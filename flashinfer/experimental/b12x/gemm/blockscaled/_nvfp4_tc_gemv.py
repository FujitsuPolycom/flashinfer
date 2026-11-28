"""Single-launch SM120a NVFP4 A16 Tensor-Core GEMV for up to 32 BF16 rows.

Streams packed E2M1 weights (`[N, K // 2]` uint8) + F8_128x4 E4M3 block scales
(group=16) directly from global memory without repacking, dequantizes 8 weights
per quarter-step on-the-fly with native SM120a PTX (``cvt.rn.bf16x2.e2m1x2`` +
``cvt.rn.bf16x2.e4m3x2`` + ``mul.bf16x2`` — lossless in BF16 because every
finite E2M1 * E4M3 product is exact in BF16), and contracts against up to
4×8 = 32 BF16 activation rows via ``bf16_mma_m16n8k16_f32``.

Two CTA layouts share the same launch/epilogue contract:
- ``n_per_cta in (16, 32)``: K-warp layout (zero barriers in the K-loop) with
  256-bit (``chunk_k=256``), 128-bit (``chunk_k=128``), or 64-bit
  (``chunk_k=64``) weight loads, optimal for small/medium ``N`` and large-vocab
  ``M <= 8`` projections.
- ``n_per_cta == 64``: 4-N-warp CTA (128 threads, 16 output rows per warp) with
  2-stage bank-conflict-free shared-memory activation staging, cutting L2
  activation traffic 4× and keeping register pressure low across ``M = 1..32``.

The scalar FP32 ``global_scale`` (or its reciprocal when
``alpha_reciprocal=True``) is applied in FP32 in the epilogue before rounding to
BF16, matching ``DenseGemmKernel(weight_only="nvfp4")``.

When ``split_k > 1``, FP32 partials are written to caller-provided workspace
scratch with coalesced 128-bit L2-coherent vector stores/loads and reduced
inside the same launch by the last arriving CTA per N-tile using self-clearing
persistent flags, eliminating both a separate SplitKReduce kernel and per-call
flag memsets.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.utils.pdl import sm120_consumer_pdl_enabled
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    shared_ptr_to_u32,
)
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import cuda_stream_from_int_or_current, current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

NVFP4_TC_GEMV_MAX_ROWS = 32
_LANES = 32

_FLAGS_CACHE: dict[tuple[int, int, int, int, int], torch.Tensor] = {}


def _smem_allocator():
    memory_mod = getattr(cutlass, "memory", None)
    if memory_mod is not None and hasattr(memory_mod, "SmemAllocator"):
        return memory_mod.SmemAllocator()
    return cutlass.utils.SmemAllocator()


def nvfp4_tc_gemv_workspace_bytes(
    num_tokens: int, out_features: int, split_k: int, n_per_cta: int = 16
) -> int:
    if split_k <= 1:
        return 0
    n_per_cta = int(n_per_cta)
    n_tiles = n_per_cta // 16
    n_blocks = (int(out_features) + n_per_cta - 1) // n_per_cta
    m_tiles = (int(num_tokens) + 7) // 8
    return n_blocks * int(split_k) * n_tiles * m_tiles * 128 * 4


def nvfp4_tc_gemv_flags_bytes(
    out_features: int, split_k: int, n_per_cta: int = 16
) -> int:
    if split_k <= 1:
        return 0
    n_per_cta = int(n_per_cta)
    return ((int(out_features) + n_per_cta - 1) // n_per_cta) * 4


def cached_nvfp4_tc_gemv_flags(
    device: torch.device,
    out_features: int,
    padded_k: int,
    split_k: int,
    n_per_cta: int = 16,
) -> torch.Tensor:
    n_per_cta = int(n_per_cta)
    key = (
        int(device.index or 0),
        int(out_features),
        int(padded_k),
        int(split_k),
        n_per_cta,
    )
    cached = _FLAGS_CACHE.get(key)
    if cached is not None:
        return cached
    flags = torch.zeros(
        ((int(out_features) + n_per_cta - 1) // n_per_cta,),
        dtype=torch.uint32,
        device=device,
    )
    _FLAGS_CACHE[key] = flags
    return flags


def effective_nvfp4_tc_gemv_chunk_k(
    padded_k: int, split_k: int, mma_warps: int, n_per_cta: int = 16
) -> int:
    padded_k = int(padded_k)
    split_k = int(split_k)
    mma_warps = int(mma_warps)
    n_per_cta = int(n_per_cta)
    if n_per_cta == 64:
        if padded_k % (256 * split_k) == 0:
            return 256
        if padded_k % (128 * split_k) == 0:
            return 128
        return 64
    if padded_k % (256 * split_k) == 0:
        chunks_256 = (padded_k // 256) // split_k
        if chunks_256 % mma_warps == 0:
            return 256
    if padded_k % (128 * split_k) == 0:
        chunks_128 = (padded_k // 128) // split_k
        if chunks_128 % mma_warps == 0:
            return 128
    return 64


def effective_nvfp4_tc_gemv_warps(
    padded_k: int, split_k: int, mma_warps: int, n_per_cta: int = 16
) -> int:
    n_per_cta = int(n_per_cta)
    if n_per_cta == 64:
        return 4
    chunk_k = effective_nvfp4_tc_gemv_chunk_k(padded_k, split_k, mma_warps, n_per_cta)
    chunks_per_split = (int(padded_k) // chunk_k) // int(split_k)
    w = min(int(mma_warps), chunks_per_split)
    while chunks_per_split % w != 0:
        w -= 1
    return w


@dsl_user_op
def _e4m3_to_bf16x2_sm120(scale_byte: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(scale_byte).ir_value(loc=loc, ip=ip)],
            """{
                .reg .b16 sf;
                .reg .b32 sf_pair;
                shl.b32 sf_pair, $1, 8;
                or.b32 sf_pair, $1, sf_pair;
                cvt.u16.u32 sf, sf_pair;
                cvt.rn.bf16x2.e4m3x2 $0, sf;
            }""",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x2_to_bf16x2_pair_sm120(scale_u16: Uint32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(scale_u16).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b16 sf;
            .reg .b32 both;
            cvt.u16.u32 sf, $2;
            cvt.rn.bf16x2.e4m3x2 both, sf;
            prmt.b32 $0, both, both, 0x1010;
            prmt.b32 $1, both, both, 0x3232;
        }""",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e4m3x4_to_bf16x2_quad_sm120(scale_u32: Uint32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(scale_u32).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b16 s01, s23;
            .reg .b32 b01, b23;
            mov.b32 {s01, s23}, $4;
            cvt.rn.bf16x2.e4m3x2 b01, s01;
            cvt.rn.bf16x2.e4m3x2 b23, s23;
            prmt.b32 $0, b01, b01, 0x1010;
            prmt.b32 $1, b01, b01, 0x3232;
            prmt.b32 $2, b23, b23, 0x1010;
            prmt.b32 $3, b23, b23, 0x3232;
        }""",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e2m1x8_scaled_to_bf16x2x4_sm120(
    packed8: Uint32, factors_bf16x2: Uint32, *, loc=None, ip=None
):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(packed8).ir_value(loc=loc, ip=ip),
            Uint32(factors_bf16x2).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 t1, t2, t3;
            .reg .b32 v0, v1, v2, v3;
            shr.u32 t1, $4, 8;
            shr.u32 t2, $4, 16;
            shr.u32 t3, $4, 24;
            cvt.u8.u32 b0, $4;
            cvt.u8.u32 b1, t1;
            cvt.u8.u32 b2, t2;
            cvt.u8.u32 b3, t3;
            cvt.rn.bf16x2.e2m1x2 v0, b0;
            cvt.rn.bf16x2.e2m1x2 v1, b1;
            cvt.rn.bf16x2.e2m1x2 v2, b2;
            cvt.rn.bf16x2.e2m1x2 v3, b3;
            mul.bf16x2 $0, v0, $5;
            mul.bf16x2 $1, v1, $5;
            mul.bf16x2 $2, v2, $5;
            mul.bf16x2 $3, v3, $5;
        }""",
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_nc_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_v2_u32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_nc_evict_v8_u32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L2::evict_first.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(8)
    )


@dsl_user_op
def _ld_global_nc_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_v4_u32(
    addr: Uint32, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Uint32(addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4_u32(addr: Uint32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _st_global_wb_v4_f32(
    addr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.wb.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_cg_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _atomic_add_gpu_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_release_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _f8_128x4_offset(row: Int32, g: Int32, k_blocks4: cutlass.Constexpr[int]) -> Int64:
    tile_r = row >> Int32(7)
    tile_k = g >> Int32(2)
    r_in = row & Int32(127)
    return (
        (Int64(tile_r) * Int64(k_blocks4) + Int64(tile_k)) * Int64(32)
        + Int64(r_in & Int32(31))
    ) * Int64(16) + Int64((r_in >> Int32(5)) * Int32(4) + (g & Int32(3)))


@cute.jit
def _load_weight_chunk(
    w_base: Int64,
    row_off: Int64,
    w_off: Int64,
    chunk_k: cutlass.Constexpr[int],
    evict_first: cutlass.Constexpr[bool],
):
    addr = w_base + row_off + w_off
    if cutlass.const_expr(chunk_k == 256):
        if cutlass.const_expr(evict_first):
            p0 = Uint32(0)
            p1 = Uint32(0)
            p2 = Uint32(0)
            p3 = Uint32(0)
            p4 = Uint32(0)
            p5 = Uint32(0)
            p6 = Uint32(0)
            p7 = Uint32(0)
            if (w_base & Int64(31)) == Int64(0):
                p0, p1, p2, p3, p4, p5, p6, p7 = _ld_global_nc_evict_v8_u32(addr)
            else:
                p0, p1, p2, p3 = ld_global_nc_v4_u32(addr)
                p4, p5, p6, p7 = ld_global_nc_v4_u32(addr + Int64(16))
            return (p0, p1, p2, p3, p4, p5, p6, p7)
        p0, p1, p2, p3 = ld_global_nc_v4_u32(addr)
        p4, p5, p6, p7 = ld_global_nc_v4_u32(addr + Int64(16))
        return (p0, p1, p2, p3, p4, p5, p6, p7)
    if cutlass.const_expr(chunk_k == 128):
        p0, p1, p2, p3 = ld_global_nc_v4_u32(addr)
        return (p0, p1, p2, p3)
    p0, p1 = _ld_global_nc_v2_u32(addr)
    return (p0, p1)


@cute.jit
def _load_scale_chunk(
    s_base: Int64,
    row: Int32,
    g0: Int32,
    k_blocks4: cutlass.Constexpr[int],
    chunk_k: cutlass.Constexpr[int],
):
    s_addr = s_base + _f8_128x4_offset(row, g0, k_blocks4)
    if cutlass.const_expr(chunk_k == 256):
        return _e4m3x4_to_bf16x2_quad_sm120(_ld_global_nc_u32(s_addr))
    if cutlass.const_expr(chunk_k == 128):
        return _e4m3x2_to_bf16x2_pair_sm120(_ld_global_nc_u16(s_addr))
    return (_e4m3_to_bf16x2_sm120(_ld_global_nc_u8(s_addr)),)


class NVFP4MultiRowTCGemvKernel:
    """Single-launch SM120a NVFP4 A16 Tensor-Core GEMV for M <= 32 (N=16, 32, or 64/CTA)."""

    def __init__(
        self,
        n: int,
        in_features: int,
        padded_k: int,
        max_m_tiles: int,
        split_k: int = 1,
        warps: int = 4,
        n_per_cta: int = 16,
        alpha_reciprocal: bool = False,
        use_pdl: bool = False,
    ):
        self.n = int(n)
        self.in_features = int(in_features)
        self.k = int(padded_k)
        self.packed_k = self.k // 2
        self.max_m_tiles = int(max_m_tiles)
        self.max_m = self.max_m_tiles * 8
        self.split_k = int(split_k)
        self.n_per_cta = int(n_per_cta)
        self.alpha_reciprocal = bool(alpha_reciprocal)
        self.use_pdl = bool(use_pdl)
        if self.n_per_cta not in (16, 32, 64):
            raise ValueError(
                f"NVFP4 TC-GEMV requires n_per_cta in (16, 32, 64), got {self.n_per_cta}"
            )
        self.n_tiles = self.n_per_cta // 16
        self.k_blocks4 = (self.k // 16 + 3) // 4
        if self.k % (64 * self.split_k) != 0:
            raise ValueError(
                f"NVFP4 TC-GEMV requires padded_k={self.k} divisible by 64*split_k"
            )
        if self.in_features % 8 != 0 or self.in_features > self.k:
            raise ValueError("NVFP4 TC-GEMV requires 8-aligned in_features <= padded_k")
        self.chunk_k = effective_nvfp4_tc_gemv_chunk_k(
            self.k, self.split_k, warps, self.n_per_cta
        )
        self.quarters = self.chunk_k // 32
        self.n_warp_mode = self.n_per_cta == 64
        self.warps = effective_nvfp4_tc_gemv_warps(
            self.k, self.split_k, warps, self.n_per_cta
        )
        self.chunks_per_split = (self.k // self.chunk_k) // self.split_k
        self.chunks_per_warp = (
            self.chunks_per_split
            if self.n_warp_mode
            else self.chunks_per_split // self.warps
        )
        # Streaming L2::evict_first keeps activations resident in L2 for M > 16
        # and large vocab matrices (> L2 capacity) while letting M <= 16 weights
        # stay L2-cacheable when re-read across sequential decode steps.
        self.evict_first = (self.chunk_k == 256) and (
            self.max_m_tiles >= 3 or self.n >= 65536
        )
        # 2-stage SMEM activation tile for n_warp_mode: [2, quarters, max_m, 4 quads, 4 u32].
        self.stage_u32s = self.quarters * self.max_m * 4 * 4

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        if self.n_warp_mode:
            SharedStorage.__annotations__ = {
                "x_smem": cute.struct.Align[
                    cute.struct.MemRange[Uint32, 2 * self.stage_u32s],
                    16,
                ],
                "last": cute.struct.Align[cute.struct.MemRange[Uint32, 1], 4],
            }
        else:
            SharedStorage.__annotations__ = {
                "partials": cute.struct.Align[
                    cute.struct.MemRange[
                        Float32,
                        self.warps * _LANES * 4 * self.max_m_tiles * self.n_tiles,
                    ],
                    16,
                ],
                "last": cute.struct.Align[cute.struct.MemRange[Uint32, 1], 4],
            }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        global_scale: cute.Pointer,
        out: cute.Pointer,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride_elems: Int64,
        stream: cuda.CUstream,
    ):
        x_t = cute.make_tensor(x, cute.make_layout((Int64(1) << Int64(50),)))
        out_t = cute.make_tensor(out, cute.make_layout((Int64(1) << Int64(50),)))
        launch_kwargs = dict(
            grid=((self.n + self.n_per_cta - 1) // self.n_per_cta, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        if cutlass.const_expr(self.n_warp_mode):
            self.kernel_nwarp(
                x_t,
                w_u8,
                s_u8,
                global_scale,
                out_t,
                scratch,
                flags,
                rows,
                x_stride_elems,
            ).launch(**launch_kwargs)
        else:
            self.kernel_kwarp(
                x_t,
                w_u8,
                s_u8,
                global_scale,
                out_t,
                scratch,
                flags,
                rows,
                x_stride_elems,
            ).launch(**launch_kwargs)

    @cute.kernel
    def kernel_nwarp(
        self,
        x: cute.Tensor,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        global_scale: cute.Pointer,
        out: cute.Tensor,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride_elems: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        n_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(tid) // _LANES
        lane = Int32(tid) % _LANES
        group = lane // 4
        quad = lane % 4
        first_n = Int32(n_tile) * self.n_per_cta
        w_base = Int64(w_u8.toint())
        s_base = Int64(s_u8.toint())
        x_base = get_ptr_as_int64(x, Int64(0))
        x_row_stride_bytes = x_stride_elems * Int64(2)
        split_chunk_base = Int32(k_split) * self.chunks_per_split

        storage = _smem_allocator().allocate(self._shared_storage_cls())
        alpha_val = _ld_global_nc_f32(Int64(global_scale.toint()))
        if cutlass.const_expr(self.alpha_reciprocal):
            alpha_val = Float32(1.0) / alpha_val

        x_smem_ptr = shared_ptr_to_u32(storage.x_smem.data_ptr())
        stage_bytes = Uint32(self.stage_u32s * 4)

        acc = cute.make_rmem_tensor((self.max_m_tiles, 4), Float32)
        for mt in cutlass.range_constexpr(self.max_m_tiles):
            for e in cutlass.range_constexpr(4):
                acc[mt, e] = Float32(0.0)

        row_a = cutlass.min(first_n + warp * Int32(16) + group, Int32(self.n - 1))
        row_b = cutlass.min(first_n + warp * Int32(16) + group + 8, Int32(self.n - 1))
        row_a_off = Int64(row_a) * Int64(self.packed_k)
        row_b_off = Int64(row_b) * Int64(self.packed_k)

        n_threads = self.warps * _LANES
        kv_per_row = self.chunk_k // 8
        total_vecs = self.max_m * kv_per_row
        iters_x = (total_vecs + n_threads - 1) // n_threads
        q_mask = self.quarters - 1
        q_shift = 3 if self.quarters == 8 else 2 if self.quarters == 4 else 1

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        for step in cutlass.range(self.chunks_per_warp, unroll=1):
            stage = Uint32(step & Int32(1))
            s_off_base = x_smem_ptr + stage * stage_bytes
            chunk = split_chunk_base + step
            k_base = chunk * Int32(self.chunk_k)

            for xi in cutlass.range_constexpr(iters_x):
                vidx = Int32(tid) + Int32(xi * n_threads)
                if cutlass.const_expr(total_vecs % n_threads == 0) or vidx < Int32(
                    total_vecs
                ):
                    xr = vidx // Int32(kv_per_row)
                    xkv = vidx % Int32(kv_per_row)
                    k_q = k_base + xkv * Int32(8)
                    vx0, vx1, vx2, vx3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if cutlass.const_expr(self.in_features == self.k) or k_q < Int32(
                        self.in_features
                    ):
                        tok = cutlass.min(xr, rows - 1)
                        px = (
                            x_base
                            + Int64(tok) * x_row_stride_bytes
                            + Int64(k_q) * Int64(2)
                        )
                        vx0, vx1, vx2, vx3 = ld_global_nc_v4_u32(px)
                    xqd = xkv >> Int32(q_shift)
                    xq = xkv & Int32(q_mask)
                    sm_off = Uint32(
                        ((xq * Int32(self.max_m) + xr) * Int32(4) + xqd) * Int32(16)
                    )
                    _st_shared_v4_u32(s_off_base + sm_off, vx0, vx1, vx2, vx3)

            k_col = k_base + quad * Int32(self.chunk_k // 4)
            g0 = chunk * Int32(self.chunk_k // 16) + quad * Int32(self.chunk_k // 64)
            w_off = Int64(k_col >> Int32(1))
            pa = _load_weight_chunk(
                w_base, row_a_off, w_off, self.chunk_k, self.evict_first
            )
            pb = _load_weight_chunk(
                w_base, row_b_off, w_off, self.chunk_k, self.evict_first
            )
            sa = _load_scale_chunk(s_base, row_a, g0, self.k_blocks4, self.chunk_k)
            sb = _load_scale_chunk(s_base, row_b, g0, self.k_blocks4, self.chunk_k)

            cute.arch.sync_threads()

            for q in cutlass.range_constexpr(self.quarters):
                wa0, wa1, wa2, wa3 = _e2m1x8_scaled_to_bf16x2x4_sm120(pa[q], sa[q // 2])
                wb0, wb1, wb2, wb3 = _e2m1x8_scaled_to_bf16x2x4_sm120(pb[q], sb[q // 2])
                if cutlass.const_expr(self.in_features < self.k):
                    k_q = k_col + Int32(q * 8)
                    if k_q >= Int32(self.in_features):
                        wa0, wa1, wa2, wa3 = (
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                        )
                        wb0, wb1, wb2, wb3 = (
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                        )
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    xr = Int32(mt * 8) + group
                    sm_off = Uint32(
                        ((Int32(q) * Int32(self.max_m) + xr) * Int32(4) + quad)
                        * Int32(16)
                    )
                    x0, x1, x2, x3 = _ld_shared_v4_u32(s_off_base + sm_off)
                    a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                        acc[mt, 0],
                        acc[mt, 1],
                        acc[mt, 2],
                        acc[mt, 3],
                        wa0,
                        wb0,
                        wa1,
                        wb1,
                        x0,
                        x1,
                    )
                    a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                        a0,
                        a1,
                        a2,
                        a3,
                        wa2,
                        wb2,
                        wa3,
                        wb3,
                        x2,
                        x3,
                    )
                    acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3] = a0, a1, a2, a3

        if cutlass.const_expr(self.split_k == 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                for e in cutlass.range_constexpr(4):
                    feature = first_n + warp * Int32(16) + group + (8 if e >= 2 else 0)
                    row = Int32(mt * 8) + quad * 2 + (e % 2)
                    if feature < self.n and row < rows:
                        out[Int64(row) * Int64(self.n) + Int64(feature)] = (
                            acc[mt, e] * alpha_val
                        ).to(BFloat16)
        else:
            tile_elems = self.n_tiles * self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            sbase = (Int64(n_tile) * self.split_k + Int64(k_split)) * tile_elems
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                slot_idx = warp * Int32(self.max_m_tiles) + Int32(mt)
                vaddr = scratch_base + (
                    sbase + Int64((slot_idx * _LANES + lane) * 4)
                ) * Int64(4)
                _st_global_wb_v4_f32(
                    vaddr, acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3]
                )
            cute.arch.sync_threads()
            last_flag = storage.last.get_tensor(cute.make_layout((1,)))
            flag_addr = Int64(flags.toint()) + Int64(n_tile) * Int64(4)
            if Int32(tid) == 0:
                old = _atomic_add_gpu_u32(flag_addr, Uint32(1))
                last_flag[Int32(0)] = (
                    Uint32(1) if old == Uint32(self.split_k - 1) else Uint32(0)
                )
            cute.arch.sync_threads()
            if last_flag[Int32(0)] == Uint32(1):
                tbase = Int64(n_tile) * self.split_k * tile_elems
                f_lo = first_n + warp * Int32(16) + group
                f_hi = f_lo + Int32(8)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = warp * Int32(self.max_m_tiles) + Int32(mt)
                    s0 = Float32(0.0)
                    s1 = Float32(0.0)
                    s2 = Float32(0.0)
                    s3 = Float32(0.0)
                    lane_off = Int64((slot_idx * _LANES + lane) * 4)
                    for sp in cutlass.range_constexpr(self.split_k):
                        raddr = scratch_base + (
                            tbase + Int64(sp) * tile_elems + lane_off
                        ) * Int64(4)
                        v0, v1, v2, v3 = _ld_global_cg_v4_f32(raddr)
                        s0 = s0 + v0
                        s1 = s1 + v1
                        s2 = s2 + v2
                        s3 = s3 + v3
                    r0 = Int32(mt * 8) + quad * 2
                    r1 = r0 + 1
                    if f_lo < self.n:
                        if r0 < rows:
                            out[Int64(r0) * Int64(self.n) + Int64(f_lo)] = (
                                s0 * alpha_val
                            ).to(BFloat16)
                        if r1 < rows:
                            out[Int64(r1) * Int64(self.n) + Int64(f_lo)] = (
                                s1 * alpha_val
                            ).to(BFloat16)
                    if f_hi < self.n:
                        if r0 < rows:
                            out[Int64(r0) * Int64(self.n) + Int64(f_hi)] = (
                                s2 * alpha_val
                            ).to(BFloat16)
                        if r1 < rows:
                            out[Int64(r1) * Int64(self.n) + Int64(f_hi)] = (
                                s3 * alpha_val
                            ).to(BFloat16)
                cute.arch.sync_threads()
                if Int32(tid) == 0:
                    _st_global_release_u32(flag_addr, Uint32(0))

    @cute.kernel
    def kernel_kwarp(
        self,
        x: cute.Tensor,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        global_scale: cute.Pointer,
        out: cute.Tensor,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride_elems: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        n_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(tid) // _LANES
        lane = Int32(tid) % _LANES
        group = lane // 4
        quad = lane % 4
        first_n = Int32(n_tile) * self.n_per_cta
        w_base = Int64(w_u8.toint())
        s_base = Int64(s_u8.toint())
        x_base = get_ptr_as_int64(x, Int64(0))
        x_row_stride_bytes = x_stride_elems * Int64(2)
        split_chunk_base = Int32(k_split) * self.chunks_per_split

        storage = _smem_allocator().allocate(self._shared_storage_cls())
        alpha_val = _ld_global_nc_f32(Int64(global_scale.toint()))
        if cutlass.const_expr(self.alpha_reciprocal):
            alpha_val = Float32(1.0) / alpha_val

        # K-warp mode (n_per_cta in (16, 32)): zero barriers in the K-loop.
        acc = cute.make_rmem_tensor((self.n_tiles, self.max_m_tiles, 4), Float32)
        for nt in cutlass.range_constexpr(self.n_tiles):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                for e in cutlass.range_constexpr(4):
                    acc[nt, mt, e] = Float32(0.0)

        if cutlass.const_expr(self.n_tiles == 1):
            row_a = cutlass.min(first_n + group, Int32(self.n - 1))
            row_b = cutlass.min(first_n + group + 8, Int32(self.n - 1))
            row_a_off = Int64(row_a) * Int64(self.packed_k)
            row_b_off = Int64(row_b) * Int64(self.packed_k)
            for step in cutlass.range(self.chunks_per_warp, unroll=1):
                chunk = split_chunk_base + warp * self.chunks_per_warp + step
                k_col = chunk * Int32(self.chunk_k) + quad * Int32(self.chunk_k // 4)
                g0 = chunk * Int32(self.chunk_k // 16) + quad * Int32(
                    self.chunk_k // 64
                )
                w_off = Int64(k_col >> Int32(1))
                pa = _load_weight_chunk(
                    w_base, row_a_off, w_off, self.chunk_k, self.evict_first
                )
                pb = _load_weight_chunk(
                    w_base, row_b_off, w_off, self.chunk_k, self.evict_first
                )
                sa = _load_scale_chunk(s_base, row_a, g0, self.k_blocks4, self.chunk_k)
                sb = _load_scale_chunk(s_base, row_b, g0, self.k_blocks4, self.chunk_k)
                if cutlass.const_expr(self.use_pdl):
                    if step == Int32(0):
                        cute.arch.griddepcontrol_wait()
                x_off = Int64(k_col) * Int64(2)
                for q in cutlass.range_constexpr(self.quarters):
                    wa0, wa1, wa2, wa3 = _e2m1x8_scaled_to_bf16x2x4_sm120(
                        pa[q], sa[q // 2]
                    )
                    wb0, wb1, wb2, wb3 = _e2m1x8_scaled_to_bf16x2x4_sm120(
                        pb[q], sb[q // 2]
                    )
                    k_q = k_col + Int32(q * 8)
                    q_off = x_off + Int64(q * 16)
                    if cutlass.const_expr(self.in_features < self.k):
                        if k_q >= Int32(self.in_features):
                            wa0, wa1, wa2, wa3 = (
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                            )
                            wb0, wb1, wb2, wb3 = (
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                            )
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                        px = x_base + Int64(token) * x_row_stride_bytes + q_off
                        x0 = Uint32(0)
                        x1 = Uint32(0)
                        x2 = Uint32(0)
                        x3 = Uint32(0)
                        if cutlass.const_expr(
                            self.in_features == self.k
                        ) or k_q < Int32(self.in_features):
                            x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            acc[0, mt, 0],
                            acc[0, mt, 1],
                            acc[0, mt, 2],
                            acc[0, mt, 3],
                            wa0,
                            wb0,
                            wa1,
                            wb1,
                            x0,
                            x1,
                        )
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa2,
                            wb2,
                            wa3,
                            wb3,
                            x2,
                            x3,
                        )
                        (
                            acc[0, mt, 0],
                            acc[0, mt, 1],
                            acc[0, mt, 2],
                            acc[0, mt, 3],
                        ) = (a0, a1, a2, a3)
        else:
            row_a0 = cutlass.min(first_n + group, Int32(self.n - 1))
            row_b0 = cutlass.min(first_n + group + 8, Int32(self.n - 1))
            row_a1 = cutlass.min(first_n + 16 + group, Int32(self.n - 1))
            row_b1 = cutlass.min(first_n + 16 + group + 8, Int32(self.n - 1))
            row_a0_off = Int64(row_a0) * Int64(self.packed_k)
            row_b0_off = Int64(row_b0) * Int64(self.packed_k)
            row_a1_off = Int64(row_a1) * Int64(self.packed_k)
            row_b1_off = Int64(row_b1) * Int64(self.packed_k)
            x_buf = cute.make_rmem_tensor((self.max_m_tiles, 4), Uint32)
            for step in cutlass.range(self.chunks_per_warp, unroll=1):
                chunk = split_chunk_base + warp * self.chunks_per_warp + step
                k_col = chunk * Int32(self.chunk_k) + quad * Int32(self.chunk_k // 4)
                g0 = chunk * Int32(self.chunk_k // 16) + quad * Int32(
                    self.chunk_k // 64
                )
                w_off = Int64(k_col >> Int32(1))
                pa = _load_weight_chunk(
                    w_base, row_a0_off, w_off, self.chunk_k, self.evict_first
                )
                pb = _load_weight_chunk(
                    w_base, row_b0_off, w_off, self.chunk_k, self.evict_first
                )
                pc = _load_weight_chunk(
                    w_base, row_a1_off, w_off, self.chunk_k, self.evict_first
                )
                pd = _load_weight_chunk(
                    w_base, row_b1_off, w_off, self.chunk_k, self.evict_first
                )
                sa0 = _load_scale_chunk(
                    s_base, row_a0, g0, self.k_blocks4, self.chunk_k
                )
                sb0 = _load_scale_chunk(
                    s_base, row_b0, g0, self.k_blocks4, self.chunk_k
                )
                sa1 = _load_scale_chunk(
                    s_base, row_a1, g0, self.k_blocks4, self.chunk_k
                )
                sb1 = _load_scale_chunk(
                    s_base, row_b1, g0, self.k_blocks4, self.chunk_k
                )
                if cutlass.const_expr(self.use_pdl):
                    if step == Int32(0):
                        cute.arch.griddepcontrol_wait()
                x_off = Int64(k_col) * Int64(2)
                for q in cutlass.range_constexpr(self.quarters):
                    k_q = k_col + Int32(q * 8)
                    q_off = x_off + Int64(q * 16)
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                        px = x_base + Int64(token) * x_row_stride_bytes + q_off
                        x0 = Uint32(0)
                        x1 = Uint32(0)
                        x2 = Uint32(0)
                        x3 = Uint32(0)
                        if cutlass.const_expr(
                            self.in_features == self.k
                        ) or k_q < Int32(self.in_features):
                            x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                        x_buf[mt, 0], x_buf[mt, 1], x_buf[mt, 2], x_buf[mt, 3] = (
                            x0,
                            x1,
                            x2,
                            x3,
                        )
                    # N-tile 0
                    wa0, wa1, wa2, wa3 = _e2m1x8_scaled_to_bf16x2x4_sm120(
                        pa[q], sa0[q // 2]
                    )
                    wb0, wb1, wb2, wb3 = _e2m1x8_scaled_to_bf16x2x4_sm120(
                        pb[q], sb0[q // 2]
                    )
                    if cutlass.const_expr(self.in_features < self.k):
                        if k_q >= Int32(self.in_features):
                            wa0, wa1, wa2, wa3 = (
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                            )
                            wb0, wb1, wb2, wb3 = (
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                            )
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            acc[0, mt, 0],
                            acc[0, mt, 1],
                            acc[0, mt, 2],
                            acc[0, mt, 3],
                            wa0,
                            wb0,
                            wa1,
                            wb1,
                            x_buf[mt, 0],
                            x_buf[mt, 1],
                        )
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa2,
                            wb2,
                            wa3,
                            wb3,
                            x_buf[mt, 2],
                            x_buf[mt, 3],
                        )
                        (
                            acc[0, mt, 0],
                            acc[0, mt, 1],
                            acc[0, mt, 2],
                            acc[0, mt, 3],
                        ) = (a0, a1, a2, a3)
                    # N-tile 1
                    wc0, wc1, wc2, wc3 = _e2m1x8_scaled_to_bf16x2x4_sm120(
                        pc[q], sa1[q // 2]
                    )
                    wd0, wd1, wd2, wd3 = _e2m1x8_scaled_to_bf16x2x4_sm120(
                        pd[q], sb1[q // 2]
                    )
                    if cutlass.const_expr(self.in_features < self.k):
                        if k_q >= Int32(self.in_features):
                            wc0, wc1, wc2, wc3 = (
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                            )
                            wd0, wd1, wd2, wd3 = (
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                                Uint32(0),
                            )
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            acc[1, mt, 0],
                            acc[1, mt, 1],
                            acc[1, mt, 2],
                            acc[1, mt, 3],
                            wc0,
                            wd0,
                            wc1,
                            wd1,
                            x_buf[mt, 0],
                            x_buf[mt, 1],
                        )
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wc2,
                            wd2,
                            wc3,
                            wd3,
                            x_buf[mt, 2],
                            x_buf[mt, 3],
                        )
                        (
                            acc[1, mt, 0],
                            acc[1, mt, 1],
                            acc[1, mt, 2],
                            acc[1, mt, 3],
                        ) = (a0, a1, a2, a3)

        partials = storage.partials.get_tensor(
            cute.make_layout(
                (self.warps * _LANES * 4 * self.max_m_tiles * self.n_tiles,)
            )
        )
        if cutlass.const_expr(self.warps > 1):
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = nt * self.max_m_tiles + mt
                    base = ((slot_idx * self.warps + warp) * _LANES + lane) * 4
                    for e in cutlass.range_constexpr(4):
                        partials[base + e] = acc[nt, mt, e]
            cute.arch.sync_threads()

        if cutlass.const_expr(self.split_k == 1):
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = nt * self.max_m_tiles + mt
                    if warp == Int32(slot_idx % self.warps):
                        for e in cutlass.range_constexpr(4):
                            total = (
                                acc[nt, mt, e]
                                if cutlass.const_expr(self.warps == 1)
                                else Float32(0.0)
                            )
                            if cutlass.const_expr(self.warps > 1):
                                for w_i in cutlass.range_constexpr(self.warps):
                                    total = (
                                        total
                                        + partials[
                                            (
                                                (slot_idx * self.warps + w_i) * _LANES
                                                + lane
                                            )
                                            * 4
                                            + e
                                        ]
                                    )
                            feature = first_n + nt * 16 + group + (8 if e >= 2 else 0)
                            row = Int32(mt * 8) + quad * 2 + (e % 2)
                            if feature < self.n and row < rows:
                                out[Int64(row) * Int64(self.n) + Int64(feature)] = (
                                    total * alpha_val
                                ).to(BFloat16)
        else:
            tile_elems = self.n_tiles * self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            sbase = (Int64(n_tile) * self.split_k + Int64(k_split)) * tile_elems
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = nt * self.max_m_tiles + mt
                    if warp == Int32(slot_idx % self.warps):
                        t0 = (
                            acc[nt, mt, 0]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        t1 = (
                            acc[nt, mt, 1]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        t2 = (
                            acc[nt, mt, 2]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        t3 = (
                            acc[nt, mt, 3]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        if cutlass.const_expr(self.warps > 1):
                            for w_i in cutlass.range_constexpr(self.warps):
                                pbase = (
                                    (slot_idx * self.warps + w_i) * _LANES + lane
                                ) * 4
                                t0 = t0 + partials[pbase + 0]
                                t1 = t1 + partials[pbase + 1]
                                t2 = t2 + partials[pbase + 2]
                                t3 = t3 + partials[pbase + 3]
                        vaddr = scratch_base + (
                            sbase + Int64((slot_idx * _LANES + lane) * 4)
                        ) * Int64(4)
                        _st_global_wb_v4_f32(vaddr, t0, t1, t2, t3)
            cute.arch.sync_threads()
            last_flag = storage.last.get_tensor(cute.make_layout((1,)))
            flag_addr = Int64(flags.toint()) + Int64(n_tile) * Int64(4)
            if Int32(tid) == 0:
                old = _atomic_add_gpu_u32(flag_addr, Uint32(1))
                last_flag[Int32(0)] = (
                    Uint32(1) if old == Uint32(self.split_k - 1) else Uint32(0)
                )
            cute.arch.sync_threads()
            if last_flag[Int32(0)] == Uint32(1):
                tbase = Int64(n_tile) * self.split_k * tile_elems
                for nt in cutlass.range_constexpr(self.n_tiles):
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        slot_idx = nt * self.max_m_tiles + mt
                        if warp == Int32(slot_idx % self.warps):
                            s0 = Float32(0.0)
                            s1 = Float32(0.0)
                            s2 = Float32(0.0)
                            s3 = Float32(0.0)
                            lane_off = Int64((slot_idx * _LANES + lane) * 4)
                            for sp in cutlass.range_constexpr(self.split_k):
                                raddr = scratch_base + (
                                    tbase + Int64(sp) * tile_elems + lane_off
                                ) * Int64(4)
                                v0, v1, v2, v3 = _ld_global_cg_v4_f32(raddr)
                                s0 = s0 + v0
                                s1 = s1 + v1
                                s2 = s2 + v2
                                s3 = s3 + v3
                            f_lo = first_n + nt * 16 + group
                            f_hi = first_n + nt * 16 + group + 8
                            r0 = Int32(mt * 8) + quad * 2
                            r1 = r0 + 1
                            if f_lo < self.n:
                                if r0 < rows:
                                    out[Int64(r0) * Int64(self.n) + Int64(f_lo)] = (
                                        s0 * alpha_val
                                    ).to(BFloat16)
                                if r1 < rows:
                                    out[Int64(r1) * Int64(self.n) + Int64(f_lo)] = (
                                        s1 * alpha_val
                                    ).to(BFloat16)
                            if f_hi < self.n:
                                if r0 < rows:
                                    out[Int64(r0) * Int64(self.n) + Int64(f_hi)] = (
                                        s2 * alpha_val
                                    ).to(BFloat16)
                                if r1 < rows:
                                    out[Int64(r1) * Int64(self.n) + Int64(f_hi)] = (
                                        s3 * alpha_val
                                    ).to(BFloat16)
                cute.arch.sync_threads()
                if Int32(tid) == 0:
                    _st_global_release_u32(flag_addr, Uint32(0))


@program_cache
def _compile_nvfp4_tc_gemv_cached(
    ordinal: int,
    n: int,
    in_features: int,
    padded_k: int,
    max_rows: int,
    split_k: int,
    warps: int,
    n_per_cta: int,
    alpha_reciprocal: bool,
    use_pdl: bool,
):
    max_m_tiles = (int(max_rows) + 7) // 8
    n_per_cta = int(n_per_cta)
    chunk_k = effective_nvfp4_tc_gemv_chunk_k(padded_k, split_k, warps, n_per_cta)
    eff_warps = effective_nvfp4_tc_gemv_warps(padded_k, split_k, warps, n_per_cta)
    alpha_reciprocal = bool(alpha_reciprocal)
    use_pdl = bool(use_pdl)
    kernel = NVFP4MultiRowTCGemvKernel(
        n,
        in_features,
        padded_k,
        max_m_tiles,
        split_k=split_k,
        warps=eff_warps,
        n_per_cta=n_per_cta,
        alpha_reciprocal=alpha_reciprocal,
        use_pdl=use_pdl,
    )
    key = (
        int(n),
        int(in_features),
        int(padded_k),
        max_m_tiles,
        int(split_k),
        chunk_k,
        eff_warps,
        n_per_cta,
        alpha_reciprocal,
        use_pdl,
    )
    fake = (
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4),
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            *fake,
            Int32(1),
            Int64(in_features),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.blockscaled.nvfp4_tc_gemv", 4, key
            ),
        )

    def run(source, values, scale_bytes, global_scale, output, scratch, flags, stream):
        if split_k > 1 and (scratch is None or flags is None):
            raise ValueError("split_k > 1 requires scratch and flags buffers")
        sc_ptr = (
            make_ptr(
                Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            if scratch is not None
            else make_ptr(
                Float32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        fl_ptr = (
            make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
            if flags is not None
            else make_ptr(
                Uint32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        run_compiled(
            raw,
            (
                make_ptr(
                    BFloat16,
                    source.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    Uint8, values.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    Uint8,
                    scale_bytes.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    Float32,
                    global_scale.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                make_ptr(
                    BFloat16,
                    output.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                sc_ptr,
                fl_ptr,
                Int32(source.shape[0]),
                Int64(source.stride(0)),
                cuda_stream_from_int_or_current(stream),
            ),
        )

    return attach_programs(run, raw)


def compile_nvfp4_tc_gemv(
    ordinal: int,
    n: int,
    in_features: int,
    padded_k: int,
    max_rows: int,
    split_k: int,
    warps: int,
    n_per_cta: int = 16,
    *,
    alpha_reciprocal: bool = False,
    use_pdl: bool | None = None,
):
    resolved_pdl = (
        sm120_consumer_pdl_enabled(ordinal, env_var="B12X_NVFP4_TC_GEMV_PDL")
        if use_pdl is None
        else bool(use_pdl)
    )
    return _compile_nvfp4_tc_gemv_cached(
        ordinal,
        n,
        in_features,
        padded_k,
        max_rows,
        split_k,
        warps,
        n_per_cta,
        alpha_reciprocal,
        resolved_pdl,
    )


compile_nvfp4_tc_gemv.cache_clear = _compile_nvfp4_tc_gemv_cached.cache_clear
compile_nvfp4_tc_gemv.cache_info = _compile_nvfp4_tc_gemv_cached.cache_info
