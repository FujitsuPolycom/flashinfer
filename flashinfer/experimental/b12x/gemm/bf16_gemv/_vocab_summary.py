# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Epilogue-fused zero-HBM BF16 lm_head online-LSE, Gumbel-max, and probe logit.

Projects ``x[M, K] @ weight[N, K].T`` in FP32 tensor-core accumulators and
reduces each CTA's vocabulary column tile directly in warp registers to compact
L2-resident tile summaries ``(tile_max, tile_sum_exp, tile_argmax)`` plus
optional Philox-4x32-10 Gumbel-max summaries and exact single-token draft
logit probes.  When ``write_logits=False`` (the default), zero ``[M, N]`` logit
bytes are written to HBM.

A lightweight Stage-2 reduction kernel (``reduce_vocab_tile_summaries``,
``grid=(M, 1, 1)``) reduces the L2-resident tile summaries into per-row
``(global_max, global_sum_exp, global_lse, argmax_id, gumbel_max, gumbel_id,
probe_prob)`` ready for TP4 word-exchange collectives and scalar speculative
verification.

Row regimes:
- ``1 <= max_rows <= 32``: ``TILE_N = 16``, ``warps = 8`` splitting ``K``
  (exact K-reduction order of ``TensorCoreGemvKernel`` for 0-ULP dot-product
  parity).
- ``max_rows > 32`` (e.g. ``M = 64, 128``): ``TILE_N = 64``,
  ``warps = 8 = 4 (N) x 2 (M)``, streaming ``weight`` once per 128 rows while
  reusing ``x`` across 4 N-warps in L1 and emitting 4x fewer tile summaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.tensor_utils import LayoutEnum

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    compile as b12x_compile,
    run_compiled,
    tensor_key,
)
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_f32,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

from ._prefill import _assume_tma_source_aligned, _to_kernel_tensor
from ._tensor_core import TC_K_MULTIPLE, _LANES, _flat, tc_chunk_k

try:
    from cutlass.memory import SmemAllocator as _SmemAllocator
except ImportError:
    from cutlass.utils import SmemAllocator as _SmemAllocator

_SMALL_M_THRESHOLD = 32
_LARGE_M_BLOCK = 128
_REDUCE_THREADS = 512
_REDUCE_WARPS = _REDUCE_THREADS // _LANES
_GUMBEL_COUNTER_DOMAIN = 0x47554D42


def vocab_summary_tile_n(max_rows: int) -> int:
    """Return the N-tile width used by Stage-1 for a given ``max_rows`` bucket."""
    return 16 if int(max_rows) <= _SMALL_M_THRESHOLD else 64


def vocab_summary_num_tiles(n: int, max_rows: int) -> int:
    """Return the number of vocabulary summary tiles per row for ``(n, max_rows)``."""
    tile_n = vocab_summary_tile_n(max_rows)
    return (int(n) + tile_n - 1) // tile_n


def _max_rows_bucket(rows: int) -> int:
    rows_i = int(rows)
    if rows_i <= 0:
        raise ValueError("rows must be positive")
    if rows_i <= _SMALL_M_THRESHOLD:
        return ((rows_i + 7) // 8) * 8
    if rows_i <= _LARGE_M_BLOCK:
        return ((rows_i + 15) // 16) * 16
    return _LARGE_M_BLOCK


@dataclass
class VocabSummaryBuffers:
    """Caller-owned CUDA-graph-safe buffers for fused lm_head summary and reduction."""

    max_rows: int
    n: int
    tile_n: int
    num_tiles: int
    tile_max: torch.Tensor
    tile_sum_exp: torch.Tensor
    tile_argmax: torch.Tensor
    global_max: torch.Tensor
    global_sum_exp: torch.Tensor
    global_lse: torch.Tensor
    argmax_id: torch.Tensor
    tile_gumbel_max: torch.Tensor | None = None
    tile_gumbel_arg: torch.Tensor | None = None
    gumbel_max: torch.Tensor | None = None
    gumbel_id: torch.Tensor | None = None
    probe_logits: torch.Tensor | None = None
    probe_prob: torch.Tensor | None = None
    logits_out: torch.Tensor | None = None


@dataclass
class VocabSummaryResult:
    """Per-row outputs of ``bf16_gemv_vocab_summary`` (sliced to live ``M``)."""

    global_max: torch.Tensor
    global_sum_exp: torch.Tensor
    global_lse: torch.Tensor
    argmax_id: torch.Tensor
    gumbel_max: torch.Tensor | None
    gumbel_id: torch.Tensor | None
    probe_logits: torch.Tensor | None
    probe_prob: torch.Tensor | None
    logits_out: torch.Tensor | None
    buffers: VocabSummaryBuffers


def allocate_vocab_summary_outputs(
    max_rows: int,
    n: int,
    device: torch.device | str,
    *,
    use_gumbel: bool = False,
    has_probe: bool = False,
    write_logits: bool = False,
    logits_dtype: torch.dtype = torch.bfloat16,
) -> VocabSummaryBuffers:
    """Allocate reusable L2-resident tile summaries and final per-row outputs."""
    max_rows_i = int(max_rows)
    n_i = int(n)
    if max_rows_i <= 0 or n_i <= 0:
        raise ValueError("max_rows and n must be positive")
    dev = torch.device(device)
    tile_n = vocab_summary_tile_n(max_rows_i)
    num_tiles = vocab_summary_num_tiles(n_i, max_rows_i)
    tile_max = torch.empty((max_rows_i, num_tiles), dtype=torch.float32, device=dev)
    tile_sum_exp = torch.empty((max_rows_i, num_tiles), dtype=torch.float32, device=dev)
    tile_argmax = torch.empty((max_rows_i, num_tiles), dtype=torch.int32, device=dev)
    global_max = torch.empty((max_rows_i,), dtype=torch.float32, device=dev)
    global_sum_exp = torch.empty((max_rows_i,), dtype=torch.float32, device=dev)
    global_lse = torch.empty((max_rows_i,), dtype=torch.float32, device=dev)
    argmax_id = torch.empty((max_rows_i,), dtype=torch.int32, device=dev)
    tile_gumbel_max = (
        torch.empty((max_rows_i, num_tiles), dtype=torch.float32, device=dev)
        if use_gumbel
        else None
    )
    tile_gumbel_arg = (
        torch.empty((max_rows_i, num_tiles), dtype=torch.int32, device=dev)
        if use_gumbel
        else None
    )
    gumbel_max = (
        torch.empty((max_rows_i,), dtype=torch.float32, device=dev)
        if use_gumbel
        else None
    )
    gumbel_id = (
        torch.empty((max_rows_i,), dtype=torch.int32, device=dev)
        if use_gumbel
        else None
    )
    probe_logits = (
        torch.empty((max_rows_i,), dtype=torch.float32, device=dev)
        if has_probe
        else None
    )
    probe_prob = (
        torch.empty((max_rows_i,), dtype=torch.float32, device=dev)
        if has_probe
        else None
    )
    if write_logits:
        if logits_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("logits_dtype must be bfloat16 or float32")
        logits_out = torch.empty((max_rows_i, n_i), dtype=logits_dtype, device=dev)
    else:
        logits_out = None
    return VocabSummaryBuffers(
        max_rows=max_rows_i,
        n=n_i,
        tile_n=tile_n,
        num_tiles=num_tiles,
        tile_max=tile_max,
        tile_sum_exp=tile_sum_exp,
        tile_argmax=tile_argmax,
        global_max=global_max,
        global_sum_exp=global_sum_exp,
        global_lse=global_lse,
        argmax_id=argmax_id,
        tile_gumbel_max=tile_gumbel_max,
        tile_gumbel_arg=tile_gumbel_arg,
        gumbel_max=gumbel_max,
        gumbel_id=gumbel_id,
        probe_logits=probe_logits,
        probe_prob=probe_prob,
        logits_out=logits_out,
    )


# =============================================================================
# PTX Intrinsics for Warp Shuffle, Bitmask, and Exact Philox-4x32-10 Gumbel
# =============================================================================


@dsl_user_op
def _shfl_bfly_f32(val: Float32, offset: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(val).ir_value(loc=loc, ip=ip),
                Int32(offset).ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_i32(val: Int32, offset: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(val).ir_value(loc=loc, ip=ip),
                Int32(offset).ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bit_allowed(word: Uint32, bit_pos: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(word).ir_value(loc=loc, ip=ip),
                Int32(bit_pos).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .u32 s; shr.u32 s, $1, $2; and.b32 $0, s, 1; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _split_i64_u32(val: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(val).ir_value(loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;",
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
def _philox_round(
    c0: Uint32,
    c1: Uint32,
    c2: Uint32,
    c3: Uint32,
    k0: Uint32,
    k1: Uint32,
    *,
    loc=None,
    ip=None,
):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(c0).ir_value(loc=loc, ip=ip),
            Uint32(c1).ir_value(loc=loc, ip=ip),
            Uint32(c2).ir_value(loc=loc, ip=ip),
            Uint32(c3).ir_value(loc=loc, ip=ip),
            Uint32(k0).ir_value(loc=loc, ip=ip),
            Uint32(k1).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 hi0, lo0, hi1, lo1; "
        "mul.hi.u32 hi0, $4, 0xD2511F53; "
        "mul.lo.u32 lo0, $4, 0xD2511F53; "
        "mul.hi.u32 hi1, $6, 0xCD9E8D57; "
        "mul.lo.u32 lo1, $6, 0xCD9E8D57; "
        "xor.b32 $0, hi1, $5; "
        "xor.b32 $0, $0, $8; "
        "mov.u32 $1, lo1; "
        "xor.b32 $2, hi0, $7; "
        "xor.b32 $2, $2, $9; "
        "mov.u32 $3, lo0; }",
        "=r,=r,=r,=r,r,r,r,r,r,r",
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
def _add_u32_imm(val: Uint32, imm: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(val).ir_value(loc=loc, ip=ip),
                Uint32(imm).ir_value(loc=loc, ip=ip),
            ],
            "add.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits_to_uniform_f32(bits: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert uint32 Philox output to U(0, 1) matching Hyperflare's _token_noise."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(bits).ir_value(loc=loc, ip=ip)],
            "{ .reg .u32 hi, lo; .reg .f32 fhi, flo, t; "
            "shr.u32 hi, $1, 16; and.b32 lo, $1, 65535; "
            "cvt.rn.f32.u32 fhi, hi; cvt.rn.f32.u32 flo, lo; "
            "fma.rn.f32 t, fhi, 0f47800000, flo; "
            "add.rn.f32 t, t, 0f3f000000; "
            "mul.rn.f32 t, t, 0f2f800000; "
            "min.f32 $0, t, 0f3f7fffff; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_i64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.s64 $0, [$1];",
            "=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.s32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _philox_gumbel_noise(seed: Int64, offset: Int64, global_col: Int32) -> Float32:
    """Standard Random123 Philox-4x32-10 Gumbel(0, 1) noise matching Triton tl.philox."""
    k0, k1 = _split_i64_u32(seed)
    off_lo, off_hi = _split_i64_u32(offset)
    c0 = Uint32(global_col)
    c1 = off_lo
    c2 = off_hi
    c3 = Uint32(_GUMBEL_COUNTER_DOMAIN)
    w0 = Uint32(0x9E3779B9)
    w1 = Uint32(0xBB67AE85)
    for _ in cutlass.range_constexpr(10):
        c0, c1, c2, c3 = _philox_round(c0, c1, c2, c3, k0, k1)
        k0 = _add_u32_imm(k0, w0)
        k1 = _add_u32_imm(k1, w1)
    u = _bits_to_uniform_f32(c0)
    return -cute.math.log(-cute.math.log1p(-u))


@cute.jit
def _argmax_pair(
    va: Float32, ia: Int32, vb: Float32, ib: Int32
) -> tuple[Float32, Int32]:
    """Associative max+argmax with tie_break_left=True and -1 invalid sentinel."""
    take_b = (vb > va) or (
        (vb == va) and (ib >= Int32(0)) and ((ia < Int32(0)) or (ib < ia))
    )
    out_v = vb if take_b else va
    out_i = ib if take_b else ia
    return out_v, out_i


@cute.jit
def _reduce_16col_row(
    val_lo: Float32,
    val_hi: Float32,
    valid_lo: cutlass.Boolean,
    valid_hi: cutlass.Boolean,
    gcol_lo: Int32,
    gcol_hi: Int32,
    is_greedy: cutlass.Boolean,
    seed: Int64,
    offset: Int64,
    use_gumbel: cutlass.Constexpr[bool],
):
    """Reduce 16 columns across 8 lanes sharing ``quad = lane % 4`` via 3 shuffles."""
    neg_inf = u32_as_f32(Uint32(0xFF800000))
    pos_inf = u32_as_f32(Uint32(0x7F800000))
    z_lo = val_lo if valid_lo else neg_inf
    z_hi = val_hi if valid_hi else neg_inf
    i_lo = gcol_lo if valid_lo else Int32(-1)
    i_hi = gcol_hi if valid_hi else Int32(-1)

    m_val, m_idx = _argmax_pair(z_lo, i_lo, z_hi, i_hi)
    for shfl_off in cutlass.range_constexpr(3):
        off = Int32(4 << shfl_off)
        om = _shfl_bfly_f32(m_val, off)
        oi = _shfl_bfly_i32(m_idx, off)
        m_val, m_idx = _argmax_pair(m_val, m_idx, om, oi)

    is_fin = (m_val > neg_inf) and (m_val < pos_inf)
    stable_max = m_val if is_fin else Float32(0.0)
    e_lo = cute.math.exp(z_lo - stable_max) if valid_lo else Float32(0.0)
    e_hi = cute.math.exp(z_hi - stable_max) if valid_hi else Float32(0.0)
    s_exp = e_lo + e_hi
    for shfl_off in cutlass.range_constexpr(3):
        s_exp = s_exp + _shfl_bfly_f32(s_exp, Int32(4 << shfl_off))

    g_val = m_val
    g_idx = m_idx
    if cutlass.const_expr(use_gumbel):
        n_lo = (
            Float32(0.0)
            if (is_greedy or (not valid_lo))
            else _philox_gumbel_noise(seed, offset, gcol_lo)
        )
        n_hi = (
            Float32(0.0)
            if (is_greedy or (not valid_hi))
            else _philox_gumbel_noise(seed, offset, gcol_hi)
        )
        c_lo = (z_lo + n_lo) if valid_lo else neg_inf
        c_hi = (z_hi + n_hi) if valid_hi else neg_inf
        g_val, g_idx = _argmax_pair(c_lo, i_lo, c_hi, i_hi)
        for shfl_off in cutlass.range_constexpr(3):
            off = Int32(4 << shfl_off)
            og = _shfl_bfly_f32(g_val, off)
            oi = _shfl_bfly_i32(g_idx, off)
            g_val, g_idx = _argmax_pair(g_val, g_idx, og, oi)

    return m_val, s_exp, m_idx, g_val, g_idx


# =============================================================================
# Stage-1 Kernel A: Small-M (1 <= max_rows <= 32), TILE_N = 16, warps = 8 (K-split)
# =============================================================================


class TensorCoreVocabSummarySmallMKernel:
    """Fused BF16 GEMV + 16-column warp-register vocab summary for 1..32 rows."""

    def __init__(
        self,
        n: int,
        k: int,
        max_rows: int,
        *,
        use_inv_temp: bool = False,
        use_bias: bool = False,
        use_grammar: bool = False,
        use_probe: bool = False,
        use_gumbel: bool = False,
        write_logits: bool = False,
    ):
        if k <= 0 or k % TC_K_MULTIPLE:
            raise ValueError(
                f"vocab-summary TC-GEMV requires K to be a positive multiple of {TC_K_MULTIPLE}"
            )
        if not 1 <= max_rows <= _SMALL_M_THRESHOLD:
            raise ValueError(
                f"small-M vocab-summary kernel supports 1..{_SMALL_M_THRESHOLD} rows"
            )
        self.n = int(n)
        self.k = int(k)
        self.chunk_k = tc_chunk_k(k)
        chunks = self.k // self.chunk_k
        self.max_m_tiles = (int(max_rows) + 7) // 8
        warps = min(8, chunks)
        while chunks % warps:
            warps -= 1
        self.warps = warps
        self.chunks_per_warp = chunks // warps
        self.num_tiles = (self.n + 15) // 16
        self.use_inv_temp = bool(use_inv_temp)
        self.use_bias = bool(use_bias)
        self.use_grammar = bool(use_grammar)
        self.use_probe = bool(use_probe)
        self.use_gumbel = bool(use_gumbel)
        self.write_logits = bool(write_logits)

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.warps * _LANES * 4 * self.max_m_tiles
                ],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        tile_max: cute.Pointer,
        tile_sum_exp: cute.Pointer,
        tile_argmax: cute.Pointer,
        tile_gumbel_max: cute.Pointer,
        tile_gumbel_arg: cute.Pointer,
        inv_temperature: cute.Pointer,
        logit_bias: cute.Pointer,
        grammar_mask: cute.Pointer,
        probe_token_ids: cute.Pointer,
        probe_logits: cute.Pointer,
        rng_seed: cute.Pointer,
        rng_offset: cute.Pointer,
        logits_out: cute.Pointer,
        rows: Int32,
        vocab_offset: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        summary_stride: Int64,
        mask_stride: Int64,
        logits_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(tile_max),
            _flat(tile_sum_exp),
            _flat(tile_argmax),
            _flat(tile_gumbel_max),
            _flat(tile_gumbel_arg),
            inv_temperature,
            logit_bias,
            grammar_mask,
            probe_token_ids,
            _flat(probe_logits),
            rng_seed,
            rng_offset,
            _flat(logits_out),
            rows,
            vocab_offset,
            x_stride,
            weight_stride,
            summary_stride,
            mask_stride,
            logits_stride,
        ).launch(
            grid=(self.num_tiles, 1, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        tile_max: cute.Tensor,
        tile_sum_exp: cute.Tensor,
        tile_argmax: cute.Tensor,
        tile_gumbel_max: cute.Tensor,
        tile_gumbel_arg: cute.Tensor,
        inv_temperature: cute.Pointer,
        logit_bias: cute.Pointer,
        grammar_mask: cute.Pointer,
        probe_token_ids: cute.Pointer,
        probe_logits: cute.Tensor,
        rng_seed: cute.Pointer,
        rng_offset: cute.Pointer,
        logits_out: cute.Tensor,
        rows: Int32,
        vocab_offset: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        summary_stride: Int64,
        mask_stride: Int64,
        logits_stride: Int64,
    ):
        logits_dtype = logits_out.element_type
        thread, _, _ = cute.arch.thread_idx()
        n_tile, _, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        group = lane // 4
        quad = lane % 4
        first = Int32(n_tile) * 16
        row_a = cutlass.min(first + group, Int32(self.n - 1))
        row_b = cutlass.min(first + group + 8, Int32(self.n - 1))
        weight_base = get_ptr_as_int64(weight, Int64(0))
        x_base = get_ptr_as_int64(x, Int64(0))

        acc = cute.make_rmem_tensor((self.max_m_tiles, 4), Float32)
        for mt in cutlass.range_constexpr(self.max_m_tiles):
            for e in cutlass.range_constexpr(4):
                acc[mt, e] = Float32(0.0)

        for step in cutlass.range_constexpr(self.chunks_per_warp):
            chunk = warp * self.chunks_per_warp + step
            if cutlass.const_expr(self.chunk_k == 64):
                offset = Int64(chunk * 128 + quad * 32)
                pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
                pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wa4, wa5, wa6, wa7 = ld_global_nc_v4_u32(pa + Int64(16))
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wb4, wb5, wb6, wb7 = ld_global_nc_v4_u32(pb + Int64(16))
                wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
                wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                    px = x_base + Int64(token) * x_stride * 2 + offset
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    x4, x5, x6, x7 = ld_global_nc_v4_u32(px + Int64(16))
                    xs = (x0, x1, x2, x3, x4, x5, x6, x7)
                    a0, a1, a2, a3 = acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3]
                    for s in cutlass.range_constexpr(4):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa[2 * s],
                            wb[2 * s],
                            wa[2 * s + 1],
                            wb[2 * s + 1],
                            xs[2 * s],
                            xs[2 * s + 1],
                        )
                    acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3] = a0, a1, a2, a3
            else:
                offset = Int64(chunk * 64 + quad * 16)
                pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
                pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wa = (wa0, wa1, wa2, wa3)
                wb = (wb0, wb1, wb2, wb3)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                    px = x_base + Int64(token) * x_stride * 2 + offset
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    xs = (x0, x1, x2, x3)
                    a0, a1, a2, a3 = acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3]
                    for s in cutlass.range_constexpr(2):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa[2 * s],
                            wb[2 * s],
                            wa[2 * s + 1],
                            wb[2 * s + 1],
                            xs[2 * s],
                            xs[2 * s + 1],
                        )
                    acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3] = a0, a1, a2, a3

        storage = _SmemAllocator().allocate(self._shared_storage_cls())
        partials = storage.partials.get_tensor(
            cute.make_layout((self.warps * _LANES * 4 * self.max_m_tiles,))
        )
        if cutlass.const_expr(self.warps > 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                base = ((mt * self.warps + warp) * _LANES + lane) * 4
                for e in cutlass.range_constexpr(4):
                    partials[base + e] = acc[mt, e]
            cute.arch.sync_threads()

        f_lo = first + group
        f_hi = first + group + 8
        in_lo = f_lo < Int32(self.n)
        in_hi = f_hi < Int32(self.n)
        gcol_lo = vocab_offset + f_lo
        gcol_hi = vocab_offset + f_hi

        b_lo = Float32(0.0)
        b_hi = Float32(0.0)
        if cutlass.const_expr(self.use_bias):
            bias_base = Int64(logit_bias.toint())
            if in_lo:
                b_lo = ld_global_nc_f32(bias_base + Int64(f_lo) * Int64(4))
            if in_hi:
                b_hi = ld_global_nc_f32(bias_base + Int64(f_hi) * Int64(4))

        inv_t_base = Int64(inv_temperature.toint())
        mask_base = Int64(grammar_mask.toint())
        probe_id_base = Int64(probe_token_ids.toint())
        seed_base = Int64(rng_seed.toint())
        off_base = Int64(rng_offset.toint())
        word_col = Int64(first >> 5)
        bit_lo = f_lo & Int32(31)
        bit_hi = f_hi & Int32(31)
        neg_inf = u32_as_f32(Uint32(0xFF800000))

        for mt in cutlass.range_constexpr(self.max_m_tiles):
            if warp == Int32(mt % self.warps):
                t0 = acc[mt, 0] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                t1 = acc[mt, 1] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                t2 = acc[mt, 2] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                t3 = acc[mt, 3] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                if cutlass.const_expr(self.warps > 1):
                    for w in cutlass.range_constexpr(self.warps):
                        pbase = ((mt * self.warps + w) * _LANES + lane) * 4
                        t0 = t0 + partials[pbase + 0]
                        t1 = t1 + partials[pbase + 1]
                        t2 = t2 + partials[pbase + 2]
                        t3 = t3 + partials[pbase + 3]

                if cutlass.const_expr(self.use_bias):
                    t0 = t0 + b_lo
                    t1 = t1 + b_lo
                    t2 = t2 + b_hi
                    t3 = t3 + b_hi

                r0 = Int32(mt * 8) + quad * 2
                r1 = r0 + Int32(1)
                for r_sub in cutlass.range_constexpr(2):
                    row = r0 if cutlass.const_expr(r_sub == 0) else r1
                    z_lo = t0 if cutlass.const_expr(r_sub == 0) else t1
                    z_hi = t2 if cutlass.const_expr(r_sub == 0) else t3
                    row_active = row < rows
                    clamped_row = cutlass.min(row, rows - Int32(1))

                    is_greedy = cutlass.Boolean(False)
                    if cutlass.const_expr(self.use_inv_temp):
                        inv_t = ld_global_nc_f32(
                            inv_t_base + Int64(clamped_row) * Int64(4)
                        )
                        is_greedy = inv_t == Float32(0.0)
                        scale = Float32(1.0) if is_greedy else inv_t
                        z_lo = z_lo * scale
                        z_hi = z_hi * scale

                    v_lo = in_lo
                    v_hi = in_hi
                    if cutlass.const_expr(self.use_grammar):
                        mword = ld_global_nc_u32(
                            mask_base
                            + (Int64(clamped_row) * mask_stride + word_col) * Int64(4)
                        )
                        v_lo = v_lo and (_bit_allowed(mword, bit_lo) != Int32(0))
                        v_hi = v_hi and (_bit_allowed(mword, bit_hi) != Int32(0))

                    out_lo = z_lo if v_lo else neg_inf
                    out_hi = z_hi if v_hi else neg_inf

                    if cutlass.const_expr(self.write_logits):
                        if row_active:
                            if in_lo:
                                logits_out[Int64(row) * logits_stride + Int64(f_lo)] = (
                                    out_lo.to(logits_dtype)
                                )
                            if in_hi:
                                logits_out[Int64(row) * logits_stride + Int64(f_hi)] = (
                                    out_hi.to(logits_dtype)
                                )

                    if cutlass.const_expr(self.use_probe):
                        if row_active:
                            pid = _ld_global_nc_i64(
                                probe_id_base + Int64(row) * Int64(8)
                            )
                            local_p = pid - Int64(vocab_offset)
                            if local_p == Int64(f_lo) and in_lo:
                                probe_logits[Int64(row)] = out_lo
                            if local_p == Int64(f_hi) and in_hi:
                                probe_logits[Int64(row)] = out_hi
                            if (
                                Int32(n_tile) == Int32(0)
                                and group == Int32(0)
                                and (local_p < Int64(0) or local_p >= Int64(self.n))
                            ):
                                probe_logits[Int64(row)] = neg_inf

                    seed_val = Int64(0)
                    off_val = Int64(0)
                    if cutlass.const_expr(self.use_gumbel):
                        seed_val = _ld_global_nc_i64(
                            seed_base + Int64(clamped_row) * Int64(8)
                        )
                        off_val = _ld_global_nc_i64(
                            off_base + Int64(clamped_row) * Int64(8)
                        )

                    m_val, s_exp, m_idx, g_val, g_idx = _reduce_16col_row(
                        z_lo,
                        z_hi,
                        v_lo,
                        v_hi,
                        gcol_lo,
                        gcol_hi,
                        is_greedy,
                        seed_val,
                        off_val,
                        self.use_gumbel,
                    )

                    if row_active and group == Int32(0):
                        dst = Int64(row) * summary_stride + Int64(n_tile)
                        tile_max[dst] = m_val
                        tile_sum_exp[dst] = s_exp
                        tile_argmax[dst] = m_idx
                        if cutlass.const_expr(self.use_gumbel):
                            tile_gumbel_max[dst] = g_val
                            tile_gumbel_arg[dst] = g_idx


# =============================================================================
# Stage-1 Kernel B: Large-M (max_rows > 32, e.g. M = 64, 128), TILE_N = 64
# =============================================================================


class TensorCoreVocabSummaryLargeMKernel:
    """Fused BF16 GEMV + 64-column vocab summary for M > 32 (4 N-warps x 2 M-warps)."""

    def __init__(
        self,
        n: int,
        k: int,
        max_rows: int,
        *,
        use_inv_temp: bool = False,
        use_bias: bool = False,
        use_grammar: bool = False,
        use_probe: bool = False,
        use_gumbel: bool = False,
        write_logits: bool = False,
    ):
        if k <= 0 or k % TC_K_MULTIPLE:
            raise ValueError(
                f"vocab-summary TC-GEMV requires K to be a positive multiple of {TC_K_MULTIPLE}"
            )
        self.n = int(n)
        self.k = int(k)
        self.chunk_k = tc_chunk_k(k)
        self.chunks = self.k // self.chunk_k
        self.block_max_rows = min(int(max_rows), _LARGE_M_BLOCK)
        self.m_tiles_block = (self.block_max_rows + 7) // 8
        self.mt_per_wm = (self.m_tiles_block + 1) // 2
        self.warps_n = 4
        self.warps_m = 2
        self.warps = self.warps_n * self.warps_m
        self.num_tiles = (self.n + 63) // 64
        self.use_inv_temp = bool(use_inv_temp)
        self.use_bias = bool(use_bias)
        self.use_grammar = bool(use_grammar)
        self.use_probe = bool(use_probe)
        self.use_gumbel = bool(use_gumbel)
        self.write_logits = bool(write_logits)

    def _shared_storage_cls(self):
        # Each (wm, local_mt, quad, r_sub) row has 4 wn entries.
        # Slot count = warps_m * mt_per_wm * 8 * warps_n.
        slots = self.warps_m * self.mt_per_wm * 8 * self.warps_n

        class SharedStorage:
            pass

        ann = {
            "s_max": cute.struct.Align[cute.struct.MemRange[Float32, slots], 16],
            "s_sum": cute.struct.Align[cute.struct.MemRange[Float32, slots], 16],
            "s_arg": cute.struct.Align[cute.struct.MemRange[Int32, slots], 16],
        }
        if self.use_gumbel:
            ann["s_gmax"] = cute.struct.Align[cute.struct.MemRange[Float32, slots], 16]
            ann["s_garg"] = cute.struct.Align[cute.struct.MemRange[Int32, slots], 16]
        SharedStorage.__annotations__ = ann
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        tile_max: cute.Pointer,
        tile_sum_exp: cute.Pointer,
        tile_argmax: cute.Pointer,
        tile_gumbel_max: cute.Pointer,
        tile_gumbel_arg: cute.Pointer,
        inv_temperature: cute.Pointer,
        logit_bias: cute.Pointer,
        grammar_mask: cute.Pointer,
        probe_token_ids: cute.Pointer,
        probe_logits: cute.Pointer,
        rng_seed: cute.Pointer,
        rng_offset: cute.Pointer,
        logits_out: cute.Pointer,
        rows: Int32,
        vocab_offset: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        summary_stride: Int64,
        mask_stride: Int64,
        logits_stride: Int64,
        stream: cuda.CUstream,
    ):
        m_blocks = (rows + Int32(_LARGE_M_BLOCK - 1)) // Int32(_LARGE_M_BLOCK)
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(tile_max),
            _flat(tile_sum_exp),
            _flat(tile_argmax),
            _flat(tile_gumbel_max),
            _flat(tile_gumbel_arg),
            inv_temperature,
            logit_bias,
            grammar_mask,
            probe_token_ids,
            _flat(probe_logits),
            rng_seed,
            rng_offset,
            _flat(logits_out),
            rows,
            vocab_offset,
            x_stride,
            weight_stride,
            summary_stride,
            mask_stride,
            logits_stride,
        ).launch(
            grid=(self.num_tiles, m_blocks, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        tile_max: cute.Tensor,
        tile_sum_exp: cute.Tensor,
        tile_argmax: cute.Tensor,
        tile_gumbel_max: cute.Tensor,
        tile_gumbel_arg: cute.Tensor,
        inv_temperature: cute.Pointer,
        logit_bias: cute.Pointer,
        grammar_mask: cute.Pointer,
        probe_token_ids: cute.Pointer,
        probe_logits: cute.Tensor,
        rng_seed: cute.Pointer,
        rng_offset: cute.Pointer,
        logits_out: cute.Tensor,
        rows: Int32,
        vocab_offset: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        summary_stride: Int64,
        mask_stride: Int64,
        logits_stride: Int64,
    ):
        logits_dtype = logits_out.element_type
        thread, _, _ = cute.arch.thread_idx()
        n_tile, m_block, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        wn = warp // Int32(2)
        wm = warp % Int32(2)
        group = lane // 4
        quad = lane % 4

        first = Int32(n_tile) * 64 + wn * 16
        row_a = cutlass.min(first + group, Int32(self.n - 1))
        row_b = cutlass.min(first + group + 8, Int32(self.n - 1))
        m_row_origin = Int32(m_block) * Int32(_LARGE_M_BLOCK)
        weight_base = get_ptr_as_int64(weight, Int64(0))
        x_base = get_ptr_as_int64(x, Int64(0))

        acc = cute.make_rmem_tensor((self.mt_per_wm, 4), Float32)
        for lmt in cutlass.range_constexpr(self.mt_per_wm):
            for e in cutlass.range_constexpr(4):
                acc[lmt, e] = Float32(0.0)

        for chunk in range(self.chunks):
            if cutlass.const_expr(self.chunk_k == 64):
                offset = Int64(chunk * 128 + quad * 32)
                pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
                pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wa4, wa5, wa6, wa7 = ld_global_nc_v4_u32(pa + Int64(16))
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wb4, wb5, wb6, wb7 = ld_global_nc_v4_u32(pb + Int64(16))
                wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
                wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
                for lmt in cutlass.range_constexpr(self.mt_per_wm):
                    mt = wm + Int32(lmt * 2)
                    token = cutlass.min(m_row_origin + mt * 8 + group, rows - Int32(1))
                    px = x_base + Int64(token) * x_stride * 2 + offset
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    x4, x5, x6, x7 = ld_global_nc_v4_u32(px + Int64(16))
                    xs = (x0, x1, x2, x3, x4, x5, x6, x7)
                    a0, a1, a2, a3 = acc[lmt, 0], acc[lmt, 1], acc[lmt, 2], acc[lmt, 3]
                    for s in cutlass.range_constexpr(4):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa[2 * s],
                            wb[2 * s],
                            wa[2 * s + 1],
                            wb[2 * s + 1],
                            xs[2 * s],
                            xs[2 * s + 1],
                        )
                    acc[lmt, 0], acc[lmt, 1], acc[lmt, 2], acc[lmt, 3] = a0, a1, a2, a3
            else:
                offset = Int64(chunk * 64 + quad * 16)
                pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
                pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wa = (wa0, wa1, wa2, wa3)
                wb = (wb0, wb1, wb2, wb3)
                for lmt in cutlass.range_constexpr(self.mt_per_wm):
                    mt = wm + Int32(lmt * 2)
                    token = cutlass.min(m_row_origin + mt * 8 + group, rows - Int32(1))
                    px = x_base + Int64(token) * x_stride * 2 + offset
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    xs = (x0, x1, x2, x3)
                    a0, a1, a2, a3 = acc[lmt, 0], acc[lmt, 1], acc[lmt, 2], acc[lmt, 3]
                    for s in cutlass.range_constexpr(2):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa[2 * s],
                            wb[2 * s],
                            wa[2 * s + 1],
                            wb[2 * s + 1],
                            xs[2 * s],
                            xs[2 * s + 1],
                        )
                    acc[lmt, 0], acc[lmt, 1], acc[lmt, 2], acc[lmt, 3] = a0, a1, a2, a3

        f_lo = first + group
        f_hi = first + group + 8
        in_lo = f_lo < Int32(self.n)
        in_hi = f_hi < Int32(self.n)
        gcol_lo = vocab_offset + f_lo
        gcol_hi = vocab_offset + f_hi

        b_lo = Float32(0.0)
        b_hi = Float32(0.0)
        if cutlass.const_expr(self.use_bias):
            bias_base = Int64(logit_bias.toint())
            if in_lo:
                b_lo = ld_global_nc_f32(bias_base + Int64(f_lo) * Int64(4))
            if in_hi:
                b_hi = ld_global_nc_f32(bias_base + Int64(f_hi) * Int64(4))

        inv_t_base = Int64(inv_temperature.toint())
        mask_base = Int64(grammar_mask.toint())
        probe_id_base = Int64(probe_token_ids.toint())
        seed_base = Int64(rng_seed.toint())
        off_base = Int64(rng_offset.toint())
        word_col = Int64(first >> 5)
        bit_lo = f_lo & Int32(31)
        bit_hi = f_hi & Int32(31)
        neg_inf = u32_as_f32(Uint32(0xFF800000))
        pos_inf = u32_as_f32(Uint32(0x7F800000))

        slots = self.warps_m * self.mt_per_wm * 8 * self.warps_n
        storage = _SmemAllocator().allocate(self._shared_storage_cls())
        s_max = storage.s_max.get_tensor(cute.make_layout((slots,)))
        s_sum = storage.s_sum.get_tensor(cute.make_layout((slots,)))
        s_arg = storage.s_arg.get_tensor(cute.make_layout((slots,)))
        if cutlass.const_expr(self.use_gumbel):
            s_gmax = storage.s_gmax.get_tensor(cute.make_layout((slots,)))
            s_garg = storage.s_garg.get_tensor(cute.make_layout((slots,)))

        for lmt in cutlass.range_constexpr(self.mt_per_wm):
            t0 = acc[lmt, 0]
            t1 = acc[lmt, 1]
            t2 = acc[lmt, 2]
            t3 = acc[lmt, 3]
            if cutlass.const_expr(self.use_bias):
                t0 = t0 + b_lo
                t1 = t1 + b_lo
                t2 = t2 + b_hi
                t3 = t3 + b_hi

            mt = wm + Int32(lmt * 2)
            r0 = m_row_origin + mt * 8 + quad * 2
            r1 = r0 + Int32(1)
            for r_sub in cutlass.range_constexpr(2):
                row = r0 if cutlass.const_expr(r_sub == 0) else r1
                z_lo = t0 if cutlass.const_expr(r_sub == 0) else t1
                z_hi = t2 if cutlass.const_expr(r_sub == 0) else t3
                row_active = (row < rows) and (
                    (mt * 8 + quad * 2 + Int32(r_sub)) < Int32(_LARGE_M_BLOCK)
                )
                clamped_row = cutlass.min(row, rows - Int32(1))

                is_greedy = cutlass.Boolean(False)
                if cutlass.const_expr(self.use_inv_temp):
                    inv_t = ld_global_nc_f32(inv_t_base + Int64(clamped_row) * Int64(4))
                    is_greedy = inv_t == Float32(0.0)
                    scale = Float32(1.0) if is_greedy else inv_t
                    z_lo = z_lo * scale
                    z_hi = z_hi * scale

                v_lo = in_lo
                v_hi = in_hi
                if cutlass.const_expr(self.use_grammar):
                    mword = ld_global_nc_u32(
                        mask_base
                        + (Int64(clamped_row) * mask_stride + word_col) * Int64(4)
                    )
                    v_lo = v_lo and (_bit_allowed(mword, bit_lo) != Int32(0))
                    v_hi = v_hi and (_bit_allowed(mword, bit_hi) != Int32(0))

                out_lo = z_lo if v_lo else neg_inf
                out_hi = z_hi if v_hi else neg_inf

                if cutlass.const_expr(self.write_logits):
                    if row_active:
                        if in_lo:
                            logits_out[Int64(row) * logits_stride + Int64(f_lo)] = (
                                out_lo.to(logits_dtype)
                            )
                        if in_hi:
                            logits_out[Int64(row) * logits_stride + Int64(f_hi)] = (
                                out_hi.to(logits_dtype)
                            )

                if cutlass.const_expr(self.use_probe):
                    if row_active:
                        pid = _ld_global_nc_i64(probe_id_base + Int64(row) * Int64(8))
                        local_p = pid - Int64(vocab_offset)
                        if local_p == Int64(f_lo) and in_lo:
                            probe_logits[Int64(row)] = out_lo
                        if local_p == Int64(f_hi) and in_hi:
                            probe_logits[Int64(row)] = out_hi
                        if (
                            Int32(n_tile) == Int32(0)
                            and wn == Int32(0)
                            and group == Int32(0)
                            and (local_p < Int64(0) or local_p >= Int64(self.n))
                        ):
                            probe_logits[Int64(row)] = neg_inf

                seed_val = Int64(0)
                off_val = Int64(0)
                if cutlass.const_expr(self.use_gumbel):
                    seed_val = _ld_global_nc_i64(
                        seed_base + Int64(clamped_row) * Int64(8)
                    )
                    off_val = _ld_global_nc_i64(
                        off_base + Int64(clamped_row) * Int64(8)
                    )

                m_val, s_exp, m_idx, g_val, g_idx = _reduce_16col_row(
                    z_lo,
                    z_hi,
                    v_lo,
                    v_hi,
                    gcol_lo,
                    gcol_hi,
                    is_greedy,
                    seed_val,
                    off_val,
                    self.use_gumbel,
                )

                if group == Int32(0):
                    row_slot = (wm * Int32(self.mt_per_wm) + Int32(lmt)) * Int32(8) + (
                        quad * Int32(2) + Int32(r_sub)
                    )
                    slot_idx = row_slot * Int32(4) + wn
                    s_max[slot_idx] = m_val
                    s_sum[slot_idx] = s_exp
                    s_arg[slot_idx] = m_idx
                    if cutlass.const_expr(self.use_gumbel):
                        s_gmax[slot_idx] = g_val
                        s_garg[slot_idx] = g_idx

        cute.arch.sync_threads()

        # wn == 0 combines the 4 16-column subtile summaries into one 64-column summary.
        if wn == Int32(0) and group == Int32(0):
            for lmt in cutlass.range_constexpr(self.mt_per_wm):
                mt = wm + Int32(lmt * 2)
                for r_sub in cutlass.range_constexpr(2):
                    local_row = mt * 8 + quad * 2 + Int32(r_sub)
                    row = m_row_origin + local_row
                    if row < rows and local_row < Int32(_LARGE_M_BLOCK):
                        row_slot = (wm * Int32(self.mt_per_wm) + Int32(lmt)) * Int32(
                            8
                        ) + (quad * Int32(2) + Int32(r_sub))
                        base = row_slot * Int32(4)
                        m0, m1, m2, m3 = (
                            s_max[base + 0],
                            s_max[base + 1],
                            s_max[base + 2],
                            s_max[base + 3],
                        )
                        s0, s1, s2, s3 = (
                            s_sum[base + 0],
                            s_sum[base + 1],
                            s_sum[base + 2],
                            s_sum[base + 3],
                        )
                        i0, i1, i2, i3 = (
                            s_arg[base + 0],
                            s_arg[base + 1],
                            s_arg[base + 2],
                            s_arg[base + 3],
                        )
                        cm, ci = _argmax_pair(m0, i0, m1, i1)
                        cm, ci = _argmax_pair(cm, ci, m2, i2)
                        cm, ci = _argmax_pair(cm, ci, m3, i3)
                        is_fin = (cm > neg_inf) and (cm < pos_inf)
                        origin = cm if is_fin else Float32(0.0)
                        cs = Float32(0.0)
                        if (m0 > neg_inf) and (m0 < pos_inf) and (s0 > Float32(0.0)):
                            cs = cs + s0 * cute.math.exp(m0 - origin)
                        if (m1 > neg_inf) and (m1 < pos_inf) and (s1 > Float32(0.0)):
                            cs = cs + s1 * cute.math.exp(m1 - origin)
                        if (m2 > neg_inf) and (m2 < pos_inf) and (s2 > Float32(0.0)):
                            cs = cs + s2 * cute.math.exp(m2 - origin)
                        if (m3 > neg_inf) and (m3 < pos_inf) and (s3 > Float32(0.0)):
                            cs = cs + s3 * cute.math.exp(m3 - origin)

                        dst = Int64(row) * summary_stride + Int64(n_tile)
                        tile_max[dst] = cm
                        tile_sum_exp[dst] = cs
                        tile_argmax[dst] = ci

                        if cutlass.const_expr(self.use_gumbel):
                            cg, cgi = _argmax_pair(
                                s_gmax[base + 0],
                                s_garg[base + 0],
                                s_gmax[base + 1],
                                s_garg[base + 1],
                            )
                            cg, cgi = _argmax_pair(
                                cg, cgi, s_gmax[base + 2], s_garg[base + 2]
                            )
                            cg, cgi = _argmax_pair(
                                cg, cgi, s_gmax[base + 3], s_garg[base + 3]
                            )
                            tile_gumbel_max[dst] = cg
                            tile_gumbel_arg[dst] = cgi


# =============================================================================
# Stage-1 Kernel C: TMA Pipelined Large-M (M > 32, K % 64 == 0), TILE_N = 64
# =============================================================================


@cute.jit
def _tma_warp_gemm_f32(
    tiled_mma,
    acc,
    fragment_a,
    fragment_b,
    shared_a,
    shared_b,
    copy_a,
    copy_b,
):
    """Full-throughput FP32-accumulating m16n8k16 warp GEMM over one K64 stage."""
    target_a = copy_a.retile(fragment_a)
    target_b = copy_b.retile(fragment_b)
    cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
    cute.copy(copy_b, shared_b[None, None, 0], target_b[None, None, 0])
    for k_step in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
        if k_step < cute.size(shared_a.shape[2]) - 1:
            cute.copy(
                copy_a,
                shared_a[None, None, k_step + 1],
                target_a[None, None, k_step + 1],
            )
            cute.copy(
                copy_b,
                shared_b[None, None, k_step + 1],
                target_b[None, None, k_step + 1],
            )
        cute.gemm(
            tiled_mma,
            acc,
            fragment_a[None, None, k_step],
            fragment_b[None, None, k_step],
            acc,
        )


class TmaVocabSummaryLargeMKernel:
    """TMA-pipelined BF16 lm_head + register-only 64-col vocab summary for M > 32.

    Each compute warp owns 16 rows x 64 vocabulary columns in ``m16n8k16``
    registers, so the 4 lanes with identical ``group = lane // 4`` hold all 64
    columns of 2 rows.  Two warp butterfly shuffles (offsets 1, 2) reduce the
    entire 64-column tile in registers with zero shared-memory epilogue traffic.
    """

    tile_n = 64
    tile_k = 64
    buffer_align_bytes = 128

    def __init__(
        self,
        n: int,
        k: int,
        tile_m: int,
        *,
        use_inv_temp: bool = False,
        use_bias: bool = False,
        use_grammar: bool = False,
        use_probe: bool = False,
        use_gumbel: bool = False,
        write_logits: bool = False,
    ):
        if k <= 0 or k % self.tile_k != 0:
            raise ValueError("TMA vocab-summary kernel requires K divisible by 64")
        if tile_m not in (64, 128):
            raise ValueError("TMA vocab-summary tile_m must be 64 or 128")
        self.n = int(n)
        self.k = int(k)
        self.tile_m = int(tile_m)
        # tile_m=64 fits 3 CTAs/SM at 2 stages (32 KiB/CTA); tile_m=128 runs
        # 1 CTA/SM due to register footprint, so 3 stages (72 KiB) hides cold DRAM.
        self.num_stages = 3 if self.tile_m == 128 else 2
        self.num_compute_warps = self.tile_m // 16
        self.producer_warp = self.num_compute_warps
        self.num_threads = (self.num_compute_warps + 1) * _LANES
        self.k_tiles = self.k // self.tile_k
        self.num_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.use_inv_temp = bool(use_inv_temp)
        self.use_bias = bool(use_bias)
        self.use_grammar = bool(use_grammar)
        self.use_probe = bool(use_probe)
        self.use_gumbel = bool(use_gumbel)
        self.write_logits = bool(write_logits)

    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    def _get_smem_layouts(self) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                self.tile_k,
            ),
            cutlass.BFloat16,
        )
        sA_layout = cute.tile_to_shape(
            layout_atom,
            (self.tile_m, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        sB_layout = cute.tile_to_shape(
            layout_atom,
            (self.tile_n, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        return sA_layout, sB_layout

    def _get_shared_storage_cls(
        self,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sA": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)],
                self.buffer_align_bytes,
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sB_layout)],
                self.buffer_align_bytes,
            ],
            "mbar_ptr": cute.struct.Align[
                cute.struct.MemRange[Int64, self.num_stages * 2], 8
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        tile_max: cute.Pointer,
        tile_sum_exp: cute.Pointer,
        tile_argmax: cute.Pointer,
        tile_gumbel_max: cute.Pointer,
        tile_gumbel_arg: cute.Pointer,
        inv_temperature: cute.Pointer,
        logit_bias: cute.Pointer,
        grammar_mask: cute.Pointer,
        probe_token_ids: cute.Pointer,
        probe_logits: cute.Pointer,
        rng_seed: cute.Pointer,
        rng_offset: cute.Pointer,
        logits_out: cute.Pointer,
        rows: Int32,
        vocab_offset: Int32,
        summary_stride: Int64,
        mask_stride: Int64,
        logits_stride: Int64,
        stream: cuda.CUstream,
    ):
        sA_layout, sB_layout = self._get_smem_layouts()
        tiled_mma = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
        sA_tma_layout = cute.slice_(sA_layout, (None, None, 0))
        sB_tma_layout = cute.slice_(sB_layout, (None, None, 0))
        src_tma = _assume_tma_source_aligned(source)
        w_tma = _assume_tma_source_aligned(weight)
        tma_atom_A, tma_tensor_A = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            src_tma,
            sA_tma_layout,
            (self.tile_m, self.tile_k),
            num_multicast=1,
        )
        tma_atom_B, tma_tensor_B = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w_tma,
            sB_tma_layout,
            (self.tile_n, self.tile_k),
            num_multicast=1,
        )
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            tma_atom_A,
            tma_atom_B,
            sA_layout,
            sB_layout,
            tiled_mma,
            SharedStorage,
            _flat(tile_max),
            _flat(tile_sum_exp),
            _flat(tile_argmax),
            _flat(tile_gumbel_max),
            _flat(tile_gumbel_arg),
            inv_temperature,
            logit_bias,
            grammar_mask,
            probe_token_ids,
            _flat(probe_logits),
            rng_seed,
            rng_offset,
            _flat(logits_out),
            rows,
            vocab_offset,
            summary_stride,
            mask_stride,
            logits_stride,
        ).launch(
            grid=(grid_m, self.num_tiles, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_max: cute.Tensor,
        tile_sum_exp: cute.Tensor,
        tile_argmax: cute.Tensor,
        tile_gumbel_max: cute.Tensor,
        tile_gumbel_arg: cute.Tensor,
        inv_temperature: cute.Pointer,
        logit_bias: cute.Pointer,
        grammar_mask: cute.Pointer,
        probe_token_ids: cute.Pointer,
        probe_logits: cute.Tensor,
        rng_seed: cute.Pointer,
        rng_offset: cute.Pointer,
        logits_out: cute.Tensor,
        rows: Int32,
        vocab_offset: Int32,
        summary_stride: Int64,
        mask_stride: Int64,
        logits_stride: Int64,
    ):
        logits_dtype = logits_out.element_type
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_B)

        storage = _SmemAllocator().allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (
            (self.tile_m + self.tile_n) * self.tile_k * cutlass.BFloat16.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_compute_warps,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        gA = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None))
        gB = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None))
        cta_layout = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_A,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_B,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_stages,
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
            acc = cute.make_rmem_tensor(acc_shape, Float32)
            acc.fill(0.0)
            smem_copy_atom_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            smem_thr_copy_A = cute.make_tiled_copy_A(
                smem_copy_atom_A, tiled_mma
            ).get_slice(tidx)
            smem_thr_copy_B = cute.make_tiled_copy_B(
                smem_copy_atom_B, tiled_mma
            ).get_slice(tidx)
            tSsA = smem_thr_copy_A.partition_S(sA)
            tSsB = smem_thr_copy_B.partition_S(sB)

            for _k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.consumer_wait(consumer_state)
                _tma_warp_gemm_f32(
                    thr_mma,
                    acc,
                    tCrA,
                    tCrB,
                    tSsA[None, None, None, consumer_state.index],
                    tSsB[None, None, None, consumer_state.index],
                    smem_thr_copy_A,
                    smem_thr_copy_B,
                )
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # Epilogue: each warp owns 16 rows x 64 columns in registers.
            # Within a warp, lanes with the same group = lane // 4 own the same
            # 2 rows (r0, r1) across all 64 columns -> 2 shuffles (offset 1, 2)!
            lane = Int32(tidx) % _LANES
            group = lane // 4
            quad = lane % 4
            col_origin = Int32(n_tile) * Int32(self.tile_n)
            row_origin = Int32(m_tile) * Int32(self.tile_m) + warp_idx * Int32(16)
            r0 = row_origin + group
            r1 = r0 + Int32(8)

            coordinates = thr_mma.partition_C(
                cute.make_identity_tensor((self.tile_m, self.tile_n))
            )
            neg_inf = u32_as_f32(Uint32(0xFF800000))
            pos_inf = u32_as_f32(Uint32(0x7F800000))
            inv_t_base = Int64(inv_temperature.toint())
            bias_base = Int64(logit_bias.toint())
            mask_base = Int64(grammar_mask.toint())
            probe_id_base = Int64(probe_token_ids.toint())
            seed_base = Int64(rng_seed.toint())
            off_base = Int64(rng_offset.toint())
            word0_col = Int64(n_tile) * Int64(2)
            word1_col = word0_col + Int64(1)
            n_words = Int64((self.n + 31) // 32)

            # Preload per-column bias once for this thread's 16 columns.
            col_bias = cute.make_rmem_tensor((16,), Float32)
            if cutlass.const_expr(self.use_bias):
                for ns in cutlass.range_constexpr(8):
                    for cs in cutlass.range_constexpr(2):
                        local_c = Int32(coordinates[ns * 4 + cs][1])
                        col = col_origin + local_c
                        bval = Float32(0.0)
                        if col < Int32(self.n):
                            bval = ld_global_nc_f32(bias_base + Int64(col) * Int64(4))
                        col_bias[ns * 2 + cs] = bval

            z_row = cute.make_rmem_tensor((16,), Float32)
            v_row = cute.make_rmem_tensor((16,), Int32)

            for r_sub in cutlass.range_constexpr(2):
                row = r0 if cutlass.const_expr(r_sub == 0) else r1
                row_active = row < rows
                clamped_row = cutlass.min(row, rows - Int32(1))

                is_greedy = cutlass.Boolean(False)
                scale = Float32(1.0)
                if cutlass.const_expr(self.use_inv_temp):
                    inv_t = ld_global_nc_f32(inv_t_base + Int64(clamped_row) * Int64(4))
                    is_greedy = inv_t == Float32(0.0)
                    scale = Float32(1.0) if is_greedy else inv_t

                mword0 = Uint32(0xFFFFFFFF)
                mword1 = Uint32(0xFFFFFFFF)
                if cutlass.const_expr(self.use_grammar):
                    row_mbase = mask_base + Int64(clamped_row) * mask_stride * Int64(4)
                    if word0_col < n_words:
                        mword0 = ld_global_nc_u32(row_mbase + word0_col * Int64(4))
                    else:
                        mword0 = Uint32(0)
                    if word1_col < n_words:
                        mword1 = ld_global_nc_u32(row_mbase + word1_col * Int64(4))
                    else:
                        mword1 = Uint32(0)

                local_p = Int64(-1)
                if cutlass.const_expr(self.use_probe):
                    if row_active:
                        pid = _ld_global_nc_i64(probe_id_base + Int64(row) * Int64(8))
                        local_p = pid - Int64(vocab_offset)
                        if (
                            Int32(n_tile) == Int32(0)
                            and quad == Int32(0)
                            and (local_p < Int64(0) or local_p >= Int64(self.n))
                        ):
                            probe_logits[Int64(row)] = neg_inf

                seed_val = Int64(0)
                off_val = Int64(0)
                if cutlass.const_expr(self.use_gumbel):
                    seed_val = _ld_global_nc_i64(
                        seed_base + Int64(clamped_row) * Int64(8)
                    )
                    off_val = _ld_global_nc_i64(
                        off_base + Int64(clamped_row) * Int64(8)
                    )

                m_val = neg_inf
                m_idx = Int32(-1)
                g_val = neg_inf
                g_idx = Int32(-1)

                for ns in cutlass.range_constexpr(8):
                    for cs in cutlass.range_constexpr(2):
                        acc_idx = (
                            ns * 4 + (0 if cutlass.const_expr(r_sub == 0) else 2) + cs
                        )
                        local_c = Int32(coordinates[acc_idx][1])
                        col = col_origin + local_c
                        in_n = col < Int32(self.n)
                        val = acc[acc_idx]
                        if cutlass.const_expr(self.use_bias):
                            val = val + col_bias[ns * 2 + cs]
                        if cutlass.const_expr(self.use_inv_temp):
                            val = val * scale

                        valid = in_n
                        if cutlass.const_expr(self.use_grammar):
                            mw = mword0 if local_c < Int32(32) else mword1
                            valid = valid and (
                                _bit_allowed(mw, local_c & Int32(31)) != Int32(0)
                            )

                        out_z = val if valid else neg_inf
                        z_row[ns * 2 + cs] = out_z
                        v_row[ns * 2 + cs] = Int32(1) if valid else Int32(0)

                        if cutlass.const_expr(self.write_logits):
                            if row_active and in_n:
                                logits_out[Int64(row) * logits_stride + Int64(col)] = (
                                    out_z.to(logits_dtype)
                                )
                        if cutlass.const_expr(self.use_probe):
                            if row_active and in_n and local_p == Int64(col):
                                probe_logits[Int64(row)] = out_z

                        gcol = vocab_offset + col
                        ci = gcol if valid else Int32(-1)
                        m_val, m_idx = _argmax_pair(m_val, m_idx, out_z, ci)
                        if cutlass.const_expr(self.use_gumbel):
                            noise = (
                                Float32(0.0)
                                if (is_greedy or (not valid))
                                else _philox_gumbel_noise(seed_val, off_val, gcol)
                            )
                            cz = (out_z + noise) if valid else neg_inf
                            g_val, g_idx = _argmax_pair(g_val, g_idx, cz, ci)

                # Reduce across the 4 quad lanes in the same group (offsets 1 and 2).
                for shfl_off in cutlass.range_constexpr(2):
                    off = Int32(1 << shfl_off)
                    om = _shfl_bfly_f32(m_val, off)
                    oi = _shfl_bfly_i32(m_idx, off)
                    m_val, m_idx = _argmax_pair(m_val, m_idx, om, oi)
                    if cutlass.const_expr(self.use_gumbel):
                        og = _shfl_bfly_f32(g_val, off)
                        ogi = _shfl_bfly_i32(g_idx, off)
                        g_val, g_idx = _argmax_pair(g_val, g_idx, og, ogi)

                is_fin = (m_val > neg_inf) and (m_val < pos_inf)
                stable_max = m_val if is_fin else Float32(0.0)
                s_exp = Float32(0.0)
                for j in cutlass.range_constexpr(16):
                    if v_row[j] != Int32(0):
                        s_exp = s_exp + cute.math.exp(z_row[j] - stable_max)
                s_exp = s_exp + _shfl_bfly_f32(s_exp, Int32(1))
                s_exp = s_exp + _shfl_bfly_f32(s_exp, Int32(2))

                if row_active and quad == Int32(0):
                    dst = Int64(row) * summary_stride + Int64(n_tile)
                    tile_max[dst] = m_val
                    tile_sum_exp[dst] = s_exp
                    tile_argmax[dst] = m_idx
                    if cutlass.const_expr(self.use_gumbel):
                        tile_gumbel_max[dst] = g_val
                        tile_gumbel_arg[dst] = g_idx

        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_stages,
            )
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.arch.fence_view_async_shared()
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, m_tile, k_tile)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_B,
                    tBgB[(None, n_tile, k_tile)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


# =============================================================================
# Stage-2 Kernel: L2-Resident Tile Summary Reduction (grid = (M, 1, 1))
# =============================================================================


class VocabTileSummaryReduceKernel:
    """Two-pass L2-resident tile summary reducer producing exact LSE, argmax, Gumbel, probe_prob."""

    def __init__(
        self,
        *,
        use_gumbel: bool = False,
        use_probe: bool = False,
        use_inv_temp: bool = False,
    ):
        self.use_gumbel = bool(use_gumbel)
        self.use_probe = bool(use_probe)
        self.use_inv_temp = bool(use_inv_temp)

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        ann = {
            "w_max": cute.struct.Align[
                cute.struct.MemRange[Float32, _REDUCE_WARPS], 16
            ],
            "w_arg": cute.struct.Align[cute.struct.MemRange[Int32, _REDUCE_WARPS], 16],
            "w_sum": cute.struct.Align[
                cute.struct.MemRange[Float32, _REDUCE_WARPS], 16
            ],
            "bcast_max": cute.struct.Align[cute.struct.MemRange[Float32, 1], 4],
            "bcast_arg": cute.struct.Align[cute.struct.MemRange[Int32, 1], 4],
        }
        if self.use_gumbel:
            ann["w_gmax"] = cute.struct.Align[
                cute.struct.MemRange[Float32, _REDUCE_WARPS], 16
            ]
            ann["w_garg"] = cute.struct.Align[
                cute.struct.MemRange[Int32, _REDUCE_WARPS], 16
            ]
            ann["bcast_gmax"] = cute.struct.Align[cute.struct.MemRange[Float32, 1], 4]
            ann["bcast_garg"] = cute.struct.Align[cute.struct.MemRange[Int32, 1], 4]
        SharedStorage.__annotations__ = ann
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        tile_max: cute.Pointer,
        tile_sum_exp: cute.Pointer,
        tile_argmax: cute.Pointer,
        tile_gumbel_max: cute.Pointer,
        tile_gumbel_arg: cute.Pointer,
        probe_logits: cute.Pointer,
        probe_token_ids: cute.Pointer,
        inv_temperature: cute.Pointer,
        global_max: cute.Pointer,
        global_sum_exp: cute.Pointer,
        global_lse: cute.Pointer,
        argmax_id: cute.Pointer,
        gumbel_max: cute.Pointer,
        gumbel_id: cute.Pointer,
        probe_prob: cute.Pointer,
        rows: Int32,
        num_tiles: Int32,
        summary_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tile_max,
            tile_sum_exp,
            tile_argmax,
            tile_gumbel_max,
            tile_gumbel_arg,
            probe_logits,
            probe_token_ids,
            inv_temperature,
            _flat(global_max),
            _flat(global_sum_exp),
            _flat(global_lse),
            _flat(argmax_id),
            _flat(gumbel_max),
            _flat(gumbel_id),
            _flat(probe_prob),
            num_tiles,
            summary_stride,
        ).launch(
            grid=(rows, 1, 1),
            block=(_REDUCE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tile_max: cute.Pointer,
        tile_sum_exp: cute.Pointer,
        tile_argmax: cute.Pointer,
        tile_gumbel_max: cute.Pointer,
        tile_gumbel_arg: cute.Pointer,
        probe_logits: cute.Pointer,
        probe_token_ids: cute.Pointer,
        inv_temperature: cute.Pointer,
        global_max: cute.Tensor,
        global_sum_exp: cute.Tensor,
        global_lse: cute.Tensor,
        argmax_id: cute.Tensor,
        gumbel_max: cute.Tensor,
        gumbel_id: cute.Tensor,
        probe_prob: cute.Tensor,
        num_tiles: Int32,
        summary_stride: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        t_i = Int32(tid)
        warp = t_i // _LANES
        lane = t_i % _LANES

        neg_inf = u32_as_f32(Uint32(0xFF800000))
        pos_inf = u32_as_f32(Uint32(0x7F800000))
        row_off = Int64(row) * summary_stride
        tmax_base = Int64(tile_max.toint()) + row_off * Int64(4)
        tsum_base = Int64(tile_sum_exp.toint()) + row_off * Int64(4)
        targ_base = Int64(tile_argmax.toint()) + row_off * Int64(4)
        tgmax_base = Int64(tile_gumbel_max.toint()) + row_off * Int64(4)
        tgarg_base = Int64(tile_gumbel_arg.toint()) + row_off * Int64(4)

        # Pass 1: exact max, argmax, and optional Gumbel-max across all tiles.
        best_m = neg_inf
        best_i = Int32(-1)
        best_gm = neg_inf
        best_gi = Int32(-1)
        idx = t_i
        while idx < num_tiles:
            off4 = Int64(idx) * Int64(4)
            m = ld_global_nc_f32(tmax_base + off4)
            ai = _ld_global_nc_i32(targ_base + off4)
            best_m, best_i = _argmax_pair(best_m, best_i, m, ai)
            if cutlass.const_expr(self.use_gumbel):
                gm = ld_global_nc_f32(tgmax_base + off4)
                gi = _ld_global_nc_i32(tgarg_base + off4)
                best_gm, best_gi = _argmax_pair(best_gm, best_gi, gm, gi)
            idx = idx + Int32(_REDUCE_THREADS)

        for s in cutlass.range_constexpr(5):
            sh = Int32(1 << s)
            om = _shfl_bfly_f32(best_m, sh)
            oi = _shfl_bfly_i32(best_i, sh)
            best_m, best_i = _argmax_pair(best_m, best_i, om, oi)
            if cutlass.const_expr(self.use_gumbel):
                ogm = _shfl_bfly_f32(best_gm, sh)
                ogi = _shfl_bfly_i32(best_gi, sh)
                best_gm, best_gi = _argmax_pair(best_gm, best_gi, ogm, ogi)

        storage = _SmemAllocator().allocate(self._shared_storage_cls())
        w_max = storage.w_max.get_tensor(cute.make_layout((_REDUCE_WARPS,)))
        w_arg = storage.w_arg.get_tensor(cute.make_layout((_REDUCE_WARPS,)))
        w_sum = storage.w_sum.get_tensor(cute.make_layout((_REDUCE_WARPS,)))
        bcast_max = storage.bcast_max.get_tensor(cute.make_layout((1,)))
        bcast_arg = storage.bcast_arg.get_tensor(cute.make_layout((1,)))
        if cutlass.const_expr(self.use_gumbel):
            w_gmax = storage.w_gmax.get_tensor(cute.make_layout((_REDUCE_WARPS,)))
            w_garg = storage.w_garg.get_tensor(cute.make_layout((_REDUCE_WARPS,)))
            bcast_gmax = storage.bcast_gmax.get_tensor(cute.make_layout((1,)))
            bcast_garg = storage.bcast_garg.get_tensor(cute.make_layout((1,)))

        if lane == Int32(0):
            w_max[warp] = best_m
            w_arg[warp] = best_i
            if cutlass.const_expr(self.use_gumbel):
                w_gmax[warp] = best_gm
                w_garg[warp] = best_gi
        cute.arch.sync_threads()

        if warp == Int32(0):
            wm_v = w_max[lane] if lane < Int32(_REDUCE_WARPS) else neg_inf
            wm_i = w_arg[lane] if lane < Int32(_REDUCE_WARPS) else Int32(-1)
            wgm_v = neg_inf
            wgm_i = Int32(-1)
            if cutlass.const_expr(self.use_gumbel):
                wgm_v = w_gmax[lane] if lane < Int32(_REDUCE_WARPS) else neg_inf
                wgm_i = w_garg[lane] if lane < Int32(_REDUCE_WARPS) else Int32(-1)
            for s in cutlass.range_constexpr(4):
                sh = Int32(1 << s)
                om = _shfl_bfly_f32(wm_v, sh)
                oi = _shfl_bfly_i32(wm_i, sh)
                wm_v, wm_i = _argmax_pair(wm_v, wm_i, om, oi)
                if cutlass.const_expr(self.use_gumbel):
                    ogm = _shfl_bfly_f32(wgm_v, sh)
                    ogi = _shfl_bfly_i32(wgm_i, sh)
                    wgm_v, wgm_i = _argmax_pair(wgm_v, wgm_i, ogm, ogi)
            if lane == Int32(0):
                bcast_max[Int32(0)] = wm_v
                bcast_arg[Int32(0)] = wm_i
                if cutlass.const_expr(self.use_gumbel):
                    bcast_gmax[Int32(0)] = wgm_v
                    bcast_garg[Int32(0)] = wgm_i
        cute.arch.sync_threads()

        row_max = bcast_max[Int32(0)]
        row_fin = (row_max > neg_inf) and (row_max < pos_inf)

        # Pass 2: exact rescaled sum-exp over L1-cached tile_max and L2-cached tile_sum_exp.
        local_sum = Float32(0.0)
        if row_fin:
            idx2 = t_i
            while idx2 < num_tiles:
                off4 = Int64(idx2) * Int64(4)
                m = ld_global_nc_f32(tmax_base + off4)
                s = ld_global_nc_f32(tsum_base + off4)
                if (m > neg_inf) and (m < pos_inf) and (s > Float32(0.0)):
                    local_sum = local_sum + s * cute.math.exp(m - row_max)
                idx2 = idx2 + Int32(_REDUCE_THREADS)

        for s in cutlass.range_constexpr(5):
            local_sum = local_sum + _shfl_bfly_f32(local_sum, Int32(1 << s))

        if lane == Int32(0):
            w_sum[warp] = local_sum
        cute.arch.sync_threads()

        if warp == Int32(0):
            ws = w_sum[lane] if lane < Int32(_REDUCE_WARPS) else Float32(0.0)
            for s in cutlass.range_constexpr(4):
                ws = ws + _shfl_bfly_f32(ws, Int32(1 << s))
            if lane == Int32(0):
                r_i64 = Int64(row)
                r_arg = bcast_arg[Int32(0)]
                valid_row = row_fin and (ws > Float32(0.0)) and (ws < pos_inf)
                lse = (row_max + cute.math.log(ws)) if valid_row else neg_inf
                global_max[r_i64] = row_max if valid_row else neg_inf
                global_sum_exp[r_i64] = ws if valid_row else Float32(0.0)
                global_lse[r_i64] = lse
                argmax_id[r_i64] = r_arg if valid_row else Int32(-1)
                if cutlass.const_expr(self.use_gumbel):
                    rgm = bcast_gmax[Int32(0)]
                    rgi = bcast_garg[Int32(0)]
                    g_valid = valid_row and (rgm > neg_inf) and (rgm < pos_inf)
                    gumbel_max[r_i64] = rgm if g_valid else neg_inf
                    gumbel_id[r_i64] = rgi if g_valid else Int32(-1)
                if cutlass.const_expr(self.use_probe):
                    p_logit = ld_global_nc_f32(
                        Int64(probe_logits.toint()) + r_i64 * Int64(4)
                    )
                    p_fin = (p_logit > neg_inf) and (p_logit < pos_inf)
                    is_greedy = cutlass.Boolean(False)
                    if cutlass.const_expr(self.use_inv_temp):
                        inv_t = ld_global_nc_f32(
                            Int64(inv_temperature.toint()) + r_i64 * Int64(4)
                        )
                        is_greedy = inv_t == Float32(0.0)
                    prob = Float32(0.0)
                    if valid_row and p_fin:
                        if is_greedy:
                            pid = _ld_global_nc_i64(
                                Int64(probe_token_ids.toint()) + r_i64 * Int64(8)
                            )
                            prob = Float32(1.0) if pid == Int64(r_arg) else Float32(0.0)
                        else:
                            prob = cute.math.exp(p_logit - lse)
                    probe_prob[r_i64] = prob


# =============================================================================
# Compilation Cache & Launchers
# =============================================================================


@program_cache
def _compile_vocab_summary_stage1_tma(
    ordinal: int,
    tile_m: int,
    n: int,
    k: int,
    use_inv_temp: bool,
    use_bias: bool,
    use_grammar: bool,
    use_probe: bool,
    use_gumbel: bool,
    write_logits: bool,
    logits_dtype_name: str,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    y_type = getattr(torch, logits_dtype_name)
    ct_y = Float32 if y_type == torch.float32 else BFloat16
    kernel = TmaVocabSummaryLargeMKernel(
        n,
        k,
        tile_m,
        use_inv_temp=use_inv_temp,
        use_bias=use_bias,
        use_grammar=use_grammar,
        use_probe=use_probe,
        use_gumbel=use_gumbel,
        write_logits=write_logits,
    )
    fake_f32 = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_i32 = make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_u32 = make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_i64 = make_ptr(Int64, 16, cute.AddressSpace.gmem, assumed_align=8)
    fake_y = make_ptr(ct_y, 16, cute.AddressSpace.gmem, assumed_align=y_type.itemsize)
    with torch.cuda.device(ordinal), FakeTensorMode():
        dev = torch.device("cuda", ordinal)
        fx = torch.empty((tile_m, k), dtype=torch.bfloat16, device=dev)
        fw = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
        t_x = _to_kernel_tensor(fx, cutlass.BFloat16, dynamic_layout=True)
        t_w = _to_kernel_tensor(fw, cutlass.BFloat16, dynamic_layout=(n == 1))
        key = (
            int(tile_m),
            tensor_key("source", fx, dims=(DimKey.dynamic(), DimKey.exact(k))),
            tensor_key("weight", fw, dims=(DimKey.exact(n), DimKey.exact(k))),
            bool(use_inv_temp),
            bool(use_bias),
            bool(use_grammar),
            bool(use_probe),
            bool(use_gumbel),
            bool(write_logits),
            logits_dtype_name,
        )
        raw = b12x_compile(
            kernel,
            t_x,
            t_w,
            fake_f32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_f32,
            fake_u32,
            fake_i64,
            fake_f32,
            fake_i64,
            fake_i64,
            fake_y,
            Int32(tile_m),
            Int32(0),
            Int64(kernel.num_tiles),
            Int64((n + 31) // 32),
            Int64(n),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.bf16_vocab_summary.tma_large_m", 3, key
            ),
        )

    def run(
        x: torch.Tensor,
        weight: torch.Tensor,
        tile_max: torch.Tensor,
        tile_sum_exp: torch.Tensor,
        tile_argmax: torch.Tensor,
        *,
        tile_gumbel_max: torch.Tensor | None,
        tile_gumbel_arg: torch.Tensor | None,
        inv_temperature: torch.Tensor | None,
        logit_bias: torch.Tensor | None,
        grammar_mask: torch.Tensor | None,
        probe_token_ids: torch.Tensor | None,
        probe_logits: torch.Tensor | None,
        rng_seed: torch.Tensor | None,
        rng_offset: torch.Tensor | None,
        logits_out: torch.Tensor | None,
        vocab_offset: int,
    ):
        dummy_f32 = make_ptr(
            Float32, tile_max.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_i32 = make_ptr(
            Int32, tile_argmax.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_u32 = make_ptr(
            Uint32, tile_argmax.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_i64 = make_ptr(
            Int64, tile_max.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_y = make_ptr(
            ct_y,
            tile_max.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=y_type.itemsize,
        )
        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
                    _to_kernel_tensor(
                        weight, cutlass.BFloat16, dynamic_layout=(weight.shape[0] == 1)
                    ),
                    make_ptr(
                        Float32,
                        tile_max.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Float32,
                        tile_sum_exp.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Int32,
                        tile_argmax.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    (
                        make_ptr(
                            Float32,
                            tile_gumbel_max.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if tile_gumbel_max is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int32,
                            tile_gumbel_arg.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if tile_gumbel_arg is not None
                        else dummy_i32
                    ),
                    (
                        make_ptr(
                            Float32,
                            inv_temperature.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if inv_temperature is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Float32,
                            logit_bias.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if logit_bias is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Uint32,
                            grammar_mask.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if grammar_mask is not None
                        else dummy_u32
                    ),
                    (
                        make_ptr(
                            Int64,
                            probe_token_ids.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if probe_token_ids is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            Float32,
                            probe_logits.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if probe_logits is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int64,
                            rng_seed.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if rng_seed is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            Int64,
                            rng_offset.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if rng_offset is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            ct_y,
                            logits_out.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=y_type.itemsize,
                        )
                        if logits_out is not None
                        else dummy_y
                    ),
                    Int32(x.shape[0]),
                    Int32(vocab_offset),
                    Int64(tile_max.stride(0)),
                    Int64(grammar_mask.stride(0) if grammar_mask is not None else 0),
                    Int64(logits_out.stride(0) if logits_out is not None else 0),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)


@program_cache
def _compile_vocab_summary_stage1(
    ordinal: int,
    max_rows_bucket: int,
    n: int,
    k: int,
    use_inv_temp: bool,
    use_bias: bool,
    use_grammar: bool,
    use_probe: bool,
    use_gumbel: bool,
    write_logits: bool,
    logits_dtype_name: str,
):
    y_type = getattr(torch, logits_dtype_name)
    ct_y = Float32 if y_type == torch.float32 else BFloat16
    if max_rows_bucket <= _SMALL_M_THRESHOLD:
        kernel = TensorCoreVocabSummarySmallMKernel(
            n,
            k,
            max_rows_bucket,
            use_inv_temp=use_inv_temp,
            use_bias=use_bias,
            use_grammar=use_grammar,
            use_probe=use_probe,
            use_gumbel=use_gumbel,
            write_logits=write_logits,
        )
        spec_name = "gemm.bf16_vocab_summary.tc_small_m"
    else:
        kernel = TensorCoreVocabSummaryLargeMKernel(
            n,
            k,
            max_rows_bucket,
            use_inv_temp=use_inv_temp,
            use_bias=use_bias,
            use_grammar=use_grammar,
            use_probe=use_probe,
            use_gumbel=use_gumbel,
            write_logits=write_logits,
        )
        spec_name = "gemm.bf16_vocab_summary.tc_large_m"

    key = (
        int(max_rows_bucket),
        int(n),
        int(k),
        bool(use_inv_temp),
        bool(use_bias),
        bool(use_grammar),
        bool(use_probe),
        bool(use_gumbel),
        bool(write_logits),
        logits_dtype_name,
    )
    fake_bf16 = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_f32 = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_i32 = make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_u32 = make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_i64 = make_ptr(Int64, 16, cute.AddressSpace.gmem, assumed_align=8)
    fake_y = make_ptr(ct_y, 16, cute.AddressSpace.gmem, assumed_align=y_type.itemsize)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            fake_bf16,
            fake_bf16,
            fake_f32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_f32,
            fake_u32,
            fake_i64,
            fake_f32,
            fake_i64,
            fake_i64,
            fake_y,
            Int32(1),
            Int32(0),
            Int64(k),
            Int64(k),
            Int64(kernel.num_tiles),
            Int64((n + 31) // 32),
            Int64(n),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(spec_name, 1, key),
        )

    def run(
        x: torch.Tensor,
        weight: torch.Tensor,
        tile_max: torch.Tensor,
        tile_sum_exp: torch.Tensor,
        tile_argmax: torch.Tensor,
        *,
        tile_gumbel_max: torch.Tensor | None,
        tile_gumbel_arg: torch.Tensor | None,
        inv_temperature: torch.Tensor | None,
        logit_bias: torch.Tensor | None,
        grammar_mask: torch.Tensor | None,
        probe_token_ids: torch.Tensor | None,
        probe_logits: torch.Tensor | None,
        rng_seed: torch.Tensor | None,
        rng_offset: torch.Tensor | None,
        logits_out: torch.Tensor | None,
        vocab_offset: int,
    ):
        dummy_f32 = make_ptr(
            Float32, tile_max.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_i32 = make_ptr(
            Int32, tile_argmax.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_u32 = make_ptr(
            Uint32, tile_argmax.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_i64 = make_ptr(
            Int64, tile_max.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_y = make_ptr(
            ct_y,
            tile_max.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=y_type.itemsize,
        )
        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    make_ptr(
                        BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                    ),
                    make_ptr(
                        BFloat16,
                        weight.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    make_ptr(
                        Float32,
                        tile_max.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Float32,
                        tile_sum_exp.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Int32,
                        tile_argmax.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    (
                        make_ptr(
                            Float32,
                            tile_gumbel_max.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if tile_gumbel_max is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int32,
                            tile_gumbel_arg.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if tile_gumbel_arg is not None
                        else dummy_i32
                    ),
                    (
                        make_ptr(
                            Float32,
                            inv_temperature.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if inv_temperature is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Float32,
                            logit_bias.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if logit_bias is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Uint32,
                            grammar_mask.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if grammar_mask is not None
                        else dummy_u32
                    ),
                    (
                        make_ptr(
                            Int64,
                            probe_token_ids.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if probe_token_ids is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            Float32,
                            probe_logits.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if probe_logits is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int64,
                            rng_seed.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if rng_seed is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            Int64,
                            rng_offset.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if rng_offset is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            ct_y,
                            logits_out.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=y_type.itemsize,
                        )
                        if logits_out is not None
                        else dummy_y
                    ),
                    Int32(x.shape[0]),
                    Int32(vocab_offset),
                    Int64(x.stride(0)),
                    Int64(weight.stride(0)),
                    Int64(tile_max.stride(0)),
                    Int64(grammar_mask.stride(0) if grammar_mask is not None else 0),
                    Int64(logits_out.stride(0) if logits_out is not None else 0),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)


@program_cache
def _compile_vocab_summary_stage2(
    ordinal: int,
    use_gumbel: bool,
    use_probe: bool,
    use_inv_temp: bool,
):
    kernel = VocabTileSummaryReduceKernel(
        use_gumbel=use_gumbel,
        use_probe=use_probe,
        use_inv_temp=use_inv_temp,
    )
    key = (bool(use_gumbel), bool(use_probe), bool(use_inv_temp))
    fake_f32 = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_i32 = make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_i64 = make_ptr(Int64, 16, cute.AddressSpace.gmem, assumed_align=8)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            fake_f32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_i64,
            fake_f32,
            fake_f32,
            fake_f32,
            fake_f32,
            fake_i32,
            fake_f32,
            fake_i32,
            fake_f32,
            Int32(1),
            Int32(1),
            Int64(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.bf16_vocab_summary.reduce", 2, key
            ),
        )

    def run(
        tile_max: torch.Tensor,
        tile_sum_exp: torch.Tensor,
        tile_argmax: torch.Tensor,
        global_max: torch.Tensor,
        global_sum_exp: torch.Tensor,
        global_lse: torch.Tensor,
        argmax_id: torch.Tensor,
        *,
        tile_gumbel_max: torch.Tensor | None,
        tile_gumbel_arg: torch.Tensor | None,
        gumbel_max: torch.Tensor | None,
        gumbel_id: torch.Tensor | None,
        probe_logits: torch.Tensor | None,
        probe_token_ids: torch.Tensor | None,
        probe_prob: torch.Tensor | None,
        inv_temperature: torch.Tensor | None,
        rows: int,
        num_tiles: int,
    ):
        dummy_f32 = make_ptr(
            Float32, global_max.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_i32 = make_ptr(
            Int32, argmax_id.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        dummy_i64 = make_ptr(
            Int64, global_max.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    make_ptr(
                        Float32,
                        tile_max.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Float32,
                        tile_sum_exp.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Int32,
                        tile_argmax.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    (
                        make_ptr(
                            Float32,
                            tile_gumbel_max.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if tile_gumbel_max is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int32,
                            tile_gumbel_arg.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if tile_gumbel_arg is not None
                        else dummy_i32
                    ),
                    (
                        make_ptr(
                            Float32,
                            probe_logits.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if probe_logits is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int64,
                            probe_token_ids.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        if probe_token_ids is not None
                        else dummy_i64
                    ),
                    (
                        make_ptr(
                            Float32,
                            inv_temperature.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if inv_temperature is not None
                        else dummy_f32
                    ),
                    make_ptr(
                        Float32,
                        global_max.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Float32,
                        global_sum_exp.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Float32,
                        global_lse.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    make_ptr(
                        Int32,
                        argmax_id.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    (
                        make_ptr(
                            Float32,
                            gumbel_max.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if gumbel_max is not None
                        else dummy_f32
                    ),
                    (
                        make_ptr(
                            Int32,
                            gumbel_id.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if gumbel_id is not None
                        else dummy_i32
                    ),
                    (
                        make_ptr(
                            Float32,
                            probe_prob.data_ptr(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        if probe_prob is not None
                        else dummy_f32
                    ),
                    Int32(rows),
                    Int32(num_tiles),
                    Int64(tile_max.stride(0)),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)


def _validate_stage1_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    tile_max: torch.Tensor,
    tile_sum_exp: torch.Tensor,
    tile_argmax: torch.Tensor,
    *,
    max_rows_cap: int,
    inv_temperature: torch.Tensor | None,
    logit_bias: torch.Tensor | None,
    grammar_mask: torch.Tensor | None,
    probe_token_ids: torch.Tensor | None,
    probe_logits: torch.Tensor | None,
    rng_seed: torch.Tensor | None,
    rng_offset: torch.Tensor | None,
    tile_gumbel_max: torch.Tensor | None,
    tile_gumbel_arg: torch.Tensor | None,
    logits_out: torch.Tensor | None,
    write_logits: bool,
) -> tuple[int, int, int, int]:
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("vocab summary requires x[M, K] and weight[N, K]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("vocab summary requires BF16 x and weight")
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("vocab summary requires CUDA tensors on the same device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("vocab summary requires contiguous x and weight")
    if x.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0:
        raise ValueError("vocab summary requires 16-byte aligned x and weight")
    rows, k = int(x.shape[0]), int(x.shape[1])
    n = int(weight.shape[0])
    if n <= 0 or k <= 0 or k % TC_K_MULTIPLE != 0:
        raise ValueError(f"K must be a positive multiple of {TC_K_MULTIPLE} and N > 0")
    if rows > max_rows_cap:
        raise ValueError(f"rows={rows} exceeds max_rows={max_rows_cap}")
    num_tiles = vocab_summary_num_tiles(n, max_rows_cap)
    for name, t, dt in (
        ("tile_max", tile_max, torch.float32),
        ("tile_sum_exp", tile_sum_exp, torch.float32),
        ("tile_argmax", tile_argmax, torch.int32),
    ):
        if (
            t.device != x.device
            or t.dtype != dt
            or t.ndim != 2
            or t.shape[0] < rows
            or t.shape[1] < num_tiles
            or t.stride(1) != 1
        ):
            raise ValueError(
                f"{name} must be {dt} [>=M, >={num_tiles}] with stride(-1)==1"
            )
    if tile_sum_exp.stride(0) != tile_max.stride(0) or tile_argmax.stride(
        0
    ) != tile_max.stride(0):
        raise ValueError("tile_max, tile_sum_exp, and tile_argmax must share stride(0)")

    if inv_temperature is not None:
        if (
            inv_temperature.device != x.device
            or inv_temperature.dtype != torch.float32
            or inv_temperature.ndim != 1
            or inv_temperature.shape[0] < rows
            or not inv_temperature.is_contiguous()
        ):
            raise ValueError("inv_temperature must be contiguous float32[>=M]")
    if logit_bias is not None:
        if (
            logit_bias.device != x.device
            or logit_bias.dtype != torch.float32
            or logit_bias.ndim != 1
            or logit_bias.shape[0] != n
            or not logit_bias.is_contiguous()
        ):
            raise ValueError("logit_bias must be contiguous float32[N]")
    if grammar_mask is not None:
        min_words = (n + 31) // 32
        if (
            grammar_mask.device != x.device
            or grammar_mask.dtype not in (torch.uint32, torch.int32)
            or grammar_mask.ndim != 2
            or grammar_mask.shape[0] < rows
            or grammar_mask.shape[1] < min_words
            or grammar_mask.stride(1) != 1
        ):
            raise ValueError(
                f"grammar_mask must be uint32/int32 [>=M, >={min_words}] with stride(-1)==1"
            )
    has_probe = probe_token_ids is not None or probe_logits is not None
    if has_probe:
        if probe_token_ids is None or probe_logits is None:
            raise ValueError("probe_token_ids and probe_logits must be passed together")
        if (
            probe_token_ids.device != x.device
            or probe_token_ids.dtype != torch.int64
            or probe_token_ids.ndim != 1
            or probe_token_ids.shape[0] < rows
            or not probe_token_ids.is_contiguous()
        ):
            raise ValueError("probe_token_ids must be contiguous int64[>=M]")
        if (
            probe_logits.device != x.device
            or probe_logits.dtype != torch.float32
            or probe_logits.ndim != 1
            or probe_logits.shape[0] < rows
            or not probe_logits.is_contiguous()
        ):
            raise ValueError("probe_logits must be contiguous float32[>=M]")

    gumbel_args = (rng_seed, rng_offset, tile_gumbel_max, tile_gumbel_arg)
    if any(a is not None for a in gumbel_args):
        if not all(a is not None for a in gumbel_args):
            raise ValueError(
                "rng_seed, rng_offset, tile_gumbel_max, and tile_gumbel_arg must all be provided for Gumbel mode"
            )
        for name, t in (("rng_seed", rng_seed), ("rng_offset", rng_offset)):
            if (
                t.device != x.device
                or t.dtype not in (torch.int64, torch.uint64)
                or t.ndim != 1
                or t.shape[0] < rows
                or not t.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous int64/uint64[>=M]")
        for name, t, dt in (
            ("tile_gumbel_max", tile_gumbel_max, torch.float32),
            ("tile_gumbel_arg", tile_gumbel_arg, torch.int32),
        ):
            if (
                t.device != x.device
                or t.dtype != dt
                or t.ndim != 2
                or t.shape[0] < rows
                or t.shape[1] < num_tiles
                or t.stride(1) != 1
                or t.stride(0) != tile_max.stride(0)
            ):
                raise ValueError(
                    f"{name} must be {dt} [>=M, >={num_tiles}] sharing stride(0) with tile_max"
                )
    if write_logits:
        if logits_out is None:
            raise ValueError("logits_out is required when write_logits=True")
        if (
            logits_out.device != x.device
            or logits_out.dtype not in (torch.bfloat16, torch.float32)
            or logits_out.ndim != 2
            or logits_out.shape[0] < rows
            or logits_out.shape[1] != n
            or logits_out.stride(1) != 1
        ):
            raise ValueError(
                "logits_out must be bfloat16/float32 [>=M, N] with stride(-1)==1"
            )
    return rows, n, k, num_tiles


def bf16_gemv_vocab_summary_stage1(
    x: torch.Tensor,
    weight: torch.Tensor,
    tile_max: torch.Tensor,
    tile_sum_exp: torch.Tensor,
    tile_argmax: torch.Tensor,
    *,
    inv_temperature: torch.Tensor | None = None,
    logit_bias: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
    vocab_offset: int = 0,
    probe_token_ids: torch.Tensor | None = None,
    probe_logits: torch.Tensor | None = None,
    rng_seed: torch.Tensor | None = None,
    rng_offset: torch.Tensor | None = None,
    tile_gumbel_max: torch.Tensor | None = None,
    tile_gumbel_arg: torch.Tensor | None = None,
    logits_out: torch.Tensor | None = None,
    write_logits: bool = False,
    max_rows: int | None = None,
) -> None:
    """Launch Stage-1 epilogue-fused BF16 lm_head tile-summary kernel."""
    cap = int(x.shape[0]) if max_rows is None else int(max_rows)
    if cap <= 0:
        return
    rows, n, k, _ = _validate_stage1_inputs(
        x,
        weight,
        tile_max,
        tile_sum_exp,
        tile_argmax,
        max_rows_cap=cap,
        inv_temperature=inv_temperature,
        logit_bias=logit_bias,
        grammar_mask=grammar_mask,
        probe_token_ids=probe_token_ids,
        probe_logits=probe_logits,
        rng_seed=rng_seed,
        rng_offset=rng_offset,
        tile_gumbel_max=tile_gumbel_max,
        tile_gumbel_arg=tile_gumbel_arg,
        logits_out=logits_out,
        write_logits=write_logits,
    )
    if rows == 0:
        return
    bucket = _max_rows_bucket(cap)
    use_gumbel = rng_seed is not None
    logits_dtype_name = (
        str(logits_out.dtype).removeprefix("torch.")
        if (write_logits and logits_out is not None)
        else "bfloat16"
    )
    if bucket > _SMALL_M_THRESHOLD and k % 64 == 0:
        tma_tile_m = 64 if bucket <= 64 else 128
        launcher = _compile_vocab_summary_stage1_tma(
            int(x.device.index or 0),
            tma_tile_m,
            n,
            k,
            inv_temperature is not None,
            logit_bias is not None,
            grammar_mask is not None,
            probe_token_ids is not None,
            use_gumbel,
            bool(write_logits),
            logits_dtype_name,
        )
    else:
        launcher = _compile_vocab_summary_stage1(
            int(x.device.index or 0),
            bucket,
            n,
            k,
            inv_temperature is not None,
            logit_bias is not None,
            grammar_mask is not None,
            probe_token_ids is not None,
            use_gumbel,
            bool(write_logits),
            logits_dtype_name,
        )
    launcher(
        x,
        weight,
        tile_max,
        tile_sum_exp,
        tile_argmax,
        tile_gumbel_max=tile_gumbel_max if use_gumbel else None,
        tile_gumbel_arg=tile_gumbel_arg if use_gumbel else None,
        inv_temperature=inv_temperature,
        logit_bias=logit_bias,
        grammar_mask=grammar_mask,
        probe_token_ids=probe_token_ids,
        probe_logits=probe_logits,
        rng_seed=rng_seed if use_gumbel else None,
        rng_offset=rng_offset if use_gumbel else None,
        logits_out=logits_out if write_logits else None,
        vocab_offset=int(vocab_offset),
    )


def reduce_vocab_tile_summaries(
    tile_max: torch.Tensor,
    tile_sum_exp: torch.Tensor,
    tile_argmax: torch.Tensor,
    global_max: torch.Tensor,
    global_sum_exp: torch.Tensor,
    global_lse: torch.Tensor,
    argmax_id: torch.Tensor,
    *,
    tile_gumbel_max: torch.Tensor | None = None,
    tile_gumbel_arg: torch.Tensor | None = None,
    gumbel_max: torch.Tensor | None = None,
    gumbel_id: torch.Tensor | None = None,
    probe_logits: torch.Tensor | None = None,
    probe_token_ids: torch.Tensor | None = None,
    probe_prob: torch.Tensor | None = None,
    inv_temperature: torch.Tensor | None = None,
    rows: int | None = None,
    num_tiles: int | None = None,
) -> None:
    """Launch Stage-2 L2-resident tile summary reducer (``grid = (M, 1, 1)``)."""
    m = int(tile_max.shape[0]) if rows is None else int(rows)
    nt = int(tile_max.shape[1]) if num_tiles is None else int(num_tiles)
    if m == 0:
        return
    if nt <= 0:
        raise ValueError("num_tiles must be positive")
    use_gumbel = tile_gumbel_max is not None
    if use_gumbel and (
        tile_gumbel_arg is None or gumbel_max is None or gumbel_id is None
    ):
        raise ValueError(
            "tile_gumbel_arg, gumbel_max, and gumbel_id are required with tile_gumbel_max"
        )
    use_probe = probe_logits is not None
    if use_probe and (probe_prob is None or probe_token_ids is None):
        raise ValueError(
            "probe_token_ids and probe_prob are required with probe_logits"
        )
    launcher = _compile_vocab_summary_stage2(
        int(tile_max.device.index or 0),
        use_gumbel,
        use_probe,
        inv_temperature is not None,
    )
    launcher(
        tile_max,
        tile_sum_exp,
        tile_argmax,
        global_max,
        global_sum_exp,
        global_lse,
        argmax_id,
        tile_gumbel_max=tile_gumbel_max if use_gumbel else None,
        tile_gumbel_arg=tile_gumbel_arg if use_gumbel else None,
        gumbel_max=gumbel_max if use_gumbel else None,
        gumbel_id=gumbel_id if use_gumbel else None,
        probe_logits=probe_logits if use_probe else None,
        probe_token_ids=probe_token_ids if use_probe else None,
        probe_prob=probe_prob if use_probe else None,
        inv_temperature=inv_temperature,
        rows=m,
        num_tiles=nt,
    )


def bf16_gemv_vocab_summary(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    inv_temperature: torch.Tensor | None = None,
    logit_bias: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
    vocab_offset: int = 0,
    probe_token_ids: torch.Tensor | None = None,
    rng_seed: torch.Tensor | None = None,
    rng_offset: torch.Tensor | None = None,
    write_logits: bool = False,
    logits_dtype: torch.dtype = torch.bfloat16,
    buffers: VocabSummaryBuffers | None = None,
) -> VocabSummaryResult:
    """Run zero-HBM fused lm_head projection + L2 tile-summary reduction.

    Pass pre-allocated ``buffers`` from ``allocate_vocab_summary_outputs`` for
    zero-allocation CUDA-graph capture and replay.
    """
    rows = int(x.shape[0])
    n = int(weight.shape[0])
    use_gumbel = rng_seed is not None or rng_offset is not None
    if use_gumbel and (rng_seed is None or rng_offset is None):
        raise ValueError("rng_seed and rng_offset must be passed together")
    has_probe = probe_token_ids is not None

    if buffers is None:
        buffers = allocate_vocab_summary_outputs(
            max(rows, 1),
            n,
            x.device,
            use_gumbel=use_gumbel,
            has_probe=has_probe,
            write_logits=write_logits,
            logits_dtype=logits_dtype,
        )
    else:
        if buffers.n != n or rows > buffers.max_rows:
            raise ValueError(
                f"buffers(max_rows={buffers.max_rows}, n={buffers.n}) incompatible with (M={rows}, N={n})"
            )
        if use_gumbel and (
            buffers.tile_gumbel_max is None or buffers.gumbel_id is None
        ):
            raise ValueError("buffers must be allocated with use_gumbel=True")
        if has_probe and (buffers.probe_logits is None or buffers.probe_prob is None):
            raise ValueError("buffers must be allocated with has_probe=True")
        if write_logits and buffers.logits_out is None:
            raise ValueError("buffers must be allocated with write_logits=True")

    bf16_gemv_vocab_summary_stage1(
        x,
        weight,
        buffers.tile_max,
        buffers.tile_sum_exp,
        buffers.tile_argmax,
        inv_temperature=inv_temperature,
        logit_bias=logit_bias,
        grammar_mask=grammar_mask,
        vocab_offset=vocab_offset,
        probe_token_ids=probe_token_ids,
        probe_logits=buffers.probe_logits if has_probe else None,
        rng_seed=rng_seed if use_gumbel else None,
        rng_offset=rng_offset if use_gumbel else None,
        tile_gumbel_max=buffers.tile_gumbel_max if use_gumbel else None,
        tile_gumbel_arg=buffers.tile_gumbel_arg if use_gumbel else None,
        logits_out=buffers.logits_out if write_logits else None,
        write_logits=write_logits,
        max_rows=buffers.max_rows,
    )
    reduce_vocab_tile_summaries(
        buffers.tile_max,
        buffers.tile_sum_exp,
        buffers.tile_argmax,
        buffers.global_max,
        buffers.global_sum_exp,
        buffers.global_lse,
        buffers.argmax_id,
        tile_gumbel_max=buffers.tile_gumbel_max if use_gumbel else None,
        tile_gumbel_arg=buffers.tile_gumbel_arg if use_gumbel else None,
        gumbel_max=buffers.gumbel_max if use_gumbel else None,
        gumbel_id=buffers.gumbel_id if use_gumbel else None,
        probe_logits=buffers.probe_logits if has_probe else None,
        probe_token_ids=probe_token_ids if has_probe else None,
        probe_prob=buffers.probe_prob if has_probe else None,
        inv_temperature=inv_temperature,
        rows=rows,
        num_tiles=buffers.num_tiles,
    )
    return VocabSummaryResult(
        global_max=buffers.global_max[:rows],
        global_sum_exp=buffers.global_sum_exp[:rows],
        global_lse=buffers.global_lse[:rows],
        argmax_id=buffers.argmax_id[:rows],
        gumbel_max=buffers.gumbel_max[:rows] if use_gumbel else None,
        gumbel_id=buffers.gumbel_id[:rows] if use_gumbel else None,
        probe_logits=buffers.probe_logits[:rows] if has_probe else None,
        probe_prob=buffers.probe_prob[:rows] if has_probe else None,
        logits_out=buffers.logits_out[:rows] if write_logits else None,
        buffers=buffers,
    )
