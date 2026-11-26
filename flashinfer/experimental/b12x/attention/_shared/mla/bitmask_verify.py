"""Step B: Exact Bitmask-Union Multi-Query Sparse MLA Verification Kernel (SM120).

Processes ``q_tile = 4`` verify tokens x ``H_tile = 16`` heads (``M = 64`` rows
per CTA: 4 exact ``m16`` tiles where each ``m16`` tile is all 16 heads of one
verify token) over deduplicated SWA (528 B E4M3 + UE8M0) and Indexed (288 B
NVFP4 E2M1 + E4M3) KV records.

Key SM120 architecture features:
  - **Full 16-head tile parity with single-query decode:** Matches
    ``UnifiedDecodeKernel``'s ``H_tile = 16`` (1 head block for TP4 ``H=16``,
    4 head blocks for TP1 ``H=64``) while amortizing every KV load and
    dequantization across 4 verify tokens simultaneously (``2x``–``4x`` fewer
    KV loads and ``4x`` higher PV compute-to-dequant ratio).
  - **Zero redundant PV dequantization:** All 8 math warps partition
    ``V = 512`` into ``64`` dims/warp (``8`` ``n8`` tiles), loading each
    64B-swizzled ``P`` tile once per ``ks`` and feeding all 4 ``m16`` tiles
    from a single V dequantization.
  - **100% ``mma.sync.aligned.m16n8k16`` utilization & zero spills:** Each
    ``m16`` tile maps 1-to-1 to one verify token's 16 heads (uniform 1-bit
    mask across the entire tile, zero head padding for ``H in {16, 64}``).
    Total dynamic SMEM = ``101,024 B <= 101,376 B`` (3-stage ``B_I = 32``
    NVFP4 ring + 3-stage ``B_I = 16`` SWA ring).
  - **Flexible Split-K (Direct / Workspace L2 Merge / SM120 BF16 DSMEM):**
    - ``num_splits == 1``: Direct coalesced BF16 output + optional
      ``attn_sink`` fold + optional ``return_lse`` (zero merge overhead).
    - ``split_mode == "workspace"`` (``num_splits in 2..16``): Writes BF16
      partials to caller-owned L2-resident workspace and runs the lightweight
      ``BitmaskSplitMergeKernel`` (~2 us) without cluster-placement limits.
    - ``split_mode == "dsmem"`` (``num_splits in {2, 4, 8}``): 1-pass SM120
      hardware DSMEM cluster reduction (``cluster=(1, 1, num_splits)``) over
      packed BF16 ``[64, 512]`` in dead SMEM (zero workspace allocation).
  - **Dual Union Modes:** Accepts either a single request-level union
    ``[B, U_max]`` (extracting 4-bit slice ``(mask >> (q_group * 4)) & 0xF``)
    or a dense per-group union ``[B * ceil(W / 4), U_max]`` (``U_4`` is
    30–40% smaller than ``U_8`` and runs Step A 2x faster).
  - Exact ``_mixed_pairs_to_bf16x2_sm120`` / ``mxfp8_pair_to_bf16x2_sm120`` /
    ``nvfp4_pair_to_bf16x2_sm120`` dequantization, exact ``attn_sink`` folding,
    exact empty-chunk NaN guard, NaN-safe dead-row skipping, and zero-alloc
    CUDA Graph capture/replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    launch as b12x_launch,
    tensor_compile_fact,
)
from b12x._lib.intrinsics.atomics import (
    ld_shared_f32,
    ld_shared_u32,
    st_shared_f32,
    st_shared_u32,
    store_v4_bf16x2,
)
from b12x._lib.intrinsics.constants import mma_m16n8k16_f32_bf16
from b12x._lib.intrinsics.math import fmax_f32, fmin_f32, rcp_approx_ftz
from b12x._lib.intrinsics.memory import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_global_v4_u32,
    ld_shared_u8_offset,
    ld_shared_u16_offset,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_shared_u8,
    st_shared_v4_u32,
)
from b12x._lib.intrinsics.packed import (
    byte_perm,
    mxfp8_pair_to_bf16x2_sm120,
    nvfp4_pair_to_bf16x2_sm120,
    pack_f32x2_to_bfloat2,
)
from b12x._lib.utils.device import current_cuda_stream
from b12x.attention._shared.cute.ops import LOG2_E

from .decode_math import (
    _exp2_approx_ftz_f32,
    _extract_u8,
    _log2_approx_ftz_f32,
    _mixed_pairs_to_bf16x2_sm120,
)

_LN2 = math.log(2.0)
_NEG_INF = float("-inf")

# Tile geometry: M=64 (q_tile=4 x H_tile=16 -> 4 m16 tiles = 1 token/tile).
_Q_TILE = 4
_H_TILE = 16
_M_TILE = _Q_TILE * _H_TILE  # 64 rows/CTA
_D_HEAD = 512
_BI_SWA = 16
_BI_IDX = 32
_SWA_RECORD_BYTES = 528
_SWA_SMEM_STRIDE = 528  # 528 % 16 == 0
_IDX_RECORD_BYTES = 288
_IDX_SMEM_STRIDE = 304  # 304 % 16 == 0, 304/4 % 32 == 12 (zero 4-way bank conflicts!)
_SWA_STAGES = 3
_IDX_STAGES = 3
_MATH_WARPS = 8
_MATH_THREADS = _MATH_WARPS * 32  # 256
_BLOCK_THREADS = (
    _MATH_THREADS  # 256 (8 warps = 2 warps/scheduler -> full 255 regs/thread!)
)
_MAX_SPLITS = 16

# Exact SMEM layout (total = 101,024 B <= 101,376 B SM120 99 KiB carveout).
_Q_BF16_BYTES = _M_TILE * _D_HEAD * 2  # 65,536 B (128B XOR swizzled)
_KV_STAGE_BYTES = max(
    _SWA_STAGES * _BI_SWA * _SWA_SMEM_STRIDE,  # 25,344 B
    _IDX_STAGES * _BI_IDX * _IDX_SMEM_STRIDE,  # 29,184 B
)  # 29,184 B
_SM_P_BYTES = _M_TILE * _BI_IDX * 2  # 4,096 B (64 x 32 bf16, 64B XOR swizzled)
_REDUCE_BYTES = 2 * 4 * _M_TILE * 4  # 2,048 B (4 cand_groups x 64 rows x {max,sum})
_MASK_STAGE_BYTES = 128  # 3 stages x 32 B = 96 B, 128B aligned
_RAW_SMEM_BYTES = (
    _Q_BF16_BYTES + _KV_STAGE_BYTES + _SM_P_BYTES + _REDUCE_BYTES + _MASK_STAGE_BYTES
)  # 100,992 B
_MBAR_BYTES = 32  # 3 x 8 B = 24 B, 16B aligned
assert _RAW_SMEM_BYTES + _MBAR_BYTES == 101024 <= 101376

# Dead-SMEM epilogue layout inside [0, 99,456):
#   [0, 65536)      : bf16_o[64, 512] BF16 (used for direct/workspace/dsmem!)
#   [65536, 65792)  : dsm_lse_in[64] f32 (partial base-2 LSE per row)
#   [65792, 67840)  : dsm_w[64, 8] f32 (normalized split weights per row)
#   [67840, 68096)  : dsm_lse_out[64] f32 (final LSE per row)
_DSM_O_OFF = 0
_DSM_LSE_IN_OFF = 65536
_DSM_W_OFF = 65792
_DSM_LSE_OUT_OFF = 67840


@dsl_user_op
def _popc_u32(mask: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(mask).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_dsm_f32(smem_addr: Int32, peer_rank: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Int32(smem_addr).ir_value(loc=loc, ip=ip),
                Int32(peer_rank).ir_value(loc=loc, ip=ip),
            ],
            """{
                .reg .u32 raddr;
                mapa.shared::cluster.u32 raddr, $1, $2;
                ld.shared::cluster.f32 $0, [raddr];
            }""",
            "=f,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_dsm_v4_u32(
    smem_addr: Int32, peer_rank: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .u32 raddr;
            mapa.shared::cluster.u32 raddr, $4, $5;
            ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [raddr];
        }""",
        "=r,=r,=r,=r,r,r",
        has_side_effects=True,
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
def _bf16x2_to_f32x2(packed: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """{
            .reg .u32 lo_u, hi_u;
            shl.b32 lo_u, $2, 16;
            and.b32 hi_u, $2, 0xffff0000;
            mov.b32 $0, lo_u;
            mov.b32 $1, hi_u;
        }""",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@cute.jit
def _store_p_pair_swizzled_64b(
    sm_p_addr: Int32, row: Int32, col_even: Int32, packed_bf16x2: Uint32
) -> None:
    """Store bf16x2 (4B) into [64, 32] bf16 64B-XOR-swizzled ``sm_p_full``."""
    byte_col = col_even * Int32(2)
    swiz_col = byte_col ^ ((row & Int32(3)) << Int32(4))
    st_shared_u32(sm_p_addr + row * Int32(64) + swiz_col, packed_bf16x2)


@cute.jit
def _qk_idx_32(
    q_bf16_addr: Int32,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    warp_id: Int32,
    lane: Int32,
):
    """QK for BI=32 NVFP4: warps 0..3 -> tokens {0,1}, warps 4..7 -> tokens {2,3}.

    Each warp owns 8 candidates (``cand_grp = warp_id & 3``), dequantizes them
    ONCE, and feeds both of its ``m16`` tiles (32 heads = 2 tokens x 16 heads).
    """
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    m_half = warp_id >> Int32(2)  # 0 or 1
    cand_grp = warp_id & Int32(3)  # 0..3
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_row0_base = q_bf16_addr + (m_half * Int32(32) + a_row) * Int32(1024)
    a_row1_base = a_row0_base + Int32(16 * 1024)
    a_swiz = (a_row & Int32(7)) << Int32(4)
    a_col_half = (lane >> Int32(4)) << Int32(4)

    cand_row = cand_grp * Int32(8) + gid
    row_addr = kv_stage_addr + cand_row * Int32(_IDX_SMEM_STRIDE)
    act = ld_shared_u8_offset(mask_stage_addr + cand_row, 0) != Uint32(0)
    all_act = cute.arch.vote_all_sync(Boolean(act))
    none_act = cute.arch.vote_all_sync(Boolean(not act))

    q00 = Float32(0.0)
    q01 = Float32(0.0)
    q02 = Float32(0.0)
    q03 = Float32(0.0)
    q10 = Float32(0.0)
    q11 = Float32(0.0)
    q12 = Float32(0.0)
    q13 = Float32(0.0)

    if all_act:
        sw0, sw1, sw2, sw3 = ld_shared_v4_u32(row_addr + Int32(256))
        sw4, sw5, sw6, sw7 = ld_shared_v4_u32(row_addr + Int32(272))
        sc_words = (sw0, sw1, sw2, sw3, sw4, sw5, sw6, sw7)
        d_base = row_addr + tid
        for blk in cutlass.range_constexpr(8):
            for ks in cutlass.range_constexpr(4):
                ko = blk * 64 + ks * 16
                byte_imm = blk * 32 + ks * 8
                sb = _extract_u8(sc_words[blk], ks)
                p0 = ld_shared_u8_offset(d_base, byte_imm)
                p1 = ld_shared_u8_offset(d_base, byte_imm + 4)
                b0, b1 = _mixed_pairs_to_bf16x2_sm120(p0, p1, sb, swa=False)
                a_off = (Int32(ko * 2) + a_col_half) ^ a_swiz
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_row0_base + a_off)
                c0, c1, c2, c3 = ldmatrix_m8n8x4_b16(a_row1_base + a_off)
                q00, q01, q02, q03 = mma_m16n8k16_f32_bf16(
                    q00, q01, q02, q03, a0, a1, a2, a3, b0, b1
                )
                q10, q11, q12, q13 = mma_m16n8k16_f32_bf16(
                    q10, q11, q12, q13, c0, c1, c2, c3, b0, b1
                )
    elif not none_act:
        sw0, sw1, sw2, sw3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        sw4, sw5, sw6, sw7 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if act:
            sw0, sw1, sw2, sw3 = ld_shared_v4_u32(row_addr + Int32(256))
            sw4, sw5, sw6, sw7 = ld_shared_v4_u32(row_addr + Int32(272))
        sc_words = (sw0, sw1, sw2, sw3, sw4, sw5, sw6, sw7)
        d_base = row_addr + tid
        for blk in cutlass.range_constexpr(8):
            for ks in cutlass.range_constexpr(4):
                ko = blk * 64 + ks * 16
                byte_imm = blk * 32 + ks * 8
                b0, b1 = Uint32(0), Uint32(0)
                if act:
                    sb = _extract_u8(sc_words[blk], ks)
                    p0 = ld_shared_u8_offset(d_base, byte_imm)
                    p1 = ld_shared_u8_offset(d_base, byte_imm + 4)
                    b0, b1 = _mixed_pairs_to_bf16x2_sm120(p0, p1, sb, swa=False)
                cute.arch.sync_warp()
                a_off = (Int32(ko * 2) + a_col_half) ^ a_swiz
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_row0_base + a_off)
                c0, c1, c2, c3 = ldmatrix_m8n8x4_b16(a_row1_base + a_off)
                q00, q01, q02, q03 = mma_m16n8k16_f32_bf16(
                    q00, q01, q02, q03, a0, a1, a2, a3, b0, b1
                )
                q10, q11, q12, q13 = mma_m16n8k16_f32_bf16(
                    q10, q11, q12, q13, c0, c1, c2, c3, b0, b1
                )

    return q00, q01, q02, q03, q10, q11, q12, q13


@cute.jit
def _qk_swa_16(
    q_bf16_addr: Int32,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    warp_id: Int32,
    lane: Int32,
):
    """QK for BI=16 SWA: warp ``w`` owns token ``w >> 1`` and cand group ``w & 1``."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    m_tile = warp_id >> Int32(1)  # 0..3
    cand_grp = warp_id & Int32(1)  # 0..1
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_row_base = q_bf16_addr + (m_tile * Int32(16) + a_row) * Int32(1024)
    a_swiz = (a_row & Int32(7)) << Int32(4)
    a_col_half = (lane >> Int32(4)) << Int32(4)

    cand_row = cand_grp * Int32(8) + gid
    row_addr = kv_stage_addr + cand_row * Int32(_SWA_SMEM_STRIDE)
    act = ld_shared_u8_offset(mask_stage_addr + cand_row, 0) != Uint32(0)
    all_act = cute.arch.vote_all_sync(Boolean(act))
    none_act = cute.arch.vote_all_sync(Boolean(not act))

    q0 = Float32(0.0)
    q1 = Float32(0.0)
    q2 = Float32(0.0)
    q3 = Float32(0.0)

    if all_act:
        sw0, sw1, sw2, sw3 = ld_shared_v4_u32(row_addr + Int32(512))
        sc_words = (sw0, sw1, sw2, sw3)
        d_base = row_addr + tid * Int32(2)
        for blk in cutlass.range_constexpr(8):
            for ks in cutlass.range_constexpr(4):
                ko = blk * 64 + ks * 16
                sc_idx = blk * 2 + (ks // 2)
                sb = _extract_u8(sc_words[sc_idx // 4], sc_idx % 4)
                p0 = ld_shared_u16_offset(d_base, ko)
                p1 = ld_shared_u16_offset(d_base, ko + 8)
                b0, b1 = _mixed_pairs_to_bf16x2_sm120(p0, p1, sb, swa=True)
                a_off = (Int32(ko * 2) + a_col_half) ^ a_swiz
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_row_base + a_off)
                q0, q1, q2, q3 = mma_m16n8k16_f32_bf16(
                    q0, q1, q2, q3, a0, a1, a2, a3, b0, b1
                )
    elif not none_act:
        sw0, sw1, sw2, sw3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if act:
            sw0, sw1, sw2, sw3 = ld_shared_v4_u32(row_addr + Int32(512))
        sc_words = (sw0, sw1, sw2, sw3)
        d_base = row_addr + tid * Int32(2)
        for blk in cutlass.range_constexpr(8):
            for ks in cutlass.range_constexpr(4):
                ko = blk * 64 + ks * 16
                b0, b1 = Uint32(0), Uint32(0)
                if act:
                    sc_idx = blk * 2 + (ks // 2)
                    sb = _extract_u8(sc_words[sc_idx // 4], sc_idx % 4)
                    p0 = ld_shared_u16_offset(d_base, ko)
                    p1 = ld_shared_u16_offset(d_base, ko + 8)
                    b0, b1 = _mixed_pairs_to_bf16x2_sm120(p0, p1, sb, swa=True)
                cute.arch.sync_warp()
                a_off = (Int32(ko * 2) + a_col_half) ^ a_swiz
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_row_base + a_off)
                q0, q1, q2, q3 = mma_m16n8k16_f32_bf16(
                    q0, q1, q2, q3, a0, a1, a2, a3, b0, b1
                )

    return q0, q1, q2, q3


@cute.jit
def _pv_m64_shared_dequant(
    acc_o: cute.Tensor,  # rmem (128,) f32: [m_tile*32 + v_tile*4 + 0..3]
    sm_p_addr: Int32,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    *,
    swa: cutlass.Constexpr,
):
    """Outer-ks PV over M=64: 1 V dequant feeds ALL 4 m16 tiles (4x reuse!)."""
    kv_stride = cutlass.const_expr(_SWA_SMEM_STRIDE if swa else _IDX_SMEM_STRIDE)
    n_ksteps = cutlass.const_expr(_BI_SWA // 16 if swa else _BI_IDX // 16)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    entry_parity = gid & Int32(1)
    pair_selector = Int32(0x5410)
    if entry_parity != Int32(0):
        pair_selector = Int32(0x3276)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    # sm_p_full is [64, 32] bf16 -> row stride = 64 B, 64B XOR swizzled.
    p_row0_base = sm_p_addr + a_row * Int32(64)
    p_row1_base = p_row0_base + Int32(16 * 64)
    p_row2_base = p_row0_base + Int32(32 * 64)
    p_row3_base = p_row0_base + Int32(48 * 64)
    p_swiz = (a_row & Int32(3)) << Int32(4)
    p_col_half = (lane >> Int32(4)) << Int32(4)

    for ks in cutlass.range_constexpr(n_ksteps):
        ent0 = Int32(ks * 16) + tid * Int32(2) + entry_parity
        ent8 = ent0 + Int32(8)
        act0 = ld_shared_u8_offset(mask_stage_addr + ent0, 0) != Uint32(0)
        act8 = ld_shared_u8_offset(mask_stage_addr + ent8, 0) != Uint32(0)
        all_act = cute.arch.vote_all_sync(Boolean(act0 and act8))
        none_act = cute.arch.vote_all_sync(Boolean((not act0) and (not act8)))

        if not none_act:
            # Load P once per ks for ALL 4 M-tiles, reused across all 8 V-tiles!
            p_off = (Int32(ks * 32) + p_col_half) ^ p_swiz
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(p_row0_base + p_off)
            c0, c1, c2, c3 = ldmatrix_m8n8x4_b16(p_row1_base + p_off)
            e0, e1, e2, e3 = ldmatrix_m8n8x4_b16(p_row2_base + p_off)
            g0, g1, g2, g3 = ldmatrix_m8n8x4_b16(p_row3_base + p_off)

            row0_addr = kv_stage_addr + ent0 * Int32(kv_stride)
            row8_addr = kv_stage_addr + ent8 * Int32(kv_stride)

            if cutlass.const_expr(swa):
                d0_base = row0_addr + warp_id * Int32(64) + (gid & ~Int32(1))
                d8_base = row8_addr + warp_id * Int32(64) + (gid & ~Int32(1))
                sc0_word = Uint32(0)
                sc8_word = Uint32(0)
                if all_act:
                    sc0_word = Uint32(
                        ld_shared_u16_offset(row0_addr + warp_id * Int32(2), 512)
                    )
                    sc8_word = Uint32(
                        ld_shared_u16_offset(row8_addr + warp_id * Int32(2), 512)
                    )
                    for tile in cutlass.range_constexpr(8):
                        sc0 = (
                            (sc0_word & Uint32(0xFF))
                            if cutlass.const_expr(tile < 4)
                            else ((sc0_word >> Uint32(8)) & Uint32(0xFF))
                        )
                        sc8 = (
                            (sc8_word & Uint32(0xFF))
                            if cutlass.const_expr(tile < 4)
                            else ((sc8_word >> Uint32(8)) & Uint32(0xFF))
                        )
                        p0 = ld_shared_u16_offset(d0_base, tile * 8)
                        pair0 = mxfp8_pair_to_bf16x2_sm120(p0, sc0)
                        p8 = ld_shared_u16_offset(d8_base, tile * 8)
                        pair8 = mxfp8_pair_to_bf16x2_sm120(p8, sc8)
                        peer0 = cute.arch.shuffle_sync_bfly(pair0, offset=4)
                        peer8 = cute.arch.shuffle_sync_bfly(pair8, offset=4)
                        b0 = byte_perm(pair0, peer0, pair_selector)
                        b1 = byte_perm(pair8, peer8, pair_selector)
                        for mt, (pa0, pa1, pa2, pa3) in (
                            (0, (a0, a1, a2, a3)),
                            (1, (c0, c1, c2, c3)),
                            (2, (e0, e1, e2, e3)),
                            (3, (g0, g1, g2, g3)),
                        ):
                            idx = mt * 32 + tile * 4
                            r0, r1, r2, r3 = mma_m16n8k16_f32_bf16(
                                acc_o[idx],
                                acc_o[idx + 1],
                                acc_o[idx + 2],
                                acc_o[idx + 3],
                                pa0,
                                pa1,
                                pa2,
                                pa3,
                                b0,
                                b1,
                            )
                            acc_o[idx] = r0
                            acc_o[idx + 1] = r1
                            acc_o[idx + 2] = r2
                            acc_o[idx + 3] = r3
                else:
                    if act0:
                        sc0_word = Uint32(
                            ld_shared_u16_offset(row0_addr + warp_id * Int32(2), 512)
                        )
                    if act8:
                        sc8_word = Uint32(
                            ld_shared_u16_offset(row8_addr + warp_id * Int32(2), 512)
                        )
                    for tile in cutlass.range_constexpr(8):
                        pair0 = Uint32(0)
                        pair8 = Uint32(0)
                        if act0:
                            sc0 = (
                                (sc0_word & Uint32(0xFF))
                                if cutlass.const_expr(tile < 4)
                                else ((sc0_word >> Uint32(8)) & Uint32(0xFF))
                            )
                            p0 = ld_shared_u16_offset(d0_base, tile * 8)
                            pair0 = mxfp8_pair_to_bf16x2_sm120(p0, sc0)
                        if act8:
                            sc8 = (
                                (sc8_word & Uint32(0xFF))
                                if cutlass.const_expr(tile < 4)
                                else ((sc8_word >> Uint32(8)) & Uint32(0xFF))
                            )
                            p8 = ld_shared_u16_offset(d8_base, tile * 8)
                            pair8 = mxfp8_pair_to_bf16x2_sm120(p8, sc8)
                        cute.arch.sync_warp()
                        peer0 = cute.arch.shuffle_sync_bfly(pair0, offset=4)
                        peer8 = cute.arch.shuffle_sync_bfly(pair8, offset=4)
                        b0 = byte_perm(pair0, peer0, pair_selector)
                        b1 = byte_perm(pair8, peer8, pair_selector)
                        for mt, (pa0, pa1, pa2, pa3) in (
                            (0, (a0, a1, a2, a3)),
                            (1, (c0, c1, c2, c3)),
                            (2, (e0, e1, e2, e3)),
                            (3, (g0, g1, g2, g3)),
                        ):
                            idx = mt * 32 + tile * 4
                            r0, r1, r2, r3 = mma_m16n8k16_f32_bf16(
                                acc_o[idx],
                                acc_o[idx + 1],
                                acc_o[idx + 2],
                                acc_o[idx + 3],
                                pa0,
                                pa1,
                                pa2,
                                pa3,
                                b0,
                                b1,
                            )
                            acc_o[idx] = r0
                            acc_o[idx + 1] = r1
                            acc_o[idx + 2] = r2
                            acc_o[idx + 3] = r3
            else:
                d0_base = row0_addr + warp_id * Int32(32) + (gid >> Int32(1))
                d8_base = row8_addr + warp_id * Int32(32) + (gid >> Int32(1))
                sc0_word = Uint32(0)
                sc8_word = Uint32(0)
                if all_act:
                    sc0_word = ld_shared_u32(
                        row0_addr + Int32(256) + warp_id * Int32(4)
                    )
                    sc8_word = ld_shared_u32(
                        row8_addr + Int32(256) + warp_id * Int32(4)
                    )
                    for tile in cutlass.range_constexpr(8):
                        sc0 = _extract_u8(sc0_word, tile // 2)
                        sc8 = _extract_u8(sc8_word, tile // 2)
                        p0 = ld_shared_u8_offset(d0_base, tile * 4)
                        pair0 = nvfp4_pair_to_bf16x2_sm120(p0, sc0)
                        p8 = ld_shared_u8_offset(d8_base, tile * 4)
                        pair8 = nvfp4_pair_to_bf16x2_sm120(p8, sc8)
                        peer0 = cute.arch.shuffle_sync_bfly(pair0, offset=4)
                        peer8 = cute.arch.shuffle_sync_bfly(pair8, offset=4)
                        b0 = byte_perm(pair0, peer0, pair_selector)
                        b1 = byte_perm(pair8, peer8, pair_selector)
                        for mt, (pa0, pa1, pa2, pa3) in (
                            (0, (a0, a1, a2, a3)),
                            (1, (c0, c1, c2, c3)),
                            (2, (e0, e1, e2, e3)),
                            (3, (g0, g1, g2, g3)),
                        ):
                            idx = mt * 32 + tile * 4
                            r0, r1, r2, r3 = mma_m16n8k16_f32_bf16(
                                acc_o[idx],
                                acc_o[idx + 1],
                                acc_o[idx + 2],
                                acc_o[idx + 3],
                                pa0,
                                pa1,
                                pa2,
                                pa3,
                                b0,
                                b1,
                            )
                            acc_o[idx] = r0
                            acc_o[idx + 1] = r1
                            acc_o[idx + 2] = r2
                            acc_o[idx + 3] = r3
                else:
                    if act0:
                        sc0_word = ld_shared_u32(
                            row0_addr + Int32(256) + warp_id * Int32(4)
                        )
                    if act8:
                        sc8_word = ld_shared_u32(
                            row8_addr + Int32(256) + warp_id * Int32(4)
                        )
                    for tile in cutlass.range_constexpr(8):
                        pair0 = Uint32(0)
                        pair8 = Uint32(0)
                        if act0:
                            sc0 = _extract_u8(sc0_word, tile // 2)
                            p0 = ld_shared_u8_offset(d0_base, tile * 4)
                            pair0 = nvfp4_pair_to_bf16x2_sm120(p0, sc0)
                        if act8:
                            sc8 = _extract_u8(sc8_word, tile // 2)
                            p8 = ld_shared_u8_offset(d8_base, tile * 4)
                            pair8 = nvfp4_pair_to_bf16x2_sm120(p8, sc8)
                        cute.arch.sync_warp()
                        peer0 = cute.arch.shuffle_sync_bfly(pair0, offset=4)
                        peer8 = cute.arch.shuffle_sync_bfly(pair8, offset=4)
                        b0 = byte_perm(pair0, peer0, pair_selector)
                        b1 = byte_perm(pair8, peer8, pair_selector)
                        for mt, (pa0, pa1, pa2, pa3) in (
                            (0, (a0, a1, a2, a3)),
                            (1, (c0, c1, c2, c3)),
                            (2, (e0, e1, e2, e3)),
                            (3, (g0, g1, g2, g3)),
                        ):
                            idx = mt * 32 + tile * 4
                            r0, r1, r2, r3 = mma_m16n8k16_f32_bf16(
                                acc_o[idx],
                                acc_o[idx + 1],
                                acc_o[idx + 2],
                                acc_o[idx + 3],
                                pa0,
                                pa1,
                                pa2,
                                pa3,
                                b0,
                                b1,
                            )
                            acc_o[idx] = r0
                            acc_o[idx + 1] = r1
                            acc_o[idx + 2] = r2
                            acc_o[idx + 3] = r3


@cute.jit
def _process_chunk_idx_32(
    acc_o: cute.Tensor,  # rmem (128,) f32
    global_max: cute.Tensor,  # rmem (8,) f32
    global_sum: cute.Tensor,  # rmem (8,) f32
    q_bf16_addr: Int32,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    sm_p_addr: Int32,
    reduce_addr: Int32,
    sm_scale_log2: Float32,
    warp_id: Int32,
    lane: Int32,
):
    """BI=32 Indexed NVFP4 QK + bitmask + online softmax + PV (2 barriers total)."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    m_half = warp_id >> Int32(2)  # 0 or 1
    cand_grp = warp_id & Int32(3)  # 0..3
    neg_inf = Float32(_NEG_INF)

    # 1. Shared-dequant QK across this warp's 2 tokens (32 heads) and 8 cands.
    q00, q01, q02, q03, q10, q11, q12, q13 = _qk_idx_32(
        q_bf16_addr, kv_stage_addr, mask_stage_addr, warp_id, lane
    )

    # 2. Uniform 1-bit token mask per m16 tile!
    bit_t0 = Uint32(1) << (m_half * Int32(2)).to(Uint32)
    bit_t1 = bit_t0 << Uint32(1)
    c0 = cand_grp * Int32(8) + tid * Int32(2)
    m0 = ld_shared_u8_offset(mask_stage_addr + c0, 0)
    m1 = ld_shared_u8_offset(mask_stage_addr + c0, 1)
    act_t0_0 = (m0 & bit_t0) != Uint32(0)
    act_t0_1 = (m1 & bit_t0) != Uint32(0)
    act_t1_0 = (m0 & bit_t1) != Uint32(0)
    act_t1_1 = (m1 & bit_t1) != Uint32(0)

    s0_0 = q00 * sm_scale_log2 if act_t0_0 else neg_inf
    s0_1 = q01 * sm_scale_log2 if act_t0_1 else neg_inf
    s1_0 = q02 * sm_scale_log2 if act_t0_0 else neg_inf
    s1_1 = q03 * sm_scale_log2 if act_t0_1 else neg_inf
    s2_0 = q10 * sm_scale_log2 if act_t1_0 else neg_inf
    s2_1 = q11 * sm_scale_log2 if act_t1_1 else neg_inf
    s3_0 = q12 * sm_scale_log2 if act_t1_0 else neg_inf
    s3_1 = q13 * sm_scale_log2 if act_t1_1 else neg_inf

    # 3. Per-warp max & sum over 8 candidates for its 4 rows (divergence-safe shfl!).
    scores = ((s0_0, s0_1), (s1_0, s1_1), (s2_0, s2_1), (s3_0, s3_1))
    lmax = [neg_inf, neg_inf, neg_inf, neg_inf]
    p_lo = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    p_hi = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]

    reduce_max_addr = reduce_addr
    reduce_sum_addr = reduce_addr + Int32(4 * _M_TILE * 4)

    for ri in cutlass.range_constexpr(4):
        sa, sb = scores[ri]
        lm = fmax_f32(sa, sb)
        lm = fmax_f32(lm, cute.arch.shuffle_sync_bfly(lm, offset=2))
        lm = fmax_f32(lm, cute.arch.shuffle_sync_bfly(lm, offset=1))
        lmax[ri] = lm
        ls = Float32(0.0)
        if lm > neg_inf:
            pa = _exp2_approx_ftz_f32(sa - lm) if sa > neg_inf else Float32(0.0)
            pb = _exp2_approx_ftz_f32(sb - lm) if sb > neg_inf else Float32(0.0)
            p_lo[ri] = pa
            p_hi[ri] = pb
            ls = pa + pb
        cute.arch.sync_warp()
        ls = ls + cute.arch.shuffle_sync_bfly(ls, offset=2)
        ls = ls + cute.arch.shuffle_sync_bfly(ls, offset=1)
        if tid == Int32(0):
            row_id = m_half * Int32(32) + Int32(ri * 8) + gid
            slot = (cand_grp * Int32(_M_TILE) + row_id) * Int32(4)
            st_shared_f32(reduce_max_addr + slot, lm)
            st_shared_f32(reduce_sum_addr + slot, ls)

    # Barrier #1: cross-warp softmax reduce.
    cute.arch.sync_threads()

    # 4. Reduce all 8 rows for this lane across 4 cand_groups, update online softmax & acc_o.
    alphas = [Float32(1.0)] * 8
    new_ms = [neg_inf] * 8
    for ri in cutlass.range_constexpr(8):
        row_id = Int32(ri * 8) + gid
        bm = neg_inf
        wmaxs = []
        wsums = []
        for cg in cutlass.range_constexpr(4):
            slot = (Int32(cg * _M_TILE) + row_id) * Int32(4)
            wm = ld_shared_f32(reduce_max_addr + slot)
            ws = ld_shared_f32(reduce_sum_addr + slot)
            wmaxs.append(wm)
            wsums.append(ws)
            bm = fmax_f32(bm, wm)
        if bm > neg_inf:
            old_m = global_max[ri]
            new_m = fmax_f32(old_m, bm)
            alpha = Float32(0.0)
            if old_m > neg_inf:
                alpha = _exp2_approx_ftz_f32(old_m - new_m)
            bs = Float32(0.0)
            for cg in cutlass.range_constexpr(4):
                if wmaxs[cg] > neg_inf:
                    bs = bs + wsums[cg] * _exp2_approx_ftz_f32(wmaxs[cg] - new_m)
            alphas[ri] = alpha
            new_ms[ri] = new_m
            global_sum[ri] = global_sum[ri] * alpha + bs
            global_max[ri] = new_m

    # Store this warp's 4 rows (cands c0, c0+1) into 64B-swizzled sm_p_full.
    for ri in cutlass.range_constexpr(4):
        row_id = m_half * Int32(32) + Int32(ri * 8) + gid
        nm = new_ms[ri] if m_half == Int32(0) else new_ms[4 + ri]
        rsc = (
            _exp2_approx_ftz_f32(lmax[ri] - nm)
            if (lmax[ri] > neg_inf and nm > neg_inf)
            else Float32(0.0)
        )
        _store_p_pair_swizzled_64b(
            sm_p_addr, row_id, c0, pack_f32x2_to_bfloat2(p_lo[ri] * rsc, p_hi[ri] * rsc)
        )

    # Rescale acc_o for all 4 M-tiles (bit-exact identity when alpha == 1.0).
    for mt in cutlass.range_constexpr(4):
        a_lo = alphas[mt * 2]
        a_hi = alphas[mt * 2 + 1]
        for tile in cutlass.range_constexpr(8):
            idx = mt * 32 + tile * 4
            acc_o[idx + 0] = acc_o[idx + 0] * a_lo
            acc_o[idx + 1] = acc_o[idx + 1] * a_lo
            acc_o[idx + 2] = acc_o[idx + 2] * a_hi
            acc_o[idx + 3] = acc_o[idx + 3] * a_hi

    # Barrier #2: sm_p_full ready for PV.
    cute.arch.sync_threads()

    # 5. Outer-ks shared-dequant PV over BI=32 (4 M-tiles reuse 1 V dequant!).
    _pv_m64_shared_dequant(
        acc_o, sm_p_addr, kv_stage_addr, mask_stage_addr, warp_id, lane, swa=False
    )


@cute.jit
def _process_chunk_swa_16(
    acc_o: cute.Tensor,  # rmem (128,) f32
    global_max: cute.Tensor,  # rmem (8,) f32
    global_sum: cute.Tensor,  # rmem (8,) f32
    q_bf16_addr: Int32,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    sm_p_addr: Int32,
    reduce_addr: Int32,
    sm_scale_log2: Float32,
    warp_id: Int32,
    lane: Int32,
):
    """BI=16 SWA QK + bitmask + online softmax + PV (2 barriers total)."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    m_tile = warp_id >> Int32(1)  # 0..3
    cand_grp = warp_id & Int32(1)  # 0..1
    neg_inf = Float32(_NEG_INF)

    # 1. SWA QK: warp w owns token m_tile and 8 cands cand_grp.
    q0, q1, q2, q3 = _qk_swa_16(
        q_bf16_addr, kv_stage_addr, mask_stage_addr, warp_id, lane
    )

    bit_t = Uint32(1) << m_tile.to(Uint32)
    c0 = cand_grp * Int32(8) + tid * Int32(2)
    m0 = ld_shared_u8_offset(mask_stage_addr + c0, 0)
    m1 = ld_shared_u8_offset(mask_stage_addr + c0, 1)
    act0 = (m0 & bit_t) != Uint32(0)
    act1 = (m1 & bit_t) != Uint32(0)

    s_lo0 = q0 * sm_scale_log2 if act0 else neg_inf
    s_lo1 = q1 * sm_scale_log2 if act1 else neg_inf
    s_hi0 = q2 * sm_scale_log2 if act0 else neg_inf
    s_hi1 = q3 * sm_scale_log2 if act1 else neg_inf

    lm0 = fmax_f32(s_lo0, s_lo1)
    lm0 = fmax_f32(lm0, cute.arch.shuffle_sync_bfly(lm0, offset=2))
    lm0 = fmax_f32(lm0, cute.arch.shuffle_sync_bfly(lm0, offset=1))
    p00 = Float32(0.0)
    p01 = Float32(0.0)
    ls0 = Float32(0.0)
    if lm0 > neg_inf:
        p00 = _exp2_approx_ftz_f32(s_lo0 - lm0) if s_lo0 > neg_inf else Float32(0.0)
        p01 = _exp2_approx_ftz_f32(s_lo1 - lm0) if s_lo1 > neg_inf else Float32(0.0)
        ls0 = p00 + p01
    cute.arch.sync_warp()
    ls0 = ls0 + cute.arch.shuffle_sync_bfly(ls0, offset=2)
    ls0 = ls0 + cute.arch.shuffle_sync_bfly(ls0, offset=1)

    lm1 = fmax_f32(s_hi0, s_hi1)
    lm1 = fmax_f32(lm1, cute.arch.shuffle_sync_bfly(lm1, offset=2))
    lm1 = fmax_f32(lm1, cute.arch.shuffle_sync_bfly(lm1, offset=1))
    p10 = Float32(0.0)
    p11 = Float32(0.0)
    ls1 = Float32(0.0)
    if lm1 > neg_inf:
        p10 = _exp2_approx_ftz_f32(s_hi0 - lm1) if s_hi0 > neg_inf else Float32(0.0)
        p11 = _exp2_approx_ftz_f32(s_hi1 - lm1) if s_hi1 > neg_inf else Float32(0.0)
        ls1 = p10 + p11
    cute.arch.sync_warp()
    ls1 = ls1 + cute.arch.shuffle_sync_bfly(ls1, offset=2)
    ls1 = ls1 + cute.arch.shuffle_sync_bfly(ls1, offset=1)

    reduce_max_addr = reduce_addr
    reduce_sum_addr = reduce_addr + Int32(4 * _M_TILE * 4)
    if tid == Int32(0):
        row_lo = m_tile * Int32(16) + gid
        row_hi = row_lo + Int32(8)
        slot_lo = (cand_grp * Int32(_M_TILE) + row_lo) * Int32(4)
        slot_hi = (cand_grp * Int32(_M_TILE) + row_hi) * Int32(4)
        st_shared_f32(reduce_max_addr + slot_lo, lm0)
        st_shared_f32(reduce_sum_addr + slot_lo, ls0)
        st_shared_f32(reduce_max_addr + slot_hi, lm1)
        st_shared_f32(reduce_sum_addr + slot_hi, ls1)

    cute.arch.sync_threads()

    alphas = [Float32(1.0)] * 8
    new_ms = [neg_inf] * 8
    for ri in cutlass.range_constexpr(8):
        row_id = Int32(ri * 8) + gid
        s0 = row_id * Int32(4)
        s1 = (Int32(_M_TILE) + row_id) * Int32(4)
        wm0 = ld_shared_f32(reduce_max_addr + s0)
        ws0 = ld_shared_f32(reduce_sum_addr + s0)
        wm1 = ld_shared_f32(reduce_max_addr + s1)
        ws1 = ld_shared_f32(reduce_sum_addr + s1)
        bm = fmax_f32(wm0, wm1)
        if bm > neg_inf:
            old_m = global_max[ri]
            new_m = fmax_f32(old_m, bm)
            alpha = Float32(0.0)
            if old_m > neg_inf:
                alpha = _exp2_approx_ftz_f32(old_m - new_m)
            bs = Float32(0.0)
            if wm0 > neg_inf:
                bs = bs + ws0 * _exp2_approx_ftz_f32(wm0 - new_m)
            if wm1 > neg_inf:
                bs = bs + ws1 * _exp2_approx_ftz_f32(wm1 - new_m)
            alphas[ri] = alpha
            new_ms[ri] = new_m
            global_sum[ri] = global_sum[ri] * alpha + bs
            global_max[ri] = new_m

    row_lo = m_tile * Int32(16) + gid
    row_hi = row_lo + Int32(8)
    nm_lo = new_ms[0]
    nm_hi = new_ms[1]
    if m_tile == Int32(1):
        nm_lo = new_ms[2]
        nm_hi = new_ms[3]
    elif m_tile == Int32(2):
        nm_lo = new_ms[4]
        nm_hi = new_ms[5]
    elif m_tile == Int32(3):
        nm_lo = new_ms[6]
        nm_hi = new_ms[7]

    rsc_lo = (
        _exp2_approx_ftz_f32(lm0 - nm_lo)
        if (lm0 > neg_inf and nm_lo > neg_inf)
        else Float32(0.0)
    )
    rsc_hi = (
        _exp2_approx_ftz_f32(lm1 - nm_hi)
        if (lm1 > neg_inf and nm_hi > neg_inf)
        else Float32(0.0)
    )
    _store_p_pair_swizzled_64b(
        sm_p_addr, row_lo, c0, pack_f32x2_to_bfloat2(p00 * rsc_lo, p01 * rsc_lo)
    )
    _store_p_pair_swizzled_64b(
        sm_p_addr, row_hi, c0, pack_f32x2_to_bfloat2(p10 * rsc_hi, p11 * rsc_hi)
    )

    for mt in cutlass.range_constexpr(4):
        a_lo = alphas[mt * 2]
        a_hi = alphas[mt * 2 + 1]
        for tile in cutlass.range_constexpr(8):
            idx = mt * 32 + tile * 4
            acc_o[idx + 0] = acc_o[idx + 0] * a_lo
            acc_o[idx + 1] = acc_o[idx + 1] * a_lo
            acc_o[idx + 2] = acc_o[idx + 2] * a_hi
            acc_o[idx + 3] = acc_o[idx + 3] * a_hi

    cute.arch.sync_threads()

    _pv_m64_shared_dequant(
        acc_o, sm_p_addr, kv_stage_addr, mask_stage_addr, warp_id, lane, swa=True
    )


class BitmaskSplitMergeKernel:
    """Fast L2-resident split-K merge for ``BitmaskUnionVerifyMLAKernel``.

    One warp (32 threads) owns one ``(row, head)`` pair and reduces all
    ``num_splits`` BF16 partial vectors (512 dims = 2 x 16B vectors/lane) in a
    single coalesced pass with exact ``attn_sink`` folding, optional
    ``return_lse`` (natural or base-2), and ``live_counts`` dead-token skip.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        verify_width: int,
        num_splits: int,
        has_attn_sink: bool,
        return_lse: bool,
        lse_is_natural: bool,
        has_live_counts: bool,
    ):
        self.num_heads = int(num_heads)
        self.verify_width = int(verify_width)
        self.num_splits = int(num_splits)
        self.has_attn_sink = bool(has_attn_sink)
        self.return_lse = bool(return_lse)
        self.lse_is_natural = bool(lse_is_natural)
        self.has_live_counts = bool(has_live_counts)
        self.warps_per_cta = 4

    @cute.jit
    def __call__(
        self,
        mid_out: cute.Tensor,  # (rows, H, S_cap, 512) bf16
        mid_lse: cute.Tensor,  # (rows, H, S_cap) f32 (base-2)
        attn_sink: cute.Tensor,  # (H,) f32 (natural log)
        live_counts: cute.Tensor,  # (B,) int32
        output: cute.Tensor,  # (rows, H, 512) bf16
        out_lse: cute.Tensor,  # (rows, H) f32
        total_rows: Int32,
        stream: cuda.CUstream,
    ):
        total_pairs = total_rows * Int32(self.num_heads)
        grid_x = (total_pairs + Int32(self.warps_per_cta - 1)) // Int32(
            self.warps_per_cta
        )
        self.kernel(
            mid_out,
            mid_lse,
            attn_sink,
            live_counts,
            output,
            out_lse,
            total_pairs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(self.warps_per_cta * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        live_counts: cute.Tensor,
        output: cute.Tensor,
        out_lse: cute.Tensor,
        total_pairs: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        warp_in_cta = tid >> Int32(5)
        lane = tid & Int32(31)
        pair_id = (
            Int32(cute.arch.block_idx()[0]) * Int32(self.warps_per_cta) + warp_in_cta
        )

        if pair_id < total_pairs:
            row = pair_id // Int32(self.num_heads)
            h = pair_id - row * Int32(self.num_heads)

            row_live = Boolean(True)
            if cutlass.const_expr(self.has_live_counts):
                b_idx = row // Int32(self.verify_width)
                tok = row - b_idx * Int32(self.verify_width)
                row_live = tok < Int32(live_counts[b_idx])

            if row_live:
                neg_inf = Float32(_NEG_INF)
                lse_arr = cute.make_rmem_tensor(self.num_splits, Float32)
                m_tot = neg_inf
                for s in cutlass.range_constexpr(self.num_splits):
                    lv = Float32(mid_lse[row, h, Int32(s)])
                    lse_arr[s] = lv
                    m_tot = fmax_f32(m_tot, lv)

                l_tot = Float32(0.0)
                for s in cutlass.range_constexpr(self.num_splits):
                    lv = lse_arr[s]
                    if lv > neg_inf:
                        l_tot = l_tot + _exp2_approx_ftz_f32(lv - m_tot)

                inv_d = Float32(0.0)
                final_lse = neg_inf
                if l_tot != Float32(0.0):
                    final_lse = _log2_approx_ftz_f32(l_tot) + m_tot
                    if cutlass.const_expr(self.has_attn_sink):
                        sk = Float32(attn_sink[h]) * Float32(LOG2_E)
                        nm = fmax_f32(m_tot, sk)
                        ps = _exp2_approx_ftz_f32(m_tot - nm)
                        ss = _exp2_approx_ftz_f32(sk - nm)
                        denom = l_tot * ps + ss
                        if denom != Float32(0.0):
                            inv_d = ps * rcp_approx_ftz(denom)
                        mm = fmax_f32(final_lse, sk)
                        final_lse = mm + _log2_approx_ftz_f32(
                            Float32(1.0)
                            + _exp2_approx_ftz_f32(fmin_f32(final_lse, sk) - mm)
                        )
                    else:
                        inv_d = rcp_approx_ftz(l_tot)
                else:
                    if cutlass.const_expr(self.has_attn_sink):
                        final_lse = Float32(attn_sink[h]) * Float32(LOG2_E)

                if cutlass.const_expr(self.return_lse):
                    if lane == Int32(0):
                        out_v = final_lse
                        if cutlass.const_expr(self.lse_is_natural):
                            if out_v != neg_inf:
                                out_v = out_v * Float32(_LN2)
                        out_lse[row, h] = out_v

                # Each lane owns 2 x 8 = 16 dims (d_a = lane*8, d_b = lane*8 + 256).
                d_a = lane * Int32(8)
                d_b = d_a + Int32(256)
                a0 = Float32(0.0)
                a1 = Float32(0.0)
                a2 = Float32(0.0)
                a3 = Float32(0.0)
                a4 = Float32(0.0)
                a5 = Float32(0.0)
                a6 = Float32(0.0)
                a7 = Float32(0.0)
                b0 = Float32(0.0)
                b1 = Float32(0.0)
                b2 = Float32(0.0)
                b3 = Float32(0.0)
                b4 = Float32(0.0)
                b5 = Float32(0.0)
                b6 = Float32(0.0)
                b7 = Float32(0.0)

                for s in cutlass.range_constexpr(self.num_splits):
                    lv = lse_arr[s]
                    if (lv > neg_inf) and (inv_d != Float32(0.0)):
                        ws = _exp2_approx_ftz_f32(lv - m_tot) * inv_d
                        off_a = cute.crd2idx((row, h, Int32(s), d_a), mid_out.layout)
                        off_b = cute.crd2idx((row, h, Int32(s), d_b), mid_out.layout)
                        u0, u1, u2, u3 = ld_global_v4_u32(
                            get_ptr_as_int64(mid_out, off_a)
                        )
                        v0, v1, v2, v3 = ld_global_v4_u32(
                            get_ptr_as_int64(mid_out, off_b)
                        )
                        f0, f1 = _bf16x2_to_f32x2(u0)
                        f2, f3 = _bf16x2_to_f32x2(u1)
                        f4, f5 = _bf16x2_to_f32x2(u2)
                        f6, f7 = _bf16x2_to_f32x2(u3)
                        a0 = a0 + ws * f0
                        a1 = a1 + ws * f1
                        a2 = a2 + ws * f2
                        a3 = a3 + ws * f3
                        a4 = a4 + ws * f4
                        a5 = a5 + ws * f5
                        a6 = a6 + ws * f6
                        a7 = a7 + ws * f7
                        g0, g1 = _bf16x2_to_f32x2(v0)
                        g2, g3 = _bf16x2_to_f32x2(v1)
                        g4, g5 = _bf16x2_to_f32x2(v2)
                        g6, g7 = _bf16x2_to_f32x2(v3)
                        b0 = b0 + ws * g0
                        b1 = b1 + ws * g1
                        b2 = b2 + ws * g2
                        b3 = b3 + ws * g3
                        b4 = b4 + ws * g4
                        b5 = b5 + ws * g5
                        b6 = b6 + ws * g6
                        b7 = b7 + ws * g7

                out_a = cute.crd2idx((row, h, d_a), output.layout)
                out_b = cute.crd2idx((row, h, d_b), output.layout)
                store_v4_bf16x2(
                    get_ptr_as_int64(output, out_a), a0, a1, a2, a3, a4, a5, a6, a7
                )
                store_v4_bf16x2(
                    get_ptr_as_int64(output, out_b), b0, b1, b2, b3, b4, b5, b6, b7
                )


@cute.jit
def _issue_swa_stage(
    swa_kv_u8: cute.Tensor,
    swa_unique_keys: cute.Tensor,
    swa_mask_bits: cute.Tensor,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    mbar_ptr,
    b_idx: Int32,
    swa_row_idx: Int32,
    swa_shift: Int32,
    swa_lo_off: Int32,
    swa_hist: Int32,
    swa_len: Int32,
    q_group_start: Int32,
    live_b: Int32,
    group_valid_bits: Uint32,
    stride_swa_block: Int64,
    ci: Int32,
    stg: Int32,
    lane: Int32,
    *,
    swa_contiguous: cutlass.Constexpr,
    has_swa_mask: cutlass.Constexpr,
    swa_page_size: cutlass.Constexpr,
) -> None:
    kv_dst = kv_stage_addr + stg * Int32(_BI_SWA * _SWA_SMEM_STRIDE)
    m_dst = mask_stage_addr + stg * Int32(_BI_IDX)
    full_mbar = mbar_ptr + stg
    full_mbar_u32 = shared_ptr_to_u32(full_mbar)
    chunk_start = ci * Int32(_BI_SWA)
    if cutlass.const_expr(swa_contiguous):
        if lane < Int32(_BI_SWA):
            rel_pos = chunk_start + lane
            abs_pos = swa_lo_off + rel_pos
            m4 = Uint32(0)
            if rel_pos < swa_len:
                if cutlass.const_expr(has_swa_mask):
                    m_raw = Uint32(swa_mask_bits[swa_row_idx, abs_pos])
                    m4 = (
                        (m_raw >> swa_shift.to(Uint32)) & Uint32(0xF) & group_valid_bits
                    )
                else:
                    for ql in cutlass.range_constexpr(4):
                        w_tok = q_group_start + Int32(ql)
                        hi_k = swa_hist + w_tok
                        lo_k = hi_k - Int32(127)
                        if (w_tok < live_b) and (abs_pos >= lo_k) and (abs_pos <= hi_k):
                            m4 = m4 | Uint32(1 << ql)
            st_shared_u8(m_dst + lane, m4.to(Uint8))
        cute.arch.sync_warp()
        cute.arch.fence_acq_rel_cta()
        chunk_valid = swa_len - chunk_start
        if chunk_valid > Int32(_BI_SWA):
            chunk_valid = Int32(_BI_SWA)
        if chunk_valid < Int32(0):
            chunk_valid = Int32(0)
        if lane == Int32(0):
            tx_bytes = chunk_valid * Int32(_SWA_RECORD_BYTES)
            cute.arch.mbarrier_arrive_and_expect_tx(full_mbar, tx_bytes)
            if chunk_valid > Int32(0):
                src_off = Int64(b_idx) * stride_swa_block + Int64(
                    swa_lo_off + chunk_start
                ) * Int64(_SWA_RECORD_BYTES)
                cp_async_bulk_g2s_mbar(
                    kv_dst,
                    get_ptr_as_int64(swa_kv_u8, src_off),
                    tx_bytes,
                    full_mbar_u32,
                )
        else:
            cute.arch.mbarrier_arrive(full_mbar)
    else:
        m4 = Uint32(0)
        key = Int32(-1)
        pos = chunk_start + lane
        if (lane < Int32(_BI_SWA)) and (pos < swa_len):
            key = Int32(swa_unique_keys[swa_row_idx, pos])
            if key >= Int32(0):
                m_raw = Uint32(swa_mask_bits[swa_row_idx, pos])
                m4 = (m_raw >> swa_shift.to(Uint32)) & Uint32(0xF) & group_valid_bits
        if lane < Int32(_BI_SWA):
            st_shared_u8(m_dst + lane, m4.to(Uint8))
        act = (lane < Int32(_BI_SWA)) and (m4 != Uint32(0))
        n_act = _popc_u32(cute.arch.vote_ballot_sync(Boolean(act)))
        cute.arch.sync_warp()
        cute.arch.fence_acq_rel_cta()
        if lane == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_mbar, n_act * Int32(_SWA_RECORD_BYTES)
            )
        else:
            cute.arch.mbarrier_arrive(full_mbar)
        if act:
            blk = key // Int32(swa_page_size)
            loc_k = key - blk * Int32(swa_page_size)
            src_off = Int64(blk) * stride_swa_block + Int64(loc_k) * Int64(
                _SWA_RECORD_BYTES
            )
            cp_async_bulk_g2s_mbar(
                kv_dst + lane * Int32(_SWA_SMEM_STRIDE),
                get_ptr_as_int64(swa_kv_u8, src_off),
                Int32(_SWA_RECORD_BYTES),
                full_mbar_u32,
            )


@cute.jit
def _issue_idx_stage(
    idx_kv_u8: cute.Tensor,
    idx_unique_keys: cute.Tensor,
    idx_mask_bits: cute.Tensor,
    idx_page_table: cute.Tensor,
    kv_stage_addr: Int32,
    mask_stage_addr: Int32,
    mbar_ptr,
    b_idx: Int32,
    idx_row_idx: Int32,
    idx_shift: Int32,
    idx_len: Int32,
    idx_table_width: Int32,
    idx_num_pages: Int64,
    group_valid_bits: Uint32,
    stride_idx_block: Int64,
    ci: Int32,
    stg: Int32,
    lane: Int32,
    *,
    has_indexed_page_table: cutlass.Constexpr,
    indexed_page_size: cutlass.Constexpr,
) -> None:
    kv_dst = kv_stage_addr + stg * Int32(_BI_IDX * _IDX_SMEM_STRIDE)
    m_dst = mask_stage_addr + stg * Int32(_BI_IDX)
    full_mbar = mbar_ptr + stg
    full_mbar_u32 = shared_ptr_to_u32(full_mbar)
    pos = ci * Int32(_BI_IDX) + lane
    m4 = Uint32(0)
    slot = Int32(-1)
    if pos < idx_len:
        raw_k = Int32(idx_unique_keys[idx_row_idx, pos])
        if raw_k >= Int32(0):
            m_raw = Uint32(idx_mask_bits[idx_row_idx, pos])
            m4 = (m_raw >> idx_shift.to(Uint32)) & Uint32(0xF) & group_valid_bits
            if m4 != Uint32(0):
                if cutlass.const_expr(has_indexed_page_table):
                    page = raw_k // Int32(indexed_page_size)
                    if (page >= Int32(0)) and (page < idx_table_width):
                        pid = Int64(idx_page_table[b_idx, page])
                        if (pid >= Int64(0)) and (pid < idx_num_pages):
                            slot_i64 = pid * Int64(indexed_page_size) + Int64(
                                raw_k - page * Int32(indexed_page_size)
                            )
                            if slot_i64 <= Int64(2147483647):
                                slot = slot_i64.to(Int32)
                    if slot < Int32(0):
                        m4 = Uint32(0)
                else:
                    slot = raw_k
    st_shared_u8(m_dst + lane, m4.to(Uint8))
    act = m4 != Uint32(0)
    n_act = _popc_u32(cute.arch.vote_ballot_sync(Boolean(act)))
    cute.arch.sync_warp()
    cute.arch.fence_acq_rel_cta()
    if lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar, n_act * Int32(_IDX_RECORD_BYTES)
        )
    else:
        cute.arch.mbarrier_arrive(full_mbar)
    if act:
        blk = slot // Int32(indexed_page_size)
        loc_k = slot - blk * Int32(indexed_page_size)
        src_off = Int64(blk) * stride_idx_block + Int64(loc_k) * Int64(
            _IDX_RECORD_BYTES
        )
        cp_async_bulk_g2s_mbar(
            kv_dst + lane * Int32(_IDX_SMEM_STRIDE),
            get_ptr_as_int64(idx_kv_u8, src_off),
            Int32(_IDX_RECORD_BYTES),
            full_mbar_u32,
        )


class BitmaskUnionVerifyMLAKernel:
    """SM120 multi-query bitmask-union Sparse MLA verification kernel (M=64, BI=32/16).

    Grid = ``(B * Q_groups, H_blocks, num_splits)``,
    Block = ``(256, 1, 1)`` (8 warps = 2 warps/scheduler -> full 255 regs/thread, 0 spills,
    with warp 0 issuing 3-stage async TMA prefetch 2 chunks ahead),
    Cluster = ``(1, 1, num_splits)`` when ``use_dsmem=True``, else ``(1, 1, 1)``.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        verify_width: int,
        num_splits: int,
        use_dsmem: bool,
        has_indexed: bool,
        swa_contiguous: bool,
        has_swa_mask: bool,
        swa_mask_is_u8: bool,
        swa_per_group: bool,
        idx_mask_is_u8: bool,
        idx_per_group: bool,
        has_indexed_page_table: bool,
        has_attn_sink: bool,
        return_lse: bool,
        lse_is_natural: bool,
        has_live_counts: bool,
        swa_page_size: int,
        indexed_page_size: int,
    ):
        if not (1 <= int(num_splits) <= _MAX_SPLITS):
            raise ValueError(
                f"num_splits must be in 1..{_MAX_SPLITS}, got {num_splits}"
            )
        if use_dsmem and int(num_splits) not in (2, 4, 8):
            raise ValueError(
                f"DSMEM cluster mode requires num_splits in (2, 4, 8), got {num_splits}"
            )
        if not (1 <= int(verify_width) <= 32):
            raise ValueError(f"verify_width must be in 1..32, got {verify_width}")
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.num_heads = int(num_heads)
        self.verify_width = int(verify_width)
        self.q_groups = (self.verify_width + _Q_TILE - 1) // _Q_TILE
        self.h_blocks = (self.num_heads + _H_TILE - 1) // _H_TILE
        self.num_splits = int(num_splits)
        self.use_dsmem = bool(use_dsmem and self.num_splits > 1)
        self.has_indexed = bool(has_indexed)
        self.swa_contiguous = bool(swa_contiguous)
        self.has_swa_mask = bool(has_swa_mask)
        self.swa_mask_is_u8 = bool(swa_mask_is_u8)
        self.swa_per_group = bool(swa_per_group)
        self.idx_mask_is_u8 = bool(idx_mask_is_u8)
        self.idx_per_group = bool(idx_per_group)
        self.has_indexed_page_table = bool(has_indexed_page_table)
        self.has_attn_sink = bool(has_attn_sink)
        self.return_lse = bool(return_lse)
        self.lse_is_natural = bool(lse_is_natural)
        self.has_live_counts = bool(has_live_counts)
        self.swa_page_size = max(1, int(swa_page_size))
        self.indexed_page_size = max(1, int(indexed_page_size))

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (B*W, H, 512) bf16
        swa_kv_u8: cute.Tensor,  # flat u8
        swa_unique_keys: cute.Tensor,  # (B or B*G, S_max) int32
        swa_mask_bits: cute.Tensor,  # (B or B*G, S_max) uint8 or int32
        swa_union_len: cute.Tensor,  # (B or B*G,) int32
        idx_kv_u8: cute.Tensor,  # flat u8
        idx_unique_keys: cute.Tensor,  # (B or B*G, U_max) int32
        idx_mask_bits: cute.Tensor,  # (B or B*G, U_max) uint8 or int32
        idx_union_len: cute.Tensor,  # (B or B*G,) int32
        idx_page_table: cute.Tensor,  # (B, max_pages) int32
        attn_sink: cute.Tensor,  # (H,) f32
        live_counts: cute.Tensor,  # (B,) int32
        output: cute.Tensor,  # (B*W, H, 512) bf16
        out_lse: cute.Tensor,  # (B*W, H) f32
        mid_out: cute.Tensor,  # (B*W, H, S_cap, 512) bf16
        mid_lse: cute.Tensor,  # (B*W, H, S_cap) f32
        sm_scale_log2: Float32,
        stride_swa_block: Int64,
        stride_idx_block: Int64,
        swa_max_keys: Int32,
        idx_max_keys: Int32,
        idx_table_width: Int32,
        idx_num_pages: Int64,
        batch_size: Int32,
        stream: cuda.CUstream,
    ):
        cluster_z = self.num_splits if self.use_dsmem else 1
        self.kernel(
            q,
            swa_kv_u8,
            swa_unique_keys,
            swa_mask_bits,
            swa_union_len,
            idx_kv_u8,
            idx_unique_keys,
            idx_mask_bits,
            idx_union_len,
            idx_page_table,
            attn_sink,
            live_counts,
            output,
            out_lse,
            mid_out,
            mid_lse,
            sm_scale_log2,
            stride_swa_block,
            stride_idx_block,
            swa_max_keys,
            idx_max_keys,
            idx_table_width,
            idx_num_pages,
        ).launch(
            grid=(batch_size * Int32(self.q_groups), self.h_blocks, self.num_splits),
            block=(_BLOCK_THREADS, 1, 1),
            cluster=(1, 1, cluster_z),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        swa_kv_u8: cute.Tensor,
        swa_unique_keys: cute.Tensor,
        swa_mask_bits: cute.Tensor,
        swa_union_len: cute.Tensor,
        idx_kv_u8: cute.Tensor,
        idx_unique_keys: cute.Tensor,
        idx_mask_bits: cute.Tensor,
        idx_union_len: cute.Tensor,
        idx_page_table: cute.Tensor,
        attn_sink: cute.Tensor,
        live_counts: cute.Tensor,
        output: cute.Tensor,
        out_lse: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        stride_swa_block: Int64,
        stride_idx_block: Int64,
        swa_max_keys: Int32,
        idx_max_keys: Int32,
        idx_table_width: Int32,
        idx_num_pages: Int64,
    ):
        bq_idx, h_blk, split_idx = cute.arch.block_idx()
        bq_idx = Int32(bq_idx)
        h_blk = Int32(h_blk)
        split_idx = Int32(split_idx)

        b_idx = bq_idx // Int32(self.q_groups)
        q_group = bq_idx - b_idx * Int32(self.q_groups)
        h_base = h_blk * Int32(_H_TILE)

        tid = Int32(cute.arch.thread_idx()[0])
        warp_id = tid >> Int32(5)
        lane = tid & Int32(31)

        smem = cutlass_utils.SmemAllocator()
        s_raw = smem.allocate_tensor(Uint8, cute.make_layout((_RAW_SMEM_BYTES,)), 128)
        s_mbar = smem.allocate_tensor(Uint64, cute.make_layout((3,)), 16)
        raw_base = shared_ptr_to_u32(s_raw.iterator)
        q_bf16_addr = raw_base
        kv_stage_addr = raw_base + Int32(_Q_BF16_BYTES)
        sm_p_addr = kv_stage_addr + Int32(_KV_STAGE_BYTES)
        reduce_addr = sm_p_addr + Int32(_SM_P_BYTES)
        mask_stage_addr = reduce_addr + Int32(_REDUCE_BYTES)
        mbar_ptr = s_mbar.iterator

        # Resolve per-request live token width [0, W] and 4-bit group valid mask.
        live_b = Int32(self.verify_width)
        if cutlass.const_expr(self.has_live_counts):
            live_b = Int32(live_counts[b_idx])
            if live_b < Int32(0):
                live_b = Int32(0)
            if live_b > Int32(self.verify_width):
                live_b = Int32(self.verify_width)
        q_group_start = q_group * Int32(_Q_TILE)
        group_live = live_b - q_group_start
        if group_live < Int32(0):
            group_live = Int32(0)
        if group_live > Int32(_Q_TILE):
            group_live = Int32(_Q_TILE)
        group_valid_bits = (Uint32(1) << group_live.to(Uint32)) - Uint32(1)

        # Resolve SWA (BI=16) and Indexed (BI=32) slice & chunk counts.
        swa_row_idx = bq_idx if cutlass.const_expr(self.swa_per_group) else b_idx
        swa_shift = (
            Int32(0) if cutlass.const_expr(self.swa_per_group) else (q_group * Int32(4))
        )
        idx_row_idx = bq_idx if cutlass.const_expr(self.idx_per_group) else b_idx
        idx_shift = (
            Int32(0) if cutlass.const_expr(self.idx_per_group) else (q_group * Int32(4))
        )

        swa_lo_off = Int32(0)
        swa_hist = Int32(0)
        swa_len = Int32(swa_union_len[swa_row_idx])
        if cutlass.const_expr(self.swa_contiguous and not self.has_swa_mask):
            swa_hist = swa_len
            if swa_hist < Int32(0):
                swa_hist = Int32(0)
            if group_live > Int32(0):
                swa_lo_off = swa_hist + q_group_start - Int32(127)
                if swa_lo_off < Int32(0):
                    swa_lo_off = Int32(0)
                swa_hi_end = swa_hist + q_group_start + group_live
                if swa_hi_end > swa_max_keys:
                    swa_hi_end = swa_max_keys
                swa_len = swa_hi_end - swa_lo_off
            else:
                swa_len = Int32(0)
        if swa_len < Int32(0):
            swa_len = Int32(0)
        if swa_len > swa_max_keys:
            swa_len = swa_max_keys
        swa_chunks = (swa_len + Int32(_BI_SWA - 1)) // Int32(_BI_SWA)

        idx_len = Int32(0)
        idx_chunks = Int32(0)
        if cutlass.const_expr(self.has_indexed):
            if group_live > Int32(0):
                idx_len = Int32(idx_union_len[idx_row_idx])
                if idx_len < Int32(0):
                    idx_len = Int32(0)
                if idx_len > idx_max_keys:
                    idx_len = idx_max_keys
                idx_chunks = (idx_len + Int32(_BI_IDX - 1)) // Int32(_BI_IDX)

        # Weighted split-K slicing: SWA BI=16 chunk = 1 unit, Indexed BI=32 chunk = 2 units.
        total_chunks = swa_chunks + idx_chunks
        if cutlass.const_expr(self.num_splits == 1):
            split_start = Int32(0)
            split_end = total_chunks
        else:
            c_s = Int32(self.num_splits)
            total_w = swa_chunks + idx_chunks * Int32(2)
            u_start = (split_idx * total_w) // c_s
            u_end = ((split_idx + Int32(1)) * total_w) // c_s
            split_start = (
                u_start
                if u_start <= swa_chunks
                else (swa_chunks + ((u_start - swa_chunks) >> Int32(1)))
            )
            split_end = (
                u_end
                if u_end <= swa_chunks
                else (swa_chunks + ((u_end - swa_chunks) >> Int32(1)))
            )

        swa_end = split_end if split_end < swa_chunks else swa_chunks
        swa_cnt = swa_end - split_start
        if swa_cnt < Int32(0):
            swa_cnt = Int32(0)
        swa_first = split_start

        idx_first_abs = split_start if split_start > swa_chunks else swa_chunks
        idx_cnt = split_end - idx_first_abs
        if idx_cnt < Int32(0):
            idx_cnt = Int32(0)
        idx_first = idx_first_abs - swa_chunks

        # Initialize mbarriers (full[0..2] arrive_count=32 so all 32 producer lanes establish happens-before).
        if tid == Int32(0):
            for s in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(mbar_ptr + s, Int32(32))
        cute.arch.sync_threads()

        # ── S0: Vectorized 16B BF16 Q staging (64 rows x 64 vecs = 4096 vecs = 16 iters/thread) ──
        if (swa_cnt + idx_cnt) > Int32(0):
            q_row_base_global = b_idx * Int32(self.verify_width)
            for it in cutlass.range_constexpr(16):
                v_idx = tid + Int32(it * _MATH_THREADS)
                r = v_idx >> Int32(6)  # 0..63: row = tok_local * 16 + h_local
                vec = v_idx & Int32(63)  # 0..63
                tok_local = r >> Int32(4)  # 0..3
                h_local = r & Int32(15)  # 0..15
                tok = q_group_start + tok_local
                h = h_base + h_local
                w0 = Uint32(0)
                w1 = Uint32(0)
                w2 = Uint32(0)
                w3 = Uint32(0)
                if (tok < live_b) and (h < Int32(self.num_heads)):
                    q_off = cute.crd2idx(
                        (q_row_base_global + tok, h, vec * Int32(8)), q.layout
                    )
                    w0, w1, w2, w3 = ld_global_nc_v4_u32(get_ptr_as_int64(q, q_off))
                swiz_col = (vec * Int32(16)) ^ ((r & Int32(7)) << Int32(4))
                st_shared_v4_u32(
                    q_bf16_addr + r * Int32(1024) + swiz_col, w0, w1, w2, w3
                )
            cute.arch.sync_threads()

        # Accumulators (128 FP32 regs/thread) + online-softmax state (8 rows/lane).
        acc_o = cute.make_rmem_tensor(128, Float32)
        for i_z in cutlass.range_constexpr(128):
            acc_o[i_z] = Float32(0.0)
        global_max = cute.make_rmem_tensor(8, Float32)
        global_sum = cute.make_rmem_tensor(8, Float32)
        for ri in cutlass.range_constexpr(8):
            global_max[ri] = Float32(_NEG_INF)
            global_sum[ri] = Float32(0.0)

        # ════════════════════════════════════════════════════════════════════
        # SECTION 1: SWA (528 B E4M3 + UE8M0, 3-stage inline TMA ring, BI=16)
        # ════════════════════════════════════════════════════════════════════
        if swa_cnt > Int32(0):
            if warp_id == Int32(0):
                _issue_swa_stage(
                    swa_kv_u8,
                    swa_unique_keys,
                    swa_mask_bits,
                    kv_stage_addr,
                    mask_stage_addr,
                    mbar_ptr,
                    b_idx,
                    swa_row_idx,
                    swa_shift,
                    swa_lo_off,
                    swa_hist,
                    swa_len,
                    q_group_start,
                    live_b,
                    group_valid_bits,
                    stride_swa_block,
                    swa_first,
                    Int32(0),
                    lane,
                    swa_contiguous=self.swa_contiguous,
                    has_swa_mask=self.has_swa_mask,
                    swa_page_size=self.swa_page_size,
                )
                if swa_cnt > Int32(1):
                    _issue_swa_stage(
                        swa_kv_u8,
                        swa_unique_keys,
                        swa_mask_bits,
                        kv_stage_addr,
                        mask_stage_addr,
                        mbar_ptr,
                        b_idx,
                        swa_row_idx,
                        swa_shift,
                        swa_lo_off,
                        swa_hist,
                        swa_len,
                        q_group_start,
                        live_b,
                        group_valid_bits,
                        stride_swa_block,
                        swa_first + Int32(1),
                        Int32(1),
                        lane,
                        swa_contiguous=self.swa_contiguous,
                        has_swa_mask=self.has_swa_mask,
                        swa_page_size=self.swa_page_size,
                    )
            cons_idx = Int32(0)
            cons_phase = Int32(0)
            for lc in cutlass.range(swa_cnt, unroll=1):
                if (warp_id == Int32(0)) and (Int32(lc) + Int32(2) < swa_cnt):
                    iss_stg = cons_idx + Int32(2)
                    if iss_stg >= Int32(_SWA_STAGES):
                        iss_stg = iss_stg - Int32(_SWA_STAGES)
                    _issue_swa_stage(
                        swa_kv_u8,
                        swa_unique_keys,
                        swa_mask_bits,
                        kv_stage_addr,
                        mask_stage_addr,
                        mbar_ptr,
                        b_idx,
                        swa_row_idx,
                        swa_shift,
                        swa_lo_off,
                        swa_hist,
                        swa_len,
                        q_group_start,
                        live_b,
                        group_valid_bits,
                        stride_swa_block,
                        swa_first + Int32(lc) + Int32(2),
                        iss_stg,
                        lane,
                        swa_contiguous=self.swa_contiguous,
                        has_swa_mask=self.has_swa_mask,
                        swa_page_size=self.swa_page_size,
                    )
                cute.arch.mbarrier_wait(mbar_ptr + cons_idx, phase=cons_phase)
                kv_buf_addr = kv_stage_addr + cons_idx * Int32(
                    _BI_SWA * _SWA_SMEM_STRIDE
                )
                m_buf_addr = mask_stage_addr + cons_idx * Int32(_BI_IDX)
                _process_chunk_swa_16(
                    acc_o,
                    global_max,
                    global_sum,
                    q_bf16_addr,
                    kv_buf_addr,
                    m_buf_addr,
                    sm_p_addr,
                    reduce_addr,
                    sm_scale_log2,
                    warp_id,
                    lane,
                )
                cute.arch.sync_threads()
                cons_idx = cons_idx + Int32(1)
                if cons_idx == Int32(_SWA_STAGES):
                    cons_idx = Int32(0)
                    cons_phase = cons_phase ^ Int32(1)

        # ════════════════════════════════════════════════════════════════════
        # SECTION 2: INDEXED NVFP4 (288 B E2M1 + E4M3, 3-stage inline TMA ring, BI=32)
        # ════════════════════════════════════════════════════════════════════
        if cutlass.const_expr(self.has_indexed):
            if (swa_cnt > Int32(0)) and (idx_cnt > Int32(0)):
                if tid == Int32(0):
                    for s in cutlass.range_constexpr(3):
                        cute.arch.mbarrier_init(mbar_ptr + s, Int32(32))
                cute.arch.sync_threads()

            if idx_cnt > Int32(0):
                if warp_id == Int32(0):
                    _issue_idx_stage(
                        idx_kv_u8,
                        idx_unique_keys,
                        idx_mask_bits,
                        idx_page_table,
                        kv_stage_addr,
                        mask_stage_addr,
                        mbar_ptr,
                        b_idx,
                        idx_row_idx,
                        idx_shift,
                        idx_len,
                        idx_table_width,
                        idx_num_pages,
                        group_valid_bits,
                        stride_idx_block,
                        idx_first,
                        Int32(0),
                        lane,
                        has_indexed_page_table=self.has_indexed_page_table,
                        indexed_page_size=self.indexed_page_size,
                    )
                    if idx_cnt > Int32(1):
                        _issue_idx_stage(
                            idx_kv_u8,
                            idx_unique_keys,
                            idx_mask_bits,
                            idx_page_table,
                            kv_stage_addr,
                            mask_stage_addr,
                            mbar_ptr,
                            b_idx,
                            idx_row_idx,
                            idx_shift,
                            idx_len,
                            idx_table_width,
                            idx_num_pages,
                            group_valid_bits,
                            stride_idx_block,
                            idx_first + Int32(1),
                            Int32(1),
                            lane,
                            has_indexed_page_table=self.has_indexed_page_table,
                            indexed_page_size=self.indexed_page_size,
                        )
                cons_idx = Int32(0)
                cons_phase = Int32(0)
                for lc in cutlass.range(idx_cnt, unroll=1):
                    if (warp_id == Int32(0)) and (Int32(lc) + Int32(2) < idx_cnt):
                        iss_stg = cons_idx + Int32(2)
                        if iss_stg >= Int32(_IDX_STAGES):
                            iss_stg = iss_stg - Int32(_IDX_STAGES)
                        _issue_idx_stage(
                            idx_kv_u8,
                            idx_unique_keys,
                            idx_mask_bits,
                            idx_page_table,
                            kv_stage_addr,
                            mask_stage_addr,
                            mbar_ptr,
                            b_idx,
                            idx_row_idx,
                            idx_shift,
                            idx_len,
                            idx_table_width,
                            idx_num_pages,
                            group_valid_bits,
                            stride_idx_block,
                            idx_first + Int32(lc) + Int32(2),
                            iss_stg,
                            lane,
                            has_indexed_page_table=self.has_indexed_page_table,
                            indexed_page_size=self.indexed_page_size,
                        )
                    cute.arch.mbarrier_wait(mbar_ptr + cons_idx, phase=cons_phase)
                    kv_buf_addr = kv_stage_addr + cons_idx * Int32(
                        _BI_IDX * _IDX_SMEM_STRIDE
                    )
                    m_buf_addr = mask_stage_addr + cons_idx * Int32(_BI_IDX)
                    _process_chunk_idx_32(
                        acc_o,
                        global_max,
                        global_sum,
                        q_bf16_addr,
                        kv_buf_addr,
                        m_buf_addr,
                        sm_p_addr,
                        reduce_addr,
                        sm_scale_log2,
                        warp_id,
                        lane,
                    )
                    cute.arch.sync_threads()
                    cons_idx = cons_idx + Int32(1)
                    if cons_idx == Int32(_IDX_STAGES):
                        cons_idx = Int32(0)
                        cons_phase = cons_phase ^ Int32(1)

        # ════════════════════════════════════════════════════════════════════
        # EPILOGUE: Direct (s=1) | Workspace L2 Partial (s>1) | BF16 DSMEM (s in 2,4,8)
        # ════════════════════════════════════════════════════════════════════
        gid = lane >> Int32(2)
        tid_in_g = lane & Int32(3)
        q_row_base_global = b_idx * Int32(self.verify_width)
        neg_inf = Float32(_NEG_INF)

        if cutlass.const_expr(self.num_splits == 1):
            # ── Mode 1: Direct BF16 Output + Optional Sink Fold + Optional LSE ──
            if cutlass.const_expr(True):
                cute.arch.sync_threads()
                h_lo = h_base + gid
                h_hi = h_lo + Int32(8)
                sink_lo = Float32(0.0)
                sink_hi = Float32(0.0)
                if cutlass.const_expr(self.has_attn_sink):
                    if h_lo < Int32(self.num_heads):
                        sink_lo = Float32(attn_sink[h_lo]) * Float32(LOG2_E)
                    if h_hi < Int32(self.num_heads):
                        sink_hi = Float32(attn_sink[h_hi]) * Float32(LOG2_E)

                invs = [Float32(0.0)] * 8
                for ri in cutlass.range_constexpr(8):
                    gs = global_sum[ri]
                    if gs != Float32(0.0):
                        if cutlass.const_expr(self.has_attn_sink):
                            sk = (
                                sink_lo
                                if cutlass.const_expr((ri & 1) == 0)
                                else sink_hi
                            )
                            gm = global_max[ri]
                            nm = fmax_f32(gm, sk)
                            ps = _exp2_approx_ftz_f32(gm - nm)
                            ss = _exp2_approx_ftz_f32(sk - nm)
                            denom = gs * ps + ss
                            if denom != Float32(0.0):
                                invs[ri] = ps * rcp_approx_ftz(denom)
                        else:
                            invs[ri] = rcp_approx_ftz(gs)

                # Pack normalized BF16 into dead q_bf16 [64, 512] BF16 (65,536 B).
                v_base = warp_id * Int32(64)
                for mt in cutlass.range_constexpr(4):
                    inv_lo = invs[mt * 2]
                    inv_hi = invs[mt * 2 + 1]
                    r_lo = Int32(mt * 16) + gid
                    r_hi = r_lo + Int32(8)
                    for tile in cutlass.range_constexpr(8):
                        col = v_base + Int32(tile * 8) + tid_in_g * Int32(2)
                        idx = mt * 32 + tile * 4
                        p0 = pack_f32x2_to_bfloat2(
                            acc_o[idx + 0] * inv_lo, acc_o[idx + 1] * inv_lo
                        )
                        p1 = pack_f32x2_to_bfloat2(
                            acc_o[idx + 2] * inv_hi, acc_o[idx + 3] * inv_hi
                        )
                        st_shared_u32(
                            raw_base + (r_lo * Int32(512) + col) * Int32(2), p0
                        )
                        st_shared_u32(
                            raw_base + (r_hi * Int32(512) + col) * Int32(2), p1
                        )

                if cutlass.const_expr(self.return_lse):
                    if (warp_id == Int32(0)) and (tid_in_g == Int32(0)):
                        for ri in cutlass.range_constexpr(8):
                            tok = q_group_start + Int32(ri >> 1)
                            h_cur = h_lo if cutlass.const_expr((ri & 1) == 0) else h_hi
                            if (tok < live_b) and (h_cur < Int32(self.num_heads)):
                                lse_v = neg_inf
                                if global_sum[ri] != Float32(0.0):
                                    lse_v = (
                                        _log2_approx_ftz_f32(global_sum[ri])
                                        + global_max[ri]
                                    )
                                if cutlass.const_expr(self.has_attn_sink):
                                    sk = (
                                        sink_lo
                                        if cutlass.const_expr((ri & 1) == 0)
                                        else sink_hi
                                    )
                                    if lse_v != neg_inf:
                                        mm = fmax_f32(lse_v, sk)
                                        lse_v = mm + _log2_approx_ftz_f32(
                                            Float32(1.0)
                                            + _exp2_approx_ftz_f32(
                                                fmin_f32(lse_v, sk) - mm
                                            )
                                        )
                                    else:
                                        lse_v = sk
                                if cutlass.const_expr(self.lse_is_natural):
                                    if lse_v != neg_inf:
                                        lse_v = lse_v * Float32(_LN2)
                                out_lse[q_row_base_global + tok, h_cur] = lse_v

                cute.arch.sync_threads()

                # Coalesced 16B global stores (16 iters x 256 threads = 4096 vectors).
                for it in cutlass.range_constexpr(16):
                    v_idx = tid + Int32(it * _MATH_THREADS)
                    r = v_idx >> Int32(6)
                    vec = v_idx & Int32(63)
                    tok = q_group_start + (r >> Int32(4))
                    h = h_base + (r & Int32(15))
                    if (tok < live_b) and (h < Int32(self.num_heads)):
                        v0, v1, v2, v3 = ld_shared_v4_u32(
                            raw_base + (r * Int32(512) + vec * Int32(8)) * Int32(2)
                        )
                        out_off = cute.crd2idx(
                            (q_row_base_global + tok, h, vec * Int32(8)), output.layout
                        )
                        st_global_v4_u32(
                            get_ptr_as_int64(output, out_off), v0, v1, v2, v3
                        )

        elif cutlass.const_expr(not self.use_dsmem):
            # ── Mode 2: Workspace Split-K Partial Output (BF16 mid_out + f32 mid_lse) ──
            if cutlass.const_expr(True):
                cute.arch.sync_threads()
                invs = [Float32(0.0)] * 8
                for ri in cutlass.range_constexpr(8):
                    gs = global_sum[ri]
                    if gs != Float32(0.0):
                        invs[ri] = rcp_approx_ftz(gs)

                # Write base-2 partial LSE (neg_inf when this split had no active keys).
                if (warp_id == Int32(0)) and (tid_in_g == Int32(0)):
                    h_lo = h_base + gid
                    h_hi = h_lo + Int32(8)
                    for ri in cutlass.range_constexpr(8):
                        tok = q_group_start + Int32(ri >> 1)
                        h_cur = h_lo if cutlass.const_expr((ri & 1) == 0) else h_hi
                        if (tok < live_b) and (h_cur < Int32(self.num_heads)):
                            lse_p = neg_inf
                            if global_sum[ri] != Float32(0.0):
                                lse_p = (
                                    _log2_approx_ftz_f32(global_sum[ri])
                                    + global_max[ri]
                                )
                            mid_lse[q_row_base_global + tok, h_cur, split_idx] = lse_p

                if (swa_cnt + idx_cnt) > Int32(0):
                    v_base = warp_id * Int32(64)
                    for mt in cutlass.range_constexpr(4):
                        inv_lo = invs[mt * 2]
                        inv_hi = invs[mt * 2 + 1]
                        r_lo = Int32(mt * 16) + gid
                        r_hi = r_lo + Int32(8)
                        for tile in cutlass.range_constexpr(8):
                            col = v_base + Int32(tile * 8) + tid_in_g * Int32(2)
                            idx = mt * 32 + tile * 4
                            p0 = pack_f32x2_to_bfloat2(
                                acc_o[idx + 0] * inv_lo, acc_o[idx + 1] * inv_lo
                            )
                            p1 = pack_f32x2_to_bfloat2(
                                acc_o[idx + 2] * inv_hi, acc_o[idx + 3] * inv_hi
                            )
                            st_shared_u32(
                                raw_base + (r_lo * Int32(512) + col) * Int32(2), p0
                            )
                            st_shared_u32(
                                raw_base + (r_hi * Int32(512) + col) * Int32(2), p1
                            )

                    cute.arch.sync_threads()

                    for it in cutlass.range_constexpr(16):
                        v_idx = tid + Int32(it * _MATH_THREADS)
                        r = v_idx >> Int32(6)
                        vec = v_idx & Int32(63)
                        tok = q_group_start + (r >> Int32(4))
                        h = h_base + (r & Int32(15))
                        if (tok < live_b) and (h < Int32(self.num_heads)):
                            v0, v1, v2, v3 = ld_shared_v4_u32(
                                raw_base + (r * Int32(512) + vec * Int32(8)) * Int32(2)
                            )
                            mo_off = cute.crd2idx(
                                (
                                    q_row_base_global + tok,
                                    h,
                                    split_idx,
                                    vec * Int32(8),
                                ),
                                mid_out.layout,
                            )
                            st_global_v4_u32(
                                get_ptr_as_int64(mid_out, mo_off), v0, v1, v2, v3
                            )

        else:
            # ── Mode 3: 1-Pass BF16 SM120 Hardware DSMEM Cluster Reduction (s in 2,4,8) ──
            cute.arch.sync_threads()
            dsm_o_addr = raw_base + Int32(_DSM_O_OFF)
            dsm_lse_in_addr = raw_base + Int32(_DSM_LSE_IN_OFF)
            dsm_w_addr = raw_base + Int32(_DSM_W_OFF)
            dsm_lse_out_addr = raw_base + Int32(_DSM_LSE_OUT_OFF)

            if cutlass.const_expr(True):
                invs = [Float32(0.0)] * 8
                for ri in cutlass.range_constexpr(8):
                    gs = global_sum[ri]
                    if gs != Float32(0.0):
                        invs[ri] = rcp_approx_ftz(gs)

                if (warp_id == Int32(0)) and (tid_in_g == Int32(0)):
                    for ri in cutlass.range_constexpr(8):
                        row_id = Int32(ri * 8) + gid
                        lse_p = neg_inf
                        if global_sum[ri] != Float32(0.0):
                            lse_p = (
                                _log2_approx_ftz_f32(global_sum[ri]) + global_max[ri]
                            )
                        st_shared_f32(dsm_lse_in_addr + row_id * Int32(4), lse_p)

                v_base = warp_id * Int32(64)
                for mt in cutlass.range_constexpr(4):
                    inv_lo = invs[mt * 2]
                    inv_hi = invs[mt * 2 + 1]
                    r_lo = Int32(mt * 16) + gid
                    r_hi = r_lo + Int32(8)
                    for tile in cutlass.range_constexpr(8):
                        col = v_base + Int32(tile * 8) + tid_in_g * Int32(2)
                        idx = mt * 32 + tile * 4
                        p0 = pack_f32x2_to_bfloat2(
                            acc_o[idx + 0] * inv_lo, acc_o[idx + 1] * inv_lo
                        )
                        p1 = pack_f32x2_to_bfloat2(
                            acc_o[idx + 2] * inv_hi, acc_o[idx + 3] * inv_hi
                        )
                        st_shared_u32(
                            dsm_o_addr + (r_lo * Int32(512) + col) * Int32(2), p0
                        )
                        st_shared_u32(
                            dsm_o_addr + (r_hi * Int32(512) + col) * Int32(2), p1
                        )

            cute.arch.sync_threads()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            rows_per_cta = cutlass.const_expr(_M_TILE // self.num_splits)
            vecs_total = cutlass.const_expr(rows_per_cta * 64)

            if tid < Int32(rows_per_cta):
                row64 = split_idx * Int32(rows_per_cta) + tid
                h_r = h_base + (row64 & Int32(15))
                lse_arr = cute.make_rmem_tensor(self.num_splits, Float32)
                m_tot = neg_inf
                for cr in cutlass.range_constexpr(self.num_splits):
                    lv = _ld_dsm_f32(dsm_lse_in_addr + row64 * Int32(4), Int32(cr))
                    lse_arr[cr] = lv
                    m_tot = fmax_f32(m_tot, lv)

                l_tot = Float32(0.0)
                for cr in cutlass.range_constexpr(self.num_splits):
                    if lse_arr[cr] > neg_inf:
                        l_tot = l_tot + _exp2_approx_ftz_f32(lse_arr[cr] - m_tot)

                inv_d = Float32(0.0)
                final_lse = neg_inf
                if l_tot != Float32(0.0):
                    final_lse = _log2_approx_ftz_f32(l_tot) + m_tot
                    if cutlass.const_expr(self.has_attn_sink):
                        sk = Float32(0.0)
                        if h_r < Int32(self.num_heads):
                            sk = Float32(attn_sink[h_r]) * Float32(LOG2_E)
                        nm = fmax_f32(m_tot, sk)
                        ps = _exp2_approx_ftz_f32(m_tot - nm)
                        ss = _exp2_approx_ftz_f32(sk - nm)
                        denom = l_tot * ps + ss
                        if denom != Float32(0.0):
                            inv_d = ps * rcp_approx_ftz(denom)
                        mm = fmax_f32(final_lse, sk)
                        final_lse = mm + _log2_approx_ftz_f32(
                            Float32(1.0)
                            + _exp2_approx_ftz_f32(fmin_f32(final_lse, sk) - mm)
                        )
                    else:
                        inv_d = rcp_approx_ftz(l_tot)
                else:
                    if cutlass.const_expr(self.has_attn_sink):
                        if h_r < Int32(self.num_heads):
                            final_lse = Float32(attn_sink[h_r]) * Float32(LOG2_E)

                for cr in cutlass.range_constexpr(self.num_splits):
                    w_cr = Float32(0.0)
                    if (lse_arr[cr] > neg_inf) and (inv_d != Float32(0.0)):
                        w_cr = _exp2_approx_ftz_f32(lse_arr[cr] - m_tot) * inv_d
                    st_shared_f32(
                        dsm_w_addr + (tid * Int32(8) + Int32(cr)) * Int32(4), w_cr
                    )
                if cutlass.const_expr(self.return_lse):
                    if cutlass.const_expr(self.lse_is_natural):
                        if final_lse != neg_inf:
                            final_lse = final_lse * Float32(_LN2)
                    st_shared_f32(dsm_lse_out_addr + tid * Int32(4), final_lse)

            cute.arch.sync_threads()

            if cutlass.const_expr(True):
                if cutlass.const_expr(self.return_lse):
                    if tid < Int32(rows_per_cta):
                        row64 = split_idx * Int32(rows_per_cta) + tid
                        tok = q_group_start + (row64 >> Int32(4))
                        h = h_base + (row64 & Int32(15))
                        if (tok < live_b) and (h < Int32(self.num_heads)):
                            out_lse[q_row_base_global + tok, h] = ld_shared_f32(
                                dsm_lse_out_addr + tid * Int32(4)
                            )

                v_idx = tid
                while v_idx < Int32(vecs_total):
                    r_off = v_idx >> Int32(6)
                    d8 = (v_idx & Int32(63)) * Int32(8)
                    row64 = split_idx * Int32(rows_per_cta) + r_off
                    tok = q_group_start + (row64 >> Int32(4))
                    h = h_base + (row64 & Int32(15))
                    if (tok < live_b) and (h < Int32(self.num_heads)):
                        o0 = Float32(0.0)
                        o1 = Float32(0.0)
                        o2 = Float32(0.0)
                        o3 = Float32(0.0)
                        o4 = Float32(0.0)
                        o5 = Float32(0.0)
                        o6 = Float32(0.0)
                        o7 = Float32(0.0)
                        elem_off = dsm_o_addr + (row64 * Int32(512) + d8) * Int32(2)
                        for cr in cutlass.range_constexpr(self.num_splits):
                            w_cr = ld_shared_f32(
                                dsm_w_addr + (r_off * Int32(8) + Int32(cr)) * Int32(4)
                            )
                            if w_cr != Float32(0.0):
                                u0, u1, u2, u3 = _ld_dsm_v4_u32(elem_off, Int32(cr))
                                f0, f1 = _bf16x2_to_f32x2(u0)
                                f2, f3 = _bf16x2_to_f32x2(u1)
                                f4, f5 = _bf16x2_to_f32x2(u2)
                                f6, f7 = _bf16x2_to_f32x2(u3)
                                o0 = o0 + w_cr * f0
                                o1 = o1 + w_cr * f1
                                o2 = o2 + w_cr * f2
                                o3 = o3 + w_cr * f3
                                o4 = o4 + w_cr * f4
                                o5 = o5 + w_cr * f5
                                o6 = o6 + w_cr * f6
                                o7 = o7 + w_cr * f7
                        out_off = cute.crd2idx(
                            (q_row_base_global + tok, h, d8), output.layout
                        )
                        store_v4_bf16x2(
                            get_ptr_as_int64(output, out_off),
                            o0,
                            o1,
                            o2,
                            o3,
                            o4,
                            o5,
                            o6,
                            o7,
                        )
                    v_idx += Int32(_MATH_THREADS)

            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


@dataclass(frozen=True)
class BitmaskVerifySplitWorkspace:
    """Caller-allocated L2-resident split-K workspace for zero-alloc CUDA Graphs."""

    mid_out: torch.Tensor  # [max_rows, num_heads, max_splits, 512] bf16
    mid_lse: torch.Tensor  # [max_rows, num_heads, max_splits] f32


def allocate_bitmask_verify_workspace(
    max_rows: int,
    num_heads: int,
    *,
    max_splits: int = _MAX_SPLITS,
    device: torch.device | str = "cuda",
) -> BitmaskVerifySplitWorkspace:
    """Pre-allocate split-K scratch for zero-allocation CUDA Graph verification."""
    if max_rows <= 0 or num_heads <= 0 or not (1 <= max_splits <= _MAX_SPLITS):
        raise ValueError(
            f"Invalid workspace dimensions: max_rows={max_rows}, "
            f"num_heads={num_heads}, max_splits={max_splits}"
        )
    mid_out = torch.empty(
        (int(max_rows), int(num_heads), int(max_splits), _D_HEAD),
        dtype=torch.bfloat16,
        device=device,
    )
    mid_lse = torch.empty(
        (int(max_rows), int(num_heads), int(max_splits)),
        dtype=torch.float32,
        device=device,
    )
    return BitmaskVerifySplitWorkspace(mid_out=mid_out, mid_lse=mid_lse)


def _to_cute_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    leading_dim: int | None = None,
    assumed_align: int = 16,
) -> cute.Tensor:
    t = from_dlpack(tensor, assumed_align=assumed_align)
    t.element_type = dtype
    if tensor.ndim == 1:
        t = t.mark_layout_dynamic(leading_dim=0)
    elif leading_dim is not None:
        t = t.mark_layout_dynamic(leading_dim=leading_dim)
    return t


@lru_cache(maxsize=256)
def _build_bitmask_verify_kernel(
    *,
    num_heads: int,
    verify_width: int,
    num_splits: int,
    use_dsmem: bool,
    has_indexed: bool,
    swa_contiguous: bool,
    has_swa_mask: bool,
    swa_mask_is_u8: bool,
    swa_per_group: bool,
    idx_mask_is_u8: bool,
    idx_per_group: bool,
    has_indexed_page_table: bool,
    has_attn_sink: bool,
    return_lse: bool,
    lse_is_natural: bool,
    has_live_counts: bool,
    swa_page_size: int,
    indexed_page_size: int,
) -> BitmaskUnionVerifyMLAKernel:
    return BitmaskUnionVerifyMLAKernel(
        num_heads=num_heads,
        verify_width=verify_width,
        num_splits=num_splits,
        use_dsmem=use_dsmem,
        has_indexed=has_indexed,
        swa_contiguous=swa_contiguous,
        has_swa_mask=has_swa_mask,
        swa_mask_is_u8=swa_mask_is_u8,
        swa_per_group=swa_per_group,
        idx_mask_is_u8=idx_mask_is_u8,
        idx_per_group=idx_per_group,
        has_indexed_page_table=has_indexed_page_table,
        has_attn_sink=has_attn_sink,
        return_lse=return_lse,
        lse_is_natural=lse_is_natural,
        has_live_counts=has_live_counts,
        swa_page_size=swa_page_size,
        indexed_page_size=indexed_page_size,
    )


@lru_cache(maxsize=128)
def _build_bitmask_split_merge_kernel(
    *,
    num_heads: int,
    verify_width: int,
    num_splits: int,
    has_attn_sink: bool,
    return_lse: bool,
    lse_is_natural: bool,
    has_live_counts: bool,
) -> BitmaskSplitMergeKernel:
    return BitmaskSplitMergeKernel(
        num_heads=num_heads,
        verify_width=verify_width,
        num_splits=num_splits,
        has_attn_sink=has_attn_sink,
        return_lse=return_lse,
        lse_is_natural=lse_is_natural,
        has_live_counts=has_live_counts,
    )


def clear_bitmask_verify_cache() -> None:
    _build_bitmask_verify_kernel.cache_clear()
    _build_bitmask_split_merge_kernel.cache_clear()


def supports_bitmask_union_verify(
    *,
    num_heads: int,
    verify_width: int,
    head_dim: int = _D_HEAD,
    device: torch.device | str | None = None,
) -> bool:
    """Return True if ``BitmaskUnionVerifyMLAKernel`` supports the configuration."""
    if int(head_dim) != _D_HEAD:
        return False
    if not (1 <= int(verify_width) <= 32):
        return False
    if int(num_heads) <= 0:
        return False
    if not torch.cuda.is_available():
        return False
    dev = torch.device(device) if device is not None else torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(dev)
    return major == 12 and minor in (0, 1)


def plan_bitmask_union_splits(
    *,
    batch_size: int,
    num_heads: int,
    verify_width: int,
    has_indexed: bool,
    sm_count: int = 170,
    split_mode: str = "auto",
    forced_num_splits: int | None = None,
    has_workspace: bool = False,
) -> tuple[int, bool]:
    """Select ``(num_splits, use_dsmem)`` to maximize SM120 occupancy cleanly."""
    if split_mode not in ("auto", "workspace", "dsmem", "direct"):
        raise ValueError(
            f"split_mode must be 'auto', 'workspace', 'dsmem', or 'direct', got {split_mode!r}"
        )
    if forced_num_splits is not None:
        s = int(forced_num_splits)
        if not (1 <= s <= _MAX_SPLITS):
            raise ValueError(f"forced_num_splits must be in 1..{_MAX_SPLITS}, got {s}")
        if s == 1:
            return 1, False
        if split_mode == "dsmem":
            if s not in (2, 4, 8):
                raise ValueError(f"dsmem requires num_splits in (2,4,8), got {s}")
            return s, True
        if split_mode == "workspace":
            return s, False
        if split_mode == "direct":
            raise ValueError("split_mode='direct' requires num_splits == 1")
        # auto
        if has_workspace:
            return s, False
        return (s, True) if s in (2, 4, 8) else (s, False)

    if split_mode == "direct":
        return 1, False

    q_groups = (int(verify_width) + _Q_TILE - 1) // _Q_TILE
    h_blocks = (int(num_heads) + _H_TILE - 1) // _H_TILE
    base_ctas = max(1, int(batch_size) * q_groups * h_blocks)

    if split_mode == "dsmem" or (split_mode == "auto" and not has_workspace):
        # DSMEM cluster path (zero workspace): best at s=2 or s=4 when under-subscribed.
        max_s = 4 if has_indexed else 2
        best = 1
        for cand in (2, 4, 8):
            if cand <= max_s and base_ctas * cand <= sm_count:
                best = cand
        return best, (best > 1)

    # Workspace L2 split-K path: ~2 us merge overhead, zero cluster restriction!
    # Target ~1 full wave (<= sm_count CTAs) capped by expected chunk count.
    max_useful = 16 if has_indexed else 8
    max_fit = max(1, sm_count // base_ctas)
    s = min(max_fit, max_useful)
    # Only pay ~2 us merge if split reduces chunk work meaningfully.
    if s == 2 and not has_indexed and base_ctas >= sm_count // 2:
        return 1, False
    return s, False


def bitmask_union_verify_mla(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    *,
    verify_width: int | None = None,
    sm_scale: float,
    swa_unique_keys: torch.Tensor | None = None,
    swa_mask_bits: torch.Tensor | None = None,
    swa_union_len: torch.Tensor | None = None,
    swa_history_lens: torch.Tensor | None = None,
    swa_contiguous: bool = False,
    swa_page_size: int = 64,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_unique_keys: torch.Tensor | None = None,
    indexed_mask_bits: torch.Tensor | None = None,
    indexed_union_len: torch.Tensor | None = None,
    indexed_page_table: torch.Tensor | None = None,
    indexed_page_size: int = 64,
    attn_sink: torch.Tensor | None = None,
    live_counts: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_lse: torch.Tensor | None = None,
    return_lse: bool = False,
    lse_scale: str = "natural",
    num_splits: int | None = None,
    split_mode: str = "auto",
    split_workspace: BitmaskVerifySplitWorkspace | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run exact multi-query bitmask-union Sparse MLA verification on SM120.

    Supports both SWA-only (528 B E4M3 + UE8M0) and Dual (SWA + 288 B NVFP4
    Indexed) DeepSeek-V4.1 caches, ``H in {16, 64}`` (general ``H >= 1``),
    ``W in 1..32``, optional ``indexed_page_table``, optional ``live_counts``,
    optional ``attn_sink``, and zero-allocation CUDA Graph capture/replay.
    """
    orig_ndim = q.ndim
    if q.ndim == 4:
        b, w_q, h, d = (int(x) for x in q.shape)
        if verify_width is not None and int(verify_width) != w_q:
            raise ValueError(
                f"verify_width={verify_width} disagrees with q.shape[1]={w_q}"
            )
        q_3d = q.reshape(b * w_q, h, d)
    elif q.ndim == 3:
        rows, h, d = (int(x) for x in q.shape)
        if verify_width is None:
            raise ValueError("verify_width is required when q is 3D [B*W, H, 512]")
        w_q = int(verify_width)
        if rows % w_q != 0:
            raise ValueError(
                f"q.shape[0]={rows} must be divisible by verify_width={w_q}"
            )
        b = rows // w_q
        q_3d = q
    else:
        raise ValueError(f"q must be 3D or 4D, got {tuple(q.shape)}")

    if d != _D_HEAD:
        raise ValueError(f"head_dim must be {_D_HEAD}, got {d}")
    if q_3d.dtype != torch.bfloat16 or not q_3d.is_cuda:
        raise TypeError("q must be a CUDA bfloat16 tensor")
    if q_3d.stride(2) != 1 or (int(q_3d.data_ptr()) % 16 != 0):
        raise ValueError("q must be 16-byte aligned with unit innermost stride")
    if lse_scale not in ("natural", "base2"):
        raise ValueError(f"lse_scale must be 'natural' or 'base2', got {lse_scale!r}")

    device = q_3d.device
    q_groups = (w_q + _Q_TILE - 1) // _Q_TILE
    rows = b * w_q

    if out is None:
        out_3d = torch.empty_like(q_3d)
        out_ret = out_3d.view_as(q) if orig_ndim == 4 else out_3d
    else:
        out_ret = out
        out_3d = out.reshape(rows, h, d) if out.ndim == 4 else out
        if out_3d.shape != q_3d.shape or out_3d.dtype != torch.bfloat16:
            raise ValueError("out shape/dtype must match q")

    if return_lse:
        if out_lse is None:
            lse_2d = torch.empty((rows, h), dtype=torch.float32, device=device)
            lse_ret = lse_2d.view(b, w_q, h) if orig_ndim == 4 else lse_2d
        else:
            lse_ret = out_lse
            lse_2d = out_lse.reshape(rows, h) if out_lse.ndim == 3 else out_lse
    else:
        lse_2d = (
            attn_sink.unsqueeze(0)
            if attn_sink is not None
            else torch.empty((1, 1), dtype=torch.float32, device=device)
        )
        lse_ret = lse_2d

    if b == 0:
        return (out_ret, lse_ret) if return_lse else out_ret

    swa_u8_raw = swa_k_cache.view(torch.uint8)
    swa_per_group = False
    if swa_contiguous:
        if swa_u8_raw.ndim == 3:
            if swa_u8_raw.shape[0] != b or swa_u8_raw.shape[2] != _SWA_RECORD_BYTES:
                raise ValueError(
                    f"contiguous swa_k_cache 3D shape must be [B, S, 528], got {tuple(swa_u8_raw.shape)}"
                )
            stride_swa_block = int(swa_u8_raw.stride(0))
            swa_max_keys = int(swa_u8_raw.shape[1])
        elif swa_u8_raw.ndim == 2:
            if swa_u8_raw.shape[0] != b or swa_u8_raw.shape[1] % _SWA_RECORD_BYTES != 0:
                raise ValueError(
                    f"contiguous swa_k_cache 2D shape must be [B, S*528], got {tuple(swa_u8_raw.shape)}"
                )
            stride_swa_block = int(swa_u8_raw.stride(0))
            swa_max_keys = int(swa_u8_raw.shape[1]) // _SWA_RECORD_BYTES
        else:
            raise ValueError("contiguous swa_k_cache must be 2D or 3D uint8")
        swa_flat = swa_u8_raw.reshape(-1)
        has_swa_mask = swa_mask_bits is not None
        if has_swa_mask:
            assert swa_mask_bits is not None and swa_union_len is not None
            swa_len_t = swa_union_len
            swa_mask_t = swa_mask_bits
            swa_max_keys = min(swa_max_keys, int(swa_mask_bits.shape[1]))
            swa_per_group = int(swa_mask_bits.shape[0]) == b * q_groups and q_groups > 1
        else:
            if swa_history_lens is None:
                raise ValueError(
                    "swa_history_lens is required when swa_contiguous=True and swa_mask_bits is None"
                )
            swa_len_t = swa_history_lens
            swa_mask_t = swa_history_lens.unsqueeze(1)
        swa_keys_t = swa_len_t.unsqueeze(1)
    else:
        if (
            swa_u8_raw.ndim != 2
            or swa_u8_raw.shape[1] != int(swa_page_size) * _SWA_RECORD_BYTES
        ):
            raise ValueError(
                f"paged swa_k_cache must be [pages, {int(swa_page_size) * _SWA_RECORD_BYTES}], got {tuple(swa_u8_raw.shape)}"
            )
        if swa_unique_keys is None or swa_mask_bits is None or swa_union_len is None:
            raise ValueError(
                "paged SWA requires swa_unique_keys, swa_mask_bits, and swa_union_len"
            )
        stride_swa_block = int(swa_u8_raw.stride(0))
        swa_flat = swa_u8_raw.reshape(-1)
        has_swa_mask = True
        swa_keys_t = swa_unique_keys
        swa_mask_t = swa_mask_bits
        swa_len_t = swa_union_len
        swa_max_keys = int(swa_unique_keys.shape[1])
        swa_per_group = int(swa_unique_keys.shape[0]) == b * q_groups and q_groups > 1

    swa_mask_is_u8 = bool(has_swa_mask and swa_mask_t.dtype == torch.uint8)
    if has_swa_mask and swa_mask_is_u8 and (not swa_per_group) and w_q > 8:
        raise ValueError(f"uint8 swa_mask_bits requires verify_width <= 8, got {w_q}")

    has_indexed = indexed_k_cache is not None
    idx_per_group = False
    if has_indexed:
        assert indexed_k_cache is not None
        if (
            indexed_unique_keys is None
            or indexed_mask_bits is None
            or indexed_union_len is None
        ):
            raise ValueError(
                "indexed_k_cache requires indexed_unique_keys, indexed_mask_bits, and indexed_union_len"
            )
        idx_u8_raw = indexed_k_cache.view(torch.uint8)
        if (
            idx_u8_raw.ndim != 2
            or idx_u8_raw.shape[1] != int(indexed_page_size) * _IDX_RECORD_BYTES
        ):
            raise ValueError(
                f"indexed_k_cache must be [pages, {int(indexed_page_size) * _IDX_RECORD_BYTES}], got {tuple(idx_u8_raw.shape)}"
            )
        stride_idx_block = int(idx_u8_raw.stride(0))
        idx_num_pages = int(idx_u8_raw.shape[0])
        idx_flat = idx_u8_raw.reshape(-1)
        idx_keys_t = indexed_unique_keys
        idx_mask_t = indexed_mask_bits
        idx_len_t = indexed_union_len
        idx_max_keys = int(indexed_unique_keys.shape[1])
        idx_per_group = (
            int(indexed_unique_keys.shape[0]) == b * q_groups and q_groups > 1
        )
        idx_mask_is_u8 = idx_mask_t.dtype == torch.uint8
        if idx_mask_is_u8 and (not idx_per_group) and w_q > 8:
            raise ValueError(
                f"uint8 indexed_mask_bits requires verify_width <= 8, got {w_q}"
            )
        has_pt = indexed_page_table is not None
        if has_pt:
            assert indexed_page_table is not None
            idx_pt_t = indexed_page_table
            idx_table_width = int(indexed_page_table.shape[1])
        else:
            idx_pt_t = idx_keys_t
            idx_table_width = 0
    else:
        idx_flat = swa_flat
        idx_keys_t = swa_keys_t
        idx_mask_t = swa_mask_t
        idx_len_t = swa_len_t
        idx_pt_t = swa_keys_t
        stride_idx_block = 0
        idx_num_pages = 0
        idx_max_keys = 0
        idx_table_width = 0
        idx_mask_is_u8 = swa_mask_is_u8
        has_pt = False

    has_attn_sink = attn_sink is not None
    sink_arg = attn_sink if has_attn_sink else lse_2d[0]
    has_live_counts = live_counts is not None
    live_arg = live_counts if has_live_counts else swa_len_t

    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    c_split, use_dsmem = plan_bitmask_union_splits(
        batch_size=b,
        num_heads=h,
        verify_width=w_q,
        has_indexed=has_indexed,
        sm_count=sm_count,
        split_mode=split_mode,
        forced_num_splits=num_splits,
        has_workspace=(split_workspace is not None),
    )

    need_workspace = (c_split > 1) and (not use_dsmem)
    if need_workspace:
        if split_workspace is None:
            ws = allocate_bitmask_verify_workspace(
                rows, h, max_splits=c_split, device=device
            )
        else:
            ws = split_workspace
            if (
                int(ws.mid_out.shape[0]) < rows
                or int(ws.mid_out.shape[1]) < h
                or int(ws.mid_out.shape[2]) < c_split
            ):
                raise ValueError(
                    f"split_workspace too small: got {tuple(ws.mid_out.shape)}, "
                    f"need >= ({rows}, {h}, {c_split}, {_D_HEAD})"
                )
        mid_out_t = ws.mid_out[:rows, :h, :c_split, :]
        mid_lse_t = ws.mid_lse[:rows, :h, :c_split]
    else:
        # Alias output/lse as 4D/3D dummy views (never touched when s==1 or use_dsmem).
        mid_out_t = out_3d.unsqueeze(2)
        mid_lse_t = lse_2d.unsqueeze(2)

    kernel = _build_bitmask_verify_kernel(
        num_heads=h,
        verify_width=w_q,
        num_splits=c_split,
        use_dsmem=use_dsmem,
        has_indexed=has_indexed,
        swa_contiguous=swa_contiguous,
        has_swa_mask=has_swa_mask,
        swa_mask_is_u8=swa_mask_is_u8,
        swa_per_group=swa_per_group,
        idx_mask_is_u8=idx_mask_is_u8,
        idx_per_group=idx_per_group,
        has_indexed_page_table=has_pt,
        has_attn_sink=has_attn_sink,
        return_lse=return_lse,
        lse_is_natural=(lse_scale == "natural"),
        has_live_counts=has_live_counts,
        swa_page_size=int(swa_page_size),
        indexed_page_size=int(indexed_page_size),
    )

    swa_mask_launch = (
        swa_mask_t.view(torch.int32) if swa_mask_t.dtype == torch.uint32 else swa_mask_t
    )
    idx_mask_launch = (
        idx_mask_t.view(torch.int32) if idx_mask_t.dtype == torch.uint32 else idx_mask_t
    )
    swa_mask_cute = cutlass.Uint8 if swa_mask_is_u8 else cutlass.Int32
    idx_mask_cute = cutlass.Uint8 if idx_mask_is_u8 else cutlass.Int32

    stream = current_cuda_stream()
    args = (
        _to_cute_tensor(q_3d, cutlass.BFloat16, leading_dim=2, assumed_align=16),
        _to_cute_tensor(swa_flat, cutlass.Uint8, assumed_align=16),
        _to_cute_tensor(swa_keys_t, cutlass.Int32, leading_dim=1, assumed_align=4),
        _to_cute_tensor(
            swa_mask_launch,
            swa_mask_cute,
            leading_dim=1,
            assumed_align=1 if swa_mask_is_u8 else 4,
        ),
        _to_cute_tensor(swa_len_t, cutlass.Int32, assumed_align=4),
        _to_cute_tensor(idx_flat, cutlass.Uint8, assumed_align=16),
        _to_cute_tensor(idx_keys_t, cutlass.Int32, leading_dim=1, assumed_align=4),
        _to_cute_tensor(
            idx_mask_launch,
            idx_mask_cute,
            leading_dim=1,
            assumed_align=1 if idx_mask_is_u8 else 4,
        ),
        _to_cute_tensor(idx_len_t, cutlass.Int32, assumed_align=4),
        _to_cute_tensor(idx_pt_t, cutlass.Int32, leading_dim=1, assumed_align=4),
        _to_cute_tensor(sink_arg, cutlass.Float32, assumed_align=4),
        _to_cute_tensor(live_arg, cutlass.Int32, assumed_align=4),
        _to_cute_tensor(out_3d, cutlass.BFloat16, leading_dim=2, assumed_align=16),
        _to_cute_tensor(lse_2d, cutlass.Float32, leading_dim=1, assumed_align=4),
        _to_cute_tensor(mid_out_t, cutlass.BFloat16, leading_dim=3, assumed_align=16),
        _to_cute_tensor(mid_lse_t, cutlass.Float32, leading_dim=2, assumed_align=4),
        Float32(float(sm_scale) * float(LOG2_E)),
        Int64(stride_swa_block),
        Int64(stride_idx_block),
        Int32(swa_max_keys),
        Int32(idx_max_keys),
        Int32(idx_table_width),
        Int64(idx_num_pages),
        Int32(b),
        stream,
    )

    cache_key = (
        h,
        w_q,
        c_split,
        int(use_dsmem),
        int(has_indexed),
        int(swa_contiguous),
        int(has_swa_mask),
        int(swa_mask_is_u8),
        int(swa_per_group),
        int(idx_mask_is_u8),
        int(idx_per_group),
        int(has_pt),
        int(has_attn_sink),
        int(return_lse),
        int(lse_scale == "natural"),
        int(has_live_counts),
        int(swa_page_size),
        int(indexed_page_size),
        tensor_compile_fact("q", q_3d, dynamic_dims=(0,), dynamic_strides=(0, 1)),
        tensor_compile_fact("out", out_3d, dynamic_dims=(0,), dynamic_strides=(0, 1)),
        tensor_compile_fact(
            "mid_out", mid_out_t, dynamic_dims=(0, 2), dynamic_strides=(0, 1, 2)
        ),
        tensor_compile_fact(
            "mid_lse", mid_lse_t, dynamic_dims=(0, 2), dynamic_strides=(0, 1)
        ),
        tensor_compile_fact(
            "swa_mask", swa_mask_launch, dynamic_dims=(0, 1), dynamic_strides=(0,)
        ),
        tensor_compile_fact(
            "idx_mask", idx_mask_launch, dynamic_dims=(0, 1), dynamic_strides=(0,)
        ),
        (DimKey.dynamic(), str(swa_flat.dtype), swa_flat.device.type),
    )
    spec = KernelCompileSpec.from_key(
        "attention.mla.bitmask_union_verify_m64",
        5,
        cache_key,
        labels=(
            "num_heads",
            "verify_width",
            "num_splits",
            "use_dsmem",
            "has_indexed",
            "swa_contiguous",
            "has_swa_mask",
            "swa_mask_is_u8",
            "swa_per_group",
            "idx_mask_is_u8",
            "idx_per_group",
            "has_indexed_page_table",
            "has_attn_sink",
            "return_lse",
            "lse_is_natural",
            "has_live_counts",
            "swa_page_size",
            "indexed_page_size",
            "q",
            "out",
            "mid_out",
            "mid_lse",
            "swa_mask",
            "idx_mask",
            "swa_flat",
        ),
    )
    b12x_launch(
        kernel,
        compile_spec=spec,
        compile_args=args,
        runtime_args=args,
    )

    if need_workspace:
        merge_kernel = _build_bitmask_split_merge_kernel(
            num_heads=h,
            verify_width=w_q,
            num_splits=c_split,
            has_attn_sink=has_attn_sink,
            return_lse=return_lse,
            lse_is_natural=(lse_scale == "natural"),
            has_live_counts=has_live_counts,
        )
        m_args = (
            _to_cute_tensor(
                mid_out_t, cutlass.BFloat16, leading_dim=3, assumed_align=16
            ),
            _to_cute_tensor(mid_lse_t, cutlass.Float32, leading_dim=2, assumed_align=4),
            _to_cute_tensor(sink_arg, cutlass.Float32, assumed_align=4),
            _to_cute_tensor(live_arg, cutlass.Int32, assumed_align=4),
            _to_cute_tensor(out_3d, cutlass.BFloat16, leading_dim=2, assumed_align=16),
            _to_cute_tensor(lse_2d, cutlass.Float32, leading_dim=1, assumed_align=4),
            Int32(rows),
            stream,
        )
        m_key = (
            h,
            w_q,
            c_split,
            int(has_attn_sink),
            int(return_lse),
            int(lse_scale == "natural"),
            int(has_live_counts),
            tensor_compile_fact(
                "mid_out", mid_out_t, dynamic_dims=(0, 2), dynamic_strides=(0, 1, 2)
            ),
            tensor_compile_fact(
                "mid_lse", mid_lse_t, dynamic_dims=(0, 2), dynamic_strides=(0, 1)
            ),
            tensor_compile_fact(
                "out", out_3d, dynamic_dims=(0,), dynamic_strides=(0, 1)
            ),
        )
        m_spec = KernelCompileSpec.from_key(
            "attention.mla.bitmask_split_merge",
            3,
            m_key,
            labels=(
                "num_heads",
                "verify_width",
                "num_splits",
                "has_attn_sink",
                "return_lse",
                "lse_is_natural",
                "has_live_counts",
                "mid_out",
                "mid_lse",
                "out",
            ),
        )
        b12x_launch(
            merge_kernel,
            compile_spec=m_spec,
            compile_args=m_args,
            runtime_args=m_args,
        )

    return (out_ret, lse_ret) if return_lse else out_ret


__all__ = [
    "BitmaskSplitMergeKernel",
    "BitmaskUnionVerifyMLAKernel",
    "BitmaskVerifySplitWorkspace",
    "allocate_bitmask_verify_workspace",
    "bitmask_union_verify_mla",
    "clear_bitmask_verify_cache",
    "plan_bitmask_union_splits",
    "supports_bitmask_union_verify",
]
