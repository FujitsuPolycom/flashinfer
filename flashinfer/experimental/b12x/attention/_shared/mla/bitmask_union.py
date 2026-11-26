"""Step A: GPU-resident sorted multi-way bitmask-union merge kernel.

Merges ``W`` pre-sorted ascending index lists per request (as emitted by
``dsa_indexer._position_sort._sort_positions``, with ``-1`` tail padding) into
a single strictly-ascending deduplicated key list plus a per-key bitmask
recording which verify tokens ``w in [0, W)`` selected each key.

Architecture:
  - For ``W <= 16`` and ``W_pow2 * K_sel <= 8192`` (covering ``W=1..8`` DSpark
    verify, ``W=12..14`` Markov Tree verify, and ``W=16`` Wide Copy-Verify at
    ``K_sel = 512``): **O(log W) Parallel 2-Way Merge-Path Tree** (512 threads /
    16 warps, zero serial frontier bottleneck, exact nudged Merge-Path diagonal
    deduplication, 41 KiB SMEM for ``W <= 8`` and 96 KiB SMEM for ``W <= 16``).
  - For ``W in 17..32`` (e.g. ``W = 24, 32``): 8-warp quantile-partitioned
    ``vote_ballot_sync`` frontier merge with 256-entry coalesced SMEM flush.

Supports:
  - ``W in 1..32``, ``uint8`` masks for ``W <= 8`` and ``uint32`` / ``int32``
    masks for ``W <= 32``.
  - Optional ``live_counts[B]`` (per-request active verify width for dead-token
    early exit) and optional ``row_lengths[B, W]`` (per-row valid prefixes).
  - Intra-row duplicates, empty lists (all ``-1``), 100% overlap, and 0%
    overlap with bit-exact parity against CPU set-union.
  - Zero runtime allocations (caller-allocated workspace) and full CUDA Graph
    capture/replay compatibility.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Boolean, Int32, Uint8, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    launch as b12x_launch,
    tensor_compile_fact,
)
from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_shared_u8_offset,
    ld_shared_u16_offset,
    shared_ptr_to_u32,
    st_shared_u8,
)
from b12x._lib.utils.device import current_cuda_stream

_TREE_THREADS = 512
_TREE_WARPS = _TREE_THREADS // 32
_FRONTIER_WARPS = 8
_FRONTIER_THREADS = _FRONTIER_WARPS * 32
_SENTINEL_U32 = 0xFFFFFFFF
_FLUSH_BUF_LEN = 256


@dsl_user_op
def _st_shared_u16(
    smem_addr: Int32, value: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u16 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _umin_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "min.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _umax_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_u32(val: Uint32, offset: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(val).ir_value(loc=loc, ip=ip),
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
def _shfl_idx_u32(val: Uint32, src_lane: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(val).ir_value(loc=loc, ip=ip),
                Int32(src_lane).ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_down_i32(val: Int32, delta: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(val).ir_value(loc=loc, ip=ip),
                Int32(delta).ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.down.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_up_i32(val: Int32, delta: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(val).ir_value(loc=loc, ip=ip),
                Int32(delta).ir_value(loc=loc, ip=ip),
            ],
            """{
                .reg .pred p;
                .reg .s32 tmp;
                shfl.sync.up.b32 tmp|p, $1, $2, 0, 0xffffffff;
                selp.s32 $0, tmp, 0, p;
            }""",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _warp_excl_scan_i32(val: Int32, lane: Int32) -> tuple[Int32, Int32]:
    """Warp exclusive prefix sum returning ``(excl_val, warp_total)``."""
    v = val
    v = v + _shfl_up_i32(v, Int32(1))
    v = v + _shfl_up_i32(v, Int32(2))
    v = v + _shfl_up_i32(v, Int32(4))
    v = v + _shfl_up_i32(v, Int32(8))
    v = v + _shfl_up_i32(v, Int32(16))
    total = cute.arch.shuffle_sync(v, Int32(31))
    excl = v - val
    return excl, total


@cute.jit
def _group_excl_scan_i32(
    val: Int32,
    tid: Int32,
    s_scan: cute.Tensor,
    *,
    group_warps: cutlass.Constexpr,
) -> tuple[Int32, Int32]:
    """Exclusive prefix sum across a contiguous group of ``group_warps`` warps."""
    lane = tid & Int32(31)
    warp_id = tid >> Int32(5)
    excl_in_warp, warp_tot = _warp_excl_scan_i32(val, lane)
    if cutlass.const_expr(group_warps == 1):
        return excl_in_warp, warp_tot
    if lane == Int32(31):
        s_scan[warp_id] = warp_tot
    cute.arch.sync_threads()
    group_id = warp_id // Int32(group_warps)
    w_in_group = warp_id - group_id * Int32(group_warps)
    g_base = group_id * Int32(group_warps)
    prefix = Int32(0)
    total = Int32(0)
    for w in cutlass.range_constexpr(group_warps):
        wt = s_scan[g_base + Int32(w)]
        if Int32(w) < w_in_group:
            prefix = prefix + wt
        total = total + wt
    return prefix + excl_in_warp, total


@cute.jit
def _warp_min_u32_broadcast(
    val: Uint32,
    *,
    active_width: cutlass.Constexpr,
) -> Uint32:
    v = val
    if cutlass.const_expr(active_width > 1):
        v = _umin_u32(v, _shfl_bfly_u32(v, Int32(1)))
    if cutlass.const_expr(active_width > 2):
        v = _umin_u32(v, _shfl_bfly_u32(v, Int32(2)))
    if cutlass.const_expr(active_width > 4):
        v = _umin_u32(v, _shfl_bfly_u32(v, Int32(4)))
    if cutlass.const_expr(active_width > 8):
        v = _umin_u32(v, _shfl_bfly_u32(v, Int32(8)))
    if cutlass.const_expr(active_width > 16):
        v = _umin_u32(v, _shfl_bfly_u32(v, Int32(16)))
    if cutlass.const_expr(active_width <= 16):
        v = _shfl_idx_u32(v, Int32(0))
    return v


@cute.jit
def _warp_max_u32_broadcast(
    val: Uint32,
    *,
    active_width: cutlass.Constexpr,
) -> Uint32:
    v = val
    if cutlass.const_expr(active_width > 1):
        v = _umax_u32(v, _shfl_bfly_u32(v, Int32(1)))
    if cutlass.const_expr(active_width > 2):
        v = _umax_u32(v, _shfl_bfly_u32(v, Int32(2)))
    if cutlass.const_expr(active_width > 4):
        v = _umax_u32(v, _shfl_bfly_u32(v, Int32(4)))
    if cutlass.const_expr(active_width > 8):
        v = _umax_u32(v, _shfl_bfly_u32(v, Int32(8)))
    if cutlass.const_expr(active_width > 16):
        v = _umax_u32(v, _shfl_bfly_u32(v, Int32(16)))
    if cutlass.const_expr(active_width <= 16):
        v = _shfl_idx_u32(v, Int32(0))
    return v


@cute.jit
def _warp_sum_i32_broadcast(
    val: Int32,
    *,
    active_width: cutlass.Constexpr,
) -> Int32:
    v = val
    if cutlass.const_expr(active_width > 1):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1)
    if cutlass.const_expr(active_width > 2):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=2)
    if cutlass.const_expr(active_width > 4):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=4)
    if cutlass.const_expr(active_width > 8):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=8)
    if cutlass.const_expr(active_width > 16):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=16)
    if cutlass.const_expr(active_width <= 16):
        v = cute.arch.shuffle_sync(v, Int32(0))
    return v


@cute.jit
def _lower_bound_smem_u32(
    row_base_addr: Int32,
    lo_init: Int32,
    hi_init: Int32,
    target: Uint32,
) -> Int32:
    lo = lo_init
    hi = hi_init
    while lo < hi:
        mid = (lo + hi) >> Int32(1)
        val = ld_shared_u32(row_base_addr + mid * Int32(4))
        if val < target:
            lo = mid + Int32(1)
        else:
            hi = mid
    return lo


@cute.jit
def _upper_bound_smem_u32(
    row_base_addr: Int32,
    lo_init: Int32,
    hi_init: Int32,
    target: Uint32,
) -> Int32:
    lo = lo_init
    hi = hi_init
    while lo < hi:
        mid = (lo + hi) >> Int32(1)
        val = ld_shared_u32(row_base_addr + mid * Int32(4))
        if val <= target:
            lo = mid + Int32(1)
        else:
            hi = mid
    return lo


@cute.jit
def _merge_path_nudged(
    a_addr: Int32,
    len_a: Int32,
    b_addr: Int32,
    len_b: Int32,
    diag: Int32,
) -> tuple[Int32, Int32]:
    """Exact 2-way Merge-Path intersection nudged so equal pairs stay in one thread.

    Given strictly increasing ``A[0..len_a)`` and ``B[0..len_b)``, returns
    ``(a_pos, b_pos)`` such that every matching pair ``A[i] == B[j]`` lies
    entirely on one side of the partition boundary.
    """
    lo = diag - len_b
    if lo < Int32(0):
        lo = Int32(0)
    hi = diag
    if hi > len_a:
        hi = len_a
    while lo < hi:
        mid = (lo + hi) >> Int32(1)
        b_idx = diag - mid - Int32(1)
        ka = ld_shared_u32(a_addr + mid * Int32(4))
        kb = ld_shared_u32(b_addr + b_idx * Int32(4))
        if ka < kb:
            lo = mid + Int32(1)
        else:
            hi = mid
    a_pos = lo
    b_pos = diag - a_pos
    if (b_pos > Int32(0)) and (a_pos < len_a):
        kb_prev = ld_shared_u32(b_addr + (b_pos - Int32(1)) * Int32(4))
        ka_cur = ld_shared_u32(a_addr + a_pos * Int32(4))
        if kb_prev == ka_cur:
            a_pos = a_pos + Int32(1)
    return a_pos, b_pos


@cute.jit
def _ld_smem_mask(addr: Int32, idx: Int32, *, mask_bytes: cutlass.Constexpr) -> Uint32:
    if cutlass.const_expr(mask_bytes == 1):
        return Uint32(ld_shared_u8_offset(addr + idx, 0))
    if cutlass.const_expr(mask_bytes == 2):
        return Uint32(ld_shared_u16_offset(addr + idx * Int32(2), 0))
    return ld_shared_u32(addr + idx * Int32(4))


@cute.jit
def _st_smem_mask(
    addr: Int32, idx: Int32, val: Uint32, *, mask_bytes: cutlass.Constexpr
) -> None:
    if cutlass.const_expr(mask_bytes == 1):
        st_shared_u8(addr + idx, (val & Uint32(0xFF)).to(Uint8))
    elif cutlass.const_expr(mask_bytes == 2):
        _st_shared_u16(addr + idx * Int32(2), val & Uint32(0xFFFF))
    else:
        st_shared_u32(addr + idx * Int32(4), val)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _next_pow2(value: int) -> int:
    p = 1
    while p < value:
        p <<= 1
    return p


class BitmaskUnionMergeKernel:
    """SM120 sorted multi-way bitmask-union merge kernel.

    Uses a 512-thread ``O(log W)`` parallel 2-way Merge-Path tree for
    ``W <= 16, W_pow2 * K_sel <= 8192`` (``<= 2.5 us`` at ``W=8, K_sel=512``),
    and an 8-warp quantile-partitioned ``vote_ballot_sync`` frontier merge for
    larger ``W in 17..32``.
    """

    def __init__(
        self,
        width: int,
        k_sel: int,
        *,
        mask_is_u8: bool,
        has_live_counts: bool = False,
        has_row_lengths: bool = False,
        vector_load: bool = True,
    ):
        if not (1 <= int(width) <= 32):
            raise ValueError(f"width must be in 1..32, got {width}")
        if int(k_sel) <= 0:
            raise ValueError(f"k_sel must be positive, got {k_sel}")
        if mask_is_u8 and int(width) > 8:
            raise ValueError(f"uint8 mask requires width <= 8, got {width}")
        self.width = int(width)
        self.k_sel = int(k_sel)
        self.mask_is_u8 = bool(mask_is_u8)
        self.has_live_counts = bool(has_live_counts)
        self.has_row_lengths = bool(has_row_lengths)
        self.vector_load = bool(vector_load and (self.k_sel % 4 == 0))

        self.w_pow2 = _next_pow2(self.width)
        self.tree_stages = self.w_pow2.bit_length() - 1  # 0..4 for W <= 16
        self.tree_cap = self.w_pow2 * self.k_sel
        self.internal_mask_bytes = 1 if self.width <= 8 else 2
        tree_smem_bytes = 2 * self.tree_cap * (4 + self.internal_mask_bytes) + 512
        self.use_tree = (
            self.width <= 16
            and self.tree_cap <= 8192
            and tree_smem_bytes <= 101376
        )

        if self.use_tree:
            self.block_threads = _TREE_THREADS
            self.list_stride = self.k_sel
            self.list_elems = self.tree_cap
        else:
            self.block_threads = _FRONTIER_THREADS
            self.list_stride = _align_up(self.k_sel + 1, 4)
            self.list_elems = self.width * self.list_stride
            two_pass_bytes = (
                self.list_elems * 4 + _FRONTIER_WARPS * _FLUSH_BUF_LEN * 8 + 512
            )
            if two_pass_bytes > 101376:
                raise ValueError(
                    f"BitmaskUnionMergeKernel SMEM {two_pass_bytes}B exceeds SM120 limit "
                    f"for width={self.width}, k_sel={self.k_sel}"
                )

    @cute.jit
    def __call__(
        self,
        selected_indices: cute.Tensor,  # (B, W, K_sel) int32
        unique_keys: cute.Tensor,  # (B, U_max) int32
        mask_bits: cute.Tensor,  # (B, U_max) uint8 or int32 (bit-exact uint32)
        union_len: cute.Tensor,  # (B,) int32
        live_counts: cute.Tensor,  # (B,) int32 (dummy if not has_live_counts)
        row_lengths: cute.Tensor,  # (B, W) int32 (dummy if not has_row_lengths)
        batch_size: Int32,
        u_max: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            selected_indices,
            unique_keys,
            mask_bits,
            union_len,
            live_counts,
            row_lengths,
            u_max,
        ).launch(
            grid=(batch_size, 1, 1),
            block=(self.block_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        selected_indices: cute.Tensor,
        unique_keys: cute.Tensor,
        mask_bits: cute.Tensor,
        union_len: cute.Tensor,
        live_counts: cute.Tensor,
        row_lengths: cute.Tensor,
        u_max: Int32,
    ):
        if cutlass.const_expr(self.use_tree):
            self._kernel_tree(
                selected_indices,
                unique_keys,
                mask_bits,
                union_len,
                live_counts,
                row_lengths,
                u_max,
            )
        else:
            self._kernel_frontier(
                selected_indices,
                unique_keys,
                mask_bits,
                union_len,
                live_counts,
                row_lengths,
                u_max,
            )

    @cute.jit
    def _kernel_tree(
        self,
        selected_indices: cute.Tensor,
        unique_keys: cute.Tensor,
        mask_bits: cute.Tensor,
        union_len: cute.Tensor,
        live_counts: cute.Tensor,
        row_lengths: cute.Tensor,
        u_max: Int32,
    ):
        b_idx = Int32(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])

        W = cutlass.const_expr(self.width)
        W_P2 = cutlass.const_expr(self.w_pow2)
        K = cutlass.const_expr(self.k_sel)
        CAP = cutlass.const_expr(self.tree_cap)
        MB = cutlass.const_expr(self.internal_mask_bytes)
        SENTINEL = Uint32(_SENTINEL_U32)

        smem = cutlass_utils.SmemAllocator()
        s_keys_a = smem.allocate_tensor(Uint32, cute.make_layout((CAP,)), 16)
        s_keys_b = smem.allocate_tensor(Uint32, cute.make_layout((CAP,)), 16)
        if cutlass.const_expr(MB == 1):
            s_mask_a = smem.allocate_tensor(Uint8, cute.make_layout((CAP,)), 16)
            s_mask_b = smem.allocate_tensor(Uint8, cute.make_layout((CAP,)), 16)
        else:
            s_mask_a = smem.allocate_tensor(Uint16, cute.make_layout((CAP,)), 16)
            s_mask_b = smem.allocate_tensor(Uint16, cute.make_layout((CAP,)), 16)
        s_seg_lens = smem.allocate_tensor(Int32, cute.make_layout((32,)), 16)
        s_scan = smem.allocate_tensor(Int32, cute.make_layout((_TREE_WARPS,)), 16)
        # Share merge-path diagonal (a0, b0) via warp shfl_down(1) plus 16 cross-warp
        # boundary entries (128 B total SMEM), so each thread runs _merge_path_nudged once.
        s_warp_diag_a = smem.allocate_tensor(Int32, cute.make_layout((_TREE_WARPS,)), 16)
        s_warp_diag_b = smem.allocate_tensor(Int32, cute.make_layout((_TREE_WARPS,)), 16)
        s_needs_dedup = smem.allocate_tensor(Int32, cute.make_layout((4,)), 16)

        ka_addr = shared_ptr_to_u32(s_keys_a.iterator)
        kb_addr = shared_ptr_to_u32(s_keys_b.iterator)
        ma_addr = shared_ptr_to_u32(s_mask_a.iterator)
        mb_addr = shared_ptr_to_u32(s_mask_b.iterator)

        if tid == Int32(0):
            s_needs_dedup[Int32(0)] = Int32(0)

        live_w = Int32(W)
        if cutlass.const_expr(self.has_live_counts):
            live_w = Int32(live_counts[b_idx])
            if live_w < Int32(0):
                live_w = Int32(0)
            if live_w > Int32(W):
                live_w = Int32(W)

        # ── Step 0a: Vectorized load directly into s_keys_a + s_mask_a, tracking fast-path ──
        bad_local = Boolean(False)
        if cutlass.const_expr(self.vector_load):
            vecs_per_row = K // 4
            total_vecs = W * vecs_per_row
            v_idx = tid
            while v_idx < Int32(total_vecs):
                w = v_idx // Int32(vecs_per_row)
                v = v_idx - w * Int32(vecs_per_row)
                k0 = v * Int32(4)
                x0 = SENTINEL
                x1 = SENTINEL
                x2 = SENTINEL
                x3 = SENTINEL
                if w < live_w:
                    r_lim = Int32(K)
                    if cutlass.const_expr(self.has_row_lengths):
                        r_lim = Int32(row_lengths[b_idx, w])
                        if r_lim < Int32(0):
                            r_lim = Int32(0)
                        if r_lim > Int32(K):
                            r_lim = Int32(K)
                    if k0 < r_lim:
                        off = cute.crd2idx((b_idx, w, k0), selected_indices.layout)
                        r0, r1, r2, r3 = ld_global_nc_v4_u32(
                            get_ptr_as_int64(selected_indices, off)
                        )
                        if (r0.to(Int32) >= Int32(0)) and (k0 < r_lim):
                            x0 = r0
                        if (r1.to(Int32) >= Int32(0)) and (k0 + Int32(1) < r_lim):
                            x1 = r1
                        if (r2.to(Int32) >= Int32(0)) and (k0 + Int32(2) < r_lim):
                            x2 = r2
                        if (r3.to(Int32) >= Int32(0)) and (k0 + Int32(3) < r_lim):
                            x3 = r3
                # Detect intra-4-vec duplicates or interior sentinels!
                if (x1 != SENTINEL) and (x1 <= x0):
                    bad_local = Boolean(True)
                if (x2 != SENTINEL) and (x2 <= x1):
                    bad_local = Boolean(True)
                if (x3 != SENTINEL) and (x3 <= x2):
                    bad_local = Boolean(True)
                base_e = w * Int32(K) + k0
                dst = ka_addr + base_e * Int32(4)
                st_shared_u32(dst, x0)
                st_shared_u32(dst + Int32(4), x1)
                st_shared_u32(dst + Int32(8), x2)
                st_shared_u32(dst + Int32(12), x3)
                row_bit = Uint32(1) << w.to(Uint32)
                if cutlass.const_expr(MB == 1):
                    packed_m = (
                        row_bit
                        | (row_bit << Uint32(8))
                        | (row_bit << Uint32(16))
                        | (row_bit << Uint32(24))
                    )
                    st_shared_u32(ma_addr + base_e, packed_m)
                else:
                    packed_m2 = row_bit | (row_bit << Uint32(16))
                    st_shared_u32(ma_addr + base_e * Int32(2), packed_m2)
                    st_shared_u32(ma_addr + (base_e + Int32(2)) * Int32(2), packed_m2)
                v_idx += Int32(_TREE_THREADS)
        else:
            total_elems = W * K
            e_idx = tid
            while e_idx < Int32(total_elems):
                w = e_idx // Int32(K)
                k = e_idx - w * Int32(K)
                val_u32 = SENTINEL
                if w < live_w:
                    r_lim = Int32(K)
                    if cutlass.const_expr(self.has_row_lengths):
                        r_lim = Int32(row_lengths[b_idx, w])
                        if r_lim < Int32(0):
                            r_lim = Int32(0)
                        if r_lim > Int32(K):
                            r_lim = Int32(K)
                    if k < r_lim:
                        raw = Int32(selected_indices[b_idx, w, k])
                        if raw >= Int32(0):
                            val_u32 = Uint32(raw)
                st_shared_u32(ka_addr + (w * Int32(K) + k) * Int32(4), val_u32)
                _st_smem_mask(
                    ma_addr + w * Int32(K * MB),
                    k,
                    Uint32(1) << w.to(Uint32),
                    mask_bytes=MB,
                )
                e_idx += Int32(_TREE_THREADS)
            bad_local = Boolean(True)

        cute.arch.sync_threads()

        # Check cross-vector boundaries (only 1 load per 4-vec!) + find dense lengths in 10 cycles!
        if cutlass.const_expr(self.vector_load):
            vecs_per_row = K // 4
            total_vecs = W * vecs_per_row
            v_idx = tid
            while v_idx < Int32(total_vecs):
                w = v_idx // Int32(vecs_per_row)
                v = v_idx - w * Int32(vecs_per_row)
                if v > Int32(0):
                    k0 = v * Int32(4)
                    rb = ka_addr + w * Int32(K * 4)
                    cur0 = ld_shared_u32(rb + k0 * Int32(4))
                    if cur0 != SENTINEL:
                        prv = ld_shared_u32(rb + (k0 - Int32(1)) * Int32(4))
                        if cur0 <= prv:
                            bad_local = Boolean(True)
                v_idx += Int32(_TREE_THREADS)

        if cute.arch.vote_any_sync(bad_local):
            if (tid & Int32(31)) == Int32(0):
                s_needs_dedup[Int32(0)] = Int32(1)

        # Speculatively compute exact dense row lengths via 9-step binary search!
        if tid < Int32(W_P2):
            if tid < Int32(W):
                s_seg_lens[tid] = _lower_bound_smem_u32(
                    ka_addr + tid * Int32(K * 4), Int32(0), Int32(K), SENTINEL
                )
            else:
                s_seg_lens[tid] = Int32(0)

        cute.arch.sync_threads()

        # ── Step 0b: Slow-path intra-row dedup ONLY when duplicates/gaps exist ──
        if s_needs_dedup[Int32(0)] != Int32(0):
            # Copy s_keys_a -> s_keys_b, then compact deduplicated keys back into s_keys_a.
            c_idx = tid
            while c_idx < Int32(W * K):
                st_shared_u32(
                    kb_addr + c_idx * Int32(4), ld_shared_u32(ka_addr + c_idx * Int32(4))
                )
                c_idx += Int32(_TREE_THREADS)
            cute.arch.sync_threads()

            threads_per_row = cutlass.const_expr(_TREE_THREADS // W_P2)
            warps_per_row = cutlass.const_expr(threads_per_row // 32)
            row_step = cutlass.const_expr((K + threads_per_row - 1) // threads_per_row)

            my_row = tid // Int32(threads_per_row)
            lt_row = tid - my_row * Int32(threads_per_row)
            k_start = lt_row * Int32(row_step)
            k_end = k_start + Int32(row_step)
            if k_end > Int32(K):
                k_end = Int32(K)

            local_cnt = Int32(0)
            if my_row < Int32(W):
                row_in_base = kb_addr + my_row * Int32(K * 4)
                ki = k_start
                while ki < k_end:
                    v = ld_shared_u32(row_in_base + ki * Int32(4))
                    if v != SENTINEL:
                        keep = Boolean(True)
                        if ki > Int32(0):
                            prev = ld_shared_u32(
                                row_in_base + (ki - Int32(1)) * Int32(4)
                            )
                            keep = prev != v
                        if keep:
                            local_cnt = local_cnt + Int32(1)
                    ki = ki + Int32(1)

            my_off, row_tot = _group_excl_scan_i32(
                local_cnt, tid, s_scan, group_warps=warps_per_row
            )
            if my_row < Int32(W):
                row_in_base = kb_addr + my_row * Int32(K * 4)
                row_out_k = ka_addr + my_row * Int32(K * 4)
                row_out_m = ma_addr + my_row * Int32(K * MB)
                row_bit = Uint32(1) << my_row.to(Uint32)
                out_p = my_off
                ki = k_start
                while ki < k_end:
                    v = ld_shared_u32(row_in_base + ki * Int32(4))
                    if v != SENTINEL:
                        keep = Boolean(True)
                        if ki > Int32(0):
                            prev = ld_shared_u32(
                                row_in_base + (ki - Int32(1)) * Int32(4)
                            )
                            keep = prev != v
                        if keep:
                            st_shared_u32(row_out_k + out_p * Int32(4), v)
                            _st_smem_mask(row_out_m, out_p, row_bit, mask_bytes=MB)
                            out_p = out_p + Int32(1)
                    ki = ki + Int32(1)
                if lt_row == Int32(0):
                    s_seg_lens[my_row] = row_tot
            else:
                if lt_row == Int32(0):
                    s_seg_lens[my_row] = Int32(0)

            cute.arch.sync_threads()

        # ── Step 1..S: log2(W_pow2) parallel 2-way Merge-Path stages ──
        for stage in cutlass.range_constexpr(self.tree_stages):
            seg_cap = K << stage
            num_pairs = W_P2 >> (stage + 1)
            threads_per_pair = _TREE_THREADS // num_pairs
            warps_per_pair = threads_per_pair // 32
            is_last_stage = stage == self.tree_stages - 1
            # Keep register-file spill-free: runtime-indexed rmem arrays spill to local memory.
            use_reg_buf = False
            max_step = 0
            lens_src_off = Int32(0) if cutlass.const_expr((stage & 1) == 0) else Int32(16)
            lens_dst_off = Int32(16) if cutlass.const_expr((stage & 1) == 0) else Int32(0)

            if cutlass.const_expr((stage & 1) == 0):
                src_k_base = ka_addr
                src_m_base = ma_addr
                dst_k_base = kb_addr
                dst_m_base = mb_addr
            else:
                src_k_base = kb_addr
                src_m_base = mb_addr
                dst_k_base = ka_addr
                dst_m_base = ma_addr

            pair_id = tid // Int32(threads_per_pair)
            lt = tid - pair_id * Int32(threads_per_pair)
            len_a = s_seg_lens[lens_src_off + pair_id * Int32(2)]
            len_b = s_seg_lens[lens_src_off + pair_id * Int32(2) + Int32(1)]
            tot = len_a + len_b

            a_k_addr = src_k_base + (pair_id * Int32(2)) * Int32(seg_cap * 4)
            b_k_addr = src_k_base + (pair_id * Int32(2) + Int32(1)) * Int32(seg_cap * 4)
            a_m_addr = src_m_base + (pair_id * Int32(2)) * Int32(seg_cap * MB)
            b_m_addr = src_m_base + (pair_id * Int32(2) + Int32(1)) * Int32(seg_cap * MB)

            step = (tot + Int32(threads_per_pair - 1)) // Int32(threads_per_pair)
            diag0 = lt * step
            if diag0 > tot:
                diag0 = tot

            a0, b0 = _merge_path_nudged(a_k_addr, len_a, b_k_addr, len_b, diag0)
            a1_next = _shfl_down_i32(a0, Int32(1))
            b1_next = _shfl_down_i32(b0, Int32(1))
            lane_t = tid & Int32(31)
            warp_t = tid >> Int32(5)
            if cutlass.const_expr(warps_per_pair > 1):
                if lane_t == Int32(0):
                    s_warp_diag_a[warp_t] = a0
                    s_warp_diag_b[warp_t] = b0
                cute.arch.sync_threads()
            a1 = len_a
            b1 = len_b
            if lt + Int32(1) < Int32(threads_per_pair):
                if lane_t < Int32(31):
                    a1 = a1_next
                    b1 = b1_next
                else:
                    if cutlass.const_expr(warps_per_pair > 1):
                        a1 = s_warp_diag_a[warp_t + Int32(1)]
                        b1 = s_warp_diag_b[warp_t + Int32(1)]

            if cutlass.const_expr(use_reg_buf):
                # Single-pass register-buffered merge (eliminates Pass 2 SMEM re-reads & comparisons!).
                buf_cap = cutlass.const_expr(max_step + 1)
                rk = cute.make_rmem_tensor(buf_cap, Uint32)
                rm = cute.make_rmem_tensor(buf_cap, Uint32)
                ia = a0
                ib = b0
                cnt = Int32(0)
                while (ia < a1) and (ib < b1):
                    ka = ld_shared_u32(a_k_addr + ia * Int32(4))
                    kb = ld_shared_u32(b_k_addr + ib * Int32(4))
                    out_k = ka
                    out_m = Uint32(0)
                    if ka < kb:
                        out_m = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                        ia = ia + Int32(1)
                    elif kb < ka:
                        out_k = kb
                        out_m = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                        ib = ib + Int32(1)
                    else:
                        out_m = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB) | _ld_smem_mask(
                            b_m_addr, ib, mask_bytes=MB
                        )
                        ia = ia + Int32(1)
                        ib = ib + Int32(1)
                    rk[cnt] = out_k
                    rm[cnt] = out_m
                    cnt = cnt + Int32(1)
                while ia < a1:
                    rk[cnt] = ld_shared_u32(a_k_addr + ia * Int32(4))
                    rm[cnt] = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                    ia = ia + Int32(1)
                    cnt = cnt + Int32(1)
                while ib < b1:
                    rk[cnt] = ld_shared_u32(b_k_addr + ib * Int32(4))
                    rm[cnt] = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                    ib = ib + Int32(1)
                    cnt = cnt + Int32(1)

                out_off, pair_tot = _group_excl_scan_i32(
                    cnt, tid, s_scan, group_warps=warps_per_pair
                )

                if cutlass.const_expr(is_last_stage):
                    clamped_tot = pair_tot if pair_tot <= u_max else u_max
                    if tid == Int32(0):
                        union_len[b_idx] = clamped_tot
                    i_emit = Int32(0)
                    while i_emit < cnt:
                        op = out_off + i_emit
                        if op < u_max:
                            unique_keys[b_idx, op] = rk[i_emit].to(Int32)
                            mv = rm[i_emit]
                            if cutlass.const_expr(self.mask_is_u8):
                                mask_bits[b_idx, op] = (mv & Uint32(0xFF)).to(Uint8)
                            else:
                                mask_bits[b_idx, op] = mv.to(mask_bits.element_type)
                        i_emit = i_emit + Int32(1)
                else:
                    if lt == Int32(0):
                        s_seg_lens[lens_dst_off + pair_id] = pair_tot
                    dst_k = dst_k_base + pair_id * Int32(seg_cap * 2 * 4)
                    dst_m = dst_m_base + pair_id * Int32(seg_cap * 2 * MB)
                    i_emit = Int32(0)
                    while i_emit < cnt:
                        op = out_off + i_emit
                        st_shared_u32(dst_k + op * Int32(4), rk[i_emit])
                        _st_smem_mask(dst_m, op, rm[i_emit], mask_bytes=MB)
                        i_emit = i_emit + Int32(1)
                    cute.arch.sync_threads()
            else:
                # Two-pass merge for larger W (max_step > 8, e.g. W=12..16).
                ia = a0
                ib = b0
                cnt = Int32(0)
                while (ia < a1) and (ib < b1):
                    ka = ld_shared_u32(a_k_addr + ia * Int32(4))
                    kb = ld_shared_u32(b_k_addr + ib * Int32(4))
                    if ka < kb:
                        ia = ia + Int32(1)
                    elif kb < ka:
                        ib = ib + Int32(1)
                    else:
                        ia = ia + Int32(1)
                        ib = ib + Int32(1)
                    cnt = cnt + Int32(1)
                cnt = cnt + (a1 - ia) + (b1 - ib)

                out_off, pair_tot = _group_excl_scan_i32(
                    cnt, tid, s_scan, group_warps=warps_per_pair
                )

                if cutlass.const_expr(is_last_stage):
                    clamped_tot = pair_tot if pair_tot <= u_max else u_max
                    if tid == Int32(0):
                        union_len[b_idx] = clamped_tot
                    ia = a0
                    ib = b0
                    op = out_off
                    while (ia < a1) and (ib < b1):
                        ka = ld_shared_u32(a_k_addr + ia * Int32(4))
                        kb = ld_shared_u32(b_k_addr + ib * Int32(4))
                        out_k = ka
                        out_m = Uint32(0)
                        if ka < kb:
                            out_m = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                            ia = ia + Int32(1)
                        elif kb < ka:
                            out_k = kb
                            out_m = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                            ib = ib + Int32(1)
                        else:
                            out_m = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB) | _ld_smem_mask(
                                b_m_addr, ib, mask_bytes=MB
                            )
                            ia = ia + Int32(1)
                            ib = ib + Int32(1)
                        if op < u_max:
                            unique_keys[b_idx, op] = out_k.to(Int32)
                            if cutlass.const_expr(self.mask_is_u8):
                                mask_bits[b_idx, op] = (out_m & Uint32(0xFF)).to(Uint8)
                            else:
                                mask_bits[b_idx, op] = out_m.to(mask_bits.element_type)
                        op = op + Int32(1)
                    while ia < a1:
                        if op < u_max:
                            ka = ld_shared_u32(a_k_addr + ia * Int32(4))
                            ma = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                            unique_keys[b_idx, op] = ka.to(Int32)
                            if cutlass.const_expr(self.mask_is_u8):
                                mask_bits[b_idx, op] = (ma & Uint32(0xFF)).to(Uint8)
                            else:
                                mask_bits[b_idx, op] = ma.to(mask_bits.element_type)
                        ia = ia + Int32(1)
                        op = op + Int32(1)
                    while ib < b1:
                        if op < u_max:
                            kb = ld_shared_u32(b_k_addr + ib * Int32(4))
                            mb = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                            unique_keys[b_idx, op] = kb.to(Int32)
                            if cutlass.const_expr(self.mask_is_u8):
                                mask_bits[b_idx, op] = (mb & Uint32(0xFF)).to(Uint8)
                            else:
                                mask_bits[b_idx, op] = mb.to(mask_bits.element_type)
                        ib = ib + Int32(1)
                        op = op + Int32(1)
                else:
                    if lt == Int32(0):
                        s_seg_lens[lens_dst_off + pair_id] = pair_tot
                    dst_k = dst_k_base + pair_id * Int32(seg_cap * 2 * 4)
                    dst_m = dst_m_base + pair_id * Int32(seg_cap * 2 * MB)
                    ia = a0
                    ib = b0
                    op = out_off
                    while (ia < a1) and (ib < b1):
                        ka = ld_shared_u32(a_k_addr + ia * Int32(4))
                        kb = ld_shared_u32(b_k_addr + ib * Int32(4))
                        if ka < kb:
                            ma = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                            st_shared_u32(dst_k + op * Int32(4), ka)
                            _st_smem_mask(dst_m, op, ma, mask_bytes=MB)
                            ia = ia + Int32(1)
                        elif kb < ka:
                            mb = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                            st_shared_u32(dst_k + op * Int32(4), kb)
                            _st_smem_mask(dst_m, op, mb, mask_bytes=MB)
                            ib = ib + Int32(1)
                        else:
                            ma = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                            mb = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                            st_shared_u32(dst_k + op * Int32(4), ka)
                            _st_smem_mask(dst_m, op, ma | mb, mask_bytes=MB)
                            ia = ia + Int32(1)
                            ib = ib + Int32(1)
                        op = op + Int32(1)
                    while ia < a1:
                        ka = ld_shared_u32(a_k_addr + ia * Int32(4))
                        ma = _ld_smem_mask(a_m_addr, ia, mask_bytes=MB)
                        st_shared_u32(dst_k + op * Int32(4), ka)
                        _st_smem_mask(dst_m, op, ma, mask_bytes=MB)
                        ia = ia + Int32(1)
                        op = op + Int32(1)
                    while ib < b1:
                        kb = ld_shared_u32(b_k_addr + ib * Int32(4))
                        mb = _ld_smem_mask(b_m_addr, ib, mask_bytes=MB)
                        st_shared_u32(dst_k + op * Int32(4), kb)
                        _st_smem_mask(dst_m, op, mb, mask_bytes=MB)
                        ib = ib + Int32(1)
                        op = op + Int32(1)
                    cute.arch.sync_threads()

        # W == 1 (tree_stages == 0): flush directly from s_keys_a / s_mask_a.
        if cutlass.const_expr(self.tree_stages == 0):
            total_u = s_seg_lens[Int32(0)]
            clamped_u = total_u if total_u <= u_max else u_max
            if tid == Int32(0):
                union_len[b_idx] = clamped_u
            idx = tid
            while idx < clamped_u:
                unique_keys[b_idx, idx] = ld_shared_u32(ka_addr + idx * Int32(4)).to(
                    Int32
                )
                mv = _ld_smem_mask(ma_addr, idx, mask_bytes=MB)
                if cutlass.const_expr(self.mask_is_u8):
                    mask_bits[b_idx, idx] = (mv & Uint32(0xFF)).to(Uint8)
                else:
                    mask_bits[b_idx, idx] = mv.to(mask_bits.element_type)
                idx = idx + Int32(_TREE_THREADS)

    @cute.jit
    def _kernel_frontier(
        self,
        selected_indices: cute.Tensor,
        unique_keys: cute.Tensor,
        mask_bits: cute.Tensor,
        union_len: cute.Tensor,
        live_counts: cute.Tensor,
        row_lengths: cute.Tensor,
        u_max: Int32,
    ):
        """8-warp quantile-partitioned ballot-frontier fallback for W in 17..32."""
        b_idx = Int32(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        warp_id = tid >> Int32(5)
        lane = tid & Int32(31)

        W = cutlass.const_expr(self.width)
        K = cutlass.const_expr(self.k_sel)
        STRIDE = cutlass.const_expr(self.list_stride)
        SENTINEL = Uint32(_SENTINEL_U32)

        smem = cutlass_utils.SmemAllocator()
        s_lists = smem.allocate_tensor(
            Uint32, cute.make_layout((self.list_elems,)), 16
        )
        s_splits = smem.allocate_tensor(
            Uint32, cute.make_layout((_FRONTIER_WARPS + 4,)), 16
        )
        s_warp_counts = smem.allocate_tensor(
            Int32, cute.make_layout((_FRONTIER_WARPS,)), 16
        )
        s_out_keys = smem.allocate_tensor(
            Uint32, cute.make_layout((_FRONTIER_WARPS * _FLUSH_BUF_LEN,)), 16
        )
        s_out_mask = smem.allocate_tensor(
            Uint32, cute.make_layout((_FRONTIER_WARPS * _FLUSH_BUF_LEN,)), 16
        )

        lists_addr = shared_ptr_to_u32(s_lists.iterator)
        out_keys_addr = shared_ptr_to_u32(s_out_keys.iterator)
        out_mask_addr = shared_ptr_to_u32(s_out_mask.iterator)

        live_w = Int32(W)
        if cutlass.const_expr(self.has_live_counts):
            live_w = Int32(live_counts[b_idx])
            if live_w < Int32(0):
                live_w = Int32(0)
            if live_w > Int32(W):
                live_w = Int32(W)

        if cutlass.const_expr(self.vector_load):
            vecs_per_row = K // 4
            total_vecs = W * vecs_per_row
            v_idx = tid
            while v_idx < Int32(total_vecs):
                w = v_idx // Int32(vecs_per_row)
                v = v_idx - w * Int32(vecs_per_row)
                k0 = v * Int32(4)
                x0 = SENTINEL
                x1 = SENTINEL
                x2 = SENTINEL
                x3 = SENTINEL
                if w < live_w:
                    r_lim = Int32(K)
                    if cutlass.const_expr(self.has_row_lengths):
                        r_lim = Int32(row_lengths[b_idx, w])
                        if r_lim < Int32(0):
                            r_lim = Int32(0)
                        if r_lim > Int32(K):
                            r_lim = Int32(K)
                    if k0 < r_lim:
                        off = cute.crd2idx((b_idx, w, k0), selected_indices.layout)
                        r0, r1, r2, r3 = ld_global_nc_v4_u32(
                            get_ptr_as_int64(selected_indices, off)
                        )
                        if (r0.to(Int32) >= Int32(0)) and (k0 < r_lim):
                            x0 = r0
                        if (r1.to(Int32) >= Int32(0)) and (k0 + Int32(1) < r_lim):
                            x1 = r1
                        if (r2.to(Int32) >= Int32(0)) and (k0 + Int32(2) < r_lim):
                            x2 = r2
                        if (r3.to(Int32) >= Int32(0)) and (k0 + Int32(3) < r_lim):
                            x3 = r3
                dst = lists_addr + (w * Int32(STRIDE) + k0) * Int32(4)
                st_shared_u32(dst, x0)
                st_shared_u32(dst + Int32(4), x1)
                st_shared_u32(dst + Int32(8), x2)
                st_shared_u32(dst + Int32(12), x3)
                v_idx += Int32(_FRONTIER_THREADS)
        else:
            total_elems = W * K
            e_idx = tid
            while e_idx < Int32(total_elems):
                w = e_idx // Int32(K)
                k = e_idx - w * Int32(K)
                val_u32 = SENTINEL
                if w < live_w:
                    r_lim = Int32(K)
                    if cutlass.const_expr(self.has_row_lengths):
                        r_lim = Int32(row_lengths[b_idx, w])
                        if r_lim < Int32(0):
                            r_lim = Int32(0)
                        if r_lim > Int32(K):
                            r_lim = Int32(K)
                    if k < r_lim:
                        raw = Int32(selected_indices[b_idx, w, k])
                        if raw >= Int32(0):
                            val_u32 = Uint32(raw)
                st_shared_u32(
                    lists_addr + (w * Int32(STRIDE) + k) * Int32(4), val_u32
                )
                e_idx += Int32(_FRONTIER_THREADS)

        if tid < Int32(W):
            st_shared_u32(
                lists_addr + (tid * Int32(STRIDE) + Int32(K)) * Int32(4), SENTINEL
            )
        cute.arch.sync_threads()

        row_base_addr = lists_addr + lane * Int32(STRIDE * 4)
        row_len = Int32(0)
        if lane < Int32(W):
            row_len = _lower_bound_smem_u32(
                row_base_addr, Int32(0), Int32(K), SENTINEL
            )

        total_valid = _warp_sum_i32_broadcast(row_len, active_width=W)
        if total_valid == Int32(0):
            if tid == Int32(0):
                union_len[b_idx] = Int32(0)
        else:
            r_min = SENTINEL
            r_max = Uint32(0)
            if (lane < Int32(W)) and (row_len > Int32(0)):
                r_min = ld_shared_u32(row_base_addr)
                r_max = ld_shared_u32(row_base_addr + (row_len - Int32(1)) * Int32(4))
            min_v = _warp_min_u32_broadcast(r_min, active_width=W)
            max_v = _warp_max_u32_broadcast(r_max, active_width=W)

            if warp_id >= Int32(1):
                target_rank = (warp_id * total_valid) >> Int32(3)
                v_lo = min_v
                v_hi = max_v
                b_lo = Int32(0)
                b_hi = row_len
                while v_lo < v_hi:
                    v_mid = v_lo + ((v_hi - v_lo) >> Uint32(1))
                    ub = Int32(0)
                    if lane < Int32(W):
                        ub = _upper_bound_smem_u32(row_base_addr, b_lo, b_hi, v_mid)
                    cnt = _warp_sum_i32_broadcast(ub, active_width=W)
                    if cnt <= target_rank:
                        v_lo = v_mid + Uint32(1)
                        b_lo = ub
                    else:
                        v_hi = v_mid
                        b_hi = ub
                if lane == Int32(0):
                    s_splits[warp_id] = v_lo
            else:
                if lane == Int32(0):
                    s_splits[Int32(0)] = min_v
                    s_splits[Int32(_FRONTIER_WARPS)] = SENTINEL
            cute.arch.sync_threads()

            my_v_lo = s_splits[warp_id]
            my_v_hi = s_splits[warp_id + Int32(1)]

            start_ptr = Int32(0)
            start_cur = SENTINEL
            if lane < Int32(W):
                start_ptr = _lower_bound_smem_u32(
                    row_base_addr, Int32(0), row_len, my_v_lo
                )
                start_cur = ld_shared_u32(row_base_addr + start_ptr * Int32(4))

            # Pass 1: count
            ptr = start_ptr
            cur = start_cur
            out_cnt = Int32(0)
            if my_v_lo < my_v_hi:
                u = _warp_min_u32_broadcast(cur, active_width=W)
                while u < my_v_hi:
                    hit = (lane < Int32(W)) and (cur == u)
                    out_cnt = out_cnt + Int32(1)
                    if hit:
                        ptr = ptr + Int32(1)
                        cur = ld_shared_u32(row_base_addr + ptr * Int32(4))
                        while cur == u:
                            ptr = ptr + Int32(1)
                            cur = ld_shared_u32(row_base_addr + ptr * Int32(4))
                    u = _warp_min_u32_broadcast(cur, active_width=W)

            if lane == Int32(0):
                s_warp_counts[warp_id] = out_cnt
            cute.arch.sync_threads()

            w_off = Int32(0)
            total_u = Int32(0)
            for wi in cutlass.range_constexpr(_FRONTIER_WARPS):
                cw = s_warp_counts[Int32(wi)]
                if Int32(wi) < warp_id:
                    w_off = w_off + cw
                total_u = total_u + cw
            clamped_u = total_u if total_u <= u_max else u_max
            if tid == Int32(0):
                union_len[b_idx] = clamped_u

            # Pass 2: stream into 256-entry warp SMEM buffer & flush coalesced.
            buf_keys_base = out_keys_addr + warp_id * Int32(_FLUSH_BUF_LEN * 4)
            buf_mask_base = out_mask_addr + warp_id * Int32(_FLUSH_BUF_LEN * 4)
            ptr = start_ptr
            cur = start_cur
            buf_n = Int32(0)
            g_base = w_off
            if my_v_lo < my_v_hi:
                u = _warp_min_u32_broadcast(cur, active_width=W)
                while u < my_v_hi:
                    hit = (lane < Int32(W)) and (cur == u)
                    mask = cute.arch.vote_ballot_sync(Boolean(hit))
                    if lane == Int32(0):
                        st_shared_u32(buf_keys_base + buf_n * Int32(4), u)
                        st_shared_u32(buf_mask_base + buf_n * Int32(4), mask)
                    buf_n = buf_n + Int32(1)
                    if buf_n == Int32(_FLUSH_BUF_LEN):
                        cute.arch.sync_warp()
                        fi = lane
                        while fi < Int32(_FLUSH_BUF_LEN):
                            dst_idx = g_base + fi
                            if dst_idx < u_max:
                                unique_keys[b_idx, dst_idx] = ld_shared_u32(
                                    buf_keys_base + fi * Int32(4)
                                ).to(Int32)
                                mask_bits[b_idx, dst_idx] = ld_shared_u32(
                                    buf_mask_base + fi * Int32(4)
                                ).to(mask_bits.element_type)
                            fi += Int32(32)
                        cute.arch.sync_warp()
                        g_base = g_base + Int32(_FLUSH_BUF_LEN)
                        buf_n = Int32(0)
                    if hit:
                        ptr = ptr + Int32(1)
                        cur = ld_shared_u32(row_base_addr + ptr * Int32(4))
                        while cur == u:
                            ptr = ptr + Int32(1)
                            cur = ld_shared_u32(row_base_addr + ptr * Int32(4))
                    u = _warp_min_u32_broadcast(cur, active_width=W)

            if buf_n > Int32(0):
                cute.arch.sync_warp()
                fi = lane
                while fi < buf_n:
                    dst_idx = g_base + fi
                    if dst_idx < u_max:
                        unique_keys[b_idx, dst_idx] = ld_shared_u32(
                            buf_keys_base + fi * Int32(4)
                        ).to(Int32)
                        mask_bits[b_idx, dst_idx] = ld_shared_u32(
                            buf_mask_base + fi * Int32(4)
                        ).to(mask_bits.element_type)
                    fi += Int32(32)


@lru_cache(maxsize=None)
def _build_bitmask_union_kernel(
    width: int,
    k_sel: int,
    mask_is_u8: bool,
    has_live_counts: bool,
    has_row_lengths: bool,
    vector_load: bool,
) -> BitmaskUnionMergeKernel:
    return BitmaskUnionMergeKernel(
        width=width,
        k_sel=k_sel,
        mask_is_u8=mask_is_u8,
        has_live_counts=has_live_counts,
        has_row_lengths=has_row_lengths,
        vector_load=vector_load,
    )


def clear_bitmask_union_cache() -> None:
    _build_bitmask_union_kernel.cache_clear()


def _to_cute_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    leading_dim: int | None = None,
    assumed_align: int = 4,
) -> cute.Tensor:
    ct = from_dlpack(tensor, assumed_align=assumed_align)
    ct.element_type = dtype
    if leading_dim is not None and tensor.ndim >= 2:
        ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def supports_bitmask_union_merge(
    *,
    width: int,
    k_sel: int,
    device: torch.device | None = None,
) -> bool:
    """Return True when ``BitmaskUnionMergeKernel`` supports this geometry."""
    if device is not None:
        if device.type != "cuda":
            return False
        major, _ = torch.cuda.get_device_capability(device)
        if major < 12:
            return False
    if not (1 <= int(width) <= 32) or int(k_sel) <= 0:
        return False
    w_p2 = _next_pow2(int(width))
    if int(width) <= 16 and w_p2 * int(k_sel) <= 8192:
        return True
    stride = _align_up(int(k_sel) + 1, 4)
    lists_bytes = int(width) * stride * 4
    two_pass_bytes = lists_bytes + _FRONTIER_WARPS * _FLUSH_BUF_LEN * 8 + 512
    return two_pass_bytes <= 101376


def allocate_bitmask_union_workspace(
    batch_size: int,
    width: int,
    k_sel: int,
    *,
    device: torch.device,
    u_max: int | None = None,
    mask_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate caller-owned zero-runtime-alloc output buffers for Step A."""
    b = int(batch_size)
    w = int(width)
    cap = int(u_max) if u_max is not None else w * int(k_sel)
    if mask_dtype is None:
        mask_dtype = torch.uint8 if w <= 8 else torch.uint32
    unique_keys = torch.empty((b, cap), dtype=torch.int32, device=device)
    mask_bits = torch.empty((b, cap), dtype=mask_dtype, device=device)
    union_len = torch.empty((b,), dtype=torch.int32, device=device)
    return unique_keys, mask_bits, union_len


def bitmask_union_merge(
    selected_indices: torch.Tensor,
    *,
    unique_keys: torch.Tensor | None = None,
    mask_bits: torch.Tensor | None = None,
    union_len: torch.Tensor | None = None,
    live_counts: torch.Tensor | None = None,
    row_lengths: torch.Tensor | None = None,
    mask_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge ``[B, W, K_sel]`` sorted ascending int32 lists into unique keys + bitmask.

    When ``unique_keys``, ``mask_bits``, and ``union_len`` are all supplied by
    the caller, performs zero runtime GPU allocations (CUDA-Graph safe).
    """
    if selected_indices.ndim != 3:
        raise ValueError(
            f"selected_indices must have shape [B, W, K_sel], got {tuple(selected_indices.shape)}"
        )
    if selected_indices.dtype != torch.int32:
        raise TypeError(
            f"selected_indices must be int32, got {selected_indices.dtype}"
        )
    if not selected_indices.is_cuda:
        raise ValueError("selected_indices must be a CUDA tensor")
    if selected_indices.stride(2) != 1:
        raise ValueError("selected_indices last dimension must be contiguous (stride 1)")

    b, w, k_sel = (int(d) for d in selected_indices.shape)
    if not (1 <= w <= 32):
        raise ValueError(f"width W must be in 1..32, got {w}")
    if k_sel <= 0:
        raise ValueError(f"k_sel must be positive, got {k_sel}")

    if unique_keys is None or mask_bits is None or union_len is None:
        alloc_keys, alloc_mask, alloc_len = allocate_bitmask_union_workspace(
            b, w, k_sel, device=selected_indices.device, mask_dtype=mask_dtype
        )
        if unique_keys is None:
            unique_keys = alloc_keys
        if mask_bits is None:
            mask_bits = alloc_mask
        if union_len is None:
            union_len = alloc_len

    if unique_keys.ndim != 2 or unique_keys.shape[0] != b or unique_keys.dtype != torch.int32:
        raise ValueError(
            f"unique_keys must be int32 [B, U_max] with B={b}, got {unique_keys.dtype} {tuple(unique_keys.shape)}"
        )
    if mask_bits.ndim != 2 or mask_bits.shape != unique_keys.shape:
        raise ValueError(
            f"mask_bits must match unique_keys shape {tuple(unique_keys.shape)}, got {tuple(mask_bits.shape)}"
        )
    if union_len.shape != (b,) or union_len.dtype != torch.int32:
        raise ValueError(
            f"union_len must be int32 [{b}], got {union_len.dtype} {tuple(union_len.shape)}"
        )
    if unique_keys.stride(1) != 1 or mask_bits.stride(1) != 1 or not union_len.is_contiguous():
        raise ValueError("output tensors must be contiguous along their innermost dimension")

    if mask_bits.dtype == torch.uint8:
        if w > 8:
            raise ValueError(f"uint8 mask_bits requires W <= 8, got W={w}")
        mask_is_u8 = True
    elif mask_bits.dtype in (torch.uint32, torch.int32):
        mask_is_u8 = False
    else:
        raise TypeError(
            f"mask_bits must be uint8, uint32, or int32, got {mask_bits.dtype}"
        )

    has_live_counts = live_counts is not None
    if has_live_counts:
        assert live_counts is not None
        if live_counts.shape != (b,) or live_counts.dtype != torch.int32:
            raise ValueError(
                f"live_counts must be int32 [{b}], got {live_counts.dtype} {tuple(live_counts.shape)}"
            )
        live_arg = live_counts
    else:
        live_arg = union_len

    has_row_lengths = row_lengths is not None
    if has_row_lengths:
        assert row_lengths is not None
        if row_lengths.shape != (b, w) or row_lengths.dtype != torch.int32:
            raise ValueError(
                f"row_lengths must be int32 [{b}, {w}], got {row_lengths.dtype} {tuple(row_lengths.shape)}"
            )
        rlen_arg = row_lengths
    else:
        rlen_arg = unique_keys

    if b == 0:
        return unique_keys, mask_bits, union_len

    vector_load = bool(
        (k_sel % 4 == 0)
        and (int(selected_indices.data_ptr()) % 16 == 0)
        and ((int(selected_indices.stride(0)) * 4) % 16 == 0)
        and ((int(selected_indices.stride(1)) * 4) % 16 == 0)
    )

    kernel = _build_bitmask_union_kernel(
        w,
        k_sel,
        mask_is_u8,
        has_live_counts,
        has_row_lengths,
        vector_load,
    )

    u_max = int(unique_keys.shape[1])
    mask_launch_tensor = (
        mask_bits.view(torch.int32)
        if mask_bits.dtype == torch.uint32
        else mask_bits
    )
    mask_cute_dtype = cutlass.Uint8 if mask_is_u8 else cutlass.Int32

    args = (
        _to_cute_tensor(
            selected_indices,
            cutlass.Int32,
            leading_dim=2,
            assumed_align=16 if vector_load else 4,
        ),
        _to_cute_tensor(unique_keys, cutlass.Int32, leading_dim=1, assumed_align=4),
        _to_cute_tensor(
            mask_launch_tensor,
            mask_cute_dtype,
            leading_dim=1,
            assumed_align=1 if mask_is_u8 else 4,
        ),
        _to_cute_tensor(union_len, cutlass.Int32, assumed_align=4),
        _to_cute_tensor(live_arg, cutlass.Int32, assumed_align=4),
        _to_cute_tensor(rlen_arg, cutlass.Int32, leading_dim=1, assumed_align=4),
        Int32(b),
        Int32(u_max),
        current_cuda_stream(),
    )

    cache_key = (
        w,
        k_sel,
        int(mask_is_u8),
        int(has_live_counts),
        int(has_row_lengths),
        int(vector_load),
        tensor_compile_fact(
            "selected_indices",
            selected_indices,
            dynamic_dims=(0,),
            dynamic_strides=(0, 1),
        ),
        tensor_compile_fact(
            "unique_keys",
            unique_keys,
            dynamic_dims=(0, 1),
            dynamic_strides=(0,),
        ),
        tensor_compile_fact(
            "mask_bits",
            mask_launch_tensor,
            dynamic_dims=(0, 1),
            dynamic_strides=(0,),
        ),
        (DimKey.dynamic(), str(union_len.dtype), union_len.device.type),
    )
    spec = KernelCompileSpec.from_key(
        "attention.mla.bitmask_union_merge",
        2,
        cache_key,
        labels=(
            "width",
            "k_sel",
            "mask_is_u8",
            "has_live_counts",
            "has_row_lengths",
            "vector_load",
            "selected_indices",
            "unique_keys",
            "mask_bits",
            "union_len",
        ),
    )
    b12x_launch(
        kernel,
        compile_spec=spec,
        compile_args=args,
        runtime_args=args,
    )
    return unique_keys, mask_bits, union_len


def bitmask_union_merge_reference(
    selected_indices: torch.Tensor,
    *,
    live_counts: torch.Tensor | None = None,
    row_lengths: torch.Tensor | None = None,
    u_max: int | None = None,
    mask_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact CPU reference for ``bitmask_union_merge``."""
    sel_cpu = selected_indices.detach().cpu().tolist()
    b, w, k_sel = selected_indices.shape
    cap = int(u_max) if u_max is not None else w * k_sel
    if mask_dtype is None:
        mask_dtype = torch.uint8 if w <= 8 else torch.uint32
    live_cpu = (
        live_counts.detach().cpu().tolist() if live_counts is not None else [w] * b
    )
    rlen_cpu = (
        row_lengths.detach().cpu().tolist()
        if row_lengths is not None
        else [[k_sel] * w for _ in range(b)]
    )

    out_keys = torch.zeros((b, cap), dtype=torch.int32)
    out_mask = torch.zeros((b, cap), dtype=torch.int64)
    out_len = torch.zeros((b,), dtype=torch.int32)

    for bi in range(b):
        lw = max(0, min(w, int(live_cpu[bi])))
        key_to_mask: dict[int, int] = {}
        for wi in range(lw):
            rl = max(0, min(k_sel, int(rlen_cpu[bi][wi])))
            bit = 1 << wi
            for ki in range(rl):
                val = int(sel_cpu[bi][wi][ki])
                if val >= 0:
                    key_to_mask[val] = key_to_mask.get(val, 0) | bit
        sorted_items = sorted(key_to_mask.items())
        n = min(len(sorted_items), cap)
        out_len[bi] = n
        for idx in range(n):
            k_val, m_val = sorted_items[idx]
            out_keys[bi, idx] = k_val
            out_mask[bi, idx] = m_val

    if mask_dtype == torch.uint8:
        mask_t = out_mask.to(torch.uint8)
    elif mask_dtype == torch.int32:
        mask_t = out_mask.to(torch.int64).to(torch.uint32).view(torch.int32)
    else:
        mask_t = out_mask.to(torch.uint32)
    return (
        out_keys.to(selected_indices.device),
        mask_t.to(selected_indices.device),
        out_len.to(selected_indices.device),
    )
