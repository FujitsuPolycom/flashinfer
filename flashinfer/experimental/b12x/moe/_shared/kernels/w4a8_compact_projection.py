"""Independent compact W4A8 FC1 gate/up projection kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x.moe._shared.kernels.routing_ids import local_expert_id

from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async4_shared_global_l2hint,
    cp_async_u32_shared_global,
    cp_async_u32_shared_global_l2hint,
    create_l2_evict_first_policy,
    get_ptr_as_int64,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_u8,
    st_shared_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.constants import FLOAT8_E4M3_MAX
from b12x._lib.intrinsics.conversion import (
    cvt_f32x4_to_e4m3x4,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.mma import (
    e2m1x8_to_qmma_e2m1x8,
    mxfp8_mma_m16n8k32_f32_e2m1_e4m3,
)
from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.packed import pack_f32x2_to_bfloat2
from b12x.moe._shared.kernels.w4a8_compact_activation import (
    W4A8CompactMicroActivationKernel,
)


@cute.jit
def _stage_compact_input_k128_swizzled(
    source: cute.Tensor,
    a_row_base: Int32,
    sfa_row_base: Int32,
    lane: Int32,
    token: Int32,
    col_idx: Int32,
    k128_slice: Int32,
    k: cutlass.Constexpr,
):
    """Quantize one token's K128 slice into an XOR-swizzled 128-byte SMEM row."""
    values = cute.make_rmem_tensor((4,), cutlass.Float32)
    offset = (
        Int64(token) * Int64(k)
        + Int64(k128_slice) * Int64(128)
        + Int64(lane * Int32(4))
    )
    for elem in cutlass.range_constexpr(4):
        values[elem] = cutlass.Float32(source[offset + Int64(elem)])
    max_abs = fabs_f32(values[0])
    for elem in cutlass.range_constexpr(1, 4):
        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
    for shift in cutlass.range_constexpr(3):
        max_abs = fmax_f32(
            max_abs,
            cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
        )
    _, scale_byte = pow2_ceil_ue8m0(max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX))
    inv_scale = ue8m0_to_output_scale(scale_byte)
    payload = cvt_f32x4_to_e4m3x4(
        values[0] * inv_scale,
        values[1] * inv_scale,
        values[2] * inv_scale,
        values[3] * inv_scale,
    )
    vec = (lane >> Int32(2)) ^ col_idx
    word_in_vec = lane & Int32(3)
    st_shared_u32(
        a_row_base + (vec << Int32(4)) + (word_in_vec << Int32(2)),
        payload,
    )
    if lane % Int32(8) == Int32(0):
        st_shared_u8(sfa_row_base + (lane >> Int32(3)), cutlass.Uint8(scale_byte))


@cute.jit
def _stage_compact_input_k128(
    source: cute.Tensor,
    a_base: Int32,
    sfa_base: Int32,
    tid: Int32,
    token: Int32,
    k128_slice: Int32,
    k: cutlass.Constexpr,
):
    """Quantize a contiguous K128 slice with eight lanes per K32 block."""
    if tid < Int32(32):
        values = cute.make_rmem_tensor((4,), cutlass.Float32)
        offset = (
            Int64(token) * Int64(k)
            + Int64(k128_slice) * Int64(128)
            + Int64(tid * Int32(4))
        )
        for elem in cutlass.range_constexpr(4):
            values[elem] = cutlass.Float32(source[offset + Int64(elem)])
        max_abs = fabs_f32(values[0])
        for elem in cutlass.range_constexpr(1, 4):
            max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
        for shift in cutlass.range_constexpr(3):
            max_abs = fmax_f32(
                max_abs,
                cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
            )
        _, scale_byte = pow2_ceil_ue8m0(
            max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
        )
        inv_scale = ue8m0_to_output_scale(scale_byte)
        payload = cvt_f32x4_to_e4m3x4(
            values[0] * inv_scale,
            values[1] * inv_scale,
            values[2] * inv_scale,
            values[3] * inv_scale,
        )
        st_shared_u32(a_base + tid * Int32(4), payload)
        if tid % Int32(8) == Int32(0):
            st_shared_u8(sfa_base + tid // Int32(8), cutlass.Uint8(scale_byte))


@dsl_user_op
def _mma_fp4_weights_fp8_activation(
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    block: int,
    *,
    loc=None,
    ip=None,
):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(value).ir_value(loc=loc, ip=ip)
            for value in (a0, a1, a2, a3, b0, b1, sfa, sfb)
        ],
        f"""
        {{
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0
            {{$0, $1, $2, $3}},
            {{$4, $5, $6, $7}},
            {{$8, $9}},
            {{zero, zero, zero, zero}},
            {{$10}}, {{{int(block)}, 0}},
            {{$11}}, {{{int(block)}, 0}};
        }}
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


class W4A8CompactMicroProjectionKernel:
    """Project compact output tiles with expert weights in the QMMA row operand."""

    tile_m = 16
    tile_n = 128
    tile_k = 128
    stages = 2
    num_warps = 4
    threads_per_cta = 128

    def __init__(
        self,
        k: int,
        n: int,
        num_topk: int,
        *,
        tile_n: int = 128,
        csf_inline: bool = False,
        max_copy_groups: int = 2,
        pipeline_stages: int = 2,
    ):
        if k <= 0 or n < 64 or k % 128 or n % 64:
            raise ValueError(
                "compact W4A8 projection requires K divisible by 128 and "
                "N divisible by 64"
            )
        if num_topk <= 0:
            raise ValueError("num_topk must be positive")
        if type(tile_n) is not int or tile_n not in (64, 128):
            raise ValueError("compact FC1 tile N must be 64 or 128")
        if tile_n == 64 and n % 128 != 64:
            raise ValueError("compact FC1 N64 tiles require the native N64-tail layout")
        if type(max_copy_groups) is not int or max_copy_groups not in (1, 2):
            raise ValueError("compact FC1 permits one or two pending copy groups")
        if type(pipeline_stages) is not int or pipeline_stages not in (1, 2):
            raise ValueError("compact FC1 pipeline stages must be one or two")
        if pipeline_stages == 1 and max_copy_groups != 1:
            raise ValueError("one compact FC1 stage requires one pending copy group")
        self.max_copy_groups = max_copy_groups
        self.stages = pipeline_stages
        self.k = int(k)
        self.n = int(n)
        self.num_topk = int(num_topk)
        self.csf_inline = bool(csf_inline)
        self.n64_tail = self.n % 128 == 64
        self.tile_n = tile_n
        self.num_warps = self.tile_n // 32
        self.threads_per_cta = self.num_warps * 32
        self.n_fragments = self.tile_n // (self.num_warps * 16)
        self.fuse_input_quantization = False
        self.share_projection_input = False
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.b_payload_bytes = self.tile_n * self.tile_k // 2
        self.sfb_bytes = 512 if self.csf_inline else self.tile_n * 4
        self.a_offset = 0
        self.sfa_offset = 128
        self.b_offset = 256
        self.sfb_offset = self.b_offset + self.b_payload_bytes
        self.stage_bytes = self.sfb_offset + self.sfb_bytes
        self.shared_words = self.stages * self.stage_bytes // 4

    @cute.jit
    def __call__(
        self,
        a8: cute.Tensor,
        scale_rows: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        a_source = cute.recast_tensor(a8, cutlass.Uint32)
        projected_flat = cute.make_tensor(
            projections.iterator,
            cute.make_layout((projections.shape[0] * projections.shape[1],)),
        )
        grid = num_pairs * Int32(2 * self.n_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(grid, max_active_clusters * Int32(2))
        self.kernel(
            a_source,
            scale_rows,
            w13_rp,
            w13_sfb,
            projected_flat,
            topk_ids,
            alpha,
            input_scale,
            num_pairs,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_b_k128(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        tid: Int32,
        l2_policy=None,
    ):
        # Prepared RP is [K32, N32-chunk, lane, n8-in-chunk].  A CTA owns
        # one N128 half, retaining coalesced four-word lane loads.
        for i in cutlass.range_constexpr(4 * self.tile_n // self.threads_per_cta):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(512):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc % Int32(4)
                kb = kc // Int32(4)
                src_word = (
                    tile_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64((packed_half * Int32(4) + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                if cutlass.const_expr(l2_policy is not None):
                    cp_async4_shared_global_l2hint(
                        dst_base + (idx << Int32(4)),
                        get_ptr_as_int64(weights, src_word),
                        l2_policy,
                    )
                else:
                    cp_async4_shared_global(
                        dst_base + (idx << Int32(4)),
                        get_ptr_as_int64(weights, src_word),
                    )

    @cute.jit
    def _stage_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        tid: Int32,
        l2_policy=None,
    ):
        if tid < Int32(32):
            src_word = tile_word_base + Int64(packed_half * Int32(128) + tid * 4)
            if cutlass.const_expr(l2_policy is not None):
                cp_async4_shared_global_l2hint(
                    dst_base + (tid << Int32(4)),
                    get_ptr_as_int64(scales, src_word),
                    l2_policy,
                )
            else:
                cp_async4_shared_global(
                    dst_base + (tid << Int32(4)),
                    get_ptr_as_int64(scales, src_word),
                )

    @cute.jit
    def _stage_n64_b_k128(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
        l2_policy=None,
    ):
        physical_tile = output_tile // Int32(128 // self.tile_n)
        packed_half = output_tile % Int32(128 // self.tile_n)
        n_rows = Int32(128)
        if physical_tile == Int32(self.n // 128):
            n_rows = Int32(64)
        for i in cutlass.range_constexpr(4 * self.tile_n // self.threads_per_cta):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(4 * (self.tile_n // 32) * 32):
                lane = idx & Int32(31)
                tmp = idx >> Int32(5)
                n32 = tmp % Int32(self.tile_n // 32)
                kb = tmp // Int32(self.tile_n // 32)
                copy_words = Int32(1)
                if cutlass.const_expr(self.tile_n == 128):
                    copy_words = (n32 < n_rows // Int32(32)).to(Int32)
                dst_addr = dst_base + (idx << Int32(4))
                if copy_words != Int32(0):
                    source_word = (
                        Int64(expert_idx) * Int64(self.n * (self.k // 32) * 8)
                        + Int64(projection) * Int64(self.n * (self.k // 32) * 4)
                        + Int64(physical_tile) * Int64(128 * (self.k // 32) * 4)
                        + Int64(k128_slice * Int32(4) + kb)
                        * Int64(n_rows // Int32(32))
                        * Int64(128)
                        + Int64(
                            (packed_half * Int32(self.tile_n // 32) + n32) * Int32(128)
                            + lane * Int32(4)
                        )
                    )
                    if cutlass.const_expr(l2_policy is not None):
                        cp_async4_shared_global_l2hint(
                            dst_addr,
                            get_ptr_as_int64(weights, source_word),
                            l2_policy,
                        )
                    else:
                        cp_async4_shared_global(
                            dst_addr, get_ptr_as_int64(weights, source_word)
                        )
                else:
                    st_shared_u32(dst_addr, Uint32(0))
                    st_shared_u32(dst_addr + Int32(4), Uint32(0))
                    st_shared_u32(dst_addr + Int32(8), Uint32(0))
                    st_shared_u32(dst_addr + Int32(12), Uint32(0))

    @cute.jit
    def _stage_n64_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
        bounds=None,
        value_offset=None,
        base_word=None,
        l2_policy=None,
    ):
        physical_tile = output_tile // Int32(128 // self.tile_n)
        packed_half = output_tile % Int32(128 // self.tile_n)
        n_rows = Int32(128)
        if physical_tile == Int32(self.n // 128):
            n_rows = Int32(64)
        if cutlass.const_expr(self.csf_inline):
            if tid < Int32(32):
                target = cute.make_ptr(
                    cutlass.Uint8, dst_base, cute.AddressSpace.smem, assumed_align=16
                )
                group = (
                    projection * Int32(self.csf_reader.plane.storage.groups)
                    + physical_tile * Int32(self.csf_reader.plane.storage.column_groups)
                    + k128_slice
                )
                cached = None
                if cutlass.const_expr(bounds is not None):
                    first, end = Uint32(0), Uint32(0)
                    for chunk in cutlass.range_constexpr(cute.size(bounds, mode=[0])):
                        selected_first = cute.arch.shuffle_sync(
                            bounds[chunk, 0], k128_slice % Int32(32)
                        )
                        selected_end = cute.arch.shuffle_sync(
                            bounds[chunk, 1], k128_slice % Int32(32)
                        )
                        if k128_slice // Int32(32) == Int32(chunk):
                            first, end = selected_first, selected_end
                    cached = (first, end)
                self.csf_reader.stage_compressed(
                    scales.iterator,
                    expert_idx,
                    group,
                    target,
                    tid,
                    cached,
                    value_offset,
                    base_word,
                    coordinates=(projection, physical_tile, k128_slice),
                )
            return
        if tid < Int32(self.tile_n // 4):
            dst_addr = dst_base + (tid << Int32(4))
            copy_words = Int32(1)
            if cutlass.const_expr(self.tile_n == 128):
                copy_words = (tid < n_rows // Int32(4)).to(Int32)
            if copy_words != Int32(0):
                source_word = (
                    Int64(expert_idx) * Int64(self.n * (self.k // 32) // 2)
                    + Int64(projection) * Int64(self.n * (self.k // 32) // 4)
                    + Int64(physical_tile) * Int64(128 * (self.k // 32) // 4)
                    + Int64(
                        k128_slice * n_rows
                        + packed_half * Int32(self.tile_n)
                        + tid * Int32(4)
                    )
                )
                if cutlass.const_expr(l2_policy is not None):
                    cp_async4_shared_global_l2hint(
                        dst_addr,
                        get_ptr_as_int64(scales, source_word),
                        l2_policy,
                    )
                else:
                    cp_async4_shared_global(
                        dst_addr, get_ptr_as_int64(scales, source_word)
                    )
            else:
                st_shared_u32(dst_addr, Uint32(0))
                st_shared_u32(dst_addr + Int32(4), Uint32(0))
                st_shared_u32(dst_addr + Int32(8), Uint32(0))
                st_shared_u32(dst_addr + Int32(12), Uint32(0))

    @cute.jit
    def _stage_slice(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pair: Int32,
        expert: Int32,
        projection_tile: Int32,
        k128_slice: Int32,
        csf_bounds=None,
        csf_value_offset=None,
        csf_base_word=None,
        l2_policy=None,
    ):
        stage_base = smem_base
        if cutlass.const_expr(self.stages == 2):
            stage_base += (k128_slice & Int32(1)) * Int32(self.stage_bytes)
        token = pair // Int32(self.num_topk)
        if cutlass.const_expr(not self.fuse_input_quantization):
            copy_input = Int32(1)
            if cutlass.const_expr(self.share_projection_input):
                copy_input = (projection_tile < Int32(self.n_tiles)).to(Int32)
            if copy_input != Int32(0):
                if tid < Int32(8):
                    src_word = (
                        Int64(token) * Int64(self.k // 4)
                        + Int64(k128_slice) * Int64(32)
                        + Int64(tid * 4)
                    )
                    cp_async4_shared_global(
                        stage_base + Int32(self.a_offset) + tid * Int32(16),
                        get_ptr_as_int64(a_source, src_word),
                    )
                if tid == Int32(0):
                    sf_src = Int64(token) * Int64(self.k // 32) + Int64(
                        k128_slice
                    ) * Int64(4)
                    cp_async_u32_shared_global(
                        stage_base + Int32(self.sfa_offset),
                        get_ptr_as_int64(scale_words, sf_src),
                    )
        b_base = stage_base + Int32(self.b_offset)
        sfb_base = stage_base + Int32(self.sfb_offset)
        if cutlass.const_expr(self.n64_tail):
            projection = projection_tile // Int32(self.n_tiles)
            output_tile = projection_tile % Int32(self.n_tiles)
            self._stage_n64_b_k128(
                w13_rp,
                b_base,
                expert,
                projection,
                output_tile,
                k128_slice,
                tid,
                l2_policy,
            )
            self._stage_n64_sfb(
                w13_sfb,
                sfb_base,
                expert,
                projection,
                output_tile,
                k128_slice,
                tid,
                csf_bounds,
                csf_value_offset,
                csf_base_word,
                l2_policy,
            )
        else:
            packed_tile = projection_tile // Int32(2)
            packed_half = projection_tile % Int32(2)
            packed_tile_index = (
                Int64(expert) * Int64(self.n // 128) + Int64(packed_tile)
            ) * Int64(self.k // 128) + Int64(k128_slice)
            tile_word_base = packed_tile_index * Int64(4096)
            self._stage_b_k128(
                w13_rp,
                b_base,
                tile_word_base,
                packed_half,
                tid,
                l2_policy,
            )
            scale_word_base = packed_tile_index * Int64(256)
            self._stage_sfb(
                w13_sfb,
                sfb_base,
                scale_word_base,
                packed_half,
                tid,
                l2_policy,
            )

        if cutlass.const_expr(self.fuse_input_quantization):
            if projection_tile < Int32(self.n_tiles):
                # Weight copies are in flight before the shared input is encoded.
                _stage_compact_input_k128(
                    a_source,
                    stage_base + Int32(self.a_offset),
                    stage_base + Int32(self.sfa_offset),
                    tid,
                    token,
                    k128_slice,
                    self.k,
                )

    @cute.jit
    def _run_task(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp: Int32,
        pair: Int32,
        expert: Int32,
        projection_tile: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        input_smem_base = smem_base
        if cutlass.const_expr(self.share_projection_input):
            # Paired projection groups consume group0's encoded input ring.
            input_smem_base -= (projection_tile // Int32(self.n_tiles)) * Int32(
                self.shared_words * 4
            )
        csf_bounds, csf_base_word, csf_value_offset = None, None, None
        if cutlass.const_expr(self.csf_inline):
            # Each task keeps one validated expert and projection row group
            # throughout its K sweep; another persistent-grid task reloads them.
            projection = projection_tile // Int32(self.n_tiles)
            output_tile = projection_tile % Int32(self.n_tiles)
            physical_tile = output_tile // Int32(128 // self.tile_n)
            section_expert = expert * Int32(self.csf_reader.plane.sections) + projection
            csf_value_offset = self.csf_reader.plane.group_value_offset(
                w13_sfb.iterator
            )
            csf_bounds = self.csf_reader.prefetch_column_bounds(
                w13_sfb.iterator, section_expert, physical_tile
            )
            csf_base_word = self.csf_reader.prefetch_row_bases(
                w13_sfb.iterator, section_expert, physical_tile, tid
            )
        l2_policy = create_l2_evict_first_policy()
        for preload in cutlass.range_constexpr(1):
            if cutlass.const_expr(preload < self.k // 128):
                self._stage_slice(
                    a_source,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    smem_base,
                    tid,
                    pair,
                    expert,
                    projection_tile,
                    Int32(preload),
                    csf_bounds,
                    csf_value_offset,
                    csf_base_word,
                    l2_policy,
                )
            cute.arch.cp_async_commit_group()

        acc = tuple(
            cute.make_rmem_tensor((4,), cutlass.Float32)
            for _ in range(self.n_fragments)
        )
        for nt in cutlass.range_constexpr(self.n_fragments):
            acc[nt].fill(0.0)
        k128_slice = Int32(0)
        k128_tiles = Int32(self.k // 128)
        b_scale_words = cute.make_rmem_tensor((self.n_fragments,), Uint32)
        b_scale_words.fill(Uint32(0))
        while k128_slice < k128_tiles:
            stage_base = smem_base
            input_stage_base = input_smem_base
            if cutlass.const_expr(self.stages == 2):
                stage_offset = (k128_slice & Int32(1)) * Int32(self.stage_bytes)
                stage_base += stage_offset
                input_stage_base += stage_offset
            b_base = stage_base + Int32(self.b_offset)
            sfb_base = stage_base + Int32(self.sfb_offset)
            if cutlass.const_expr(self.max_copy_groups == 1):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            next_slice = k128_slice + Int32(1)
            if cutlass.const_expr(self.stages == 2):
                if next_slice < k128_tiles:
                    self._stage_slice(
                        a_source,
                        scale_words,
                        w13_rp,
                        w13_sfb,
                        smem_base,
                        tid,
                        pair,
                        expert,
                        projection_tile,
                        next_slice,
                        csf_bounds,
                        csf_value_offset,
                        csf_base_word,
                        l2_policy,
                    )
                cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.max_copy_groups == 2):
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
            if cutlass.const_expr(self.csf_inline):
                self.csf_reader.expand_slots(
                    w13_sfb.iterator, sfb_base, tid, self.threads_per_cta, 1
                )
                cute.arch.sync_threads()
                if cutlass.const_expr(self.tile_n == 64):
                    sfb_base += (
                        projection_tile % Int32(self.n_tiles) & Int32(1)
                    ) * Int32(256)
            for nt in cutlass.range_constexpr(self.n_fragments):
                n8 = (
                    warp * Int32(self.n_fragments * 2)
                    + Int32(nt * 2)
                    + (lane & Int32(1))
                )
                b_scale_words[nt] = ld_shared_u32(
                    sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                )
            asc = ld_shared_u32(input_stage_base + Int32(self.sfa_offset))
            for kb in cutlass.range_constexpr(4):
                # One activation is repeated across the eight hardware columns.
                a_src = (
                    input_stage_base
                    + Int32(self.a_offset)
                    + Int32(kb * 32)
                    + c * Int32(8)
                )
                a0, a2 = ld_shared_v2_u32(a_src)
                b_words = cute.make_rmem_tensor((self.n_fragments * 2,), Uint32)
                for chunk in cutlass.range_constexpr(self.n_fragments // 2):
                    source_chunk = (
                        Int32(kb * (self.tile_n // 32))
                        + warp * Int32(self.n_fragments // 2)
                        + Int32(chunk)
                    )
                    bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                        b_base + ((source_chunk * Int32(32) + lane) << Int32(4))
                    )
                    b_words[chunk * 4] = bw0
                    b_words[chunk * 4 + 1] = bw1
                    b_words[chunk * 4 + 2] = bw2
                    b_words[chunk * 4 + 3] = bw3
                for nt in cutlass.range_constexpr(self.n_fragments):
                    w0, w2 = e2m1x8_to_qmma_e2m1x8(b_words[nt * 2])
                    w1, w3 = e2m1x8_to_qmma_e2m1x8(b_words[nt * 2 + 1])
                    sfb = b_scale_words[nt]
                    fragment = acc[nt]
                    x0, x1, x2, x3 = _mma_fp4_weights_fp8_activation(
                        w0,
                        w1,
                        w2,
                        w3,
                        a0,
                        a2,
                        sfb,
                        asc,
                        block=kb,
                    )
                    fragment[0] = fragment[0] + x0
                    fragment[1] = fragment[1] + x1
                    fragment[2] = fragment[2] + x2
                    fragment[3] = fragment[3] + x3
            if cutlass.const_expr(self.max_copy_groups == 2 or self.stages == 1):
                cute.arch.sync_threads()
            if cutlass.const_expr(self.stages == 1):
                if next_slice < k128_tiles:
                    self._stage_slice(
                        a_source,
                        scale_words,
                        w13_rp,
                        w13_sfb,
                        smem_base,
                        tid,
                        pair,
                        expert,
                        projection_tile,
                        next_slice,
                        csf_bounds,
                        csf_value_offset,
                        csf_base_word,
                        l2_policy,
                    )
                cute.arch.cp_async_commit_group()
            k128_slice += Int32(1)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        scale_expert = expert
        if cutlass.const_expr(input_scale.shape[0] == 1):
            scale_expert = Int32(0)
        value_scale = alpha[expert].to(cutlass.Float32) * input_scale[
            Int64(scale_expert)
        ].to(cutlass.Float32)
        # The prepared RP stores up first.  The seam is canonical gate then up.
        source_tile = projection_tile % Int32(self.n_tiles)
        dst_base = source_tile * Int32(self.tile_n)
        if projection_tile < Int32(self.n_tiles):
            dst_base += Int32(self.n)
        if c == Int32(0):
            col_base = warp * Int32(self.tile_n // self.num_warps) + q
            for nt in cutlass.range_constexpr(self.n_fragments):
                tile_col = col_base + Int32(nt * 16)
                if source_tile * Int32(self.tile_n) + tile_col < Int32(self.n):
                    col = dst_base + tile_col
                    fragment = acc[nt]
                    projections[Int64(pair) * Int64(2 * self.n) + Int64(col)] = (
                        value_scale * fragment[0]
                    ).to(cutlass.BFloat16)
                    projections[
                        Int64(pair) * Int64(2 * self.n) + Int64(col + Int32(8))
                    ] = (value_scale * fragment[2]).to(cutlass.BFloat16)

    @cute.kernel
    def kernel(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        task = Int32(bidx)
        projection_tiles = Int32(2 * self.n_tiles)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        while task < num_pairs * projection_tiles:
            pair = task % num_pairs
            projection_tile = task // num_pairs
            expert = local_expert_id(topk_ids[Int64(pair)], Int32(alpha.shape[0]))
            if expert >= Int32(0) and expert < Int32(alpha.shape[0]):
                self._run_task(
                    a_source,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    projections,
                    alpha,
                    input_scale,
                    shared_ptr_to_u32(storage.words.data_ptr()),
                    tid,
                    cute.arch.make_warp_uniform(cute.arch.warp_idx()),
                    pair,
                    expert,
                    projection_tile,
                )
            task += Int32(gdimx)


class W4A8CompactMicroProjectionActivationKernel(W4A8CompactMicroProjectionKernel):
    """Pair gate/up projection groups and materialize their MXFP8 activation."""

    def __init__(
        self,
        k: int,
        n: int,
        num_topk: int,
        activation: W4A8CompactMicroActivationKernel,
        *,
        csf_inline: bool = False,
        max_copy_groups: int = 2,
        fuse_input_quantization: bool = False,
        pipeline_stages: int = 2,
    ):
        super().__init__(
            k,
            n,
            num_topk,
            csf_inline=csf_inline,
            max_copy_groups=max_copy_groups,
            pipeline_stages=pipeline_stages,
        )
        self.num_warps = 2
        self.threads_per_cta = self.num_warps * 32
        self.n_fragments = self.tile_n // (self.num_warps * 16)
        self.activation = activation
        self.share_projection_input = True
        self.fuse_input_quantization = bool(fuse_input_quantization)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        scale_rows: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        grid = num_pairs * Int32(self.n_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(grid, max_active_clusters * Int32(2))
        a_source = a
        if cutlass.const_expr(not self.fuse_input_quantization):
            a_source = cute.recast_tensor(a, cutlass.Uint32)
        self.fused_kernel(
            a_source,
            scale_rows,
            w13_rp,
            w13_sfb,
            projections,
            intermediate,
            topk_ids,
            alpha,
            input_scale,
            num_pairs,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta * 2, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.kernel
    def fused_kernel(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        group = tid // Int32(self.threads_per_cta)
        local_tid = tid % Int32(self.threads_per_cta)
        local_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(
            self.num_warps
        )
        task = Int32(bidx)
        projected_flat = cute.make_tensor(
            projections.iterator,
            cute.make_layout((projections.shape[0] * projections.shape[1],)),
        )
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words * 2], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        while task < num_pairs * Int32(self.n_tiles):
            pair = task % num_pairs
            output_tile = task // num_pairs
            expert = local_expert_id(topk_ids[Int64(pair)], Int32(alpha.shape[0]))
            if expert >= Int32(0) and expert < Int32(alpha.shape[0]):
                self._run_task(
                    a_source,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    projected_flat,
                    alpha,
                    input_scale,
                    smem_base + group * Int32(self.shared_words * 4),
                    local_tid,
                    local_warp,
                    pair,
                    expert,
                    output_tile + group * Int32(self.n_tiles),
                )
                # Both BF16 projections must be visible before activation reads.
                cute.arch.sync_threads()
                if tid < Int32(32):
                    self.activation.activate_tile(
                        projections, intermediate, pair, output_tile, tid
                    )
                cute.arch.sync_threads()
            task += Int32(gdimx)


class W4A8CompactCoalesceRoutesKernel:
    """Group active (token, slot) pairs by expert into <=8-column MMA groups.

    Metadata layout in ``coalesce_meta`` (Int32[1 + 10 * max_pairs]):
      - [0]: num_groups
      - [1 .. max_pairs]: group_expert[g]
      - [1 + max_pairs .. 2 * max_pairs]: group_count[g] in 1..8
      - [1 + 2 * max_pairs .. 1 + 10 * max_pairs - 1]: group_pairs[g * 8 + c]
    """

    def __init__(self, max_pairs: int, num_experts: int):
        if max_pairs <= 0 or max_pairs > 256:
            raise ValueError(
                f"coalesced compact routing requires 1 <= max_pairs <= 256, got {max_pairs}"
            )
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.max_pairs = int(max_pairs)
        self.num_experts = int(num_experts)
        n_sort = 1 << (max(self.max_pairs, 32) - 1).bit_length()
        self.n_sort = int(n_sort)
        self.log_n_sort = int(n_sort.bit_length() - 1)
        self.threads = self.n_sort
        self.num_warps = self.threads // 32
        self.keys_offset = 0
        self.warp_sums_offset = self.n_sort * 4
        self.shared_words = (self.warp_sums_offset + 16 * 4 + 3) // 4

    @cute.jit
    def __call__(
        self,
        topk_ids: cute.Tensor,
        coalesce_meta: cute.Tensor,
        num_pairs: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(topk_ids, coalesce_meta, num_pairs).launch(
            grid=(1, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        topk_ids: cute.Tensor,
        coalesce_meta: cute.Tensor,
        num_pairs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tid = Int32(tidx)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 128
            ]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())
        keys_base = base + Int32(self.keys_offset)
        warp_sums_base = base + Int32(self.warp_sums_offset)

        key = Uint32(0xFFFFFFFF)
        if tid < num_pairs:
            expert = local_expert_id(topk_ids[Int64(tid)], Int32(self.num_experts))
            if expert >= Int32(0):
                key = (expert.to(Uint32) << Uint32(9)) | tid.to(Uint32)

        # In-register warp bitonic sort for stages within 32 lanes (0 barriers!),
        # followed by SMEM merge stages for N > 32.
        for stage in cutlass.range_constexpr(5):
            k_size = 1 << (stage + 1)
            dir_asc = (tid & Int32(k_size)) == Int32(0)
            for sub in cutlass.range_constexpr(stage + 1):
                j_size = 1 << (stage - sub)
                other = cute.arch.shuffle_sync_bfly(key, offset=j_size)
                is_hi = (tid & Int32(j_size)) != Int32(0)
                want_min = Int32(1)
                if dir_asc:
                    if is_hi:
                        want_min = Int32(0)
                else:
                    if not is_hi:
                        want_min = Int32(0)
                if want_min != Int32(0):
                    if other < key:
                        key = other
                else:
                    if other > key:
                        key = other

        if cutlass.const_expr(self.log_n_sort > 5):
            for stage in cutlass.range_constexpr(5, self.log_n_sort):
                k_size = 1 << (stage + 1)
                dir_asc = (tid & Int32(k_size)) == Int32(0)
                for sub in cutlass.range_constexpr(stage + 1):
                    j_size = 1 << (stage - sub)
                    if cutlass.const_expr(j_size >= 32):
                        st_shared_u32(keys_base + (tid << Int32(2)), key)
                        cute.arch.sync_threads()
                        ixj = tid ^ Int32(j_size)
                        other = ld_shared_u32(keys_base + (ixj << Int32(2)))
                        is_hi = (tid & Int32(j_size)) != Int32(0)
                        want_min = Int32(1)
                        if dir_asc:
                            if is_hi:
                                want_min = Int32(0)
                        else:
                            if not is_hi:
                                want_min = Int32(0)
                        if want_min != Int32(0):
                            if other < key:
                                key = other
                        else:
                            if other > key:
                                key = other
                        cute.arch.sync_threads()
                    else:
                        other = cute.arch.shuffle_sync_bfly(key, offset=j_size)
                        is_hi = (tid & Int32(j_size)) != Int32(0)
                        want_min = Int32(1)
                        if dir_asc:
                            if is_hi:
                                want_min = Int32(0)
                        else:
                            if not is_hi:
                                want_min = Int32(0)
                        if want_min != Int32(0):
                            if other < key:
                                key = other
                        else:
                            if other > key:
                                key = other

        st_shared_u32(keys_base + (tid << Int32(2)), key)
        cute.arch.sync_threads()

        # Parallel group-head detection: each active thread finds its offset
        # within its contiguous expert run; every multiple-of-8 offset starts
        # a new <=8-column MMA group.
        is_head = Int32(0)
        exp = Int32(-1)
        if key != Uint32(0xFFFFFFFF):
            exp = (key >> Uint32(9)).to(Int32)
            pos = Int32(0)
            probe = tid - Int32(1)
            keep_going = Int32(1)
            while probe >= Int32(0) and keep_going != Int32(0):
                prev_k = ld_shared_u32(keys_base + (probe << Int32(2)))
                if (prev_k >> Uint32(9)).to(Int32) == exp:
                    pos += Int32(1)
                    probe -= Int32(1)
                else:
                    keep_going = Int32(0)
            if (pos & Int32(7)) == Int32(0):
                is_head = Int32(1)

        # Exclusive prefix sum of is_head across the CTA -> dense group IDs.
        w_scan = is_head
        for shift in cutlass.range_constexpr(5):
            delta = 1 << shift
            prev_val = cute.arch.shuffle_sync_up(w_scan, offset=delta)
            if lane >= Int32(delta):
                w_scan = w_scan + prev_val
        if cutlass.const_expr(self.num_warps > 1):
            if lane == Int32(31):
                st_shared_u32(warp_sums_base + (warp << Int32(2)), w_scan.to(Uint32))
            cute.arch.sync_threads()
            if warp == Int32(0):
                w_val = Int32(0)
                if lane < Int32(self.num_warps):
                    w_val = ld_shared_u32(warp_sums_base + (lane << Int32(2))).to(Int32)
                w_tot = w_val
                for shift in cutlass.range_constexpr(4):
                    delta = 1 << shift
                    pv = cute.arch.shuffle_sync_up(w_tot, offset=delta)
                    if lane >= Int32(delta):
                        w_tot = w_tot + pv
                if lane < Int32(self.num_warps):
                    st_shared_u32(
                        warp_sums_base + (lane << Int32(2)),
                        (w_tot - w_val).to(Uint32),
                    )
                if lane == Int32(self.num_warps - 1):
                    coalesce_meta[Int64(0)] = w_tot
            cute.arch.sync_threads()
            warp_offset = ld_shared_u32(warp_sums_base + (warp << Int32(2))).to(Int32)
            group_id = warp_offset + w_scan - is_head
        else:
            if lane == Int32(31):
                coalesce_meta[Int64(0)] = w_scan
            group_id = w_scan - is_head

        if is_head != Int32(0):
            first_pair = (key & Uint32(0x1FF)).to(Int32)
            cnt = Int32(0)
            still_match = Int32(1)
            for c_off in cutlass.range_constexpr(8):
                idx = tid + Int32(c_off)
                p_out = first_pair
                if still_match != Int32(0) and idx < num_pairs:
                    ck = ld_shared_u32(keys_base + (idx << Int32(2)))
                    if ck != Uint32(0xFFFFFFFF) and (ck >> Uint32(9)).to(Int32) == exp:
                        p_out = (ck & Uint32(0x1FF)).to(Int32)
                        cnt += Int32(1)
                    else:
                        still_match = Int32(0)
                else:
                    still_match = Int32(0)
                coalesce_meta[
                    Int64(1 + 2 * self.max_pairs)
                    + Int64(group_id * Int32(8) + Int32(c_off))
                ] = p_out
            coalesce_meta[Int64(1) + Int64(group_id)] = exp
            coalesce_meta[Int64(1 + self.max_pairs) + Int64(group_id)] = cnt


class W4A8CompactCoalescedFC1Kernel:
    """8-column expert-coalesced FC1 gate/up projection with in-SMEM SwiGLU."""

    tile_n = 128
    tile_k = 128
    num_warps = 2
    threads_per_group = 64
    threads_per_cta = 128

    def __init__(
        self,
        k: int,
        n: int,
        num_topk: int,
        max_pairs: int,
        activation: W4A8CompactMicroActivationKernel,
        *,
        tile_n: int = 128,
        csf_inline: bool = False,
        max_copy_groups: int = 2,
        fuse_input_quantization: bool = False,
        pipeline_stages: int = 2,
    ):
        if k <= 0 or n < 64 or k % 128 or n % 64:
            raise ValueError(
                "coalesced compact W4A8 FC1 requires K divisible by 128 and N divisible by 64"
            )
        if num_topk <= 0 or max_pairs <= 0:
            raise ValueError("num_topk and max_pairs must be positive")
        if type(tile_n) is not int or tile_n not in (64, 128):
            raise ValueError("coalesced compact FC1 tile N must be 64 or 128")
        if tile_n == 64 and n % 128 != 64:
            raise ValueError(
                "coalesced compact FC1 N64 tiles require the native N64-tail layout"
            )
        if type(max_copy_groups) is not int or max_copy_groups not in (1, 2):
            raise ValueError("compact FC1 permits one or two pending copy groups")
        if type(pipeline_stages) is not int or pipeline_stages not in (1, 2):
            raise ValueError("compact FC1 pipeline stages must be one or two")
        if pipeline_stages == 1 and max_copy_groups != 1:
            raise ValueError("one compact FC1 stage requires one pending copy group")
        self.k = int(k)
        self.n = int(n)
        self.num_topk = int(num_topk)
        self.max_pairs = int(max_pairs)
        self.activation = activation
        self.tile_n = tile_n
        self.csf_inline = bool(csf_inline)
        self.max_copy_groups = max_copy_groups
        self.stages = pipeline_stages
        self.fuse_input_quantization = bool(fuse_input_quantization)
        self.n64_tail = self.n % 128 == 64
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.n_padded = ((self.n + 127) // 128) * 128
        self.n_fragments = self.tile_n // (self.num_warps * 16)
        self.min_blocks_per_mp = 4 if self.tile_n == 64 else 2

        # Shared input ring (shared across gate and up groups):
        # 8 rows * 128B E4M3 payload + 128B (8 * 4B UE8M0 scales + pad) = 1152B/stage.
        self.input_a_bytes = 8 * 128
        self.input_sfa_offset = self.input_a_bytes
        self.input_stage_bytes = self.input_a_bytes + 128
        self.input_ring_bytes = (
            (self.stages * self.input_stage_bytes + 255) // 256
        ) * 256

        # Weight ring per projection group (up = group 0, gate = group 1).
        self.b_payload_bytes = self.tile_n * self.tile_k // 2
        self.sfb_bytes = 512 if self.csf_inline else self.tile_n * 4
        self.weight_stage_bytes = self.b_payload_bytes + self.sfb_bytes
        self.weight_ring_bytes = self.stages * self.weight_stage_bytes
        self.total_smem_bytes = max(
            self.input_ring_bytes + 2 * self.weight_ring_bytes,
            8 * 2 * self.tile_n * 2,
        )
        self.shared_words = (self.total_smem_bytes + 3) // 4

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        scale_rows: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        intermediate: cute.Tensor,
        coalesce_meta: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        grid = num_pairs * Int32(self.n_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(
                grid, max_active_clusters * Int32(self.min_blocks_per_mp)
            )
        a_source = a
        if cutlass.const_expr(not self.fuse_input_quantization):
            a_source = cute.recast_tensor(a, cutlass.Uint32)
        self.kernel(
            a_source,
            scale_rows,
            w13_rp,
            w13_sfb,
            intermediate,
            coalesce_meta,
            alpha,
            input_scale,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )

    @cute.jit
    def _stage_b_k128(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        local_tid: Int32,
        l2_policy,
    ):
        for i in cutlass.range_constexpr(4 * self.tile_n // self.threads_per_group):
            idx = local_tid + Int32(i * self.threads_per_group)
            if idx < Int32(512):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc % Int32(4)
                kb = kc // Int32(4)
                src_word = (
                    tile_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64((packed_half * Int32(4) + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global_l2hint(
                    dst_base + (idx << Int32(4)),
                    get_ptr_as_int64(weights, src_word),
                    l2_policy,
                )

    @cute.jit
    def _stage_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        local_tid: Int32,
        l2_policy,
    ):
        if local_tid < Int32(32):
            src_word = tile_word_base + Int64(packed_half * Int32(128) + local_tid * 4)
            cp_async4_shared_global_l2hint(
                dst_base + (local_tid << Int32(4)),
                get_ptr_as_int64(scales, src_word),
                l2_policy,
            )

    @cute.jit
    def _stage_n64_b_k128(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        local_tid: Int32,
        l2_policy,
    ):
        physical_tile = output_tile // Int32(128 // self.tile_n)
        packed_half = output_tile % Int32(128 // self.tile_n)
        n_rows = Int32(128)
        if physical_tile == Int32(self.n // 128):
            n_rows = Int32(64)
        for i in cutlass.range_constexpr(4 * self.tile_n // self.threads_per_group):
            idx = local_tid + Int32(i * self.threads_per_group)
            if idx < Int32(4 * (self.tile_n // 32) * 32):
                lane = idx & Int32(31)
                tmp = idx >> Int32(5)
                n32 = tmp % Int32(self.tile_n // 32)
                kb = tmp // Int32(self.tile_n // 32)
                copy_words = Int32(1)
                if cutlass.const_expr(self.tile_n == 128):
                    copy_words = (n32 < n_rows // Int32(32)).to(Int32)
                dst_addr = dst_base + (idx << Int32(4))
                if copy_words != Int32(0):
                    source_word = (
                        Int64(expert_idx) * Int64(self.n * (self.k // 32) * 8)
                        + Int64(projection) * Int64(self.n * (self.k // 32) * 4)
                        + Int64(physical_tile) * Int64(128 * (self.k // 32) * 4)
                        + Int64(k128_slice * Int32(4) + kb)
                        * Int64(n_rows // Int32(32))
                        * Int64(128)
                        + Int64(
                            (packed_half * Int32(self.tile_n // 32) + n32) * Int32(128)
                            + lane * Int32(4)
                        )
                    )
                    cp_async4_shared_global_l2hint(
                        dst_addr,
                        get_ptr_as_int64(weights, source_word),
                        l2_policy,
                    )
                else:
                    st_shared_v4_u32(
                        dst_addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    )

    @cute.jit
    def _stage_n64_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        local_tid: Int32,
        l2_policy,
        bounds=None,
        value_offset=None,
        base_word=None,
    ):
        physical_tile = output_tile // Int32(128 // self.tile_n)
        packed_half = output_tile % Int32(128 // self.tile_n)
        n_rows = Int32(128)
        if physical_tile == Int32(self.n // 128):
            n_rows = Int32(64)
        if cutlass.const_expr(self.csf_inline):
            if local_tid < Int32(32):
                target = cute.make_ptr(
                    cutlass.Uint8, dst_base, cute.AddressSpace.smem, assumed_align=16
                )
                group = (
                    projection * Int32(self.csf_reader.plane.storage.groups)
                    + physical_tile * Int32(self.csf_reader.plane.storage.column_groups)
                    + k128_slice
                )
                cached = None
                if cutlass.const_expr(bounds is not None):
                    first, end = Uint32(0), Uint32(0)
                    for chunk in cutlass.range_constexpr(cute.size(bounds, mode=[0])):
                        selected_first = cute.arch.shuffle_sync(
                            bounds[chunk, 0], k128_slice % Int32(32)
                        )
                        selected_end = cute.arch.shuffle_sync(
                            bounds[chunk, 1], k128_slice % Int32(32)
                        )
                        if k128_slice // Int32(32) == Int32(chunk):
                            first, end = selected_first, selected_end
                    cached = (first, end)
                self.csf_reader.stage_compressed(
                    scales.iterator,
                    expert_idx,
                    group,
                    target,
                    local_tid,
                    cached,
                    value_offset,
                    base_word,
                    coordinates=(projection, physical_tile, k128_slice),
                )
            return
        if local_tid < Int32(self.tile_n // 4):
            dst_addr = dst_base + (local_tid << Int32(4))
            copy_words = Int32(1)
            if cutlass.const_expr(self.tile_n == 128):
                copy_words = (local_tid < n_rows // Int32(4)).to(Int32)
            if copy_words != Int32(0):
                source_word = (
                    Int64(expert_idx) * Int64(self.n * (self.k // 32) // 2)
                    + Int64(projection) * Int64(self.n * (self.k // 32) // 4)
                    + Int64(physical_tile) * Int64(128 * (self.k // 32) // 4)
                    + Int64(
                        k128_slice * n_rows
                        + packed_half * Int32(self.tile_n)
                        + local_tid * Int32(4)
                    )
                )
                cp_async4_shared_global_l2hint(
                    dst_addr,
                    get_ptr_as_int64(scales, source_word),
                    l2_policy,
                )
            else:
                st_shared_v4_u32(dst_addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def _stage_slice(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        coalesce_meta: cute.Tensor,
        input_ring_base: Int32,
        weight_ring_base: Int32,
        group: Int32,
        local_tid: Int32,
        local_warp: Int32,
        my_stage_tok: Int32,
        my_scale_tok: Int32,
        group_idx: Int32,
        group_count: Int32,
        expert: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        l2_policy,
        csf_bounds=None,
        csf_value_offset=None,
        csf_base_word=None,
    ):
        stage_idx = Int32(0)
        if cutlass.const_expr(self.stages == 2):
            stage_idx = k128_slice & Int32(1)
        input_stage = input_ring_base + stage_idx * Int32(self.input_stage_bytes)
        weight_stage = weight_ring_base + stage_idx * Int32(self.weight_stage_bytes)

        if cutlass.const_expr(not self.fuse_input_quantization):
            if group == Int32(0):
                col_tok = local_tid >> Int32(3)
                vec = local_tid & Int32(7)
                if col_tok < group_count:
                    phys_vec = vec ^ col_tok
                    src_word = (
                        Int64(my_stage_tok) * Int64(self.k // 4)
                        + Int64(k128_slice) * Int64(32)
                        + Int64(vec * Int32(4))
                    )
                    cp_async4_shared_global(
                        input_stage + (col_tok << Int32(7)) + (phys_vec << Int32(4)),
                        get_ptr_as_int64(a_source, src_word),
                    )
                if local_tid < group_count:
                    sf_src = Int64(my_scale_tok) * Int64(self.k // 32) + Int64(
                        k128_slice
                    ) * Int64(4)
                    cp_async_u32_shared_global(
                        input_stage
                        + Int32(self.input_sfa_offset)
                        + (local_tid << Int32(2)),
                        get_ptr_as_int64(scale_words, sf_src),
                    )

        b_base = weight_stage
        sfb_base = weight_stage + Int32(self.b_payload_bytes)
        projection_tile = output_tile + group * Int32(self.n_tiles)
        if cutlass.const_expr(self.n64_tail):
            self._stage_n64_b_k128(
                w13_rp,
                b_base,
                expert,
                group,
                output_tile,
                k128_slice,
                local_tid,
                l2_policy,
            )
            self._stage_n64_sfb(
                w13_sfb,
                sfb_base,
                expert,
                group,
                output_tile,
                k128_slice,
                local_tid,
                l2_policy,
                csf_bounds,
                csf_value_offset,
                csf_base_word,
            )
        else:
            packed_tile = projection_tile // Int32(2)
            packed_half = projection_tile % Int32(2)
            packed_tile_index = (
                Int64(expert) * Int64(self.n // 128) + Int64(packed_tile)
            ) * Int64(self.k // 128) + Int64(k128_slice)
            tile_word_base = packed_tile_index * Int64(4096)
            self._stage_b_k128(
                w13_rp, b_base, tile_word_base, packed_half, local_tid, l2_policy
            )
            scale_word_base = packed_tile_index * Int64(256)
            self._stage_sfb(
                w13_sfb, sfb_base, scale_word_base, packed_half, local_tid, l2_policy
            )

        if cutlass.const_expr(self.fuse_input_quantization):
            if group == Int32(0):
                lane = local_tid & Int32(31)
                c_idx = local_warp
                while c_idx < group_count:
                    tok = coalesce_meta[
                        Int64(1 + 2 * self.max_pairs)
                        + Int64(group_idx * Int32(8) + c_idx)
                    ] // Int32(self.num_topk)
                    _stage_compact_input_k128_swizzled(
                        a_source,
                        input_stage + (c_idx << Int32(7)),
                        input_stage
                        + Int32(self.input_sfa_offset)
                        + (c_idx << Int32(2)),
                        lane,
                        tok,
                        c_idx,
                        k128_slice,
                        self.k,
                    )
                    c_idx += Int32(2)

    @cute.kernel
    def kernel(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        intermediate: cute.Tensor,
        coalesce_meta: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_global = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        group = tid // Int32(self.threads_per_group)
        local_tid = tid % Int32(self.threads_per_group)
        local_warp = warp_global % Int32(self.num_warps)
        lane = local_tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        num_groups = coalesce_meta[Int64(0)]
        total_tasks = num_groups * Int32(self.n_tiles)
        task = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        input_ring_base = smem_base
        weight_ring_base = (
            smem_base
            + Int32(self.input_ring_bytes)
            + group * Int32(self.weight_ring_bytes)
        )
        proj_smem = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16, smem_base, cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_ordered_layout((8, 2, self.tile_n), order=(2, 1, 0)),
        )
        l2_policy = create_l2_evict_first_policy()
        acc = tuple(
            cute.make_rmem_tensor((4,), cutlass.Float32)
            for _ in range(self.n_fragments)
        )
        b_scale_words = cute.make_rmem_tensor((self.n_fragments,), Uint32)
        b_words = cute.make_rmem_tensor((self.n_fragments * 2,), Uint32)
        act_value = cute.make_rmem_tensor((4,), cutlass.Float32)
        act_gate = cute.make_rmem_tensor((4,), cutlass.Float32)
        act_up = cute.make_rmem_tensor((4,), cutlass.Float32)
        k128_tiles = Int32(self.k // 128)

        while task < total_tasks:
            group_idx = task // Int32(self.n_tiles)
            output_tile = task % Int32(self.n_tiles)
            self._run_task(
                a_source,
                scale_words,
                w13_rp,
                w13_sfb,
                intermediate,
                coalesce_meta,
                alpha,
                input_scale,
                input_ring_base,
                weight_ring_base,
                proj_smem,
                l2_policy,
                acc,
                b_scale_words,
                b_words,
                act_value,
                act_gate,
                act_up,
                warp_global,
                group,
                local_tid,
                local_warp,
                lane,
                q,
                c,
                group_idx,
                output_tile,
                k128_tiles,
            )
            task += Int32(gdimx)

    @cute.jit
    def _run_task(
        self,
        a_source: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        intermediate: cute.Tensor,
        coalesce_meta: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        input_ring_base: Int32,
        weight_ring_base: Int32,
        proj_smem: cute.Tensor,
        l2_policy,
        acc,
        b_scale_words: cute.Tensor,
        b_words: cute.Tensor,
        act_value: cute.Tensor,
        act_gate: cute.Tensor,
        act_up: cute.Tensor,
        warp_global: Int32,
        group: Int32,
        local_tid: Int32,
        local_warp: Int32,
        lane: Int32,
        q: Int32,
        c: Int32,
        group_idx: Int32,
        output_tile: Int32,
        k128_tiles: Int32,
    ):
        expert = coalesce_meta[Int64(1) + Int64(group_idx)]
        group_count = coalesce_meta[Int64(1 + self.max_pairs) + Int64(group_idx)]
        # Keep each thread's staging token indices in scalar registers (0 local spills).
        meta_pair_base = Int64(1 + 2 * self.max_pairs) + Int64(group_idx * Int32(8))
        my_stage_tok = coalesce_meta[
            meta_pair_base + Int64(local_tid >> Int32(3))
        ] // Int32(self.num_topk)
        my_scale_tok = Int32(0)
        if local_tid < Int32(8):
            my_scale_tok = coalesce_meta[meta_pair_base + Int64(local_tid)] // Int32(
                self.num_topk
            )

        q_in = q
        if q_in >= group_count:
            q_in = Int32(0)

        csf_bounds, csf_base_word, csf_value_offset = None, None, None
        if cutlass.const_expr(self.csf_inline):
            physical_tile = output_tile // Int32(128 // self.tile_n)
            section_expert = expert * Int32(self.csf_reader.plane.sections) + group
            csf_value_offset = self.csf_reader.plane.group_value_offset(
                w13_sfb.iterator
            )
            csf_bounds = self.csf_reader.prefetch_column_bounds(
                w13_sfb.iterator, section_expert, physical_tile
            )
            csf_base_word = self.csf_reader.prefetch_row_bases(
                w13_sfb.iterator, section_expert, physical_tile, local_tid
            )

        if cutlass.const_expr(0 < self.k // 128):
            self._stage_slice(
                a_source,
                scale_words,
                w13_rp,
                w13_sfb,
                coalesce_meta,
                input_ring_base,
                weight_ring_base,
                group,
                local_tid,
                local_warp,
                my_stage_tok,
                my_scale_tok,
                group_idx,
                group_count,
                expert,
                output_tile,
                Int32(0),
                l2_policy,
                csf_bounds,
                csf_value_offset,
                csf_base_word,
            )
        cute.arch.cp_async_commit_group()

        for nt in cutlass.range_constexpr(self.n_fragments):
            acc[nt].fill(0.0)
        b_scale_words.fill(Uint32(0))

        k128_slice = Int32(0)
        while k128_slice < k128_tiles:
            stage_idx = Int32(0)
            if cutlass.const_expr(self.stages == 2):
                stage_idx = k128_slice & Int32(1)
            input_stage = input_ring_base + stage_idx * Int32(self.input_stage_bytes)
            weight_stage = weight_ring_base + stage_idx * Int32(self.weight_stage_bytes)
            b_base = weight_stage
            sfb_base = weight_stage + Int32(self.b_payload_bytes)

            if cutlass.const_expr(self.max_copy_groups == 1):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            next_slice = k128_slice + Int32(1)
            if cutlass.const_expr(self.stages == 2):
                if next_slice < k128_tiles:
                    self._stage_slice(
                        a_source,
                        scale_words,
                        w13_rp,
                        w13_sfb,
                        coalesce_meta,
                        input_ring_base,
                        weight_ring_base,
                        group,
                        local_tid,
                        local_warp,
                        my_stage_tok,
                        my_scale_tok,
                        group_idx,
                        group_count,
                        expert,
                        output_tile,
                        next_slice,
                        l2_policy,
                        csf_bounds,
                        csf_value_offset,
                        csf_base_word,
                    )
                cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.max_copy_groups == 2):
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
            if cutlass.const_expr(self.csf_inline):
                self.csf_reader.expand_slots(
                    w13_sfb.iterator,
                    sfb_base,
                    local_tid,
                    self.threads_per_group,
                    1,
                )
                cute.arch.sync_threads()
                if cutlass.const_expr(self.tile_n == 64):
                    sfb_base += (output_tile & Int32(1)) * Int32(256)

            for nt in cutlass.range_constexpr(self.n_fragments):
                n8 = (
                    local_warp * Int32(self.n_fragments * 2)
                    + Int32(nt * 2)
                    + (lane & Int32(1))
                )
                b_scale_words[nt] = ld_shared_u32(
                    sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                )
            asc = ld_shared_u32(
                input_stage + Int32(self.input_sfa_offset) + (q_in << Int32(2))
            )
            for kb in cutlass.range_constexpr(4):
                phys_vec = (Int32(kb * 2) + (c >> Int32(1))) ^ q_in
                a_src = (
                    input_stage
                    + (q_in << Int32(7))
                    + (phys_vec << Int32(4))
                    + ((c & Int32(1)) << Int32(3))
                )
                a0, a2 = ld_shared_v2_u32(a_src)
                for chunk in cutlass.range_constexpr(self.n_fragments // 2):
                    source_chunk = (
                        Int32(kb * (self.tile_n // 32))
                        + local_warp * Int32(self.n_fragments // 2)
                        + Int32(chunk)
                    )
                    bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                        b_base + ((source_chunk * Int32(32) + lane) << Int32(4))
                    )
                    b_words[chunk * 4] = bw0
                    b_words[chunk * 4 + 1] = bw1
                    b_words[chunk * 4 + 2] = bw2
                    b_words[chunk * 4 + 3] = bw3
                for nt in cutlass.range_constexpr(self.n_fragments):
                    w0, w2 = e2m1x8_to_qmma_e2m1x8(b_words[nt * 2])
                    w1, w3 = e2m1x8_to_qmma_e2m1x8(b_words[nt * 2 + 1])
                    sfb = b_scale_words[nt]
                    fragment = acc[nt]
                    x0, x1, x2, x3 = _mma_fp4_weights_fp8_activation(
                        w0,
                        w1,
                        w2,
                        w3,
                        a0,
                        a2,
                        sfb,
                        asc,
                        block=kb,
                    )
                    fragment[0] = fragment[0] + x0
                    fragment[1] = fragment[1] + x1
                    fragment[2] = fragment[2] + x2
                    fragment[3] = fragment[3] + x3
            if cutlass.const_expr(self.max_copy_groups == 2 or self.stages == 1):
                cute.arch.sync_threads()
            if cutlass.const_expr(self.stages == 1):
                if next_slice < k128_tiles:
                    self._stage_slice(
                        a_source,
                        scale_words,
                        w13_rp,
                        w13_sfb,
                        coalesce_meta,
                        input_ring_base,
                        weight_ring_base,
                        group,
                        local_tid,
                        local_warp,
                        my_stage_tok,
                        my_scale_tok,
                        group_idx,
                        group_count,
                        expert,
                        output_tile,
                        next_slice,
                        l2_policy,
                        csf_bounds,
                        csf_value_offset,
                        csf_base_word,
                    )
                cute.arch.cp_async_commit_group()
            k128_slice += Int32(1)

        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        scale_expert = expert
        if cutlass.const_expr(input_scale.shape[0] == 1):
            scale_expert = Int32(0)
        value_scale = alpha[expert].to(cutlass.Float32) * input_scale[
            Int64(scale_expert)
        ].to(cutlass.Float32)
        # group == 0 computes up (proj_idx = 1); group == 1 computes gate (proj_idx = 0).
        proj_idx = Int32(1) - group
        col_base = local_warp * Int32(self.tile_n // self.num_warps) + q
        for row_part in cutlass.range_constexpr(2):
            col_idx = c * Int32(2) + Int32(row_part)
            if col_idx < group_count:
                for nt in cutlass.range_constexpr(self.n_fragments):
                    tile_col = col_base + Int32(nt * 16)
                    if output_tile * Int32(self.tile_n) + tile_col < Int32(self.n):
                        fragment = acc[nt]
                        proj_smem[col_idx, proj_idx, tile_col] = (
                            value_scale * fragment[row_part]
                        ).to(cutlass.BFloat16)
                        proj_smem[col_idx, proj_idx, tile_col + Int32(8)] = (
                            value_scale * fragment[2 + row_part]
                        ).to(cutlass.BFloat16)
        cute.arch.sync_threads()

        # All four warps cooperatively run SwiGLU + MXFP8 quantization directly
        # from SMEM into the per-pair intermediate buffer.
        col = output_tile * Int32(self.tile_n) + lane * Int32(4)
        rows = Int64(self.max_pairs)
        words_per_row = Int64(self.n_padded // 4)
        intermediate_u8 = cute.recast_tensor(intermediate, cutlass.Uint8)
        for p_pass in cutlass.range_constexpr(2):
            col_idx = warp_global + Int32(p_pass * 4)
            if col_idx < group_count:
                pair = coalesce_meta[meta_pair_base + Int64(col_idx)]
                physical_row = Int64(pair)
                if cutlass.const_expr(self.tile_n == 64):
                    act_value.fill(0.0)
                    scale_byte = Uint32(0)
                    if lane < Int32(16):
                        for elem in cutlass.range_constexpr(4):
                            act_gate[elem] = proj_smem[
                                col_idx, Int32(0), lane * Int32(4) + Int32(elem)
                            ].to(cutlass.Float32)
                            act_up[elem] = proj_smem[
                                col_idx, Int32(1), lane * Int32(4) + Int32(elem)
                            ].to(cutlass.Float32)
                            if cutlass.const_expr(self.activation.has_swiglu_limit):
                                limit = cutlass.Float32(self.activation.swiglu_limit)
                                act_gate[elem] = cutlass.min(act_gate[elem], limit)
                                act_up[elem] = cutlass.max(
                                    cutlass.min(act_up[elem], limit),
                                    -limit,
                                )
                        for elem in cutlass.range_constexpr(4):
                            sigmoid = cute.arch.rcp_approx(
                                cutlass.Float32(1.0)
                                + cute.math.exp(
                                    -act_gate[elem],
                                    fastmath=self.activation.fast_math,
                                )
                            )
                            act_value[elem] = cutlass.BFloat16(
                                act_gate[elem] * sigmoid * act_up[elem]
                            ).to(cutlass.Float32)
                    # All 32 lanes must participate in full-warp shuffle_sync_bfly.
                    amax = fabs_f32(act_value[0])
                    for elem in cutlass.range_constexpr(1, 4):
                        amax = fmax_f32(amax, fabs_f32(act_value[elem]))
                    for shift in cutlass.range_constexpr(3):
                        amax = fmax_f32(
                            amax,
                            cute.arch.shuffle_sync_bfly(amax, offset=1 << shift),
                        )
                    if lane < Int32(16):
                        _, scale_byte = pow2_ceil_ue8m0(
                            amax * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                        )
                        inv_scale = ue8m0_to_output_scale(scale_byte)
                        for elem in cutlass.range_constexpr(4):
                            act_value[elem] = act_value[elem] * inv_scale
                        intermediate[
                            physical_row * words_per_row
                            + Int64(output_tile * Int32(16) + lane)
                        ] = cvt_f32x4_to_e4m3x4(
                            act_value[0], act_value[1], act_value[2], act_value[3]
                        )
                        if lane % Int32(8) == Int32(0):
                            n128_tile = output_tile >> Int32(1)
                            half_k32 = (output_tile & Int32(1)) << Int32(1)
                            scale_byte_index = (
                                rows * words_per_row
                                + Int64(n128_tile) * rows
                                + physical_row
                            ) * Int64(4) + Int64(half_k32 + (lane >> Int32(3)))
                            intermediate_u8[scale_byte_index] = (
                                scale_byte & Uint32(0xFF)
                            ).to(cutlass.Uint8)
                    else:
                        # Zero-pad the upper 64 columns of the trailing N64 tile.
                        if output_tile == Int32(self.n_tiles - 1):
                            intermediate[
                                physical_row * words_per_row
                                + Int64(output_tile * Int32(16) + lane)
                            ] = Uint32(0)
                            if lane % Int32(8) == Int32(0):
                                n128_tile = output_tile >> Int32(1)
                                scale_byte_index = (
                                    rows * words_per_row
                                    + Int64(n128_tile) * rows
                                    + physical_row
                                ) * Int64(4) + Int64(lane >> Int32(3))
                                intermediate_u8[scale_byte_index] = cutlass.Uint8(0)
                else:
                    act_value.fill(0.0)
                    scale_byte = Uint32(0)
                    if col < Int32(self.n):
                        for elem in cutlass.range_constexpr(4):
                            act_gate[elem] = proj_smem[
                                col_idx, Int32(0), lane * Int32(4) + Int32(elem)
                            ].to(cutlass.Float32)
                            act_up[elem] = proj_smem[
                                col_idx, Int32(1), lane * Int32(4) + Int32(elem)
                            ].to(cutlass.Float32)
                            if cutlass.const_expr(self.activation.has_swiglu_limit):
                                limit = cutlass.Float32(self.activation.swiglu_limit)
                                act_gate[elem] = cutlass.min(act_gate[elem], limit)
                                act_up[elem] = cutlass.max(
                                    cutlass.min(act_up[elem], limit),
                                    -limit,
                                )
                        for elem in cutlass.range_constexpr(4):
                            sigmoid = cute.arch.rcp_approx(
                                cutlass.Float32(1.0)
                                + cute.math.exp(
                                    -act_gate[elem],
                                    fastmath=self.activation.fast_math,
                                )
                            )
                            act_value[elem] = cutlass.BFloat16(
                                act_gate[elem] * sigmoid * act_up[elem]
                            ).to(cutlass.Float32)
                    amax = fabs_f32(act_value[0])
                    for elem in cutlass.range_constexpr(1, 4):
                        amax = fmax_f32(amax, fabs_f32(act_value[elem]))
                    for shift in cutlass.range_constexpr(3):
                        amax = fmax_f32(
                            amax,
                            cute.arch.shuffle_sync_bfly(amax, offset=1 << shift),
                        )
                    if col < Int32(self.n):
                        _, scale_byte = pow2_ceil_ue8m0(
                            amax * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                        )
                        inv_scale = ue8m0_to_output_scale(scale_byte)
                        for elem in cutlass.range_constexpr(4):
                            act_value[elem] = act_value[elem] * inv_scale
                    intermediate[
                        physical_row * words_per_row
                        + Int64(output_tile * Int32(32) + lane)
                    ] = cvt_f32x4_to_e4m3x4(
                        act_value[0], act_value[1], act_value[2], act_value[3]
                    )
                    if lane % Int32(8) == Int32(0):
                        scale_byte_index = (
                            rows * words_per_row
                            + Int64(output_tile) * rows
                            + physical_row
                        ) * Int64(4) + Int64(lane // Int32(8))
                        intermediate_u8[scale_byte_index] = (
                            scale_byte & Uint32(0xFF)
                        ).to(cutlass.Uint8)
        cute.arch.sync_threads()


class W4A8CompactCoalescedFC2Kernel:
    """8-column expert-coalesced FC2 down projection with L2::evict_first weights."""

    tile_n = 128
    tile_k = 128
    num_warps = 4
    threads_per_cta = 128
    stages = 2

    def __init__(
        self,
        k: int,
        n: int,
        max_pairs: int,
        *,
        csf_inline: bool = False,
    ):
        if k <= 0 or n < 64 or k % 256 or n % 64:
            raise ValueError(
                "coalesced compact W4A8 FC2 requires K divisible by 256 and N divisible by 64"
            )
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        self.k = int(k)
        self.n = int(n)
        self.max_pairs = int(max_pairs)
        self.csf_inline = bool(csf_inline)
        self.n64_tail = self.n % 128 == 64
        self.intermediate_tiles = (self.n + 127) // 128
        self.output_tiles = self.k // 128
        self.packed_output_tiles = self.k // 256

        # 8 rows * 128B E4M3 + 128B (8 * 4B UE8M0 scales + pad) = 1152B/stage.
        self.a_payload_bytes = 8 * 128
        self.a_scale_bytes = 128
        self.a_stage_bytes = self.a_payload_bytes + self.a_scale_bytes
        self.a_storage_bytes = self.stages * self.a_stage_bytes
        self.b_storage_offset = ((self.a_storage_bytes + 255) // 256) * 256
        self.b_stage_bytes = self.tile_n * self.tile_k // 2
        self.b_storage_bytes = self.stages * self.b_stage_bytes
        self.sfb_storage_offset = self.b_storage_offset + self.b_storage_bytes
        self.sfb_stage_bytes = 512
        self.shared_bytes = self.sfb_storage_offset + self.stages * self.sfb_stage_bytes
        self.shared_words = (self.shared_bytes + 3) // 4
        self.region_words = self.shared_words + (128 if self.csf_inline else 0)

    @cute.jit
    def __call__(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        route_outputs: cute.Tensor,
        token_weights: cute.Tensor,
        coalesce_meta: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        grid = num_pairs * Int32(self.output_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(grid, max_active_clusters * Int32(4))
        self.kernel(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            route_outputs,
            token_weights,
            coalesce_meta,
            down_alpha,
            global_scale,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=4,
            stream=stream,
        )

    @cute.jit
    def _stage_n64_b(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
        l2_policy,
    ):
        total_k32 = self.n // 32
        expert_words = self.k * total_k32 * 4
        output_base = output_tile * Int32(128 * total_k32 * 4)
        k32_count = Int32(4)
        if cutlass.const_expr(self.n64_tail):
            if k128_slice == Int32(total_k32 // 4):
                k32_count = Int32(2)
        for i in cutlass.range_constexpr(4):
            idx = tid + Int32(i * self.threads_per_cta)
            lane = idx & Int32(31)
            tmp = idx >> Int32(5)
            n32 = tmp & Int32(3)
            kb = tmp >> Int32(2)
            dst_addr = dst_base + (idx << Int32(4))
            if kb < k32_count:
                source_word = (
                    Int64(expert_idx) * Int64(expert_words)
                    + Int64(output_base)
                    + Int64(k128_slice * Int32(2048))
                    + Int64(kb * Int32(512))
                    + Int64(n32 * Int32(128))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global_l2hint(
                    dst_addr, get_ptr_as_int64(weights, source_word), l2_policy
                )
            else:
                st_shared_v4_u32(dst_addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def _stage_n64_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
        l2_policy,
        bounds=None,
        value_offset=None,
        base_word=None,
    ):
        total_k32 = self.n // 32
        if cutlass.const_expr(self.csf_inline):
            if tid < Int32(32):
                target = cute.make_ptr(
                    cutlass.Uint8, dst_base, cute.AddressSpace.smem, assumed_align=16
                )
                group = (
                    output_tile * Int32(self.csf_reader.plane.storage.column_groups)
                    + k128_slice
                )
                cached = None
                if cutlass.const_expr(bounds is not None):
                    first, end = Uint32(0), Uint32(0)
                    for chunk in cutlass.range_constexpr(cute.size(bounds, mode=[0])):
                        selected_first = cute.arch.shuffle_sync(
                            bounds[chunk, 0], k128_slice % Int32(32)
                        )
                        selected_end = cute.arch.shuffle_sync(
                            bounds[chunk, 1], k128_slice % Int32(32)
                        )
                        if k128_slice // Int32(32) == Int32(chunk):
                            first, end = selected_first, selected_end
                    cached = (first, end)
                self.csf_reader.stage_compressed(
                    scales.iterator,
                    expert_idx,
                    group,
                    target,
                    tid,
                    cached,
                    value_offset,
                    base_word,
                    coordinates=(Int32(0), output_tile, k128_slice),
                )
            return
        scales_u8 = cute.recast_tensor(scales, cutlass.Uint8)
        expert_bytes = self.k * total_k32
        output_base = output_tile * Int32(128 * total_k32)
        tile_cols = Int32(4)
        if cutlass.const_expr(self.n64_tail):
            if k128_slice == Int32(total_k32 // 4):
                tile_cols = Int32(2)
        source_base = (
            Int64(expert_idx) * Int64(expert_bytes)
            + Int64(output_base)
            + Int64(k128_slice * Int32(512))
        )
        if tile_cols == Int32(4):
            cp_async_u32_shared_global_l2hint(
                dst_base + tid * Int32(4),
                get_ptr_as_int64(scales_u8, source_base + Int64(tid * Int32(4))),
                l2_policy,
            )
        else:
            scales_u16 = cute.recast_tensor(scales, cutlass.Uint16)
            value = scales_u16[source_base // Int64(2) + Int64(tid)].to(Uint32)
            st_shared_u32(dst_base + tid * Int32(4), value)

    @cute.jit
    def _stage_slice(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        my_stage_pair: Int32,
        my_scale_pair: Int32,
        group_count: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        intermediate_slice: Int32,
        l2_policy,
        csf_bounds=None,
        csf_value_offset=None,
        csf_base_word=None,
    ):
        stage = intermediate_slice & Int32(1)
        a_base = smem_base + stage * Int32(self.a_stage_bytes)
        sfa_base = a_base + Int32(self.a_payload_bytes)
        b_base = (
            smem_base + Int32(self.b_storage_offset) + stage * Int32(self.b_stage_bytes)
        )
        sfb_base = (
            smem_base
            + Int32(self.sfb_storage_offset)
            + stage * Int32(self.sfb_stage_bytes)
        )

        words_per_row = Int32(self.intermediate_tiles * 32)
        rows_capacity = Int32(self.max_pairs)
        if tid < Int32(64):
            col_tok = tid >> Int32(3)
            vec = tid & Int32(7)
            if col_tok < group_count:
                phys_vec = vec ^ col_tok
                src_word = Int64(my_stage_pair) * Int64(words_per_row) + Int64(
                    intermediate_slice * Int32(32) + vec * Int32(4)
                )
                cp_async4_shared_global(
                    a_base + (col_tok << Int32(7)) + (phys_vec << Int32(4)),
                    get_ptr_as_int64(intermediate_u32, src_word),
                )
        if tid < group_count:
            sf_src = (
                Int64(rows_capacity) * Int64(words_per_row)
                + Int64(intermediate_slice) * Int64(rows_capacity)
                + Int64(my_scale_pair)
            )
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(intermediate_u32, sf_src),
            )

        if cutlass.const_expr(self.n64_tail):
            self._stage_n64_b(
                down_rp,
                b_base,
                expert_idx,
                output_tile,
                intermediate_slice,
                tid,
                l2_policy,
            )
            self._stage_n64_sfb(
                down_sfb_rp,
                sfb_base,
                expert_idx,
                output_tile,
                intermediate_slice,
                tid,
                l2_policy,
                csf_bounds,
                csf_value_offset,
                csf_base_word,
            )
        else:
            packed_tile = output_tile // Int32(2)
            packed_n32 = (output_tile % Int32(2)) * Int32(4)
            b_tile = (
                expert_idx * Int32(self.packed_output_tiles) + packed_tile
            ) * Int32(self.intermediate_tiles) + intermediate_slice
            b_word_base = Int64(b_tile) * Int64(4096)
            for i in cutlass.range_constexpr(4):
                idx = tid + Int32(i * self.threads_per_cta)
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc % Int32(4)
                kb = kc // Int32(4)
                src_word = (
                    b_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64((packed_n32 + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global_l2hint(
                    b_base + (idx << Int32(4)),
                    get_ptr_as_int64(down_rp, src_word),
                    l2_policy,
                )
            if tid < Int32(32):
                sfb_word_base = Int64(b_tile) * Int64(256)
                src_word = sfb_word_base + Int64(
                    packed_n32 * Int32(32) + tid * Int32(4)
                )
                cp_async4_shared_global_l2hint(
                    sfb_base + (tid << Int32(4)),
                    get_ptr_as_int64(down_sfb_rp, src_word),
                    l2_policy,
                )

    @cute.kernel
    def kernel(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        route_outputs: cute.Tensor,
        token_weights: cute.Tensor,
        coalesce_meta: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        num_groups = coalesce_meta[Int64(0)]
        total_tasks = num_groups * Int32(self.output_tiles)
        task = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.region_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        l2_policy = create_l2_evict_first_policy()
        facc = tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(2))
        words = cute.make_rmem_tensor((4,), Uint32)
        dn_b0 = cute.make_rmem_tensor((4,), Uint32)
        dn_b1 = cute.make_rmem_tensor((4,), Uint32)

        while task < total_tasks:
            group_idx = task // Int32(self.output_tiles)
            output_tile = task % Int32(self.output_tiles)
            self._run_task(
                intermediate_u32,
                down_rp,
                down_sfb_rp,
                route_outputs,
                token_weights,
                coalesce_meta,
                down_alpha,
                global_scale,
                smem_base,
                l2_policy,
                facc,
                words,
                dn_b0,
                dn_b1,
                tid,
                warp_idx,
                lane,
                q,
                c,
                group_idx,
                output_tile,
            )
            task += Int32(gdimx)

    @cute.jit
    def _run_task(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        route_outputs: cute.Tensor,
        token_weights: cute.Tensor,
        coalesce_meta: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        smem_base: Int32,
        l2_policy,
        facc,
        words: cute.Tensor,
        dn_b0: cute.Tensor,
        dn_b1: cute.Tensor,
        tid: Int32,
        warp_idx: Int32,
        lane: Int32,
        q: Int32,
        c: Int32,
        group_idx: Int32,
        output_tile: Int32,
    ):
        expert_idx = coalesce_meta[Int64(1) + Int64(group_idx)]
        group_count = coalesce_meta[Int64(1 + self.max_pairs) + Int64(group_idx)]
        meta_pair_base = Int64(1 + 2 * self.max_pairs) + Int64(group_idx * Int32(8))
        my_stage_pair = Int32(0)
        if tid < Int32(64):
            my_stage_pair = coalesce_meta[meta_pair_base + Int64(tid >> Int32(3))]
        my_scale_pair = Int32(0)
        if tid < Int32(8):
            my_scale_pair = coalesce_meta[meta_pair_base + Int64(tid)]

        q_in = q
        if q_in >= group_count:
            q_in = Int32(0)

        csf_bounds, csf_base_word, csf_value_offset = None, None, None
        if cutlass.const_expr(self.csf_inline and self.n64_tail):
            csf_value_offset = self.csf_reader.plane.group_value_offset(
                down_sfb_rp.iterator
            )
            csf_bounds = self.csf_reader.prefetch_column_bounds(
                down_sfb_rp.iterator, expert_idx, output_tile
            )
            csf_base_word = self.csf_reader.prefetch_row_bases(
                down_sfb_rp.iterator, expert_idx, output_tile, tid
            )

        self._stage_slice(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            smem_base,
            tid,
            my_stage_pair,
            my_scale_pair,
            group_count,
            expert_idx,
            output_tile,
            Int32(0),
            l2_policy,
            csf_bounds,
            csf_value_offset,
            csf_base_word,
        )
        cute.arch.cp_async_commit_group()

        for nt in cutlass.range_constexpr(2):
            facc[nt].fill(0.0)

        intermediate_slice = Int32(0)
        while intermediate_slice < Int32(self.intermediate_tiles):
            stage = intermediate_slice & Int32(1)
            a_base = smem_base + stage * Int32(self.a_stage_bytes)
            sfa_base = a_base + Int32(self.a_payload_bytes)
            b_base = (
                smem_base
                + Int32(self.b_storage_offset)
                + stage * Int32(self.b_stage_bytes)
            )
            sfb_base = (
                smem_base
                + Int32(self.sfb_storage_offset)
                + stage * Int32(self.sfb_stage_bytes)
            )
            next_slice = intermediate_slice + Int32(1)
            if next_slice < Int32(self.intermediate_tiles):
                self._stage_slice(
                    intermediate_u32,
                    down_rp,
                    down_sfb_rp,
                    smem_base,
                    tid,
                    my_stage_pair,
                    my_scale_pair,
                    group_count,
                    expert_idx,
                    output_tile,
                    next_slice,
                    l2_policy,
                    csf_bounds,
                    csf_value_offset,
                    csf_base_word,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            sfb_words = sfb_base
            if cutlass.const_expr(self.csf_inline):
                sfb_words = smem_base + Int32(self.shared_words * 4)
                self.csf_reader.expand_warp_rows(
                    down_sfb_rp.iterator, sfb_base, sfb_words, warp_idx, lane
                )
            asc = ld_shared_u32(sfa_base + (q_in << Int32(2)))

            for kb in cutlass.range_constexpr(4):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q_in
                a_lo = (
                    a_base
                    + (q_in << Int32(7))
                    + (u_phys << Int32(4))
                    + ((c & Int32(1)) << Int32(3))
                )
                a0, a2 = ld_shared_v2_u32(a_lo)
                w0, w1, w2, w3 = ld_shared_v4_u32(
                    b_base
                    + (((Int32(kb * 4) + warp_idx) * Int32(32) + lane) << Int32(4))
                )
                words[0] = w0
                words[1] = w1
                words[2] = w2
                words[3] = w3
                for nt in cutlass.range_constexpr(4):
                    wb0, wb1 = e2m1x8_to_qmma_e2m1x8(words[nt])
                    dn_b0[nt] = wb0
                    dn_b1[nt] = wb1
                for nt in cutlass.range_constexpr(2):
                    n8 = warp_idx * Int32(4) + Int32(nt * 2) + (lane & Int32(1))
                    sfb_word = ld_shared_u32(
                        sfb_words + ((n8 * Int32(8) + q) << Int32(2))
                    )
                    fragment = facc[nt]
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1_e4m3(
                        fragment[0],
                        fragment[1],
                        fragment[2],
                        fragment[3],
                        dn_b0[nt * 2],
                        dn_b0[nt * 2 + 1],
                        dn_b1[nt * 2],
                        dn_b1[nt * 2 + 1],
                        a0,
                        a2,
                        sfb_word,
                        asc,
                        bid_a=kb,
                        bid_b=kb,
                    )
                    fragment[0] = d0
                    fragment[1] = d1
                    fragment[2] = d2
                    fragment[3] = d3
            cute.arch.sync_threads()
            intermediate_slice += Int32(1)

        cute.arch.cp_async_wait_group(0)
        down_scale = down_alpha[expert_idx].to(cutlass.Float32) * global_scale[
            0 if cutlass.const_expr(global_scale.shape[0] == 1) else expert_idx
        ].to(cutlass.Float32)
        for nt in cutlass.range_constexpr(2):
            fragment = facc[nt]
            peer = (
                cute.arch.shuffle_sync_bfly(fragment[0], offset=4),
                cute.arch.shuffle_sync_bfly(fragment[1], offset=4),
                cute.arch.shuffle_sync_bfly(fragment[2], offset=4),
                cute.arch.shuffle_sync_bfly(fragment[3], offset=4),
            )
            if (q & Int32(1)) == Int32(0):
                for row_part in cutlass.range_constexpr(2):
                    col_idx = c * Int32(2) + Int32(row_part)
                    if col_idx < group_count:
                        pair = coalesce_meta[meta_pair_base + Int64(col_idx)]
                        scale = down_scale * token_weights[Int64(pair)].to(
                            cutlass.Float32
                        )
                        for half in cutlass.range_constexpr(2):
                            out_col = (
                                output_tile * Int32(128)
                                + warp_idx * Int32(32)
                                + Int32(nt * 16 + half * 8)
                                + q
                            )
                            st_global_u32(
                                get_ptr_as_int64(
                                    route_outputs,
                                    Int64(pair) * Int64(self.k) + Int64(out_col),
                                ),
                                pack_f32x2_to_bfloat2(
                                    scale * fragment[half * 2 + row_part],
                                    scale * peer[half * 2 + row_part],
                                ),
                            )


class W4A8CompactTopkSumKernel:
    """Deterministic slot-ordered (0..top_k-1) FP32 reduction into BF16 output."""

    threads_per_cta = 128

    def __init__(self, k: int, num_topk: int, num_experts: int):
        if k <= 0 or k % 256:
            raise ValueError("compact topk sum requires K divisible by 256")
        if num_topk <= 0 or num_experts <= 0:
            raise ValueError("num_topk and num_experts must be positive")
        self.k = int(k)
        self.num_topk = int(num_topk)
        self.num_experts = int(num_experts)
        self.tiles_per_token = self.k // 256

    @cute.jit
    def __call__(
        self,
        route_outputs: cute.Tensor,
        topk_ids: cute.Tensor,
        output: cute.Tensor,
        num_pairs: Int32,
        stream: cuda.CUstream,
    ):
        live_tokens = num_pairs // Int32(self.num_topk)
        route_u32 = cute.recast_tensor(route_outputs, Uint32)
        self.kernel(route_u32, topk_ids, output, live_tokens).launch(
            grid=(live_tokens * Int32(self.tiles_per_token), 1, 1),
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        route_u32: cute.Tensor,
        topk_ids: cute.Tensor,
        output: cute.Tensor,
        live_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        bid = Int32(bidx)
        token = bid // Int32(self.tiles_per_token)
        if token < live_tokens:
            tile = bid % Int32(self.tiles_per_token)
            col = tile * Int32(256) + tid * Int32(2)
            col_u32 = col >> Int32(1)
            row_u32_stride = Int64(self.k // 2)
            acc0 = cutlass.Float32(0.0)
            acc1 = cutlass.Float32(0.0)
            base_pair = token * Int32(self.num_topk)
            for slot in cutlass.range_constexpr(self.num_topk):
                pair = base_pair + Int32(slot)
                expert = local_expert_id(topk_ids[Int64(pair)], Int32(self.num_experts))
                if expert >= Int32(0):
                    word = route_u32[Int64(pair) * row_u32_stride + Int64(col_u32)]
                    acc0 = acc0 + u32_as_f32((word & Uint32(0xFFFF)) << Uint32(16))
                    acc1 = acc1 + u32_as_f32(word & Uint32(0xFFFF0000))
            st_global_u32(
                get_ptr_as_int64(output, Int64(token) * Int64(self.k) + Int64(col)),
                pack_f32x2_to_bfloat2(acc0, acc1),
            )


__all__ = [
    "W4A8CompactMicroProjectionKernel",
    "W4A8CompactMicroProjectionActivationKernel",
    "W4A8CompactCoalesceRoutesKernel",
    "W4A8CompactCoalescedFC1Kernel",
    "W4A8CompactCoalescedFC2Kernel",
    "W4A8CompactTopkSumKernel",
]
