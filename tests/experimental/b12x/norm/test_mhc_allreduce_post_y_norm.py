"""CUDA tests for fused Lamport-sentinel ClusterMhcAllReducePostYNorm.

Includes:
1. True single-GPU multi-rank Lamport buffer simulation (runs on 1x SM120 GPU such as
   RTX 5090): pre-populates peer ranks' sentinel-encoded shards using the exact
   ``st.relaxed.sys.global.v4.u32`` wire protocol, executes ``ClusterMhcAllReducePostYNorm``
   live on GPU, and asserts 0-ULP bit-exact parity against both:
   - standalone ``OneshotAllReduce`` (``remote_push_full``) + ``ClusterMhcPostYNorm``, and
   - canonical rank-order FP32 summation rounded to BF16 + ``ClusterMhcPostYNorm``,
   covering M in {1, 2, 4, 7, 8, 9, 12, 16, 24, 32}, H in {4096, 5120}, world_size in
   {2, 4, 8}, with/without input_addend, BF16/FP32 norm weights, MXFP8 prequant,
   SM120 PDL ``release_dependents``, adversarial -0.0/+0.0/extreme-exponent inputs,
   Lamport slot zero-clear & epoch verification, and 50-iter zero-alloc CUDA-graph replay.
2. Live multi-GPU ``B12X_RUN_PCIE_ONESHOT_TEST=1`` verification over a real
   ``PCIeOneshotAllReducePool`` channel (for 4-GPU SM120 PCIe hosts).
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cutlass import BFloat16, Int32, Int64, Uint32

from b12x._lib.compile_plan import load_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics.memory import ld_global_v4_u32
from b12x._lib.program_cache import program_cache
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.utils.device import cuda_stream_from_int_or_current, current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.comm.pcie._cute_intrinsics import (
    add_bf16x2,
    st_relaxed_sys_global_v4_u32,
)
from b12x.comm.pcie._oneshot_cute import (
    _MAX_RANKS,
    _PLAIN_GRAPH_ARRIVED_OFFSET,
    _PLAIN_GRAPH_EPOCH_OFFSET,
    get_oneshot_launcher,
)
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool, _CuTeOneshotBackend
from b12x.norm.mhc._split_lagged import (
    CLUSTER_AR_MHC_MAX_TOKENS,
    _encode_bf16_sentinel_word,
    compile_cluster_allreduce_post_y_norm,
    compile_cluster_post_y_norm,
    oneshot_channel_sentinel_params,
    split_lagged_cluster_size,
    supports_cluster_allreduce_post_y_norm,
)


def _sm120_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120+ required for ClusterMhcAllReducePostYNorm")


class _PeerSentinelPushKernel:
    """Test helper that publishes peer rank ``src_rank``'s sentinel packs into rank 0's slot."""

    def __init__(self, has_addend: bool):
        self.has_addend = bool(has_addend)

    @cute.jit
    def __call__(
        self,
        dst_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        addend_ptr: cute.Pointer,
        size_packs: Int32,
        stream: cuda.CUstream,
    ):
        threads = 256
        blocks = (size_packs + Int32(threads - 1)) // Int32(threads)
        self.kernel(dst_ptr, x_ptr, addend_ptr, size_packs).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dst_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        addend_ptr: cute.Pointer,
        size_packs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = Int32(bidx) * Int32(256) + Int32(tidx)
        if idx < size_packs:
            off = Int64(idx) * Int64(16)
            words = ld_global_v4_u32(Int64(x_ptr.toint()) + off)
            if cutlass.const_expr(self.has_addend):
                extra = ld_global_v4_u32(Int64(addend_ptr.toint()) + off)
                e0 = _encode_bf16_sentinel_word(add_bf16x2(words[0], extra[0]))
                e1 = _encode_bf16_sentinel_word(add_bf16x2(words[1], extra[1]))
                e2 = _encode_bf16_sentinel_word(add_bf16x2(words[2], extra[2]))
                e3 = _encode_bf16_sentinel_word(add_bf16x2(words[3], extra[3]))
            else:
                e0 = _encode_bf16_sentinel_word(words[0])
                e1 = _encode_bf16_sentinel_word(words[1])
                e2 = _encode_bf16_sentinel_word(words[2])
                e3 = _encode_bf16_sentinel_word(words[3])
            st_relaxed_sys_global_v4_u32(Int64(dst_ptr.toint()) + off, e0, e1, e2, e3)


@program_cache
def _compile_peer_sentinel_push(ordinal: int, has_addend: bool):
    fake = (
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            _PeerSentinelPushKernel(has_addend),
            *fake,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "test.mhc.peer_sentinel_push", 1, (bool(has_addend),)
            ),
        )

    def run(
        dst_addr: int, x: torch.Tensor, addend: torch.Tensor | None = None, stream=None
    ):
        size_packs = (x.numel() * x.element_size()) // 16
        add_t = addend if (has_addend and addend is not None) else x
        stream_int = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
        run_compiled(
            raw,
            (
                make_ptr(
                    Uint32, int(dst_addr), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    BFloat16, add_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                Int32(size_packs),
                cuda_stream_from_int_or_current(stream_int),
            ),
        )

    return run


class _SimulatedLamportRing:
    """Single-GPU multi-rank Lamport buffer harness matching PCIeOneshotAllReduce layout."""

    def __init__(
        self,
        *,
        world_size: int,
        shard_packs: int,
        region_packs: int,
        device: torch.device,
    ):
        self.world_size = int(world_size)
        self.shard_packs = int(shard_packs)
        self.region_packs = int(region_packs)
        self.device = device
        # Two ping-pong slots; each rank has [region_packs + world_size * shard_packs] x 16B.
        slot_packs_per_rank = self.region_packs + self.world_size * self.shard_packs
        self.slots = [
            [
                torch.zeros(slot_packs_per_rank * 4, dtype=torch.int32, device=device)
                for _ in range(self.world_size)
            ]
            for _ in range(2)
        ]
        # Pointer table: 2 * _MAX_RANKS uint64 entries (slot 0 then slot 1).
        peer_ptrs = [0] * (2 * _MAX_RANKS)
        for s in range(2):
            for r in range(self.world_size):
                peer_ptrs[s * _MAX_RANKS + r] = self.slots[s][r].data_ptr()
        self.peer_table = torch.tensor(peer_ptrs, dtype=torch.int64, device=device)

        # Signal buffers (256 KiB each, matching _CuTeOneshotBackend.meta_size()).
        signal_bytes = _CuTeOneshotBackend.meta_size()
        self.signals = [
            torch.zeros(signal_bytes, dtype=torch.uint8, device=device)
            for _ in range(self.world_size)
        ]
        sig_ptrs = [0] * _MAX_RANKS
        for r in range(self.world_size):
            sig_ptrs[r] = self.signals[r].data_ptr()
        self.signal_table = torch.tensor(sig_ptrs, dtype=torch.int64, device=device)

    @property
    def peer_table_address(self) -> int:
        return int(self.peer_table.data_ptr())

    @property
    def signal_table_address(self) -> int:
        return int(self.signal_table.data_ptr())

    def slot_table_address(self, slot: int) -> int:
        return int(self.peer_table.data_ptr()) + (slot & 1) * _MAX_RANKS * 8

    def shard_address(self, slot: int, dst_rank: int, src_rank: int) -> int:
        base = self.slots[slot & 1][dst_rank].data_ptr()
        return base + (self.region_packs + src_rank * self.shard_packs) * 16

    def reset_signals(self) -> None:
        for s in self.signals:
            s.zero_()

    def reset_slots(self) -> None:
        for slot_bufs in self.slots:
            for buf in slot_bufs:
                buf.zero_()

    def epoch_and_arrived(self, rank: int) -> tuple[int, int]:
        sig = self.signals[rank]
        epoch = int(
            sig[_PLAIN_GRAPH_EPOCH_OFFSET : _PLAIN_GRAPH_EPOCH_OFFSET + 4]
            .view(torch.int32)
            .item()
        )
        arrived = int(
            sig[_PLAIN_GRAPH_ARRIVED_OFFSET : _PLAIN_GRAPH_ARRIVED_OFFSET + 4]
            .view(torch.int32)
            .item()
        )
        return epoch, arrived

    def populate_remote_peers(
        self,
        *,
        slot: int,
        target_rank: int,
        peer_xs: list[torch.Tensor],
        peer_addends: list[torch.Tensor] | None = None,
        stream=None,
    ) -> None:
        push = _compile_peer_sentinel_push(
            int(self.device.index or 0), peer_addends is not None
        )
        for src in range(self.world_size):
            if src == target_rank:
                continue
            dst_addr = self.shard_address(slot, target_rank, src)
            add = None if peer_addends is None else peer_addends[src]
            push(dst_addr, peer_xs[src], add, stream=stream)


def _canonical_bf16_sum(
    peer_xs: list[torch.Tensor],
    peer_addends: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Exact canonical rank-order FP32 sum rounded to BF16 matching sentinel encoding."""
    acc = None
    for r, x in enumerate(peer_xs):
        term = x if peer_addends is None else (x + peer_addends[r])
        # Match Lamport sentinel encoding: +0.0 (0x0000) -> -0.0 (0x8000) before FP32 unpack.
        bits = term.view(torch.uint16)
        encoded = torch.where(
            bits == 0, torch.tensor(0x8000, dtype=torch.uint16, device=x.device), bits
        )
        f32 = encoded.view(torch.bfloat16).float()
        acc = f32 if acc is None else (acc + f32)
    return acc.to(torch.bfloat16)


def _make_mhc_tensors(
    tokens: int,
    hidden: int,
    world_size: int,
    norm_dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    peer_xs = [
        (
            torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.5
        ).contiguous()
        for _ in range(world_size)
    ]
    peer_addends = [
        (
            torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.25
        ).contiguous()
        for _ in range(world_size)
    ]
    residual = (
        torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device) * 0.5
    ).contiguous()
    prev_post = torch.rand(tokens, 4, dtype=torch.float32, device=device).contiguous()
    prev_comb = torch.softmax(
        torch.randn(tokens, 4, 4, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    incoming = torch.softmax(
        torch.randn(tokens, 4, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    norm_weight = (
        torch.randn(hidden, dtype=norm_dtype, device=device) * 0.2 + 1.0
    ).contiguous()
    return peer_xs, peer_addends, residual, prev_post, prev_comb, incoming, norm_weight


@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("tokens", [1, 2, 4, 7, 8, 9, 12, 16, 24, 32])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("has_addend", [False, True])
def test_cluster_allreduce_post_y_norm_bit_exact_simulation(
    hidden: int, tokens: int, world_size: int, has_addend: bool
):
    _sm120_required()
    # Keep full matrix on DeepSeek-V4.1 (H=5120, TP4) and representative coverage on H=4096 / TP2 / TP8.
    if hidden == 4096 and (tokens not in (1, 4, 8, 16, 32) or world_size == 8):
        pytest.skip("reduced matrix for H=4096")
    if world_size == 8 and (hidden != 5120 or tokens not in (1, 8, 16, 32)):
        pytest.skip("reduced matrix for TP8")
    if world_size == 2 and tokens not in (1, 2, 4, 8, 16, 32):
        pytest.skip("reduced matrix for TP2")

    device = torch.device("cuda", torch.cuda.current_device())
    ordinal = int(device.index or 0)
    assert CLUSTER_AR_MHC_MAX_TOKENS == 32
    assert supports_cluster_allreduce_post_y_norm(
        device, tokens, hidden, torch.bfloat16, world_size
    )
    assert not supports_cluster_allreduce_post_y_norm(
        device, CLUSTER_AR_MHC_MAX_TOKENS + 1, hidden, torch.bfloat16, world_size
    )

    norm_dtype = (
        torch.float32
        if (hidden == 5120 and tokens in (7, 16) and world_size == 4)
        else torch.bfloat16
    )
    emit_mxfp8 = True
    release_dependents = tokens in (1, 8, 16, 32)

    peer_xs, peer_addends, residual, prev_post, prev_comb, incoming, norm_weight = (
        _make_mhc_tensors(
            tokens,
            hidden,
            world_size,
            norm_dtype,
            device,
            seed=2026 + hidden + tokens * 17 + world_size,
        )
    )
    addends = peer_addends if has_addend else None

    shard_packs = tokens * (hidden // 8) + 64
    region_packs = 128
    ring_fused = _SimulatedLamportRing(
        world_size=world_size,
        shard_packs=shard_packs,
        region_packs=region_packs,
        device=device,
    )
    ring_ref = _SimulatedLamportRing(
        world_size=world_size,
        shard_packs=shard_packs,
        region_packs=region_packs,
        device=device,
    )

    # Compile fused kernel (rank 0) and standalone OneshotAllReduce + ClusterMhcPostYNorm.
    fused_runner = compile_cluster_allreduce_post_y_norm(
        ordinal,
        hidden,
        world_size,
        0,
        1e-20,
        norm_dtype,
        emit_mxfp8=emit_mxfp8,
        has_addend=has_addend,
        device_slot_selection=True,
        slot_bias=0,
        release_dependents=release_dependents,
    )
    post_y_runner = compile_cluster_post_y_norm(
        ordinal,
        hidden,
        False,
        1e-20,
        norm_dtype,
        cluster_size=split_lagged_cluster_size(tokens),
        emit_mxfp8=emit_mxfp8,
    )
    ar_launcher = get_oneshot_launcher(
        "bfloat16",
        world_size,
        0,
        True,
        True,
        0,
        "remote_push_full",
        256,
        ordinal,
        has_addend=has_addend,
    )
    load_programs(fused_runner)
    load_programs(post_y_runner)

    # Pre-populate peer ranks 1..W-1 into slot 0 of both rings.
    ring_fused.populate_remote_peers(
        slot=0, target_rank=0, peer_xs=peer_xs, peer_addends=addends
    )
    ring_ref.populate_remote_peers(
        slot=0, target_rank=0, peer_xs=peer_xs, peer_addends=addends
    )

    # Path 1 (Reference): standalone OneshotAllReduce(remote_push_full) -> ClusterMhcPostYNorm.
    ar_out = torch.empty_like(peer_xs[0])
    ar_kwargs = {"addend_address": peer_addends[0].data_ptr()} if has_addend else {}
    ar_launcher(
        ring_ref.peer_table_address,
        ring_ref.signal_table_address,
        peer_xs[0].data_ptr(),
        ar_out.data_ptr(),
        tokens * (hidden // 8),
        region_packs,
        shard_packs,
        0,
        0,
        0,
        0,
        16,
        **ar_kwargs,
    )
    out_ref = torch.empty_like(residual)
    y_ref = torch.empty_like(peer_xs[0])
    av_ref = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
    as_ref = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)
    post_y_runner(
        ar_out,
        residual,
        prev_post,
        prev_comb,
        incoming,
        norm_weight,
        out_ref,
        y_ref,
        a_values=av_ref,
        a_scales=as_ref,
    )

    # Path 2 (Fused): ClusterMhcAllReducePostYNorm.
    out_fused = torch.empty_like(residual)
    y_fused = torch.empty_like(peer_xs[0])
    av_fused = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
    as_fused = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)
    fused_runner(
        ring_fused.peer_table_address,
        ring_fused.signal_table_address,
        peer_xs[0],
        residual,
        prev_post,
        prev_comb,
        incoming,
        norm_weight,
        out_fused,
        y_fused,
        shard_packs=shard_packs,
        payload_ready_region_packs=region_packs,
        input_addend=peer_addends[0] if has_addend else None,
        a_values=av_fused,
        a_scales=as_fused,
    )
    torch.cuda.synchronize()

    # 1. Exact 0-ULP bit-for-bit equality on ALL outputs (raw uint16 / uint8 bits!).
    assert torch.equal(out_fused.view(torch.uint16), out_ref.view(torch.uint16))
    assert torch.equal(y_fused.view(torch.uint16), y_ref.view(torch.uint16))
    assert torch.equal(av_fused, av_ref)
    assert torch.equal(as_fused, as_ref)

    # 2. Verify Lamport slots on rank 0 were completely zero-cleared and peers received exact sentinel packs!
    assert torch.equal(ring_fused.slots[0][0], ring_ref.slots[0][0])
    assert int(ring_fused.slots[0][0].any().item()) == 0
    for peer in range(1, world_size):
        assert torch.equal(ring_fused.slots[0][peer], ring_ref.slots[0][peer])

    # 3. Verify epoch advanced identically (epoch == 1, arrived == 0).
    assert ring_fused.epoch_and_arrived(0) == (1, 0)
    assert ring_ref.epoch_and_arrived(0) == (1, 0)

    # 4. Also verify non-zero rank (e.g. rank = world_size - 1) produces identical canonical 0-ULP output!
    if world_size == 4 and tokens in (1, 8, 16) and hidden == 5120:
        r_test = world_size - 1
        ring_r = _SimulatedLamportRing(
            world_size=world_size,
            shard_packs=shard_packs,
            region_packs=region_packs,
            device=device,
        )
        ring_r.populate_remote_peers(
            slot=0, target_rank=r_test, peer_xs=peer_xs, peer_addends=addends
        )
        fused_r = compile_cluster_allreduce_post_y_norm(
            ordinal,
            hidden,
            world_size,
            r_test,
            1e-20,
            norm_dtype,
            emit_mxfp8=emit_mxfp8,
            has_addend=has_addend,
            device_slot_selection=True,
            slot_bias=0,
            release_dependents=release_dependents,
        )
        load_programs(fused_r)
        out_r = torch.empty_like(residual)
        y_r = torch.empty_like(peer_xs[0])
        av_r = torch.empty_like(av_fused)
        as_r = torch.empty_like(as_fused)
        fused_r(
            ring_r.peer_table_address,
            ring_r.signal_table_address,
            peer_xs[r_test],
            residual,
            prev_post,
            prev_comb,
            incoming,
            norm_weight,
            out_r,
            y_r,
            shard_packs=shard_packs,
            payload_ready_region_packs=region_packs,
            input_addend=peer_addends[r_test] if has_addend else None,
            a_values=av_r,
            a_scales=as_r,
        )
        torch.cuda.synchronize()
        # Because summation is in canonical rank order 0..W-1, rank W-1 matches rank 0 bit-for-bit!
        assert torch.equal(out_r.view(torch.uint16), out_fused.view(torch.uint16))
        assert torch.equal(y_r.view(torch.uint16), y_fused.view(torch.uint16))
        assert torch.equal(av_r, av_fused)
        assert torch.equal(as_r, as_fused)
        assert int(ring_r.slots[0][r_test].any().item()) == 0
        assert ring_r.epoch_and_arrived(r_test) == (1, 0)


@pytest.mark.parametrize("has_addend", [False, True])
def test_cluster_allreduce_post_y_norm_adversarial_zero_and_exponents(has_addend: bool):
    """Test +0.0, -0.0, subnormals, extreme finite exponents, and mixed cancellation."""
    _sm120_required()
    device = torch.device("cuda", torch.cuda.current_device())
    ordinal = int(device.index or 0)
    tokens, hidden, world_size = 8, 5120, 4

    # Construct adversarial BF16 bit patterns covering +0.0 (0x0000), -0.0 (0x8000),
    # smallest subnormal (0x0001 / 0x8001), tiny normal (0x0080), cancellation pairs,
    # and large exponents (e.g. ~1e18 so squares don't overflow FP32).
    special_bits = torch.tensor(
        [
            0x0000,  # +0.0
            0x8000,  # -0.0
            0x0001,  # +subnormal
            0x8001,  # -subnormal
            0x0080,  # +min normal
            0x3F80,  # +1.0
            0xBF80,  # -1.0
            0x5500,  # large positive (~2^43, sum-of-squares stays finite in FP32)
            0xD500,  # large negative (~-2^43)
            0x3C00,  # +0.0078125
            0xBC00,  # -0.0078125
            0x42C8,  # +100.0
        ],
        dtype=torch.int32,
        device=device,
    )
    total = tokens * hidden
    idx = torch.arange(total, device=device)
    peer_xs = []
    for r in range(world_size):
        # Row 0: ALL +0.0 across all ranks (exercises pure sentinel encoding on every lane!).
        # Row 1: ALL -0.0 across all ranks.
        # Row 2: mix of +0.0 on some ranks and -0.0 on others.
        # Row 3: exact cancellation (+1.0, -1.0, +0.0, -0.0).
        # Rows 4..7: rotating special bit patterns across ranks.
        bits = (
            special_bits[(idx + r * 3) % special_bits.numel()]
            .reshape(tokens, hidden)
            .to(torch.uint16)
            .clone()
        )
        bits[0].fill_(0x0000)
        bits[1].fill_(0x8000)
        bits[2].fill_(0x0000 if (r % 2 == 0) else 0x8000)
        bits[3].fill_([0x3F80, 0xBF80, 0x0000, 0x8000][r])
        peer_xs.append(bits.view(torch.bfloat16).contiguous())

    peer_addends = None
    if has_addend:
        peer_addends = []
        for r in range(world_size):
            abits = (
                special_bits[(idx + r * 5 + 1) % special_bits.numel()]
                .reshape(tokens, hidden)
                .to(torch.uint16)
                .clone()
            )
            abits[0].fill_(0x0000)
            abits[1].fill_(0x8000)
            abits[2].fill_(0x0000)
            abits[3].fill_(0x8000 if r % 2 == 0 else 0x0000)
            peer_addends.append(abits.view(torch.bfloat16).contiguous())

    _, _, residual, prev_post, prev_comb, incoming, norm_weight = _make_mhc_tensors(
        tokens, hidden, world_size, torch.bfloat16, device, seed=999
    )
    # Also zero out row 0 of residual to verify +0.0 / -0.0 through RMSNorm + MXFP8 zero-block scale=127.
    residual[0].zero_()

    shard_packs = tokens * (hidden // 8)
    region_packs = 64
    ring_fused = _SimulatedLamportRing(
        world_size=world_size,
        shard_packs=shard_packs,
        region_packs=region_packs,
        device=device,
    )
    ring_ref = _SimulatedLamportRing(
        world_size=world_size,
        shard_packs=shard_packs,
        region_packs=region_packs,
        device=device,
    )
    ring_fused.populate_remote_peers(
        slot=0, target_rank=0, peer_xs=peer_xs, peer_addends=peer_addends
    )
    ring_ref.populate_remote_peers(
        slot=0, target_rank=0, peer_xs=peer_xs, peer_addends=peer_addends
    )

    fused_runner = compile_cluster_allreduce_post_y_norm(
        ordinal,
        hidden,
        world_size,
        0,
        1e-20,
        torch.bfloat16,
        emit_mxfp8=True,
        has_addend=has_addend,
        device_slot_selection=True,
        slot_bias=0,
    )
    post_y_runner = compile_cluster_post_y_norm(
        ordinal, hidden, False, 1e-20, torch.bfloat16, cluster_size=8, emit_mxfp8=True
    )
    ar_launcher = get_oneshot_launcher(
        "bfloat16",
        world_size,
        0,
        True,
        True,
        0,
        "remote_push_full",
        256,
        ordinal,
        has_addend=has_addend,
    )
    load_programs(fused_runner)
    load_programs(post_y_runner)

    ar_out = torch.empty_like(peer_xs[0])
    ar_kwargs = {"addend_address": peer_addends[0].data_ptr()} if has_addend else {}
    ar_launcher(
        ring_ref.peer_table_address,
        ring_ref.signal_table_address,
        peer_xs[0].data_ptr(),
        ar_out.data_ptr(),
        tokens * (hidden // 8),
        region_packs,
        shard_packs,
        0,
        0,
        0,
        0,
        16,
        **ar_kwargs,
    )
    out_ref, y_ref = torch.empty_like(residual), torch.empty_like(peer_xs[0])
    av_ref = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
    as_ref = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)
    post_y_runner(
        ar_out,
        residual,
        prev_post,
        prev_comb,
        incoming,
        norm_weight,
        out_ref,
        y_ref,
        av_ref,
        as_ref,
    )

    out_fused, y_fused = torch.empty_like(residual), torch.empty_like(peer_xs[0])
    av_fused = torch.empty_like(av_ref)
    as_fused = torch.empty_like(as_ref)
    fused_runner(
        ring_fused.peer_table_address,
        ring_fused.signal_table_address,
        peer_xs[0],
        residual,
        prev_post,
        prev_comb,
        incoming,
        norm_weight,
        out_fused,
        y_fused,
        shard_packs=shard_packs,
        payload_ready_region_packs=region_packs,
        input_addend=peer_addends[0] if has_addend else None,
        a_values=av_fused,
        a_scales=as_fused,
    )
    torch.cuda.synchronize()

    assert torch.equal(out_fused.view(torch.uint16), out_ref.view(torch.uint16))
    assert torch.equal(y_fused.view(torch.uint16), y_ref.view(torch.uint16))
    assert torch.equal(av_fused, av_ref)
    assert torch.equal(as_fused, as_ref)
    assert int(ring_fused.slots[0][0].any().item()) == 0

    # Also cross-check MXFP8 output directly against quantize_mxfp8_rows_cute.
    # (On all-zero blocks, ClusterMhcPostYNorm sets UE8M0 scale_byte=127 while
    # quantize_mxfp8_rows_cute leaves 0; both decode to exact 0.0, and as_fused == as_ref above.)
    av_q = torch.empty_like(av_fused)
    as_q = torch.empty_like(as_fused)
    am_q = torch.empty(
        ((tokens + 127) // 128 * ((hidden + 127) // 128) * 512,),
        dtype=torch.uint8,
        device=device,
    )
    quantize_mxfp8_rows_cute(y_fused, av_q, as_q, am_q, expected_m=tokens)
    torch.cuda.synchronize()
    assert torch.equal(av_fused, av_q)
    assert torch.equal(as_fused[1:], as_q[1:])
    assert torch.all(as_fused[0] == 127)


@pytest.mark.parametrize("tokens", [1, 8, 16, 32])
def test_cluster_allreduce_post_y_norm_50_iter_zero_alloc_cuda_graph(tokens: int):
    """Verify 50-iteration zero-allocation CUDA-graph replay alternating ping-pong slots."""
    _sm120_required()
    device = torch.device("cuda", torch.cuda.current_device())
    ordinal = int(device.index or 0)
    hidden, world_size = 5120, 4

    shard_packs = tokens * (hidden // 8)
    region_packs = 64
    ring = _SimulatedLamportRing(
        world_size=world_size,
        shard_packs=shard_packs,
        region_packs=region_packs,
        device=device,
    )

    fused_runner = compile_cluster_allreduce_post_y_norm(
        ordinal,
        hidden,
        world_size,
        0,
        1e-20,
        torch.bfloat16,
        emit_mxfp8=True,
        has_addend=True,
        device_slot_selection=True,
        slot_bias=0,
        release_dependents=True,
    )
    post_y_runner = compile_cluster_post_y_norm(
        ordinal,
        hidden,
        False,
        1e-20,
        torch.bfloat16,
        cluster_size=split_lagged_cluster_size(tokens),
        emit_mxfp8=True,
    )
    load_programs(fused_runner)
    load_programs(post_y_runner)
    # Pre-compile peer push helper before graph capture.
    _compile_peer_sentinel_push(ordinal, True)

    peer_xs, peer_addends, residual, prev_post, prev_comb, incoming, norm_weight = (
        _make_mhc_tensors(
            tokens, hidden, world_size, torch.bfloat16, device, seed=555 + tokens
        )
    )
    out = torch.empty_like(residual)
    y = torch.empty_like(peer_xs[0])
    a_values = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
    a_scales = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)

    stream = torch.cuda.Stream(device=device)
    # Capture a 2-step ping-pong graph: step 0 uses slot 0, step 1 uses slot 1!
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # Iteration A (slot 0)
        ring.populate_remote_peers(
            slot=0,
            target_rank=0,
            peer_xs=peer_xs,
            peer_addends=peer_addends,
            stream=stream,
        )
        fused_runner(
            ring.peer_table_address,
            ring.signal_table_address,
            peer_xs[0],
            residual,
            prev_post,
            prev_comb,
            incoming,
            norm_weight,
            out,
            y,
            shard_packs=shard_packs,
            payload_ready_region_packs=region_packs,
            input_addend=peer_addends[0],
            a_values=a_values,
            a_scales=a_scales,
            stream=stream,
        )
        # Iteration B (slot 1)
        ring.populate_remote_peers(
            slot=1,
            target_rank=0,
            peer_xs=peer_xs,
            peer_addends=peer_addends,
            stream=stream,
        )
        fused_runner(
            ring.peer_table_address,
            ring.signal_table_address,
            peer_xs[0],
            residual,
            prev_post,
            prev_comb,
            incoming,
            norm_weight,
            out,
            y,
            shard_packs=shard_packs,
            payload_ready_region_packs=region_packs,
            input_addend=peer_addends[0],
            a_values=a_values,
            a_scales=a_scales,
            stream=stream,
        )

    torch.cuda.synchronize()
    # Warm 1 replay and record expected allocation counter.
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_stats(device)["allocation.all.allocated"]

    # 25 replays x 2 steps/replay = 50 fused iterations!
    for it in range(25):
        if it in (0, 12, 24):
            new_xs, new_adds, new_res, new_pp, new_pc, new_in, _ = _make_mhc_tensors(
                tokens,
                hidden,
                world_size,
                torch.bfloat16,
                device,
                seed=7000 + tokens * 100 + it,
            )
            for r in range(world_size):
                peer_xs[r].copy_(new_xs[r])
                peer_addends[r].copy_(new_adds[r])
            residual.copy_(new_res)
            prev_post.copy_(new_pp)
            prev_comb.copy_(new_pc)
            incoming.copy_(new_in)
            torch.cuda.synchronize()

        allocated_before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.cuda.memory_stats(device)["allocation.all.allocated"]
            == allocated_before
        )

        if it in (0, 12, 24):
            expected_reduced = _canonical_bf16_sum(peer_xs, peer_addends)
            out_ref = torch.empty_like(residual)
            y_ref = torch.empty_like(peer_xs[0])
            av_ref = torch.empty_like(a_values)
            as_ref = torch.empty_like(a_scales)
            post_y_runner(
                expected_reduced,
                residual,
                prev_post,
                prev_comb,
                incoming,
                norm_weight,
                out_ref,
                y_ref,
                a_values=av_ref,
                a_scales=as_ref,
            )
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint16), out_ref.view(torch.uint16))
            assert torch.equal(y.view(torch.uint16), y_ref.view(torch.uint16))
            assert torch.equal(a_values, av_ref)
            assert torch.equal(a_scales, as_ref)

    # 1 warmup + 25 replays = 26 x 2 = 52 epochs advanced, arrived == 0, slots cleared.
    assert ring.epoch_and_arrived(0) == (52, 0)
    assert int(ring.slots[0][0].any().item()) == 0
    assert int(ring.slots[1][0].any().item()) == 0


# ---------------------------------------------------------------------------
# Live Multi-GPU PCIeOneshotAllReducePool Verification
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _multi_gpu_worker(rank: int, world_size: int, port: int, hidden: int, tokens: int):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=120),
    )
    pool = None
    try:
        max_bytes = max(2 * 1024 * 1024, tokens * hidden * 2)
        pool = PCIeOneshotAllReducePool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_input_bytes=max_bytes,
            max_size=max_bytes,
        )
        channel_id = f"fused-mhc:{hidden}:{tokens}"
        pool.prepare_channels((channel_id,))

        # Replicate shared mHC state across ranks; per-rank x and input_addend differ.
        torch.manual_seed(4242 + hidden + tokens)
        all_xs = [
            (
                torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.5
            ).contiguous()
            for _ in range(world_size)
        ]
        all_addends = [
            (
                torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.25
            ).contiguous()
            for _ in range(world_size)
        ]
        residual = (
            torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device) * 0.5
        ).contiguous()
        prev_post = torch.rand(
            tokens, 4, dtype=torch.float32, device=device
        ).contiguous()
        prev_comb = torch.softmax(
            torch.randn(tokens, 4, 4, dtype=torch.float32, device=device), dim=-1
        ).contiguous()
        incoming = torch.softmax(
            torch.randn(tokens, 4, dtype=torch.float32, device=device), dim=-1
        ).contiguous()
        norm_weight = (
            torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
        ).contiguous()

        x = all_xs[rank].clone()
        addend = all_addends[rank].clone()
        out = torch.empty_like(residual)
        y = torch.empty_like(x)
        a_values = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
        a_scales = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)

        fused_runner = compile_cluster_allreduce_post_y_norm(
            rank,
            hidden,
            world_size,
            rank,
            1e-20,
            torch.bfloat16,
            emit_mxfp8=True,
            has_addend=True,
            device_slot_selection=True,
            slot_bias=0,
            release_dependents=True,
        )
        post_y_runner = compile_cluster_post_y_norm(
            rank,
            hidden,
            False,
            1e-20,
            torch.bfloat16,
            cluster_size=split_lagged_cluster_size(tokens),
            emit_mxfp8=True,
        )
        load_programs(fused_runner)
        load_programs(post_y_runner)

        graph = torch.cuda.CUDAGraph()
        with pool.capture(channel_id=channel_id) as channel:
            with torch.cuda.graph(graph):
                peer_tbl, sig_tbl, shard_packs, region_packs, dev_sel, bias = (
                    oneshot_channel_sentinel_params(channel, capturing=True)
                )
                assert dev_sel is True and bias == 0
                fused_runner(
                    peer_tbl,
                    sig_tbl,
                    x,
                    residual,
                    prev_post,
                    prev_comb,
                    incoming,
                    norm_weight,
                    out,
                    y,
                    shard_packs=shard_packs,
                    payload_ready_region_packs=region_packs,
                    input_addend=addend,
                    a_values=a_values,
                    a_scales=a_scales,
                )

        dist.barrier()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize(device)

        expected_reduced = _canonical_bf16_sum(all_xs, all_addends)
        out_ref = torch.empty_like(residual)
        y_ref = torch.empty_like(x)
        av_ref = torch.empty_like(a_values)
        as_ref = torch.empty_like(a_scales)
        post_y_runner(
            expected_reduced,
            residual,
            prev_post,
            prev_comb,
            incoming,
            norm_weight,
            out_ref,
            y_ref,
            a_values=av_ref,
            a_scales=as_ref,
        )
        torch.cuda.synchronize(device)
        assert torch.equal(out.view(torch.uint16), out_ref.view(torch.uint16))
        assert torch.equal(y.view(torch.uint16), y_ref.view(torch.uint16))
        assert torch.equal(a_values, av_ref)
        assert torch.equal(a_scales, as_ref)
    finally:
        if pool is not None:
            pool.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ONESHOT_TEST") != "1",
    reason="set B12X_RUN_PCIE_ONESHOT_TEST=1 on a multi-GPU SM120 host to run live multi-rank PCIe test",
)
@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("tokens", [1, 8, 16])
def test_cluster_allreduce_post_y_norm_live_multi_gpu(hidden: int, tokens: int):
    _sm120_required()
    world_size = torch.cuda.device_count()
    if world_size not in (2, 4, 8):
        pytest.skip(f"requires 2, 4, or 8 GPUs, found {world_size}")
    port = _free_port()
    mp.spawn(
        _multi_gpu_worker,
        args=(world_size, port, hidden, tokens),
        nprocs=world_size,
        join=True,
    )
