"""Unit, protocol simulation, and multi-GPU tests for PCIeDmaAllReduce.all_to_all."""

from __future__ import annotations

import ctypes
import os
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from b12x.comm.pcie._dma_preparation import (
    _DmaExecutionState,
    plan as dma_plan,
    prepared_call,
    query_from_metadata,
)
from b12x.comm.pcie._tuning import PcieQuery
from b12x.comm.pcie.pcie_dma import (
    ALL_TO_ALL_DTYPES,
    ALL_TO_ALL_WORLD_SIZES,
    FLAG_SLOTS,
    FLAG_STRIDE,
    MAX_PIECES,
    PCIeDmaAllReduce,
    SUPPORTED_WORLD_SIZES,
    _A2A_DATA_FLAG_BASE,
    _A2A_DONE_FLAG_BASE,
    _A2A_READY_FLAG_BASE,
)
from b12x.preparation import FrozenMapping, Plan, PreparationSession


_skip_unless_pcie_dma_gpu = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DMA_TEST") != "1",
    reason="set B12X_RUN_PCIE_DMA_TEST=1 to run PCIe ring DMA GPU tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stub_channel(
    world_size: int = 4,
    max_bytes: int = 4096,
    shard_capacity: int | None = None,
) -> PCIeDmaAllReduce:
    runtime = PCIeDmaAllReduce.__new__(PCIeDmaAllReduce)
    runtime.rank = 0
    runtime.world_size = world_size
    runtime.device = torch.device("cpu")
    runtime.max_bytes = max_bytes
    runtime.shard_capacity = (
        max_bytes // world_size if shard_capacity is None else shard_capacity
    )
    runtime.min_bytes = 0
    runtime._pieces_override = 0
    runtime._closed = False
    runtime._capture_context_depth = 0
    return runtime


def _stub_prepared_plan(runtime: PCIeDmaAllReduce) -> tuple[Plan, _DmaExecutionState]:
    query = PcieQuery(
        surface="DmaAllReduce.all_reduce",
        world_size=runtime.world_size,
        rank=runtime.rank,
        topology="pcie_ipc",
        call=FrozenMapping({"wire_mode": ""}),
        setup=FrozenMapping({}),
    )
    state = _DmaExecutionState(query=query, runtime=runtime, launchers=None)
    plan = SimpleNamespace(
        _prepared_state=state,
        contract=SimpleNamespace(component_id="comm.pcie"),
        device=runtime.device,
    )
    return plan, state


@pytest.mark.parametrize("world_size", ALL_TO_ALL_WORLD_SIZES)
def test_should_all_to_all_contract(world_size: int) -> None:
    slice_bytes = 256
    runtime = _stub_channel(
        world_size=world_size,
        max_bytes=world_size * slice_bytes,
        shard_capacity=slice_bytes,
    )
    for dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float8_e4m3fn,
        torch.uint8,
    ):
        cols = slice_bytes // dtype.itemsize
        assert runtime.should_all_to_all(torch.zeros((world_size, cols), dtype=dtype))
        assert runtime.should_all_to_all(
            torch.zeros((world_size, 2, cols // 2), dtype=dtype)
        )

    probe = torch.zeros((world_size, 16), dtype=torch.bfloat16)
    assert not runtime.should_all_to_all(torch.tensor(1.0, dtype=torch.bfloat16))
    assert not runtime.should_all_to_all(
        torch.zeros((world_size * 16,), dtype=torch.bfloat16)
    )
    assert not runtime.should_all_to_all(
        torch.zeros((world_size + 1, 16), dtype=torch.bfloat16)
    )
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, 0), dtype=torch.bfloat16)
    )
    # 12 bytes per slice violates 16-byte alignment across every dtype width.
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, 6), dtype=torch.bfloat16)
    )
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, 3), dtype=torch.float32)
    )
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, 15), dtype=torch.uint8)
    )
    # Non-contiguous input rejected.
    assert not runtime.should_all_to_all(
        torch.zeros((16, world_size), dtype=torch.bfloat16).transpose(0, 1)
    )
    # Unsupported dtypes rejected.
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, 8), dtype=torch.int32)
    )
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, 4), dtype=torch.float64)
    )
    # Exceeding max_bytes or shard_capacity rejected.
    assert not runtime.should_all_to_all(
        torch.zeros((world_size, slice_bytes // 2 + 8), dtype=torch.bfloat16)
    )
    narrow_shard = _stub_channel(
        world_size=world_size,
        max_bytes=world_size * slice_bytes,
        shard_capacity=slice_bytes - 16,
    )
    assert not narrow_shard.should_all_to_all(
        torch.zeros((world_size, slice_bytes // 2), dtype=torch.bfloat16)
    )
    # Closed channel rejected.
    runtime._closed = True
    assert not runtime.should_all_to_all(probe)


def test_all_to_all_validates_before_allocating_and_rejects_partial_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _stub_channel(world_size=4, max_bytes=4096, shard_capacity=1024)
    _, state = _stub_prepared_plan(runtime)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dma.require_prepared",
        lambda _plan, _component, _device: state,
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)

    alloc_calls = 0
    orig_empty_like = torch.empty_like

    def counting_empty_like(*args, **kwargs):
        nonlocal alloc_calls
        alloc_calls += 1
        return orig_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", counting_empty_like)

    for bad_inp in (
        torch.tensor(1.0, dtype=torch.bfloat16),
        torch.zeros((32,), dtype=torch.bfloat16),
        torch.zeros((3, 16), dtype=torch.bfloat16),
        torch.zeros((4, 0), dtype=torch.bfloat16),
        torch.zeros((4, 6), dtype=torch.bfloat16),
        torch.zeros((16, 4), dtype=torch.bfloat16).transpose(0, 1),
        torch.zeros((4, 16), dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="all_to_all"):
            runtime.all_to_all(bad_inp, plan=None)  # type: ignore[arg-type]
    assert alloc_calls == 0

    # Wrong runtime or state missing run_all_to_all is rejected.
    other = _stub_channel(world_size=4, max_bytes=4096, shard_capacity=1024)
    with pytest.raises(ValueError, match="different native runtime"):
        other.all_to_all(torch.zeros((4, 16), dtype=torch.bfloat16), plan=None)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dma.require_prepared",
        lambda _plan, _component, _device: SimpleNamespace(
            require_runtime=lambda _: None
        ),
    )
    with pytest.raises(TypeError, match="prepared DMA channel"):
        runtime.all_to_all(torch.zeros((4, 16), dtype=torch.bfloat16), plan=None)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dma.require_prepared",
        lambda _plan, _component, _device: state,
    )

    # Partial storage overlap (non-identical overlapping buffers) is rejected.
    backing = torch.zeros((6, 16), dtype=torch.bfloat16)
    inp = backing[:4]
    for aliased_out in (backing[1:5], backing[2:6]):
        with pytest.raises(ValueError, match="disjoint or identical"):
            runtime.all_to_all(inp, plan=None, out=aliased_out)  # type: ignore[arg-type]

    # Output shape, dtype, and contiguity mismatches are rejected.
    for bad_out in (
        torch.empty((4, 8), dtype=torch.bfloat16),
        torch.empty((2, 2, 16), dtype=torch.bfloat16),
        torch.empty((4, 16), dtype=torch.float16),
        torch.empty((16, 4), dtype=torch.bfloat16).transpose(0, 1),
    ):
        with pytest.raises(ValueError, match="contiguous"):
            runtime.all_to_all(inp, plan=None, out=bad_out)  # type: ignore[arg-type]


def test_all_to_all_flag_slots_disjoint_from_ring_and_prime() -> None:
    max_world = max(ALL_TO_ALL_WORLD_SIZES)
    assert set(ALL_TO_ALL_WORLD_SIZES) <= set(SUPPORTED_WORLD_SIZES)
    tp16_runtime = _stub_channel(world_size=16, max_bytes=16 * 256, shard_capacity=256)
    assert not tp16_runtime.should_all_to_all(torch.zeros((16, 128), dtype=torch.bfloat16))
    max_steps = max_world - 1
    ring_data = set(range(2 * max_steps * MAX_PIECES))
    a2a_data = {
        _A2A_DATA_FLAG_BASE + step * MAX_PIECES + p
        for step in range(max_steps)
        for p in range(MAX_PIECES)
    }
    a2a_ready = {_A2A_READY_FLAG_BASE + step for step in range(max_steps)}
    a2a_done = {_A2A_DONE_FLAG_BASE + step for step in range(max_steps)}
    ring_done = {FLAG_SLOTS - 2}
    prime_slot = {FLAG_SLOTS - 1}

    groups = (ring_data, a2a_data, a2a_ready, a2a_done, ring_done, prime_slot)
    all_slots = set().union(*groups)
    assert len(all_slots) == sum(len(g) for g in groups)
    assert max(all_slots) < FLAG_SLOTS
    assert ALL_TO_ALL_DTYPES == frozenset(
        (torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn, torch.uint8)
    )


class _FakeEvent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.recorded_on: list[str] = []

    def record(self, stream) -> None:
        self.recorded_on.append(stream.name)


class _FakeStream:
    def __init__(self, name: str, log: list[tuple]) -> None:
        self.name = name
        self._log = log

    def wait_event(self, event: _FakeEvent) -> None:
        self._log.append(("wait_event", self.name, event.name))

    def wait_stream(self, other: "_FakeStream") -> None:
        self._log.append(("wait_stream", self.name, other.name))


class _RecordedDmaKernels:
    """Capture-only stand-in for DmaKernels recording stream-ordered ops on CPU."""

    def __init__(self, current_stream_fn, ops: list[tuple]) -> None:
        self._current_stream = current_stream_fn
        self._ops = ops

    def dma_copy(self, dst: int, src: int, nbytes: int) -> None:
        self._ops.append(("copy", self._current_stream().name, dst, src, nbytes))

    def dma_set_flag(self, flag_ptr: int, counter_ptr: int) -> None:
        self._ops.append(
            ("set_flag", self._current_stream().name, flag_ptr, counter_ptr)
        )

    def dma_wait_flag(self, flag_ptr: int, counter_ptr: int) -> None:
        self._ops.append(
            ("wait_flag", self._current_stream().name, flag_ptr, counter_ptr)
        )


def _execute_simulated_all_to_all(
    world_size: int,
    inputs: list[torch.Tensor],
    *,
    in_place: bool,
    pieces_override: int,
    iterations: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[torch.Tensor], list[PCIeDmaAllReduce], list[list[tuple]], list[list[tuple]]
]:
    """Run all `world_size` ranks' `_run_prepared_all_to_all` over real CPU buffers.

    Records each rank's emitted stream ops (`copy`, `set_flag`, `wait_flag`) and
    executes them in valid cross-rank dependency order (ready -> send -> receive
    -> done) so protocol bijections, slot uniqueness, in-place ordering, wire
    bytes, and monotonic counters are verified end to end without CUDA hardware.
    """
    slice_bytes = inputs[0].numel() * inputs[0].element_size() // world_size
    shard_capacity = max(slice_bytes, 256)
    flags_bytes = FLAG_SLOTS * FLAG_STRIDE
    steps = 2 * (world_size - 1)
    slabs = [
        torch.zeros(flags_bytes + steps * shard_capacity, dtype=torch.uint8)
        for _ in range(world_size)
    ]
    flags_base = [slab.data_ptr() for slab in slabs]
    scratch_base = [slab.data_ptr() + flags_bytes for slab in slabs]

    channels: list[PCIeDmaAllReduce] = []
    states: list[_DmaExecutionState] = []
    active_stream = [None] * world_size
    per_rank_ops: list[list[tuple]] = [[] for _ in range(world_size)]
    per_rank_events: list[list[tuple]] = [[] for _ in range(world_size)]

    for r in range(world_size):
        ch = PCIeDmaAllReduce.__new__(PCIeDmaAllReduce)
        ch.rank = r
        ch.world_size = world_size
        ch.device = torch.device("cpu")
        ch.max_bytes = max(slice_bytes * world_size, 4096)
        ch.shard_capacity = shard_capacity
        ch.min_bytes = 0
        ch._pieces_override = pieces_override
        ch._closed = False
        ch._capture_context_depth = 0
        ch._flags_base = list(flags_base)
        ch._scratch_base = list(scratch_base)
        ch._send_counters = torch.zeros(FLAG_SLOTS, dtype=torch.int32)
        ch._wait_counters = torch.zeros(FLAG_SLOTS, dtype=torch.int32)
        ch._copy_stream = _FakeStream("copy", per_rank_events[r])
        ch._flag_stream = _FakeStream("flag", per_rank_events[r])
        ch._input_ready = _FakeEvent("input_ready")
        ch._copied_events = [
            _FakeEvent(f"copied_{idx}")
            for idx in range(2 * (world_size - 1) * MAX_PIECES)
        ]
        ch._kernels = _RecordedDmaKernels(lambda r=r: active_stream[r], per_rank_ops[r])
        _, st = _stub_prepared_plan(ch)
        channels.append(ch)
        states.append(st)

    current_rank = [0]

    @contextmanager
    def fake_stream_ctx(stream):
        r = current_rank[0]
        prev = active_stream[r]
        active_stream[r] = stream
        try:
            yield
        finally:
            active_stream[r] = prev

    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _device: active_stream[current_rank[0]]
    )
    monkeypatch.setattr(torch.cuda, "stream", fake_stream_ctx)

    def _run_u32_publish(flag_ptr: int, counter_ptr: int) -> None:
        counter = ctypes.c_uint32.from_address(counter_ptr)
        flag = ctypes.c_uint32.from_address(flag_ptr)
        counter.value = (counter.value + 1) & 0xFFFFFFFF
        flag.value = counter.value

    def _run_u32_wait(flag_ptr: int, counter_ptr: int) -> None:
        counter = ctypes.c_uint32.from_address(counter_ptr)
        flag = ctypes.c_uint32.from_address(flag_ptr)
        counter.value = (counter.value + 1) & 0xFFFFFFFF
        assert flag.value == counter.value, (
            f"flag mismatch at 0x{flag_ptr:x}: observed={flag.value}, expected={counter.value}"
        )

    outputs: list[torch.Tensor] = []
    for iteration in range(iterations):
        iter_in = [inp.clone() for inp in inputs]
        iter_out = iter_in if in_place else [torch.empty_like(inp) for inp in iter_in]
        for r in range(world_size):
            per_rank_ops[r].clear()
            per_rank_events[r].clear()
            current_rank[0] = r
            active_stream[r] = _FakeStream("main", per_rank_events[r])
            ret = states[r].run_all_to_all(iter_in[r], out=iter_out[r])
            assert ret is iter_out[r]

        # Phase 1: ready set_flags + local self-copy on main, then ready wait_flags.
        # Split each rank's op stream at its natural phase boundaries and verify order.
        peer_steps = world_size - 1
        for r in range(world_size):
            ops = per_rank_ops[r]
            idx = 0
            for _ in range(peer_steps):
                op, stream_name, flag_ptr, counter_ptr = ops[idx]
                assert op == "set_flag" and stream_name == "main"
                _run_u32_publish(flag_ptr, counter_ptr)
                idx += 1
            if not in_place:
                op, stream_name, dst, src, nbytes = ops[idx]
                assert op == "copy" and stream_name == "main" and nbytes == slice_bytes
                ctypes.memmove(dst, src, nbytes)
                idx += 1

        # Phase 2: ready waits across all ranks.
        ready_end = peer_steps + (0 if in_place else 1)
        for r in range(world_size):
            ops = per_rank_ops[r]
            for offset in range(peer_steps):
                op, stream_name, flag_ptr, counter_ptr = ops[ready_end + offset]
                assert op == "wait_flag" and stream_name == "main"
                _run_u32_wait(flag_ptr, counter_ptr)

        # Phase 3: all outgoing peer copies on copy_stream + data flags on flag_stream.
        send_start = ready_end + peer_steps
        send_end = send_start
        while send_end < len(per_rank_ops[0]) and per_rank_ops[0][send_end][1] in (
            "copy",
            "flag",
        ):
            send_end += 1
        written_dst_ranges: list[tuple[int, int]] = []
        written_flag_ptrs: set[int] = set()
        for r in range(world_size):
            peer_wire_bytes = 0
            for op_tuple in per_rank_ops[r][send_start:send_end]:
                if op_tuple[0] == "copy":
                    _, stream_name, dst, src, nbytes = op_tuple
                    assert stream_name == "copy"
                    written_dst_ranges.append((dst, dst + nbytes))
                    ctypes.memmove(dst, src, nbytes)
                    peer_wire_bytes += nbytes
                else:
                    op, stream_name, flag_ptr, counter_ptr = op_tuple
                    assert op == "set_flag" and stream_name == "flag"
                    assert flag_ptr not in written_flag_ptrs, (
                        "duplicate data flag write"
                    )
                    written_flag_ptrs.add(flag_ptr)
                    _run_u32_publish(flag_ptr, counter_ptr)
            # Wire traffic per rank is strictly (world_size - 1) * slice_bytes.
            assert peer_wire_bytes == (world_size - 1) * slice_bytes

        # Every peer scratch destination write in the iteration is strictly disjoint.
        written_dst_ranges.sort()
        for (_, prev_end), (next_start, _) in zip(
            written_dst_ranges, written_dst_ranges[1:], strict=False
        ):
            assert prev_end <= next_start

        # Phase 4: receive waits + scratch->out copies on main.
        recv_start = send_end
        done_start = len(per_rank_ops[0]) - 2 * peer_steps
        for r in range(world_size):
            ops = per_rank_ops[r][recv_start:done_start]
            assert len(ops) == (send_end - send_start)
            for k in range(0, len(ops), 2):
                w_op, w_stream, flag_ptr, counter_ptr = ops[k]
                c_op, c_stream, dst, src, nbytes = ops[k + 1]
                assert w_op == "wait_flag" and w_stream == "main"
                assert c_op == "copy" and c_stream == "main"
                _run_u32_wait(flag_ptr, counter_ptr)
                ctypes.memmove(dst, src, nbytes)

        # Phase 5: exit done barrier across all ranks.
        for r in range(world_size):
            for op, stream_name, flag_ptr, counter_ptr in per_rank_ops[r][
                done_start : done_start + peer_steps
            ]:
                assert op == "set_flag" and stream_name == "main"
                _run_u32_publish(flag_ptr, counter_ptr)
        for r in range(world_size):
            for op, stream_name, flag_ptr, counter_ptr in per_rank_ops[r][
                done_start + peer_steps :
            ]:
                assert op == "wait_flag" and stream_name == "main"
                _run_u32_wait(flag_ptr, counter_ptr)

        # Verify bit-exact output on every rank and iteration.
        for dst_rank in range(world_size):
            for src_rank in range(world_size):
                assert torch.equal(
                    iter_out[dst_rank][src_rank], inputs[src_rank][dst_rank]
                )
        outputs = iter_out

    return outputs, channels, per_rank_ops, per_rank_events


@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("pieces_override", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_all_to_all_multi_rank_protocol_simulation(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    in_place: bool,
    pieces_override: int,
    dtype: torch.dtype,
) -> None:
    # Use 1 MiB slices when pieces_override == 2 so _pick_a2a_pieces chooses 2 pieces.
    slice_bytes = (1 << 20) if pieces_override == 2 else 256
    cols = slice_bytes // dtype.itemsize
    inputs = []
    for r in range(world_size):
        raw = torch.arange(world_size * cols, dtype=torch.int64).view(world_size, cols)
        vals = ((raw + r * 37) % 127).to(dtype).contiguous()
        inputs.append(vals)

    _, channels, _, events = _execute_simulated_all_to_all(
        world_size,
        inputs,
        in_place=in_place,
        pieces_override=pieces_override,
        iterations=3,
        monkeypatch=monkeypatch,
    )

    for ch, ev_log in zip(channels, events, strict=True):
        # Monotonic counters stayed in lockstep across all 3 iterations.
        assert torch.equal(ch._send_counters, ch._wait_counters)
        assert int(ch._send_counters.max().item()) == 3
        # Auxiliary streams forked from input_ready and joined back to main.
        assert ("wait_event", "copy", "input_ready") in ev_log
        assert ("wait_event", "flag", "input_ready") in ev_log
        assert ("wait_stream", "main", "copy") in ev_log
        assert ("wait_stream", "main", "flag") in ev_log
        if in_place:
            # In-place waits on copy_stream twice: once before receive copies and once at drain.
            assert ev_log.count(("wait_stream", "main", "copy")) == 2
        else:
            assert ev_log.count(("wait_stream", "main", "copy")) == 1


def _worker(rank: int, world_size: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    dma = None
    sess = None
    prep_result = None
    graph = None
    try:
        max_bytes = world_size * 2048 * 1536 * 2
        dma = PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD, max_bytes=max_bytes, device=device
        )
        probe_in = torch.zeros(
            (world_size, 1024, 1536), dtype=torch.bfloat16, device=device
        )
        probe_out = torch.empty_like(probe_in)
        plan = dma_plan(
            query_from_metadata(dma, shape=tuple(probe_in.shape), dtype=torch.bfloat16),
            runtime=dma,
        )
        sess = PreparationSession(device=device, compile_workers=0, autotune=False)
        dist.barrier()
        prep_result = sess.prepare(
            (
                plan.request(
                    name="dma_a2a",
                    prepare_call=lambda st: prepared_call(
                        st, inp=probe_in, out=probe_out
                    ),
                ),
            )
        )
        dist.barrier()

        # 1. Exact bit-for-bit parity across all supported dtypes (out=None, disjoint, in-place).
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            for rows, cols in ((16, 64), (1024, 1536)):
                if world_size * rows * cols * dtype.itemsize > max_bytes:
                    continue
                raw = (
                    torch.arange(world_size * rows * cols, device=device) % 113
                ).view(world_size, rows, cols)
                inp = ((raw + rank * 19) % 127).to(dtype).contiguous()
                ref_u8 = torch.empty_like(inp.view(torch.uint8))
                dist.all_to_all_single(ref_u8, inp.view(torch.uint8))
                ref = ref_u8.view(dtype)

                out_alloc = dma.all_to_all(inp, plan=plan)
                torch.cuda.synchronize(device)
                assert torch.equal(out_alloc.view(torch.uint8), ref.view(torch.uint8))

                out_explicit = torch.empty_like(inp)
                ret = dma.all_to_all(inp, plan=plan, out=out_explicit)
                assert ret is out_explicit
                torch.cuda.synchronize(device)
                assert torch.equal(
                    out_explicit.view(torch.uint8), ref.view(torch.uint8)
                )

                buf_inplace = inp.clone()
                ret_ip = dma.all_to_all(buf_inplace, plan=plan, out=buf_inplace)
                assert ret_ip is buf_inplace
                torch.cuda.synchronize(device)
                assert torch.equal(buf_inplace.view(torch.uint8), ref.view(torch.uint8))

        # 2. Interleaved ring all_gather -> all_to_all -> reduce_scatter -> all_to_all on one channel.
        ag_in = torch.full((256, 1536), rank + 1, dtype=torch.bfloat16, device=device)
        ag_out = torch.empty(
            (world_size, 256, 1536), dtype=torch.bfloat16, device=device
        )
        a2a_in = torch.empty(
            (world_size, 256, 1536), dtype=torch.bfloat16, device=device
        )
        for peer in range(world_size):
            a2a_in[peer].fill_(rank * 10 + peer)
        a2a_out = torch.empty_like(a2a_in)
        rs_in = torch.full(
            (world_size * 256, 1536), rank + 1, dtype=torch.bfloat16, device=device
        )
        rs_out = torch.empty((256, 1536), dtype=torch.bfloat16, device=device)
        for _ in range(5):
            dma.all_gather(ag_in, plan=plan, out=ag_out)
            dma.all_to_all(a2a_in, plan=plan, out=a2a_out)
            dma.reduce_scatter(rs_in, plan=plan, out=rs_out)
            dma.all_to_all(a2a_in, plan=plan, out=a2a_out)
        torch.cuda.synchronize(device)
        for src in range(world_size):
            assert torch.equal(
                a2a_out[src],
                torch.full(
                    (256, 1536), src * 10 + rank, dtype=torch.bfloat16, device=device
                ),
            )

        # 3. Captured CUDA graph replay overlapped with compute on another stream.
        comm_stream = torch.cuda.Stream(device=device)
        ev_fork, ev_join = torch.cuda.Event(), torch.cuda.Event()
        a = torch.randn((512, 512), dtype=torch.bfloat16, device=device)
        b = torch.randn((512, 512), dtype=torch.bfloat16, device=device)
        c = torch.empty((512, 512), dtype=torch.bfloat16, device=device)
        torch.mm(a, b, out=c)

        g_in = torch.empty(
            (world_size, 1024, 1536), dtype=torch.bfloat16, device=device
        )
        for peer in range(world_size):
            g_in[peer].fill_(rank * 16 + peer + 1)
        g_out = torch.empty_like(g_in)
        g_ref = torch.empty_like(g_in)
        dist.all_to_all_single(g_ref, g_in)

        def step() -> None:
            main_s = torch.cuda.current_stream(device)
            ev_fork.record(main_s)
            comm_stream.wait_event(ev_fork)
            with torch.cuda.stream(comm_stream):
                dma.all_to_all(g_in, plan=plan, out=g_out)
                ev_join.record(comm_stream)
            torch.mm(a, b, out=c)
            main_s.wait_event(ev_join)

        for _ in range(3):
            step()
        torch.cuda.synchronize(device)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with (
            sess.capture(),
            dma.capture(stream=comm_stream),
            torch.cuda.graph(graph),
        ):
            step()
        torch.cuda.synchronize(device)
        dist.barrier()
        before_allocs = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        for _ in range(50):
            g_out.zero_()
            graph.replay()
        torch.cuda.synchronize(device)
        after_allocs = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        assert before_allocs == after_allocs
        assert torch.equal(g_out, g_ref)
        dist.barrier()
    finally:
        if graph is not None:
            graph.reset()
        if prep_result is not None:
            prep_result.close()
        if sess is not None:
            sess.close()
        if dma is not None:
            dma.close()
        dist.destroy_process_group()


@_skip_unless_pcie_dma_gpu
def test_pcie_dma_all_to_all_2gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA GPUs")
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)


@_skip_unless_pcie_dma_gpu
def test_pcie_dma_all_to_all_4gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires at least 4 CUDA GPUs")
    mp.spawn(_worker, args=(4, _free_port()), nprocs=4, join=True)
