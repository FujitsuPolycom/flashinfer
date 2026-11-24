"""Unit and multi-GPU tests for PCIeDmaAllReduce.reduce_scatter."""

from __future__ import annotations

import os
import socket
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
from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce
from b12x.preparation import FrozenMapping, Plan, PreparationSession


_skip_unless_pcie_dma_gpu = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DMA_TEST") != "1",
    reason="set B12X_RUN_PCIE_DMA_TEST=1 to run PCIe ring DMA GPU tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stub_channel(world_size: int = 4, max_bytes: int = 4096) -> PCIeDmaAllReduce:
    runtime = PCIeDmaAllReduce.__new__(PCIeDmaAllReduce)
    runtime.rank = 0
    runtime.world_size = world_size
    runtime.device = torch.device("cpu")
    runtime.max_bytes = max_bytes
    runtime.min_bytes = 0
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


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_should_reduce_scatter_contract(world_size: int) -> None:
    runtime = _stub_channel(world_size=world_size, max_bytes=world_size * 64 * 4)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        valid = torch.zeros((world_size * 2, 16), dtype=dtype)
        assert runtime.should_reduce_scatter(valid)

    probe = torch.zeros((world_size * 4, 16), dtype=torch.bfloat16)
    assert not runtime.should_reduce_scatter(torch.tensor(1.0, dtype=torch.bfloat16))
    assert not runtime.should_reduce_scatter(probe[:0])
    assert not runtime.should_reduce_scatter(probe[:1])
    assert not runtime.should_reduce_scatter(
        torch.zeros((world_size, 6), dtype=torch.bfloat16)
    )
    assert not runtime.should_reduce_scatter(probe.transpose(0, 1))
    assert not runtime.should_reduce_scatter(
        torch.zeros((world_size * 2, 16), dtype=torch.float8_e4m3fn)
    )
    assert not runtime.should_reduce_scatter(
        torch.zeros((world_size * 8, 16), dtype=torch.float32)
    )
    runtime._closed = True
    assert not runtime.should_reduce_scatter(probe)


def test_reduce_scatter_validates_before_allocating_and_rejects_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _stub_channel(world_size=4, max_bytes=4096)
    _, state = _stub_prepared_plan(runtime)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dma.require_prepared",
        lambda _plan, _component, _device: state,
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)

    alloc_calls = 0
    orig_empty = torch.empty

    def counting_empty(*args, **kwargs):
        nonlocal alloc_calls
        alloc_calls += 1
        return orig_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", counting_empty)

    # Invalid inputs with out=None must raise ValueError before allocating out.
    for bad_inp in (
        torch.tensor(1.0, dtype=torch.bfloat16),
        orig_empty((0, 16), dtype=torch.bfloat16),
        orig_empty((1, 32), dtype=torch.bfloat16),
        orig_empty((16, 16), dtype=torch.bfloat16).transpose(0, 1),
        orig_empty((8, 16), dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="reduce_scatter"):
            runtime.reduce_scatter(bad_inp, plan=None)  # type: ignore[arg-type]
    assert alloc_calls == 0

    inp = orig_empty((8, 16), dtype=torch.bfloat16)
    shard_rows = 2
    for aliased_out in (
        inp[:shard_rows],
        inp[shard_rows : 2 * shard_rows],
        inp[-shard_rows:],
        inp.view(-1)[8 : 8 + shard_rows * 16].view(shard_rows, 16),
    ):
        with pytest.raises(ValueError, match="disjoint"):
            runtime.reduce_scatter(inp, plan=None, out=aliased_out)  # type: ignore[arg-type]

    for bad_out in (
        orig_empty((shard_rows + 1, 16), dtype=torch.bfloat16),
        orig_empty((shard_rows, 16), dtype=torch.float16),
        orig_empty((16, shard_rows), dtype=torch.bfloat16).transpose(0, 1),
    ):
        with pytest.raises(ValueError, match="contiguous"):
            runtime.reduce_scatter(inp, plan=None, out=bad_out)  # type: ignore[arg-type]


def _assert_close(actual: torch.Tensor, ref: torch.Tensor, world_size: int) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-5 * world_size)
    elif actual.dtype == torch.float16:
        torch.testing.assert_close(
            actual.float(), ref, rtol=1e-2, atol=1e-2 * world_size
        )
    else:
        torch.testing.assert_close(
            actual.float(), ref, rtol=3e-2, atol=3e-2 * world_size
        )


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
        max_bytes = 4096 * 5120 * 4
        dma = PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD, max_bytes=max_bytes, device=device
        )
        probe_in = torch.zeros((4096, 5120), dtype=torch.bfloat16, device=device)
        probe_out = torch.empty_like(probe_in)
        plan = dma_plan(
            query_from_metadata(dma, shape=(4096, 5120), dtype=torch.bfloat16),
            runtime=dma,
        )
        sess = PreparationSession(device=device, compile_workers=0, autotune=False)
        dist.barrier()
        prep_result = sess.prepare(
            (
                plan.request(
                    name="dma_rs",
                    prepare_call=lambda st: prepared_call(
                        st, inp=probe_in, out=probe_out
                    ),
                ),
            )
        )
        dist.barrier()

        # Eligibility and rejection checks
        assert dma.should_reduce_scatter(probe_in)
        assert not dma.should_reduce_scatter(
            torch.tensor(1.0, dtype=torch.bfloat16, device=device)
        )
        assert not dma.should_reduce_scatter(probe_in[:0])
        assert not dma.should_reduce_scatter(probe_in[:1])
        assert not dma.should_reduce_scatter(probe_in.transpose(0, 1))
        assert not dma.should_reduce_scatter(
            torch.zeros((4096, 5120), dtype=torch.float8_e4m3fn, device=device)
        )

        probe_shard_rows = 4096 // world_size
        with pytest.raises(ValueError, match="reduce_scatter"):
            dma.reduce_scatter(
                torch.tensor(1.0, dtype=torch.bfloat16, device=device), plan=plan
            )
        with pytest.raises(ValueError, match="reduce_scatter"):
            dma.reduce_scatter(probe_in[:1], plan=plan)
        with pytest.raises(ValueError, match="disjoint"):
            dma.reduce_scatter(probe_in, plan=plan, out=probe_in[:probe_shard_rows])
        with pytest.raises(ValueError, match="disjoint"):
            dma.reduce_scatter(
                probe_in,
                plan=plan,
                out=probe_in[rank * probe_shard_rows : (rank + 1) * probe_shard_rows],
            )
        with pytest.raises(ValueError, match="contiguous"):
            dma.reduce_scatter(
                probe_in,
                plan=plan,
                out=torch.empty(
                    (probe_shard_rows + 1, 5120), dtype=torch.bfloat16, device=device
                ),
            )
        with pytest.raises(ValueError, match="contiguous"):
            dma.reduce_scatter(
                probe_in,
                plan=plan,
                out=torch.empty(
                    (probe_shard_rows, 5120), dtype=torch.float16, device=device
                ),
            )
        with pytest.raises(ValueError, match="contiguous"):
            dma.reduce_scatter(
                probe_in,
                plan=plan,
                out=torch.empty(
                    (probe_shard_rows, 5120), dtype=torch.bfloat16, device="cpu"
                ),
            )
        with pytest.raises(ValueError, match="contiguous"):
            dma.reduce_scatter(
                probe_in,
                plan=plan,
                out=torch.empty(
                    (probe_shard_rows, 5120),
                    dtype=torch.bfloat16,
                    device=torch.device(f"cuda:{(rank + 1) % world_size}"),
                ),
            )
        with pytest.raises(ValueError, match="contiguous"):
            dma.reduce_scatter(
                probe_in,
                plan=plan,
                out=torch.empty(
                    (5120, probe_shard_rows), dtype=torch.bfloat16, device=device
                ).transpose(0, 1),
            )

        # 1. Exact integer parity across supported dtypes (out=None and explicit out)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for rows, cols in ((1024, 5120), (4096, 5120)):
                shard_rows = rows // world_size
                base = (torch.arange(rows * cols, device=device) % 13 - 6).view(
                    rows, cols
                )
                inp = (base + rank * 2).to(dtype).contiguous()
                ref = torch.empty(
                    (shard_rows, cols), dtype=torch.float32, device=device
                )
                dist.reduce_scatter_tensor(ref, inp.float(), op=dist.ReduceOp.SUM)
                if rows == 1024:
                    out = dma.reduce_scatter(inp, plan=plan)
                    assert out.shape == (shard_rows, cols)
                    assert out.dtype == dtype
                    assert out.device == device
                else:
                    out = torch.empty((shard_rows, cols), dtype=dtype, device=device)
                    ret = dma.reduce_scatter(inp, plan=plan, out=out)
                    assert ret is out
                torch.cuda.synchronize(device)
                assert torch.equal(out, ref.to(dtype))

        # 2. Random floating-point parity across supported dtypes against FP32 reference
        for dtype_idx, dtype in enumerate(
            (torch.bfloat16, torch.float16, torch.float32)
        ):
            gen = torch.Generator(device="cpu").manual_seed(
                9100 + dtype_idx * 37 + rank
            )
            inp_rand = (torch.randn((4096, 5120), generator=gen) * 0.25).to(
                device=device, dtype=dtype
            )
            out_rand = torch.empty(
                (4096 // world_size, 5120), dtype=dtype, device=device
            )
            dma.reduce_scatter(inp_rand, plan=plan, out=out_rand)
            ref_f32 = torch.empty(
                (4096 // world_size, 5120), dtype=torch.float32, device=device
            )
            dist.reduce_scatter_tensor(ref_f32, inp_rand.float(), op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            _assert_close(out_rand, ref_f32, world_size)

        # 3. Captured CUDA graph replay overlapped with compute on another stream
        comm_stream = torch.cuda.Stream(device=device)
        ev_fork, ev_join = torch.cuda.Event(), torch.cuda.Event()
        a = torch.randn((512, 512), dtype=torch.bfloat16, device=device)
        b = torch.randn((512, 512), dtype=torch.bfloat16, device=device)
        c = torch.empty((512, 512), dtype=torch.bfloat16, device=device)
        torch.mm(a, b, out=c)  # initialize cuBLAS before capture
        g_in = (torch.arange(4096 * 5120, device=device) % 13 - 6).view(4096, 5120).to(
            torch.bfloat16
        ) + rank
        g_out = torch.empty(
            (4096 // world_size, 5120), dtype=torch.bfloat16, device=device
        )
        g_ref = torch.empty(
            (4096 // world_size, 5120), dtype=torch.float32, device=device
        )
        dist.reduce_scatter_tensor(g_ref, g_in.float(), op=dist.ReduceOp.SUM)

        def step() -> None:
            main_s = torch.cuda.current_stream(device)
            ev_fork.record(main_s)
            comm_stream.wait_event(ev_fork)
            with torch.cuda.stream(comm_stream):
                dma.reduce_scatter(g_in, plan=plan, out=g_out)
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
        assert torch.equal(g_out, g_ref.to(torch.bfloat16))
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
def test_pcie_dma_reduce_scatter_2gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA GPUs")
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)


@_skip_unless_pcie_dma_gpu
def test_pcie_dma_reduce_scatter_4gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires at least 4 CUDA GPUs")
    mp.spawn(_worker, args=(4, _free_port()), nprocs=4, join=True)
