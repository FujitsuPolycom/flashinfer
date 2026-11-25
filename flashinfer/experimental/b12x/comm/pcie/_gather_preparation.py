"""Fixed-geometry column gathers with static specialized barrier profiles."""

from dataclasses import dataclass
from math import prod

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import MemoryRequirements, PersistentMemory, Plan, current_plan
from ._tuning import TUNING

GATHER_MODES = {
    "selected_phased_barrier": ("barrier",),
    "selected_phased_push": ("barrier",),
    "selected_ring": ("barrier",),
    "selected_switch_push": ("barrier",),
    "selected_recursive": ("barrier",),
    "selected_recursive_ready": ("barrier",),
    "selected_recursive_outer_first": ("barrier",),
    "selected_recursive_pix_last": ("barrier",),
}


def selected_gather_profile(transport, blocks, mode, active):
    if transport not in GATHER_MODES or mode not in GATHER_MODES[transport]:
        raise ValueError("profile does not belong to the selected gather transport")
    if (
        type(active) is not int
        or type(blocks) is not int
        or not 1 <= active <= blocks <= 64
    ):
        raise ValueError("active gather CTAs must fit the physical grid")
    if transport in ("selected_phased_barrier", "selected_phased_push") and (
        blocks != 8 or active != 8
    ):
        raise ValueError("phased gather requires one complete eight-CTA cluster")
    if transport == "selected_recursive_pix_last":
        if not ((blocks == 8 and active == 8) or (blocks == 16 and active in (8, 16))):
            raise ValueError(
                "pix-last recursive gather requires 8/8 or 16/(8|16) physical/active CTAs"
            )
    elif transport == "selected_ring":
        if not ((blocks == 8 and active == 8) or (blocks == 16 and active == 16)):
            raise ValueError("ring gather requires 8/8 or 16/16 physical/active CTAs")
    elif transport == "selected_switch_push":
        if not ((blocks == 8 and active == 8) or (blocks == 16 and active in (8, 16))):
            raise ValueError(
                "switch_push gather requires 8/8 or 16/(8|16) physical/active CTAs"
            )
    elif transport in (
        "selected_recursive",
        "selected_recursive_ready",
        "selected_recursive_outer_first",
    ) and (blocks != 8 or active != 8):
        raise ValueError("recursive gathers require eight active and physical CTAs")
    return active


def validate_query(query):
    call = query.call
    shape = call["shape"]
    if (
        len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or call["dtype"]
        not in ("torch.float16", "torch.bfloat16", "torch.float8_e4m3fn")
    ):
        raise ValueError(
            "selected gather requires a positive 3D FP16, BF16, or E4M3 input"
        )
    element = 1 if call["dtype"] == "torch.float8_e4m3fn" else 2
    row_bytes = prod(shape[1:]) * element
    if (
        shape[0] > query.setup["max_batch_size"]
        or shape[1] * query.world_size != query.setup["total_heads"]
        or shape[2] != query.setup["query_head_dim"]
    ):
        raise ValueError(
            "selected gather shape exceeds or differs from its channel geometry"
        )
    transport = call["gather_transport"]
    selected_gather_profile(
        transport, call["block_limit"], call["submode"], call["active_blocks"]
    )
    if transport == "selected_ring" and (
        query.world_size not in (4, 8, 16)
        or call["threads"] not in (128, 512)
        or (query.world_size != 16 and call["block_limit"] != 8)
    ):
        raise ValueError(
            "ring gather requires DCP4/8/16 with 128 or 512 threads per CTA (16 CTAs on DCP16 only)"
        )
    if transport == "selected_switch_push" and (
        query.world_size != 4 or call["threads"] not in (128, 256, 512)
    ):
        raise ValueError(
            "switch push gather requires DCP4 with 128, 256, or 512 threads per CTA"
        )
    if transport == "selected_recursive_pix_last" and (
        query.world_size not in (4, 8, 16)
        or call["threads"] not in (128, 256, 512)
        or (query.world_size != 8 and call["block_limit"] != 8)
    ):
        raise ValueError(
            "pix-last recursive gather requires DCP4/8/16 with 128, 256, or 512 threads per CTA"
        )
    if transport == "selected_recursive_outer_first" and query.world_size != 16:
        raise ValueError("outer-first recursive doubling requires sixteen ranks")
    if transport in (
        "selected_recursive",
        "selected_recursive_ready",
        "selected_recursive_outer_first",
    ) and (query.world_size not in (8, 16) or call["threads"] != 128):
        raise ValueError(
            "recursive gathers require DCP8 or DCP16 with 128 threads per CTA"
        )
    if (
        call["threads"] not in (32, 64, 128, 256, 512)
        or row_bytes % 16
        or row_bytes < 16
        or shape[0] <= 0
        or not call.get("disjoint_output", False)
        or not call.get("pack_aligned", False)
        or tuple(call["stride"]) != (prod(shape[1:]), shape[2], 1)
        or tuple(call["out_shape"]) != (shape[0], shape[1] * query.world_size, shape[2])
        or tuple(call["out_stride"])
        != (prod(shape[1:]) * query.world_size, shape[2], 1)
    ):
        raise ValueError(
            "selected gather requires disjoint contiguous 16-byte-aligned rows"
        )


@dataclass
class SelectedGatherState:
    query: object
    runtime: object
    launcher: object
    selector: torch.Tensor

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("selected gather plan belongs to a different channel")

    def run_selected(self, source, output):
        c, runtime = self.query.call, self.runtime
        if tuple(source.shape) != tuple(c["shape"]) or str(source.dtype) != c["dtype"]:
            raise ValueError(
                "selected gather input differs from the prepared shape or dtype"
            )
        if not runtime._device_slot_selection:
            # Selected barrier kernels retire device epochs even in eager warmup.
            # Enter the same bank sequence as ordinary captured DCP operations.
            runtime._graph_base_slot = runtime._next_slot & 1
            runtime._device_slot_selection = True
        bank = runtime._graph_base_slot
        self.launcher(
            source.data_ptr(),
            output.data_ptr(),
            self.selector.data_ptr(),
            runtime._staging_ptrs[bank],
            runtime._signal_ptrs,
            source.shape[0],
            runtime._slot_bytes if bank == 0 else -runtime._slot_bytes,
            runtime._exchange_offset,
            runtime._exchange_counter_offset,
        )
        return output

    def run(self, *, local_input, out):
        return self.runtime._all_gather_heads_on_device(
            local_input,
            out,
            state=self,
            threads=self.query.call["threads"],
            block_limit=self.query.call["block_limit"],
        )


def plan(query, *, runtime, invocation, override):
    from ._runtime_gather_cute import compile_selected_gather

    validate_query(query)
    if invocation:
        raise ValueError("selected gather invocation semantics belong in its query")
    for field in (
        "max_batch_size",
        "total_heads",
        "query_head_dim",
        "slot_bytes",
        "exchange_chunks",
        "exchange_offset",
        "exchange_counter_offset",
    ):
        name = (
            field
            if field in ("max_batch_size", "total_heads", "query_head_dim")
            else "_" + field
        )
        if query.setup[field] != getattr(runtime, name):
            raise ValueError(f"selected gather {field} differs from its runtime owner")
    c = query.call
    element = 1 if c["dtype"] == "torch.float8_e4m3fn" else 2
    args = (
        query.world_size,
        query.rank,
        prod(c["shape"][1:]) * element // 4,
        c["threads"],
        c["block_limit"],
        query.setup["exchange_chunks"],
        c["gather_transport"].removeprefix("selected_"),
    )

    def materialize(selection, device):
        launcher = load_programs(compile_selected_gather(*args))
        if c["gather_transport"] in ("selected_phased_barrier", "selected_phased_push"):
            from ._launch_geometry import validate_cluster_residency

            validate_cluster_residency(launcher, c["block_limit"], c["threads"], 8)
        value = selected_gather_profile(
            c["gather_transport"], c["block_limit"], c["submode"], c["active_blocks"]
        )
        selector = torch.tensor([value], dtype=torch.uint32, device=runtime.device)
        return SelectedGatherState(query, runtime, launcher, selector)

    resident = sum(buffer.nbytes for buffer in runtime._owned_buffers)
    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._runtime_gather_cute:compile_selected_gather", *args
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(("pcie-dcp-runtime", id(runtime)), resident, resident),
                PersistentMemory(
                    ("pcie-gather-selector", current_plan(), config), 4, 4
                ),
            )
        ),
        _materialize=materialize,
        _device=runtime.device,
    )


def validate_varlen_buffers(runtime, output, counts):
    if (
        output.dtype != torch.uint8
        or output.ndim != 2
        or output.shape[0] != runtime.world_size
        or not output.is_contiguous()
        or output.shape[1] <= 0
        or output.shape[1] % 16
        or output.data_ptr() % 16
        or counts.shape != (runtime.world_size,)
        or counts.dtype != torch.int64
        or not counts.is_contiguous()
        or counts.data_ptr() % 8
        or output.device != runtime.device
        or counts.device != runtime.device
    ):
        raise ValueError(
            "variable gather requires contiguous CUDA byte rows and int64 counts on its device"
        )


@dataclass
class VarlenGatherState:
    query: object
    runtime: object
    launcher: object

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("variable gather plan belongs to another channel")

    def run(self, *, out, counts):
        runtime = self.runtime
        if out.shape[1] != self.query.call["capacity"]:
            raise ValueError("variable gather row capacity differs from its plan")
        runtime._check_stream()
        if not runtime._device_slot_selection:
            runtime._graph_base_slot = runtime._next_slot & 1
            runtime._device_slot_selection = True
        bank = runtime._graph_base_slot
        self.launcher(
            out.data_ptr(),
            counts.data_ptr(),
            runtime._staging_ptrs[bank],
            runtime._signal_ptrs,
            runtime._slot_bytes if bank == 0 else -runtime._slot_bytes,
        )
        return out


def varlen_plan(query, *, runtime, invocation, override):
    from ._ring_gather_cute import compile_varlen_ring

    TUNING.validate_query(query, None)
    for field in ("slot_bytes", "exchange_chunks", "exchange_offset"):
        if query.setup[field] != getattr(runtime, "_" + field):
            raise ValueError(f"variable gather {field} differs from its runtime owner")
    args = (
        query.world_size,
        query.rank,
        query.call["capacity"],
        query.call["bank_bytes"],
    )
    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._ring_gather_cute:compile_varlen_ring", *args
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(
                    ("pcie-dcp-runtime", id(runtime)),
                    sum(b.nbytes for b in runtime._owned_buffers),
                    sum(b.nbytes for b in runtime._owned_buffers),
                ),
            )
        ),
        _materialize=lambda selection, device: VarlenGatherState(
            query, runtime, load_programs(compile_varlen_ring(*args))
        ),
        _device=runtime.device,
    )
