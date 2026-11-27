"""Compile jobs and caller-owned bindings for split normalization."""

from dataclasses import dataclass
import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from ._tuning import Query, Config, TUNING


@program_cache(scope="preparation")
def compile_norm(payload, pdl, ordinal):
    from ._kernels import mla_norm_mxfp8_kernel

    query = Query(**dict(payload))
    q_weight_dtype = getattr(torch, query.query_weight_dtype)
    kv_weight_dtype = getattr(torch, query.kv_weight_dtype)
    rope_args = (
        (
            torch.float32,
            torch.float32,
            query.kv_rope_dim // 2,
            query.kv_rope_dim // 2,
            query.kv_rope_dim,
        )
        if query.kv_rope_dim > 0
        else (None, None, 0, 0, 0)
    )
    with torch.cuda.device(ordinal):
        return mla_norm_mxfp8_kernel.warmup(
            torch.bfloat16,
            q_weight_dtype,
            kv_weight_dtype,
            torch.bfloat16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            query.query_features,
            query.kv_features,
            query.epsilon,
            query.min_amax,
            pdl,
            query.release_dependents,
            *rope_args,
            grid=(query.rows, 2, 1),
            num_warps=8,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )


@dataclass(frozen=True)
class State:
    query: Query
    device: torch.device
    programmatic_serialization: bool
    program: object

    def run(
        self,
        projected,
        qw,
        kw,
        query,
        kv,
        values_base,
        scale_rows_base,
        scale_mma_base,
        *,
        cos=None,
        sin=None,
    ):
        q = self.query
        from b12x.quantization import mxfp8

        values_shape, rows_shape, mma_shape = mxfp8.dense_rows_storage_shapes(
            q.rows, q.query_features
        )
        facts = [
            (projected, (q.rows, q.query_features + q.kv_features), torch.bfloat16),
            (qw, (q.query_features,), getattr(torch, q.query_weight_dtype)),
            (kw, (q.kv_features,), getattr(torch, q.kv_weight_dtype)),
            (query, (q.rows, q.query_features), torch.bfloat16),
            (kv, (q.rows, q.kv_features), torch.bfloat16),
            (values_base, values_shape, torch.float8_e4m3fn),
            (scale_rows_base, rows_shape, torch.uint8),
            (scale_mma_base, mma_shape, torch.uint8),
        ]
        for tensor, shape, dtype in facts:
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "MLA normalization operand differs from its declared device, dtype, shape or alignment"
                )
        for position, (left, _, _) in enumerate(facts):
            for index, (right, _, _) in enumerate(facts[position + 1 :], position + 1):
                a, b = left.data_ptr(), right.data_ptr()
                if (
                    a < b + right.numel() * right.element_size()
                    and b < a + left.numel() * left.element_size()
                ):
                    raise ValueError(
                        "MLA normalization requires disjoint projected input, affine weights and outputs"
                    )
        if q.kv_rope_dim > 0:
            half_rope = q.kv_rope_dim // 2
            if cos is None or sin is None:
                raise ValueError(
                    "MLA normalization with fused KV RoPE requires cos and sin tensors"
                )
            for angle in (cos, sin):
                if (
                    angle.device != self.device
                    or angle.dtype != torch.float32
                    or tuple(angle.shape) != (q.rows, half_rope)
                    or angle.stride(1) != 1
                    or angle.data_ptr() % 16
                    or (angle.stride(0) * angle.element_size()) % 16
                ):
                    raise ValueError(
                        "MLA normalization KV RoPE cos/sin must be 16-byte aligned FP32 [rows, kv_rope_dim/2] tensors"
                    )
                angle_start = angle.data_ptr()
                angle_span = (
                    (q.rows - 1) * angle.stride(0) + half_rope
                ) * angle.element_size()
                for output_tensor, _, _ in facts[3:]:
                    out_start = output_tensor.data_ptr()
                    out_end = (
                        out_start + output_tensor.numel() * output_tensor.element_size()
                    )
                    if angle_start < out_end and out_start < angle_start + angle_span:
                        raise ValueError(
                            "MLA normalization KV RoPE angles must not overlap outputs"
                        )
            cos_stride, sin_stride = int(cos.stride(0)), int(sin.stride(0))
        else:
            if cos is not None or sin is not None:
                raise ValueError(
                    "MLA normalization cos/sin requires kv_rope_dim > 0 in Query"
                )
            cos_stride = sin_stride = 0
        self.program[(q.rows, 2, 1)](
            projected,
            qw,
            kw,
            query,
            kv,
            values_base,
            scale_rows_base,
            scale_mma_base,
            q.query_features,
            q.kv_features,
            q.epsilon,
            q.min_amax,
            self.programmatic_serialization,
            q.release_dependents,
            cos,
            sin,
            cos_stride,
            sin_stride,
            q.kv_rope_dim,
        )


def plan(query, *, invocation=FrozenMapping(), override=None):
    if not isinstance(query, Query):
        raise TypeError("MLA normalization requires Query")
    if invocation:
        raise ValueError("MLA normalization semantics belong in Query")

    def materialize(selection, device):
        program = compile_norm(
            TUNING.encode_query(query),
            selection.config.programmatic_serialization,
            device.ordinal,
        )
        load_programs(program)
        return attach_programs(
            State(
                query,
                torch.device("cuda", device.ordinal),
                selection.config.programmatic_serialization,
                program,
            ),
            program,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        override=override,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.quantization.mla_norm_mxfp8._preparation:compile_norm",
                TUNING.encode_query(query),
                config.programmatic_serialization,
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda c, d: MemoryRequirements(),
        _materialize=materialize,
    )


@torch.library.custom_op(
    "b12x::mla_norm_mxfp8_prepared",
    mutates_args=("query", "kv", "values_base", "scale_rows_base", "scale_mma_base"),
)
def _execute(
    projected: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    values_base: torch.Tensor,
    scale_rows_base: torch.Tensor,
    scale_mma_base: torch.Tensor,
    handle: int,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "quantization.mla_norm_mxfp8", projected.device
    )
    state.run(
        projected,
        qw,
        kw,
        query,
        kv,
        values_base,
        scale_rows_base,
        scale_mma_base,
        cos=cos,
        sin=sin,
    )


@_execute.register_fake
def _fake(
    projected,
    qw,
    kw,
    query,
    kv,
    values_base,
    scale_rows_base,
    scale_mma_base,
    handle,
    cos=None,
    sin=None,
):
    pass


def run(
    projected,
    qw,
    kw,
    query,
    kv,
    values_base,
    scale_rows_base,
    scale_mma_base,
    *,
    plan,
    cos=None,
    sin=None,
):
    _execute(
        projected,
        qw,
        kw,
        query,
        kv,
        values_base,
        scale_rows_base,
        scale_mma_base,
        plan.handle,
        cos,
        sin,
    )
