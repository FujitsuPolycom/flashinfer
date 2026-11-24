"""Validate pre-quantized MXFP8 inputs for grouped dynamic W4A8 MX MoE."""

from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe._binding_ops import (
    prepared_moe_mxfp8_out,
    prepared_moe_out,
)
from b12x.moe.fused_moe._impl import _heuristic_moe_decode_config
from b12x.moe.fused_moe._tuning import (
    MoeDecodeConfig,
    MoeDecodeQuery,
    validate_moe_decode_config,
)
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import FrozenMapping
from tests.experimental.b12x.conftest import require_b12x


def test_execution_capacity_and_query_prequantized_input_validation():
    cap_default = fused_moe.ExecutionCapacity(max_tokens=128, top_k=6)
    assert cap_default.input_format == "bf16"

    cap_mx = fused_moe.ExecutionCapacity(max_tokens=128, top_k=6, input_format="mxfp8")
    assert cap_mx.input_format == "mxfp8"

    with pytest.raises(ValueError, match="input_format"):
        fused_moe.ExecutionCapacity(max_tokens=128, top_k=6, input_format="fp16")

    base_kwargs = dict(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=32,
        hidden_size=512,
        intermediate_size=256,
        top_k=4,
        num_tokens=128,
        routed_rows=512,
        route_num_experts=None,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        w13_layout="w13",
        weight_layouts=("w13",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=False,
        numerical_recipe=None,
        controls=FrozenMapping({}),
    )
    q_false = MoeDecodeQuery(**base_kwargs, prequantized_input=False)
    assert "prequantized_input" not in q_false.to_dict()

    q_true = MoeDecodeQuery(**base_kwargs, prequantized_input=True)
    assert q_true.to_dict()["prequantized_input"] is True

    valid_cfg = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=64,
        dynamic_route_mode="grouped",
    )
    validate_moe_decode_config(q_true, valid_cfg, None)

    for bad_cfg, bad_query in (
        (
            MoeDecodeConfig(
                backend="micro", route_planner="internal", max_active_clusters=None
            ),
            q_true,
        ),
        (
            MoeDecodeConfig(
                backend="dynamic",
                route_planner="internal",
                max_active_clusters=None,
                dynamic_tile_m=64,
                dynamic_route_mode="direct",
            ),
            q_true,
        ),
        (
            MoeDecodeConfig(
                backend="dynamic",
                route_planner="internal",
                max_active_clusters=None,
                dynamic_tile_m=16,
                dynamic_route_mode="grouped",
            ),
            q_true,
        ),
        (
            valid_cfg,
            MoeDecodeQuery(
                **{**base_kwargs, "quant_mode": "nvfp4"}, prequantized_input=True
            ),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="prequantized MXFP8 MoE inputs require grouped dynamic W4A8 MX",
        ):
            validate_moe_decode_config(bad_query, bad_cfg, None)

    q_sparse = MoeDecodeQuery(
        **{
            **base_kwargs,
            "num_experts": 256,
            "hidden_size": 5120,
            "intermediate_size": 384,
            "top_k": 8,
            "num_tokens": 512,
            "routed_rows": 4096,
        },
        prequantized_input=True,
    )
    cfg_sparse = _heuristic_moe_decode_config(q_sparse, None)
    assert cfg_sparse.backend == "dynamic"
    assert cfg_sparse.dynamic_route_mode == "grouped"
    assert cfg_sparse.dynamic_tile_m == 32
    validate_moe_decode_config(q_sparse, cfg_sparse, None)

    q_n64 = MoeDecodeQuery(
        **{
            **base_kwargs,
            "num_experts": 384,
            "hidden_size": 5120,
            "intermediate_size": 576,
            "top_k": 6,
            "num_tokens": 2048,
            "routed_rows": 12288,
        },
        prequantized_input=True,
    )
    cfg_n64 = _heuristic_moe_decode_config(q_n64, None)
    assert cfg_n64.backend == "dynamic"
    assert cfg_n64.dynamic_route_mode == "grouped"
    assert cfg_n64.dynamic_tile_m == 64
    validate_moe_decode_config(q_n64, cfg_n64, None)


def test_dynamic_w4a8_mx_prequantized_matches_bf16_eager_and_compiled():
    device = require_b12x()
    from b12x.quantization import mxfp8

    E, K, N, M, top_k = 16, 512, 256, 256, 4
    gen = torch.Generator(device="cpu").manual_seed(2026)

    w13 = torch.randint(
        0, 256, (E, 2 * N, K // 2), generator=gen, dtype=torch.uint8
    ).to(device)
    s13 = torch.randint(
        120, 129, (E, 2 * N, K // 32), generator=gen, dtype=torch.uint8
    ).to(device)
    w2 = torch.randint(0, 256, (E, K, N // 2), generator=gen, dtype=torch.uint8).to(
        device
    )
    s2 = torch.randint(120, 129, (E, K, N // 32), generator=gen, dtype=torch.uint8).to(
        device
    )
    ones = torch.ones(E, dtype=torch.float32, device=device)

    wplan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=7.0
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=E, hidden_size=K, intermediate_size=N
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=wplan,
        weights=fused_moe.PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=s13,
            w2_block_scales=s2,
            w13_global_scales=ones,
            w2_global_scales=ones.clone(),
        ),
    )

    x_bf16 = (torch.randn((M, K), generator=gen) * 0.5).to(
        device=device, dtype=torch.bfloat16
    )
    # Mask trailing 24 rows as inactive (-1) to exercise padded prefill tail handling.
    live = M - 24
    x_bf16[live:].zero_()
    topk_ids = torch.stack(
        [torch.randperm(E, generator=gen)[:top_k] for _ in range(M)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[live:] = -1
    topk_weights = torch.softmax(torch.randn((M, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    topk_weights[live:] = 0.0

    shapes = mxfp8.dense_rows_storage_shapes(M, K)
    vals = torch.empty(shapes[0], dtype=torch.float8_e4m3fn, device=device)
    srows = torch.empty(shapes[1], dtype=torch.uint8, device=device)
    smma = torch.empty(shapes[2], dtype=torch.uint8, device=device)
    enc = mxfp8.dense_rows_from_bases(vals, srows, smma)
    q_plan = mxfp8.plan(
        mxfp8.query_from_call(
            x_bf16, enc.values, enc.scale_rows, enc.scale_mma, min_amax=0.0
        )
    )
    q_session = PreparationSession(compile_workers=0, autotune=False)
    q_session.prepare(
        (
            q_plan.request(
                name="quant_mxfp8",
                prepare_call=lambda st: PreparedCall(
                    run=lambda: st.run(
                        x_bf16, enc.values, enc.scale_rows, enc.scale_mma
                    )
                ),
            ),
        )
    )
    mxfp8.quantize_dense_rows(x_bf16, vals, srows, smma, plan=q_plan)

    plan_bf16 = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=M, top_k=top_k, input_format="bf16"
        ),
    )
    plan_mx = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=M, top_k=top_k, input_format="mxfp8"
        ),
    )
    out_probe = torch.zeros((M, K), dtype=torch.bfloat16, device=device)

    states = {}

    def prep_bf16(state):
        states["bf16"] = state
        sc = [
            torch.zeros(spec.nbytes, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        ]
        binding = state.bind(
            a=x_bf16,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=out_probe,
            scratch=sc,
            input_scales_static=True,
        )
        return PreparedCall(run=lambda: state.run(binding))

    def prep_mx(state):
        states["mx"] = state
        sc = [
            torch.zeros(spec.nbytes, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        ]
        binding = state.bind(
            a=vals,
            a_scales=srows,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=out_probe,
            scratch=sc,
            input_scales_static=True,
        )
        return PreparedCall(run=lambda: state.run(binding))

    moe_session = PreparationSession(compile_workers=0, autotune=False)
    moe_session.prepare(
        (
            plan_bf16.request(name="moe_bf16", prepare_call=prep_bf16),
            plan_mx.request(name="moe_mx", prepare_call=prep_mx),
        )
    )

    scratch_bf16 = torch.zeros(
        plan_bf16.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device
    )
    scratch_mx = torch.zeros(
        plan_mx.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device
    )

    # Validation checks on missing or unexpected a_scales.
    with pytest.raises(ValueError, match="requires a_scales"):
        states["mx"].bind(
            a=vals,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=out_probe,
            scratch=scratch_mx,
            input_scales_static=True,
        )
    with pytest.raises(ValueError, match="requires a prequantized_input"):
        states["bf16"].bind(
            a=x_bf16,
            a_scales=srows,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=out_probe,
            scratch=scratch_bf16,
            input_scales_static=True,
        )

    def run_bf16() -> torch.Tensor:
        out = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
        prepared_moe_out(
            x_bf16,
            topk_weights,
            topk_ids,
            out,
            scratch_bf16,
            [],
            [],
            plan_bf16.handle,
            True,
            None,
            None,
            None,
            None,
            None,
            False,
            None,
        )
        return out

    def run_mxfp8(v: torch.Tensor, sr: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
        prepared_moe_mxfp8_out(
            v,
            sr,
            topk_weights,
            topk_ids,
            out,
            scratch_mx,
            [],
            [],
            plan_mx.handle,
            True,
            None,
            None,
            None,
            None,
            None,
            False,
            None,
        )
        return out

    bf16_runs = [run_bf16().clone() for _ in range(16)]
    bf16_rep_max = max(
        (a.float() - b.float()).abs().max().item()
        for i, a in enumerate(bf16_runs)
        for b in bf16_runs[i + 1 :]
    )
    out_eager = run_mxfp8(vals, srows)
    compiled_fn = torch.compile(run_mxfp8, fullgraph=True)
    out_compiled = compiled_fn(vals, srows)
    torch.cuda.synchronize()

    assert torch.all(out_eager[live:] == 0)
    assert torch.all(out_compiled[live:] == 0)
    assert (bf16_runs[0].float() - out_eager.float()).abs().max().item() <= bf16_rep_max
    assert (
        bf16_runs[0].float() - out_compiled.float()
    ).abs().max().item() <= bf16_rep_max
