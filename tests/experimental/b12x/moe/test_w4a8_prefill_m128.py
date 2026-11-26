"""Validate W4A8 MX M128xN64 prefill kernels and multi-CTA Triton route planner."""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from b12x.loader._csf_blocks import export_csf_blocks
from b12x.moe import fused_moe
from b12x.moe.fused_moe._impl import _heuristic_moe_decode_config
from b12x.moe.fused_moe._tuning import (
    MoeDecodeConfig,
    MoeDecodeQuery,
    validate_moe_decode_config,
)
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import FrozenMapping
from b12x.quantization import mxfp8
from tests.experimental.b12x.conftest import require_b12x


_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_fp4_e8m0(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize [E, rows, cols // 2] packed E2M1 weights with UE8M0 block-32 scales."""
    lut = _E2M1_LUT.to(packed.device)
    lo = lut[(packed & 0x0F).to(torch.long)]
    hi = lut[((packed >> 4) & 0x0F).to(torch.long)]
    vals = torch.stack((lo, hi), dim=-1).reshape(
        packed.shape[0], packed.shape[1], packed.shape[2] * 2
    )
    sf = torch.pow(2.0, scales.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    return vals * sf


def _dequant_mxfp8_rows(vals: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    sf = (
        torch.pow(2.0, scales.reshape(vals.shape[0], -1).to(torch.float32) - 127.0)
        .repeat_interleave(32, dim=-1)
    )
    return vals.to(torch.float32) * sf


def _fp32_dequant_moe_reference(
    a_vals: torch.Tensor,
    a_scales: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    w13_layout: str = "w13",
) -> torch.Tensor:
    """FP32 dequantized reference for w4a8_mx SiLU MoE (w13=[gate; up] or w31=[up; gate])."""
    M, K = a_vals.shape
    E, two_n, _ = w13.shape
    N = two_n // 2
    x = _dequant_mxfp8_rows(a_vals, a_scales)
    w13_f32 = _dequant_fp4_e8m0(w13, s13)
    w2_f32 = _dequant_fp4_e8m0(w2, s2)
    out = torch.zeros((M, K), dtype=torch.float32, device=a_vals.device)
    ids_cpu = topk_ids.cpu()
    for e in range(E):
        tok_idx, slot_idx = torch.where(ids_cpu == e)
        if tok_idx.numel() == 0:
            continue
        tok_idx = tok_idx.to(a_vals.device)
        slot_idx = slot_idx.to(a_vals.device)
        xe = x[tok_idx]
        if w13_layout == "w13":
            w_up, w_gate = w13_f32[e, :N], w13_f32[e, N:]
        else:
            w_gate, w_up = w13_f32[e, :N], w13_f32[e, N:]
        gate = xe @ w_gate.t()
        up = xe @ w_up.t()
        act = (F.silu(gate) * up).to(torch.bfloat16).to(torch.float32)
        down = act @ w2_f32[e].t()
        weights = topk_weights[tok_idx, slot_idx].unsqueeze(-1)
        out.index_add_(0, tok_idx, down * weights)
    return out.to(torch.bfloat16)


def _quantize_mxfp8(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = x_bf16.shape
    device = x_bf16.device
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
    return vals, srows


def _make_csf_planes(
    num_experts: int, rows: int, cols: int, seed: int
) -> tuple[fused_moe.CsfBlockPlanes, torch.Tensor]:
    rng = np.random.default_rng(seed)
    planes = []
    grids = []
    for _ in range(num_experts):
        bases = rng.integers(121, 126, rows, dtype=np.uint8)
        bits = rng.integers(0, 2, (rows, cols), dtype=np.uint8)
        logical = bases[:, None] + bits
        positions = np.arange(0, rows * cols, 97, dtype=np.uint32)
        values = rng.integers(119, 127, len(positions), dtype=np.uint32)
        logical.ravel()[positions] = values
        selectors = np.packbits(bits, axis=1, bitorder="little")
        fixed = torch.from_numpy(
            np.concatenate(
                (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
            )
        )
        exc = torch.from_numpy(positions | (values << 24))
        planes.append(
            export_csf_blocks(
                fixed,
                exc,
                rows=rows,
                columns=cols,
                quantization="mxfp4",
            )
        )
        grids.append(logical)
    return fused_moe.CsfBlockPlanes(tuple(planes)), torch.from_numpy(np.stack(grids))


def _build_experts(
    E: int,
    K: int,
    N: int,
    device: torch.device,
    *,
    csf: bool = False,
    seed: int = 42,
    w13_layout: str = "w13",
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), generator=gen, dtype=torch.uint8).to(
        device
    )
    w2 = torch.randint(0, 256, (E, K, N // 2), generator=gen, dtype=torch.uint8).to(
        device
    )
    ones = torch.ones(E, dtype=torch.float32, device=device)
    wplan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout=w13_layout),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=None
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=E, hidden_size=K, intermediate_size=N
        ),
    )
    if csf:
        w13_csf, s13_cpu = _make_csf_planes(E, 2 * N, K // 32, seed + 101)
        w2_csf, s2_cpu = _make_csf_planes(E, K, N // 32, seed + 202)
        s13 = s13_cpu.to(device)
        s2 = s2_cpu.to(device)
        experts = fused_moe.prepare_weights(
            plan=wplan,
            weights=fused_moe.Mxfp4CsfWeights(
                w13=w13.clone(),
                w2=w2.clone(),
                w13_scales=w13_csf,
                w2_scales=w2_csf,
                w13_scale_scratch=torch.empty_like(s13),
                w2_scale_scratch=torch.empty_like(s2),
            ),
        )
    else:
        s13 = torch.randint(
            122, 127, (E, 2 * N, K // 32), generator=gen, dtype=torch.uint8
        ).to(device)
        s2 = torch.randint(
            122, 127, (E, K, N // 32), generator=gen, dtype=torch.uint8
        ).to(device)
        experts = fused_moe.prepare_weights(
            plan=wplan,
            weights=fused_moe.PackedWeights(
                w13=w13.clone(),
                w2=w2.clone(),
                w13_block_scales=s13.clone(),
                w2_block_scales=s2.clone(),
                w13_global_scales=ones,
                w2_global_scales=ones.clone(),
            ),
        )
    return experts, w13, s13, w2, s2


def _make_skewed_routes(
    M: int,
    top_k: int,
    E: int,
    *,
    counts: tuple[int, ...] = (0, 1, 17, 32, 33, 64, 65, 128, 129, 400),
    inactive_routes: int = 0,
    dtype: torch.dtype = torch.int32,
    seed: int = 77,
    device: torch.device,
) -> torch.Tensor:
    """Build routes with exact per-expert token counts covering all M128/M64/M32 boundaries."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total = M * top_k
    flat: list[int] = []
    for eid, cnt in enumerate(counts):
        if eid < E and len(flat) + cnt <= total - inactive_routes:
            flat.extend([eid] * cnt)
    if inactive_routes > 0:
        flat.extend([-1] * inactive_routes)
    rem = total - len(flat)
    if rem > 0:
        start_e = min(len(counts), max(E - 1, 0))
        flat.extend(torch.randint(start_e, E, (rem,), generator=gen).tolist())
    flat = flat[:total]
    perm = torch.randperm(total, generator=gen)
    return torch.tensor(flat, dtype=dtype)[perm].view(M, top_k).to(device).contiguous()


def _make_uniform_routes(
    M: int, top_k: int, E: int, *, seed: int = 33, device: torch.device
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((M, E), generator=gen, dtype=torch.float32)
    return torch.topk(scores, k=top_k, dim=-1).indices.to(
        device=device, dtype=torch.int32
    ).contiguous()


def _run_moe(
    experts,
    *,
    x_bf16: torch.Tensor | None = None,
    a_vals: torch.Tensor | None = None,
    a_scales: torch.Tensor | None = None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tile_m: int,
    route_planner: str,
    csf_inline: bool = False,
    deterministic: bool = True,
) -> torch.Tensor:
    prequant = a_vals is not None
    M = a_vals.shape[0] if prequant else x_bf16.shape[0]
    K = a_vals.shape[1] if prequant else x_bf16.shape[1]
    top_k = topk_ids.shape[1]
    device = topk_ids.device
    out = torch.full((M, K), -99.0, dtype=torch.bfloat16, device=device)

    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=M,
            top_k=top_k,
            input_format="mxfp8" if prequant else "bf16",
        ),
        routing=fused_moe.RoutingSpec(deterministic_output=deterministic),
        override=MoeDecodeConfig(
            backend="dynamic",
            route_planner=route_planner,
            max_active_clusters=None,
            dynamic_tile_m=tile_m,
            dynamic_route_mode="grouped",
            csf_inline_scales=csf_inline,
        ),
    )
    session = PreparationSession(compile_workers=0, autotune=False)
    box = {}

    def _prep(state):
        sc = [
            torch.zeros(spec.nbytes, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        ]
        if prequant:
            b = state.bind(
                a=a_vals,
                a_scales=a_scales,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=out,
                scratch=sc,
                input_scales_static=True,
            )
        else:
            b = state.bind(
                a=x_bf16,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=out,
                scratch=sc,
                input_scales_static=True,
            )
        box["state"] = state
        box["b"] = b
        return PreparedCall(run=lambda: state.run(b))

    session.prepare(
        (plan.request(name=f"m{tile_m}_{route_planner}_{int(deterministic)}", prepare_call=_prep),)
    )
    out.fill_(-99.0)
    box["state"].run(box["b"])
    torch.cuda.synchronize()
    return out


def test_w4a8_m128_validation_and_heuristics():
    """Config validation and heuristic selection for M128 compact-N64 and Triton planner."""
    base_kwargs = dict(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=384,
        hidden_size=5120,
        intermediate_size=576,
        top_k=6,
        route_num_experts=None,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=None,
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
    for num_tokens, expected_tile, expected_planner in (
        (2048, 64, "triton"),
        (4096, 128, "internal"),
        (8192, 128, "internal"),
    ):
        q = MoeDecodeQuery(
            **base_kwargs,
            num_tokens=num_tokens,
            routed_rows=num_tokens * 6,
            prequantized_input=True,
        )
        cfg = _heuristic_moe_decode_config(q, None)
        assert cfg.backend == "dynamic"
        assert cfg.dynamic_route_mode == "grouped"
        assert cfg.dynamic_tile_m == expected_tile
        assert cfg.route_planner == expected_planner
        validate_moe_decode_config(q, cfg, None)

    # Compact N64 M128 with inline CSF scales validates cleanly.
    q_csf = MoeDecodeQuery(
        **base_kwargs,
        num_tokens=4096,
        routed_rows=24576,
        prequantized_input=True,
        csf_inline_scales=True,
    )
    cfg_csf = MoeDecodeConfig(
        backend="dynamic",
        route_planner="triton",
        max_active_clusters=None,
        dynamic_tile_m=128,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
    )
    validate_moe_decode_config(q_csf, cfg_csf, None)

    # Non-prequantized M128 validates with internal route planner.
    q_bf16 = MoeDecodeQuery(
        **base_kwargs,
        num_tokens=4096,
        routed_rows=24576,
        prequantized_input=False,
    )
    cfg_bf16 = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=128,
        dynamic_route_mode="grouped",
    )
    validate_moe_decode_config(q_bf16, cfg_bf16, None)

    # Triton route planner supports up to 131072 routed rows (M=16384, k=8) and rejects >131072.
    q_max = MoeDecodeQuery(
        **{**base_kwargs, "intermediate_size": 1024, "top_k": 8},
        num_tokens=16384,
        routed_rows=131072,
        prequantized_input=True,
    )
    validate_moe_decode_config(
        q_max,
        MoeDecodeConfig(
            backend="dynamic",
            route_planner="triton",
            max_active_clusters=None,
            dynamic_tile_m=128,
            dynamic_route_mode="grouped",
        ),
        None,
    )
    q_over = MoeDecodeQuery(
        **{**base_kwargs, "intermediate_size": 1024, "top_k": 8},
        num_tokens=16385,
        routed_rows=131080,
        prequantized_input=True,
    )
    with pytest.raises(ValueError, match="Triton route planner"):
        validate_moe_decode_config(
            q_over,
            MoeDecodeConfig(
                backend="dynamic",
                route_planner="triton",
                max_active_clusters=None,
                dynamic_tile_m=128,
                dynamic_route_mode="grouped",
            ),
            None,
        )


@pytest.mark.parametrize(
    ("N", "csf_inline"),
    [(576, False), (576, True), (768, False), (1024, False)],
)
@pytest.mark.parametrize("routing", ["uniform", "skewed"])
@pytest.mark.parametrize("prequant", [True, False])
def test_w4a8_m128_exact_0ulp_parity_vs_m64(
    N: int, csf_inline: bool, routing: str, prequant: bool
):
    """Strict 0-ULP BF16 bitwise parity between M128 and baseline M64 across N, CSF, and skew."""
    device = require_b12x()
    E, K, M, top_k = 32, 5120, 512, 6
    experts, *_ = _build_experts(E, K, N, device, csf=csf_inline, seed=42 + N)
    gen = torch.Generator(device="cpu").manual_seed(101 + N)
    x_bf16 = (torch.randn((M, K), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    a_vals, a_scales = _quantize_mxfp8(x_bf16) if prequant else (None, None)

    if routing == "uniform":
        topk_ids = _make_uniform_routes(M, top_k, E, seed=19 + N, device=device)
    else:
        topk_ids = _make_skewed_routes(
            M,
            top_k,
            E,
            counts=(0, 1, 17, 32, 33, 64, 65, 128, 129, 400),
            inactive_routes=24,
            dtype=torch.int64 if prequant else torch.int32,
            seed=77 + N,
            device=device,
        )
    topk_w = torch.softmax(torch.randn((M, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    topk_w = torch.where(topk_ids >= 0, topk_w, torch.zeros_like(topk_w))

    ref = _run_moe(
        experts,
        x_bf16=None if prequant else x_bf16,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=64,
        route_planner="internal",
        csf_inline=csf_inline,
        deterministic=True,
    )
    got = _run_moe(
        experts,
        x_bf16=None if prequant else x_bf16,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=128,
        route_planner="internal",
        csf_inline=csf_inline,
        deterministic=True,
    )
    assert torch.isfinite(ref).all() and torch.count_nonzero(ref) > 0
    assert torch.equal(ref.view(torch.uint16), got.view(torch.uint16))

    if prequant:
        got_tri = _run_moe(
            experts,
            a_vals=a_vals,
            a_scales=a_scales,
            topk_ids=topk_ids,
            topk_weights=topk_w,
            tile_m=128,
            route_planner="triton",
            csf_inline=csf_inline,
            deterministic=False,
        )
        ref_nondet = _run_moe(
            experts,
            a_vals=a_vals,
            a_scales=a_scales,
            topk_ids=topk_ids,
            topk_weights=topk_w,
            tile_m=64,
            route_planner="internal",
            csf_inline=csf_inline,
            deterministic=False,
        )
        denom = ref.float().abs().max().clamp(min=1.0)
        rel_tri = (ref.float() - got_tri.float()).abs().max() / denom
        rel_base = (ref.float() - ref_nondet.float()).abs().max() / denom
        assert rel_tri.item() <= max(rel_base.item() * 2.0, 0.02)

    del experts, x_bf16, a_vals, a_scales, topk_ids, topk_w, ref, got
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("N", [576, 768, 1024])
def test_w4a8_m128_matches_fp32_dequant_reference(N: int):
    """Verify M128 exact 0-ULP vs M64 AND close agreement with FP32 dequantized MoE math."""
    device = require_b12x()
    E, K, M, top_k = 16, 512, 256, 4
    experts, w13, s13, w2, s2 = _build_experts(E, K, N, device, csf=False, seed=300 + N)
    gen = torch.Generator(device="cpu").manual_seed(400 + N)
    x_bf16 = (torch.randn((M, K), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    a_vals, a_scales = _quantize_mxfp8(x_bf16)
    topk_ids = _make_skewed_routes(
        M,
        top_k,
        E,
        counts=(0, 1, 17, 32, 33, 64, 65, 128, 129, 200),
        inactive_routes=12,
        seed=500 + N,
        device=device,
    )
    topk_w = torch.softmax(torch.randn((M, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    topk_w = torch.where(topk_ids >= 0, topk_w, torch.zeros_like(topk_w))

    ref_fp32 = _fp32_dequant_moe_reference(
        a_vals, a_scales, w13, s13, w2, s2, topk_ids, topk_w
    )
    out_m64 = _run_moe(
        experts,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=64,
        route_planner="internal",
        deterministic=True,
    )
    out_m128 = _run_moe(
        experts,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=128,
        route_planner="internal",
        deterministic=True,
    )
    out_triton = _run_moe(
        experts,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=128,
        route_planner="triton",
        deterministic=False,
    )

    assert torch.equal(out_m64.view(torch.uint16), out_m128.view(torch.uint16))
    cos = F.cosine_similarity(
        out_m128.float().flatten(), ref_fp32.float().flatten(), dim=0
    )
    cos_tri = F.cosine_similarity(
        out_triton.float().flatten(), ref_fp32.float().flatten(), dim=0
    )
    assert cos.item() > 0.999
    assert cos_tri.item() > 0.999
    rel_err = (out_m128.float() - ref_fp32.float()).norm() / ref_fp32.float().norm()
    assert rel_err.item() < 0.03


@pytest.mark.parametrize("M", [128, 1024, 4096, 8192])
def test_w4a8_m128_dsv41_large_m_parity(M: int):
    """Verify real DeepSeek-V4.1-Flash TP4 geometry (E=384, k=6, N=576, H=5120) up to M=8192."""
    device = require_b12x()
    E, K, N, top_k = 384, 5120, 576, 6
    experts, *_ = _build_experts(E, K, N, device, csf=False, seed=888)
    gen = torch.Generator(device="cpu").manual_seed(900 + M)
    x_bf16 = (torch.randn((M, K), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    a_vals, a_scales = _quantize_mxfp8(x_bf16)
    del x_bf16
    topk_ids = _make_skewed_routes(
        M,
        top_k,
        E,
        counts=(0, 1, 17, 32, 33, 64, 65, 128, 129, 400),
        seed=950 + M,
        device=device,
    )
    topk_w = torch.softmax(torch.randn((M, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )

    ref = _run_moe(
        experts,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=64,
        route_planner="internal",
        deterministic=True,
    )
    got_det = _run_moe(
        experts,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=128,
        route_planner="internal",
        deterministic=True,
    )
    assert torch.equal(ref.view(torch.uint16), got_det.view(torch.uint16))

    got_tri = _run_moe(
        experts,
        a_vals=a_vals,
        a_scales=a_scales,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        tile_m=128,
        route_planner="triton",
        deterministic=False,
    )
    cos = F.cosine_similarity(
        got_tri.float().flatten(), ref.float().flatten(), dim=0
    )
    assert cos.item() > 0.9999

    del experts, a_vals, a_scales, topk_ids, topk_w, ref, got_det, got_tri
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("csf_inline", [False, True])
def test_w4a8_m128_triton_cuda_graph_capture_and_live_route_mutation(csf_inline: bool):
    """Zero-allocation CUDA-graph capture and replay with live route/input mutation."""
    device = require_b12x()
    E, K, N, M, top_k = 64, 5120, 576, 512, 6
    experts, *_ = _build_experts(E, K, N, device, csf=csf_inline, seed=600)
    gen = torch.Generator(device="cpu").manual_seed(601)
    x1 = (torch.randn((M, K), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    a1, s1 = _quantize_mxfp8(x1)
    ids1 = _make_uniform_routes(M, top_k, E, seed=602, device=device)
    w1 = torch.softmax(torch.randn((M, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    out = torch.full((M, K), -99.0, dtype=torch.bfloat16, device=device)

    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=M, top_k=top_k, input_format="mxfp8"
        ),
        override=MoeDecodeConfig(
            backend="dynamic",
            route_planner="triton",
            max_active_clusters=None,
            dynamic_tile_m=128,
            dynamic_route_mode="grouped",
            csf_inline_scales=csf_inline,
        ),
    )
    session = PreparationSession(compile_workers=0, autotune=False)
    box = {}

    def _prep(state):
        sc = [
            torch.zeros(spec.nbytes, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        ]
        b = state.bind(
            a=a1,
            a_scales=s1,
            topk_ids=ids1,
            topk_weights=w1,
            output=out,
            scratch=sc,
            input_scales_static=True,
        )
        box["state"] = state
        box["b"] = b
        return PreparedCall(run=lambda: state.run(b))

    session.prepare((plan.request(name="graph_m128", prepare_call=_prep),))
    st, b = box["state"], box["b"]

    # Warmup then capture in a CUDA graph.
    st.run(b)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        st.run(b)
    torch.cuda.synchronize()

    # Mutate inputs, routes (skew + inactive), and poison output buffer before replay.
    x2 = (torch.randn((M, K), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    a2, s2 = _quantize_mxfp8(x2)
    ids2 = _make_skewed_routes(
        M,
        top_k,
        E,
        counts=(0, 1, 17, 32, 33, 64, 65, 128, 129, 300),
        inactive_routes=18,
        seed=603,
        device=device,
    )
    w2 = torch.softmax(torch.randn((M, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    w2 = torch.where(ids2 >= 0, w2, torch.zeros_like(w2))
    a1.copy_(a2)
    s1.copy_(s2)
    ids1.copy_(ids2)
    w1.copy_(w2)
    out.fill_(-99.0)

    alloc_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == alloc_before
    replayed = out.clone()

    ref = _run_moe(
        experts,
        a_vals=a2,
        a_scales=s2,
        topk_ids=ids2,
        topk_weights=w2,
        tile_m=64,
        route_planner="internal",
        csf_inline=csf_inline,
        deterministic=True,
    )
    assert torch.isfinite(replayed).all()
    assert (replayed != -99.0).any()
    cos = F.cosine_similarity(
        replayed.float().flatten(), ref.float().flatten(), dim=0
    )
    assert cos.item() > 0.9999

    graph.reset()
    gc.collect()
    torch.cuda.empty_cache()
