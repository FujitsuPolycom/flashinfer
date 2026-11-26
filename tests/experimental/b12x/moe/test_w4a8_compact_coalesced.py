"""0-ULP parity, CUDA-graph replay, and M<=128 prequant tests for coalesced W4A8CompactMicro."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.loader._csf_blocks import export_csf_blocks
from b12x.moe import fused_moe as moe
from b12x.moe._shared.kernels.w4a8_compact_micro import (
    launch_w4a8_compact_micro,
    micro_scratch_nbytes,
)
from b12x.moe.fused_moe._binding_ops import (
    prepared_moe_mxfp8_out,
    prepared_moe_out,
    prepared_moe_prequant_out,
    supports_prequant_mxfp8,
)
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
from tests.experimental.b12x.quantization.test_mxfp4_csf import fixture as csf_fixture


_FP4_LUT = torch.tensor(
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


def _dequant_mxfp4_e8m0(packed_u8: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize [E, N, K//2] E2M1 + [E, N, K//32] UE8M0 to FP32 [E, N, K]."""
    lut = _FP4_LUT.to(packed_u8.device)
    lo = lut[(packed_u8 & 0xF).to(torch.int64)]
    hi = lut[((packed_u8 >> 4) & 0xF).to(torch.int64)]
    vals = torch.stack((lo, hi), dim=-1).reshape(
        packed_u8.shape[0], packed_u8.shape[1], packed_u8.shape[2] * 2
    )
    exp = scales_u8.to(torch.int32) - 127
    scale = torch.where(
        scales_u8 == 0,
        torch.zeros_like(exp, dtype=torch.float32),
        torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp),
    )
    return vals * scale.repeat_interleave(32, dim=-1)


def _fp32_moe_reference(
    a_bf16: torch.Tensor,
    w13_u8: torch.Tensor,
    s13_u8: torch.Tensor,
    w2_u8: torch.Tensor,
    s2_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    swiglu_limit: float | None = 10.0,
) -> torch.Tensor:
    """Exact FP32 reference for W4A8 SiLU MoE (w13_layout='w13' = up_gate)."""
    w13 = _dequant_mxfp4_e8m0(w13_u8, s13_u8)
    w2 = _dequant_mxfp4_e8m0(w2_u8, s2_u8)
    n = w2.shape[2]
    t, k = a_bf16.shape
    e = w13.shape[0]
    x = a_bf16.float()
    out = torch.zeros((t, k), dtype=torch.float32, device=a_bf16.device)
    for tok in range(t):
        for slot in range(topk_ids.shape[1]):
            exp = int(topk_ids[tok, slot].item())
            if 0 <= exp < e:
                w_up = w13[exp, :n]
                w_gate = w13[exp, n:]
                up = x[tok] @ w_up.T
                gate = x[tok] @ w_gate.T
                if swiglu_limit is not None:
                    gate = gate.clamp(max=swiglu_limit)
                    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
                act = torch.nn.functional.silu(gate) * up
                proj2 = act @ w2[exp].T
                out[tok] += topk_weights[tok, slot].float() * proj2
    return out


def _make_experts(
    device: torch.device,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    csf: bool = False,
    seed: int = 2026,
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        generator=gen,
        dtype=torch.uint8,
    ).to(device)
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        generator=gen,
        dtype=torch.uint8,
    ).to(device)
    wplan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
        ),
        geometry=moe.MoEGeometry(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ),
    )
    if csf:
        first_batch, s13 = csf_fixture(
            2 * intermediate_size,
            hidden_size // 32,
            experts=num_experts,
            finite_scales=True,
        )
        second_batch, s2 = csf_fixture(
            hidden_size,
            intermediate_size // 32,
            experts=num_experts,
            finite_scales=True,
        )

        def to_planes(batch):
            offsets = batch.exception_offsets.cpu().tolist()
            records = batch.exceptions.cpu()
            return moe.CsfBlockPlanes(
                tuple(
                    export_csf_blocks(
                        f.cpu(),
                        records[a:b],
                        rows=batch.rows,
                        columns=batch.columns,
                        quantization="mxfp4",
                    )
                    for f, a, b in zip(
                        batch.fixed, offsets[:-1], offsets[1:], strict=True
                    )
                )
            )

        experts = moe.prepare_weights(
            plan=wplan,
            weights=moe.Mxfp4CsfWeights(
                w13=w13,
                w2=w2,
                w13_scales=to_planes(first_batch),
                w2_scales=to_planes(second_batch),
                w13_scale_scratch=torch.empty(
                    num_experts * 2 * intermediate_size * (hidden_size // 32),
                    dtype=torch.uint8,
                    device=device,
                ),
                w2_scale_scratch=torch.empty(
                    num_experts * hidden_size * (intermediate_size // 32),
                    dtype=torch.uint8,
                    device=device,
                ),
            ),
        )
    else:
        s13 = torch.randint(
            122,
            128,
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            generator=gen,
            dtype=torch.uint8,
        ).to(device)
        s2 = torch.randint(
            122,
            128,
            (num_experts, hidden_size, intermediate_size // 32),
            generator=gen,
            dtype=torch.uint8,
        ).to(device)
        w13_logical = w13.clone()
        s13_logical = s13.clone()
        w2_logical = w2.clone()
        s2_logical = s2.clone()
        ones = torch.ones(num_experts, dtype=torch.float32, device=device)
        experts = moe.prepare_weights(
            plan=wplan,
            weights=moe.PackedWeights(
                w13=w13,
                w2=w2,
                w13_block_scales=s13,
                w2_block_scales=s2,
                w13_global_scales=ones,
                w2_global_scales=ones.clone(),
            ),
        )
        return experts, w13_logical, s13_logical, w2_logical, s2_logical
    return experts, w13, s13, w2, s2


@pytest.fixture(scope="module")
def deepseek_experts_by_csf():
    """Build E=384, H=5120, N=576 experts once per module for both dense and CSF scales."""
    device = require_b12x()
    cache: dict[bool, object] = {}

    def _get(csf: bool):
        if csf not in cache:
            experts, _, _, _, _ = _make_experts(
                device, num_experts=384, hidden_size=5120, intermediate_size=576, csf=csf
            )
            cache[csf] = experts
        return cache[csf]

    yield _get
    cache.clear()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 7, 8, 12, 14, 16])
@pytest.mark.parametrize("csf_inline", [False, True])
@pytest.mark.parametrize("prequant", [False, True])
def test_w4a8_compact_coalesced_0ulp_parity_deepseek_shape(
    deepseek_experts_by_csf,
    num_tokens: int,
    csf_inline: bool,
    prequant: bool,
):
    """Strict 0-ULP bit-exact parity vs uncoalesced W4A8CompactMicro on E=384, k=6, N=576, H=5120."""
    device = require_b12x()
    k, n, top_k = 5120, 576, 6
    experts = deepseek_experts_by_csf(csf_inline)
    impl = experts._impl
    rep = impl.representation.value
    w13_scales = impl.nvfp4_csf.first.scales.storage if csf_inline else rep.w13_sfb
    w2_scales = impl.nvfp4_csf.second.scales.storage if csf_inline else rep.w2_sfb

    gen = torch.Generator(device="cpu").manual_seed(100 + num_tokens)
    a = (torch.randn((num_tokens, k), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    # Mix heavy expert sharing (all tokens hitting expert 7 -> tests >8 split at T=12,14,16),
    # partial overlap, and inactive -1 tail routes when T >= 4.
    ids = torch.randint(0, 18, (num_tokens, top_k), generator=gen, dtype=torch.int32)
    ids[:, 0] = 7
    if num_tokens >= 4:
        ids[-1, :] = -1
    ids = ids.to(device)
    weights = torch.softmax(
        torch.randn((num_tokens, top_k), generator=gen), dim=-1
    ).to(device=device, dtype=torch.float32)

    pre_v = pre_s = None
    if prequant:
        pre_v = torch.empty((num_tokens, k), dtype=torch.uint8, device=device)
        pre_s = torch.empty((num_tokens, k // 32), dtype=torch.uint8, device=device)
        scale_mma = torch.empty(
            ((num_tokens + 127) // 128) * (k // 128) * 512,
            dtype=torch.uint8,
            device=device,
        )
        quantize_mxfp8_rows_cute(a, pre_v, pre_s, scale_mma, expected_m=num_tokens)

    scratch = torch.empty(
        micro_scratch_nbytes(num_tokens, k, n, top_k), dtype=torch.uint8, device=device
    )
    out_ref = torch.empty((num_tokens, k), dtype=torch.bfloat16, device=device)
    out_got = torch.empty((num_tokens, k), dtype=torch.bfloat16, device=device)

    common = dict(
        scratch=scratch,
        a=a,
        topk_ids=ids,
        topk_weights=weights,
        w13=rep.w13_rp,
        w13_scales=w13_scales,
        w2=rep.w2_rp,
        w2_scales=w2_scales,
        alpha1=impl.w1_alphas,
        alpha2=impl.w2_alphas,
        input_scale=impl.a1_gscale,
        down_scale=impl.a2_gscale,
        max_tokens=num_tokens,
        num_topk=top_k,
        swiglu_limit=10.0,
        fast_math=True,
        csf_inline_scales=csf_inline,
        prequant_a_values=pre_v,
        prequant_a_scales=pre_s,
    )
    launch_w4a8_compact_micro(**common, output=out_ref, coalesced=False)
    launch_w4a8_compact_micro(**common, output=out_got, coalesced=True)
    torch.cuda.synchronize()

    assert torch.equal(out_got.view(torch.uint16), out_ref.view(torch.uint16))
    if num_tokens >= 4:
        assert torch.all(out_got[-1] == 0)


@pytest.mark.parametrize(
    "shape",
    [
        (128, 5120, 576, 3, 128),  # DSpark draft shape (E=128, k=3)
        (32, 512, 576, 6, 64),  # Coalesced FC1 N64 tiles
        (32, 512, 1024, 6, 128),  # N % 128 == 0 (non-N64-tail)
    ],
)
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_w4a8_compact_coalesced_shapes_heavy_split_and_fp32_ref(shape, ids_dtype):
    """Verify DSpark, N64-tile, N=1024, heavy 16-hit split, all-inactive, and FP32 ref."""
    device = require_b12x()
    e, k, n, top_k, fc1_tile_n = shape
    t = 16
    experts, w13_u8, s13_u8, w2_u8, s2_u8 = _make_experts(
        device, num_experts=e, hidden_size=k, intermediate_size=n, csf=False, seed=77
    )
    impl = experts._impl
    rep = impl.representation.value
    gen = torch.Generator(device="cpu").manual_seed(321)
    a = (torch.randn((t, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )

    # All 16 tokens hit the exact same top_k experts -> every expert splits into 2x 8-col groups!
    ids_shared = (
        torch.arange(top_k, dtype=ids_dtype, device=device)
        .unsqueeze(0)
        .expand(t, top_k)
        .contiguous()
    )
    weights = torch.softmax(torch.randn((t, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    scratch = torch.empty(
        micro_scratch_nbytes(t, k, n, top_k), dtype=torch.uint8, device=device
    )
    out_ref = torch.empty((t, k), dtype=torch.bfloat16, device=device)
    out_got = torch.empty((t, k), dtype=torch.bfloat16, device=device)
    common = dict(
        scratch=scratch,
        a=a,
        topk_ids=ids_shared,
        topk_weights=weights,
        w13=rep.w13_rp,
        w13_scales=rep.w13_sfb,
        w2=rep.w2_rp,
        w2_scales=rep.w2_sfb,
        alpha1=impl.w1_alphas,
        alpha2=impl.w2_alphas,
        input_scale=impl.a1_gscale,
        down_scale=impl.a2_gscale,
        max_tokens=t,
        num_topk=top_k,
        swiglu_limit=10.0,
        fast_math=False,
        fc1_tile_n=fc1_tile_n,
    )
    launch_w4a8_compact_micro(**common, output=out_ref, coalesced=False)
    launch_w4a8_compact_micro(**common, output=out_got, coalesced=True)
    torch.cuda.synchronize()
    assert torch.equal(out_got.view(torch.uint16), out_ref.view(torch.uint16))

    # Compare against FP32 dequantized reference on the smaller hidden shape.
    if k <= 512:
        fp32_ref = _fp32_moe_reference(
            a, w13_u8, s13_u8, w2_u8, s2_u8, ids_shared, weights, swiglu_limit=10.0
        )
        cos = torch.nn.functional.cosine_similarity(
            out_got.float().reshape(-1), fp32_ref.reshape(-1), dim=0
        ).item()
        assert cos > 0.995, f"FP32 reference cosine too low: {cos}"

    # All-inactive (-1) routes (live_counts == 0): must produce exact zeros without OOB.
    ids_dead = torch.full_like(ids_shared, -1)
    out_dead_ref = torch.full((t, k), 1.0, dtype=torch.bfloat16, device=device)
    out_dead_got = torch.full((t, k), 1.0, dtype=torch.bfloat16, device=device)
    launch_w4a8_compact_micro(
        **{**common, "topk_ids": ids_dead}, output=out_dead_ref, coalesced=False
    )
    launch_w4a8_compact_micro(
        **{**common, "topk_ids": ids_dead}, output=out_dead_got, coalesced=True
    )
    torch.cuda.synchronize()
    assert torch.all(out_dead_got == 0)
    assert torch.equal(out_dead_got.view(torch.uint16), out_dead_ref.view(torch.uint16))


@pytest.mark.parametrize("fuse_input_quant", [False, True])
def test_w4a8_compact_coalesced_cuda_graph_live_route_mutation(fuse_input_quant: bool):
    """Zero-allocation CUDA-graph replay with live route/live_counts mutation."""
    device = require_b12x()
    e, k, n, t, top_k = 64, 512, 576, 14, 6
    experts, _, _, _, _ = _make_experts(
        device, num_experts=e, hidden_size=k, intermediate_size=n, csf=True, seed=55
    )
    impl = experts._impl
    rep = impl.representation.value
    w13_csf = impl.nvfp4_csf.first.scales.storage
    w2_csf = impl.nvfp4_csf.second.scales.storage

    a = torch.randn((t, k), dtype=torch.bfloat16, device=device) * 0.25
    ids = torch.randint(0, e, (t, top_k), dtype=torch.int32, device=device)
    weights = torch.softmax(torch.randn((t, top_k), device=device), dim=-1)
    scratch = torch.empty(
        micro_scratch_nbytes(t, k, n, top_k), dtype=torch.uint8, device=device
    )
    out_got = torch.empty((t, k), dtype=torch.bfloat16, device=device)
    out_ref = torch.empty((t, k), dtype=torch.bfloat16, device=device)

    common = dict(
        scratch=scratch,
        a=a,
        topk_ids=ids,
        topk_weights=weights,
        w13=rep.w13_rp,
        w13_scales=w13_csf,
        w2=rep.w2_rp,
        w2_scales=w2_csf,
        alpha1=impl.w1_alphas,
        alpha2=impl.w2_alphas,
        input_scale=impl.a1_gscale,
        down_scale=impl.a2_gscale,
        max_tokens=t,
        num_topk=top_k,
        swiglu_limit=10.0,
        fast_math=True,
        csf_inline_scales=True,
        fuse_input_quantization=fuse_input_quant,
    )
    # Warmup compile before capture.
    launch_w4a8_compact_micro(**common, output=out_got, coalesced=True)
    launch_w4a8_compact_micro(
        **{**common, "fuse_input_quantization": False}, output=out_ref, coalesced=False
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_w4a8_compact_micro(**common, output=out_got, coalesced=True)

    route_patterns = [
        # 1. All 14 tokens hit the exact same 6 experts (>8 split)
        torch.arange(top_k, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(t, top_k)
        .contiguous(),
        # 2. Half live tokens (live_counts = 7) with partial overlap
        torch.cat(
            [
                torch.randint(0, 12, (7, top_k), dtype=torch.int32, device=device),
                torch.full((t - 7, top_k), -1, dtype=torch.int32, device=device),
            ],
            dim=0,
        ),
        # 3. Distinct uniform experts across all 14 tokens
        torch.stack(
            [torch.randperm(e, device=device)[:top_k].to(torch.int32) for _ in range(t)]
        ),
        # 4. Zero live tokens (all -1)
        torch.full((t, top_k), -1, dtype=torch.int32, device=device),
    ]
    try:
        with kernel_resolution_guard("coalesced compact replay"):
            for idx, pattern in enumerate(route_patterns):
                a.mul_(-0.9 if idx % 2 else 1.1)
                ids.copy_(pattern)
                out_got.fill_(float("nan"))
                alloc_before = torch.cuda.memory_stats(device)[
                    "allocation.all.allocated"
                ]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == alloc_before
                )
                launch_w4a8_compact_micro(
                    **{**common, "fuse_input_quantization": False},
                    output=out_ref,
                    coalesced=False,
                )
                torch.cuda.synchronize()
                assert torch.equal(
                    out_got.view(torch.uint16), out_ref.view(torch.uint16)
                ), f"CUDA-graph replay mismatch on pattern {idx}"
    finally:
        graph.reset()


def test_coupled_prequant_m_le_128_tuning_policy():
    """Heuristic and validator select coalesced micro for M<=16 and Dynamic M16 for 17<=M<=128."""
    base = dict(
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
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        w13_layout="w13",
        weight_layouts=("w13",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=True,
        numerical_recipe=None,
        controls=FrozenMapping({}),
        prequantized_input=True,
    )
    for t in (1, 8, 12, 16):
        q = MoeDecodeQuery(**base, num_tokens=t, routed_rows=t * 6)
        cfg = _heuristic_moe_decode_config(q, None)
        assert cfg.backend == "micro"
        assert not cfg.compact_fuse_input_quantization
        validate_moe_decode_config(q, cfg, None)
        # Fused input quantization must be rejected when input is already prequantized.
        with pytest.raises(ValueError):
            validate_moe_decode_config(
                q,
                MoeDecodeConfig(
                    backend="micro",
                    route_planner="internal",
                    max_active_clusters=None,
                    compact_fuse_input_quantization=True,
                ),
                None,
            )

    for t in (17, 32, 64, 128):
        q = MoeDecodeQuery(**base, num_tokens=t, routed_rows=t * 6)
        cfg = _heuristic_moe_decode_config(q, None)
        assert cfg.backend == "dynamic"
        assert cfg.dynamic_tile_m == 16
        assert cfg.dynamic_route_mode == "grouped"
        validate_moe_decode_config(q, cfg, None)

    q_prefill = MoeDecodeQuery(**base, num_tokens=512, routed_rows=512 * 6)
    cfg_prefill = _heuristic_moe_decode_config(q_prefill, None)
    assert cfg_prefill.backend == "dynamic"
    assert cfg_prefill.dynamic_tile_m == 64


@pytest.mark.parametrize("num_tokens", [8, 16, 32, 64, 128])
def test_coupled_prequant_m_le_128_gpu_parity(num_tokens: int):
    """Verify supports_prequant_mxfp8 + prepared_moe_prequant_out / prepared_moe_mxfp8_out for M<=128."""
    device = require_b12x()
    e, k, n, top_k = 32, 512, 576, 6
    experts, _, _, _, _ = _make_experts(
        device, num_experts=e, hidden_size=k, intermediate_size=n, csf=False, seed=88
    )
    gen = torch.Generator(device="cpu").manual_seed(900 + num_tokens)
    x_bf16 = (torch.randn((num_tokens, k), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    live = max(1, num_tokens - 3)
    x_bf16[live:].zero_()
    ids = torch.stack(
        [torch.randperm(e, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(device=device, dtype=torch.int32)
    ids[live:] = -1
    weights = torch.softmax(
        torch.randn((num_tokens, top_k), generator=gen), dim=-1
    ).to(device=device, dtype=torch.float32)
    weights[live:] = 0.0

    shapes = mxfp8.dense_rows_storage_shapes(num_tokens, k)
    vals = torch.empty(shapes[0], dtype=torch.float8_e4m3fn, device=device)
    srows = torch.empty(shapes[1], dtype=torch.uint8, device=device)
    smma = torch.empty(shapes[2], dtype=torch.uint8, device=device)
    quantize_mxfp8_rows_cute(
        x_bf16, vals.view(torch.uint8), srows, smma, expected_m=num_tokens
    )

    plan_bf16 = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(
            max_tokens=num_tokens, top_k=top_k, input_format="bf16"
        ),
    )
    plan_mx = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(
            max_tokens=num_tokens, top_k=top_k, input_format="mxfp8"
        ),
    )
    out_probe = torch.zeros((num_tokens, k), dtype=torch.bfloat16, device=device)

    def prep_bf16(st):
        sc = [
            torch.zeros(sp.nbytes, dtype=sp.dtype, device=device)
            for sp in st.scratch.scratch_specs()
        ]
        b = st.bind(
            a=x_bf16,
            topk_ids=ids,
            topk_weights=weights,
            output=out_probe,
            scratch=sc,
            input_scales_static=True,
        )
        return PreparedCall(run=lambda: st.run(b), owners=(sc, b))

    def prep_mx(st):
        sc = [
            torch.zeros(sp.nbytes, dtype=sp.dtype, device=device)
            for sp in st.scratch.scratch_specs()
        ]
        b = st.bind(
            a=vals,
            a_scales=srows,
            topk_ids=ids,
            topk_weights=weights,
            output=out_probe,
            scratch=sc,
            input_scales_static=True,
        )
        return PreparedCall(run=lambda: st.run(b), owners=(sc, b))

    session = PreparationSession(compile_workers=0, autotune=False)
    session.prepare(
        (
            plan_bf16.request(name=f"bf16_{num_tokens}", prepare_call=prep_bf16),
            plan_mx.request(name=f"mx_{num_tokens}", prepare_call=prep_mx),
        )
    )

    assert supports_prequant_mxfp8(plan_mx, num_tokens)
    if num_tokens <= 16:
        assert supports_prequant_mxfp8(plan_bf16, num_tokens)
        assert plan_mx._prepared.state.config.backend == "micro"
    else:
        assert plan_mx._prepared.state.config.backend == "dynamic"
        assert plan_mx._prepared.state.config.dynamic_tile_m == 16

    sc_bf16 = torch.zeros(
        plan_bf16.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device
    )
    sc_mx = torch.zeros(
        plan_mx.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device
    )

    def run_bf16():
        out = torch.zeros((num_tokens, k), dtype=torch.bfloat16, device=device)
        prepared_moe_out(
            x_bf16,
            weights,
            ids,
            out,
            sc_bf16,
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

    def run_mx_op():
        out = torch.zeros((num_tokens, k), dtype=torch.bfloat16, device=device)
        prepared_moe_mxfp8_out(
            vals,
            srows,
            weights,
            ids,
            out,
            sc_mx,
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

    def run_prequant_op(handle, scratch):
        out = torch.zeros((num_tokens, k), dtype=torch.bfloat16, device=device)
        prepared_moe_prequant_out(
            x_bf16,
            vals.view(torch.uint8),
            srows,
            weights,
            ids,
            out,
            scratch,
            [],
            [],
            handle,
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
    out_mx = run_mx_op()
    out_pq = run_prequant_op(plan_mx.handle, sc_mx)
    torch.cuda.synchronize()

    assert torch.all(out_mx[live:] == 0)
    assert torch.all(out_pq[live:] == 0)
    if num_tokens <= 16:
        # Compact micro is deterministic -> strict 0-ULP bit-exact!
        assert torch.equal(out_mx.view(torch.uint16), bf16_runs[0].view(torch.uint16))
        assert torch.equal(out_pq.view(torch.uint16), bf16_runs[0].view(torch.uint16))
        out_bf16_pq = run_prequant_op(plan_bf16.handle, sc_bf16)
        torch.cuda.synchronize()
        assert torch.equal(
            out_bf16_pq.view(torch.uint16), bf16_runs[0].view(torch.uint16)
        )
    else:
        # Grouped dynamic M16 uses BF16 scatter-add atomics -> match within BF16 repeatability.
        rep_max = max(
            (a.float() - b.float()).abs().max().item()
            for i, a in enumerate(bf16_runs)
            for b in bf16_runs[i + 1 :]
        )
        tol = max(2.0 * rep_max, 1e-2)
        assert (bf16_runs[0].float() - out_mx.float()).abs().max().item() <= tol
        assert (bf16_runs[0].float() - out_pq.float()).abs().max().item() <= tol
        assert (
            torch.nn.functional.cosine_similarity(
                bf16_runs[0].float().reshape(-1), out_mx.float().reshape(-1), dim=0
            ).item()
            > 0.9999
        )
        assert (
            torch.nn.functional.cosine_similarity(
                bf16_runs[0].float().reshape(-1), out_pq.float().reshape(-1), dim=0
            ).item()
            > 0.9999
        )
