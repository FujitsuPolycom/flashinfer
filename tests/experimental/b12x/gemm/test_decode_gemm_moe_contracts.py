"""CPU contract tests for b12x decode GEMM, WO projection, KV RoPE, and compact MoE optimizations."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import DeviceIdentity, FrozenMapping

RTX = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000 Blackwell")
GB10 = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")


# ---------------------------------------------------------------------------
# 1. wo_projection persistent split-K flags & zero-memset execution
# ---------------------------------------------------------------------------


def test_wo_a16_splitk_flags_live_in_persistent_memory_and_skip_memset():
    from b12x.gemm._shared.wo_mxfp8 import WOProjectionScratchCaps
    from b12x.gemm.wo_projection._preparation import _PreparedWO, plan as wo_plan
    from b12x.gemm.wo_projection._tc_gemv import (
        wo_a16_tc_gemv_flags_bytes,
        wo_a16_tc_gemv_scratch_bytes,
        wo_a16_tc_gemv_splitk_bytes,
        wo_a16_tc_gemv_tmp_bytes,
    )
    from b12x.gemm.wo_projection._tuning import WoProjectionConfig

    caps = WOProjectionScratchCaps(
        device="cpu",
        max_tokens=8,
        groups=2,
        group_width=4096,
        rank=1024,
        hidden=5120,
    )
    # Scratch holds tmp + split-K partials only; completion flags are persistent.
    assert wo_a16_tc_gemv_scratch_bytes(8, 2, 1024, split_k=2) == (
        wo_a16_tc_gemv_tmp_bytes(8, 2, 1024)
        + wo_a16_tc_gemv_splitk_bytes(8, 2, 1024, 2)
    )
    assert wo_a16_tc_gemv_flags_bytes(2, 1024, split_k=2) == 128 * 4
    assert wo_a16_tc_gemv_flags_bytes(2, 1024, split_k=1) == 0

    declaration = wo_plan(
        caps,
        invocation=dict(
            operation="inv_rope",
            heads_per_group=8,
            nope_dim=448,
            rope_dim=64,
            positions_dtype="int64",
            cos_sin_dtype="float32",
        ),
    )
    dev = SimpleNamespace(ordinal=0, identity=RTX)
    cfg_sk2 = WoProjectionConfig(
        backend="a16_tc_gemv",
        decode_tile_n=16,
        wo_a_split_k=2,
        wo_a_warps=4,
        wo_b_warps=4,
    )
    mem_sk2 = declaration._memory_requirements(cfg_sk2, dev)
    assert len(mem_sk2.persistent) == 1
    assert mem_sk2.persistent[0].key == (
        "wo_projection.a16_woa_flags",
        0,
        2,
        1024,
        4096,
        2,
    )
    assert mem_sk2.persistent[0].required_nbytes == 512
    assert mem_sk2.persistent[0].resident_nbytes == 0

    cfg_sk1 = replace(cfg_sk2, wo_a_split_k=1)
    mem_sk1 = declaration._memory_requirements(cfg_sk1, dev)
    assert mem_sk1.persistent == ()

    # Verify _PreparedWO.run and run_inv_rope pass persistent flags without zero_().
    class _GuardFlags:
        def zero_(self):
            raise AssertionError(
                "persistent split-K flags must not be memset per launch"
            )

    guard_flags = _GuardFlags()
    captured = {}

    def fake_wo_a(*args):
        captured["wo_a"] = args

    def fake_wo_b(*args):
        captured["wo_b"] = args

    prepared = _PreparedWO(
        _scratch_state=None,
        query=declaration.query,
        ordinary_a=None,
        ordinary_b=None,
        fused_b=None,
        quantizers=None,
        a16_wo_a=fake_wo_a,
        a16_wo_b=fake_wo_b,
        a16_woa_flags=guard_flags,
        a16_split_k=2,
    )
    dummy_out = torch.empty((8, 5120, 1), dtype=torch.bfloat16)
    dummy_weights = SimpleNamespace(
        wo_a=SimpleNamespace(values="wa", scale_mma="sa"),
        wo_b=SimpleNamespace(values="wb", scale_mma="sb"),
    )
    binding = SimpleNamespace(
        o="o",
        source_tgd="src",
        positions="pos",
        cos_sin_cache="cs",
        weights=dummy_weights,
        tmp="tmp",
        x_q="splitk_scratch",
        tmp_q=None,
        output=dummy_out,
        return_3d=False,
    )
    prepared.run_inv_rope(binding)
    assert captured["wo_a"][7] is guard_flags
    prepared.run(binding)
    assert captured["wo_a"][7] is guard_flags


# ---------------------------------------------------------------------------
# 2. K % 64 == 0 (K=576 shared.down & K=160) active-K warp math & dequant
# ---------------------------------------------------------------------------


def test_mxfp8_tc_gemv_k576_uses_9_active_chunks_and_3_balanced_warps():
    from b12x.gemm.blockscaled._mxfp8_tc_gemv import (
        MXFP8MultiRowTCGemvKernel,
        effective_mxfp8_tc_gemv_warps,
        mxfp8_tc_gemv_active_k,
    )
    from b12x.gemm.blockscaled._preparation import _workspace_bytes
    from b12x.gemm.blockscaled._tuning import (
        BlockscaledConfig,
        BlockscaledQuery,
        TUNING as BLOCKSCALED_TUNING,
        mxfp8_a16_dequant,
        mxfp8_a16_tc_gemv_splits,
    )

    assert mxfp8_tc_gemv_active_k(576, 640) == 576
    assert mxfp8_tc_gemv_active_k(160, 256) == 192
    assert mxfp8_tc_gemv_active_k(5120, 5120) == 5120

    # 9 chunks with mma_warps=4 uses 3 balanced warps instead of dropping to 2 warps on 10 padded chunks.
    assert effective_mxfp8_tc_gemv_warps(576, 1, 4) == 3
    assert effective_mxfp8_tc_gemv_warps(576, 3, 4) == 3
    assert effective_mxfp8_tc_gemv_warps(192, 1, 4) == 3

    kern = MXFP8MultiRowTCGemvKernel(
        n=5120,
        in_features=576,
        padded_k=640,
        max_m_tiles=1,
        split_k=1,
        warps=4,
    )
    assert kern.k == 640  # physical row stride preserved for K128 TMA weights
    assert kern.active_k == 576
    assert kern.chunks_per_split == 9
    assert kern.warps == 3
    assert kern.chunks_per_warp == 3

    # Default config for shared.down [5120, 576] (padded to 640) at M=8 uses tc_gemv cleanly.
    q_m8 = BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=8,
        in_features=576,
        padded_in_features=640,
        out_features=5120,
        output_mode="provided",
        workspace_form="provided",
    )
    default_m8 = BLOCKSCALED_TUNING.default_config(q_m8, RTX)
    assert default_m8.mode == "a16"
    assert default_m8.a16_backend == "tc_gemv"
    assert default_m8.split_k == 1
    BLOCKSCALED_TUNING.validate_config(q_m8, default_m8, RTX)

    # Small-N K=576 exposes split_k=3 cleanly (9 chunks divisible by 3).
    q_small_n = replace(q_m8, out_features=512)
    assert 3 in mxfp8_a16_tc_gemv_splits(q_small_n)
    cfg_sk3 = replace(default_m8, split_k=3, mma_warps=4)
    BLOCKSCALED_TUNING.validate_config(q_small_n, cfg_sk3, RTX)

    # At M=128 (C=16 verify), shared.down [5120, 576] unlocks mxfp8_a16_dequant without padding waste.
    q_m128 = replace(q_m8, num_tokens=128)
    cfg_a16 = BlockscaledConfig(
        mode="a16", tile_m=16, tile_n=64, tile_k=64, split_k=1, mma_warps=4
    )
    assert mxfp8_a16_dequant(q_m128, cfg_a16) is True
    assert _workspace_bytes(q_m128, cfg_a16, RTX) == 5120 * 576 * 2


# ---------------------------------------------------------------------------
# 3. bf16_gemv router split_k selection & dense GEMM evict_first=(False,)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_features,expected_min_ctas",
    [(384, 96), (128, 96), (32, 32)],
)
def test_bf16_gemv_router_defaults_use_split_k_on_sm120(
    out_features, expected_min_ctas
):
    from b12x.gemm.bf16_gemv._tuning import GemvQuery, TUNING as BF16_TUNING

    q = GemvQuery(
        source_dtype="bfloat16",
        weight_dtype="bfloat16",
        max_rows=8,
        in_features=5120,
        out_features=out_features,
        source_contiguous=True,
        source_aligned=True,
        weight_contiguous=True,
        weight_aligned=True,
        output_dtype="float32",
    )
    cfg = BF16_TUNING.default_config(q, RTX)
    assert cfg.backend == "tc"
    assert cfg.split_k > 1
    n_blocks = (out_features + 15) // 16
    assert n_blocks * cfg.split_k >= expected_min_ctas
    BF16_TUNING.validate_config(q, cfg, RTX)
    choices = {c for _, c in BF16_TUNING.choices(q, RTX)}
    assert cfg in choices


def test_dense_gemm_ready_small_mxfp8_disables_evict_first():
    from b12x.gemm._tuning import DenseGemmQuery, TUNING as DENSE_TUNING, knob_values

    for effort in ("default", "audit"):
        q = DenseGemmQuery(
            recipe="mxfp8",
            entry_point="gemm.mm",
            weight_storage="native",
            output_dtype="bfloat16",
            batch=1,
            max_rows=8,
            in_features=1280,
            out_features=8192,
            output_mode="provided",
            alpha_mode="unit",
            rhs_prefetch_ready=True,
            expected_m=8,
            autotune_effort=effort,
        )
        assert knob_values(q)["evict_first"] == (False,)
        assert all(not cfg.evict_first for _, cfg in DENSE_TUNING.choices(q, RTX))


# ---------------------------------------------------------------------------
# 4. MoE compact W4A8 inline CSF default, token cutover, & expert_id < 0 gating
# ---------------------------------------------------------------------------


def test_compact_w4a8_defaults_enable_inline_csf_and_bound_micro_tokens():
    from b12x.moe.fused_moe._impl import (
        _COMPACT_W4A8_MAX_TOKENS,
        _COMPACT_W4A8_MAX_TOKENS_SM121,
    )
    from b12x.moe.fused_moe._tuning import TUNING as MOE_TUNING
    from tests.experimental.b12x.moe.test_compact_w4a8_backend_race import _query

    assert 8 <= _COMPACT_W4A8_MAX_TOKENS <= 16
    assert _COMPACT_W4A8_MAX_TOKENS_SM121 == 4

    # SM120: micro through _COMPACT_W4A8_MAX_TOKENS, dynamic M16 above the
    # threshold; inline CSF enabled by default.
    sm120_tokens = (
        1,
        4,
        7,
        8,
        14,
        16,
        _COMPACT_W4A8_MAX_TOKENS,
        _COMPACT_W4A8_MAX_TOKENS + 1,
        64,
        128,
    )
    for tokens in sm120_tokens:
        cfg = MOE_TUNING.default_config(_query(tokens), RTX)
        if tokens <= _COMPACT_W4A8_MAX_TOKENS:
            assert cfg.backend == "micro"
        else:
            assert cfg.backend == "dynamic"
            assert cfg.dynamic_tile_m == 16
            assert cfg.dynamic_route_mode == "grouped"
        assert cfg.csf_inline_scales is True

    # High-density prefill amortizes expansion (routed_rows > 16 * 384).
    cfg_prefill = MOE_TUNING.default_config(_query(2048), RTX)
    assert cfg_prefill.backend == "dynamic"
    assert cfg_prefill.csf_inline_scales is False

    # SM121 (GB10 Spark): micro through _COMPACT_W4A8_MAX_TOKENS_SM121,
    # dynamic M16 above the threshold.
    for tokens in (4, 7, 8, 16):
        cfg = MOE_TUNING.default_config(_query(tokens), GB10)
        expected = "micro" if tokens <= _COMPACT_W4A8_MAX_TOKENS_SM121 else "dynamic"
        assert cfg.backend == expected


def test_w4a8_compact_kernels_gate_negative_expert_ids_cleanly():
    from b12x.moe._shared.kernels.w4a8_compact_activation import (
        W4A8CompactMicroActivationKernel,
    )
    from b12x.moe._shared.kernels.w4a8_compact_projection import (
        W4A8CompactMicroProjectionActivationKernel,
        W4A8CompactMicroProjectionKernel,
    )
    from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel

    # Projection and activation kernels skip _run_task / activate_tile when expert < 0.
    proj_src = inspect.getsource(W4A8CompactMicroProjectionKernel.kernel)
    fused_src = inspect.getsource(
        W4A8CompactMicroProjectionActivationKernel.fused_kernel
    )
    act_src = inspect.getsource(W4A8CompactMicroActivationKernel.kernel)
    assert "expert >= Int32(0) and expert < Int32(alpha.shape[0])" in proj_src
    assert "expert >= Int32(0) and expert < Int32(alpha.shape[0])" in fused_src
    assert "expert >= Int64(0) and expert < Int64(self.num_experts)" in act_src

    # Phase 2 direct-routes kernel instantiates cleanly and has CTA-uniform skip for invalid pairs.
    phase2 = W4A8MaterializedPhase2Kernel(
        source_tile_m=1,
        deterministic_output=True,
        direct_routes=True,
        fused_topk=6,
        n64_repacked=True,
        n64_tail=True,
    )
    assert phase2.direct_routes and phase2.fused_topk == 6 and phase2.fused_groups == 2
    phase2_src = inspect.getsource(W4A8MaterializedPhase2Kernel.kernel)
    assert "if valid0 or valid1:" in phase2_src
