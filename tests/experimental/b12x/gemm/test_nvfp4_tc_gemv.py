"""SM120/SM121 NVFP4MultiRowTCGemvKernel parity, contract, and CUDA-graph tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from b12x._lib.intrinsics.reference import swizzle_block_scale
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _preparation
from b12x.gemm.blockscaled._nvfp4_tc_gemv import (
    effective_nvfp4_tc_gemv_chunk_k,
    effective_nvfp4_tc_gemv_warps,
    nvfp4_tc_gemv_workspace_bytes,
)
from b12x.gemm.blockscaled._tuning import (
    TUNING,
    BlockscaledConfig,
    BlockscaledQuery,
    nvfp4_a16_tc_gemv_eligible,
)
from b12x.preparation import DeviceIdentity
from b12x.testing.reference.helpers import require_b12x

from .test_blockscaled_a16 import (
    assert_close,
    make_weight,
    make_workspace,
    prepared_execution,
)

IDENTITY_SM120 = DeviceIdentity("nvidia", (12, 0), 170, "NVIDIA GeForce RTX 5090")
IDENTITY_SM121 = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")


def _assert_bf16_parity(
    actual: torch.Tensor, tma_ref: torch.Tensor, fp32_ref: torch.Tensor
) -> None:
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert_close(actual, fp32_ref)
    assert_close(actual, tma_ref.float())
    # Exact W4A16 BF16 arithmetic: compare per-element ULP distance against TMA A16.
    diff_ulp = (
        actual.view(torch.int16).to(torch.int32)
        - tma_ref.view(torch.int16).to(torch.int32)
    ).abs()
    # Allow at most 1-2 BF16 ULPs from FP32 K-reduction summation order differences
    # (and check relative error vs FP32 oracle is strictly within BF16 precision).
    rel_err = torch.linalg.vector_norm(
        actual.float() - fp32_ref
    ) / torch.linalg.vector_norm(fp32_ref).clamp_min(1e-12)
    tma_rel_err = torch.linalg.vector_norm(
        tma_ref.float() - fp32_ref
    ) / torch.linalg.vector_norm(fp32_ref).clamp_min(1e-12)
    assert float(rel_err) <= max(float(tma_rel_err) * 1.5 + 1e-4, 0.004)
    assert int(diff_ulp.max()) <= 2 or float(rel_err) < 0.002


@pytest.mark.parametrize("identity", [IDENTITY_SM120, IDENTITY_SM121])
def test_nvfp4_tc_gemv_tuning_contracts_and_defaults(identity):
    # M <= 32 without activation scale -> defaults to tc_gemv.
    for m in (1, 7, 8, 14, 16, 28, 32):
        q = BlockscaledQuery(
            recipe="nvfp4",
            num_tokens=m,
            in_features=5120,
            padded_in_features=5120,
            out_features=32320,
            activation_mode="auto",
            activation_scale_available=False,
        )
        assert nvfp4_a16_tc_gemv_eligible(q, identity)
        cfg = TUNING.default_config(q, identity)
        assert cfg.mode == "a16"
        assert cfg.a16_backend == "tc_gemv"
        TUNING.validate_config(q, cfg, identity)

    # M <= 8 with activation scale and mode="auto" -> automatic A16 -> tc_gemv.
    q_m8 = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=8,
        in_features=5120,
        padded_in_features=5120,
        out_features=5120,
        activation_mode="auto",
        activation_scale_available=True,
    )
    assert TUNING.default_config(q_m8, identity).a16_backend == "tc_gemv"

    # M = 9 with activation scale and mode="auto" -> stays quantized.
    q_m9 = replace(q_m8, num_tokens=9)
    assert TUNING.default_config(q_m9, identity).mode == "quantized"

    # M = 33 -> falls back to TMA DenseGemmKernel.
    q_m33 = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=33,
        in_features=5120,
        padded_in_features=5120,
        out_features=32320,
        activation_mode="a16",
    )
    assert not nvfp4_a16_tc_gemv_eligible(q_m33, identity)
    assert TUNING.default_config(q_m33, identity).a16_backend == "auto"

    # Non-K64-aligned K=96 -> ineligible for tc_gemv, falls back to TMA.
    q_k96 = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=8,
        in_features=96,
        padded_in_features=96,
        out_features=136,
        activation_mode="a16",
    )
    assert not nvfp4_a16_tc_gemv_eligible(q_k96, identity)
    assert TUNING.default_config(q_k96, identity).a16_backend == "auto"

    # Default effort keeps quantized + tc_gemv; audit also includes TMA A16.
    default_choices = [c for _, c in TUNING.choices(q_m8, identity)]
    assert any(c.mode == "quantized" for c in default_choices)
    assert any(c.mode == "a16" and c.a16_backend == "tc_gemv" for c in default_choices)
    assert not any(c.mode == "a16" and c.a16_backend == "auto" for c in default_choices)

    audit_choices = [
        c for _, c in TUNING.choices(replace(q_m8, autotune_effort="audit"), identity)
    ]
    assert any(c.mode == "a16" and c.a16_backend == "tc_gemv" for c in audit_choices)
    assert any(c.mode == "a16" and c.a16_backend == "auto" for c in audit_choices)
    for c in audit_choices:
        TUNING.validate_config(q_m8, c, identity)

    # Chunk-K and warp selection checks.
    assert effective_nvfp4_tc_gemv_chunk_k(5120, 1, 4, 16) == 256
    assert effective_nvfp4_tc_gemv_chunk_k(5120, 1, 8, 16) == 128
    assert effective_nvfp4_tc_gemv_chunk_k(5120, 4, 4, 16) == 64
    assert effective_nvfp4_tc_gemv_chunk_k(192, 1, 4, 16) == 64
    assert effective_nvfp4_tc_gemv_chunk_k(5120, 1, 4, 64) == 256
    assert effective_nvfp4_tc_gemv_chunk_k(640, 1, 4, 64) == 128
    assert effective_nvfp4_tc_gemv_warps(5120, 4, 4, 16) == 4
    assert effective_nvfp4_tc_gemv_warps(512, 4, 2, 16) == 2
    assert effective_nvfp4_tc_gemv_warps(5120, 1, 4, 64) == 4

    # Workspace validation rejects undersized split-K buffer.
    split_cfg = BlockscaledConfig(
        mode="a16",
        tile_m=16,
        tile_n=16,
        tile_k=64,
        split_k=4,
        mma_warps=4,
        a16_backend="tc_gemv",
    )
    needed = nvfp4_tc_gemv_workspace_bytes(8, 5120, 4, 16)
    TUNING.validate_config(replace(q_m8, workspace_nbytes=needed), split_cfg, identity)
    with pytest.raises(ValueError, match="workspace"):
        TUNING.validate_config(
            replace(q_m8, workspace_nbytes=needed - 1), split_cfg, identity
        )


@pytest.mark.parametrize("m", [1, 2, 4, 7, 8, 13, 16, 24, 32])
@pytest.mark.parametrize("tile_n", [16, 32, 64])
@pytest.mark.parametrize("split_k", [1, 2, 4])
def test_nvfp4_tc_gemv_parity_rows_and_splits(m, tile_n, split_k):
    require_b12x()
    torch.manual_seed(1200 + m * 31 + tile_n * 7 + split_k)
    n, k = 384, 512
    weight, decoded, _ = make_weight("nvfp4", n, k)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    fp32_ref = source.float() @ decoded.T

    tma_cfg = BlockscaledConfig(mode="a16", tile_m=16, tile_n=64, tile_k=128, split_k=1)
    with prepared_execution(
        source, weight, activation_mode="a16", expected_m=m, config=tma_cfg
    ) as (_, tma_plan):
        tma_out = blockscaled.mm(source, weight, plan=tma_plan)

    warps = 4 if tile_n == 64 else (2 if (k // 64) // split_k <= 4 else 4)
    tc_cfg = BlockscaledConfig(
        mode="a16",
        tile_m=16,
        tile_n=tile_n,
        tile_k=64,
        split_k=split_k,
        mma_warps=warps,
        a16_backend="tc_gemv",
    )
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    ws = make_workspace(
        source, weight, activation_mode="a16", out=out, expected_m=m, config=tc_cfg
    )
    ws.fill_(255)
    with prepared_execution(
        source,
        weight,
        activation_mode="a16",
        out=out,
        workspace=ws,
        expected_m=m,
        config=tc_cfg,
    ) as (_, tc_plan):
        actual = blockscaled.mm(source, weight, out=out, workspace=ws, plan=tc_plan)
        assert actual.data_ptr() == out.data_ptr()
        _assert_bf16_parity(actual, tma_out, fp32_ref)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (7, 32320, 5120),
        (14, 32320, 5120),
        (28, 32320, 5120),
        (1, 38720, 4096),
        (8, 38720, 4096),
        (16, 2048, 2560),
        (4, 5120, 5120),
        (13, 136, 192),
        (3, 40, 64),
    ],
)
def test_nvfp4_tc_gemv_production_and_edge_shapes(m, n, k):
    require_b12x()
    torch.manual_seed(4200 + m + n + k)
    weight, decoded, _ = make_weight("nvfp4", n, k)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    fp32_ref = source.float() @ decoded.T

    tma_cfg = BlockscaledConfig(mode="a16", tile_m=16, tile_n=128, tile_k=64, split_k=1)
    with prepared_execution(
        source, weight, activation_mode="a16", expected_m=m, config=tma_cfg
    ) as (_, tma_plan):
        tma_out = blockscaled.mm(source, weight, plan=tma_plan)

    with prepared_execution(source, weight, activation_mode="a16", expected_m=m) as (
        _,
        default_plan,
    ):
        assert default_plan.selection.config.a16_backend == "tc_gemv"
        actual = blockscaled.mm(source, weight, plan=default_plan)
        _assert_bf16_parity(actual, tma_out, fp32_ref)


@pytest.mark.parametrize("kind", ["multiplier", "reciprocal"])
@pytest.mark.parametrize("tile_n", [16, 64])
@pytest.mark.parametrize("split_k", [1, 4])
def test_nvfp4_tc_gemv_global_scale_kind_and_live_mutation(kind, tile_n, split_k):
    require_b12x()
    torch.manual_seed(7711)
    m, n, k = 7, 512, 1024
    weight, decoded, storage = make_weight("nvfp4", n, k)
    saved_scales = storage.view(torch.uint8).clone()
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    global_scale = (
        weight.global_scale.clone()
        if kind == "multiplier"
        else weight.global_scale.reciprocal().clone()
    )
    w_kind = blockscaled.pack_weight(
        weight.values,
        storage,
        recipe="nvfp4",
        global_scale=global_scale,
        global_scale_kind=kind,
    )
    cfg = BlockscaledConfig(
        mode="a16",
        tile_m=16,
        tile_n=tile_n,
        tile_k=64,
        split_k=split_k,
        mma_warps=4,
        a16_backend="tc_gemv",
    )
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ws = make_workspace(
        source, w_kind, activation_mode="a16", out=out, expected_m=m, config=cfg
    )
    with prepared_execution(
        source,
        w_kind,
        activation_mode="a16",
        out=out,
        workspace=ws,
        expected_m=m,
        config=cfg,
    ) as (_, plan):
        blockscaled.mm(source, w_kind, out=out, workspace=ws, plan=plan)
        assert_close(out, source.float() @ decoded.T)

        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard("nvfp4 tc_gemv graph capture"):
            with torch.cuda.graph(graph):
                blockscaled.mm(source, w_kind, out=out, workspace=ws, plan=plan)
            pointers = (source.data_ptr(), out.data_ptr(), ws.data_ptr())
            scale_factor = 1.0
            for step in range(4):
                source.normal_()
                if step > 0:
                    global_scale.mul_(0.5)
                    scale_factor = scale_factor * (0.5 if kind == "multiplier" else 2.0)
                ws.fill_(255)
                out.fill_(float("nan"))
                alloc_before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"]
                    == alloc_before
                )
                assert pointers == (source.data_ptr(), out.data_ptr(), ws.data_ptr())
                assert_close(out, (source.float() @ decoded.T) * scale_factor)
            graph.reset()
    torch.testing.assert_close(storage.view(torch.uint8), saved_scales, atol=0, rtol=0)


@pytest.mark.parametrize("logical_k,stored_k", [(136, 256), (240, 256), (576, 640)])
@pytest.mark.parametrize("tile_n", [16, 32, 64])
def test_nvfp4_tc_gemv_padded_k_ignores_poisoned_tail(logical_k, stored_k, tile_n):
    require_b12x()
    torch.manual_seed(9901 + logical_k + tile_n)
    m, n = 13, 256
    source = torch.randn(m, logical_k, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(0, 256, (n, stored_k // 2), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(n, stored_k // 16, device="cuda") * 2 + 0.125).to(
        torch.float8_e4m3fn
    )
    # Poison scale bytes strictly past logical_k with E4M3 NaN (0x7F) to prove
    # padded weight columns never enter the MMA accumulator.
    poison_start = (logical_k + 15) // 16
    scales_u8 = scales.view(torch.uint8).clone()
    scales_u8[:, poison_start:] = 0x7F
    scales_poisoned = scales_u8.view(torch.float8_e4m3fn)
    gain = torch.tensor([0.0625], dtype=torch.float32, device="cuda")
    weight = replace(
        blockscaled.pack_weight(
            codes,
            swizzle_block_scale(scales_poisoned),
            recipe="nvfp4",
            global_scale=gain,
        ),
        in_features=logical_k,
    )
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device="cuda",
    )
    decoded = torch.stack(
        (lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1
    ).flatten(-2)
    scale_expanded = scales.float().repeat_interleave(16, dim=1)
    expected = (
        source.float()
        @ (
            (decoded[:, :logical_k] * scale_expanded[:, :logical_k])
            .to(torch.bfloat16)
            .float()
            * gain
        ).T
    )

    cfg = BlockscaledConfig(
        mode="a16",
        tile_m=16,
        tile_n=tile_n,
        tile_k=64,
        split_k=1,
        mma_warps=4 if tile_n == 64 else 2,
        a16_backend="tc_gemv",
    )
    with prepared_execution(
        source, weight, activation_mode="a16", expected_m=m, config=cfg
    ) as (_, plan):
        actual = blockscaled.mm(source, weight, plan=plan)
        assert torch.isfinite(actual).all()
        assert_close(actual, expected)


@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("tile_n", [16, 32, 64])
def test_nvfp4_tc_gemv_dynamic_rows_and_strided_source(split_k, tile_n):
    require_b12x()
    torch.manual_seed(5512)
    capacity, n, k = 32, 512, 512
    weight, decoded, _ = make_weight("nvfp4", n, k)
    # Create a non-contiguous 2D source with 8-element aligned row stride.
    backing = torch.randn(capacity, k + 16, device="cuda", dtype=torch.bfloat16)
    source = backing[:, :k]
    assert not source.is_contiguous() and source.stride(0) % 8 == 0
    out = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)
    cfg = BlockscaledConfig(
        mode="a16",
        tile_m=16,
        tile_n=tile_n,
        tile_k=64,
        split_k=split_k,
        mma_warps=4,
        a16_backend="tc_gemv",
    )
    ws = make_workspace(source, weight, activation_mode="a16", out=out, config=cfg)
    with prepared_execution(
        source, weight, activation_mode="a16", out=out, workspace=ws, config=cfg
    ) as (_, plan):
        with kernel_resolution_guard("nvfp4 tc_gemv dynamic rows"):
            for rows in (1, 7, 8, 9, 16, 17, 28, 32):
                out.fill_(float("nan"))
                ws.fill_(255)
                blockscaled.mm(
                    source[:rows], weight, out=out[:rows], workspace=ws, plan=plan
                )
                assert_close(out[:rows], source[:rows].float() @ decoded.T)
                assert torch.isnan(out[rows:]).all()


def test_nvfp4_tc_gemv_swiglu_bf16():
    require_b12x()
    torch.manual_seed(6603)
    m, n, k = 7, 256, 512
    weight, decoded, _ = make_weight("nvfp4", n, k)
    source = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    gate, up = source.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up).float() @ decoded.T
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    cfg = BlockscaledConfig(
        mode="a16",
        tile_m=16,
        tile_n=16,
        tile_k=64,
        split_k=2,
        mma_warps=4,
        a16_backend="tc_gemv",
    )
    ws = make_workspace(
        source,
        weight,
        activation_mode="a16",
        source_activation="swiglu_bf16",
        out=out,
        config=cfg,
    )
    with prepared_execution(
        source,
        weight,
        activation_mode="a16",
        source_activation="swiglu_bf16",
        out=out,
        workspace=ws,
        config=cfg,
    ) as (_, plan):
        blockscaled.mm(source, weight, out=out, workspace=ws, plan=plan)
        assert_close(out, expected)
