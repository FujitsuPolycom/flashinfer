"""CPU contract and live SM120 0-ULP parity tests for decode PDL and TC-GEMV DSMEM split-K.

Covers:
  1. Consumer-side SM120 PDL (``use_pdl=True`` + ``griddepcontrol_wait``) across
     decode GEMM, MLA, and coalesced W4A8 MoE kernels.
  2. Opt-in SM120 DSMEM cluster split-K reduction and occupancy gating for
     ``TensorCoreGemvKernel``, ``DualBf16TCGemvKernel``,
     ``GroupedWoaTCGemvKernel``, and ``MXFP8MultiRowTCGemvKernel``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.utils.dsmem import (
    SM120_DSMEM_CLUSTER_SIZES,
    sm120_dsmem_split_eligible,
    sm120_dsmem_split_enabled,
)
from b12x._lib.utils.pdl import parse_pdl_env, sm120_consumer_pdl_enabled
from b12x.attention._shared.mla import kernel as mla_kernel
from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_HEAD_DIM,
    COMPRESSED_SPARSE_MLA_NOPE_DIM,
    COMPRESSED_SPARSE_MLA_ROPE_DIM,
    pack_compressed_sparse_mla_kv_cache_reference,
)
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps,
    _compressed_sparse_mla_scratch_layout,
    _materialize_compressed_sparse_mla_scratch,
)
from b12x.gemm.bf16_gemv import _dual_tc as dual_tc_mod
from b12x.gemm.bf16_gemv import _tensor_core as tc_mod
from b12x.gemm.bf16_gemv._dual_tc import (
    compile_dual_tc_projection,
    dual_tc_flags_numel,
    dual_tc_scratch_numel,
)
from b12x.gemm.bf16_gemv._tensor_core import (
    compile_tc_projection,
    tc_splitk_flags_bytes,
    tc_splitk_scratch_bytes,
)
from b12x.gemm.blockscaled import _mxfp8_tc_gemv as mxfp8_tc_mod
from b12x.gemm.blockscaled._mxfp8_tc_gemv import (
    _mxfp8_tc_gemv_auto_dsmem,
    compile_mxfp8_tc_gemv,
    compile_mxfp8_tc_gemv_rope,
    mxfp8_tc_gemv_flags_bytes,
    mxfp8_tc_gemv_workspace_bytes,
)
from b12x.gemm.blockscaled._nvfp4_tc_gemv import (
    compile_nvfp4_tc_gemv,
    nvfp4_tc_gemv_flags_bytes,
    nvfp4_tc_gemv_workspace_bytes,
)
from b12x.gemm.wo_projection._tc_gemv import (
    compile_grouped_woa_tc_gemv,
    wo_a16_tc_gemv_flags_bytes,
    wo_a16_tc_gemv_splitk_bytes,
)
from b12x.moe._shared.kernels.w4a8_compact_micro import (
    launch_w4a8_compact_micro,
    micro_scratch_nbytes,
)


def _require_sm120() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip(f"SM120+ required, found SM{major}x")
    return torch.device("cuda", torch.cuda.current_device())


def _make_mxfp8_weight(
    n: int, k: int, device: torch.device, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build finite E4M3 weight bytes [n, k] and F8_128x4 UE8M0 scale bytes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mag = torch.randint(0, 120, (n, k), generator=gen, dtype=torch.uint8)
    sign = torch.randint(0, 2, (n, k), generator=gen, dtype=torch.uint8) << 7
    vals = (mag | sign).to(device)
    n_tiles = (n + 127) // 128
    k_blocks4 = ((k // 32) + 3) // 4
    scales = torch.randint(
        122, 127, (n_tiles * k_blocks4 * 512,), generator=gen, dtype=torch.uint8
    ).to(device)
    return vals, scales


def _make_nvfp4_weight(
    n: int, k: int, device: torch.device, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed E2M1 weight bytes [n, k//2] and F8_128x4 E4M3 scales."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vals = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(
        device
    )
    n_tiles = (n + 127) // 128
    k_blocks4 = ((k // 16) + 3) // 4
    scales = torch.randint(
        0x28, 0x44, (n_tiles * k_blocks4 * 512,), generator=gen, dtype=torch.uint8
    ).to(device)
    gscale = torch.tensor([0.5], dtype=torch.float32, device=device)
    return vals, scales, gscale


def _make_mla_workspace(
    *, rows: int, heads: int, width: int, max_chunks: int, device: torch.device
):
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=width,
        head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
        v_head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
        max_chunks_per_row=max_chunks,
        page_size=64,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(layout.nbytes, dtype=torch.uint8, device=device)
    return _materialize_compressed_sparse_mla_scratch(caps, storage, layout, None)


def _make_dsv4_mla_case(
    *,
    rows: int,
    heads: int,
    main_width: int,
    extra_width: int,
    device: torch.device,
    seed: int = 42,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.empty(
        (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(0.0, 0.25, generator=gen)

    def _pack(tokens: int):
        kn = torch.empty(
            (tokens, COMPRESSED_SPARSE_MLA_NOPE_DIM), dtype=torch.float32, device=device
        ).normal_(0.0, 0.3, generator=gen)
        kr = torch.empty(
            (tokens, COMPRESSED_SPARSE_MLA_ROPE_DIM), dtype=torch.float32, device=device
        ).normal_(0.0, 0.3, generator=gen)
        return pack_compressed_sparse_mla_kv_cache_reference(kn, kr, page_size=64)

    main_cache = _pack(max(main_width, 64))
    main_indices = torch.arange(main_width, dtype=torch.int32, device=device).repeat(
        rows, 1
    )
    extra_cache = _pack(max(extra_width, 64)) if extra_width > 0 else None
    extra_indices = (
        torch.arange(extra_width, dtype=torch.int32, device=device).repeat(rows, 1)
        if extra_width > 0
        else None
    )
    return q, main_cache, main_indices, extra_cache, extra_indices


# =============================================================================
# 1. CPU Contract Tests (no GPU required)
# =============================================================================


def test_pdl_helper_gating_and_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_pdl_env(None) is None
    assert parse_pdl_env("1") is True
    assert parse_pdl_env("true") is True
    assert parse_pdl_env("0") is False
    assert parse_pdl_env("off") is False
    with pytest.raises(ValueError, match="Invalid boolean PDL toggle"):
        parse_pdl_env("garbage")

    monkeypatch.setenv("B12X_SM120_CONSUMER_PDL", "1")
    assert sm120_consumer_pdl_enabled(torch.device("cpu")) is False
    assert sm120_consumer_pdl_enabled("cpu") is False

    monkeypatch.setattr("b12x._lib.utils.pdl.get_sm_version", lambda _dev: 90)
    assert sm120_consumer_pdl_enabled(0) is False

    monkeypatch.setattr("b12x._lib.utils.pdl.get_sm_version", lambda _dev: 120)
    monkeypatch.delenv("B12X_SM120_CONSUMER_PDL", raising=False)
    monkeypatch.delenv("B12X_TEST_OP_PDL", raising=False)
    assert sm120_consumer_pdl_enabled(0, env_var="B12X_TEST_OP_PDL") is True

    monkeypatch.setenv("B12X_SM120_CONSUMER_PDL", "0")
    assert sm120_consumer_pdl_enabled(0, env_var="B12X_TEST_OP_PDL") is False

    monkeypatch.setenv("B12X_TEST_OP_PDL", "1")
    assert sm120_consumer_pdl_enabled(0, env_var="B12X_TEST_OP_PDL") is True


def test_pdl_ast_contract_all_decode_kernels() -> None:
    """Every consumer-PDL decode kernel file must wire use_pdl= and griddepcontrol_wait."""
    b12x_root = Path(inspect.getfile(mla_kernel)).parents[3]
    targets = [
        b12x_root / "gemm" / "blockscaled" / "_mxfp8_tc_gemv.py",
        b12x_root / "gemm" / "blockscaled" / "_nvfp4_tc_gemv.py",
        b12x_root / "gemm" / "wo_projection" / "_tc_gemv.py",
        b12x_root / "gemm" / "bf16_gemv" / "_tensor_core.py",
        b12x_root / "gemm" / "bf16_gemv" / "_dual_tc.py",
        b12x_root / "gemm" / "bf16_gemv" / "_kernel.py",
        b12x_root / "attention" / "_shared" / "mla" / "kernel.py",
        b12x_root / "attention" / "_shared" / "mla" / "merge.py",
        b12x_root / "moe" / "_shared" / "kernels" / "w4a8_compact_projection.py",
    ]
    for path in targets:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        has_use_pdl_kw = any(
            isinstance(node, ast.keyword) and node.arg == "use_pdl"
            for node in ast.walk(tree)
        )
        has_gdc_wait = any(
            isinstance(node, ast.Attribute) and node.attr == "griddepcontrol_wait"
            for node in ast.walk(tree)
        )
        assert has_use_pdl_kw, f"{path.name}: missing use_pdl= launch keyword"
        assert has_gdc_wait, f"{path.name}: missing griddepcontrol_wait()"


def test_dsmem_helper_and_tc_gemv_auto_occupancy_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DSMEM helper gating and occupancy ceilings for MXFP8 and BF16 TC-GEMV."""
    for s in SM120_DSMEM_CLUSTER_SIZES:
        assert sm120_dsmem_split_eligible(s) is True
    for s in (0, 1, 3, 5, 6, 7, 16):
        assert sm120_dsmem_split_eligible(s) is False

    monkeypatch.setenv("B12X_SM120_DSMEM_SPLIT", "1")
    assert sm120_dsmem_split_enabled(torch.device("cpu"), num_splits=4) is False

    monkeypatch.setattr("b12x._lib.utils.dsmem.get_sm_version", lambda _dev: 90)
    assert sm120_dsmem_split_enabled(0, num_splits=4) is False

    monkeypatch.setattr("b12x._lib.utils.dsmem.get_sm_version", lambda _dev: 120)
    monkeypatch.delenv("B12X_SM120_DSMEM_SPLIT", raising=False)
    monkeypatch.delenv("B12X_TC_GEMV_DSMEM_SPLIT", raising=False)
    assert (
        sm120_dsmem_split_enabled(0, num_splits=4, env_var="B12X_TC_GEMV_DSMEM_SPLIT")
        is False
    )

    monkeypatch.setenv("B12X_TC_GEMV_DSMEM_SPLIT", "1")
    assert (
        sm120_dsmem_split_enabled(0, num_splits=4, env_var="B12X_TC_GEMV_DSMEM_SPLIT")
        is True
    )
    assert (
        sm120_dsmem_split_enabled(0, num_splits=3, env_var="B12X_TC_GEMV_DSMEM_SPLIT")
        is False
    )

    # MXFP8 TC-GEMV auto occupancy gating on 188-SM RTX PRO 6000 (2 * sm_count = 376 CTAs):
    monkeypatch.setattr(
        mxfp8_tc_mod.torch.cuda,
        "get_device_properties",
        lambda _ord: SimpleNamespace(multi_processor_count=188),
    )
    # qkv_a (n=1792, n_per_cta=16 -> 112 blocks): sk=2 allowed; sk=4 (448 > 376) blocked.
    assert _mxfp8_tc_gemv_auto_dsmem(0, 1792, 2, 16) is True
    assert _mxfp8_tc_gemv_auto_dsmem(0, 1792, 4, 16) is False
    # shared_gate_up (n=1152, n_per_cta=16 -> 72 blocks): sk=4 (288 <= 376) allowed; sk=8 (576 > 376) blocked.
    assert _mxfp8_tc_gemv_auto_dsmem(0, 1152, 4, 16) is True
    assert _mxfp8_tc_gemv_auto_dsmem(0, 1152, 8, 16) is False
    # n_per_cta=32 halves n_blocks (1792 -> 56 blocks): sk=4 (224 <= 376) allowed.
    assert _mxfp8_tc_gemv_auto_dsmem(0, 1792, 4, 32) is True

    # BF16 TC-GEMV auto gating: allow split_k == 2 or (split_k == 4 and max_rows <= 8).
    monkeypatch.setattr("b12x._lib.utils.pdl.get_sm_version", lambda _dev: 120)
    captured_tc: list[bool] = []
    monkeypatch.setattr(
        tc_mod,
        "_compile_tc_projection_cached",
        lambda _o, _m, _sk, _n, _k, _dt, _pdl, dsmem: captured_tc.append(dsmem),
    )
    compile_tc_projection(0, 32, 2, 384, 5120, "float32")
    compile_tc_projection(0, 4, 4, 384, 5120, "float32")
    compile_tc_projection(0, 8, 4, 384, 5120, "float32")
    compile_tc_projection(0, 16, 4, 384, 5120, "float32")
    compile_tc_projection(0, 4, 8, 384, 5120, "float32")
    compile_tc_projection(0, 16, 4, 384, 5120, "float32", use_dsmem=True)
    compile_tc_projection(0, 4, 2, 384, 5120, "float32", use_dsmem=False)
    assert captured_tc == [True, True, True, False, False, True, False]

    captured_dual: list[bool] = []
    monkeypatch.setattr(
        dual_tc_mod,
        "_compile_dual_tc_projection_cached",
        lambda _o, _m, _sk, _na, _nb, _k, _pdl, dsmem: captured_dual.append(dsmem),
    )
    compile_dual_tc_projection(0, 32, 2, 512, 384, 5120)
    compile_dual_tc_projection(0, 8, 4, 512, 384, 5120)
    compile_dual_tc_projection(0, 16, 4, 512, 384, 5120)
    compile_dual_tc_projection(0, 4, 8, 512, 384, 5120)
    compile_dual_tc_projection(0, 16, 4, 512, 384, 5120, use_dsmem=True)
    assert captured_dual == [True, True, False, False, True]


# =============================================================================
# 2. Live SM120 GPU 0-ULP Parity Tests — Consumer-Side PDL
# =============================================================================


@torch.inference_mode()
def test_pdl_0ulp_parity_decode_gemm_kernels() -> None:
    """Every consumer-PDL decode GEMM kernel is 0-ULP bit-exact with use_pdl=True vs False."""
    device = _require_sm120()
    ordinal = int(device.index or 0)
    gen = torch.Generator(device="cpu").manual_seed(2026)

    # 1. MXFP8MultiRowTCGemvKernel (split_k=1 and split_k=2)
    for n, k, sk in ((5120, 2048, 1), (2112, 5120, 2)):
        m = 8
        x = (torch.randn((m, k), generator=gen) * 0.2).to(
            device=device, dtype=torch.bfloat16
        )
        w, s = _make_mxfp8_weight(n, k, device, seed=10 + sk)
        sc = (
            torch.empty(
                mxfp8_tc_gemv_workspace_bytes(m, n, sk) // 4,
                dtype=torch.float32,
                device=device,
            )
            if sk > 1
            else None
        )
        fl = (
            torch.zeros(
                mxfp8_tc_gemv_flags_bytes(n, sk) // 4,
                dtype=torch.uint32,
                device=device,
            )
            if sk > 1
            else None
        )
        out_off = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        out_on = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        compile_mxfp8_tc_gemv(
            ordinal, n, k, k, m, sk, 4, use_pdl=False, use_dsmem=False
        )(x, w, s, out_off, sc, fl, None)
        compile_mxfp8_tc_gemv(
            ordinal, n, k, k, m, sk, 4, use_pdl=True, use_dsmem=False
        )(x, w, s, out_on, sc, fl, None)
        torch.cuda.synchronize(device)
        assert torch.equal(out_on.view(torch.uint16), out_off.view(torch.uint16))

    # 2. MXFP8MultiRowTCGemvKernel fused RoPE
    m, n, k, sk = 8, 1024, 5120, 2
    x = (torch.randn((m, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    w, s = _make_mxfp8_weight(n, k, device, seed=19)
    cos_sin = torch.randn((m, 64), generator=gen, dtype=torch.float32).to(device)
    sc = torch.empty(
        mxfp8_tc_gemv_workspace_bytes(m, n, sk) // 4, dtype=torch.float32, device=device
    )
    fl = torch.zeros(
        mxfp8_tc_gemv_flags_bytes(n, sk) // 4, dtype=torch.uint32, device=device
    )
    out_off = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    out_on = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    compile_mxfp8_tc_gemv_rope(
        ordinal, n, k, k, m, sk, 4, use_pdl=False, use_dsmem=False
    )(x, w, s, out_off, cos_sin, sc, fl, None)
    compile_mxfp8_tc_gemv_rope(
        ordinal, n, k, k, m, sk, 4, use_pdl=True, use_dsmem=False
    )(x, w, s, out_on, cos_sin, sc, fl, None)
    torch.cuda.synchronize(device)
    assert torch.equal(out_on.view(torch.uint16), out_off.view(torch.uint16))

    # 3. NVFP4MultiRowTCGemvKernel
    m, n, k, sk = 8, 2048, 5120, 2
    x = (torch.randn((m, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    w, s, gs = _make_nvfp4_weight(n, k, device, seed=21)
    sc = torch.empty(
        nvfp4_tc_gemv_workspace_bytes(m, n, sk) // 4, dtype=torch.float32, device=device
    )
    fl = torch.zeros(
        nvfp4_tc_gemv_flags_bytes(n, sk) // 4, dtype=torch.uint32, device=device
    )
    out_off = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    out_on = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    compile_nvfp4_tc_gemv(ordinal, n, k, k, m, sk, 4, use_pdl=False)(
        x, w, s, gs, out_off, sc, fl, None
    )
    compile_nvfp4_tc_gemv(ordinal, n, k, k, m, sk, 4, use_pdl=True)(
        x, w, s, gs, out_on, sc, fl, None
    )
    torch.cuda.synchronize(device)
    assert torch.equal(out_on.view(torch.uint16), out_off.view(torch.uint16))

    # 4. GroupedWoaTCGemvKernel (plain + inv-RoPE)
    m, groups, rank, width, sk = 8, 2, 1024, 4096, 2
    src = (torch.randn((m, groups, width), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    wa, sa = _make_mxfp8_weight(groups * rank, width, device, seed=25)
    pos = torch.arange(m, dtype=torch.int64, device=device)
    cs_cache = torch.randn((16, 64), generator=gen, dtype=torch.float32).to(device)
    sc = torch.empty(
        wo_a16_tc_gemv_splitk_bytes(m, groups, rank, sk) // 4,
        dtype=torch.float32,
        device=device,
    )
    fl = torch.zeros(
        wo_a16_tc_gemv_flags_bytes(groups, rank, sk) // 4,
        dtype=torch.uint32,
        device=device,
    )
    for inv_rope in (False, True):
        out_off = torch.empty((m, groups * rank), dtype=torch.bfloat16, device=device)
        out_on = torch.empty((m, groups * rank), dtype=torch.bfloat16, device=device)
        compile_grouped_woa_tc_gemv(
            ordinal,
            groups,
            rank,
            width,
            m,
            sk,
            4,
            fuse_inv_rope=inv_rope,
            use_pdl=False,
            use_dsmem=False,
        )(src, pos, cs_cache, wa, sa, out_off, sc, fl, None)
        compile_grouped_woa_tc_gemv(
            ordinal,
            groups,
            rank,
            width,
            m,
            sk,
            4,
            fuse_inv_rope=inv_rope,
            use_pdl=True,
            use_dsmem=False,
        )(src, pos, cs_cache, wa, sa, out_on, sc, fl, None)
        torch.cuda.synchronize(device)
        assert torch.equal(out_on.view(torch.uint16), out_off.view(torch.uint16))

    # 5. TensorCoreGemvKernel + DualBf16TCGemvKernel
    m, n, k, sk = 8, 384, 5120, 4
    x = (torch.randn((m, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    wb = (torch.randn((n, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    sc = torch.empty(
        tc_splitk_scratch_bytes(m, n, sk) // 4, dtype=torch.float32, device=device
    )
    fl = torch.zeros(
        tc_splitk_flags_bytes(n, sk) // 4, dtype=torch.uint32, device=device
    )
    out_off = torch.empty((m, n), dtype=torch.float32, device=device)
    out_on = torch.empty((m, n), dtype=torch.float32, device=device)
    compile_tc_projection(
        ordinal, m, sk, n, k, "float32", use_pdl=False, use_dsmem=False
    )(x, wb, out_off, scratch=sc, flags=fl)
    compile_tc_projection(
        ordinal, m, sk, n, k, "float32", use_pdl=True, use_dsmem=False
    )(x, wb, out_on, scratch=sc, flags=fl)
    torch.cuda.synchronize(device)
    assert torch.equal(out_on.view(torch.uint32), out_off.view(torch.uint32))

    n_a, n_b = 512, 384
    w_a = (torch.randn((n_a, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    w_b = (torch.randn((n_b, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    sc_d = torch.empty(
        dual_tc_scratch_numel(n_a, n_b, m, sk), dtype=torch.float32, device=device
    )
    fl_d = torch.zeros(
        dual_tc_flags_numel(n_a, n_b), dtype=torch.uint32, device=device
    )
    oa_off = torch.empty((m, n_a), dtype=torch.bfloat16, device=device)
    ob_off = torch.empty((m, n_b), dtype=torch.float32, device=device)
    oa_on, ob_on = torch.empty_like(oa_off), torch.empty_like(ob_off)
    compile_dual_tc_projection(
        ordinal, m, sk, n_a, n_b, k, use_pdl=False, use_dsmem=False
    )(x, w_a, w_b, oa_off, ob_off, scratch=sc_d, flags=fl_d)
    compile_dual_tc_projection(
        ordinal, m, sk, n_a, n_b, k, use_pdl=True, use_dsmem=False
    )(x, w_a, w_b, oa_on, ob_on, scratch=sc_d, flags=fl_d)
    torch.cuda.synchronize(device)
    assert torch.equal(oa_on.view(torch.uint16), oa_off.view(torch.uint16))
    assert torch.equal(ob_on.view(torch.uint32), ob_off.view(torch.uint32))


@torch.inference_mode()
def test_pdl_0ulp_parity_mla_and_w4a8_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    """UnifiedDecodeKernel + Merge and W4A8CompactMicro are 0-ULP bit-exact with PDL ON vs OFF."""
    device = _require_sm120()
    rows, heads = 4, 16
    q, mc, mi, ec, ei = _make_dsv4_mla_case(
        rows=rows, heads=heads, main_width=128, extra_width=256, device=device, seed=31
    )
    ml = torch.tensor([128, 96, 64, 128], dtype=torch.int32, device=device)
    el = torch.tensor([256, 128, 64, 192], dtype=torch.int32, device=device)
    ws = _make_mla_workspace(
        rows=rows, heads=heads, width=384, max_chunks=8, device=device
    )
    out_off = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=device)
    out_on = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=device)

    monkeypatch.setenv("B12X_MLA_DECODE_PDL", "0")
    monkeypatch.setenv("B12X_MLA_MERGE_PDL", "0")
    mla_kernel.run_unified_decode(
        q_all=q,
        swa_k_cache=mc,
        swa_indices=mi,
        swa_topk_lengths=ml,
        indexed_k_cache=ec,
        indexed_indices=ei,
        indexed_topk_lengths=el,
        indexed_page_size=64,
        workspace=ws,
        sm_scale=512**-0.5,
        swa_page_size=64,
        forced_num_splits=4,
        out=out_off,
    )

    monkeypatch.setenv("B12X_MLA_DECODE_PDL", "1")
    monkeypatch.setenv("B12X_MLA_MERGE_PDL", "1")
    mla_kernel.run_unified_decode(
        q_all=q,
        swa_k_cache=mc,
        swa_indices=mi,
        swa_topk_lengths=ml,
        indexed_k_cache=ec,
        indexed_indices=ei,
        indexed_topk_lengths=el,
        indexed_page_size=64,
        workspace=ws,
        sm_scale=512**-0.5,
        swa_page_size=64,
        forced_num_splits=4,
        out=out_on,
    )
    torch.cuda.synchronize(device)
    assert torch.equal(out_on.view(torch.uint16), out_off.view(torch.uint16))

    from tests.experimental.b12x.moe.test_w4a8_compact_coalesced import _make_experts

    m, k, n, top_k, num_experts = 8, 5120, 576, 6, 32
    experts, *_ = _make_experts(
        device,
        num_experts=num_experts,
        hidden_size=k,
        intermediate_size=n,
        csf=False,
        seed=44,
    )
    impl = experts._impl
    rep = impl.representation.value
    gen = torch.Generator(device="cpu").manual_seed(45)
    a = (torch.randn((m, k), generator=gen) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    ids = torch.randint(0, num_experts, (m, top_k), generator=gen, dtype=torch.int32).to(
        device
    )
    ids[-1, :] = -1
    weights = torch.softmax(torch.randn((m, top_k), generator=gen), dim=-1).to(
        device=device, dtype=torch.float32
    )
    scratch = torch.empty(
        micro_scratch_nbytes(m, k, n, top_k), dtype=torch.uint8, device=device
    )
    moe_off = torch.empty((m, k), dtype=torch.bfloat16, device=device)
    moe_on = torch.empty((m, k), dtype=torch.bfloat16, device=device)
    common = dict(
        scratch=scratch,
        a=a,
        topk_ids=ids,
        topk_weights=weights,
        w13=rep.w13_rp,
        w13_scales=rep.w13_sfb,
        w2=rep.w2_rp,
        w2_scales=rep.w2_sfb,
        alpha1=impl.w1_alphas,
        alpha2=impl.w2_alphas,
        input_scale=impl.a1_gscale,
        down_scale=impl.a2_gscale,
        max_tokens=m,
        num_topk=top_k,
        swiglu_limit=10.0,
        fast_math=True,
        coalesced=True,
    )
    launch_w4a8_compact_micro(**common, output=moe_off, use_pdl=False)
    launch_w4a8_compact_micro(**common, output=moe_on, use_pdl=True)
    torch.cuda.synchronize(device)
    assert torch.equal(moe_on.view(torch.uint16), moe_off.view(torch.uint16))


# =============================================================================
# 3. Live SM120 GPU 0-ULP Parity Tests — TC-GEMV DSMEM Cluster Split-K
# =============================================================================


@torch.inference_mode()
@pytest.mark.parametrize("split_k", [2, 4, 8])
def test_tc_gemv_dsmem_cluster_splitk_0ulp_parity(split_k: int) -> None:
    """All four TC-GEMV DSMEM cluster split-K kernels are 0-ULP bit-exact vs L2 split-K."""
    device = _require_sm120()
    ordinal = int(device.index or 0)
    gen = torch.Generator(device="cpu").manual_seed(1200 + split_k)

    # 1. TensorCoreGemvKernel (router N=384, K=5120, FP32 & BF16, M in {1, 8, 16})
    for m, out_dtype in ((1, "float32"), (8, "float32"), (16, "bfloat16")):
        n, k = 384, 5120
        dt = getattr(torch, out_dtype)
        x = (torch.randn((m, k), generator=gen) * 0.2).to(
            device=device, dtype=torch.bfloat16
        )
        w = (torch.randn((n, k), generator=gen) * 0.2).to(
            device=device, dtype=torch.bfloat16
        )
        sc = torch.empty(
            tc_splitk_scratch_bytes(m, n, split_k) // 4,
            dtype=torch.float32,
            device=device,
        )
        fl = torch.zeros(
            tc_splitk_flags_bytes(n, split_k) // 4, dtype=torch.uint32, device=device
        )
        out_ref = torch.empty((m, n), dtype=dt, device=device)
        out_got = torch.empty((m, n), dtype=dt, device=device)
        compile_tc_projection(ordinal, m, split_k, n, k, out_dtype, use_dsmem=False)(
            x, w, out_ref, scratch=sc, flags=fl
        )
        compile_tc_projection(ordinal, m, split_k, n, k, out_dtype, use_dsmem=True)(
            x, w, out_got, scratch=None, flags=None
        )
        torch.cuda.synchronize(device)
        view_dt = torch.uint32 if dt == torch.float32 else torch.uint16
        assert torch.equal(out_got.view(view_dt), out_ref.view(view_dt))

    # 2. DualBf16TCGemvKernel
    m, n_a, n_b, k = 8, 512, 384, 5120
    x = (torch.randn((m, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    w_a = (torch.randn((n_a, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    w_b = (torch.randn((n_b, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    sc = torch.empty(
        dual_tc_scratch_numel(n_a, n_b, m, split_k), dtype=torch.float32, device=device
    )
    fl = torch.zeros(
        dual_tc_flags_numel(n_a, n_b), dtype=torch.uint32, device=device
    )
    oa_ref = torch.empty((m, n_a), dtype=torch.bfloat16, device=device)
    ob_ref = torch.empty((m, n_b), dtype=torch.float32, device=device)
    oa_got, ob_got = torch.empty_like(oa_ref), torch.empty_like(ob_ref)
    compile_dual_tc_projection(ordinal, m, split_k, n_a, n_b, k, use_dsmem=False)(
        x, w_a, w_b, oa_ref, ob_ref, scratch=sc, flags=fl
    )
    compile_dual_tc_projection(ordinal, m, split_k, n_a, n_b, k, use_dsmem=True)(
        x, w_a, w_b, oa_got, ob_got, scratch=None, flags=None
    )
    torch.cuda.synchronize(device)
    assert torch.equal(oa_got.view(torch.uint16), oa_ref.view(torch.uint16))
    assert torch.equal(ob_got.view(torch.uint32), ob_ref.view(torch.uint32))

    # 3. GroupedWoaTCGemvKernel (plain + inv-RoPE)
    m, groups, rank, width = 8, 2, 1024, 4096
    src = (torch.randn((m, groups, width), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    wa, sa = _make_mxfp8_weight(groups * rank, width, device, seed=1210 + split_k)
    pos = torch.arange(m, dtype=torch.int64, device=device)
    cs_cache = torch.randn((16, 64), generator=gen, dtype=torch.float32).to(device)
    sc = torch.empty(
        wo_a16_tc_gemv_splitk_bytes(m, groups, rank, split_k) // 4,
        dtype=torch.float32,
        device=device,
    )
    fl = torch.zeros(
        wo_a16_tc_gemv_flags_bytes(groups, rank, split_k) // 4,
        dtype=torch.uint32,
        device=device,
    )
    for inv_rope in (False, True):
        out_ref = torch.empty((m, groups * rank), dtype=torch.bfloat16, device=device)
        out_got = torch.empty((m, groups * rank), dtype=torch.bfloat16, device=device)
        compile_grouped_woa_tc_gemv(
            ordinal,
            groups,
            rank,
            width,
            m,
            split_k,
            4,
            fuse_inv_rope=inv_rope,
            use_dsmem=False,
        )(src, pos, cs_cache, wa, sa, out_ref, sc, fl, None)
        compile_grouped_woa_tc_gemv(
            ordinal,
            groups,
            rank,
            width,
            m,
            split_k,
            4,
            fuse_inv_rope=inv_rope,
            use_dsmem=True,
        )(src, pos, cs_cache, wa, sa, out_got, None, None, None)
        torch.cuda.synchronize(device)
        assert torch.equal(out_got.view(torch.uint16), out_ref.view(torch.uint16))

    # 4. MXFP8MultiRowTCGemvKernel (plain qkv_a + fused RoPE + CUDA-graph replay)
    m, n, k = 8, 2112, 5120
    x = (torch.randn((m, k), generator=gen) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    wm, sm = _make_mxfp8_weight(n, k, device, seed=1220 + split_k)
    sc = torch.empty(
        mxfp8_tc_gemv_workspace_bytes(m, n, split_k) // 4,
        dtype=torch.float32,
        device=device,
    )
    fl = torch.zeros(
        mxfp8_tc_gemv_flags_bytes(n, split_k) // 4, dtype=torch.uint32, device=device
    )
    out_ref = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    out_got = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    compile_mxfp8_tc_gemv(ordinal, n, k, k, m, split_k, 4, use_dsmem=False)(
        x, wm, sm, out_ref, sc, fl, None
    )
    dsmem_fn = compile_mxfp8_tc_gemv(ordinal, n, k, k, m, split_k, 4, use_dsmem=True)
    dsmem_fn(x, wm, sm, out_got, None, None, None)
    torch.cuda.synchronize(device)
    assert torch.equal(out_got.view(torch.uint16), out_ref.view(torch.uint16))

    out_graph = torch.empty_like(out_got)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsmem_fn(x, wm, sm, out_graph, None, None, None)
    out_graph.zero_()
    alloc_before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == alloc_before
    assert torch.equal(out_graph.view(torch.uint16), out_ref.view(torch.uint16))

    n_rope = 1024
    wr, sr = _make_mxfp8_weight(n_rope, k, device, seed=1230 + split_k)
    cos_sin = torch.randn((m, 64), generator=gen, dtype=torch.float32).to(device)
    sc_r = torch.empty(
        mxfp8_tc_gemv_workspace_bytes(m, n_rope, split_k) // 4,
        dtype=torch.float32,
        device=device,
    )
    fl_r = torch.zeros(
        mxfp8_tc_gemv_flags_bytes(n_rope, split_k) // 4,
        dtype=torch.uint32,
        device=device,
    )
    or_ref = torch.empty((m, n_rope), dtype=torch.bfloat16, device=device)
    or_got = torch.empty((m, n_rope), dtype=torch.bfloat16, device=device)
    compile_mxfp8_tc_gemv_rope(ordinal, n_rope, k, k, m, split_k, 4, use_dsmem=False)(
        x, wr, sr, or_ref, cos_sin, sc_r, fl_r, None
    )
    compile_mxfp8_tc_gemv_rope(ordinal, n_rope, k, k, m, split_k, 4, use_dsmem=True)(
        x, wr, sr, or_got, cos_sin, None, None, None
    )
    torch.cuda.synchronize(device)
    assert torch.equal(or_got.view(torch.uint16), or_ref.view(torch.uint16))
