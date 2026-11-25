"""CPU/host tests for DSV4.1 decode BF16 fast path, single-split merge bypass, and split planner."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from b12x.attention import compressed_sparse_mla as mla
from b12x.attention._shared.mla import decode_math, io as mla_io, kernel as mla_kernel
from b12x.attention._shared.mla.smem import SM120_SMEM_CARVEOUT_BYTES, make_smem_layout
from b12x.attention._shared.mla.traits import (
    ComputeMode,
    DSV41_INDEXED_SMEM_STRIDE,
    DSV41_SWA_SMEM_STRIDE,
    ModelType,
    ScaleFormat,
    make_unified_traits,
)
from b12x.attention.compressed_sparse_mla._tuning import TUNING, _single_pass
from b12x.preparation import DeviceIdentity


SM120 = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
SM121 = DeviceIdentity("nvidia", (12, 1), 48, "SM121")


def _dummy_workspace(
    *,
    rows: int,
    heads: int,
    max_chunks_per_row: int = 16,
    num_chunks_value: int = 16,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_chunks_per_row=max_chunks_per_row,
        num_chunks_value=num_chunks_value,
        tmp_output=torch.empty(
            (rows, heads, max_chunks_per_row, 512), dtype=torch.bfloat16
        ),
        tmp_lse=torch.empty((rows, heads, max_chunks_per_row), dtype=torch.float32),
        output_buffer=torch.empty((rows, heads, 512), dtype=torch.bfloat16),
        final_lse=torch.empty((rows, heads), dtype=torch.float32),
        num_chunks_ptr=torch.tensor([max_chunks_per_row], dtype=torch.int32),
    )


def _dsv41_bf16_traits():
    return make_unified_traits(
        ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3
    )


def test_dsv41_304b_indexed_stride_fits_double_buffer_and_is_conflict_free() -> None:
    traits = _dsv41_bf16_traits()
    layout = make_smem_layout(traits)

    assert traits.kv_smem_stride == DSV41_SWA_SMEM_STRIDE == 544
    assert DSV41_INDEXED_SMEM_STRIDE == 304
    assert DSV41_INDEXED_SMEM_STRIDE % 16 == 0
    assert math.gcd(DSV41_INDEXED_SMEM_STRIDE // 4, 32) == 4
    assert layout.kv_fp8_buf_bytes == 64 * 544
    assert layout.kv_fp8_buf_bytes >= 64 * DSV41_INDEXED_SMEM_STRIDE
    assert layout.total_bytes < SM120_SMEM_CARVEOUT_BYTES


def test_unified_decode_kernel_enables_vector_q_bf16_for_nvfp4_bf16_only() -> None:
    dsv41_bf16 = _dsv41_bf16_traits()
    dsv41_fp8 = replace(
        dsv41_bf16,
        compute_mode=ComputeMode.FP8,
        fp8_internal=True,
        q_nope_stride=528,
        kv_smem_stride=624,
    )
    glm_nvfp4 = make_unified_traits(
        ModelType.GLM_NSA, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3, fp8_rope=False
    )
    dsv4_fp8 = make_unified_traits(
        ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE
    )

    def _make_kernel(
        traits,
        *,
        vector_q=True,
        native_dsv41_fp8=False,
        direct_output=False,
        has_attn_sink=False,
    ):
        return mla_kernel.UnifiedDecodeKernel(
            traits,
            make_smem_layout(traits),
            page_block_size=64,
            chunks_per_split=10,
            h_blocks=1,
            num_splits=1 if direct_output else 2,
            num_heads=16,
            q_head_dim=traits.d_nope + traits.d_rope,
            topk=128,
            extra_topk=512,
            q_stride=(
                16 * (traits.d_nope + traits.d_rope),
                traits.d_nope + traits.d_rope,
                1,
            ),
            swa_indices_stride0=128,
            extra_indices_stride0=512,
            mid_out_stride=(16 * 512, 512, 512, 1),
            mid_lse_stride=(16, 1, 1),
            has_extra=traits.has_extra_cache,
            pbs_extra=64,
            valid_hpb=16,
            native_dsv41_fp8=native_dsv41_fp8,
            vector_q=vector_q,
            direct_output=direct_output,
            has_attn_sink=has_attn_sink,
        )

    k_v41_bf16 = _make_kernel(
        dsv41_bf16, vector_q=True, direct_output=True, has_attn_sink=True
    )
    assert k_v41_bf16.vector_q is True
    assert k_v41_bf16.vector_q_bf16 is True
    assert k_v41_bf16.direct_output is True
    assert k_v41_bf16.has_attn_sink is True

    k_v41_unaligned = _make_kernel(dsv41_bf16, vector_q=False)
    assert k_v41_unaligned.vector_q is False
    assert k_v41_unaligned.vector_q_bf16 is False
    assert k_v41_unaligned.direct_output is False
    assert k_v41_unaligned.has_attn_sink is False

    k_v41_fp8 = _make_kernel(dsv41_fp8, vector_q=True, native_dsv41_fp8=True)
    assert k_v41_fp8.vector_q is False
    assert k_v41_fp8.vector_q_bf16 is False

    k_glm_nvfp4 = _make_kernel(glm_nvfp4, vector_q=True)
    assert k_glm_nvfp4.vector_q is True
    assert k_glm_nvfp4.vector_q_bf16 is True

    k_dsv4 = _make_kernel(dsv4_fp8, vector_q=True)
    assert k_dsv4.vector_q is True
    assert k_dsv4.vector_q_bf16 is False


def test_dsv41_split_planner_uses_wave_balancing_instead_of_forced_max_splits() -> None:
    traits = _dsv41_bf16_traits()
    swa_cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    idx_cache = torch.empty((8, 64 * 288), dtype=torch.uint8)

    # SWA-only layer (2 chunks) at P=128 on 188-SM SM120: wave balancing selects
    # num_splits=1 (128 CTAs in 1 clean wave) and enables direct_output merge bypass,
    # even when workspace.num_chunks_value == 11 (chunk_size=12 contract).
    q_128 = torch.empty((128, 16, 512), dtype=torch.bfloat16)
    swa_idx_128 = torch.zeros((128, 128), dtype=torch.int32)
    ws_swa = _dummy_workspace(
        rows=128, heads=16, max_chunks_per_row=16, num_chunks_value=11
    )
    launch_swa = mla_kernel.prepare_unified_decode_launch(
        q_all=q_128,
        q_alignment=16,
        swa_k_cache=swa_cache,
        swa_indices=swa_idx_128,
        workspace=ws_swa,
        swa_page_size=64,
        sm_count=188,
        v41_compute_mode="bf16",
        traits_override=traits,
    )
    assert launch_swa.num_chunks == 2
    assert launch_swa.num_splits == 1
    assert launch_swa.chunks_per_split == 2
    assert launch_swa.direct_output is True

    # Dual-cache layer (10 chunks) at P=128 on 188-SM SM120: wave balancing selects
    # num_splits=4 (512 CTAs across 3 waves, gap=0.277) instead of forced max 10 splits.
    idx_idx_128 = torch.zeros((128, 512), dtype=torch.int32)
    ws_dual = _dummy_workspace(
        rows=128, heads=16, max_chunks_per_row=54, num_chunks_value=54
    )
    launch_dual = mla_kernel.prepare_unified_decode_launch(
        q_all=q_128,
        q_alignment=16,
        swa_k_cache=swa_cache,
        swa_indices=swa_idx_128,
        indexed_k_cache=idx_cache,
        indexed_indices=idx_idx_128,
        indexed_page_size=64,
        workspace=ws_dual,
        swa_page_size=64,
        sm_count=188,
        v41_compute_mode="bf16",
        traits_override=traits,
    )
    assert launch_dual.num_chunks == 10
    assert launch_dual.num_splits == 4
    assert launch_dual.direct_output is False

    # When max_chunks_per_row == 1 (the single-split autotune candidate), num_splits == 1
    # triggers direct_output merge bypass for dual-cache 10-chunk layers as well.
    ws_single = _dummy_workspace(
        rows=128, heads=16, max_chunks_per_row=1, num_chunks_value=1
    )
    launch_single = mla_kernel.prepare_unified_decode_launch(
        q_all=q_128,
        q_alignment=16,
        swa_k_cache=swa_cache,
        swa_indices=swa_idx_128,
        indexed_k_cache=idx_cache,
        indexed_indices=idx_idx_128,
        indexed_page_size=64,
        workspace=ws_single,
        swa_page_size=64,
        sm_count=188,
        v41_compute_mode="bf16",
        traits_override=traits,
    )
    assert launch_single.num_splits == 1
    assert launch_single.chunks_per_split == 10
    assert launch_single.direct_output is True

    # At P=188 (fills 188 SMs in 1 wave), wave balancing chooses num_splits=1 even
    # with max_chunks_per_row=54.
    q_188 = torch.empty((188, 16, 512), dtype=torch.bfloat16)
    swa_idx_188 = torch.zeros((188, 128), dtype=torch.int32)
    idx_idx_188 = torch.zeros((188, 512), dtype=torch.int32)
    ws_188 = _dummy_workspace(
        rows=188, heads=16, max_chunks_per_row=54, num_chunks_value=54
    )
    launch_188 = mla_kernel.prepare_unified_decode_launch(
        q_all=q_188,
        q_alignment=16,
        swa_k_cache=swa_cache,
        swa_indices=swa_idx_188,
        indexed_k_cache=idx_cache,
        indexed_indices=idx_idx_188,
        indexed_page_size=64,
        workspace=ws_188,
        swa_page_size=64,
        sm_count=188,
        v41_compute_mode="bf16",
        traits_override=traits,
    )
    assert launch_188.num_splits == 1
    assert launch_188.direct_output is True


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize(
    "return_lse,lse_scale", [(False, "base2"), (True, "base2"), (True, "natural")]
)
def test_num_splits_one_bypasses_merge_and_handles_sink_and_lse(
    monkeypatch, with_sink: bool, return_lse: bool, lse_scale: str
) -> None:
    import b12x.attention._shared.mla.api as mla_api
    import b12x.attention._shared.mla.merge as mla_merge

    rows, heads = 4, 16
    traits = _dsv41_bf16_traits()
    q = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    swa_cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    swa_indices = torch.zeros((rows, 128), dtype=torch.int32)
    swa_lengths = torch.full((rows,), 128, dtype=torch.int32)
    sink = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32) if with_sink else None
    out = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    ws = _dummy_workspace(
        rows=rows, heads=heads, max_chunks_per_row=1, num_chunks_value=1
    )

    grid_calls = []

    def fake_grid_launch(*args, **kwargs):
        grid_calls.append((args, kwargs))
        # Simulate s7_epilogue writing base-2 LSE = 2.0 into grid_lse (args[4]).
        args[4].fill_(2.0)
        return "grid-program"

    def forbid_merge(*args, **kwargs):
        raise AssertionError("merge kernel must not run when num_splits == 1")

    def forbid_lse(*args, **kwargs):
        raise AssertionError(
            "_final_lse_from_split_workspace must not run when num_splits == 1"
        )

    monkeypatch.setattr(
        mla_kernel, "_sparse_mla_decode_grid_flat_launch", fake_grid_launch
    )
    monkeypatch.setattr(mla_merge, "run_sparse_mla_split_decode_merge", forbid_merge)
    monkeypatch.setattr(mla_api, "_final_lse_from_split_workspace", forbid_lse)

    prepared = mla_kernel.prepare_unified_decode_launch(
        q_all=q,
        q_alignment=16,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        workspace=ws,
        swa_page_size=64,
        sm_count=188,
        attn_sink=sink,
        return_lse=return_lse,
        v41_compute_mode="bf16",
        traits_override=traits,
    )
    assert prepared.num_splits == 1
    assert prepared.direct_output is True

    compiled = mla_kernel.compile_unified_decode_launch(
        prepared=prepared,
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        workspace=ws,
        sm_scale=512**-0.5,
        swa_page_size=64,
        attn_sink=sink,
        return_lse=return_lse,
        lse_scale=lse_scale,
        out=out,
        v41_compute_mode="bf16",
    )
    assert compiled.grid_programs == ("grid-program",)
    assert compiled.merge_program is None
    assert compiled.lse_program is None
    assert len(grid_calls) == 1

    grid_calls.clear()
    result = mla_kernel.run_prepared_unified_decode(
        prepared=compiled,
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        workspace=ws,
        sm_scale=512**-0.5,
        swa_page_size=64,
        attn_sink=sink,
        return_lse=return_lse,
        lse_scale=lse_scale,
        out=out,
        v41_compute_mode="bf16",
    )
    assert len(grid_calls) == 1
    args, kwargs = grid_calls[0]
    assert kwargs["direct_output"] is True
    if with_sink:
        assert kwargs["attn_sink"] is not None
        assert kwargs["attn_sink"].data_ptr() == sink.data_ptr()
    else:
        assert kwargs["attn_sink"] is None
    assert args[3].data_ptr() == out.data_ptr()
    assert args[4].data_ptr() == ws.final_lse.data_ptr()
    assert mla_kernel.LAST_DECODE_PLAN["direct_output"] is True
    assert mla_kernel.LAST_DECODE_PLAN["num_splits"] == 1

    if return_lse:
        ret_out, ret_lse = result
        assert ret_out is out
        expected_lse = 2.0 * math.log(2.0) if lse_scale == "natural" else 2.0
        torch.testing.assert_close(ret_lse, torch.full_like(ret_lse, expected_lse))
    else:
        assert result is out


def test_num_splits_gt_one_retains_merge_and_lse_programs(monkeypatch) -> None:
    import b12x.attention._shared.mla.api as mla_api
    import b12x.attention._shared.mla.merge as mla_merge

    rows, heads = 4, 16
    traits = _dsv41_bf16_traits()
    q = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    swa_cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    swa_indices = torch.zeros((rows, 128), dtype=torch.int32)
    swa_lengths = torch.full((rows,), 128, dtype=torch.int32)
    ws = _dummy_workspace(
        rows=rows, heads=heads, max_chunks_per_row=4, num_chunks_value=4
    )

    monkeypatch.setattr(
        mla_kernel, "_sparse_mla_decode_grid_flat_launch", lambda *a, **k: "grid-prog"
    )
    monkeypatch.setattr(
        mla_merge, "run_sparse_mla_split_decode_merge", lambda *a, **k: "merge-prog"
    )

    def fake_lse(*args, resolved_programs=None, **kwargs):
        if resolved_programs is not None:
            resolved_programs.append("lse-prog")
        return ws.final_lse[:rows, :heads]

    monkeypatch.setattr(mla_api, "_final_lse_from_split_workspace", fake_lse)

    prepared = mla_kernel.prepare_unified_decode_launch(
        q_all=q,
        q_alignment=16,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        workspace=ws,
        swa_page_size=64,
        sm_count=188,
        return_lse=True,
        v41_compute_mode="bf16",
        traits_override=traits,
    )
    assert prepared.num_splits == 2
    assert prepared.direct_output is False

    compiled = mla_kernel.compile_unified_decode_launch(
        prepared=prepared,
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        workspace=ws,
        sm_scale=512**-0.5,
        swa_page_size=64,
        return_lse=True,
        lse_scale="base2",
        v41_compute_mode="bf16",
    )
    assert compiled.grid_programs == ("grid-prog",)
    assert compiled.merge_program == "merge-prog"
    assert compiled.lse_program == "lse-prog"


@pytest.mark.parametrize("cache_format", ["deepseek_v4", "deepseek_v41"])
def test_sm121_single_pass_decode_covers_deepseek_v41(cache_format: str) -> None:
    rows, heads = 16, 32
    q = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    swa = torch.empty((2, 64 * 528), dtype=torch.uint8)
    idx = torch.empty((8, 64 * 288), dtype=torch.uint8)
    plan = mla.plan(
        mla.Caps(
            device="cpu",
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=640,
            swa_width=128,
            indexed_width=512,
            swa_page_size=64,
            indexed_page_size=64,
            cache_format=cache_format,
            mode="decode",
        ),
        invocation=mla.invocation_from_tensors(
            q=q, swa_k_cache=swa, indexed_k_cache=idx, out=torch.empty_like(q)
        ),
    )

    assert _single_pass(plan.query, SM121) is True
    assert _single_pass(plan.query, SM120) is False

    cfg_121 = TUNING.configure(plan.query, device=SM121).default
    assert cfg_121.single_pass is True
    assert cfg_121.max_chunks_per_row == 1
    assert cfg_121.split_chunk_size == 1
    assert cfg_121.v41_heads_per_block == 16

    # mode="verify" is always single-pass on both SM120 and SM121.
    verify_query = replace(plan.query, mode="verify")
    assert _single_pass(verify_query, SM120) is True
    assert _single_pass(verify_query, SM121) is True


def test_decode_grid_custom_op_fake_dispatch_accepts_direct_output_and_sink() -> None:
    with FakeTensorMode():
        q_all = torch.empty((2, 16, 512), dtype=torch.bfloat16)
        cache = torch.empty((4, 64 * 528), dtype=torch.uint8)
        indices = torch.empty((2, 128), dtype=torch.int32)
        lengths = torch.empty((2,), dtype=torch.int32)
        direct_out = torch.empty((2, 16, 1, 512), dtype=torch.bfloat16)
        direct_lse = torch.empty((2, 16, 1), dtype=torch.float32)
        attn_sink = torch.empty((16,), dtype=torch.float32)

        torch.ops.b12x.sparse_mla_sm120_decode_grid(
            q_all,
            cache,
            indices,
            direct_out,
            direct_lse,
            lengths,
            cache,
            indices,
            lengths,
            0.1,
            1.0,
            ModelType.DSV41,
            ComputeMode.BF16,
            ScaleFormat.NVFP4_E4M3,
            False,
            64,
            128,
            0,
            2,
            1,
            2,
            64 * 528,
            64,
            64 * 288,
            1,
            16,
            0,
            False,
            True,
            False,
            False,
            16,
            attn_sink,
            True,
        )


def test_kernel_math_and_io_ast_contracts() -> None:
    """Verify #142 BF16 decode primitives, 304B stride, overlap IO, and FINAL_BF16 wiring."""
    q_vec_src = inspect.getsource(decode_math.s0_load_q_bf16_vec)
    assert "ld_global_nc_v4_u32" in q_vec_src
    assert "st_shared_v4_u32" in q_vec_src

    s1_src = inspect.getsource(decode_math.s1_qk_dsv41_source_decode)
    assert "544 if swa else 304" in s1_src
    assert "528 if swa else 288" in s1_src
    assert "ld_shared_v4_u32" in s1_src
    assert "vote_all_sync" in s1_src

    s6_src = inspect.getsource(decode_math.s6_xv_dsv41_source_decode)
    assert "544 if swa else 304" in s6_src
    assert "valid_mask" in s6_src
    assert "vote_all_sync" in s6_src

    io_src = inspect.getsource(mla_io.io_issue_gather)
    assert "overlap_dsv41_fp8" in io_src
    assert "528 if (dsv41_swa or kv_smem_stride >= 544) else 288" in io_src

    body_src = inspect.getsource(mla_kernel.UnifiedDecodeKernel._kernel_body)
    tree = ast.parse(textwrap.dedent(body_src))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "s0_load_q_bf16_vec",
        "s1_qk_dsv41_source_decode",
        "s6_xv_dsv41_source_decode",
        "s7_epilogue",
    } <= called_names
    assert "DSV41_INDEXED_SMEM_STRIDE" in body_src
    assert "EPILOGUE_FINAL_BF16" in body_src
