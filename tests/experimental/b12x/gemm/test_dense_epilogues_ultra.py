"""0-ULP parity and CUDA-graph tests for Stage 2 Ultra dense MXFP8 epilogues."""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from b12x._lib.dense_gemm.execution import dense_gemm
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm import wo_projection as wo
from b12x.gemm._shared.wo_mxfp8 import (
    empty_mxfp8_rows_for_dense_gemm,
    quantize_mxfp8_rows_torch,
    quantize_wo_a_input_mxfp8,
    quantize_wo_b_input_mxfp8,
    quantize_wo_projection_weights_mxfp8_torch,
    wo_a_dense_gemm_mxfp8,
    wo_b_dense_gemm_mxfp8,
)
from b12x.gemm.blockscaled._linear import pack_mxfp8_linear_weight
from b12x.norm import hyperconnection
from b12x.preparation import PreparationSession, PreparedCall
from b12x.quantization import mxfp8
from b12x.testing.reference.helpers import require_b12x


ALL_M = (1, 8, 16, 32, 64, 128, 512, 1024, 2048)


def _reference_swiglu_bf16(
    gate_up: torch.Tensor, *, limit: float = 7.0, round_silu: bool = False
) -> torch.Tensor:
    rows, total = gate_up.shape
    width = total // 2
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=gate_up.device)
    plan = hyperconnection.plan(
        hyperconnection.Caps(device=gate_up.device, max_tokens=rows, hidden_size=width),
        invocation={
            "operation": "swiglu",
            "limit": limit,
            "round_silu": round_silu,
        },
    )
    with PreparationSession(device=gate_up.device, autotune=False) as session:
        session.prepare(
            (
                plan.request(
                    name="ref.swiglu",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.launch(gate_up, gate_up, out)
                    ),
                ),
            )
        )
        hyperconnection.run_swiglu(
            gate_up, limit=limit, round_silu=round_silu, out=out, plan=plan
        )
    return out


@triton.jit
def _triton_rotate_grouped_ref(
    Values,
    Angles,
    ROW_STRIDE,
    HEAD_STRIDE,
    OFFSET,
    HEADS,
    SIGN: tl.constexpr,
    SPLIT_ANGLES: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
):
    """Exact rotary.py _rotate_grouped kernel."""
    row, h_block = tl.program_id(0), tl.program_id(1)
    pair = tl.arange(0, 32)
    if SPLIT_ANGLES:
        cos = tl.load(Angles + row * 64 + pair)
        sin = tl.load(Angles + row * 64 + 32 + pair) * SIGN
    else:
        cos = tl.load(Angles + row * 64 + 2 * pair)
        sin = tl.load(Angles + row * 64 + 2 * pair + 1) * SIGN
    row_base = Values + row * ROW_STRIDE + OFFSET + 2 * pair
    h0 = h_block * HEADS_PER_CTA
    for hi in tl.static_range(HEADS_PER_CTA):
        h = h0 + hi
        if h < HEADS:
            base = row_base + h * HEAD_STRIDE
            a = tl.load(base).to(tl.float32)
            b = tl.load(base + 1).to(tl.float32)
            tl.store(base, (a * cos - b * sin).to(Values.dtype.element_ty))
            tl.store(base + 1, (a * sin + b * cos).to(Values.dtype.element_ty))


def _reference_rotary_tail64(
    values: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    head_dim: int,
    split: bool = False,
    inverse: bool = False,
) -> torch.Tensor:
    """Exact rotary.py Triton _rotate_grouped reference."""
    m, n = values.shape
    heads = n // head_dim
    out = values.clone().view(m, heads, head_dim)
    per_cta = 8 if heads % 8 == 0 else 4
    _triton_rotate_grouped_ref[(m, triton.cdiv(heads, per_cta))](
        out,
        cos_sin,
        out.stride(0),
        out.stride(1),
        head_dim - 64,
        heads,
        -1 if inverse else 1,
        split,
        per_cta,
        num_warps=1,
    )
    return out.view(m, n)


def _make_mxfp8_weight(n: int, k: int, *, seed: int, k_alignment: int = 128):
    torch.manual_seed(seed)
    w_bf16 = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    w_q = quantize_mxfp8_rows_torch(w_bf16)
    return pack_mxfp8_linear_weight(
        w_q.values,
        w_q.scale_rows[0].view(torch.uint8),
        k_alignment=k_alignment,
    )


@pytest.mark.parametrize("m", ALL_M)
@pytest.mark.parametrize("groups", (2, 8))
def test_wo_a_quantize_c_epilogue_zero_ulp_parity(m: int, groups: int) -> None:
    """Fused wo_a BF16-rounded MXFP8 epilogue matches wo_a -> quantize_b at 0-ULP."""
    require_b12x()
    torch.manual_seed(1000 + m * 17 + groups)
    group_width = 512
    rank = 256
    hidden = 512

    source_tgd = (
        torch.randn((m, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4.0
    ).contiguous()
    # Include an all-zero 32-element block to exercise the max_abs == 0 scale=127 path.
    source_tgd[0, 0, :32] = 0
    wo_a_bf16 = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / 8.0
    ).contiguous()
    wo_b_bf16 = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / 8.0
    ).contiguous()
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_bf16, wo_b_bf16)

    x_q = quantize_wo_a_input_mxfp8(source_tgd)

    # Baseline: wo_a -> BF16 tmp -> standalone quantize_b -> wo_b.
    tmp_ref = wo_a_dense_gemm_mxfp8(x_q, weights.wo_a, expected_m=m)
    tmp_q_ref = quantize_wo_b_input_mxfp8(tmp_ref)
    out_ref = wo_b_dense_gemm_mxfp8(tmp_q_ref, weights.wo_b, expected_m=m)[:, :, 0]

    # Treatment: wo_a with fused BF16-rounded quantize_c epilogue -> wo_b (0 quantize_b).
    tmp_q_fused = empty_mxfp8_rows_for_dense_gemm(
        m, groups * rank, num_groups=1, device="cuda"
    )
    wo_a_dense_gemm_mxfp8(
        x_q, weights.wo_a, quantized_out=tmp_q_fused, expected_m=m
    )
    out_fused = wo_b_dense_gemm_mxfp8(tmp_q_fused, weights.wo_b, expected_m=m)[:, :, 0]
    torch.cuda.synchronize()

    assert torch.equal(tmp_q_fused.values.view(torch.uint8), tmp_q_ref.values.view(torch.uint8))
    assert torch.equal(
        tmp_q_fused.scale_rows.view(torch.uint8), tmp_q_ref.scale_rows.view(torch.uint8)
    )
    assert torch.equal(out_fused, out_ref)


@pytest.mark.parametrize("m", (16, 64, 128, 1024))
def test_wo_projection_plan_uses_quantize_c_and_captures_cuda_graph(m: int) -> None:
    """Prepared wo.plan uses fused wo_a quantize_c for M > 8 with 0-ULP parity and CUDA graph replay."""
    require_b12x()
    torch.manual_seed(2026 + m)
    groups, group_width, rank, hidden = 2, 512, 256, 512
    source_tgd = (
        torch.randn((m, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4.0
    ).contiguous()
    wo_a_bf16 = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / 8.0
    ).contiguous()
    wo_b_bf16 = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / 8.0
    ).contiguous()
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_bf16, wo_b_bf16)

    # Standalone baseline oracle (quantize_a -> ordinary wo_a -> quantize_b -> wo_b).
    x_q = quantize_wo_a_input_mxfp8(source_tgd)
    tmp_ref = wo_a_dense_gemm_mxfp8(x_q, weights.wo_a, expected_m=m)
    tmp_q_ref = quantize_wo_b_input_mxfp8(tmp_ref)
    out_ref = wo_b_dense_gemm_mxfp8(tmp_q_ref, weights.wo_b, expected_m=m)[:, :, 0]

    plan = wo.plan(
        wo.Caps(
            device=source_tgd.device,
            max_tokens=m,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
            dtype=torch.bfloat16,
        ),
    )
    with PreparationSession(device=source_tgd.device, autotune=False) as session:
        def prepare(state):
            assert state.ordinary_a.lowering.quantize_c is True
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=source_tgd.device)
                for spec in state._scratch_state.scratch_specs()
            )
            binding = state.bind(
                scratch=scratch, source_tgd=source_tgd, weights=weights
            )
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        session.prepare((plan.request(name="wo", prepare_call=prepare),))
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=source_tgd.device)
            for spec in plan.scratch_specs()
        )
        out_buf = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda")
        binding = wo.bind(
            plan, scratch=scratch, source_tgd=source_tgd, weights=weights, out=out_buf
        )
        eager = wo.run(binding=binding, plan=plan).clone()
        torch.cuda.synchronize()
        assert torch.equal(eager, out_ref)

        # Capture and replay in CUDA graph with new input values.
        replacement = (torch.randn_like(source_tgd) / 4.0).contiguous()
        x_q2 = quantize_wo_a_input_mxfp8(replacement)
        out_ref2 = wo_b_dense_gemm_mxfp8(
            quantize_wo_b_input_mxfp8(
                wo_a_dense_gemm_mxfp8(x_q2, weights.wo_a, expected_m=m)
            ),
            weights.wo_b,
            expected_m=m,
        )[:, :, 0]

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wo.run(binding=binding, plan=plan)
        source_tgd.copy_(replacement)
        out_buf.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_buf, out_ref2)
        graph.reset()


@pytest.mark.parametrize("m", ALL_M)
@pytest.mark.parametrize("physical_k", (576, 640))
@pytest.mark.parametrize("min_amax", (0.0, 1e-4))
def test_swiglu_mxfp8_quant_zero_ulp_parity(
    m: int, physical_k: int, min_amax: float
) -> None:
    """Fused swiglu_mxfp8_quant matches hyperconnection.run_swiglu + quantize_mxfp8_rows_cute at 0-ULP."""
    require_b12x()
    torch.manual_seed(3000 + m * 13 + physical_k)
    width = 576
    gate_up = (torch.randn((m, 2 * width), device="cuda", dtype=torch.bfloat16) * 3.0).contiguous()
    # Include extreme values (> limit=7.0) and an all-zero 32-element block.
    gate_up[0, :32] = 0
    gate_up[0, width : width + 32] = 0
    gate_up[0, 32:64] = 12.0
    gate_up[0, width + 32 : width + 64] = -12.0

    # Baseline: standalone SwiGLU -> BF16 -> standalone MXFP8 quantize.
    inter = _reference_swiglu_bf16(gate_up, limit=7.0, round_silu=False)
    k_align = 64 if physical_k == 576 else 128
    val_shape, row_shape, mma_shape = mxfp8.dense_rows_storage_shapes(
        m, physical_k, k_alignment=k_align
    )
    ref_v = torch.empty(val_shape, dtype=torch.float8_e4m3fn, device="cuda")
    ref_sr = torch.empty(row_shape, dtype=torch.uint8, device="cuda")
    ref_sm = torch.full(mma_shape, 127, dtype=torch.uint8, device="cuda")
    quantize_mxfp8_rows_cute(
        inter,
        ref_v,
        ref_sr.view(m, -1),
        ref_sm.view(-1),
        expected_m=m,
        min_amax=min_amax,
        physical_k=physical_k,
    )

    # Fused SwiGLU -> BF16 round -> MXFP8 quantize.
    fused_v = torch.empty(val_shape, dtype=torch.float8_e4m3fn, device="cuda")
    fused_sr = torch.empty(row_shape, dtype=torch.uint8, device="cuda")
    fused_sm = torch.full(mma_shape, 127, dtype=torch.uint8, device="cuda")
    mxfp8.swiglu_quantize_rows(
        gate_up,
        fused_v,
        fused_sr.view(m, -1),
        fused_sm.view(-1),
        limit=7.0,
        round_silu=False,
        expected_m=m,
        min_amax=min_amax,
    )
    torch.cuda.synchronize()

    assert torch.equal(fused_v.view(torch.uint8), ref_v.view(torch.uint8))
    assert torch.equal(fused_sr, ref_sr)
    assert torch.equal(fused_sm, ref_sm)


@pytest.mark.parametrize("m", ALL_M)
@pytest.mark.parametrize("inplace_residual", (False, True))
def test_unpadded_k576_and_residual_addend_zero_ulp_parity(
    m: int, inplace_residual: bool
) -> None:
    """Unpadded K=576 MXFP8 GEMM + optional BF16 residual addend matches padded K=640 + add at 0-ULP."""
    require_b12x()
    torch.manual_seed(4000 + m * 19 + int(inplace_residual))
    n, k = 5120, 576
    gate_up = (torch.randn((m, 2 * k), device="cuda", dtype=torch.bfloat16) * 2.0).contiguous()
    routed = (torch.randn((m, n), device="cuda", dtype=torch.bfloat16) / 2.0).contiguous()

    # Pack exact same weight with default k_alignment=128 (padded K=640) and k_alignment=64 (unpadded K=576).
    w_bf16 = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    w_q = quantize_mxfp8_rows_torch(w_bf16)
    w_pad640 = pack_mxfp8_linear_weight(
        w_q.values, w_q.scale_rows[0].view(torch.uint8), k_alignment=128
    )
    w_unpad576 = pack_mxfp8_linear_weight(
        w_q.values, w_q.scale_rows[0].view(torch.uint8), k_alignment=64
    )
    assert w_pad640.padded_in_features == 640
    assert w_unpad576.padded_in_features == 576

    # Baseline: SwiGLU -> padded K=640 quant -> padded K=640 dense_gemm -> + routed.
    inter = _reference_swiglu_bf16(gate_up, limit=7.0, round_silu=False)
    v640, sr640, sm640 = (
        torch.empty(s, dtype=d, device="cuda")
        for s, d in zip(
            mxfp8.dense_rows_storage_shapes(m, 640, k_alignment=128),
            (torch.float8_e4m3fn, torch.uint8, torch.uint8),
        )
    )
    sm640.fill_(127)
    quantize_mxfp8_rows_cute(
        inter, v640, sr640.view(m, -1), sm640.view(-1), expected_m=m, physical_k=640
    )
    rows640 = mxfp8.dense_rows_from_bases(v640, sr640, sm640, k_alignment=128)
    shared_ref = dense_gemm(
        (rows640.values.unsqueeze(-1), rows640.scale_mma),
        (w_pad640.weight.values.unsqueeze(-1), w_pad640.weight.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
        _split_k_slices_override=1 if m <= 16 else None,
    )[:, :, 0]
    out_ref = (shared_ref.float() + routed.float()).to(torch.bfloat16)

    # Treatment: fused SwiGLU+quant (unpadded K=576) -> unpadded K=576 dense_gemm + fused residual epilogue!
    v576, sr576, sm576 = (
        torch.empty(s, dtype=d, device="cuda")
        for s, d in zip(
            mxfp8.dense_rows_storage_shapes(m, 576, k_alignment=64),
            (torch.float8_e4m3fn, torch.uint8, torch.uint8),
        )
    )
    sm576.fill_(127)
    mxfp8.swiglu_quantize_rows(
        gate_up,
        v576,
        sr576.view(m, -1),
        sm576.view(-1),
        limit=7.0,
        round_silu=False,
        expected_m=m,
    )
    rows576 = mxfp8.dense_rows_from_bases(v576, sr576, sm576, k_alignment=64)

    # First verify plain unpadded K=576 matches padded K=640 at 0-ULP!
    shared_576 = dense_gemm(
        (rows576.values.unsqueeze(-1), rows576.scale_mma),
        (w_unpad576.weight.values.unsqueeze(-1), w_unpad576.weight.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
    )[:, :, 0]

    # Now verify fused residual addend (both out-of-place and in-place out=routed)!
    res_buf = routed.clone()
    out_buf = res_buf if inplace_residual else torch.empty_like(routed)
    dense_gemm(
        (rows576.values.unsqueeze(-1), rows576.scale_mma),
        (w_unpad576.weight.values.unsqueeze(-1), w_unpad576.weight.scale_mma),
        out=out_buf.unsqueeze(-1),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
        residual=res_buf.unsqueeze(-1),
    )
    torch.cuda.synchronize()

    assert torch.equal(shared_576, shared_ref)
    assert torch.equal(out_buf, out_ref)


@pytest.mark.parametrize("m", ALL_M)
@pytest.mark.parametrize(
    "n,head_dim,split",
    [
        (8192, 512, False),  # decode TP4 q_b (16 heads x 512), interleaved cos/sin
        (8192, 512, True),   # decode TP4 q_b (16 heads x 512), split cos/sin
        (4096, 128, False),  # index_q (32 heads x 128), interleaved cos/sin
    ],
)
def test_q_rope_epilogue_zero_ulp_parity(
    m: int, n: int, head_dim: int, split: bool
) -> None:
    """Optional tail-64 Q-RoPE epilogue on DenseGemmKernel matches dense_gemm + rotary at 0-ULP."""
    require_b12x()
    torch.manual_seed(5000 + m * 23 + head_dim + int(split))
    k = 1280
    x_bf16 = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 4.0).contiguous()
    x_q = quantize_mxfp8_rows_torch(x_bf16)
    w_packed = _make_mxfp8_weight(n, k, seed=6000 + head_dim)

    angles = torch.randn((m, 32), device="cuda", dtype=torch.float32) * 3.0
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if split:
        cos_sin = torch.cat([cos, sin], dim=-1).contiguous()
    else:
        cos_sin = torch.stack([cos, sin], dim=-1).contiguous()

    # Baseline: plain dense_gemm -> standalone rotary on trailing 64 dims.
    raw = dense_gemm(
        (x_q.values.unsqueeze(-1), x_q.scale_mma),
        (w_packed.weight.values.unsqueeze(-1), w_packed.weight.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
        _split_k_slices_override=1 if m <= 16 else None,
    )[:, :, 0]
    expected = _reference_rotary_tail64(raw, cos_sin, head_dim=head_dim, split=split)

    # Treatment: fused Q-RoPE epilogue inside DenseGemmKernel.
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    dense_gemm(
        (x_q.values.unsqueeze(-1), x_q.scale_mma),
        (w_packed.weight.values.unsqueeze(-1), w_packed.weight.scale_mma),
        out=actual.unsqueeze(-1),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
        _split_k_slices_override=1 if m <= 16 else None,
        rope_cos_sin=cos_sin,
        rope_head_dim=head_dim,
        rope_dim=64,
        rope_split=split,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_ultra_epilogues_cuda_graph_capture_and_replay() -> None:
    """Zero-allocation CUDA graph capture and replay for swiglu_quant + K=576 residual + Q-RoPE."""
    require_b12x()
    torch.manual_seed(7777)
    m = 128

    # 1. shared_mlp: swiglu_mxfp8_quant -> unpadded K=576 down + residual.
    gate_up = (torch.randn((m, 1152), device="cuda", dtype=torch.bfloat16) * 2.0).contiguous()
    routed = (torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16) / 2.0).contiguous()
    w_down = _make_mxfp8_weight(5120, 576, seed=7778, k_alignment=64)
    v576, sr576, sm576 = (
        torch.empty(s, dtype=d, device="cuda")
        for s, d in zip(
            mxfp8.dense_rows_storage_shapes(m, 576, k_alignment=64),
            (torch.float8_e4m3fn, torch.uint8, torch.uint8),
        )
    )
    sm576.fill_(127)
    rows576 = mxfp8.dense_rows_from_bases(v576, sr576, sm576, k_alignment=64)
    mlp_out = torch.empty((m, 5120), dtype=torch.bfloat16, device="cuda")

    # 2. q_b with fused tail-64 Q-RoPE.
    q_in = quantize_mxfp8_rows_torch(
        (torch.randn((m, 1280), device="cuda", dtype=torch.bfloat16) / 4.0).contiguous()
    )
    w_qb = _make_mxfp8_weight(8192, 1280, seed=7779)
    ang = torch.randn((m, 32), device="cuda", dtype=torch.float32) * 3.0
    cos_sin = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).contiguous()
    qb_out = torch.empty((m, 8192), dtype=torch.bfloat16, device="cuda")

    def run_all():
        mxfp8.swiglu_quantize_rows(
            gate_up, v576, sr576.view(m, -1), sm576.view(-1), limit=7.0, round_silu=False, expected_m=m
        )
        dense_gemm(
            (rows576.values.unsqueeze(-1), rows576.scale_mma),
            (w_down.weight.values.unsqueeze(-1), w_down.weight.scale_mma),
            out=mlp_out.unsqueeze(-1),
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            expected_m=m,
            residual=routed.unsqueeze(-1),
        )
        dense_gemm(
            (q_in.values.unsqueeze(-1), q_in.scale_mma),
            (w_qb.weight.values.unsqueeze(-1), w_qb.weight.scale_mma),
            out=qb_out.unsqueeze(-1),
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            expected_m=m,
            rope_cos_sin=cos_sin,
            rope_head_dim=512,
            rope_dim=64,
            rope_split=False,
        )

    # Warmup outside capture.
    run_all()
    torch.cuda.synchronize()
    ref_mlp = mlp_out.clone()
    ref_qb = qb_out.clone()

    # Capture + replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    mlp_out.zero_()
    qb_out.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(mlp_out, ref_mlp)
    assert torch.equal(qb_out, ref_qb)
    graph.reset()
