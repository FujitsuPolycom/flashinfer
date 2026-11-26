# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical, statistical, and CUDA-graph parity tests for zero-HBM lm_head vocab summary."""

from __future__ import annotations

import math

import pytest
import torch

from b12x.gemm import bf16_gemv
from b12x.preparation import PreparationSession, PreparedCall

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_GUMBEL_DOMAIN = 0x47554D42


def _fp64_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    inv_temperature: torch.Tensor | None = None,
    logit_bias: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
    vocab_offset: int = 0,
    probe_token_ids: torch.Tensor | None = None,
):
    """Exact FP64 reference matching the vocab-summary math contract."""
    m, n = x.shape[0], weight.shape[0]
    logits = x.double() @ weight.double().T
    if logit_bias is not None:
        logits = logits + logit_bias.double().unsqueeze(0)

    is_greedy = torch.zeros(m, dtype=torch.bool, device=x.device)
    if inv_temperature is not None:
        inv_t = inv_temperature.double()
        is_greedy = inv_t == 0.0
        scale = torch.where(is_greedy, torch.ones_like(inv_t), inv_t)
        logits = logits * scale.unsqueeze(1)

    allowed = torch.ones((m, n), dtype=torch.bool, device=x.device)
    if grammar_mask is not None:
        cols = torch.arange(n, device=x.device, dtype=torch.int64)
        mask_i32 = grammar_mask.view(torch.int32)
        words = mask_i32[:m, cols // 32].to(torch.int64) & 0xFFFFFFFF
        bits = (words >> (cols % 32)) & 1
        allowed = bits != 0

    masked_logits = torch.where(
        allowed, logits, torch.full_like(logits, float("-inf"))
    )
    row_valid = allowed.any(dim=1)

    global_max = torch.where(
        row_valid,
        masked_logits.max(dim=1).values,
        torch.full((m,), float("-inf"), dtype=torch.float64, device=x.device),
    )
    # torch.argmax breaks ties left-to-right (smallest index wins).
    argmax_local = masked_logits.argmax(dim=1).to(torch.int32)
    argmax_id = torch.where(
        row_valid,
        argmax_local + int(vocab_offset),
        torch.full((m,), -1, dtype=torch.int32, device=x.device),
    )
    stable_max = torch.where(row_valid, global_max, torch.zeros_like(global_max))
    exp_vals = torch.where(
        allowed,
        torch.exp(masked_logits - stable_max.unsqueeze(1)),
        torch.zeros_like(masked_logits),
    )
    global_sum_exp = torch.where(
        row_valid, exp_vals.sum(dim=1), torch.zeros_like(global_max)
    )
    global_lse = torch.where(
        row_valid,
        global_max + torch.log(global_sum_exp.clamp_min(1e-300)),
        torch.full((m,), float("-inf"), dtype=torch.float64, device=x.device),
    )

    probe_logits = None
    probe_prob = None
    if probe_token_ids is not None:
        local_p = probe_token_ids[:m] - int(vocab_offset)
        in_range = (local_p >= 0) & (local_p < n)
        safe_p = local_p.clamp(0, n - 1)
        gathered = masked_logits[torch.arange(m, device=x.device), safe_p]
        probe_logits = torch.where(
            in_range,
            gathered,
            torch.full((m,), float("-inf"), dtype=torch.float64, device=x.device),
        )
        p_fin = torch.isfinite(probe_logits) & row_valid
        soft_prob = torch.where(
            p_fin,
            torch.exp(probe_logits - global_lse),
            torch.zeros_like(probe_logits),
        )
        greedy_prob = torch.where(
            p_fin & (probe_token_ids[:m] == argmax_id.to(torch.int64)),
            torch.ones_like(probe_logits),
            torch.zeros_like(probe_logits),
        )
        probe_prob = torch.where(is_greedy, greedy_prob, soft_prob)

    return {
        "logits": masked_logits,
        "global_max": global_max,
        "global_sum_exp": global_sum_exp,
        "global_lse": global_lse,
        "argmax_id": argmax_id,
        "probe_logits": probe_logits,
        "probe_prob": probe_prob,
    }


def _pack_grammar_mask(allowed: torch.Tensor, *, dtype=torch.uint32) -> torch.Tensor:
    """Pack boolean [M, N] into [M, ceil(N / 32)] bitmask matching Hyperflare."""
    m, n = allowed.shape
    words = (n + 31) // 32
    padded = torch.zeros((m, words * 32), dtype=torch.int64, device=allowed.device)
    padded[:, :n] = allowed.to(torch.int64)
    bits = padded.view(m, words, 32)
    shifts = torch.arange(32, device=allowed.device, dtype=torch.int64)
    packed = (bits << shifts).sum(dim=-1)
    if dtype == torch.uint32:
        return packed.to(torch.uint32)
    # Cast unsigned 32-bit patterns into signed int32 view.
    return packed.to(torch.uint32).view(torch.int32)


def _reference_bf16_gemv_tc(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    with PreparationSession(device=x.device, autotune=False, compile_workers=1) as session:
        decl = bf16_gemv.plan(
            bf16_gemv.query_from_call(x, weight, out=out),
            override=bf16_gemv.GemvConfig(backend="tc", split_k=1),
        )
        session.prepare(
            (
                decl.request(
                    name="tc_ref",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(x, weight, out=out)
                    ),
                ),
            )
        )
        bf16_gemv.mm(x, weight, out=out, plan=decl)
    return out


def _reference_gumbel_noise_gpu(
    seed: torch.Tensor, offset: torch.Tensor, n: int, vocab_offset: int
) -> torch.Tensor:
    """Reference Philox-4x32-10 Gumbel noise matching Triton tl.philox exactly."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _ref_gumbel_kernel(
        Out, Seeds, Offsets, vocab_off, N, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        col = tile * BLOCK + tl.arange(0, BLOCK)
        mask = col < N
        gcol = vocab_off + col
        s = tl.load(Seeds + row)
        off = tl.load(Offsets + row).to(tl.uint64)
        bits, _, _, _ = tl.philox(
            s,
            gcol.to(tl.uint32),
            off.to(tl.uint32),
            (off >> 32).to(tl.uint32),
            tl.full((), 0x47554D42, tl.uint32),
        )
        u = (
            (bits >> 16).to(tl.float32) * 65536.0
            + (bits & 65535).to(tl.float32)
            + 0.5
        ) * 2.3283064365386963e-10
        u = tl.minimum(u, 0.9999999403953552)
        noise = -tl.log(-libdevice.log1p(-u))
        tl.store(Out + row * N + col, noise, mask=mask)

    m = seed.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=seed.device)
    block = 256
    _ref_gumbel_kernel[(m, (n + block - 1) // block)](
        out, seed, offset, int(vocab_offset), int(n), BLOCK=block
    )
    return out


@cuda_required
@pytest.mark.parametrize("m", [1, 4, 8, 16, 32, 64, 128])
def test_ds41_tp4_vocab_summary_parity(m):
    """DeepSeek-V4.1 TP4 lm_head [32320, 5120] across M = 1..128."""
    torch.manual_seed(1000 + m)
    n, k = 32320, 5120
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()

    res = bf16_gemv.bf16_gemv_vocab_summary(x, w)
    assert res.logits_out is None

    ref = _fp64_reference(x, w)
    assert torch.equal(res.argmax_id, ref["argmax_id"])
    torch.testing.assert_close(
        res.global_max.double(), ref["global_max"], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        res.global_lse.double(), ref["global_lse"], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        res.global_sum_exp.double(), ref["global_sum_exp"], rtol=2e-4, atol=2e-4
    )


@cuda_required
@pytest.mark.parametrize(
    "n,k,m",
    [
        (38720, 4096, 1),
        (38720, 4096, 8),
        (38720, 4096, 32),
        (62080, 5120, 1),
        (62080, 5120, 8),
        (62080, 5120, 32),
        (248320, 5120, 1),
        (248320, 5120, 8),
        (248320, 5120, 32),
    ],
)
def test_glm53_and_qwen38_vocab_summary_parity(n, k, m):
    """GLM-5.3 TP4 [38720, 4096] and Qwen3.8 TP4/TP1 [62080/248320, 5120]."""
    torch.manual_seed(2000 + m + (n % 997))
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()

    res = bf16_gemv.bf16_gemv_vocab_summary(x, w)
    assert res.logits_out is None

    ref = _fp64_reference(x, w)
    assert torch.equal(res.argmax_id, ref["argmax_id"])
    torch.testing.assert_close(
        res.global_max.double(), ref["global_max"], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        res.global_lse.double(), ref["global_lse"], rtol=2e-5, atol=2e-5
    )


@cuda_required
@pytest.mark.parametrize("m", [1, 8, 32])
def test_zero_ulp_exact_match_vs_tensor_core_gemv(m):
    """For M <= 32, fused dot products and write_logits match TensorCoreGemvKernel 0-ULP."""
    torch.manual_seed(31415 + m)
    n, k = 32320, 5120
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    tc_logits = _reference_bf16_gemv_tc(x, w)
    res = bf16_gemv.bf16_gemv_vocab_summary(
        x, w, write_logits=True, logits_dtype=torch.float32
    )
    # Bit-for-bit identical FP32 dot products (0-ULP)!
    assert torch.equal(
        res.logits_out.view(torch.int32), tc_logits.view(torch.int32)
    )
    assert torch.equal(res.argmax_id, tc_logits.argmax(dim=1).to(torch.int32))
    torch.testing.assert_close(
        res.global_max, tc_logits.max(dim=1).values, rtol=0, atol=0
    )
    ref_lse = torch.logsumexp(tc_logits.double(), dim=1).float()
    torch.testing.assert_close(res.global_lse, ref_lse, rtol=1e-6, atol=1e-6)


@cuda_required
@pytest.mark.parametrize("m,n,k", [(1, 100, 128), (7, 129, 160), (40, 1025, 160), (65, 193, 256)])
def test_non_multiple_tail_shapes_and_chunk_k32(m, n, k):
    """Unaligned N tails, chunk_k=32 (K=128/160), and M=65 large-M non-multiple."""
    torch.manual_seed(4000 + m + n)
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    res = bf16_gemv.bf16_gemv_vocab_summary(x, w)
    ref = _fp64_reference(x, w)
    assert torch.equal(res.argmax_id, ref["argmax_id"])
    torch.testing.assert_close(
        res.global_max.double(), ref["global_max"], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        res.global_lse.double(), ref["global_lse"], rtol=2e-5, atol=2e-5
    )


@cuda_required
@pytest.mark.parametrize("m", [8, 48])
@pytest.mark.parametrize("mask_dtype", [torch.uint32, torch.int32])
def test_temperature_bias_grammar_mask_probe_and_edge_cases(m, mask_dtype):
    """Covers inv_temperature (incl. greedy 0.0), logit_bias, grammar_mask edges, and probes."""
    torch.manual_seed(5000 + m)
    n, k = 4096, 512
    vocab_offset = 32320  # simulates TP4 rank 1
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    bias = torch.linspace(-0.5, 0.5, n, device="cuda", dtype=torch.float32)

    inv_temp = torch.full((m,), 1.0 / 0.7, device="cuda", dtype=torch.float32)
    inv_temp[0] = 0.0  # greedy row
    inv_temp[1] = 0.0  # greedy row with non-argmax probe
    inv_temp[2] = 1.0 / 1.3
    inv_temp[3] = 2.5

    allowed = torch.rand((m, n), device="cuda") > 0.35
    # Row 4: ALL masked -> -inf / 0.0 / -1 without NaNs!
    allowed[4, :] = False
    # Row 5: SINGLE unmasked token (in last partial tile)!
    allowed[5, :] = False
    single_tok = n - 3
    allowed[5, single_tok] = True
    # Row 6: tie-break left check -> make exact equal logits across multiple tiles!
    x[6].zero_()
    bias_tie = bias.clone()
    bias_tie.fill_(1.25)
    # Keep distinct bias for rows != 6; test tie-breaking explicitly below too.
    mask_packed = _pack_grammar_mask(allowed, dtype=mask_dtype)
    # Test strided view of grammar_mask (simulating slicing global mask on TP rank)
    padded_mask = torch.zeros(
        (m, mask_packed.shape[1] + 8), dtype=mask_dtype, device="cuda"
    )
    padded_mask[:, 4 : 4 + mask_packed.shape[1]] = mask_packed
    mask_view = padded_mask[:, 4 : 4 + mask_packed.shape[1]]

    # First get exact reference argmax for row 0 so we can probe it in greedy mode.
    pre_ref = _fp64_reference(
        x,
        w,
        inv_temperature=inv_temp,
        logit_bias=bias,
        grammar_mask=mask_view,
        vocab_offset=vocab_offset,
    )
    probe_ids = torch.full((m,), vocab_offset + 17, dtype=torch.int64, device="cuda")
    probe_ids[0] = pre_ref["argmax_id"][0].to(torch.int64)  # greedy HIT -> prob 1.0
    probe_ids[1] = (
        vocab_offset + ((int(pre_ref["argmax_id"][1].item()) - vocab_offset + 1) % n)
    )  # greedy MISS -> prob 0.0
    probe_ids[2] = vocab_offset - 1  # out of range (below)
    probe_ids[3] = vocab_offset + n  # out of range (above)
    probe_ids[4] = vocab_offset + 5  # all-masked row -> prob 0.0
    probe_ids[5] = vocab_offset + single_tok  # single-unmasked -> prob 1.0
    probe_ids[6] = -1  # disabled probe -> -inf / 0.0
    # Masked token in active row 7:
    masked_col = int((~allowed[7]).nonzero()[0].item())
    probe_ids[7] = vocab_offset + masked_col

    res = bf16_gemv.bf16_gemv_vocab_summary(
        x,
        w,
        inv_temperature=inv_temp,
        logit_bias=bias,
        grammar_mask=mask_view,
        vocab_offset=vocab_offset,
        probe_token_ids=probe_ids,
    )
    ref = _fp64_reference(
        x,
        w,
        inv_temperature=inv_temp,
        logit_bias=bias,
        grammar_mask=mask_view,
        vocab_offset=vocab_offset,
        probe_token_ids=probe_ids,
    )

    assert torch.equal(res.argmax_id, ref["argmax_id"])
    # Check all-masked row 4 has no NaNs and exact sentinels
    assert res.argmax_id[4].item() == -1
    assert math.isinf(res.global_max[4].item()) and res.global_max[4].item() < 0
    assert math.isinf(res.global_lse[4].item()) and res.global_lse[4].item() < 0
    assert res.global_sum_exp[4].item() == 0.0
    assert res.probe_prob[4].item() == 0.0
    # Check single-unmasked row 5
    assert res.argmax_id[5].item() == vocab_offset + single_tok
    assert res.global_sum_exp[5].item() == pytest.approx(1.0, abs=1e-6)
    assert res.probe_prob[5].item() == pytest.approx(1.0, abs=1e-6)
    # Check greedy hit/miss on rows 0, 1
    assert res.probe_prob[0].item() == 1.0
    assert res.probe_prob[1].item() == 0.0
    # Check out-of-range / masked probes
    for r in (2, 3, 6, 7):
        assert math.isinf(res.probe_logits[r].item()) and res.probe_logits[r].item() < 0
        assert res.probe_prob[r].item() == 0.0

    # Check all finite rows match FP64 oracle closely
    fin = torch.isfinite(ref["global_max"])
    torch.testing.assert_close(
        res.global_max[fin].double(), ref["global_max"][fin], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        res.global_lse[fin].double(), ref["global_lse"][fin], rtol=2e-5, atol=2e-5
    )
    pfin = torch.isfinite(ref["probe_logits"])
    torch.testing.assert_close(
        res.probe_logits[pfin].double(),
        ref["probe_logits"][pfin],
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        res.probe_prob.double(), ref["probe_prob"], rtol=2e-5, atol=2e-5
    )


@cuda_required
@pytest.mark.parametrize("m", [4, 40])
def test_tie_break_left_exact(m):
    """Equal logits across columns and tiles must always pick the smallest global token ID."""
    n, k = 1024, 128
    vocab_offset = 6400
    x = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)
    allowed = torch.zeros((m, n), dtype=torch.bool, device="cuda")
    expected_ids = []
    for r in range(m):
        first_col = (r * 37 + 13) % (n - 200)
        allowed[r, first_col] = True
        allowed[r, first_col + 1] = True
        allowed[r, first_col + 17] = True
        allowed[r, first_col + 129] = True
        expected_ids.append(vocab_offset + first_col)
    mask = _pack_grammar_mask(allowed)
    res = bf16_gemv.bf16_gemv_vocab_summary(
        x, w, logit_bias=bias, grammar_mask=mask, vocab_offset=vocab_offset
    )
    assert res.argmax_id.cpu().tolist() == expected_ids


@cuda_required
@pytest.mark.parametrize("m", [8, 48])
def test_philox_gumbel_exact_parity_with_triton(m):
    """Philox-4x32-10 Gumbel-max matches Hyperflare Triton tl.philox bit-identically."""
    torch.manual_seed(6000 + m)
    n, k = 8192, 512
    vocab_offset = 64640
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    inv_temp = torch.full((m,), 1.25, device="cuda", dtype=torch.float32)
    inv_temp[0] = 0.0  # greedy row -> gumbel_id must equal argmax_id!
    seeds = torch.arange(m, device="cuda", dtype=torch.int64) * 0x9E3779B97F4A7C15 + 42
    offsets = torch.arange(m, device="cuda", dtype=torch.int64) * 4 + 17

    res = bf16_gemv.bf16_gemv_vocab_summary(
        x,
        w,
        inv_temperature=inv_temp,
        vocab_offset=vocab_offset,
        rng_seed=seeds,
        rng_offset=offsets,
        write_logits=True,
        logits_dtype=torch.float32,
    )

    # Greedy row 0 must have zero Gumbel noise -> exact argmax!
    assert res.gumbel_id[0].item() == res.argmax_id[0].item()
    assert res.gumbel_max[0].item() == res.global_max[0].item()

    noise = _reference_gumbel_noise_gpu(seeds, offsets, n, vocab_offset)
    noise[0].zero_()
    scores = res.logits_out + noise
    ref_gumbel_max = scores.max(dim=1).values
    ref_gumbel_id = scores.argmax(dim=1).to(torch.int32) + vocab_offset

    assert torch.equal(res.gumbel_id, ref_gumbel_id)
    torch.testing.assert_close(res.gumbel_max, ref_gumbel_max, rtol=1e-6, atol=1e-6)


@cuda_required
def test_philox_gumbel_100k_statistical_distribution():
    """100,000 independent Philox Gumbel-max draws match exact softmax probabilities."""
    n, k = 32, 64
    batch = 100
    iters = 1000  # 100,000 draws total
    torch.manual_seed(7777)
    x = torch.zeros((batch, k), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    # Non-trivial peaked + long-tail distribution over 32 tokens
    bias = torch.linspace(-2.0, 2.2, n, device="cuda", dtype=torch.float32)
    expected_probs = torch.softmax(bias.double(), dim=0).cpu()

    buffers = bf16_gemv.allocate_vocab_summary_outputs(
        batch, n, "cuda", use_gumbel=True
    )
    seeds = torch.full((batch,), 0x517CC1B727220A95, device="cuda", dtype=torch.int64)
    base_offsets = torch.arange(batch, device="cuda", dtype=torch.int64)
    counts = torch.zeros(n, dtype=torch.int64, device="cuda")

    for step in range(iters):
        offsets = base_offsets + step * batch
        res = bf16_gemv.bf16_gemv_vocab_summary(
            x,
            w,
            logit_bias=bias,
            rng_seed=seeds,
            rng_offset=offsets,
            buffers=buffers,
        )
        counts.scatter_add_(
            0, res.gumbel_id.to(torch.int64), torch.ones(batch, dtype=torch.int64, device="cuda")
        )

    empirical = counts.double().cpu() / (batch * iters)
    # Pearson chi-square statistic across 32 bins (df=31; 99.9% critical value ~61.1)
    expected_counts = expected_probs * (batch * iters)
    chi2 = (((counts.double().cpu() - expected_counts) ** 2) / expected_counts).sum().item()
    max_abs = (empirical - expected_probs).abs().max().item()
    assert max_abs < 0.005, f"max probability deviation too high: {max_abs}"
    assert chi2 < 65.0, f"chi-square goodness-of-fit failed: chi2={chi2:.2f}"


@cuda_required
@pytest.mark.parametrize("m", [8, 64])
def test_cuda_graph_zero_alloc_capture_and_live_mutation(m):
    """Zero-allocation CUDA-graph capture and replay with live input mutations."""
    torch.manual_seed(8000 + m)
    n, k = 32320, 5120
    vocab_offset = 32320
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()
    inv_temp = torch.full((m,), 1.0 / 0.8, device="cuda", dtype=torch.float32)
    allowed = torch.rand((m, n), device="cuda") > 0.2
    mask = _pack_grammar_mask(allowed, dtype=torch.uint32)
    probe_ids = torch.randint(
        vocab_offset, vocab_offset + n, (m,), device="cuda", dtype=torch.int64
    )
    probe_ids[0] = vocab_offset - 5  # out of range
    seeds = torch.arange(m, device="cuda", dtype=torch.int64) + 12345
    offsets = torch.arange(m, device="cuda", dtype=torch.int64) * 4

    buffers = bf16_gemv.allocate_vocab_summary_outputs(
        m, n, "cuda", use_gumbel=True, has_probe=True, write_logits=False
    )

    def run_once():
        return bf16_gemv.bf16_gemv_vocab_summary(
            x,
            w,
            inv_temperature=inv_temp,
            grammar_mask=mask,
            vocab_offset=vocab_offset,
            probe_token_ids=probe_ids,
            rng_seed=seeds,
            rng_offset=offsets,
            buffers=buffers,
        )

    # Warmup compile before capture
    run_once()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run_once()
        assert captured.logits_out is None

        for step in range(3):
            x.neg_()
            inv_temp.fill_(0.0 if step == 1 else (1.0 / (0.6 + 0.2 * step)))
            # Flip probe_ids between in-range and out-of-range live!
            probe_ids.copy_(
                torch.randint(
                    vocab_offset, vocab_offset + n, (m,), device="cuda", dtype=torch.int64
                )
            )
            if step == 2:
                probe_ids[1] = -1
            offsets.add_(4)

            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated

            ref = _fp64_reference(
                x,
                w,
                inv_temperature=inv_temp,
                grammar_mask=mask,
                vocab_offset=vocab_offset,
                probe_token_ids=probe_ids,
            )
            assert torch.equal(captured.argmax_id, ref["argmax_id"])
            torch.testing.assert_close(
                captured.global_max.double(), ref["global_max"], rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                captured.global_lse.double(), ref["global_lse"], rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                captured.probe_prob.double(), ref["probe_prob"], rtol=2e-5, atol=2e-5
            )
            if step == 1:
                assert torch.equal(captured.gumbel_id, captured.argmax_id)
    finally:
        graph.reset()


@cuda_required
def test_validation_and_negative_sensitivity():
    """Invalid inputs fail fast and parity checks reject deliberately wrong results."""
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rng_seed and rng_offset"):
        bf16_gemv.bf16_gemv_vocab_summary(
            x, w, rng_seed=torch.zeros(4, dtype=torch.int64, device="cuda")
        )
    res = bf16_gemv.bf16_gemv_vocab_summary(x, w)
    ref = _fp64_reference(x, w)
    assert torch.equal(res.argmax_id, ref["argmax_id"])
    # Negative sensitivity: perturbing global_lse must fail our tolerance check
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            res.global_lse.double() + 1e-3, ref["global_lse"], rtol=2e-5, atol=2e-5
        )
