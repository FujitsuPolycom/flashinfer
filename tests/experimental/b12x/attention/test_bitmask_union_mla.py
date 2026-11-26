"""GPU parity, edge-case, and CUDA Graph tests for Bitmask-Union Sparse MLA Verification.

Covers:
  - Step A (``BitmaskUnionMergeKernel``): bit-exact parity vs CPU set-union across
    ``W in {1, 4, 8, 12, 14, 16, 24}``, ``B in {1, 4, 16}``, empty/disjoint/
    identical/duplicate/Zipf inputs, ``live_counts``, ``row_lengths``,
    ``uint8``/``uint32`` masks, and zero-alloc CUDA Graph replay.
  - Step B (``BitmaskUnionVerifyMLAKernel`` + ``BitmaskSplitMergeKernel``):
    parity against BOTH exact FP32 ``compressed_sparse_mla_reference`` AND
    existing ``UnifiedDecodeKernel`` across SWA-only and Dual (128+512 NVFP4)
    caches, ``H=16`` (TP4) and ``H=64`` (TP1), ``B in {1, 4, 8, 16}``,
    ``W in {8, 14, 16}``, partial ``live_counts``, empty-chunk NaN guard,
    NaN-poisoned page 0, ``indexed_page_table``, ``attn_sink``, ``return_lse``
    (natural/base2), split-K (``1, 2, 4, 8`` DSMEM & workspace), and 50-iter
    zero-alloc CUDA Graph capture/replay.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention import compressed_sparse_mla, sparse_mla
from b12x.attention._shared.mla.bitmask_union import (
    allocate_bitmask_union_workspace,
    bitmask_union_merge,
    bitmask_union_merge_reference,
    supports_bitmask_union_merge,
)
from b12x.attention._shared.mla.bitmask_verify import (
    allocate_bitmask_verify_workspace,
    bitmask_union_verify_mla,
    plan_bitmask_union_splits,
    supports_bitmask_union_verify,
)
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_deepseek_v41_cache_reference,
)
from b12x.preparation import PreparationSession, PreparedCall

SM_SCALE = 512.0 ** -0.5

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and supports_bitmask_union_merge(width=8, k_sel=512)
        and supports_bitmask_union_verify(num_heads=16, verify_width=8)
    ),
    reason="Requires SM120/SM121 CUDA GPU with CUTLASS DSL b12x support",
)


def _make_zipf_indices(
    b: int,
    w: int,
    k_sel: int,
    total_keys: int,
    overlap_frac: float = 0.70,
    seed: int = 42,
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    step = max(1, int(round(k_sel * (1.0 - overlap_frac) * 0.42)))
    out = torch.empty((b, w, k_sel), dtype=torch.int32)
    for bi in range(b):
        perm = torch.randperm(total_keys, generator=g).to(torch.int32)
        for wi in range(w):
            start = (wi * step) % max(1, total_keys - k_sel)
            row, _ = torch.sort(perm[start : start + k_sel])
            out[bi, wi] = row
    return out


def _assert_union_matches_ref(
    sel_gpu: torch.Tensor,
    *,
    live_counts: torch.Tensor | None = None,
    row_lengths: torch.Tensor | None = None,
    mask_dtype: torch.dtype | None = None,
) -> None:
    uk, mb, ul = bitmask_union_merge(
        sel_gpu,
        live_counts=live_counts,
        row_lengths=row_lengths,
        mask_dtype=mask_dtype,
    )
    torch.cuda.synchronize()
    ref_k, ref_m, ref_l = bitmask_union_merge_reference(
        sel_gpu,
        live_counts=live_counts,
        row_lengths=row_lengths,
        mask_dtype=mb.dtype,
    )
    assert torch.equal(ul.cpu(), ref_l.cpu()), f"union_len mismatch: {ul.cpu()} vs {ref_l.cpu()}"
    uk_c = uk.cpu()
    mb_c = mb.cpu()
    ref_kc = ref_k.cpu()
    ref_mc = ref_m.cpu()
    for bi in range(sel_gpu.shape[0]):
        n = int(ref_l[bi].item())
        assert torch.equal(uk_c[bi, :n], ref_kc[bi, :n]), f"keys mismatch at b={bi}"
        assert torch.equal(mb_c[bi, :n], ref_mc[bi, :n]), f"mask mismatch at b={bi}"


def _run_unified_baseline(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    swa_page_size: int,
    indexed_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    indexed_page_size: int = 64,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    dev = q.device
    rows, h, _ = q.shape
    has_idx = indexed_cache is not None
    out = torch.empty_like(q)
    with PreparationSession(device=dev, autotune=False, compile_workers=2) as sess:
        caps = compressed_sparse_mla.Caps(
            device=dev,
            num_q_heads=h,
            max_q_rows=rows,
            max_width=int(swa_indices.shape[1])
            + (int(indexed_indices.shape[1]) if has_idx else 0),
            swa_width=int(swa_indices.shape[1]),
            indexed_width=int(indexed_indices.shape[1]) if has_idx else 0,
            max_page_table_width=1,
            swa_page_size=swa_page_size,
            indexed_page_size=indexed_page_size,
            cache_format="deepseek_v41",
            compute_mode="bf16",
            mode="decode",
            use_cuda_graph=True,
        )
        inv = compressed_sparse_mla.invocation_from_tensors(
            q=q,
            swa_k_cache=swa_cache,
            indexed_k_cache=indexed_cache,
            attn_sink=attn_sink,
            out=out,
        )
        plan = compressed_sparse_mla.plan(caps, invocation=inv)
        bind_kw = dict(q=q, swa_indices=swa_indices, swa_lengths=swa_lengths)
        run_kw = dict(
            swa_k_cache=swa_cache,
            swa_page_size=swa_page_size,
            attn_sink=attn_sink,
            sm_scale=SM_SCALE,
        )
        if has_idx:
            bind_kw.update(
                indexed_indices=indexed_indices, indexed_lengths=indexed_lengths
            )
            run_kw.update(
                indexed_k_cache=indexed_cache, indexed_page_size=indexed_page_size
            )

        def _prep(st):
            (sp,) = st.scratch_specs()
            sc = torch.empty(sp.shape, dtype=sp.dtype, device=dev)
            to = torch.empty_like(out)
            bd = st.bind_for_preparation(scratch=sc, **bind_kw)
            return PreparedCall(
                run=lambda: st.run(bd, **run_kw, out=to), output=to, owners=(sc, bd)
            )

        sess.prepare((plan.request(name="base_ref", prepare_call=_prep),))
        (sp,) = plan.scratch_specs()
        sc = torch.empty(sp.shape, dtype=sp.dtype, device=dev)
        binding = compressed_sparse_mla.bind(plan, scratch=sc, **bind_kw)
        sess.freeze()
        compressed_sparse_mla.run(binding=binding, out=out, **run_kw)
        torch.cuda.synchronize()
    return out


# ════════════════════════════════════════════════════════════════════════════
# Public API & Step A (BitmaskUnionMergeKernel) Tests
# ════════════════════════════════════════════════════════════════════════════


def test_public_sparse_mla_exports():
    for name in (
        "BitmaskUnionMergeKernel",
        "BitmaskUnionVerifyMLAKernel",
        "BitmaskSplitMergeKernel",
        "BitmaskVerifySplitWorkspace",
        "allocate_bitmask_union_workspace",
        "allocate_bitmask_verify_workspace",
        "bitmask_union_merge",
        "bitmask_union_merge_reference",
        "bitmask_union_verify_mla",
        "plan_bitmask_union_splits",
        "supports_bitmask_union_merge",
        "supports_bitmask_union_verify",
    ):
        assert hasattr(sparse_mla, name), f"Missing sparse_mla export: {name}"


@pytest.mark.parametrize("w", [1, 4, 8, 12, 14, 16, 24])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_bitmask_union_merge_zipf_widths_and_batches(b: int, w: int):
    dev = torch.device("cuda:0")
    sel = _make_zipf_indices(b, w, 512, 8192, overlap_frac=0.70, seed=10 + b + w).to(
        dev
    )
    _assert_union_matches_ref(sel)


def test_bitmask_union_merge_edge_cases():
    dev = torch.device("cuda:0")
    k_sel = 512

    # 1. All empty (-1)
    _assert_union_matches_ref(torch.full((4, 8, k_sel), -1, dtype=torch.int32, device=dev))

    # 2. 100% overlap (identical rows -> U = 512, mask = 0xFF)
    base = torch.arange(0, k_sel * 3, 3, dtype=torch.int32, device=dev)
    identical = base.view(1, 1, k_sel).expand(4, 8, k_sel).contiguous()
    uk, mb, ul = bitmask_union_merge(identical)
    torch.cuda.synchronize()
    assert (ul == k_sel).all()
    assert (mb[:, :k_sel] == 0xFF).all()
    _assert_union_matches_ref(identical)

    # 3. 0% overlap (completely disjoint -> U = W * 512)
    disjoint = torch.arange(8 * k_sel, dtype=torch.int32, device=dev).view(1, 8, k_sel)
    uk, mb, ul = bitmask_union_merge(disjoint)
    torch.cuda.synchronize()
    assert ul.item() == 8 * k_sel
    _assert_union_matches_ref(disjoint)

    # 4. Intra-row duplicates + interior -1 gaps + INT32_MAX
    dup = torch.zeros((2, 8, 64), dtype=torch.int32)
    for bi in range(2):
        for wi in range(8):
            row = [0, 0, 5, 5, 5, -1, 12, 12, 2147483647] + [-1] * (64 - 9)
            dup[bi, wi] = torch.tensor(row, dtype=torch.int32)
    _assert_union_matches_ref(dup.to(dev))

    # 5. live_counts + row_lengths + uint32 mask for W <= 8
    sel = _make_zipf_indices(4, 8, k_sel, 4096, overlap_frac=0.65, seed=99).to(dev)
    lc = torch.tensor([0, 1, 5, 8], dtype=torch.int32, device=dev)
    rl = torch.randint(0, k_sel + 1, (4, 8), dtype=torch.int32, device=dev)
    _assert_union_matches_ref(sel, live_counts=lc, row_lengths=rl, mask_dtype=torch.uint32)


def test_bitmask_union_merge_cuda_graph_zero_alloc():
    dev = torch.device("cuda:0")
    b, w, k_sel = 8, 8, 512
    sel_a = _make_zipf_indices(b, w, k_sel, 8192, overlap_frac=0.70, seed=1).to(dev)
    sel_b = _make_zipf_indices(b, w, k_sel, 8192, overlap_frac=0.85, seed=2).to(dev)
    sel_buf = sel_a.clone()
    uk, mb, ul = allocate_bitmask_union_workspace(b, w, k_sel, device=dev)

    # Warmup compile
    bitmask_union_merge(sel_buf, unique_keys=uk, mask_bits=mb, union_len=ul)
    torch.cuda.synchronize()

    mem_before = torch.cuda.memory_allocated(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        bitmask_union_merge(sel_buf, unique_keys=uk, mask_bits=mb, union_len=ul)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(dev) == mem_before

    # Replay 50 times with mutated inputs and check bit-exactness
    for i in range(50):
        sel_buf.copy_(sel_a if (i & 1) == 0 else sel_b)
        g.replay()
    torch.cuda.synchronize()
    ref_k, ref_m, ref_l = bitmask_union_merge_reference(sel_b, mask_dtype=mb.dtype)
    assert torch.equal(ul.cpu(), ref_l.cpu())
    for bi in range(b):
        n = int(ref_l[bi].item())
        assert torch.equal(uk[bi, :n].cpu(), ref_k[bi, :n].cpu())
        assert torch.equal(mb[bi, :n].cpu(), ref_m[bi, :n].cpu())


# ════════════════════════════════════════════════════════════════════════════
# Step B (BitmaskUnionVerifyMLAKernel + BitmaskSplitMergeKernel) Tests
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "b,w,h,has_indexed,swa_contig,num_splits,split_mode,use_sink,use_pt,per_group",
    [
        # SWA-only: direct, DSMEM, workspace, paged w/ NaN poison
        (1, 8, 16, False, True, 1, "direct", True, False, False),
        (1, 8, 16, False, True, 4, "dsmem", True, False, False),
        (4, 8, 16, False, True, 4, "workspace", True, False, False),
        (2, 8, 16, False, False, 2, "dsmem", False, False, False),
        # Dual (128 + 512): TP4 H=16 across B=1,4,8,16, splits 1,2,4,5,8, DSMEM & workspace
        (1, 8, 16, True, True, 1, "direct", True, False, False),
        (1, 8, 16, True, True, 8, "dsmem", True, False, False),
        (4, 8, 16, True, False, 4, "dsmem", True, True, False),
        (4, 8, 16, True, True, 5, "workspace", True, True, True),
        (8, 8, 16, True, True, 8, "workspace", True, False, True),
        (16, 8, 16, True, True, 5, "workspace", True, False, True),
        # TP1 H=64
        (1, 8, 64, True, True, 1, "direct", True, False, True),
        (4, 8, 64, True, True, 4, "workspace", True, False, True),
        # Variable W=14 (Markov Tree) and W=16 (Wide Copy-Verify)
        (2, 14, 16, True, True, 4, "dsmem", True, False, False),
        (2, 16, 16, True, True, 2, "workspace", True, True, True),
    ],
)
def test_bitmask_union_verify_parity_vs_fp32_and_unified(
    b: int,
    w: int,
    h: int,
    has_indexed: bool,
    swa_contig: bool,
    num_splits: int,
    split_mode: str,
    use_sink: bool,
    use_pt: bool,
    per_group: bool,
):
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(500 + b * 13 + w * 7 + h + num_splits)
    page_size = 64
    rows = b * w

    q = (torch.randn((rows, h, 512), device=dev, generator=gen) * 0.25).bfloat16()
    sink = (
        torch.linspace(-0.4, 0.4, h, device=dev, dtype=torch.float32)
        if use_sink
        else None
    )

    n_swa = 128 + w
    swa_vals = (
        torch.randn((b, n_swa, 512), device=dev, generator=gen) * 0.25
    ).bfloat16()
    if swa_contig:
        swa_cache = pack_deepseek_v41_cache_reference(
            swa_vals.reshape(-1, 512), page_size=n_swa, cache_kind="swa"
        )
        swa_hist = torch.full((b,), 128, dtype=torch.int32, device=dev)
        swa_indices_ref = torch.empty((b, w, 128), dtype=torch.int32, device=dev)
        for bi in range(b):
            for wi in range(w):
                lo_k = wi + 1
                swa_indices_ref[bi, wi] = bi * n_swa + torch.arange(
                    lo_k, lo_k + 128, device=dev, dtype=torch.int32
                )
        swa_u_keys = swa_u_mask = swa_u_len = None
        swa_ref_ps = n_swa
    else:
        swa_packed = pack_deepseek_v41_cache_reference(
            swa_vals.reshape(-1, 512), page_size=page_size, cache_kind="swa"
        )
        swa_cache = torch.empty(
            (1 + swa_packed.shape[0], swa_packed.shape[1]), dtype=torch.uint8, device=dev
        )
        swa_cache[0].fill_(127)  # FP8 E4M3 NaN poison page 0!
        swa_cache[1:].copy_(swa_packed)
        swa_indices_ref = torch.empty((b, w, 128), dtype=torch.int32, device=dev)
        for bi in range(b):
            base_slot = page_size + bi * n_swa
            for wi in range(w):
                lo_k = wi + 1
                swa_indices_ref[bi, wi] = base_slot + torch.arange(
                    lo_k, lo_k + 128, device=dev, dtype=torch.int32
                )
        swa_u_keys, swa_u_mask, swa_u_len = bitmask_union_merge(swa_indices_ref)
        swa_hist = None
        swa_ref_ps = page_size
    swa_lens_ref = torch.full((rows,), 128, dtype=torch.int32, device=dev)

    idx_cache = idx_u_keys = idx_u_mask = idx_u_len = idx_pt = None
    idx_indices_phys = idx_lens_ref = None
    if has_indexed:
        n_idx_pages = 32
        idx_vals = (
            torch.randn((n_idx_pages * page_size, 512), device=dev, generator=gen)
            * 0.25
        ).bfloat16()
        idx_packed = pack_deepseek_v41_cache_reference(
            idx_vals, page_size=page_size, cache_kind="indexed"
        )
        idx_cache = torch.empty(
            (1 + n_idx_pages, idx_packed.shape[1]), dtype=torch.uint8, device=dev
        )
        idx_cache[0].fill_(127)  # NaN poison page 0!
        idx_cache[1:].copy_(idx_packed)

        logical_sel = _make_zipf_indices(
            b, w, 512, n_idx_pages * page_size, overlap_frac=0.70, seed=31 + b + w
        ).to(dev)
        if use_pt:
            idx_pt = (
                torch.arange(n_idx_pages, 0, -1, dtype=torch.int32, device=dev)
                .unsqueeze(0)
                .repeat(b, 1)
            )
            sel_for_union = (
                logical_sel.reshape(b * (w // 4), 4, 512)
                if (per_group and w % 4 == 0)
                else logical_sel
            )
            idx_u_keys, idx_u_mask, idx_u_len = bitmask_union_merge(sel_for_union)
            pg = (logical_sel // page_size).long()
            off = (logical_sel % page_size).long()
            phys_pg = idx_pt.long().unsqueeze(1).expand(b, w, -1).gather(2, pg)
            idx_indices_phys = (phys_pg * page_size + off).int()
        else:
            phys_sel = logical_sel + page_size
            sel_for_union = (
                phys_sel.reshape(b * (w // 4), 4, 512)
                if (per_group and w % 4 == 0)
                else phys_sel
            )
            idx_u_keys, idx_u_mask, idx_u_len = bitmask_union_merge(sel_for_union)
            idx_indices_phys = phys_sel
        idx_lens_ref = torch.full((rows,), 512, dtype=torch.int32, device=dev)

    ws = (
        allocate_bitmask_verify_workspace(rows, h, max_splits=max(1, num_splits), device=dev)
        if split_mode == "workspace"
        else None
    )
    out_act, lse_act = bitmask_union_verify_mla(
        q,
        swa_cache,
        verify_width=w,
        sm_scale=SM_SCALE,
        swa_unique_keys=swa_u_keys,
        swa_mask_bits=swa_u_mask,
        swa_union_len=swa_u_len,
        swa_history_lens=swa_hist,
        swa_contiguous=swa_contig,
        swa_page_size=swa_ref_ps,
        indexed_k_cache=idx_cache,
        indexed_unique_keys=idx_u_keys,
        indexed_mask_bits=idx_u_mask,
        indexed_union_len=idx_u_len,
        indexed_page_table=idx_pt,
        indexed_page_size=page_size,
        attn_sink=sink,
        return_lse=True,
        lse_scale="natural",
        num_splits=num_splits,
        split_mode=split_mode,
        split_workspace=ws,
    )
    torch.cuda.synchronize()
    assert not torch.isnan(out_act).any()
    assert not torch.isnan(lse_act).any()

    # 1. Compare against exact FP32 reference (on B <= 4 to keep test runtime snappy)
    if b <= 4:
        ref_out, ref_lse = compressed_sparse_mla_reference(
            q,
            swa_cache,
            swa_indices_ref.reshape(rows, 128),
            swa_lens_ref,
            sm_scale=SM_SCALE,
            attn_sink=sink,
            extra_k_cache=idx_cache,
            extra_indices=idx_indices_phys.reshape(rows, 512) if has_indexed else None,
            extra_topk_lengths=idx_lens_ref if has_indexed else None,
            swa_page_size=swa_ref_ps,
            extra_page_size=page_size if has_indexed else None,
            return_lse=True,
            cache_format="deepseek_v41",
        )
        max_abs_fp32 = (out_act.float() - ref_out.float()).abs().max().item()
        lse_err = (lse_act - ref_lse).abs().max().item()
        assert max_abs_fp32 < 2.5e-3, f"FP32 max_abs={max_abs_fp32}"
        assert lse_err < 1e-3, f"LSE err={lse_err}"

    # 2. Compare directly against existing UnifiedDecodeKernel baseline!
    if swa_contig:
        base_out = _run_unified_baseline(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices_ref.reshape(rows, 128),
            swa_lengths=swa_lens_ref,
            swa_page_size=swa_ref_ps,
            indexed_cache=idx_cache,
            indexed_indices=idx_indices_phys.reshape(rows, 512) if has_indexed else None,
            indexed_lengths=idx_lens_ref if has_indexed else None,
            indexed_page_size=page_size,
            attn_sink=sink,
        )
        max_abs_base = (out_act.float() - base_out.float()).abs().max().item()
        assert max_abs_base < 2.5e-3, f"UnifiedDecodeKernel max_abs={max_abs_base}"


def test_bitmask_union_verify_empty_chunks_and_live_counts():
    """Verify exact NaN guard when an entire 32-key chunk has mask == 0 for some rows,
    plus partial live_counts (0, 2, 5, 8) and base-2 vs natural LSE scaling."""
    dev = torch.device("cuda:0")
    b, w, h, page_size = 4, 8, 16, 64
    gen = torch.Generator(device=dev).manual_seed(909)
    q = (torch.randn((b * w, h, 512), device=dev, generator=gen) * 0.25).bfloat16()
    sink = torch.linspace(-0.2, 0.2, h, device=dev, dtype=torch.float32)

    swa_vals = (torch.randn((b * 136, 512), device=dev, generator=gen) * 0.25).bfloat16()
    swa_cache = pack_deepseek_v41_cache_reference(swa_vals, page_size=136, cache_kind="swa")
    hist = torch.full((b,), 128, dtype=torch.int32, device=dev)

    idx_vals = (torch.randn((16 * page_size, 512), device=dev, generator=gen) * 0.25).bfloat16()
    idx_cache = pack_deepseek_v41_cache_reference(idx_vals, page_size=page_size, cache_kind="indexed")

    # Construct disjoint token blocks so every token sees multiple ALL-ZERO 32-key chunks!
    sel = torch.full((b, w, 64), -1, dtype=torch.int32, device=dev)
    for bi in range(b):
        for wi in range(w):
            sel[bi, wi] = torch.arange(wi * 64, (wi + 1) * 64, dtype=torch.int32, device=dev)
    lc = torch.tensor([0, 2, 5, 8], dtype=torch.int32, device=dev)
    uk, um, ul = bitmask_union_merge(sel, live_counts=lc)

    ws = allocate_bitmask_verify_workspace(b * w, h, max_splits=4, device=dev)
    for split_mode, ns in (("direct", 1), ("dsmem", 4), ("workspace", 4)):
        out_nat, lse_nat = bitmask_union_verify_mla(
            q,
            swa_cache,
            verify_width=w,
            sm_scale=SM_SCALE,
            swa_history_lens=hist,
            swa_contiguous=True,
            swa_page_size=136,
            indexed_k_cache=idx_cache,
            indexed_unique_keys=uk,
            indexed_mask_bits=um,
            indexed_union_len=ul,
            indexed_page_size=page_size,
            attn_sink=sink,
            live_counts=lc,
            return_lse=True,
            lse_scale="natural",
            num_splits=ns,
            split_mode=split_mode,
            split_workspace=ws if split_mode == "workspace" else None,
        )
        out_b2, lse_b2 = bitmask_union_verify_mla(
            q,
            swa_cache,
            verify_width=w,
            sm_scale=SM_SCALE,
            swa_history_lens=hist,
            swa_contiguous=True,
            swa_page_size=136,
            indexed_k_cache=idx_cache,
            indexed_unique_keys=uk,
            indexed_mask_bits=um,
            indexed_union_len=ul,
            indexed_page_size=page_size,
            attn_sink=sink,
            live_counts=lc,
            return_lse=True,
            lse_scale="base2",
            num_splits=ns,
            split_mode=split_mode,
            split_workspace=ws if split_mode == "workspace" else None,
        )
        torch.cuda.synchronize()
        for bi in range(b):
            lb = int(lc[bi].item())
            if lb == 0:
                continue
            r0, r1 = bi * w, bi * w + lb
            assert not torch.isnan(out_nat[r0:r1]).any()
            assert not torch.isnan(lse_nat[r0:r1]).any()
            assert torch.equal(out_nat[r0:r1], out_b2[r0:r1])
            assert torch.allclose(lse_nat[r0:r1], lse_b2[r0:r1] * math.log(2.0), atol=1e-4)


def test_bitmask_union_verify_cuda_graph_50_iter_zero_alloc():
    dev = torch.device("cuda:0")
    b, w, h, page_size = 4, 8, 16, 64
    rows = b * w
    gen = torch.Generator(device=dev).manual_seed(2026)

    q = (torch.randn((rows, h, 512), device=dev, generator=gen) * 0.25).bfloat16()
    sink = torch.linspace(-0.3, 0.3, h, device=dev, dtype=torch.float32)
    swa_cache = pack_deepseek_v41_cache_reference(
        (torch.randn((b * 136, 512), device=dev, generator=gen) * 0.25).bfloat16(),
        page_size=136,
        cache_kind="swa",
    )
    hist = torch.full((b,), 128, dtype=torch.int32, device=dev)
    idx_cache = pack_deepseek_v41_cache_reference(
        (torch.randn((32 * page_size, 512), device=dev, generator=gen) * 0.25).bfloat16(),
        page_size=page_size,
        cache_kind="indexed",
    )
    sel_g = _make_zipf_indices(b, w, 512, 32 * page_size, overlap_frac=0.70, seed=7).to(
        dev
    ).reshape(b * 2, 4, 512)
    uk, um, ul = allocate_bitmask_union_workspace(b * 2, 4, 512, device=dev)
    split_ws = allocate_bitmask_verify_workspace(rows, h, max_splits=8, device=dev)
    out = torch.empty_like(q)
    out_lse = torch.empty((rows, h), dtype=torch.float32, device=dev)
    s_plan, _ = plan_bitmask_union_splits(
        batch_size=b, num_heads=h, verify_width=w, has_indexed=True, has_workspace=True
    )
    s_use = min(s_plan, 8)

    def _step():
        bitmask_union_merge(sel_g, unique_keys=uk, mask_bits=um, union_len=ul)
        bitmask_union_verify_mla(
            q,
            swa_cache,
            verify_width=w,
            sm_scale=SM_SCALE,
            swa_history_lens=hist,
            swa_contiguous=True,
            swa_page_size=136,
            indexed_k_cache=idx_cache,
            indexed_unique_keys=uk,
            indexed_mask_bits=um,
            indexed_union_len=ul,
            indexed_page_size=page_size,
            attn_sink=sink,
            out=out,
            out_lse=out_lse,
            return_lse=True,
            num_splits=s_use,
            split_mode="workspace",
            split_workspace=split_ws,
        )

    # Warmup compile + capture eager reference
    _step()
    torch.cuda.synchronize()
    eager_out = out.clone()
    eager_lse = out_lse.clone()

    mem_before = torch.cuda.memory_allocated(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _step()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(dev) == mem_before

    out.zero_()
    out_lse.zero_()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out)
    assert torch.equal(out_lse, eager_lse)
