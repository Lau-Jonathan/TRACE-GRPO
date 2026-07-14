"""CPU-only equivalence test for CAPA reference path.

Strategy: build a synthetic trajectory of L tokens with random K/V; pick a
turn list ``[(ctx_end_t, seg_len_t, crit_len_t)]``; for *each* turn,

  1. Compute the gold attention output with **plain causal attention** over
     ``[K_traj | K_new]`` of length ``ctx_end_t + seg_len_t``.
  2. Compute the same output via :func:`reference_capa_attention` going
     through the page-table → stub-pages plumbing.

Bit-exact equality is required (fp32 + eager attn). This pins down all the
indexing/scatter logic before we wire the FA2 kernel.

The test deliberately stresses these edge cases:
  - turns where ``ctx_end`` is *not* a multiple of PAGE_SIZE (forces tail>0)
  - turns where ``ctx_end`` *is* aligned (tail=0, no carryover into stub)
  - turns where ``seg_len + tail`` spans multiple stub pages
  - the smallest possible trajectory (1 page) and the largest stub fan-out
    (so trailing zero-pad slots are exercised)
"""

from __future__ import annotations

import math

import pytest
import torch

from trace_grpo.patches.capa import (
    PAGE_SIZE,
    TurnLayout,
    build_page_tables,
    capa_attention_forward,
    kv_to_pages,
    reference_capa_attention,
    scatter_into_stub_pages,
)


# Use a small page size in tests so we can stay in pages of a few tokens.
TEST_PAGE_SIZE = 8


def _full_causal_attention(
    Q: torch.Tensor,                      # (seg, n_heads, d_head)
    K: torch.Tensor,                      # (T, h_kv, d_head)
    V: torch.Tensor,
    abs_q_start: int,                     # absolute position of Q[0]
) -> torch.Tensor:
    """Plain reference: attention with absolute-position causal mask."""
    n_heads, d_head = Q.shape[1], Q.shape[-1]
    h_kv = K.shape[1]
    group = n_heads // h_kv
    if group > 1:
        K = K.repeat_interleave(group, dim=1)
        V = V.repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(d_head)
    seg = Q.shape[0]
    T = K.shape[0]
    scores = torch.einsum("snd,knd->nsk", Q, K) * scale
    q_pos = torch.arange(seg) + abs_q_start
    k_pos = torch.arange(T)
    mask = k_pos[None, :] <= q_pos[:, None]
    scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("nsk,knd->snd", attn, V)


def _build_synthetic(
    L_traj: int,
    turn_specs: list[tuple[int, int, int]],   # (ctx_end, seg_len, crit_len)
    n_heads: int = 2,
    h_kv: int = 1,
    d_head: int = 4,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)

    # --- Trajectory KV (the "old_log_prob" cache).
    K_traj = torch.randn(1, h_kv, L_traj, d_head, generator=g)
    V_traj = torch.randn(1, h_kv, L_traj, d_head, generator=g)

    # --- Per-turn new K/V (projection of [<teacher_note>+critique | response]).
    Q_per_turn = []
    K_new_per_turn = []
    V_new_per_turn = []
    turns: list[TurnLayout] = []
    for ctx_end, seg_len, crit_len in turn_specs:
        Q_per_turn.append(torch.randn(seg_len, n_heads, d_head, generator=g))
        K_new_per_turn.append(torch.randn(seg_len, h_kv, d_head, generator=g))
        V_new_per_turn.append(torch.randn(seg_len, h_kv, d_head, generator=g))
        turns.append(
            TurnLayout(
                ctx_end=ctx_end,
                seg_len=seg_len,
                crit_len=crit_len,
                resp_len=seg_len - crit_len,
            )
        )
    Q_packed = torch.cat(Q_per_turn, dim=0)
    K_new = torch.cat(K_new_per_turn, dim=0)
    V_new = torch.cat(V_new_per_turn, dim=0)

    # --- Page the trajectory KV.
    K_traj_pages, V_traj_pages, _ = kv_to_pages(K_traj, V_traj, page_size=TEST_PAGE_SIZE)
    n_traj_pages = K_traj_pages.shape[0]

    # --- Build page tables.
    pt = build_page_tables(turns, n_traj_pages, page_size=TEST_PAGE_SIZE)

    # --- Allocate stub tensors and scatter into them.
    K_stub = torch.zeros(pt.n_stub_pages_total, TEST_PAGE_SIZE, h_kv, d_head)
    V_stub = torch.zeros(pt.n_stub_pages_total, TEST_PAGE_SIZE, h_kv, d_head)
    # Need un-paged trajectory KV for the scatter (Algorithm 3 reads
    # contiguously from the original KV stream).
    K_traj_flat = K_traj.squeeze(0).permute(1, 0, 2).contiguous()  # (L, h_kv, d_head)
    V_traj_flat = V_traj.squeeze(0).permute(1, 0, 2).contiguous()
    scatter_into_stub_pages(
        K_stub, V_stub, K_traj_flat, V_traj_flat, K_new, V_new,
        turns, pt.stub_layout, pt.cu_seqlens_q, page_size=TEST_PAGE_SIZE,
    )

    # --- Concatenate clean + stub for FA2-style indexing.
    K_pages = torch.cat([K_traj_pages, K_stub], dim=0)
    V_pages = torch.cat([V_traj_pages, V_stub], dim=0)

    return {
        "Q_packed": Q_packed,
        "K_pages": K_pages,
        "V_pages": V_pages,
        "K_traj_flat": K_traj_flat,
        "V_traj_flat": V_traj_flat,
        "K_new": K_new,
        "V_new": V_new,
        "turns": turns,
        "page_tables": pt,
        "n_heads": n_heads,
        "h_kv": h_kv,
        "d_head": d_head,
    }


def _gold_per_turn(d) -> torch.Tensor:
    """Compute attention output via plain reference (one full pass per turn)."""
    Q = d["Q_packed"]
    K_traj_flat = d["K_traj_flat"]
    V_traj_flat = d["V_traj_flat"]
    K_new = d["K_new"]
    V_new = d["V_new"]
    cu = d["page_tables"].cu_seqlens_q.tolist()
    out = torch.zeros_like(Q)
    for t, turn in enumerate(d["turns"]):
        ctx_end = turn.ctx_end
        seg_lo, seg_hi = cu[t], cu[t + 1]
        K_full = torch.cat([K_traj_flat[:ctx_end], K_new[seg_lo:seg_hi]], dim=0)
        V_full = torch.cat([V_traj_flat[:ctx_end], V_new[seg_lo:seg_hi]], dim=0)
        Q_t = Q[seg_lo:seg_hi]
        out[seg_lo:seg_hi] = _full_causal_attention(Q_t, K_full, V_full, abs_q_start=ctx_end)
    return out


def _capa(d) -> torch.Tensor:
    return reference_capa_attention(
        Q_packed=d["Q_packed"],
        K_pages_concat=d["K_pages"],
        V_pages_concat=d["V_pages"],
        block_table=d["page_tables"].block_table,
        cache_seqlens=d["page_tables"].cache_seqlens,
        cu_seqlens_q=d["page_tables"].cu_seqlens_q,
        turns=d["turns"],
        page_size=TEST_PAGE_SIZE,
    )


# ---------------------------------------------------------------------------
# Algorithm 1: kv_to_pages padding & shape.
# ---------------------------------------------------------------------------


def test_kv_to_pages_no_pad_when_aligned():
    K = torch.randn(1, 2, 16, 4)  # 16 = 2 pages of 8
    V = torch.randn_like(K)
    K_p, V_p, L = kv_to_pages(K, V, page_size=TEST_PAGE_SIZE)
    assert K_p.shape == (2, TEST_PAGE_SIZE, 2, 4)
    assert V_p.shape == (2, TEST_PAGE_SIZE, 2, 4)
    assert L == 16


def test_kv_to_pages_pads_partial_last_page():
    K = torch.randn(1, 2, 11, 4)  # 11 = 1 full page + 3 tail tokens
    V = torch.randn_like(K)
    K_p, V_p, L = kv_to_pages(K, V, page_size=TEST_PAGE_SIZE)
    assert K_p.shape == (2, TEST_PAGE_SIZE, 2, 4)
    assert L == 11
    # First 11 token slots match original; padding slots are zero.
    K_flat = K_p.permute(2, 0, 1, 3).reshape(2, -1, 4)  # (h_kv, n_pages*P, d)
    assert torch.allclose(K_flat[:, :11], K.squeeze(0))
    assert torch.allclose(K_flat[:, 11:], torch.zeros_like(K_flat[:, 11:]))


# ---------------------------------------------------------------------------
# Algorithm 2: build_page_tables structure.
# ---------------------------------------------------------------------------


def test_build_page_tables_basic_two_turns():
    """Two turns, one mid-page tail and one page-aligned ctx_end."""
    turns = [
        TurnLayout(ctx_end=10, seg_len=5, crit_len=2, resp_len=3),  # tail=2, n_clean=1
        TurnLayout(ctx_end=16, seg_len=4, crit_len=1, resp_len=3),  # tail=0, n_clean=2
    ]
    pt = build_page_tables(turns, n_traj_pages=3, page_size=TEST_PAGE_SIZE)

    # Turn 0: 1 clean page + ceil((2+5)/8) = 1 stub page; abs page ids = [0, 3].
    assert pt.stub_layout[0].n_clean == 1
    assert pt.stub_layout[0].tail == 2
    assert pt.stub_layout[0].n_stub_pages == 1
    assert pt.stub_layout[0].stub_start == 0

    # Turn 1: 2 clean pages + ceil((0+4)/8) = 1 stub page; abs page ids = [0, 1, 4].
    assert pt.stub_layout[1].n_clean == 2
    assert pt.stub_layout[1].tail == 0
    assert pt.stub_layout[1].n_stub_pages == 1
    assert pt.stub_layout[1].stub_start == 1

    # block_table must be (N=2, max_pages_per_turn=3); turn-0 row uses 2 of 3 slots.
    assert pt.max_pages_per_turn == 3
    assert pt.block_table.tolist() == [[0, 3, 0], [0, 1, 4]]
    assert pt.cache_seqlens.tolist() == [15, 20]
    assert pt.cu_seqlens_q.tolist() == [0, 5, 9]


def test_build_page_tables_stub_spans_multiple_pages():
    """A long critique+response that spills across two stub pages."""
    turns = [TurnLayout(ctx_end=5, seg_len=15, crit_len=3, resp_len=12)]
    pt = build_page_tables(turns, n_traj_pages=1, page_size=TEST_PAGE_SIZE)
    # tail=5, n_clean=0; 5 + 15 = 20 -> ceil(20/8) = 3 stub pages.
    assert pt.stub_layout[0].n_clean == 0
    assert pt.stub_layout[0].tail == 5
    assert pt.stub_layout[0].n_stub_pages == 3
    assert pt.block_table.tolist() == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# End-to-end equivalence tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "L_traj, turn_specs, seed",
    [
        # Single turn, ctx_end = 1 page exactly (tail=0).
        (24, [(8, 5, 2)], 0),
        # Single turn, ctx_end mid-page (tail=2).
        (24, [(10, 5, 2)], 1),
        # Two turns, second one starts mid-page.
        (32, [(10, 5, 2), (15, 6, 1)], 2),
        # Three turns, monotonically increasing ctx_end. Note that
        # in real usage the trajectory tokens for *later* turns must
        # already be in K_traj — here we simply use a long enough L_traj.
        (40, [(8, 4, 1), (16, 6, 2), (24, 5, 1)], 3),
        # Stub fan-out: long seg_len that needs multiple stub pages.
        (16, [(5, 12, 3)], 4),
        # Edge: ctx_end == 0 (first turn of a fresh trajectory).
        (24, [(0, 6, 2)], 5),
    ],
)
def test_capa_equivalence(L_traj, turn_specs, seed):
    """Reference CAPA path == per-turn full-attention reference, fp32 eager."""
    d = _build_synthetic(L_traj=L_traj, turn_specs=turn_specs, seed=seed)
    out_gold = _gold_per_turn(d)
    out_capa = _capa(d)
    diff = (out_gold - out_capa).abs().max().item()
    # fp32 with the same arithmetic order should be bit-equal modulo
    # negligible floating jitter from concat order.
    assert diff < 1e-5, (
        f"CAPA != gold (max|diff|={diff:.3e})\n"
        f"out_gold[0]={out_gold[0]}\nout_capa[0]={out_capa[0]}"
    )


def test_capa_equivalence_gqa():
    """Same equivalence with multi-head attention (n_heads > h_kv)."""
    g = torch.Generator().manual_seed(123)
    L_traj, n_heads, h_kv, d_head = 24, 4, 2, 4
    K_traj = torch.randn(1, h_kv, L_traj, d_head, generator=g)
    V_traj = torch.randn(1, h_kv, L_traj, d_head, generator=g)
    turn = TurnLayout(ctx_end=10, seg_len=5, crit_len=2, resp_len=3)
    Q = torch.randn(turn.seg_len, n_heads, d_head, generator=g)
    K_new = torch.randn(turn.seg_len, h_kv, d_head, generator=g)
    V_new = torch.randn(turn.seg_len, h_kv, d_head, generator=g)

    K_traj_pages, V_traj_pages, _ = kv_to_pages(K_traj, V_traj, page_size=TEST_PAGE_SIZE)
    pt = build_page_tables([turn], n_traj_pages=K_traj_pages.shape[0], page_size=TEST_PAGE_SIZE)
    K_stub = torch.zeros(pt.n_stub_pages_total, TEST_PAGE_SIZE, h_kv, d_head)
    V_stub = torch.zeros(pt.n_stub_pages_total, TEST_PAGE_SIZE, h_kv, d_head)
    K_traj_flat = K_traj.squeeze(0).permute(1, 0, 2).contiguous()
    V_traj_flat = V_traj.squeeze(0).permute(1, 0, 2).contiguous()
    scatter_into_stub_pages(
        K_stub, V_stub, K_traj_flat, V_traj_flat, K_new, V_new,
        [turn], pt.stub_layout, pt.cu_seqlens_q, page_size=TEST_PAGE_SIZE,
    )
    K_pages = torch.cat([K_traj_pages, K_stub], dim=0)
    V_pages = torch.cat([V_traj_pages, V_stub], dim=0)

    out_capa = reference_capa_attention(
        Q_packed=Q, K_pages_concat=K_pages, V_pages_concat=V_pages,
        block_table=pt.block_table, cache_seqlens=pt.cache_seqlens,
        cu_seqlens_q=pt.cu_seqlens_q, turns=[turn], page_size=TEST_PAGE_SIZE,
    )

    K_full = torch.cat([K_traj_flat[: turn.ctx_end], K_new], dim=0)
    V_full = torch.cat([V_traj_flat[: turn.ctx_end], V_new], dim=0)
    out_gold = _full_causal_attention(Q, K_full, V_full, abs_q_start=turn.ctx_end)

    diff = (out_gold - out_capa).abs().max().item()
    assert diff < 1e-5, f"GQA: max|diff|={diff:.3e}"


def test_capa_attention_forward_registered_with_transformers():
    pytest.importorskip("transformers")
    import trace_grpo.patches  # noqa: F401
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert ALL_ATTENTION_FUNCTIONS["capa"] is capa_attention_forward


def test_capa_attention_forward_consumes_tracegrpo_payload():
    """The registered hook accepts the tracegrpo capa_payload fields."""
    d = _build_synthetic(L_traj=24, turn_specs=[(10, 5, 2), (16, 4, 1)], seed=42)
    pt = d["page_tables"]
    Q = d["Q_packed"]
    K_new = d["K_new"]
    V_new = d["V_new"]

    # HF attention functions receive (B, H, S, D). CAPA packs all query
    # turns into a single B=1 sequence and uses cu_seqlens_q for ragged turns.
    q_hf = Q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    k_hf = K_new.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    v_hf = V_new.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

    K_clean = d["K_traj_flat"].permute(1, 0, 2).unsqueeze(0).contiguous()
    V_clean = d["V_traj_flat"].permute(1, 0, 2).unsqueeze(0).contiguous()
    K_traj_pages, V_traj_pages, L_orig = kv_to_pages(K_clean, V_clean, page_size=TEST_PAGE_SIZE)
    payload = {
        "block_table": pt.block_table,
        "cache_seqlens": pt.cache_seqlens,
        "cu_seqlens_q": pt.cu_seqlens_q,
        "stub_layout": pt.stub_layout,
        "turns": d["turns"],
        "K_traj_pages": K_traj_pages,
        "V_traj_pages": V_traj_pages,
        "traj_kv_len": L_orig,
        "page_size": TEST_PAGE_SIZE,
    }

    out_hf, attn_weights = capa_attention_forward(
        None,
        q_hf,
        k_hf,
        v_hf,
        None,
        scaling=1.0 / math.sqrt(d["d_head"]),
        capa_payload=payload,
    )

    assert attn_weights is None
    out_packed = out_hf.squeeze(0).contiguous()
    expected = _capa(d)
    assert torch.allclose(out_packed, expected, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FA2 paged CAPA")
def test_capa_attention_forward_uses_fa2_paged_kernel(monkeypatch):
    """FA2 block_table path matches eager CAPA within bf16 tolerance."""
    from flash_attn import flash_attn_interface

    calls = []
    real_flash_attn_varlen_func = flash_attn_interface.flash_attn_varlen_func

    def spy_flash_attn_varlen_func(*args, **kwargs):
        calls.append(kwargs)
        return real_flash_attn_varlen_func(*args, **kwargs)

    monkeypatch.setattr(
        flash_attn_interface,
        "flash_attn_varlen_func",
        spy_flash_attn_varlen_func,
    )

    page_size = PAGE_SIZE
    d = _build_synthetic(
        L_traj=300,
        turn_specs=[(129, 17, 5), (260, 19, 4)],
        n_heads=4,
        h_kv=2,
        d_head=16,
        seed=7,
    )
    turns = d["turns"]
    K_clean = d["K_traj_flat"].permute(1, 0, 2).unsqueeze(0).contiguous()
    V_clean = d["V_traj_flat"].permute(1, 0, 2).unsqueeze(0).contiguous()
    K_traj_pages, V_traj_pages, L_orig = kv_to_pages(K_clean, V_clean, page_size=page_size)
    pt = build_page_tables(turns, n_traj_pages=K_traj_pages.shape[0], page_size=page_size)

    q_hf = d["Q_packed"].unsqueeze(0).permute(0, 2, 1, 3).contiguous().cuda().bfloat16()
    k_hf = d["K_new"].unsqueeze(0).permute(0, 2, 1, 3).contiguous().cuda().bfloat16()
    v_hf = d["V_new"].unsqueeze(0).permute(0, 2, 1, 3).contiguous().cuda().bfloat16()
    payload = {
        "block_table": pt.block_table.cuda(),
        "cache_seqlens": pt.cache_seqlens.cuda(),
        "cu_seqlens_q": pt.cu_seqlens_q.cuda(),
        "stub_layout": pt.stub_layout,
        "turns": turns,
        "K_traj_pages": K_traj_pages.cuda().bfloat16(),
        "V_traj_pages": V_traj_pages.cuda().bfloat16(),
        "traj_kv_len": L_orig,
        "page_size": page_size,
        "use_fa2": True,
    }

    out_hf, _ = capa_attention_forward(
        None,
        q_hf,
        k_hf,
        v_hf,
        None,
        scaling=1.0 / math.sqrt(d["d_head"]),
        capa_payload=payload,
    )

    fa2 = out_hf.squeeze(0).float().cpu()
    expected = reference_capa_attention(
        Q_packed=d["Q_packed"],
        K_pages_concat=d["K_pages"],
        V_pages_concat=d["V_pages"],
        block_table=d["page_tables"].block_table,
        cache_seqlens=d["page_tables"].cache_seqlens,
        cu_seqlens_q=d["page_tables"].cu_seqlens_q,
        turns=turns,
        page_size=TEST_PAGE_SIZE,
    )
    assert calls and calls[-1].get("block_table") is not None
    assert torch.allclose(fa2, expected, atol=1.5e-2, rtol=1.5e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
