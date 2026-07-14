"""Critique-Aware Paged Attention (CAPA).

Implements the algorithms in the paper §6 + Appendix C. The goal is
to compute, for one packed trajectory ``x = [s_0, a_1, s_1, ..., s_T, a_T]``
and a per-turn critique list ``c_1, ..., c_N``, the per-token log-prob shift

    δ_{i,k} = log p_θ(token_k | prefix_k, critique_t) − log p_θ(token_k | prefix_k)

at the **cost of one packed forward pass** (plus one cheap KV cache capture
on the original trajectory) instead of N full forwards. The savings are
realised by:

  1. **Page table**. The trajectory's KV cache is sliced into fixed-size
     pages (PAGE_SIZE=256, FA2 constraint). Each turn's attention reads from
     - the *clean* pages of the trajectory KV (read-only, shared), and
     - a small set of *stub* pages that hold the tail of trajectory KV
       (last partial page, masked off later tokens) prepended to that turn's
       new K/V (from the ``<teacher_note>+critique`` + response query).
  2. **Stub pages**. They are necessary because a clean trajectory page may
     contain tokens belonging to *later* turns. If we let turn ``t`` read the
     clean page directly, FA2's causal mask would still allow attention into
     those future tokens within the same page (FA2 masks at the *query token
     position* level, not the page level — but the page boundary doesn't
     align with the turn boundary). Stub pages give each turn a private copy
     of the tail history with only its own future tokens.

This module exposes both the eager reference attention used for indexing
equivalence tests and the Transformers ``ALL_ATTENTION_FUNCTIONS["capa"]``
backend used by the packed Qwen2/Qwen2.5 path. On CUDA fp16/bf16 the backend
routes to FlashAttention-2 varlen paged attention with ``block_table``.

Public entry points:

    kv_to_pages(K, V, page_size=256)
        Algorithm 1.
    build_page_tables(turns, n_traj_pages, page_size=256)
        Algorithm 2.
    scatter_into_stub_pages(K_stub, V_stub, K_traj_flat, V_traj_flat,
                            K_new, V_new, turns, stub_layout,
                            cu_seqlens_q, page_size=256)
        Algorithm 3.

The end-to-end model wrapper lives in :mod:`trace_grpo.patches.capa_forward`.
It captures per-layer trajectory KV once, builds these page tables, and runs
one packed conditioned forward for all annotated turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import math

import torch


PAGE_SIZE: int = 256


# ---------------------------------------------------------------------------
# Data classes describing the per-turn layout.
# ---------------------------------------------------------------------------


@dataclass
class TurnLayout:
    """Per-turn metadata required by CAPA.

    Attributes:
        ctx_end: Position right after the last trajectory token visible to
            this turn (``ctx_end_t`` in the spec). For turn ``t``, that's the
            length of ``[s_0, a_1, ..., s_{t-1}, a_{t-1}, s_t]`` — i.e. up to
            and including the env observation that triggered the turn, but
            *before* the assistant's response.
        seg_len: Length of the new tokens for this turn — ``|c_t| + |r_t|``
            (critique + response). Combined with ``ctx_end`` this fully
            describes where this turn's tokens live in the trajectory's RoPE
            frame.
        crit_len: ``|c_t|`` — length of the spliced ``<teacher_note>...
            </teacher_note>`` segment.
        resp_len: ``|r_t|`` — length of the assistant response.
    """

    ctx_end: int
    seg_len: int
    crit_len: int
    resp_len: int

    def __post_init__(self) -> None:
        if self.crit_len + self.resp_len != self.seg_len:
            raise ValueError(
                f"TurnLayout: crit_len({self.crit_len}) + resp_len({self.resp_len}) "
                f"must equal seg_len({self.seg_len})"
            )
        if self.ctx_end < 0 or self.seg_len <= 0:
            raise ValueError(f"TurnLayout: invalid ctx_end={self.ctx_end} or seg_len={self.seg_len}")


@dataclass
class StubEntry:
    """Per-turn stub page record built by :func:`build_page_tables`.

    Attributes:
        n_clean: Number of fully-occupied trajectory pages this turn can
            read directly (``ctx_end // PAGE_SIZE``).
        tail: ``ctx_end % PAGE_SIZE`` — number of trajectory tokens in the
            partial last page that this turn needs but cannot read from the
            shared cache (would leak future turns' tokens).
        n_stub_pages: Number of stub pages allocated for this turn:
            ``ceil((tail + seg_len) / PAGE_SIZE)``.
        stub_start: First stub-page slot in the global K_stub/V_stub tensors
            (these slots are appended *after* all clean trajectory pages,
            so the absolute page index in ``block_table`` is
            ``n_traj_pages + stub_start``).
    """

    n_clean: int
    tail: int
    n_stub_pages: int
    stub_start: int  # local stub offset (before adding n_traj_pages)


@dataclass
class PageTables:
    """Output bundle of :func:`build_page_tables`.

    Attributes:
        block_table: ``(N, max_pages_per_turn)`` int32. Row ``t`` lists the
            page indices (in the *combined* clean+stub page tensor) that
            turn ``t`` attends to, in order. Padded with zeros (FA2 ignores
            the padding because ``cache_seqlens`` says how many tokens are
            actually live).
        cache_seqlens: ``(N,)`` int32. Total key length attended to by turn
            ``t`` — i.e. ``ctx_end_t + seg_len_t``.
        cu_seqlens_q: ``(N+1,)`` int32. Cumulative ragged Q lengths for the
            packed query (FA2 varlen API).
        stub_layout: ``list[StubEntry]`` of length ``N``.
        max_pages_per_turn: width of ``block_table`` second dimension.
        n_stub_pages_total: total stub pages allocated across all turns.
    """

    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    stub_layout: List[StubEntry]
    max_pages_per_turn: int
    n_stub_pages_total: int


# ---------------------------------------------------------------------------
# Algorithm 1: KvToPages.
# ---------------------------------------------------------------------------


def kv_to_pages(
    K: torch.Tensor,
    V: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Slice a single trajectory's KV cache into fixed-size pages.

    Args:
        K: ``(1, h_kv, L, d_head)``. Either eager attention's stored cache
            or a shape constructed from a forward run.
        V: same shape as K.
        page_size: pages are PAGE_SIZE-aligned tokens (FA2 constraint).

    Returns:
        Tuple ``(K_pages, V_pages, L_orig)`` where:
          - ``K_pages``: ``(n_pages, page_size, h_kv, d_head)``. Note the
            FA2 paged-API convention puts ``page_size`` *before* ``h_kv``.
          - ``V_pages``: same shape.
          - ``L_orig``: original sequence length (so the caller can mask
            out the padding when reconstructing).
    """
    if K.ndim != 4 or V.ndim != 4 or K.shape[0] != 1:
        raise ValueError(f"kv_to_pages: expected (1, h_kv, L, d_head), got K{tuple(K.shape)}")
    if K.shape != V.shape:
        raise ValueError(f"kv_to_pages: K{tuple(K.shape)} != V{tuple(V.shape)}")

    _, h_kv, L, d_head = K.shape
    L_orig = L
    pad_len = (page_size - L % page_size) % page_size
    if pad_len > 0:
        zeros_K = K.new_zeros(1, h_kv, pad_len, d_head)
        zeros_V = V.new_zeros(1, h_kv, pad_len, d_head)
        K = torch.cat([K, zeros_K], dim=2)
        V = torch.cat([V, zeros_V], dim=2)

    L_pad = L + pad_len
    n_pages = L_pad // page_size
    # (1, h_kv, n_pages * page_size, d_head)
    # → (n_pages, page_size, h_kv, d_head)
    K_pages = (
        K.view(1, h_kv, n_pages, page_size, d_head)
        .squeeze(0)                        # (h_kv, n_pages, page_size, d_head)
        .permute(1, 2, 0, 3)               # (n_pages, page_size, h_kv, d_head)
        .contiguous()
    )
    V_pages = (
        V.view(1, h_kv, n_pages, page_size, d_head)
        .squeeze(0)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    return K_pages, V_pages, L_orig


# ---------------------------------------------------------------------------
# Algorithm 2: BuildPageTables.
# ---------------------------------------------------------------------------


def build_page_tables(
    turns: Sequence[TurnLayout],
    n_traj_pages: int,
    page_size: int = PAGE_SIZE,
    device: torch.device | str | None = None,
    dtype_int: torch.dtype = torch.int32,
) -> PageTables:
    """Build per-turn page tables for FA2 paged attention.

    Implements ``_build_page_tables`` from spec §6.2 plus the variable-page
    accounting described in Appendix C.2.

    Important: pages are referenced in the *combined* tensor that has clean
    trajectory pages first (indices ``[0, n_traj_pages)``) and stub pages
    after (indices ``[n_traj_pages, n_traj_pages + stub_total)``).

    Args:
        turns: ordered list of :class:`TurnLayout`.
        n_traj_pages: number of clean trajectory pages (output of Algorithm 1).
        page_size: must match Algorithm 1.

    Returns:
        :class:`PageTables` (see field docstrings for shapes).
    """
    if len(turns) == 0:
        raise ValueError("build_page_tables: turns must be non-empty")

    n_turns = len(turns)
    cache_seqlens = torch.zeros(n_turns, dtype=dtype_int, device=device)
    seg_lens = torch.zeros(n_turns, dtype=dtype_int, device=device)

    stub_layout: List[StubEntry] = []
    cursor_stub = 0
    for t, turn in enumerate(turns):
        ctx_end = turn.ctx_end
        seg_len = turn.seg_len
        if ctx_end + seg_len > n_traj_pages * page_size + 10**9:  # noqa: just a sanity bound
            pass
        tail = ctx_end % page_size
        n_clean = ctx_end // page_size
        # Stub pages must cover both the `tail` history tokens (which we cannot
        # read from clean pages without leaking later turns) and the new
        # `seg_len` tokens generated by this turn.
        n_stub = math.ceil((tail + seg_len) / page_size)
        stub_layout.append(
            StubEntry(
                n_clean=n_clean,
                tail=tail,
                n_stub_pages=n_stub,
                stub_start=cursor_stub,
            )
        )
        cursor_stub += n_stub
        cache_seqlens[t] = ctx_end + seg_len
        seg_lens[t] = seg_len

    n_stub_pages_total = cursor_stub
    max_pages_per_turn = max(s.n_clean + s.n_stub_pages for s in stub_layout)
    block_table = torch.zeros(
        n_turns, max_pages_per_turn, dtype=dtype_int, device=device
    )

    for t, s in enumerate(stub_layout):
        # Clean pages: direct reference to trajectory pages [0, n_clean).
        if s.n_clean > 0:
            block_table[t, : s.n_clean] = torch.arange(
                s.n_clean, dtype=dtype_int, device=device
            )
        # Stub pages: appended after the n_traj_pages clean pages.
        if s.n_stub_pages > 0:
            stub_abs_start = n_traj_pages + s.stub_start
            block_table[t, s.n_clean : s.n_clean + s.n_stub_pages] = torch.arange(
                stub_abs_start,
                stub_abs_start + s.n_stub_pages,
                dtype=dtype_int,
                device=device,
            )

    # cu_seqlens_q is the FA2 ragged-Q convention: prepend 0 to cumsum(seg_len).
    cu_seqlens_q = torch.zeros(n_turns + 1, dtype=dtype_int, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seg_lens, dim=0)

    return PageTables(
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        stub_layout=stub_layout,
        max_pages_per_turn=max_pages_per_turn,
        n_stub_pages_total=n_stub_pages_total,
    )


# ---------------------------------------------------------------------------
# Algorithm 3: ScatterIntoStubPages.
# ---------------------------------------------------------------------------


def scatter_into_stub_pages(
    K_stub: torch.Tensor,                # (n_stub_total, page_size, h_kv, d_head)
    V_stub: torch.Tensor,                # same shape
    K_traj_flat: torch.Tensor,           # (L_traj, h_kv, d_head)  -- un-paged
    V_traj_flat: torch.Tensor,
    K_new: torch.Tensor,                 # (sum(seg_len), h_kv, d_head)
    V_new: torch.Tensor,
    turns: Sequence[TurnLayout],
    stub_layout: Sequence[StubEntry],
    cu_seqlens_q: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> None:
    """In-place fill of ``K_stub`` / ``V_stub`` so each turn's stub pages
    start with the trajectory's tail-history slice and continue with this
    turn's freshly-projected K/V.

    Implements Appendix C.3 / spec §6.3 ``_scatter_into_stub_pages``.

    The stub block layout for turn t is:

        | tail history (tail tokens) | K_new for turn t (seg_len tokens) | (zero-padded to next page boundary)

    Reads the appropriate slice of the un-paged trajectory K/V for the tail
    and ``K_new[cu_q[t] : cu_q[t+1]]`` for the new tokens.
    """
    if K_stub.shape != V_stub.shape:
        raise ValueError(f"K_stub{tuple(K_stub.shape)} != V_stub{tuple(V_stub.shape)}")
    n_stub_total, ps, h_kv, d_head = K_stub.shape
    if ps != page_size:
        raise ValueError(f"K_stub page_size {ps} != argument page_size {page_size}")
    L_traj = K_traj_flat.shape[0]

    cu = cu_seqlens_q.tolist() if isinstance(cu_seqlens_q, torch.Tensor) else list(cu_seqlens_q)

    for t, (turn, s) in enumerate(zip(turns, stub_layout)):
        ctx_end = turn.ctx_end
        seg_len = turn.seg_len
        tail = s.tail
        n_stub = s.n_stub_pages
        stub_start = s.stub_start  # local stub-tensor offset

        # Build the contiguous stream [tail history | K_new for this turn].
        if tail > 0:
            tail_slice_K = K_traj_flat[ctx_end - tail : ctx_end]  # (tail, h_kv, d_head)
            tail_slice_V = V_traj_flat[ctx_end - tail : ctx_end]
        else:
            tail_slice_K = K_traj_flat[0:0]
            tail_slice_V = V_traj_flat[0:0]

        new_K = K_new[cu[t] : cu[t + 1]]
        new_V = V_new[cu[t] : cu[t + 1]]
        if new_K.shape[0] != seg_len or new_V.shape[0] != seg_len:
            raise ValueError(
                f"scatter_into_stub_pages: turn {t} expected seg_len={seg_len} new tokens, "
                f"got {new_K.shape[0]}"
            )

        K_seg = torch.cat([tail_slice_K, new_K], dim=0)  # (tail+seg_len, h_kv, d_head)
        V_seg = torch.cat([tail_slice_V, new_V], dim=0)
        L_seg = K_seg.shape[0]

        # Distribute the segment across n_stub stub pages, each holding up
        # to page_size tokens. Trailing slots in the last page stay zero
        # (they will be masked out by `cache_seqlens`).
        for p in range(n_stub):
            tok_lo = p * page_size
            tok_hi = min((p + 1) * page_size, L_seg)
            if tok_hi <= tok_lo:
                break
            n_w = tok_hi - tok_lo
            K_stub[stub_start + p, 0:n_w] = K_seg[tok_lo:tok_hi]
            V_stub[stub_start + p, 0:n_w] = V_seg[tok_lo:tok_hi]


# ---------------------------------------------------------------------------
# Reference (eager) implementation of Algorithm 5: full CAPA forward.
# ---------------------------------------------------------------------------


def reference_capa_attention(
    Q_packed: torch.Tensor,              # (sum(seg_len), n_heads, d_head)
    K_pages_concat: torch.Tensor,        # (n_pages_total, page_size, h_kv, d_head)
    V_pages_concat: torch.Tensor,
    block_table: torch.Tensor,           # (N, max_pages)
    cache_seqlens: torch.Tensor,         # (N,)
    cu_seqlens_q: torch.Tensor,          # (N+1,)
    turns: Sequence[TurnLayout],
    *,
    page_size: int = PAGE_SIZE,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """CPU-only reference implementation of CAPA attention using eager torch
    ops, suitable for bit-exact equivalence testing against a non-paged
    full forward.

    Uses absolute-position causal masking: a query token at trajectory
    position ``pos_q`` (= ``ctx_end_t + offset_within_turn``) attends to
    *all* key positions ``pos_k`` such that ``pos_k <= pos_q``. The page
    table simply gathers those keys; the causal mask is applied on the
    *resulting* (turn_seg_len, T_t) attention score matrix.

    Args:
        Q_packed: ``(sum_seg, n_heads, d_head)`` packed query tokens for all
            turns concatenated in order.
        K_pages_concat / V_pages_concat: concatenation of clean trajectory
            pages and stub pages, in that order, shape
            ``(n_pages_total, page_size, h_kv, d_head)``.
        block_table: from :func:`build_page_tables`.
        cache_seqlens: from :func:`build_page_tables`.
        cu_seqlens_q: from :func:`build_page_tables`.
        turns: per-turn layout (used to compute absolute positions for the
            causal mask).

    Returns:
        ``(sum_seg, n_heads, d_head)`` packed attention output, ready to be
        scattered back into per-turn rows.
    """
    n_heads = Q_packed.shape[1]
    d_head = Q_packed.shape[-1]
    h_kv = K_pages_concat.shape[-2]
    if n_heads % h_kv != 0:
        raise ValueError(f"n_heads {n_heads} must be a multiple of h_kv {h_kv}")
    group_size = n_heads // h_kv  # MQA/GQA expansion factor
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_head)

    cu = cu_seqlens_q.tolist()
    n_turns = len(turns)
    out = torch.zeros_like(Q_packed)

    for t in range(n_turns):
        seg_lo = cu[t]
        seg_hi = cu[t + 1]
        seg_len = seg_hi - seg_lo
        ctx_end = turns[t].ctx_end
        T_t = int(cache_seqlens[t].item())  # total live key length for turn t

        # Gather this turn's keys/values by walking block_table[t].
        # We ignore zero-padded pages by trusting cache_seqlens to bound T_t.
        page_ids = block_table[t].tolist()
        keys = []
        vals = []
        remaining = T_t
        for pid in page_ids:
            take = min(page_size, remaining)
            if take <= 0:
                break
            keys.append(K_pages_concat[pid, :take])  # (take, h_kv, d_head)
            vals.append(V_pages_concat[pid, :take])
            remaining -= take
        if remaining > 0:
            raise RuntimeError(
                f"reference_capa_attention: turn {t} expected {T_t} keys, "
                f"only gathered {T_t - remaining} from block_table — page "
                f"table is malformed"
            )
        K_t = torch.cat(keys, dim=0)               # (T_t, h_kv, d_head)
        V_t = torch.cat(vals, dim=0)

        # Expand h_kv -> n_heads (MQA/GQA).
        if group_size > 1:
            K_t = K_t.repeat_interleave(group_size, dim=1)
            V_t = V_t.repeat_interleave(group_size, dim=1)

        Q_t = Q_packed[seg_lo:seg_hi]              # (seg_len, n_heads, d_head)

        # Attention scores: (n_heads, seg_len, T_t).
        scores = torch.einsum("snd,knd->nsk", Q_t, K_t) * softmax_scale

        # Causal mask in *absolute* trajectory positions.
        # Q absolute positions: [ctx_end + 0, ctx_end + 1, ..., ctx_end + seg_len - 1]
        # K absolute positions: [0, 1, ..., T_t - 1]   (FA2 paged convention:
        # the live KV is indexed by the natural trajectory order — this works
        # here because clean pages are concatenated in trajectory order and
        # stub pages append the *contiguous* tail+new run, see Algorithm 3).
        q_pos = torch.arange(seg_len, device=Q_t.device) + ctx_end
        k_pos = torch.arange(T_t, device=Q_t.device)
        mask = k_pos[None, :] <= q_pos[:, None]   # (seg_len, T_t), True = keep
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # (n_heads, seg_len, T_t) @ (T_t, n_heads, d_head) -> (seg_len, n_heads, d_head)
        out_t = torch.einsum("nsk,knd->snd", attn, V_t)
        out[seg_lo:seg_hi] = out_t

    return out


# ---------------------------------------------------------------------------
# Transformers attention-dispatch hook.
# ---------------------------------------------------------------------------


def _flatten_hf_states(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert HF attention states ``(B, H, S, D)`` to ``(B*S, H, D)``."""
    if x.ndim != 4:
        raise ValueError(f"expected HF attention states (B, H, S, D), got {tuple(x.shape)}")
    bsz, n_heads, seq_len, d_head = x.shape
    packed = x.permute(0, 2, 1, 3).reshape(bsz * seq_len, n_heads, d_head).contiguous()
    return packed, (bsz, seq_len)


def _unflatten_hf_attention_output(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Convert ``(B*S, H, D)`` to HF attention output ``(B, S, H, D)``.

    Transformers attention backends receive Q/K/V as ``(B, H, S, D)``, but
    return attention output as ``(B, S, H, D)`` before the model reshapes it
    back to hidden size.
    """
    bsz, seq_len = shape
    return x.reshape(bsz, seq_len, x.shape[-2], x.shape[-1]).contiguous()


def _pages_to_flat(pages: torch.Tensor, *, upto: int | None = None) -> torch.Tensor:
    """Convert ``(n_pages, page_size, H, D)`` to ``(L, H, D)``."""
    flat = pages.reshape(-1, pages.shape[-2], pages.shape[-1]).contiguous()
    return flat if upto is None else flat[:upto]


def _new_kv_from_hf_states(x: torch.Tensor) -> torch.Tensor:
    """Convert HF K/V states ``(B, H_kv, S, D)`` to ``(B*S, H_kv, D)``."""
    packed, _ = _flatten_hf_states(x)
    return packed


def _payload_get(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    joined = ", ".join(names)
    raise KeyError(f"capa_payload missing required field: one of {joined}")


def capa_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Transformers ``ALL_ATTENTION_FUNCTIONS['capa']`` entry.

    The model layer supplies Q/K/V for the packed new sequence
    ``[c_1 r_1 c_2 r_2 ...]``. ``capa_payload`` supplies the read-only
    trajectory KV pages and per-turn page table metadata:

    ``block_table``, ``cache_seqlens``, ``cu_seqlens_q``, ``stub_layout``,
    ``turns``/``turn_layouts``, ``K_traj_pages``, ``V_traj_pages``.

    If the caller has already materialized stub pages it may pass
    ``K_stub_pages``/``V_stub_pages`` or ``K_pages_concat``/``V_pages_concat``.
    Otherwise this function builds stub pages from the incoming ``key`` and
    ``value`` states. On CUDA fp16/bf16 it routes to FA2 varlen paged
    attention with ``block_table``. CPU/fp32 keeps the eager reference path
    so unit tests can still check exact indexing semantics without requiring
    a GPU kernel.
    """
    payload = kwargs.get("capa_payload")
    if payload is None:
        raise ValueError("capa attention requires a capa_payload dict")
    if "layers" in payload:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            raise ValueError("layered capa_payload requires attention module.layer_idx")
        layer_payload = dict(payload["layers"][int(layer_idx)])
        shared_payload = {k: v for k, v in payload.items() if k != "layers"}
        shared_payload.update(layer_payload)
        payload = shared_payload
    if attention_mask is not None:
        raise ValueError("capa attention builds its own causal/page mask; attention_mask must be None")

    page_size = int(payload.get("page_size", PAGE_SIZE))
    turns = _payload_get(payload, "turns", "turn_layouts")
    block_table = _payload_get(payload, "block_table").to(device=query.device)
    cache_seqlens = _payload_get(payload, "cache_seqlens").to(device=query.device)
    cu_seqlens_q = _payload_get(payload, "cu_seqlens_q").to(device=query.device)

    q_packed, out_shape = _flatten_hf_states(query)
    q_packed = q_packed.to(dtype=query.dtype)

    K_pages_concat = payload.get("K_pages_concat")
    V_pages_concat = payload.get("V_pages_concat")
    if K_pages_concat is None or V_pages_concat is None:
        K_traj_pages = _payload_get(payload, "K_traj_pages").to(device=query.device, dtype=key.dtype)
        V_traj_pages = _payload_get(payload, "V_traj_pages").to(device=query.device, dtype=value.dtype)
        K_stub = payload.get("K_stub_pages")
        V_stub = payload.get("V_stub_pages")
        if K_stub is None or V_stub is None:
            n_stub_pages_total = sum(int(s.n_stub_pages) for s in payload["stub_layout"])
            K_stub = torch.zeros(
                n_stub_pages_total,
                page_size,
                key.shape[1],
                key.shape[-1],
                dtype=key.dtype,
                device=query.device,
            )
            V_stub = torch.zeros_like(K_stub)
            K_traj_flat = payload.get("K_traj_flat")
            V_traj_flat = payload.get("V_traj_flat")
            if K_traj_flat is None:
                K_traj_flat = _pages_to_flat(K_traj_pages, upto=payload.get("traj_kv_len"))
            else:
                K_traj_flat = K_traj_flat.to(device=query.device, dtype=key.dtype)
            if V_traj_flat is None:
                V_traj_flat = _pages_to_flat(V_traj_pages, upto=payload.get("traj_kv_len"))
            else:
                V_traj_flat = V_traj_flat.to(device=query.device, dtype=value.dtype)
            scatter_into_stub_pages(
                K_stub,
                V_stub,
                K_traj_flat,
                V_traj_flat,
                _new_kv_from_hf_states(key),
                _new_kv_from_hf_states(value),
                turns,
                payload["stub_layout"],
                cu_seqlens_q,
                page_size=page_size,
            )
        else:
            K_stub = K_stub.to(device=query.device, dtype=key.dtype)
            V_stub = V_stub.to(device=query.device, dtype=value.dtype)
        K_pages_concat = torch.cat([K_traj_pages, K_stub], dim=0)
        V_pages_concat = torch.cat([V_traj_pages, V_stub], dim=0)
    else:
        K_pages_concat = K_pages_concat.to(device=query.device, dtype=key.dtype)
        V_pages_concat = V_pages_concat.to(device=query.device, dtype=value.dtype)

    use_fa2 = bool(payload.get("use_fa2", True))
    if use_fa2 and query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        cu_seqlens_k = torch.zeros(
            cache_seqlens.shape[0] + 1,
            dtype=torch.int32,
            device=query.device,
        )
        cu_seqlens_k[1:] = torch.cumsum(cache_seqlens.to(torch.int32), dim=0)
        out = flash_attn_varlen_func(
            q=q_packed.contiguous(),
            k=K_pages_concat.contiguous(),
            v=V_pages_concat.contiguous(),
            cu_seqlens_q=cu_seqlens_q.to(torch.int32).contiguous(),
            cu_seqlens_k=cu_seqlens_k.contiguous(),
            max_seqlen_q=int(q_lens.max().item()),
            max_seqlen_k=int(cache_seqlens.max().item()),
            dropout_p=float(dropout),
            softmax_scale=scaling,
            causal=True,
            block_table=block_table.to(torch.int32).contiguous(),
        )
    else:
        out = reference_capa_attention(
            Q_packed=q_packed,
            K_pages_concat=K_pages_concat,
            V_pages_concat=V_pages_concat,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            turns=turns,
            page_size=page_size,
            softmax_scale=scaling,
        )
    return _unflatten_hf_attention_output(out.to(dtype=query.dtype), out_shape), None


def register_capa_attention() -> None:
    """Register ``_attn_implementation='capa'`` with Transformers."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    if ALL_ATTENTION_FUNCTIONS.get("capa") is not capa_attention_forward:
        ALL_ATTENTION_FUNCTIONS.register("capa", capa_attention_forward)
