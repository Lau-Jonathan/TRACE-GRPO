"""Real-model implementations of :class:`CritiqueConditionedForwardFn`.

Two implementations live here:

  1. :class:`ReferenceCritiqueForward` — runs ONE full ``model.forward``
     per annotated turn with the spliced
     ``[prefix | <teacher_note>critique</teacher_note> | response]``
     input. Slow (``O(N)`` forwards per trajectory) but trivially correct.
     This is the **golden** path used in unit tests and as a fallback when
     CAPA is disabled.

  2. :class:`CapaCritiqueForward` — runs ONE packed forward per
     trajectory using the CAPA paged-attention scaffolding. Production
     path for Qwen2/Qwen2.5-style HF causal LMs. It captures trajectory
     KV once, registers ``ALL_ATTENTION_FUNCTIONS["capa"]``, and feeds the
     model a packed ``[c_1 r_1 c_2 r_2 ...]`` sequence. Other architectures
     fall back to :class:`ReferenceCritiqueForward`.

Both classes share the contract from
:class:`~trace_grpo.patches.critique_conditioned_provider.CritiqueConditionedForwardFn`:

    Inputs:
      - trajectory_tokens (1D LongTensor, length L)
      - prompt_length (int)
      - turns (Sequence[TurnAnnotation])
    Output:
      - cond_logprobs (1D FloatTensor of length L − prompt_length),
        valid at response-axis positions of *annotated* turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .critique_conditioned_provider import (
    CritiqueConditionedForwardFn,
    TurnAnnotation,
)
from .capa import (
    PAGE_SIZE,
    TurnLayout,
    build_page_tables,
    kv_to_pages,
    register_capa_attention,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _logprobs_for_response_segment(
    model,
    full_input_ids: torch.Tensor,                # (1, L_total)
    *,
    response_start_in_full: int,
    response_len: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run model on ``full_input_ids`` and return per-token log-prob for
    the response segment ``[response_start_in_full : response_start_in_full
    + response_len]`` (positions in the *full* input frame).

    Implements the shifted-by-one convention: at full position ``k``, the
    log-prob of the *predicted* token ``input_ids[k]`` is
    ``log_softmax(logits[k-1])[input_ids[k]]``. So to score the response
    span ``[s, s+L)`` we look up logits at positions ``[s-1, s+L-1)``.
    """
    if full_input_ids.ndim == 1:
        full_input_ids = full_input_ids.unsqueeze(0)
    full_input_ids = full_input_ids.to(device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(full_input_ids, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
    logits = logits.to(dtype=dtype)
    if response_start_in_full <= 0:
        raise ValueError(
            f"response_start_in_full must be > 0 to score the first response token; "
            f"got {response_start_in_full}"
        )
    score_logits = logits[0, response_start_in_full - 1 : response_start_in_full - 1 + response_len, :]
    target_ids = full_input_ids[0, response_start_in_full : response_start_in_full + response_len]
    log_probs_full = F.log_softmax(score_logits, dim=-1)
    return log_probs_full.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Reference (one full forward per annotated turn).
# ---------------------------------------------------------------------------


@dataclass
class ReferenceCritiqueForward(CritiqueConditionedForwardFn):
    """Slow but obviously correct critique-conditioned forward.

    For each annotated turn, builds the conditioned input
    ``[trajectory[0:resp_start_full] | critique_token_ids |
    trajectory[resp_start_full : resp_end_full]]`` and reads the log-prob
    of the response tokens off that single forward. The unannotated
    positions of the returned tensor stay 0.

    Args:
        model: HF-style model with ``model(input_ids).logits``. Must be
            on a single device; we do not handle TP/DP here.
        device: where to run the forward. Defaults to model's device.
        dtype: precision of the log-prob computation. Use ``torch.float32``
            for bit-equal tests; ``torch.float16``/``torch.bfloat16`` for
            production.
    """

    model: torch.nn.Module
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    def __call__(
        self,
        trajectory_tokens: torch.Tensor,
        prompt_length: int,
        turns: Sequence[TurnAnnotation],
    ) -> torch.Tensor:
        L = int(trajectory_tokens.shape[0])
        out = torch.zeros(L - prompt_length, dtype=self.dtype, device="cpu")
        if not turns:
            return out

        traj = trajectory_tokens.to(dtype=torch.long)

        for turn in turns:
            if not turn.critique_token_ids:
                continue  # caller filtered already, but be defensive
            # response axis (turn) → trajectory-flat absolute positions
            resp_start_full = prompt_length + int(turn.response_start)
            resp_end_full = prompt_length + int(turn.response_end)
            if resp_start_full <= 0 or resp_end_full <= resp_start_full:
                continue
            critique_ids = torch.as_tensor(turn.critique_token_ids, dtype=torch.long)
            # Splice critique BEFORE the response in the full sequence.
            full_input = torch.cat(
                [traj[:resp_start_full], critique_ids, traj[resp_start_full:resp_end_full]],
                dim=0,
            )
            new_resp_start = resp_start_full + critique_ids.shape[0]
            response_len = resp_end_full - resp_start_full
            cond_lp = _logprobs_for_response_segment(
                self.model,
                full_input,
                response_start_in_full=new_resp_start,
                response_len=response_len,
                device=self.device,
                dtype=self.dtype,
            ).to(device="cpu")
            out[turn.response_start : turn.response_end] = cond_lp
        return out


# ---------------------------------------------------------------------------
# CAPA-backed (single packed forward).
# ---------------------------------------------------------------------------


@dataclass
class CapaCritiqueForward(CritiqueConditionedForwardFn):
    """Packed-forward critique-conditioned log-prob computation.

    For Qwen2/Qwen2.5-style HF causal LMs this runs one baseline forward to
    capture RoPE-applied trajectory KV, then one packed CAPA forward over
    ``[c_1 r_1 c_2 r_2 ...]``. Each decoder layer receives its own
    ``K_traj_pages`` / ``V_traj_pages`` through ``capa_payload["layers"]``.
    Non-Qwen2 architectures fall back to :class:`ReferenceCritiqueForward`.

    Args:
        model: HF model.
        device, dtype: forwarded to the underlying reference
            implementation.
    """

    model: torch.nn.Module
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    page_size: int = PAGE_SIZE
    use_fa2: bool = True

    def __post_init__(self) -> None:
        register_capa_attention()
        self._reference = ReferenceCritiqueForward(
            model=self.model, device=self.device, dtype=self.dtype
        )
        self.last_used_packed: bool = False

    @staticmethod
    def _inner_model(model):
        return getattr(model, "module", getattr(model, "_fsdp_wrapped_module", model))

    def _supports_qwen2_packed(self) -> bool:
        root = self._inner_model(self.model)
        inner = getattr(root, "model", None)
        return (
            inner is not None
            and hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(root, "lm_head")
        )

    @staticmethod
    def _get_attn_impl(model) -> str | None:
        root = CapaCritiqueForward._inner_model(model)
        cfg = getattr(root, "config", getattr(model, "config", None))
        return getattr(cfg, "_attn_implementation", None) if cfg is not None else None

    @staticmethod
    def _set_attn_impl(model, value: str | None) -> None:
        root = CapaCritiqueForward._inner_model(model)
        for cfg in (
            getattr(model, "config", None),
            getattr(root, "config", None),
            getattr(getattr(root, "model", None), "config", None),
        ):
            if cfg is not None and value is not None:
                setattr(cfg, "_attn_implementation", value)

    def _call_qwen2_packed(
        self,
        trajectory_tokens: torch.Tensor,
        prompt_length: int,
        turns: Sequence[TurnAnnotation],
    ) -> torch.Tensor:
        if isinstance(self.device, int):
            device = torch.device("cuda", self.device)
        else:
            device = torch.device(self.device)
        traj = trajectory_tokens.to(device=device, dtype=torch.long)
        L = int(traj.shape[0])
        out = torch.zeros(L - prompt_length, dtype=self.dtype, device="cpu")

        annotated = [t for t in turns if t.critique_token_ids]
        if not annotated:
            return out

        layouts: list[TurnLayout] = []
        packed_parts: list[torch.Tensor] = []
        pos_parts: list[torch.Tensor] = []
        response_slices: list[tuple[TurnAnnotation, int, int]] = []
        cursor = 0
        for turn in annotated:
            resp_start_full = prompt_length + int(turn.response_start)
            resp_end_full = prompt_length + int(turn.response_end)
            if resp_start_full <= 0 or resp_end_full <= resp_start_full:
                continue
            critique = torch.as_tensor(turn.critique_token_ids, dtype=torch.long, device=device)
            response = traj[resp_start_full:resp_end_full]
            crit_len = int(critique.shape[0])
            resp_len = int(response.shape[0])
            seg_len = crit_len + resp_len
            if crit_len <= 0 or resp_len <= 0:
                continue
            ctx_end = resp_start_full
            layouts.append(
                TurnLayout(ctx_end=ctx_end, seg_len=seg_len, crit_len=crit_len, resp_len=resp_len)
            )
            packed_parts.append(torch.cat([critique, response], dim=0))
            pos_parts.append(torch.arange(ctx_end, ctx_end + seg_len, device=device, dtype=torch.long))
            response_slices.append((turn, cursor + crit_len, cursor + seg_len))
            cursor += seg_len

        if not layouts:
            return out

        packed_ids = torch.cat(packed_parts, dim=0).unsqueeze(0)
        position_ids = torch.cat(pos_parts, dim=0).unsqueeze(0)

        with torch.no_grad():
            baseline = self.model(traj.unsqueeze(0), use_cache=True)
        kv_cache = [
            (k.detach().clone(), v.detach().clone())
            for k, v in baseline.past_key_values
        ]
        first_k = kv_cache[0][0]
        K_first_pages, _, _ = kv_to_pages(
            first_k.to(device=device),
            kv_cache[0][1].to(device=device),
            page_size=self.page_size,
        )
        page_tables = build_page_tables(
            layouts,
            n_traj_pages=K_first_pages.shape[0],
            page_size=self.page_size,
            device=device,
        )

        layer_payloads = []
        for K_layer, V_layer in kv_cache:
            K_layer = K_layer.to(device=device)
            V_layer = V_layer.to(device=device)
            K_pages, V_pages, L_orig = kv_to_pages(K_layer, V_layer, page_size=self.page_size)
            layer_payloads.append(
                {
                    "K_traj_pages": K_pages,
                    "V_traj_pages": V_pages,
                    "K_traj_flat": K_layer.squeeze(0).permute(1, 0, 2).contiguous(),
                    "V_traj_flat": V_layer.squeeze(0).permute(1, 0, 2).contiguous(),
                    "traj_kv_len": L_orig,
                }
            )

        payload = {
            "block_table": page_tables.block_table,
            "cache_seqlens": page_tables.cache_seqlens,
            "cu_seqlens_q": page_tables.cu_seqlens_q,
            "stub_layout": page_tables.stub_layout,
            "turns": layouts,
            "page_size": self.page_size,
            "use_fa2": self.use_fa2,
            "layers": layer_payloads,
        }

        old_impl = self._get_attn_impl(self.model)
        self._set_attn_impl(self.model, "capa")
        try:
            with torch.no_grad():
                packed_out = self.model(
                    packed_ids,
                    attention_mask={"full_attention": None, "sliding_attention": None},
                    position_ids=position_ids,
                    use_cache=False,
                    capa_payload=payload,
                )
        finally:
            self._set_attn_impl(self.model, old_impl)

        logits = packed_out.logits.to(dtype=self.dtype)
        for turn, resp_lo, resp_hi in response_slices:
            score_logits = logits[0, resp_lo - 1 : resp_hi - 1, :]
            target = packed_ids[0, resp_lo:resp_hi]
            cond_lp = F.log_softmax(score_logits, dim=-1).gather(
                -1, target.unsqueeze(-1)
            ).squeeze(-1)
            out[turn.response_start : turn.response_end] = cond_lp.detach().to("cpu")

        self.last_used_packed = True
        return out

    def __call__(
        self,
        trajectory_tokens: torch.Tensor,
        prompt_length: int,
        turns: Sequence[TurnAnnotation],
    ) -> torch.Tensor:
        self.last_used_packed = False
        if self._supports_qwen2_packed():
            return self._call_qwen2_packed(trajectory_tokens, prompt_length, turns)
        return self._reference(trajectory_tokens, prompt_length, turns)
