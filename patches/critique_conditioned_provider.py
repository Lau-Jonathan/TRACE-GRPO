"""L3 critique-conditioned forward provider (``critique_v1``).

For each trajectory in the batch, builds a CAPA payload that delivers — in
**one** packed forward — the conditioned log-probs

    log p_θ(response_token_k | prefix_k, <teacher_note>critique_t</teacher_note>)

for every annotated turn. Subtracting verl's already-computed
``old_log_probs`` produces ``critique_delta (bs, response_length)`` which is
exactly what L3 in :mod:`trace_grpo.patches.level3_patch` consumes.

Design choices:

  1. **Decoupled from model internals**. The provider takes a
     :class:`CritiqueConditionedForwardFn` callable, so we can unit-test the
     scheduling logic with a deterministic mock forward and *separately*
     swap in the real CAPA-backed forward when integrating with verl's
     actor worker.
  2. **Trajectories with no annotated turns produce δ ≡ 0**. We skip the
     entire CAPA payload and write zeros — this matches the desired
     no-op behaviour (and saves real cost in production).
  3. **Annotated turns produce δ only at *response* token positions**;
     critique tokens themselves do not appear in the response so they are
     never queried. Unannotated turns inside an otherwise-annotated
     trajectory keep δ=0 on their response positions (we simply leave
     those rows of the per-turn output untouched).

The class is registered as ``critique_v1`` via the registry in
:mod:`trace_grpo.patches.conditioned_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from .conditioned_forward import register_delta_provider


# ---------------------------------------------------------------------------
# Public data types.
# ---------------------------------------------------------------------------


@dataclass
class TurnAnnotation:
    """Per-turn annotation produced by a teacher.

    All position fields live in the **response-axis frame**: index 0 is the
    first response token of the trajectory, indices grow rightward, and
    valid values are in ``[0, response_length)``. The provider translates
    response-axis positions to trajectory-flat positions by adding
    ``prompt_length`` (stored on :class:`TrajectoryAnnotation`).

    Attributes:
        history_index: the position of this turn in the trajectory's
            multi-turn chat list (``0`` is the first assistant turn).
        response_start: response-axis index of the first response token
            for this turn (verl convention; matches ``response_mask``).
        response_end: exclusive end. ``response_end - response_start ==
            len(response_t)``.
        critique_token_ids: token ids for ``<teacher_note>critique_t
            </teacher_note>``. Empty list ⇒ unannotated turn (δ stays 0).
        q_t: the per-turn judgment in [-1, +1]. Unused by this provider but
            stored alongside for downstream telemetry.
        has_format_error: whether this assistant turn violated the required
            ``<think>...<action>...`` format. L2 uses this to pin ``w²=1``
            only on the affected turn spans (spec §5.2).
    """

    history_index: int
    response_start: int
    response_end: int
    critique_token_ids: List[int]
    q_t: float = 0.0
    has_format_error: bool = False


@dataclass
class TrajectoryAnnotation:
    """Per-trajectory bundle of teacher annotations.

    Attributes:
        traj_token_ids: full flat token id list ``[s_0, a_1, ...]``,
            including the prompt and all response/observation tokens.
        prompt_length: number of prompt tokens at the front of
            ``traj_token_ids``. Response axis frame starts at this offset.
        response_length: length of the response axis in the verl batch
            tensors (``token_level_rewards.shape[1]``). Per-turn
            ``response_start/end`` must be in ``[0, response_length)``.
        turns: per-turn annotations, in trajectory order.
        traj_offset_in_batch: which row of the batch this trajectory
            occupies (0-based). Used to scatter δ back into the
            ``(bs, response_length)`` output.
    """

    traj_token_ids: List[int]
    prompt_length: int
    response_length: int
    turns: List[TurnAnnotation]
    traj_offset_in_batch: int


# ---------------------------------------------------------------------------
# Forward-function contract.
# ---------------------------------------------------------------------------


class CritiqueConditionedForwardFn:
    """Callable that takes the conditioned-forward payload and returns
    log-probs of the response tokens under that conditioning.

    Concrete implementations:
      - :class:`MockCritiqueForward` — deterministic stand-in for tests.
      - The CAPA-backed implementation, wired up in the actor worker
        (lives next to verl's :class:`DataParallelPPOActor`).

    Contract (so unit tests and the real version stay interchangeable):
      Inputs:
        - ``trajectory_tokens``: 1D ``LongTensor`` of length ``L`` (the
          full trajectory: prompt + responses + env observations).
        - ``prompt_length``: number of prompt tokens at the front of
          ``trajectory_tokens``. Implementations need this to translate
          turn position fields (which are in the response-axis frame) to
          absolute trajectory positions for RoPE / attention.
        - ``turns``: list of :class:`TurnAnnotation` with critique tokens
          populated; empty-critique entries are filtered out by the
          caller.
      Output:
        - ``cond_logprobs``: 1D ``FloatTensor`` of length
          ``L - prompt_length`` — i.e. **lives in the response-axis
          frame**, so position 0 is the first response/observation token.
          Only positions inside an annotated turn's response span are
          required to be valid; the caller masks the rest.
    """

    def __call__(
        self,
        trajectory_tokens: torch.Tensor,
        prompt_length: int,
        turns: Sequence[TurnAnnotation],
    ) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Provider implementation.
# ---------------------------------------------------------------------------


@register_delta_provider("critique_v1")
class CritiqueConditionedProvider:
    """Build ``critique_delta (bs, response_length)`` from per-trajectory
    annotations + per-trajectory baseline log-probs.

    Stateless aside from the injected forward function. Intended usage::

        provider = CritiqueConditionedProvider(forward_fn=actor_critique_fwd)
        delta = provider.compute_critique_delta(
            trajectories=annotated_trajectories,
            old_logprobs=old_logprobs_tensor,    # (bs, response_length)
        )

    The shape, dtype, and device of the returned tensor exactly match
    ``old_logprobs`` so the caller can subtract them for free.
    """

    def __init__(self, forward_fn: CritiqueConditionedForwardFn):
        self.forward_fn = forward_fn

    # -- public ---------------------------------------------------------------

    def compute_critique_delta(
        self,
        trajectories: Sequence[TrajectoryAnnotation],
        old_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``δ = cond_logprobs − old_logprobs`` for every annotated
        response token.

        Args:
            trajectories: one entry per *trajectory* in the batch. Each
                lists its own annotated turns.
            old_logprobs: ``(bs, response_length)`` baseline log-probs from
                verl's ``compute_log_prob`` step. Indexed by trajectory
                offset, *not* turn — i.e. row ``i`` is the entire packed
                response of trajectory ``i``.

        Returns:
            ``critique_delta`` of the same shape, dtype, and device as
            ``old_logprobs``. Non-annotated tokens are 0.
        """
        if old_logprobs.ndim != 2:
            raise ValueError(
                f"old_logprobs must be (bs, response_length), got {tuple(old_logprobs.shape)}"
            )

        bs, response_length = old_logprobs.shape
        delta = torch.zeros_like(old_logprobs)

        for traj in trajectories:
            if traj.traj_offset_in_batch < 0 or traj.traj_offset_in_batch >= bs:
                raise IndexError(
                    f"trajectory offset {traj.traj_offset_in_batch} out of range for "
                    f"batch size {bs}"
                )
            if traj.response_length != response_length:
                raise ValueError(
                    f"trajectory {traj.traj_offset_in_batch} has response_length "
                    f"{traj.response_length} != batch response_length {response_length}"
                )

            annotated = [t for t in traj.turns if len(t.critique_token_ids) > 0]
            if not annotated:
                continue  # leave row zero

            # Bounds-check turn spans against response_length.
            for turn in annotated:
                if turn.response_start < 0 or turn.response_end > response_length:
                    raise IndexError(
                        f"turn (history_index={turn.history_index}) span "
                        f"[{turn.response_start}, {turn.response_end}) is out of "
                        f"response axis [0, {response_length})"
                    )
                if turn.response_end <= turn.response_start:
                    raise ValueError(
                        f"turn (history_index={turn.history_index}) has empty span"
                    )

            traj_tokens = torch.as_tensor(traj.traj_token_ids, dtype=torch.long)
            expected_resp_len = traj_tokens.shape[0] - traj.prompt_length
            cond_logprobs_resp = self.forward_fn(
                trajectory_tokens=traj_tokens,
                prompt_length=traj.prompt_length,
                turns=annotated,
            )
            if (
                cond_logprobs_resp.ndim != 1
                or cond_logprobs_resp.shape[0] != expected_resp_len
            ):
                raise RuntimeError(
                    f"forward_fn must return a 1D tensor of length L - prompt_length "
                    f"({expected_resp_len}); got {tuple(cond_logprobs_resp.shape)}"
                )

            row = traj.traj_offset_in_batch
            for turn in annotated:
                resp_start = turn.response_start
                resp_end = turn.response_end
                cond_seg = cond_logprobs_resp[resp_start:resp_end]
                old_seg = old_logprobs[row, resp_start:resp_end]
                delta[row, resp_start:resp_end] = (
                    cond_seg.to(device=delta.device, dtype=delta.dtype) - old_seg
                )

        return delta


# ---------------------------------------------------------------------------
# A deterministic mock forward suitable for unit tests.
# ---------------------------------------------------------------------------


class MockCritiqueForward(CritiqueConditionedForwardFn):
    """Returns a fixed bias for response tokens of annotated turns.

    Useful for verifying the provider's *scheduling* (which turns get
    forwarded, which positions get written, what shape comes back) without
    running a real model. The bias for turn ``t`` is keyed by
    ``turn.history_index`` so different turns can produce distinguishable
    log-probs.
    """

    def __init__(self, biases: Optional[dict[int, float]] = None, default_bias: float = 0.0):
        self.biases = biases or {}
        self.default_bias = default_bias

    def __call__(
        self,
        trajectory_tokens: torch.Tensor,
        prompt_length: int,
        turns: Sequence[TurnAnnotation],
    ) -> torch.Tensor:
        L = trajectory_tokens.shape[0]
        out = torch.zeros(L - prompt_length, dtype=torch.float32)
        for turn in turns:
            b = self.biases.get(turn.history_index, self.default_bias)
            out[turn.response_start : turn.response_end] = b
        return out
