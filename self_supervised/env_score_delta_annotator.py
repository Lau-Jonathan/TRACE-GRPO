"""Self-supervised env-score-delta teacher.

Implements the production-recommended teacher path from
the paper §7.1: read ``q_t`` straight off the env reward signal,
no LLM API call, ``Pearson(q, outcome) ≡ 1`` by construction.

Per-turn signal (spec literal):

    q_t = sign(score_t − score_{t-1}) ∈ {−1, 0, +1}

Per-turn critique text (templated):

    delta > 0  → "This action advanced env score by {delta} points."
    delta < 0  → "This action triggered env penalty {delta}."
    delta == 0 → "No env-score progress this turn."

Aggregation to a per-trajectory scalar ``per_sample_q`` (consumed by
``trace_l3_mask_compute``): we use the **sum** of turn-level q values,
which matches the ``group_q_sum`` decision used by the L1 gate (the gate
compares ``sign(group_mean) ↔ sign(group_q_sum)``, so propagating the
sum to the per-sample slot keeps the L1 logic literal).

Note: this teacher does *not* run an LLM forward — it only mines the env
score deltas captured during rollout. The trajectory tokens are needed
only for the L3 critique-conditioned forward, not for the q itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from trace_grpo.patches.critique_conditioned_provider import (
    TrajectoryAnnotation,
    TurnAnnotation,
)


# ---------------------------------------------------------------------------
# Public input / output types.
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """One assistant turn captured during rollout.

    Mirrors the per-turn metadata produced by an AgentLoopBase subclass and
    stored in ``AgentLoopOutput.extra_fields["assistant_turn_spans"]``.

    Attributes:
        history_index: 0-based turn index within the trajectory.
        response_start: response-axis position of the first token of this
            turn's assistant response. (Same frame as
            :class:`~trace_grpo.patches.critique_conditioned_provider.TurnAnnotation`.)
        response_end: exclusive end.
        score: env score *after* the env step that closed this turn.
            ScienceWorld's ``info["score"]`` (an integer in [-100, 100]).
        has_format_error: True if the assistant response failed to parse
            (no ``<action>`` tag etc.). Format-error turns get a -0.5
            BEACON penalty and force ``w² = 1`` in L2.
        action_text: the raw assistant response (decoded from
            ``response_ids``) — needed by the LLM-judge teacher to
            describe to the judge what the agent did. Optional;
            env-score teacher does not use it.
        env_response_text: env feedback text after this action — also
            consumed by the LLM judge to ground its decision.
    """

    history_index: int
    response_start: int
    response_end: int
    score: float
    has_format_error: bool = False
    action_text: str = ""
    env_response_text: str = ""


@dataclass
class TrajectoryRecord:
    """One full trajectory captured during rollout.

    Attributes:
        traj_token_ids: full ``[s_0, a_1, ..., s_T, a_T]`` token stream.
        prompt_length: number of prompt tokens at the front.
        response_length: response axis length in the verl batch.
        traj_offset_in_batch: row index of this trajectory in the batch.
        score_initial: env score before the first turn (usually 0).
        turns: ordered list of :class:`TurnRecord`.
        task_description: the env task prompt (what the agent is asked to
            do). Used by the LLM judge to ground its assessment of each
            turn against the global goal.
        initial_observation: the starting env observation. Optional.
        won: platform outcome for outcome-consistency gating when available.
    """

    traj_token_ids: List[int]
    prompt_length: int
    response_length: int
    traj_offset_in_batch: int
    score_initial: float
    turns: List[TurnRecord]
    task_description: str = ""
    initial_observation: str = ""
    won: Optional[bool] = None


# ---------------------------------------------------------------------------
# Templated critique strings (spec PDF §7.1).
# ---------------------------------------------------------------------------


def env_score_critique(prev_score: float, cur_score: float) -> str:
    """Reproduce the critique template printed in spec §7.1 verbatim."""
    delta = cur_score - prev_score
    if delta > 0:
        return f"This action advanced env score by {delta:g} points."
    if delta < 0:
        return f"This action triggered env penalty {delta:g}."
    return "No env-score progress this turn."


def wrap_teacher_note(critique: str) -> str:
    """Wrap a critique string in the ``<teacher_note>...</teacher_note>``
    block that gets spliced before the assistant response during the
    L3 conditioned forward."""
    return f"<teacher_note>{critique}</teacher_note>"


# ---------------------------------------------------------------------------
# Teacher implementation.
# ---------------------------------------------------------------------------


@dataclass
class EnvScoreDeltaAnnotator:
    """Teacher that derives q_t and critique text from env score deltas.

    Args:
        tokenizer: HF-compatible tokenizer (only ``encode`` is used). May
            be ``None`` if L3 is not enabled — in which case the
            ``trajectories`` output is set to ``None`` and the
            critique-conditioned forward is skipped.
        l3_enable: if True, emit per-turn ``critique_token_ids`` so the
            L3 provider can splice them. If False, ``trajectories`` is
            ``None`` and only the per-sample q is produced.
    """

    tokenizer: Optional[object] = None
    l3_enable: bool = True
    emit_turn_annotations: bool = False
    per_sample_q_aggregator: str = "sum"

    # -- spec §3.2: per-trajectory dict aggregator -------------------------

    @staticmethod
    def per_turn_q_from_score(prev_score: float, cur_score: float) -> int:
        """``sign(score_t − score_{t-1}) ∈ {−1, 0, +1}`` (Eq. 4)."""
        delta = cur_score - prev_score
        if delta > 0:
            return 1
        if delta < 0:
            return -1
        return 0

    def _aggregate_per_sample_q(self, q_values: List[float]) -> float:
        if self.per_sample_q_aggregator == "sum":
            return float(sum(q_values))
        if self.per_sample_q_aggregator == "sign_sum":
            total = float(sum(q_values))
            return float(np.sign(total))
        raise ValueError(
            f"Unsupported per_sample_q_aggregator={self.per_sample_q_aggregator!r}; "
            "expected 'sum' or 'sign_sum'."
        )

    # -- public API consumed by the trainer hook ---------------------------

    def annotate(
        self,
        batch,
    ) -> tuple[np.ndarray, Optional[List[TrajectoryAnnotation]]]:
        """Compute ``per_sample_q`` and (optionally) build trajectory
        annotations for the L3 conditioned forward.

        ``batch`` is expected to expose a ``non_tensor_batch`` mapping with
        the key ``"trajectory_records"``: a sequence of
        :class:`TrajectoryRecord` (one per row of the batch). The
        AgentLoopBase subclass populates this during rollout.

        Returns:
            (per_sample_q, trajectories) where:
              - ``per_sample_q`` is a ``(bs,)`` float32 ndarray; entry
                ``i`` is the sum of ``q_t`` across turns of trajectory
                ``i`` (the per-trajectory aggregation that L1's
                ``group_q_sum`` consumes).
              - ``trajectories`` is a list of
                :class:`TrajectoryAnnotation` if ``l3_enable`` and at
                least one turn produced a non-zero q; otherwise ``None``.
        """
        records: Sequence[TrajectoryRecord] = batch.non_tensor_batch.get(
            "trajectory_records"
        )
        if records is None:
            raise KeyError(
                "EnvScoreDeltaAnnotator: batch.non_tensor_batch is missing "
                "'trajectory_records' — make sure the agent loop populated it"
            )

        bs = len(records)
        per_sample_q = np.zeros(bs, dtype=np.float32)
        trajectories: List[TrajectoryAnnotation] = []

        for traj in records:
            row = traj.traj_offset_in_batch
            if row < 0 or row >= bs:
                raise IndexError(
                    f"trajectory offset {row} out of [0, {bs})"
                )
            prev_score = traj.score_initial
            turn_anns: List[TurnAnnotation] = []
            q_values: List[float] = []
            for turn in traj.turns:
                q = self.per_turn_q_from_score(prev_score, turn.score)
                q_values.append(float(q))
                if self.l3_enable and self.tokenizer is not None:
                    critique = env_score_critique(prev_score, turn.score)
                    critique_ids = self.tokenizer.encode(
                        wrap_teacher_note(critique),
                        add_special_tokens=False,
                    )
                else:
                    critique_ids = []
                turn_anns.append(
                    TurnAnnotation(
                        history_index=turn.history_index,
                        response_start=turn.response_start,
                        response_end=turn.response_end,
                        critique_token_ids=critique_ids,
                        q_t=float(q),
                        has_format_error=bool(turn.has_format_error),
                    )
                )
                prev_score = turn.score
            per_sample_q[row] = self._aggregate_per_sample_q(q_values)

            if any(t.critique_token_ids for t in turn_anns) or (
                self.emit_turn_annotations and turn_anns
            ):
                trajectories.append(
                    TrajectoryAnnotation(
                        traj_token_ids=traj.traj_token_ids,
                        prompt_length=traj.prompt_length,
                        response_length=traj.response_length,
                        turns=turn_anns,
                        traj_offset_in_batch=row,
                    )
                )

        return per_sample_q, (trajectories if trajectories else None)
