"""Counterfactual masked teacher scaffold.

This teacher path mirrors the paper §7.2 runtime contract:
``teacher_kind=counterfactual`` yields per-turn annotations that are later
resolved into ``q_t^cf`` on the actor worker (where the policy model lives).

Driver-side responsibilities here:
  1. Read rollout ``trajectory_records``.
  2. Build :class:`TrajectoryAnnotation` / :class:`TurnAnnotation` skeletons
     (turn spans + format-error flags).
  3. Return placeholder ``per_sample_q`` (all zeros) that will be overwritten
     by actor-side counterfactual scoring before advantage computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from trace_grpo.patches.critique_conditioned_provider import (
    TrajectoryAnnotation,
    TurnAnnotation,
)
from trace_grpo.self_supervised.env_score_delta_annotator import TrajectoryRecord


@dataclass
class CounterfactualMaskAnnotator:
    """Prepare turn-level scaffolding for actor-side counterfactual scoring."""

    tokenizer: Optional[object] = None
    l3_enable: bool = True
    emit_turn_annotations: bool = True
    per_sample_q_aggregator: str = "sum"

    # Marker consumed by trainer hook.
    requires_actor_counterfactual: bool = field(default=True, init=False)

    # Diagnostics (same names as LLM annotator for metric compatibility).
    last_total_requests: int = field(default=0, init=False)
    last_failed_requests: int = field(default=0, init=False)
    last_valid_parsed_requests: int = field(default=0, init=False)
    last_invalid_parsed_requests: int = field(default=0, init=False)
    last_outcome_compared_requests: int = field(default=0, init=False)
    last_outcome_inconsistent_requests: int = field(default=0, init=False)
    last_turns_total: int = field(default=0, init=False)
    last_turns_nonzero_q: int = field(default=0, init=False)
    last_turns_pos_q: int = field(default=0, init=False)
    last_turns_neg_q: int = field(default=0, init=False)
    last_turns_zero_q: int = field(default=0, init=False)
    last_turns_outcome_gated: int = field(default=0, init=False)
    last_trajectory_count: int = field(default=0, init=False)
    last_labels_per_trajectory_mean: float = field(default=0.0, init=False)
    last_labels_per_trajectory_std: float = field(default=0.0, init=False)
    last_labels_per_trajectory_p50: float = field(default=0.0, init=False)
    last_labels_per_trajectory_p95: float = field(default=0.0, init=False)
    last_judgment_tag_counts: dict[str, int] = field(default_factory=dict, init=False)
    last_judgment_tag_total: int = field(default=0, init=False)
    last_mean_q: float = field(default=0.0, init=False)
    last_mean_abs_q: float = field(default=0.0, init=False)
    last_latency_mean_s: float = field(default=0.0, init=False)
    last_latency_p95_s: float = field(default=0.0, init=False)
    last_latency_max_s: float = field(default=0.0, init=False)

    def annotate(
        self,
        batch,
    ) -> tuple[np.ndarray, Optional[List[TrajectoryAnnotation]]]:
        records: Sequence[TrajectoryRecord] = batch.non_tensor_batch.get("trajectory_records")
        if records is None:
            raise KeyError(
                "CounterfactualMaskAnnotator: batch.non_tensor_batch is missing "
                "'trajectory_records' — make sure the agent loop populated it"
            )

        bs = len(records)
        per_sample_q = np.zeros(bs, dtype=np.float32)
        trajectories: List[TrajectoryAnnotation] = []

        self.last_total_requests = bs
        self.last_failed_requests = 0
        self.last_valid_parsed_requests = bs
        self.last_invalid_parsed_requests = 0
        self.last_outcome_compared_requests = 0
        self.last_outcome_inconsistent_requests = 0
        self.last_turns_total = 0
        self.last_turns_nonzero_q = 0
        self.last_turns_pos_q = 0
        self.last_turns_neg_q = 0
        self.last_turns_zero_q = 0
        self.last_turns_outcome_gated = 0
        self.last_trajectory_count = bs
        self.last_judgment_tag_counts.clear()
        self.last_judgment_tag_total = 0
        self.last_mean_q = 0.0
        self.last_mean_abs_q = 0.0
        self.last_latency_mean_s = 0.0
        self.last_latency_p95_s = 0.0
        self.last_latency_max_s = 0.0

        per_traj_turn_counts: List[float] = []
        for traj in records:
            row = int(traj.traj_offset_in_batch)
            if row < 0 or row >= bs:
                raise IndexError(f"trajectory offset {row} out of [0, {bs})")
            turn_anns: List[TurnAnnotation] = []
            for turn in traj.turns:
                turn_anns.append(
                    TurnAnnotation(
                        history_index=int(turn.history_index),
                        response_start=int(turn.response_start),
                        response_end=int(turn.response_end),
                        critique_token_ids=[],
                        q_t=0.0,
                        has_format_error=bool(turn.has_format_error),
                    )
                )
            self.last_turns_total += len(turn_anns)
            self.last_turns_zero_q += len(turn_anns)
            per_traj_turn_counts.append(float(len(turn_anns)))

            if turn_anns or self.emit_turn_annotations:
                trajectories.append(
                    TrajectoryAnnotation(
                        traj_token_ids=list(traj.traj_token_ids),
                        prompt_length=int(traj.prompt_length),
                        response_length=int(traj.response_length),
                        turns=turn_anns,
                        traj_offset_in_batch=row,
                    )
                )

        if per_traj_turn_counts:
            arr = np.asarray(per_traj_turn_counts, dtype=np.float32)
            self.last_labels_per_trajectory_mean = float(arr.mean())
            self.last_labels_per_trajectory_std = float(arr.std())
            self.last_labels_per_trajectory_p50 = float(np.percentile(arr, 50))
            self.last_labels_per_trajectory_p95 = float(np.percentile(arr, 95))
        else:
            self.last_labels_per_trajectory_mean = 0.0
            self.last_labels_per_trajectory_std = 0.0
            self.last_labels_per_trajectory_p50 = 0.0
            self.last_labels_per_trajectory_p95 = 0.0

        return per_sample_q, (trajectories if trajectories else None)
