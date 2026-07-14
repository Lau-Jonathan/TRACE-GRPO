"""BEACON-aligned shaped reward for ScienceWorld.

Implements BEACON's ``SciWorldWorker.step`` shaping (envs.py:91-113):

    r_t = 1[score_t > score_{t-1}] * 1.0
        + 1[done_t ∧ score_t > 0] * 10.0

Notes:
  - ``score`` is ScienceWorld's ``info["score"]`` (an integer, range
    [-100, 100]; -100 on catastrophic failure forces ``isCompleted=True``).
  - The first +1 fires on **strict** score increase, not non-decrease.
  - The +10 terminal bonus only fires on ``done ∧ score > 0`` so a
    catastrophic-failure terminal (-100) does *not* get the bonus.
  - There is **no** format-error penalty in the trajectory reward. BEACON
    has no such concept; format-quality enforcement is handled at the
    token-level by ``apply_invalid_action_penalty`` in the trainer
    (BEACON/verl/trainer/ppo/ray_trainer.py:201). To stay strictly
    BEACON-aligned the per-step shaped reward must therefore not depend
    on ``has_format_error``.

Trajectory-level reward fed to GRPO is ``R_i = sum_t r_t``.

This module is intentionally pure: it operates on lightweight
:class:`StepRecord` dataclasses without depending on verl's DataProto so
it can be exercised by unit tests without any heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Public types.
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Per-env-step record produced by the agent loop.

    Attributes:
        score: env score *after* this step (i.e. ``info["score"]``).
        done: env termination flag for this step.
        has_format_error: True if the assistant response failed to parse.
            BEACON exposes the same flag as ``info["is_action_valid"]``;
            it is consumed by the trainer-level invalid-action penalty,
            not by the trajectory shaper.
    """

    score: float
    done: bool
    has_format_error: bool = False


@dataclass
class ShapedTrajectoryReward:
    """Result of shaping one trajectory.

    Attributes:
        per_step: shaped reward at each step ``r_t``.
        trajectory_reward: ``sum_t r_t`` (the scalar fed to GRPO).
        trajectory_reward_no_format_penalty: alias of ``trajectory_reward``
            kept for backwards-compatible plumbing. There is no format
            penalty in the BEACON-aligned shaped reward, so this field
            always equals ``trajectory_reward``.
        won: BEACON-aligned win criterion ``done ∧ score > 0`` of the
            *last* step.
        num_invalid_turns: count of turns flagged as ``has_format_error``.
            Surfaced so the trainer can apply BEACON's per-step
            invalid-action penalty on the packed trajectory's last token.
    """

    per_step: List[float] = field(default_factory=list)
    trajectory_reward: float = 0.0
    trajectory_reward_no_format_penalty: float = 0.0
    won: bool = False
    num_invalid_turns: int = 0


# ---------------------------------------------------------------------------
# Shaping function.
# ---------------------------------------------------------------------------


# Spec-named constants kept hoist-able so unit tests can monkey-patch.
SHAPED_REWARD_SCORE_INCREASE: float = 1.0
SHAPED_REWARD_TERMINAL_BONUS: float = 10.0


def shape_trajectory_reward(
    steps: List[StepRecord],
    *,
    score_initial: float = 0.0,
    is_train: bool = True,
) -> ShapedTrajectoryReward:
    """Apply BEACON shaping to one trajectory.

    Args:
        steps: ordered per-step records.
        score_initial: env score *before* the first step (usually 0).
        is_train: kept for API compatibility; the BEACON-aligned shaper
            is identical for train and val (no format penalty).

    Returns:
        :class:`ShapedTrajectoryReward`. Empty trajectories return a
        zero-filled record (no error).
    """
    del is_train  # BEACON shaping has no train/val split
    out = ShapedTrajectoryReward()
    prev = score_initial
    for step in steps:
        r = 0.0
        if step.score > prev:
            r += SHAPED_REWARD_SCORE_INCREASE
        if step.done and step.score > 0:
            r += SHAPED_REWARD_TERMINAL_BONUS
        out.per_step.append(r)
        out.trajectory_reward += r
        if step.has_format_error:
            out.num_invalid_turns += 1
        prev = step.score
    out.trajectory_reward_no_format_penalty = out.trajectory_reward

    if steps:
        last = steps[-1]
        out.won = bool(last.done and last.score > 0)
    return out
