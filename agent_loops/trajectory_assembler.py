"""Pure trajectory-assembly logic for TRACE-GRPO agent loops.

An AgentLoopBase subclass (e.g. ScienceWorldAgentLoop) interleaves three
side-effecting concerns:

  1. **env interactions** — ``env.reset() / env.step()``, parsing the
     assistant response, scoring, etc.
  2. **inference** — calling vLLM / sglang to generate the next assistant
     turn.
  3. **trajectory bookkeeping** — flat token stream, response_mask,
     per-turn span records, shaped reward accumulation.

This module isolates (3) so it can be unit-tested without scienceworld,
without an LLM, without verl. The :class:`TrajectoryAssembler` is fed
``begin_turn(...)`` / ``record_response(...)`` / ``record_observation(...)``
calls and produces the final
:class:`~trace_grpo.self_supervised.env_score_delta_annotator.TrajectoryRecord`
plus an :class:`AssembledTrajectory` (which the AgentLoopBase wraps into
an ``AgentLoopOutput``).

Design notes:
  - Token positions are tracked in the **trajectory-flat frame** during
    assembly. ``response_start / response_end`` exposed to downstream
    components are converted to the **response-axis frame** (subtract
    ``prompt_length``) so they match the verl ``response_mask`` layout
    consumed by :class:`TurnAnnotation`.
  - The assembler does *not* tokenize — the caller hands it pre-tokenized
    ids. This keeps the unit tests pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from trace_grpo.patches.sciworld_reward_manager import (
    ShapedTrajectoryReward,
    StepRecord,
    shape_trajectory_reward,
)
from trace_grpo.self_supervised.env_score_delta_annotator import (
    TrajectoryRecord,
    TurnRecord,
)


# Type of an injectable trajectory-level reward shaper.
# Signature mirrors :func:`shape_trajectory_reward` so callers can pass either
# the default sciworld/alfworld-shared shaper or an environment-specific one.
ShaperFn = Callable[..., ShapedTrajectoryReward]


# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------


@dataclass
class AssembledTrajectory:
    """Bundle returned by :meth:`TrajectoryAssembler.finalize`.

    Attributes:
        prompt_ids: token ids of the system + initial user prompt (the
            piece an AgentLoopBase subclass writes into
            ``AgentLoopOutput.prompt_ids``).
        response_ids: everything *after* the prompt — interleaved
            assistant responses and env-feedback user turns. Padded with
            zeros up to ``response_length`` if the trajectory is shorter
            than the budget.
        response_mask: ``len(response_ids)`` 0/1 mask, 1 on tokens emitted
            by the actor. Matches verl's ``response_mask`` convention.
        trajectory_reward: scalar reward fed to GRPO (BEACON-shaped sum).
        trajectory_reward_no_format_penalty: alias of
            ``trajectory_reward`` (no format penalty exists in the
            BEACON-aligned shaper). Kept for backwards-compat plumbing.
        won: ``done ∧ score > 0`` of the last step.
        num_turns: number of assistant turns recorded.
        num_invalid_turns: count of turns that failed format parsing;
            consumed by the trainer-level invalid-action penalty.
        trajectory_record: the rich :class:`TrajectoryRecord` for the
            env-score teacher / L3 provider.
    """

    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    trajectory_reward: float
    trajectory_reward_no_format_penalty: float
    won: bool
    num_turns: int
    num_invalid_turns: int
    trajectory_record: TrajectoryRecord


# ---------------------------------------------------------------------------
# Assembler.
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryAssembler:
    """Stateful trajectory builder.

    Lifecycle::

        a = TrajectoryAssembler(prompt_ids=..., response_length=512,
                                traj_offset_in_batch=3, score_initial=0)
        for turn in range(N):
            a.record_response(response_ids=[...], score=12, done=False,
                              has_format_error=False)
            a.record_observation(obs_ids=[...])
        traj = a.finalize(is_train=True)

    The class deliberately doesn't drive the env / model loop — the
    AgentLoopBase subclass owns control flow and just hands tokens here.

    Args:
        prompt_ids: pre-tokenized system + initial user prompt.
        response_length: budget for the response axis. Trailing slots
            beyond the actual trajectory get zero-padded; the response
            mask stays 0 in those slots so downstream losses ignore them.
        traj_offset_in_batch: row index of this trajectory in the verl
            batch (used by the env-score teacher to scatter δ back).
        score_initial: env score before the first step (usually 0).
    """

    prompt_ids: List[int]
    response_length: int
    traj_offset_in_batch: int
    score_initial: float = 0.0
    pad_token_id: int = 0

    response_ids: List[int] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    turn_records: List[TurnRecord] = field(default_factory=list)
    step_records: List[StepRecord] = field(default_factory=list)
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.response_length <= 0:
            raise ValueError(f"response_length must be > 0, got {self.response_length}")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")

    # -- internal -----------------------------------------------------------

    @property
    def _budget_remaining(self) -> int:
        return self.response_length - len(self.response_ids)

    def _append(self, ids: List[int], is_response: bool) -> tuple[int, int, bool]:
        """Append ``ids`` to the response stream up to the response budget.

        Returns ``(start, end, dropped)``. ``dropped`` is True if any ids
        could not fit (the caller may want to skip recording a turn whose
        tokens never went into training).
        Sets ``truncated=True`` if any ids are dropped.
        """
        start = len(self.response_ids)
        if self._budget_remaining <= 0:
            self.truncated = True
            return start, start, True
        dropped = len(ids) > self._budget_remaining
        if dropped:
            ids = ids[: self._budget_remaining]
            self.truncated = True
        mask_val = 1 if is_response else 0
        self.response_ids.extend(ids)
        self.response_mask.extend([mask_val] * len(ids))
        end = len(self.response_ids)
        return start, end, dropped

    # -- public API ---------------------------------------------------------

    def record_response(
        self,
        response_ids: List[int],
        *,
        score: float,
        done: bool,
        has_format_error: bool = False,
    ) -> bool:
        """Record one assistant turn + the env step that closed it.

        Returns True if the turn was fully recorded; False if the response
        budget was exceeded so the turn was skipped. The caller should
        break out of the rollout loop on a False return — the env had
        already produced ``score/done``, but those would otherwise enter
        the trajectory baseline for tokens we never trained on.
        """
        if not response_ids and not has_format_error:
            # Allow empty responses only when has_format_error=True (e.g.
            # the parser failed and we still need to record the step).
            raise ValueError("record_response: response_ids must be non-empty")
        if self._budget_remaining <= 0 or len(response_ids) > self._budget_remaining:
            # Budget can't fit this turn fully. Skipping prevents:
            #   (1) a phantom score delta polluting shape_trajectory_reward
            #       on tokens the actor never emitted, and
            #   (2) PPO computing advantage on partial-turn tokens whose
            #       response_mask=1 but which have no corresponding
            #       TurnRecord/annotation.
            self.truncated = True
            return False
        history_index = len(self.turn_records)
        start, end, _dropped = self._append(response_ids, is_response=True)
        self.turn_records.append(
            TurnRecord(
                history_index=history_index,
                response_start=start,
                response_end=end,
                score=float(score),
                has_format_error=bool(has_format_error),
            )
        )
        self.step_records.append(
            StepRecord(
                score=float(score),
                done=bool(done),
                has_format_error=bool(has_format_error),
            )
        )
        return True

    def record_observation(self, obs_ids: List[int]) -> None:
        """Record the env feedback (next user turn) tokens after a step.

        These tokens go into the trajectory but get response_mask=0 so
        they aren't scored as actor output.
        """
        if not obs_ids:
            return
        self._append(obs_ids, is_response=False)

    def finalize(
        self,
        *,
        is_train: bool = True,
        shaper: Optional[ShaperFn] = None,
    ) -> AssembledTrajectory:
        """Pad response axis to the budget, compute shaped reward, build
        the trajectory record, and return the bundle.

        Args:
            is_train: forwarded to the shaper (currently unused by the
                BEACON-aligned shapers; kept for API stability).
            shaper: optional environment-specific reward shaper. If not
                provided, falls back to the sciworld/alfworld-shared
                :func:`shape_trajectory_reward` (BEACON sciworld formula
                ``r = 1[score↑] + 10[done∧score>0]``). AlfWorld text-only
                runs override this with a 10*won-only shaper to match
                GiGPO baseline exactly (no score-up term, since text-only
                AlfredTWEnv only emits a binary ``info['won']``).
        """
        # Pad to response_length so the verl batch tensor has fixed shape.
        pad_len = self.response_length - len(self.response_ids)
        if pad_len > 0:
            self.response_ids.extend([self.pad_token_id] * pad_len)
            self.response_mask.extend([0] * pad_len)

        shape_fn = shaper if shaper is not None else shape_trajectory_reward
        shaped = shape_fn(
            self.step_records,
            score_initial=self.score_initial,
            is_train=is_train,
        )

        traj_token_ids = self.prompt_ids + self.response_ids
        record = TrajectoryRecord(
            traj_token_ids=traj_token_ids,
            prompt_length=len(self.prompt_ids),
            response_length=self.response_length,
            traj_offset_in_batch=self.traj_offset_in_batch,
            score_initial=self.score_initial,
            turns=list(self.turn_records),
            won=shaped.won,
        )

        return AssembledTrajectory(
            prompt_ids=list(self.prompt_ids),
            response_ids=list(self.response_ids),
            response_mask=list(self.response_mask),
            trajectory_reward=shaped.trajectory_reward,
            trajectory_reward_no_format_penalty=shaped.trajectory_reward_no_format_penalty,
            won=shaped.won,
            num_turns=len(self.turn_records),
            num_invalid_turns=shaped.num_invalid_turns,
            trajectory_record=record,
        )
