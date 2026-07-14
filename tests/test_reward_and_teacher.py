"""Unit tests for BEACON-aligned reward shaping, trajectory assembly, and
the env-score-delta teacher.

End-to-end the three components form the rollout-time signal stack:

    AgentLoopBase
        ├─ TrajectoryAssembler (record_response / record_observation)
        │   └─ trajectory_record (with TurnRecord + score history)
        └─ shape_trajectory_reward (BEACON Eq. 2)
                ↓
            verl batch with non_tensor_batch["trajectory_records"]
                ↓
            EnvScoreDeltaAnnotator.annotate(batch)
                ↓
            (per_sample_q, trajectories) → trainer hook
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trace_grpo.agent_loops.trajectory_assembler import TrajectoryAssembler
from trace_grpo.patches.sciworld_reward_manager import (
    SHAPED_REWARD_SCORE_INCREASE,
    SHAPED_REWARD_TERMINAL_BONUS,
    StepRecord,
    shape_trajectory_reward,
)
from trace_grpo.self_supervised.env_score_delta_annotator import (
    EnvScoreDeltaAnnotator,
    env_score_critique,
    wrap_teacher_note,
)
from trace_grpo.self_supervised.counterfactual_mask_annotator import (
    CounterfactualMaskAnnotator,
)


# ---------------------------------------------------------------------------
# Reward shaping (BEACON Eq. 2).
# ---------------------------------------------------------------------------


def test_shape_strict_score_increase_only():
    """Non-decrease (equal) does NOT fire the +1; strict increase does."""
    r = shape_trajectory_reward(
        [StepRecord(score=0, done=False), StepRecord(score=0, done=False)],
        score_initial=0,
    )
    assert r.per_step == [0.0, 0.0]
    assert r.trajectory_reward == 0.0


def test_shape_score_increase_fires_plus_one():
    r = shape_trajectory_reward(
        [StepRecord(score=10, done=False), StepRecord(score=20, done=False)],
        score_initial=0,
    )
    assert r.per_step == [SHAPED_REWARD_SCORE_INCREASE, SHAPED_REWARD_SCORE_INCREASE]
    assert r.trajectory_reward == 2.0


def test_shape_terminal_bonus_only_when_score_positive():
    """``done ∧ score > 0`` fires +10; ``done ∧ score < 0`` does not."""
    pos = shape_trajectory_reward(
        [StepRecord(score=50, done=True)], score_initial=0,
    )
    assert pos.trajectory_reward == SHAPED_REWARD_SCORE_INCREASE + SHAPED_REWARD_TERMINAL_BONUS

    neg = shape_trajectory_reward(
        [StepRecord(score=-100, done=True)], score_initial=0,
    )
    # No +1 (score didn't *increase* past 0 — it went down) and no terminal bonus.
    assert neg.trajectory_reward == 0.0


def test_shape_has_no_format_penalty():
    """BEACON-aligned shaper must not apply any format-error penalty.

    Format-quality enforcement lives at the trainer level (apply_invalid
    _action_penalty in verl/trainer/ppo/ray_trainer.py), so the trajectory
    shaper's output must match BEACON's per-step ``+1 / +10`` exactly
    regardless of whether the turn parsed cleanly. The trainer-side
    penalty consumes ``num_invalid_turns`` instead.
    """
    with_err = shape_trajectory_reward(
        [StepRecord(score=10, done=False, has_format_error=True)],
        score_initial=0,
        is_train=True,
    )
    without_err = shape_trajectory_reward(
        [StepRecord(score=10, done=False, has_format_error=False)],
        score_initial=0,
        is_train=True,
    )
    assert with_err.trajectory_reward == SHAPED_REWARD_SCORE_INCREASE
    assert without_err.trajectory_reward == SHAPED_REWARD_SCORE_INCREASE
    # train/val parity: BEACON has no train/val split.
    val = shape_trajectory_reward(
        [StepRecord(score=10, done=False, has_format_error=True)],
        score_initial=0,
        is_train=False,
    )
    assert val.trajectory_reward == SHAPED_REWARD_SCORE_INCREASE
    # The shaper exposes invalid-turn count for the trainer-level penalty.
    assert with_err.num_invalid_turns == 1
    assert without_err.num_invalid_turns == 0
    # ``trajectory_reward_no_format_penalty`` is now an alias of the main reward.
    assert with_err.trajectory_reward_no_format_penalty == with_err.trajectory_reward


def test_shape_won_criterion():
    """``won = done ∧ score > 0`` of the LAST step (BEACON-aligned)."""
    won = shape_trajectory_reward(
        [StepRecord(score=10, done=False), StepRecord(score=50, done=True)],
        score_initial=0,
    )
    assert won.won is True
    lost = shape_trajectory_reward(
        [StepRecord(score=80, done=False), StepRecord(score=-100, done=True)],
        score_initial=0,
    )
    assert lost.won is False


def test_shape_empty_trajectory_returns_zero():
    r = shape_trajectory_reward([], score_initial=0)
    assert r.per_step == []
    assert r.trajectory_reward == 0.0
    assert r.won is False


# ---------------------------------------------------------------------------
# TrajectoryAssembler.
# ---------------------------------------------------------------------------


def test_assembler_basic_two_turns():
    a = TrajectoryAssembler(
        prompt_ids=[7, 8, 9],
        response_length=20,
        traj_offset_in_batch=2,
        score_initial=0,
    )
    a.record_response(response_ids=[10, 11, 12], score=5, done=False)
    a.record_observation(obs_ids=[13, 14])
    a.record_response(response_ids=[15, 16], score=10, done=True)

    traj = a.finalize(is_train=True)
    assert traj.num_invalid_turns == 0

    # Response stream layout:  [10 11 12 | 13 14 | 15 16 | <pad>...]
    # Mask:                     [ 1  1  1 |  0  0 |  1  1 | 0...]
    assert traj.response_ids[:7] == [10, 11, 12, 13, 14, 15, 16]
    assert traj.response_mask[:7] == [1, 1, 1, 0, 0, 1, 1]
    assert all(m == 0 for m in traj.response_mask[7:])
    assert len(traj.response_ids) == 20  # padded to budget
    assert traj.num_turns == 2
    # Reward: t1 score 0→5 (+1), t2 5→10 (+1) + done&score>0 (+10) = 12
    assert traj.trajectory_reward == 12.0
    assert traj.won is True

    # Trajectory record carries the right per-turn spans.
    rec = traj.trajectory_record
    assert rec.prompt_length == 3
    assert rec.response_length == 20
    assert rec.traj_offset_in_batch == 2
    assert rec.score_initial == 0
    assert rec.turns[0].response_start == 0
    assert rec.turns[0].response_end == 3
    assert rec.turns[1].response_start == 5
    assert rec.turns[1].response_end == 7
    assert rec.turns[0].score == 5
    assert rec.turns[1].score == 10


def test_assembler_skips_turn_when_over_budget():
    """If a turn cannot fit fully, it is skipped entirely so the env's
    score/done don't pollute ``shape_trajectory_reward``. ``truncated``
    is set so the runner can break out of the loop."""
    a = TrajectoryAssembler(prompt_ids=[1], response_length=5, traj_offset_in_batch=0)
    ok = a.record_response(response_ids=[10, 11, 12], score=1, done=False)
    assert ok is True
    a.record_observation(obs_ids=[20, 21])  # fills budget exactly
    ok = a.record_response(response_ids=[30, 31], score=2, done=False)  # would not fit
    assert ok is False
    traj = a.finalize()
    assert a.truncated is True
    # Only one TurnRecord — the over-budget turn was skipped, not partially recorded.
    assert len(traj.trajectory_record.turns) == 1


def test_assembler_skips_partial_truncation():
    """If only part of a turn would fit, the whole turn is dropped — partial
    spans would put response_mask=1 on tokens with no TurnRecord, which
    breaks downstream advantage scattering."""
    a = TrajectoryAssembler(prompt_ids=[1], response_length=4, traj_offset_in_batch=0)
    ok = a.record_response(response_ids=[10, 11, 12, 13, 14], score=1, done=False)
    assert ok is False
    traj = a.finalize()
    assert a.truncated is True
    assert len(traj.trajectory_record.turns) == 0


def test_assembler_rejects_empty_response_without_format_error():
    a = TrajectoryAssembler(prompt_ids=[1], response_length=4, traj_offset_in_batch=0)
    with pytest.raises(ValueError):
        a.record_response(response_ids=[], score=0, done=False, has_format_error=False)


def test_assembler_allows_empty_response_with_format_error():
    a = TrajectoryAssembler(prompt_ids=[1], response_length=4, traj_offset_in_batch=0)
    # Should not raise.
    a.record_response(response_ids=[], score=0, done=False, has_format_error=True)
    traj = a.finalize(is_train=True)
    assert traj.trajectory_record.turns[0].has_format_error is True
    # No format penalty in the BEACON-aligned shaper (handled at trainer level).
    assert traj.trajectory_reward == 0.0
    # Invalid-turn count surfaced for the trainer hook.
    assert traj.num_invalid_turns == 1


# ---------------------------------------------------------------------------
# Env-score-delta teacher.
# ---------------------------------------------------------------------------


def test_critique_template_three_branches():
    assert env_score_critique(0, 5) == "This action advanced env score by 5 points."
    assert env_score_critique(10, 5) == "This action triggered env penalty -5."
    assert env_score_critique(5, 5) == "No env-score progress this turn."


def test_wrap_teacher_note_format():
    s = wrap_teacher_note("hello world")
    assert s == "<teacher_note>hello world</teacher_note>"


def test_per_turn_q_signs():
    f = EnvScoreDeltaAnnotator.per_turn_q_from_score
    assert f(10, 20) == +1
    assert f(20, 10) == -1
    assert f(10, 10) == 0


class _StubTokenizer:
    """Returns one synthetic id per character of the input string."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text[:8]]  # cap to keep tests cheap


def test_annotator_l3_disabled_returns_no_trajectories():
    """If L3 is disabled, the teacher returns ``trajectories=None`` so the
    L3 forward is skipped entirely."""
    a1 = TrajectoryAssembler(prompt_ids=[1], response_length=8, traj_offset_in_batch=0)
    a1.record_response(response_ids=[2, 3], score=10, done=False)
    traj1 = a1.finalize().trajectory_record

    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [traj1]})
    annotator = EnvScoreDeltaAnnotator(tokenizer=None, l3_enable=False)
    per_q, trajs = annotator.annotate(batch)
    assert per_q.tolist() == [1.0]
    assert trajs is None


def test_annotator_l3_enabled_emits_trajectory_with_critique_ids():
    a1 = TrajectoryAssembler(prompt_ids=[1, 2], response_length=10, traj_offset_in_batch=0)
    a1.record_response(response_ids=[10, 11], score=5, done=False)            # delta=+5 → q=+1
    a1.record_observation(obs_ids=[20])
    a1.record_response(response_ids=[30, 31], score=5, done=False)            # delta=0  → q=0
    a1.record_response(response_ids=[40], score=2, done=True)                 # delta=-3 → q=-1
    traj1 = a1.finalize().trajectory_record

    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [traj1]})
    annotator = EnvScoreDeltaAnnotator(tokenizer=_StubTokenizer(), l3_enable=True)
    per_q, trajs = annotator.annotate(batch)

    # per_sample_q is the SUM of per-turn q: +1 + 0 + (-1) = 0.
    assert per_q.tolist() == [0.0]
    # trajectories returned because at least one turn has non-empty critique.
    assert trajs is not None and len(trajs) == 1
    t = trajs[0]
    assert t.traj_offset_in_batch == 0
    assert t.prompt_length == 2

    # Three turn annotations, with critique tokens per turn.
    assert len(t.turns) == 3
    assert t.turns[0].q_t == +1.0
    assert t.turns[1].q_t == 0.0
    assert t.turns[2].q_t == -1.0
    # Critique ids should be non-empty for delta != 0 turns.
    assert len(t.turns[0].critique_token_ids) > 0
    assert len(t.turns[2].critique_token_ids) > 0
    # The "no progress" turn still gets *some* critique ("No env-score
    # progress this turn.") wrapped in <teacher_note>; that's fine since
    # δ ≡ 0 in the conditioned forward when q_t = 0.
    assert len(t.turns[1].critique_token_ids) > 0


def test_annotator_returns_none_when_no_critiques_emitted():
    """All turns score-stagnant *and* tokenizer is None → no L3 work to do."""
    a1 = TrajectoryAssembler(prompt_ids=[1], response_length=4, traj_offset_in_batch=0)
    a1.record_response(response_ids=[10], score=0, done=False)
    a1.record_response(response_ids=[20], score=0, done=False)
    traj1 = a1.finalize().trajectory_record
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [traj1]})
    annotator = EnvScoreDeltaAnnotator(tokenizer=None, l3_enable=True)
    per_q, trajs = annotator.annotate(batch)
    assert per_q.tolist() == [0.0]
    assert trajs is None


def test_annotator_validates_offset():
    a1 = TrajectoryAssembler(prompt_ids=[1], response_length=4, traj_offset_in_batch=5)
    a1.record_response(response_ids=[10], score=0, done=False)
    traj1 = a1.finalize().trajectory_record
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [traj1]})
    annotator = EnvScoreDeltaAnnotator(tokenizer=None, l3_enable=False)
    with pytest.raises(IndexError):
        annotator.annotate(batch)


def test_annotator_missing_records_raises():
    annotator = EnvScoreDeltaAnnotator(tokenizer=None, l3_enable=False)
    batch = SimpleNamespace(non_tensor_batch={})
    with pytest.raises(KeyError):
        annotator.annotate(batch)


def test_counterfactual_annotator_emits_turn_scaffold():
    a1 = TrajectoryAssembler(prompt_ids=[1, 2], response_length=8, traj_offset_in_batch=0)
    a1.record_response(response_ids=[10, 11], score=5, done=False, has_format_error=False)
    a1.record_observation(obs_ids=[21])
    a1.record_response(response_ids=[12], score=7, done=True, has_format_error=True)
    rec = a1.finalize().trajectory_record

    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})
    annotator = CounterfactualMaskAnnotator(tokenizer=None, l3_enable=True)
    per_q, trajs = annotator.annotate(batch)

    assert per_q.tolist() == [0.0]
    assert trajs is not None and len(trajs) == 1
    t = trajs[0]
    assert t.traj_offset_in_batch == 0
    assert len(t.turns) == 2
    assert t.turns[0].q_t == 0.0
    assert t.turns[1].q_t == 0.0
    assert t.turns[0].critique_token_ids == []
    assert t.turns[1].critique_token_ids == []
    assert t.turns[0].has_format_error is False
    assert t.turns[1].has_format_error is True


def test_counterfactual_annotator_missing_records_raises():
    annotator = CounterfactualMaskAnnotator(tokenizer=None, l3_enable=True)
    batch = SimpleNamespace(non_tensor_batch={})
    with pytest.raises(KeyError):
        annotator.annotate(batch)


# ---------------------------------------------------------------------------
# End-to-end: assembler → annotator → trainer hook.
# ---------------------------------------------------------------------------


def test_assembler_to_annotator_to_hook_smoke():
    """Reproduce the rollout → teacher → AdvantageContext signal path.

    Two trajectories in the same group:
      τ0: turns score 0→5→10 (won at step 2)
      τ1: turns score 0→0→-5 (failed)
    With env-score-delta teacher we expect:
      τ0: q=[+1, +1] → per_sample_q[0] = +2
      τ1: q=[0, -1]  → per_sample_q[1] = -1
    """
    bs, response_length = 2, 16

    a0 = TrajectoryAssembler(prompt_ids=[1, 2], response_length=response_length, traj_offset_in_batch=0)
    a0.record_response(response_ids=[10, 11], score=5, done=False)
    a0.record_response(response_ids=[20, 21], score=10, done=True)
    rec0 = a0.finalize().trajectory_record

    a1 = TrajectoryAssembler(prompt_ids=[1, 2], response_length=response_length, traj_offset_in_batch=1)
    a1.record_response(response_ids=[30, 31], score=0, done=False)
    a1.record_response(response_ids=[40, 41], score=-5, done=True)
    rec1 = a1.finalize().trajectory_record

    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec0, rec1]})
    annotator = EnvScoreDeltaAnnotator(tokenizer=_StubTokenizer(), l3_enable=True)
    per_q, trajs = annotator.annotate(batch)

    assert per_q.tolist() == [2.0, -1.0]
    assert trajs is not None
    # τ0 has 2 progress turns + τ1 has 1 penalty turn → both trajectories
    # produced critiques and should be in the list.
    offsets = sorted(t.traj_offset_in_batch for t in trajs)
    assert offsets == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
