"""Unit tests for the trainer-side TRACE-GRPO injection hook.

Verifies that :func:`inject_trace_context`:

  - reads ``old_log_probs`` / ``has_format_error`` from a verl-shaped
    batch and assembles the right shapes / dtypes / devices;
  - calls the teacher exactly once;
  - skips the L3 forward when no provider is given OR the teacher returns
    no trajectories;
  - the resulting context is consumed correctly by ``trace_l3_mask`` —
    i.e. an end-to-end run from ``inject_trace_context`` to
    ``trace_l3_mask_compute`` reproduces a known TRACE-GRPO advantage.

We mimic verl's ``DataProto`` with a tiny ``_Batch`` namespace so we don't
depend on verl being installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pytest
import torch

from trace_grpo.patches.context import (
    clear_advantage_context,
    get_advantage_context,
)
from trace_grpo.patches.critique_conditioned_provider import (
    CritiqueConditionedProvider,
    MockCritiqueForward,
    TrajectoryAnnotation,
    TurnAnnotation,
)
from trace_grpo.patches.level3_patch import trace_l3_mask_compute
from trace_grpo.patches.trainer_hook import inject_trace_context


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _fake_batch(
    *,
    bs: int,
    response_length: int,
    old_logprobs: Optional[torch.Tensor] = None,
    has_format_error: Optional[np.ndarray] = None,
):
    """Stand-in for verl's ``DataProto``.

    Exposes ``.batch`` (a dict of tensors) and ``.non_tensor_batch`` (a dict
    of numpy arrays). The keys we look up are ``old_log_probs`` and
    ``response_mask`` (tensors) and ``has_format_error`` (numpy).
    """
    batch_d: dict[str, torch.Tensor] = {}
    if old_logprobs is not None:
        batch_d["old_log_probs"] = old_logprobs
    batch_d["response_mask"] = torch.ones(bs, response_length, dtype=torch.float32)

    nt: dict[str, np.ndarray] = {}
    if has_format_error is not None:
        nt["has_format_error"] = has_format_error

    return SimpleNamespace(batch=batch_d, non_tensor_batch=nt)


class _Teacher:
    """Mock teacher: returns prescribed q values + (optionally) trajectories."""

    def __init__(
        self,
        per_sample_q: np.ndarray,
        trajectories: Optional[Sequence[TrajectoryAnnotation]] = None,
    ):
        self.per_sample_q = per_sample_q
        self.trajectories = trajectories
        self.calls = 0

    def annotate(self, batch):
        self.calls += 1
        return self.per_sample_q, self.trajectories


@pytest.fixture(autouse=True)
def _clear_ctx():
    """Make sure no leftover context bleeds between tests."""
    clear_advantage_context()
    yield
    clear_advantage_context()


# ---------------------------------------------------------------------------
# Basic plumbing.
# ---------------------------------------------------------------------------


def test_hook_stashes_context_with_correct_shapes():
    bs, response_length = 4, 8
    old_lp = torch.zeros(bs, response_length)
    teacher = _Teacher(per_sample_q=np.array([+1.0, 0.0, -1.0, 0.5], dtype=np.float32))
    batch = _fake_batch(bs=bs, response_length=response_length, old_logprobs=old_lp)

    ctx = inject_trace_context(batch, teacher=teacher)

    assert teacher.calls == 1
    assert ctx is get_advantage_context()
    assert ctx.per_turn_q.shape == (bs,)
    assert ctx.per_turn_q.tolist() == [1.0, 0.0, -1.0, 0.5]
    assert ctx.critique_delta is None        # no l3_provider
    assert ctx.has_format_error is None      # not provided


def test_hook_propagates_has_format_error():
    bs, response_length = 3, 4
    old_lp = torch.zeros(bs, response_length)
    teacher = _Teacher(per_sample_q=np.zeros(bs, dtype=np.float32))
    fmt = np.array([False, True, False])
    batch = _fake_batch(
        bs=bs, response_length=response_length, old_logprobs=old_lp, has_format_error=fmt
    )

    ctx = inject_trace_context(batch, teacher=teacher)
    assert ctx.has_format_error is not None
    assert ctx.has_format_error.tolist() == [False, True, False]
    assert ctx.has_format_error.dtype == torch.bool


def test_hook_validates_per_sample_q_shape():
    bs, response_length = 2, 4
    old_lp = torch.zeros(bs, response_length)
    teacher = _Teacher(per_sample_q=np.array([+1.0]))  # wrong shape
    batch = _fake_batch(bs=bs, response_length=response_length, old_logprobs=old_lp)
    with pytest.raises(ValueError):
        inject_trace_context(batch, teacher=teacher)


def test_hook_validates_format_error_shape():
    bs, response_length = 2, 4
    old_lp = torch.zeros(bs, response_length)
    teacher = _Teacher(per_sample_q=np.zeros(bs, dtype=np.float32))
    batch = _fake_batch(
        bs=bs, response_length=response_length,
        old_logprobs=old_lp, has_format_error=np.array([False, True, False]),  # wrong
    )
    with pytest.raises(ValueError):
        inject_trace_context(batch, teacher=teacher)


def test_hook_works_without_old_logprobs_when_l3_disabled():
    """When ``l3_provider=None`` the hook should not require old_log_probs."""
    bs, response_length = 2, 4
    teacher = _Teacher(per_sample_q=np.zeros(bs, dtype=np.float32))
    batch = _fake_batch(bs=bs, response_length=response_length)  # no old_lp
    ctx = inject_trace_context(batch, teacher=teacher)
    assert ctx.critique_delta is None


def test_hook_requires_old_logprobs_when_l3_enabled():
    bs, response_length = 2, 4
    teacher = _Teacher(per_sample_q=np.zeros(bs, dtype=np.float32))
    batch = _fake_batch(bs=bs, response_length=response_length)  # no old_lp
    provider = CritiqueConditionedProvider(forward_fn=MockCritiqueForward())
    with pytest.raises(KeyError):
        inject_trace_context(batch, teacher=teacher, l3_provider=provider)


# ---------------------------------------------------------------------------
# L3 path — provider invoked, critique_delta filled.
# ---------------------------------------------------------------------------


def test_hook_runs_l3_when_provider_and_trajectories_present():
    bs, response_length = 2, 8
    old_lp = torch.zeros(bs, response_length)

    trajs = [
        TrajectoryAnnotation(
            traj_token_ids=list(range(12)),
            prompt_length=4,
            response_length=response_length,
            turns=[
                TurnAnnotation(history_index=0, response_start=0, response_end=4,
                               critique_token_ids=[1, 2], q_t=-1.0),
            ],
            traj_offset_in_batch=1,
        ),
    ]
    teacher = _Teacher(
        per_sample_q=np.array([0.0, -1.0], dtype=np.float32),
        trajectories=trajs,
    )
    provider = CritiqueConditionedProvider(forward_fn=MockCritiqueForward(biases={0: -0.5}))
    batch = _fake_batch(bs=bs, response_length=response_length, old_logprobs=old_lp)

    ctx = inject_trace_context(batch, teacher=teacher, l3_provider=provider)

    # Row 0 untouched (no trajectory annotation), row 1 has δ on [0:4].
    assert ctx.critique_delta is not None
    assert "critique_delta" in batch.batch
    assert batch.batch["critique_delta"] is ctx.critique_delta
    assert batch.batch["critique_delta"].shape == (bs, response_length)
    assert torch.all(ctx.critique_delta[0] == 0)
    assert torch.allclose(ctx.critique_delta[1, 0:4], torch.full((4,), -0.5))
    assert torch.all(ctx.critique_delta[1, 4:] == 0)


def test_hook_reuses_precomputed_teacher_and_batch_critique_delta():
    bs, response_length = 2, 5
    precomputed_delta = torch.randn(bs, response_length)
    trajs = [
        TrajectoryAnnotation(
            traj_token_ids=list(range(9)),
            prompt_length=4,
            response_length=response_length,
            turns=[
                TurnAnnotation(history_index=0, response_start=1, response_end=4,
                               critique_token_ids=[7, 8], q_t=0.7),
            ],
            traj_offset_in_batch=0,
        ),
    ]
    batch = _fake_batch(bs=bs, response_length=response_length)
    batch.batch["critique_delta"] = precomputed_delta

    ctx = inject_trace_context(
        batch,
        per_sample_q=np.array([0.7, 0.0], dtype=np.float32),
        trajectories=trajs,
    )

    assert ctx.critique_delta is precomputed_delta
    assert ctx.turn_q is not None
    assert torch.allclose(ctx.turn_q[0], torch.tensor([0.0, 0.7, 0.7, 0.7, 0.0]))


def test_hook_builds_token_aligned_turn_q_without_l3_provider():
    bs, response_length = 1, 8
    trajs = [
        TrajectoryAnnotation(
            traj_token_ids=list(range(12)),
            prompt_length=4,
            response_length=response_length,
            turns=[
                TurnAnnotation(history_index=0, response_start=0, response_end=3,
                               critique_token_ids=[], q_t=1.0),
                TurnAnnotation(history_index=1, response_start=5, response_end=8,
                               critique_token_ids=[], q_t=-0.5),
            ],
            traj_offset_in_batch=0,
        )
    ]
    teacher = _Teacher(per_sample_q=np.array([0.5], dtype=np.float32), trajectories=trajs)
    batch = _fake_batch(bs=bs, response_length=response_length)

    ctx = inject_trace_context(batch, teacher=teacher)

    assert ctx.turn_q is not None
    expected = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, -0.5, -0.5, -0.5]])
    assert torch.allclose(ctx.turn_q.cpu(), expected)


def test_hook_skips_l3_when_no_trajectories_returned():
    """Teacher returns no trajectory annotations ⇒ critique_delta stays None."""
    bs, response_length = 2, 4
    old_lp = torch.zeros(bs, response_length)
    teacher = _Teacher(
        per_sample_q=np.zeros(bs, dtype=np.float32),
        trajectories=[],   # empty list = no L3 work
    )
    provider = CritiqueConditionedProvider(forward_fn=MockCritiqueForward())
    batch = _fake_batch(bs=bs, response_length=response_length, old_logprobs=old_lp)
    ctx = inject_trace_context(batch, teacher=teacher, l3_provider=provider)
    assert ctx.critique_delta is None


# ---------------------------------------------------------------------------
# End-to-end: hook → estimator produces the right advantage.
# ---------------------------------------------------------------------------


def test_hook_to_estimator_end_to_end():
    """Complete the round trip: trainer hook → AdvantageContext → estimator."""
    bs, response_length = 4, 4
    # Reproduce the τ_2.turn_1 emnlp §6 case: R = [1,0,1,0], q = [0,0,-1,0],
    # δ for τ_2 = [0, +0.1, 0, +0.5, -5.5, 0] (we use response_length=6 so
    # we don't truncate; bump bs context to fit). Easier: just do L1+L2
    # check (no critique_delta) and confirm A[2] == +0.693.
    old_lp = torch.zeros(bs, response_length)
    teacher = _Teacher(per_sample_q=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32))
    batch = _fake_batch(bs=bs, response_length=response_length, old_logprobs=old_lp)
    inject_trace_context(batch, teacher=teacher)

    # Now run the estimator with the values that would normally come from
    # verl. Build the same token_level_rewards used in the §6 case.
    rewards = torch.zeros(bs, response_length)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    response_mask = torch.ones_like(rewards)
    index = np.array(["g0"] * bs, dtype=object)

    ctx = get_advantage_context()
    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        per_turn_q=ctx.per_turn_q,
        critique_delta=ctx.critique_delta,
        has_format_error=ctx.has_format_error,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
    )
    # τ_2: success traj + detour turn (q=-1) with (sigma+sigma_eps) baseline.
    assert torch.allclose(A[2], torch.full((response_length,), 0.6916), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
