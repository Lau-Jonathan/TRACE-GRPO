"""Scheduling-logic tests for the critique_v1 delta provider.

These tests pin down the contract between
:class:`CritiqueConditionedProvider` and a forward function:

  - which turns are forwarded vs skipped (empty critique → skip)
  - how the response-axis frame maps onto the (bs, response_length) tensor
  - which positions of ``critique_delta`` are written (only annotated
    response spans) and which stay 0
  - precondition checks (out-of-range spans, mismatched shapes)

We use :class:`MockCritiqueForward`, which returns a fixed bias per
``history_index``, so we can assert exact δ values without a real model.
"""

from __future__ import annotations

import pytest
import torch

from trace_grpo.patches.conditioned_forward import (
    DELTA_PROVIDER_REGISTRY,
    get_delta_provider,
    register_delta_provider,
    resolve_delta_provider,
)
from trace_grpo.patches.critique_conditioned_provider import (
    CritiqueConditionedForwardFn,
    CritiqueConditionedProvider,
    MockCritiqueForward,
    TrajectoryAnnotation,
    TurnAnnotation,
)
from trace_grpo.patches.counterfactual_provider import CounterfactualMaskProvider


# ---------------------------------------------------------------------------
# Registry semantics.
# ---------------------------------------------------------------------------


def test_critique_v1_provider_is_registered():
    cls = get_delta_provider("critique_v1")
    assert cls is CritiqueConditionedProvider


def test_counterfactual_provider_is_registered():
    cls = get_delta_provider("counterfactual_mask_v1")
    assert cls is CounterfactualMaskProvider


def test_resolve_delta_provider_for_known_teachers():
    assert resolve_delta_provider("env_score") == "critique_v1"
    assert resolve_delta_provider("llm") == "critique_v1"
    assert resolve_delta_provider("counterfactual") == "counterfactual_mask_v1"


def test_resolve_delta_provider_unknown_teacher_raises():
    with pytest.raises(KeyError):
        resolve_delta_provider("not_a_real_teacher")


def test_register_delta_provider_rejects_duplicate():
    """Re-registering the same name with a different class must raise.

    Re-registering the *exact same* class is idempotent (covers the
    common case where ``trace_grpo.patches`` is imported twice).
    """
    @register_delta_provider("temp_provider_for_test")
    class _A:
        pass

    # Different class under the same name → ValueError.
    with pytest.raises(ValueError):
        @register_delta_provider("temp_provider_for_test")
        class _B:
            pass

    # Same class re-registered is allowed (no-op).
    register_delta_provider("temp_provider_for_test")(_A)

    # Cleanup so other tests aren't affected.
    DELTA_PROVIDER_REGISTRY.pop("temp_provider_for_test", None)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_traj(
    *,
    prompt_length: int,
    response_length: int,
    extra_obs_len: int = 0,
    turns: list[TurnAnnotation],
    traj_offset: int = 0,
):
    """Build a TrajectoryAnnotation with ``prompt_length`` prompt tokens at
    the front and ``response_length`` total response/observation tokens
    after, all populated with arbitrary token ids."""
    L = prompt_length + response_length + extra_obs_len
    return TrajectoryAnnotation(
        traj_token_ids=list(range(L)),
        prompt_length=prompt_length,
        response_length=response_length,
        turns=turns,
        traj_offset_in_batch=traj_offset,
    )


# ---------------------------------------------------------------------------
# Trivial: no annotated turns → δ ≡ 0 and forward_fn never called.
# ---------------------------------------------------------------------------


def test_no_annotated_turns_returns_zero_delta_and_skips_forward():
    bs, response_length = 4, 16
    old_logprobs = torch.randn(bs, response_length)

    calls = []
    class _Spy(CritiqueConditionedForwardFn):
        def __call__(self, trajectory_tokens, prompt_length, turns):
            calls.append((trajectory_tokens.shape[0], prompt_length, len(turns)))
            return torch.zeros(trajectory_tokens.shape[0] - prompt_length)

    provider = CritiqueConditionedProvider(forward_fn=_Spy())

    # Two trajectories, neither has any non-empty critique.
    trajs = [
        _make_traj(
            prompt_length=4,
            response_length=response_length,
            turns=[
                TurnAnnotation(history_index=0, response_start=0, response_end=5,
                               critique_token_ids=[], q_t=0.0),
                TurnAnnotation(history_index=1, response_start=5, response_end=10,
                               critique_token_ids=[], q_t=0.0),
            ],
            traj_offset=0,
        ),
        _make_traj(
            prompt_length=4,
            response_length=response_length,
            turns=[],
            traj_offset=2,
        ),
    ]

    delta = provider.compute_critique_delta(trajs, old_logprobs)
    assert delta.shape == old_logprobs.shape
    assert torch.all(delta == 0)
    assert calls == []  # forward_fn was never invoked


# ---------------------------------------------------------------------------
# Single annotated turn — δ written only on that response span.
# ---------------------------------------------------------------------------


def test_single_annotated_turn_writes_only_its_span():
    bs, response_length = 4, 16
    old_logprobs = torch.full((bs, response_length), -2.0)

    forward = MockCritiqueForward(biases={3: -0.7})  # arbitrary bias
    provider = CritiqueConditionedProvider(forward_fn=forward)

    trajs = [
        _make_traj(
            prompt_length=4,
            response_length=response_length,
            turns=[
                TurnAnnotation(history_index=3, response_start=6, response_end=10,
                               critique_token_ids=[42, 43, 44], q_t=-1.0),
            ],
            traj_offset=2,
        ),
    ]

    delta = provider.compute_critique_delta(trajs, old_logprobs)

    # Row 2, columns 6..10 should be -0.7 - (-2.0) = 1.3.
    expected_seg = torch.full((4,), 1.3)
    assert torch.allclose(delta[2, 6:10], expected_seg)
    # All other positions of row 2 stay 0.
    assert torch.all(delta[2, :6] == 0)
    assert torch.all(delta[2, 10:] == 0)
    # Other rows untouched.
    assert torch.all(delta[0] == 0)
    assert torch.all(delta[1] == 0)
    assert torch.all(delta[3] == 0)


# ---------------------------------------------------------------------------
# Mixed turns — annotated + unannotated inside the same trajectory.
# ---------------------------------------------------------------------------


def test_mixed_annotated_and_unannotated_turns():
    """Only annotated turns produce non-zero δ; unannotated turn's response
    span stays at 0."""
    bs, response_length = 2, 32
    old_logprobs = torch.zeros(bs, response_length)

    forward = MockCritiqueForward(biases={0: 0.5, 2: -0.25})
    provider = CritiqueConditionedProvider(forward_fn=forward)

    trajs = [
        _make_traj(
            prompt_length=8,
            response_length=response_length,
            turns=[
                TurnAnnotation(history_index=0, response_start=0, response_end=5,
                               critique_token_ids=[100], q_t=+1.0),
                TurnAnnotation(history_index=1, response_start=5, response_end=12,
                               critique_token_ids=[], q_t=0.0),  # NOT annotated
                TurnAnnotation(history_index=2, response_start=12, response_end=20,
                               critique_token_ids=[101, 102], q_t=-0.5),
            ],
            traj_offset=0,
        ),
    ]
    delta = provider.compute_critique_delta(trajs, old_logprobs)

    assert torch.allclose(delta[0, 0:5], torch.full((5,), 0.5))
    assert torch.all(delta[0, 5:12] == 0)
    assert torch.allclose(delta[0, 12:20], torch.full((8,), -0.25))
    assert torch.all(delta[0, 20:] == 0)


# ---------------------------------------------------------------------------
# Multiple trajectories with disjoint offsets.
# ---------------------------------------------------------------------------


def test_multiple_trajectories_scatter_to_correct_rows():
    bs, response_length = 8, 24
    old_logprobs = torch.zeros(bs, response_length)
    forward = MockCritiqueForward(biases={5: 1.0, 9: -1.0})
    provider = CritiqueConditionedProvider(forward_fn=forward)

    trajs = [
        _make_traj(
            prompt_length=4,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=5, response_start=0, response_end=4,
                                  critique_token_ids=[1])],
            traj_offset=1,
        ),
        _make_traj(
            prompt_length=4,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=9, response_start=10, response_end=14,
                                  critique_token_ids=[2])],
            traj_offset=6,
        ),
    ]
    delta = provider.compute_critique_delta(trajs, old_logprobs)
    # Row 1 [0:4] = 1.0
    assert torch.allclose(delta[1, 0:4], torch.full((4,), 1.0))
    # Row 6 [10:14] = -1.0
    assert torch.allclose(delta[6, 10:14], torch.full((4,), -1.0))
    # All other rows / positions are 0.
    for i in range(bs):
        if i not in (1, 6):
            assert torch.all(delta[i] == 0), f"row {i} should be zero"


# ---------------------------------------------------------------------------
# δ correctly subtracts old_logprobs.
# ---------------------------------------------------------------------------


def test_delta_is_cond_minus_old():
    """The provider must return ``cond − old``, not just ``cond``."""
    bs, response_length = 1, 8
    old_logprobs = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0]])
    forward = MockCritiqueForward(biases={0: -0.5})  # cond = -0.5 on response[0:5]
    provider = CritiqueConditionedProvider(forward_fn=forward)
    trajs = [
        _make_traj(
            prompt_length=2,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=0, response_start=0, response_end=5,
                                  critique_token_ids=[7])],
            traj_offset=0,
        ),
    ]
    delta = provider.compute_critique_delta(trajs, old_logprobs)
    expected = torch.tensor([
        -0.5 - 0.0,
        -0.5 - (-1.0),
        -0.5 - (-2.0),
        -0.5 - (-3.0),
        -0.5 - (-4.0),
        0.0,
        0.0,
        0.0,
    ])
    assert torch.allclose(delta[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Precondition errors.
# ---------------------------------------------------------------------------


def test_response_end_beyond_response_length_raises():
    bs, response_length = 1, 8
    old_logprobs = torch.zeros(bs, response_length)
    forward = MockCritiqueForward(biases={0: 0.0})
    provider = CritiqueConditionedProvider(forward_fn=forward)
    trajs = [
        _make_traj(
            prompt_length=2,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=0, response_start=4, response_end=10,
                                  critique_token_ids=[1])],
            traj_offset=0,
        ),
    ]
    with pytest.raises(IndexError):
        provider.compute_critique_delta(trajs, old_logprobs)


def test_empty_response_span_raises():
    bs, response_length = 1, 8
    old_logprobs = torch.zeros(bs, response_length)
    forward = MockCritiqueForward()
    provider = CritiqueConditionedProvider(forward_fn=forward)
    trajs = [
        _make_traj(
            prompt_length=2,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=0, response_start=4, response_end=4,
                                  critique_token_ids=[1])],
            traj_offset=0,
        ),
    ]
    with pytest.raises(ValueError):
        provider.compute_critique_delta(trajs, old_logprobs)


def test_traj_offset_out_of_range_raises():
    bs, response_length = 2, 8
    old_logprobs = torch.zeros(bs, response_length)
    forward = MockCritiqueForward()
    provider = CritiqueConditionedProvider(forward_fn=forward)
    trajs = [
        _make_traj(
            prompt_length=2,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=0, response_start=0, response_end=2,
                                  critique_token_ids=[1])],
            traj_offset=5,  # > bs - 1
        ),
    ]
    with pytest.raises(IndexError):
        provider.compute_critique_delta(trajs, old_logprobs)


def test_response_length_mismatch_raises():
    bs, response_length = 1, 8
    old_logprobs = torch.zeros(bs, response_length)
    forward = MockCritiqueForward()
    provider = CritiqueConditionedProvider(forward_fn=forward)
    trajs = [
        TrajectoryAnnotation(
            traj_token_ids=[0, 1, 2, 3, 4],
            prompt_length=2,
            response_length=16,                   # != 8
            turns=[TurnAnnotation(history_index=0, response_start=0, response_end=2,
                                  critique_token_ids=[1])],
            traj_offset_in_batch=0,
        ),
    ]
    with pytest.raises(ValueError):
        provider.compute_critique_delta(trajs, old_logprobs)


def test_old_logprobs_must_be_2d():
    forward = MockCritiqueForward()
    provider = CritiqueConditionedProvider(forward_fn=forward)
    with pytest.raises(ValueError):
        provider.compute_critique_delta([], torch.zeros(8))


def test_forward_fn_returning_wrong_shape_raises():
    """If the forward fn returns a tensor of unexpected length, surface that
    immediately rather than silently miscomputing."""
    bs, response_length = 1, 8
    old_logprobs = torch.zeros(bs, response_length)

    class _BadForward(CritiqueConditionedForwardFn):
        def __call__(self, trajectory_tokens, prompt_length, turns):
            return torch.zeros(trajectory_tokens.shape[0])  # wrong: should be L - prompt_length

    provider = CritiqueConditionedProvider(forward_fn=_BadForward())
    trajs = [
        _make_traj(
            prompt_length=2,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=0, response_start=0, response_end=2,
                                  critique_token_ids=[1])],
            traj_offset=0,
        ),
    ]
    with pytest.raises(RuntimeError):
        provider.compute_critique_delta(trajs, old_logprobs)


# ---------------------------------------------------------------------------
# dtype / device propagation.
# ---------------------------------------------------------------------------


def test_delta_matches_old_logprobs_dtype_and_device():
    bs, response_length = 1, 8
    old_logprobs = torch.zeros(bs, response_length, dtype=torch.float64)
    forward = MockCritiqueForward(biases={0: 0.5})
    provider = CritiqueConditionedProvider(forward_fn=forward)
    trajs = [
        _make_traj(
            prompt_length=2,
            response_length=response_length,
            turns=[TurnAnnotation(history_index=0, response_start=0, response_end=4,
                                  critique_token_ids=[1])],
            traj_offset=0,
        ),
    ]
    delta = provider.compute_critique_delta(trajs, old_logprobs)
    assert delta.dtype == old_logprobs.dtype
    assert delta.device == old_logprobs.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
