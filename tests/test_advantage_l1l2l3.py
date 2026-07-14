"""Golden tests for ``trace_l3_mask_compute``.

Reproduces the worked example updated by the paper §5:

    Group return = [1.0, 0.0, 1.0, 0.0]                 # μ=0.5, σ=0.577
    A_grpo       = [+0.866, -0.866, +0.866, -0.866]     # GRPO baseline

    τ_2.turn_1 (failed detour, q=-1):
      after L2: A = +0.866 × clamp(1 + 0.2*(+1)*(-1), 0, 1.2) = +0.693
      δ        = [0.0, +0.1, 0.0, +0.5, -5.5, 0.0]   (clipped to κ=1.0)
      δ_clip   = [0.0, +0.1, 0.0, +0.5, -1.0, 0.0]
      w_raw    = 1 + 0.3 * (-1) * δ_clip = [1.00, 0.97, 1.00, 0.85, 1.30, 1.00]
      mean(w)  = 1.020
      w_norm   = [0.980, 0.951, 0.980, 0.833, 1.275, 0.980]
      A_after_L3 = +0.693 * w_norm
                 = [0.6792, 0.6588, 0.6792, 0.5774, 0.8839, 0.6792]
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from trace_grpo.patches.level3_patch import trace_l3_mask_compute


# Tolerance for the spec-printed 4-digit numbers.
RTOL = 1e-3


def _make_inputs(
    R: list[float],
    q: list[float],
    response_length: int,
    delta_per_sample: list[list[float]] | None = None,
    has_format_error: list[bool] | None = None,
):
    """Build (token_level_rewards, response_mask, index, per_turn_q, critique_delta, has_format_error)
    such that ``token_level_rewards.sum(-1) == R``. We place the entire reward
    on the last token of each row.
    """
    bs = len(R)
    token_level_rewards = torch.zeros(bs, response_length)
    for i, r in enumerate(R):
        token_level_rewards[i, -1] = r
    response_mask = torch.ones(bs, response_length, dtype=torch.float32)
    index = np.array(["g0"] * bs, dtype=object)
    per_turn_q = torch.tensor(q, dtype=torch.float32)
    critique_delta = None
    if delta_per_sample is not None:
        critique_delta = torch.tensor(delta_per_sample, dtype=torch.float32)
        assert critique_delta.shape == (bs, response_length)
    fmt = None
    if has_format_error is not None:
        fmt = torch.tensor(has_format_error, dtype=torch.bool)
    return token_level_rewards, response_mask, index, per_turn_q, critique_delta, fmt


# ---------------------------------------------------------------------------
# Vanilla GRPO reproduction — emnlp §2.
# ---------------------------------------------------------------------------


def test_grpo_baseline_recovered_when_q_is_zero():
    """q=0 turns TRACE-GRPO into vanilla GRPO bit-equivalent (L2 w²=1, L3 w³=1)."""
    response_length = 3
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0] * 4
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        critique_delta=None,
        has_format_error=None,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
        text_feedback_sigma_eps=1e-3,
    )
    # μ=0.5, σ=std([1,0,1,0]) with Bessel correction = 0.5773
    expected = torch.tensor([+0.866, -0.866, +0.866, -0.866]).unsqueeze(-1).expand(4, response_length)
    assert torch.allclose(A, expected, atol=2e-3), f"got {A}"


# ---------------------------------------------------------------------------
# L2 turn modulation — tracegrpo §5.2.
# ---------------------------------------------------------------------------


def test_l2_failed_traj_correct_turn_softens_penalty():
    """τ_1.turn_2 (take, correct under failing trajectory): q=+1.

    A_grpo = -0.866, sign = -1
    w² = 1 + 0.2 * (-1) * (+1) = 0.8
    A_after_L2 = -0.866 * 0.8 = -0.693
    """
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, +1.0, 0.0, 0.0]  # only τ_1.turn_2 is annotated
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
    )
    # τ_1 (i=1) gets the modulation
    assert math.isclose(A[1, 0].item(), -0.6916, abs_tol=RTOL), f"got {A[1, 0]}"
    # other rows keep vanilla GRPO
    assert math.isclose(A[0, 0].item(), +0.8645, abs_tol=RTOL)


def test_l2_failed_traj_wrong_turn_amplifies_penalty():
    """τ_1.turn_3 (no heat → put, wrong): q=-1.

    sign(A_grpo) = -1, q = -1 → w² = 1 + 0.2 * (-1) * (-1) = 1.2
    A_after_L2 = -0.866 * 1.2 = -1.039
    """
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, -1.0, 0.0, 0.0]
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
    )
    assert math.isclose(A[1, 0].item(), -1.0374, abs_tol=RTOL), f"got {A[1, 0]}"


def test_l2_success_traj_detour_softens_reward():
    """τ_2.turn_1 (detour to wrong place, but trajectory still succeeds): q=-1.

    A_grpo = +0.866, sign = +1
    w² = 1 + 0.2 * (+1) * (-1) = 0.8
    A_after_L2 = +0.866 * 0.8 = +0.693
    """
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, 0.0, -1.0, 0.0]
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
    )
    assert math.isclose(A[2, 0].item(), +0.6916, abs_tol=RTOL), f"got {A[2, 0]}"


def test_l2_success_traj_critical_correct_turn_amplifies_reward():
    """τ_2.turn_5 (final put, q=+1).

    sign(A_grpo) = +1, q = +1 → w² = 1.2 → A = +0.866 * 1.2 = +1.039
    """
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, 0.0, +1.0, 0.0]
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
    )
    assert math.isclose(A[2, 0].item(), +1.0374, abs_tol=RTOL), f"got {A[2, 0]}"


def test_l2_format_error_forces_w2_one():
    """A turn flagged ``has_format_error=True`` must keep w² = 1 regardless of q."""
    R = [1.0, 0.0, 1.0, 0.0]
    # τ_1.turn_3 has q=-1 *and* format error — without the override, w² would be 1.2.
    q = [0.0, -1.0, 0.0, 0.0]
    response_length = 1
    rewards, mask, index, q_t, _, fmt = _make_inputs(
        R, q, response_length, has_format_error=[False, True, False, False]
    )

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        has_format_error=fmt,
    )
    # τ_1 should get *plain* GRPO, not the 1.2x amplification.
    assert math.isclose(A[1, 0].item(), -0.8645, abs_tol=RTOL), f"got {A[1, 0]}"


# ---------------------------------------------------------------------------
# L3 token modulation — tracegrpo §5.3 worked example.
# ---------------------------------------------------------------------------


def test_l3_token_redistribution_emnlp_example():
    """Exactly the τ_2.turn_1 scenario printed in emnlp §5.3 / §6.

    response tokens: [<action>, go, to, countertop, 2, </action>]
    pre-L3 advantage (after L2): +0.693
    δ (raw): [0.0, +0.1, 0.0, +0.5, -5.5, 0.0]
    after κ=1.0 clip: [0.0, +0.1, 0.0, +0.5, -1.0, 0.0]
    q=-1 → w_raw = 1 + 0.3 * (-1) * δ_clip = [1.00, 0.97, 1.00, 0.85, 1.30, 1.00]
    mean(w_raw) = 1.020
    w_norm = w_raw / 1.020 = [0.9804, 0.9510, 0.9804, 0.8333, 1.2745, 0.9804]
    A_after_L3 = 0.693 * w_norm
               ≈ [0.6792, 0.6588, 0.6792, 0.5774, 0.8839, 0.6792]
    """
    response_length = 6
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, 0.0, -1.0, 0.0]
    delta = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, +0.1, 0.0, +0.5, -5.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    rewards, mask, index, q_t, delta_t, _ = _make_inputs(R, q, response_length, delta)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        critique_delta=delta_t,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
        trace_l3_alpha=0.3,
        trace_l3_kappa=1.0,
    )

    # emnlp §6 prints these to 4 decimals; allow 1e-3 absolute tolerance to
    # absorb their internal rounding.
    expected = torch.tensor([0.6781, 0.6577, 0.6781, 0.5764, 0.8815, 0.6781])
    assert torch.allclose(A[2], expected, atol=1e-3), f"got {A[2]}\nexpected {expected}"


# ---------------------------------------------------------------------------
# L1 zero-variance fallback — emnlp §3.
# ---------------------------------------------------------------------------


def test_l1_all_fail_group_uses_q_fallback():
    """All-failure group: σ ≈ 0, vanilla GRPO advantage degenerates to 0.

    Per the paper §5.1, L1 fallback adds ``α · q_t`` to the token-level
    (zero) base so we recover gradient signal. L2 still uses ``sign(base)``,
    so a pure zero-variance fallback is not amplified by L2.

    All-fail group with negative q_sum:
      μ = 0, σ = 0 → zero_var=True; q_sum=-1.3 → mu_val<=0 & q_sum>0? No → gate_keep=True.
      base_grpo = 0, A¹ += 0.5·q_t:   τ_0=-0.5, τ_1=0, τ_2=-0.15, τ_3=0.

    L2 step (sign(base_grpo) = 0):
      all rows keep w² = 1.
    """
    R = [0.0, 0.0, 0.0, 0.0]                        # all fail
    q = [-1.0, 0.0, -0.3, 0.0]                      # τ_0/τ_2 annotated negative
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
        text_feedback_sigma_eps=1e-3,
    )
    assert math.isclose(A[0, 0].item(), -0.50, abs_tol=RTOL), f"got {A[0, 0]}"
    assert math.isclose(A[1, 0].item(), 0.0, abs_tol=RTOL)
    assert math.isclose(A[2, 0].item(), -0.15, abs_tol=RTOL), f"got {A[2, 0]}"
    assert math.isclose(A[3, 0].item(), 0.0, abs_tol=RTOL)


def test_l1_all_success_group_keeps_zero_when_no_q():
    """All-success group with no annotations: σ ≈ 0, q=0 everywhere, A = 0."""
    R = [1.0, 1.0, 1.0, 1.0]
    q = [0.0] * 4
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
    )
    assert torch.allclose(A, torch.zeros_like(A), atol=1e-6)


def test_l1_gate_keep_blocks_inconsistent_fallback():
    """When μ and q_sum disagree in sign, gate_keep=False drops the fallback.

    Spec literal (the paper §3.2):
        gate_keep = not (
            (group_mean <= 0 and group_q_sum > 0) or
            (group_mean > 0 and group_q_sum < 0)
        )

    Setup: all-fail group (μ=0) with positive q_sum → first clause fires →
    gate_keep=False → λ·q fallback is *not* added → A is zero across the board
    (baseline GRPO is zero, no rescue, no L2 amplification).
    """
    R = [0.0, 0.0, 0.0, 0.0]                # μ = 0, σ ≈ 0
    q = [+1.0, +0.5, 0.0, 0.0]              # q_sum = +1.5 > 0  → contradiction
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)
    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards, response_mask=mask, index=index, per_turn_q=q_t,
    )
    assert torch.allclose(A, torch.zeros_like(A), atol=1e-6), f"got {A}"


def test_l1_gate_keep_allows_consistent_fallback():
    """Sanity check the *consistent* branch — μ and q_sum align → gate_keep=True.

    All-fail group (μ=0) with **negative** q_sum: first clause is
    ``mu<=0 and q_sum>0`` (False), second is ``mu>0 and q_sum<0`` (False) →
    gate_keep=True → fallback applies, A becomes non-zero.
    """
    R = [0.0, 0.0, 0.0, 0.0]
    q = [-1.0, -0.5, 0.0, 0.0]
    response_length = 1
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)
    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards, response_mask=mask, index=index, per_turn_q=q_t,
    )
    # Annotated samples should now have non-zero advantage, unannotated stay 0.
    assert A[0, 0].item() < 0
    assert A[1, 0].item() < 0
    assert math.isclose(A[2, 0].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(A[3, 0].item(), 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# L3 disabled / no critique — graceful degrade.
# ---------------------------------------------------------------------------


def test_l3_disabled_keeps_l2_advantage_per_token():
    """If trace_l3_enable=False the per-token advantage is just A_after_L2 broadcast."""
    response_length = 4
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, -1.0, 0.0, 0.0]
    delta = [[0.0]*4]*4
    rewards, mask, index, q_t, delta_t, _ = _make_inputs(R, q, response_length, delta)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        critique_delta=delta_t,
        trace_l3_enable=False,
    )
    # τ_1 row should be -1.039 broadcast.
    assert torch.allclose(A[1], torch.full((response_length,), -1.0374), atol=RTOL)


def test_l3_with_zero_delta_is_noop():
    """δ=0 everywhere → w_raw=1, mean(w_raw)=1, w_norm=1; A_after_L3 == A_after_L2."""
    response_length = 4
    R = [1.0, 0.0, 1.0, 0.0]
    q = [0.0, 0.0, -1.0, 0.0]
    delta = [[0.0]*4]*4
    rewards, mask, index, q_t, delta_t, _ = _make_inputs(R, q, response_length, delta)

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        critique_delta=delta_t,
        trace_l3_enable=True,
    )
    assert torch.allclose(A[2], torch.full((response_length,), 0.6916), atol=RTOL)


def test_l2_uses_token_aligned_turn_q_when_available():
    """Different turns in one trajectory can receive different L2 weights."""
    response_length = 6
    R = [1.0, 0.0]
    q = [0.0, 0.0]  # sample-level q only drives L1 group logic here
    rewards, mask, index, q_t, _, _ = _make_inputs(R, q, response_length)
    turn_q = torch.tensor([
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    A, _ = trace_l3_mask_compute(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        per_turn_q=q_t,
        turn_q=turn_q,
        text_feedback_alpha=0.5,
        text_feedback_lambda=0.2,
    )

    # Row 0 has positive GRPO base ~= +0.707. First turn q=+1 -> 1.2x,
    # second turn q=-1 -> 0.8x.
    assert torch.allclose(A[0, :3], torch.full((3,), 0.8473), atol=1e-3)
    assert torch.allclose(A[0, 3:], torch.full((3,), 0.5649), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
