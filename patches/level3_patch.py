"""TRACE-GRPO three-level advantage estimator (L1 sample gate + L2 turn modulation
+ L3 token modulation) registered with verl's adv-estimator registry.

The pure-numerics function lives in :func:`trace_l3_mask_compute` and is
exercised directly by the unit tests. The verl-callable wrapper
:func:`trace_l3_mask` reads ``per_turn_q``, ``has_format_error``, and
``critique_delta`` off a stash that the trainer entry point fills in just
before calling ``compute_advantage`` (verl's plug-in adv-estimator API does
not accept ``DataProto`` directly, so we route those tensors through a
process-local context object — see :mod:`trace_grpo.patches.context`).

Algorithm reference: the paper §5.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from typing import Optional

import numpy as np
import torch


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

try:
    from verl.trainer.ppo.core_algos import register_adv_est
except Exception:  # pragma: no cover  (verl not installed in some envs)
    def register_adv_est(name):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn
        return _decorator

from .context import get_advantage_context

logger = logging.getLogger(__name__)
_L3_LOCAL = threading.local()


def set_last_l3_stats(stats: dict[str, float]) -> None:
    """Persist last-step L3 telemetry in process-local storage."""
    _L3_LOCAL.last_l3_stats = dict(stats or {})


def get_last_l3_stats() -> dict[str, float]:
    """Read the most recent L3 telemetry from process-local storage."""
    return dict(getattr(_L3_LOCAL, "last_l3_stats", {}))


def _next_l3_step_seen() -> int:
    step_seen = int(getattr(_L3_LOCAL, "step_seen", 0)) + 1
    _L3_LOCAL.step_seen = step_seen
    return step_seen


# ----------------------------------------------------------------------------
# Pure numerics (test target).
# ----------------------------------------------------------------------------


def _row_mean_with_mask(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-row mean of ``x`` over the positions where ``mask == 1``.

    Args:
        x: shape ``(bs, T)``.
        mask: shape ``(bs, T)``, values in {0, 1}.

    Returns:
        Tensor of shape ``(bs, 1)``; rows with empty mask map to 1.0 so that
        downstream division is a no-op for those rows.
    """
    mask = mask.to(x.dtype)
    num = (x * mask).sum(dim=-1, keepdim=True)
    den = mask.sum(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.where(mask.sum(dim=-1, keepdim=True) > 0, num / den, torch.ones_like(num))
    return out


def trace_l3_mask_compute(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    per_turn_q: torch.Tensor,
    turn_q: Optional[torch.Tensor] = None,
    critique_delta: Optional[torch.Tensor] = None,
    has_format_error: Optional[torch.Tensor] = None,
    *,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    text_feedback_alpha: float = 0.5,
    text_feedback_lambda: float = 0.2,
    text_feedback_sigma_eps: float = 1e-3,
    trace_l3_alpha: float = 0.3,
    trace_l3_kappa: float = 1.0,
    trace_l3_enable: bool = True,
    trace_l3_suppress_cap: float = 0.3,
    trace_l3_boost_cap: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute TRACE-GRPO advantages.

    Args:
        token_level_rewards: ``(bs, response_length)``. Sum along the response
            axis is the trajectory-level scalar reward ``R_i`` fed to GRPO.
        response_mask: ``(bs, response_length)``, 1 on actor-emitted tokens.
        index: ``(bs,)`` group ids — trajectories sharing the same id belong
            to the same GRPO group and are normalized together.
        per_turn_q: ``(bs,)`` ``q ∈ [-1, +1]``, 0 if missing. **Sample-level**:
            in the packed-trajectory data layout this is the trajectory-level
            aggregate of per-turn judgments produced by the teacher (typically
            the sum or sign-of-sum of the turn-q values inside the trajectory).
            The caller — i.e. the trainer hook — is responsible for that
            aggregation; this function does not look at turn boundaries.
        turn_q: optional ``(bs, response_length)`` token-aligned per-turn q.
            This is the spec-correct signal for L2/L3. If absent, the old
            behavior is preserved by broadcasting ``per_turn_q`` across the row.
        critique_delta: ``(bs, response_length)`` token-level
            ``log p(t|prefix+critique) − log p(t|prefix)``, ``None`` if L3 is
            disabled or no critique is available.
        has_format_error: optional bool mask. Preferred shape is
            ``(bs, response_length)`` (token-level turn mask) so only
            format-error turns keep ``w² = 1``. Legacy ``(bs,)`` is also
            accepted and pins whole rows.

    Algorithm (matches the paper §5 step-by-step):
        1. Per-group reduction: ``μ, σ`` over ``R_i``, ``q_sum`` over ``q_i``.
        2. ``base = (R − μ) / (σ + ε)`` (or ``R − μ`` if std-norm disabled).
        3. **L1 fallback (additive)** — build token-level ``A¹`` from
           ``base`` and, when ``σ < sigma_eps`` and ``gate_keep``, add
           ``α · q_t`` so zero-variance groups still get gradient.
        4. **L2** — ``w² = clamp(1 + λ · sign(base) · q_t, 0, 1+λ)``;
           overridden to 1 on format-error rows.
        5. ``A_after_L2 = A¹ · w²``.
        6. **L3** (if enabled and ``critique_delta`` provided) —
           ``δ_clip = clamp(δ, ±κ)``;
           ``w³_raw = 1 + α₃ · sign(q) · |q| · δ_clip`` (per-token);
           ``w³_raw = clamp(w³_raw, 1−suppress_cap, 1+boost_cap)``
           (asymmetric: strong suppress, weak boost);
           ``w³ = w³_raw / row_mean(w³_raw)`` so per-turn magnitude is
           preserved (L3 only redistributes weight across tokens).
        7. ``A_final = A_after_L2.unsqueeze(-1) · w³ · response_mask``.

    Returns ``(advantages, returns)`` of shape ``(bs, response_length)``;
    ``returns == advantages`` because TRACE-GRPO is outcome-only (matches verl
    GRPO convention).
    """
    # Avoid stale telemetry when the current step does not activate L3.
    set_last_l3_stats({})

    bs, response_length = token_level_rewards.shape
    device = token_level_rewards.device

    # ----- Group reduction: trajectory-level R, μ, σ, q_sum ---------------
    scores = token_level_rewards.sum(dim=-1).clone()  # (bs,) per-trajectory R
    per_turn_q = per_turn_q.to(scores.dtype).to(device)
    if turn_q is not None:
        turn_q = turn_q.to(device=device, dtype=scores.dtype)
        if turn_q.shape != response_mask.shape:
            raise ValueError(
                f"turn_q shape {tuple(turn_q.shape)} must match response_mask "
                f"{tuple(response_mask.shape)}"
            )
        q_tokens = turn_q
    else:
        q_tokens = per_turn_q.unsqueeze(-1).expand_as(response_mask)

    id2score: dict = defaultdict(list)
    id2qsum: dict = defaultdict(float)
    for i in range(bs):
        id2score[index[i]].append(scores[i])
        id2qsum[index[i]] += float(per_turn_q[i].item())

    id2mean: dict = {}
    id2std: dict = {}
    for idx, lst in id2score.items():
        if len(lst) == 1:
            id2mean[idx] = torch.tensor(0.0, device=device, dtype=scores.dtype)
            id2std[idx] = torch.tensor(1.0, device=device, dtype=scores.dtype)
        else:
            t = torch.stack(lst)
            id2mean[idx] = t.mean()
            id2std[idx] = t.std()

    # ----- L1: GRPO base + token-level q fallback for zero-var groups -----
    # Per the paper §5.1, L1 starts from vanilla GRPO and only injects
    # alpha * q_t on zero-variance groups whose teacher signal is outcome
    # consistent with the group. The injection is token-aligned when turn_q is
    # available; otherwise we preserve the historical row-broadcast fallback.
    base_score = torch.zeros_like(scores)
    zero_var_mask = torch.zeros_like(scores, dtype=torch.bool)
    gate_keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for i in range(bs):
        idx = index[i]
        mu = id2mean[idx]
        sigma = id2std[idx]
        mu_val = float(mu.item())
        q_sum = id2qsum[idx]
        zero_var = sigma.item() < text_feedback_sigma_eps

        # gate_keep: drop the fallback when group_mean and group_q_sum point in
        # opposite directions (teacher disagrees with the outcome distribution
        # at the group level — likely noisy annotation, safer to skip).
        # Spec literal — note the asymmetric `<= 0` / `> 0` boundaries:
        #   the paper §5.1:
        #     gate_keep[i] = not (
        #         (group_mean <= 0 and group_q_sum > 0) or
        #         (group_mean > 0 and group_q_sum < 0)
        #     )
        gate_keep = not ((mu_val <= 0 and q_sum > 0) or (mu_val > 0 and q_sum < 0))
        zero_var_mask[i] = bool(zero_var)
        gate_keep_mask[i] = bool(gate_keep)

        # GRPO base (always computed; ≈0 in zero-var groups).
        # Spec §5.1 uses (sigma + sigma_eps), not the generic epsilon.
        if norm_adv_by_std_in_grpo:
            base_score[i] = (scores[i] - mu) / (sigma + text_feedback_sigma_eps)
        else:
            base_score[i] = scores[i] - mu

    a_l1 = base_score.unsqueeze(-1).expand_as(response_mask).clone()
    fallback_mask = (zero_var_mask & gate_keep_mask).to(dtype=scores.dtype).unsqueeze(-1)
    a_l1 = a_l1 + text_feedback_alpha * q_tokens * fallback_mask

    # ----- L2: turn-level modulation -------------------------------------
    # w² = clamp(1 + λ · sign(base_score) · q_t, 0, 1+λ)
    # Sign-consistency: for a failing trajectory (base_score < 0) carrying a
    # positive q_t (teacher said this turn was good), the magnitude shrinks
    # toward 0; for a successful trajectory (base_score > 0) with negative
    # q_t, it also shrinks. Both are intuitively "soften the verdict".
    sign_base = torch.sign(base_score)
    w2 = (1.0 + text_feedback_lambda * sign_base.unsqueeze(-1) * q_tokens).clamp(
        min=0.0, max=1.0 + text_feedback_lambda
    )
    if has_format_error is not None:
        # Format-error turns must keep w² = 1 so the −0.5 BEACON penalty is
        # not amplified/erased by teacher feedback.
        fmask = has_format_error.to(device=device, dtype=torch.bool)
        if fmask.ndim == 1:
            if fmask.shape[0] != bs:
                raise ValueError(
                    f"has_format_error row mask has shape {tuple(fmask.shape)}, expected ({bs},)"
                )
            w2 = torch.where(fmask.unsqueeze(-1), torch.ones_like(w2), w2)
        elif fmask.ndim == 2:
            if fmask.shape != response_mask.shape:
                raise ValueError(
                    f"has_format_error token mask shape {tuple(fmask.shape)} "
                    f"must match response_mask {tuple(response_mask.shape)}"
                )
            w2 = torch.where(fmask, torch.ones_like(w2), w2)
        else:
            raise ValueError(
                f"has_format_error must be rank-1 or rank-2 bool mask, got shape {tuple(fmask.shape)}"
            )

    a_after_l2 = a_l1 * w2  # (bs, T)
    tf_stats = {}
    tf_mask = response_mask.to(device=device, dtype=torch.bool)
    tf_mask_sum = int(tf_mask.sum().item())
    if tf_mask_sum > 0:
        w2_vals = w2[tf_mask]
        fallback_token_mask = (fallback_mask > 0).expand_as(response_mask)
        effective_token_mask = fallback_token_mask & tf_mask & (q_tokens != 0)
        tf_stats = {
            "w_critique_mean": float(w2_vals.mean().item()),
            "w_critique_std": float(w2_vals.std().item()) if w2_vals.numel() > 1 else 0.0,
            "level1_effective_token_frac": float(
                effective_token_mask.to(dtype=torch.float32).sum().item() / float(tf_mask_sum)
            ),
        }

    # ----- L3: token-level modulation ------------------------------------
    if trace_l3_enable and critique_delta is not None:
        delta = critique_delta.to(device=device, dtype=base_score.dtype)
        delta_clip = delta.clamp(
            min=-trace_l3_kappa, max=trace_l3_kappa
        )
        # w³ = 1 + α₃ · sign(q_t) · |q_t| · δ_clip
        # Note that ``sign(q) · |q|`` recovers ``q`` itself (since q ∈ [-1, 1]),
        # but writing it this way keeps the spec literal so anyone tracing the
        # paper finds the exact expression.
        sign_q = torch.sign(q_tokens)                        # (bs, T)
        abs_q = q_tokens.abs()                               # (bs, T)
        w3_raw = 1.0 + trace_l3_alpha * sign_q * abs_q * delta_clip  # (bs, T)
        w3_raw = w3_raw.clamp(
            min=1.0 - trace_l3_suppress_cap,
            max=1.0 + trace_l3_boost_cap,
        )

        # Row-mean normalize so the *turn-level* advantage magnitude is
        # preserved and L3 only redistributes weight across tokens.
        w_delta = w3_raw / _row_mean_with_mask(w3_raw, response_mask)

        # No critique → row_mean is 1, w³ ≡ 1 / 1 = 1 inside the response.
        # Mask response positions so non-response tokens stay zero.
        advantages = a_after_l2 * w_delta * response_mask

        mask = response_mask.to(device=device, dtype=torch.bool)
        if int(mask.sum().item()) > 0:
            wd_vals = w_delta[mask]
            delta_vals = delta[mask]
            stats = {
                "w_delta_mean": float(wd_vals.mean().item()),
                "w_delta_std": float(wd_vals.std().item()) if wd_vals.numel() > 1 else 0.0,
                "w_delta_min": float(wd_vals.min().item()),
                "w_delta_max": float(wd_vals.max().item()),
                "delta_mean": float(delta_vals.mean().item()),
                "delta_abs_mean": float(delta_vals.abs().mean().item()),
                "delta_abs_max": float(delta_vals.abs().max().item()),
                "alpha3": float(trace_l3_alpha),
                "active_token_frac": float((w_delta[mask] != 1.0).float().mean().item()),
            }
            stats.update(tf_stats)
            set_last_l3_stats(stats)

            step_seen = _next_l3_step_seen()
            if step_seen <= 3 or step_seen % 20 == 0:
                msg = (
                    f"[trace_l3][trace_l3_mask] step#{step_seen} L3 ACTIVE | "
                    f"alpha3={trace_l3_alpha:.3f} kappa={trace_l3_kappa:.2f} | "
                    f"w_delta: mean={stats.get('w_delta_mean', float('nan')):.4f} "
                    f"std={stats.get('w_delta_std', float('nan')):.4f} "
                    f"range=[{stats.get('w_delta_min', float('nan')):.3f}, "
                    f"{stats.get('w_delta_max', float('nan')):.3f}] | "
                    f"delta: |mean|={stats.get('delta_abs_mean', float('nan')):.4f} "
                    f"|max|={stats.get('delta_abs_max', float('nan')):.4f} | "
                    f"active_frac={stats.get('active_token_frac', 0.0):.3f}"
                )
                print(msg, flush=True)
                logger.warning(msg)
    else:
        advantages = a_after_l2 * response_mask

    return advantages, advantages


# ----------------------------------------------------------------------------
# verl-facing wrapper.
# ----------------------------------------------------------------------------


@register_adv_est("trace_l3_mask")
def trace_l3_mask(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = get_advantage_context()  # process-local stash filled by trainer hook
    if ctx is None:
        raise RuntimeError(
            "trace_l3_mask: no advantage context found — the trainer "
            "entry point must call trace_grpo.patches.context.set_advantage_context "
            "with per_turn_q (and optionally critique_delta + has_format_error) "
            "before compute_advantage runs."
        )

    # Hyperparameters can come from either `config.algorithm.trace_grpo.*` (if
    # the user runs hydra in non-strict mode and adds `trace_grpo:` to the
    # algorithm yaml block) or from environment variables (spec PDF §9.2 —
    # the TRACE_GRPO_L3_* knobs). Env vars win over config to make smoke runs
    # easy to override.
    cfg = {}
    if config is not None:
        try:
            cfg = config.get("trace_grpo", None) or {}
        except Exception:
            cfg = {}

    return trace_l3_mask_compute(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        per_turn_q=ctx.per_turn_q,
        turn_q=ctx.turn_q,
        critique_delta=ctx.critique_delta,
        has_format_error=ctx.has_format_error,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        text_feedback_alpha=_env_float(
            "TRACE_GRPO_L1_ALPHA", cfg.get("text_feedback_alpha", 0.5),
        ),
        text_feedback_lambda=_env_float(
            "TRACE_GRPO_L2_LAMBDA", cfg.get("text_feedback_lambda", 0.2),
        ),
        text_feedback_sigma_eps=_env_float(
            "TRACE_GRPO_L1_SIGMA_EPS", cfg.get("text_feedback_sigma_eps", 1e-3)
        ),
        trace_l3_alpha=_env_float("TRACE_GRPO_L3_ALPHA", cfg.get("trace_l3_alpha", 0.3)),
        trace_l3_kappa=_env_float("TRACE_GRPO_L3_KAPPA", cfg.get("trace_l3_kappa", 1.0)),
        trace_l3_enable=_env_bool("TRACE_GRPO_L3_ENABLE", cfg.get("trace_l3_enable", True)),
        trace_l3_suppress_cap=_env_float(
            "TRACE_GRPO_L3_SUPPRESS_CAP", cfg.get("trace_l3_suppress_cap", 0.3)
        ),
        trace_l3_boost_cap=_env_float(
            "TRACE_GRPO_L3_BOOST_CAP", cfg.get("trace_l3_boost_cap", 0.1)
        ),
    )
