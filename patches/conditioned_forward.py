"""Delta-provider abstraction for TRACE-GRPO L3.

A *delta provider* is the thing that produces, for one packed trajectory,
a per-token log-prob shift tensor

    δ_{k} = log p_θ(token_k | prefix_k, *condition_t*) − log p_θ(token_k | prefix_k)

where ``condition_t`` is whatever the chosen teacher decides should be
spliced before the response of turn ``t``. Two providers ship in this repo:

  - ``critique_v1`` — splice ``<teacher_note>critique_t</teacher_note>``
    before the response. Used by both the LLM teacher and the env-score
    teacher (they only differ in how ``critique_t`` is generated).
  - ``counterfactual_mask_v1`` — produce ``q_t^cf`` by *removing* turn ``t``
    from the trajectory and reading the model's belief shift on the outcome
    anchor. The "critique text" for L3 in this mode is the templated
    ``"Hindsight: removing this turn would change the predicted outcome by
    q_cf={value:.2f}."`` (spec PDF §D.3).

The provider runs on the actor worker (it needs the full model + KV cache)
and is *only* invoked when ``trace_l3_enable=True`` and at least one
turn in the batch has a non-empty critique. Trajectories with all-empty
critiques get a zero δ tensor at zero cost.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import torch

logger = logging.getLogger(__name__)

DELTA_PROVIDER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_delta_provider(name: str):
    """Register a delta provider class under a string name.

    Use::

        @register_delta_provider("critique_v1")
        class CritiqueConditionedProvider:
            ...
    """
    def _decorator(cls):
        if name in DELTA_PROVIDER_REGISTRY and DELTA_PROVIDER_REGISTRY[name] is not cls:
            raise ValueError(
                f"register_delta_provider: '{name}' already registered to "
                f"{DELTA_PROVIDER_REGISTRY[name]} (got {cls})"
            )
        DELTA_PROVIDER_REGISTRY[name] = cls
        return cls
    return _decorator


def get_delta_provider(name: str):
    """Look up a delta provider class by name.

    Raises:
        KeyError: if ``name`` was never registered.
    """
    if name not in DELTA_PROVIDER_REGISTRY:
        raise KeyError(
            f"get_delta_provider: unknown provider '{name}'. "
            f"Registered: {sorted(DELTA_PROVIDER_REGISTRY)}"
        )
    return DELTA_PROVIDER_REGISTRY[name]


# Map teacher kind → default delta provider.
TEACHER_TO_PROVIDER: Dict[str, str] = {
    "llm": "critique_v1",
    "env_score": "critique_v1",
    "counterfactual": "counterfactual_mask_v1",
}


def resolve_delta_provider(teacher_kind: str) -> str:
    """Pick the default delta provider for a given teacher kind."""
    if teacher_kind not in TEACHER_TO_PROVIDER:
        raise KeyError(
            f"resolve_delta_provider: unknown teacher_kind '{teacher_kind}'. "
            f"Known: {sorted(TEACHER_TO_PROVIDER)}"
        )
    return TEACHER_TO_PROVIDER[teacher_kind]


class PolicyDisagreementProvider:
    """Fallback delta provider with explicit ``source=...`` diagnostics."""

    def __init__(self):
        self._call_count = 0
        self._none_count = 0

    def compute_delta(
        self,
        *,
        rollout_log_probs: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        source = None
        delta = None
        if (
            rollout_log_probs is not None
            and old_log_probs is not None
            and tuple(rollout_log_probs.shape) == tuple(old_log_probs.shape)
        ):
            delta = rollout_log_probs - old_log_probs
            source = "rollout_minus_old"
        elif entropy is not None:
            delta = -entropy
            source = "entropy_proxy"

        if delta is None:
            self._none_count += 1
            if self._none_count <= 2 or self._none_count % 50 == 0:
                msg = (
                    "[PolicyDisagreementProvider] source=fallback_none | "
                    "returning None (no rollout_minus_old and no entropy_proxy)"
                )
                print(msg, flush=True)
                logger.warning(msg)
            return None

        self._call_count += 1
        if self._call_count <= 2 or self._call_count % 20 == 0:
            msg = (
                f"[PolicyDisagreementProvider] call#{self._call_count} | source={source} | "
                f"shape={tuple(delta.shape)} | delta mean={float(delta.mean().item()):.4f} "
                f"std={float(delta.std().item()):.4f} |max|={float(delta.abs().max().item()):.4f}"
            )
            print(msg, flush=True)
            logger.warning(msg)
        return delta


class HybridProvider:
    """Try primary provider, then fallback provider with one-shot warning."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self._warned_primary_none = False

    def compute_delta(self, **kwargs) -> torch.Tensor | None:
        delta = None
        if self.primary is not None:
            delta = self.primary.compute_delta(**kwargs)
        if delta is not None:
            return delta
        if self.primary is not None and not self._warned_primary_none:
            msg = (
                "[HybridProvider] primary returned None; switching to fallback "
                "(rollout log-prob path may be unavailable)"
            )
            print(msg, flush=True)
            logger.warning(msg)
            self._warned_primary_none = True
        if self.fallback is None:
            return None
        return self.fallback.compute_delta(**kwargs)
