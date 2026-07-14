"""Counterfactual delta-provider registration.

the paper maps ``teacher_kind=counterfactual`` to registry name
``counterfactual_mask_v1``. The runtime data contract is identical to the
``critique_v1`` provider: actor-side code materializes per-turn hindsight
critique tokens and q values, then L3 consumes standard ``TrajectoryAnnotation``
objects to build ``critique_delta``.
"""

from __future__ import annotations

from .conditioned_forward import register_delta_provider
from .critique_conditioned_provider import CritiqueConditionedProvider


@register_delta_provider("counterfactual_mask_v1")
class CounterfactualMaskProvider(CritiqueConditionedProvider):
    """Alias provider for counterfactual teacher mode."""

