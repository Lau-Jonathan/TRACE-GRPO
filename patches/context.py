"""Process-local stash for TRACE-GRPO advantage-estimator inputs.

verl's plug-in adv-estimator API only forwards ``token_level_rewards``,
``response_mask``, ``index``, ``epsilon``, ``norm_adv_by_std_in_grpo``,
``config``. To pass TRACE-GRPO's per-turn / per-token signals through that
narrow channel without modifying verl, the trainer entry point fills this
context just before calling ``compute_advantage``; the registered estimator
reads from it and clears it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch

_LOCAL = threading.local()


@dataclass
class AdvantageContext:
    """Per-step bag of tensors handed off to ``trace_l3_mask``.

    Shapes:
      - ``per_turn_q``: ``(bs,)`` float, ``q_t ∈ [-1, +1]`` (0 if missing).
      - ``turn_q``: ``(bs, response_length)`` float, token-aligned per-turn
        q values. Optional for backward compatibility; when absent the
        estimator broadcasts ``per_turn_q`` across each row.
      - ``critique_delta``: ``(bs, response_length)`` float, or ``None``.
      - ``has_format_error``: either ``(bs,)`` bool (row-level legacy mask)
        or ``(bs, response_length)`` bool (token-level turn mask), or
        ``None``.
    """

    per_turn_q: torch.Tensor
    turn_q: Optional[torch.Tensor] = None
    critique_delta: Optional[torch.Tensor] = None
    has_format_error: Optional[torch.Tensor] = None


def set_advantage_context(ctx: AdvantageContext) -> None:
    _LOCAL.ctx = ctx


def get_advantage_context() -> Optional[AdvantageContext]:
    return getattr(_LOCAL, "ctx", None)


def clear_advantage_context() -> None:
    if hasattr(_LOCAL, "ctx"):
        del _LOCAL.ctx
