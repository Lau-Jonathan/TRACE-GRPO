"""Trainer-side hook that wires teacher annotations and critique deltas into
the TRACE-GRPO advantage estimator.

verl's :func:`compute_advantage` only forwards a narrow set of arguments to
plug-in advantage estimators. To pass TRACE-GRPO's per-sample tensors
(``per_turn_q``, ``critique_delta``, ``has_format_error``) we use a process-
local context (see :mod:`trace_grpo.patches.context`) that the trainer
populates *just before* calling ``compute_advantage`` and clears immediately
after.

This module exposes one entry point:

    inject_trace_context(batch, *, teacher, l3_provider=None,
                           old_logprobs_key="old_log_probs",
                           response_length=None, device=None)

which the trainer (or a thin wrapper around verl's ``main_ppo``) should
call after ``compute_log_prob`` and before ``compute_advantage``. It:

  1. Pulls per-sample ``has_format_error`` from ``batch.non_tensor_batch``
     (the env loop sets this; default to ``False`` if absent).
  2. Asks ``teacher.annotate(batch)`` for trajectory-level annotations
     (``per_turn_q`` and per-turn ``critique_text``).
  3. Reduces per-turn q to a per-sample tensor via configurable rule
     (sum or sign-of-sum — spec leaves this to the caller).
  4. Optionally invokes the L3 critique-conditioned provider to fill
     ``critique_delta``; if no provider or no annotation, leaves it None.
     When present, also writes it to ``batch.batch["critique_delta"]`` with
     shape ``(B, L_resp)`` as specified by tracegrpo.
  5. Stashes the result via :func:`set_advantage_context`.

The actual model-forward path that produces conditioned log-probs is
plumbed in through ``l3_provider`` so this module remains agnostic to verl
internals.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np
import torch

from .context import AdvantageContext, set_advantage_context


class TeacherProtocol(Protocol):
    """A teacher annotates a verl batch with per-turn q and critique text.

    ``annotate(batch)`` returns:
      - ``per_sample_q``: ``np.ndarray`` of shape ``(bs,)`` with ``q ∈
        [-1, +1]``. **Already aggregated to the trajectory level** (one
        scalar per batch row); how the aggregation is done (sum, sign-of-
        sum, mean) is the teacher's call.
      - ``trajectories``: a list of
        :class:`~trace_grpo.patches.critique_conditioned_provider.TrajectoryAnnotation`
        instances *or* ``None`` if no L3 forward should be run.
    """

    def annotate(self, batch) -> tuple[np.ndarray, Optional[Sequence]]:
        ...


def inject_trace_context(
    batch,
    *,
    teacher: Optional[TeacherProtocol] = None,
    per_sample_q: Optional[np.ndarray] = None,
    trajectories: Optional[Sequence] = None,
    l3_provider=None,
    old_logprobs_key: str = "old_log_probs",
    response_length: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> AdvantageContext:
    """Populate the :class:`AdvantageContext` for one trainer step.

    Args:
        batch: verl ``DataProto``-like object with ``.batch`` (TensorDict) and
            ``.non_tensor_batch`` (dict of np arrays).
        teacher: implements :class:`TeacherProtocol`. Optional when
            ``per_sample_q`` is supplied by the caller.
        per_sample_q: precomputed teacher q values. This lets the trainer
            call the teacher once, use actor-worker RPC to compute L3
            ``critique_delta``, then reuse the same annotations here.
        trajectories: precomputed trajectory annotations paired with
            ``per_sample_q``.
        l3_provider: optional
            :class:`~trace_grpo.patches.critique_conditioned_provider.CritiqueConditionedProvider`
            instance. ``None`` ⇒ skip L3 (advantages reduce to L1+L2).
        old_logprobs_key: name of the baseline log-probs tensor in
            ``batch.batch``. Defaults to verl's ``old_log_probs``.
        response_length: response axis length; if ``None``, inferred from
            ``batch.batch[old_logprobs_key].shape[1]``.
        device: where to place context tensors. Defaults to old_logprobs's
            device.

    Returns:
        The constructed :class:`AdvantageContext` (also stashed into the
        thread-local; the registered estimator will read it).
    """
    # ----- step 1: fetch baseline log-probs and shape info ---------------
    if l3_provider is not None and old_logprobs_key not in batch.batch:
        raise KeyError(
            f"inject_trace_context: L3 enabled but '{old_logprobs_key}' missing from "
            f"batch.batch — make sure compute_log_prob runs before this hook"
        )

    old_logprobs = batch.batch.get(old_logprobs_key) if l3_provider is not None else None
    if old_logprobs is not None:
        bs, inferred_resp_len = old_logprobs.shape
        response_length = response_length or inferred_resp_len
        device = device or old_logprobs.device
    else:
        # No L3 → still need bs / device from somewhere predictable.
        # Pull from response_mask if present; else fall back to per_turn_q below.
        rm = batch.batch.get("response_mask")
        if rm is None:
            raise KeyError(
                "inject_trace_context: cannot determine batch shape; provide "
                "old_log_probs or response_mask in batch.batch"
            )
        bs, inferred_resp_len = rm.shape
        response_length = response_length or inferred_resp_len
        device = device or rm.device

    # ----- step 2: ask the teacher for annotations ----------------------
    if per_sample_q is None:
        if teacher is None:
            raise ValueError("inject_trace_context requires either teacher or per_sample_q")
        per_sample_q_np, trajectories = teacher.annotate(batch)
    else:
        per_sample_q_np = per_sample_q
    per_sample_q_np = np.asarray(per_sample_q_np, dtype=np.float32)
    if per_sample_q_np.shape != (bs,):
        raise ValueError(
            f"teacher.annotate returned per_sample_q of shape {per_sample_q_np.shape}, "
            f"expected ({bs},)"
        )
    per_turn_q = torch.from_numpy(per_sample_q_np).to(device=device)

    # ----- step 3: format-error mask ------------------------------------
    has_format_error: Optional[torch.Tensor] = None
    if response_length is not None:
        recs = batch.non_tensor_batch.get("trajectory_records")
        if recs is None and trajectories is not None:
            recs = trajectories
        if recs is not None:
            fmt_turn = torch.zeros(bs, int(response_length), dtype=torch.bool, device=device)
            for rec in recs:
                row = int(getattr(rec, "traj_offset_in_batch", -1))
                if row < 0 or row >= bs:
                    continue
                for turn in (getattr(rec, "turns", None) or []):
                    if not bool(getattr(turn, "has_format_error", False)):
                        continue
                    start = int(getattr(turn, "response_start", -1))
                    end = int(getattr(turn, "response_end", -1))
                    if start < 0 or end > int(response_length) or end <= start:
                        continue
                    fmt_turn[row, start:end] = True
            if bool(fmt_turn.any().item()):
                has_format_error = fmt_turn

    # Fallback: row-level format-error mask (legacy path).
    if has_format_error is None:
        fmt_np = batch.non_tensor_batch.get("has_format_error")
        if fmt_np is not None:
            fmt_arr = np.asarray(fmt_np, dtype=bool)
            if fmt_arr.shape != (bs,):
                raise ValueError(
                    f"non_tensor_batch['has_format_error'] shape {fmt_arr.shape} != ({bs},)"
                )
            has_format_error = torch.from_numpy(fmt_arr).to(device=device)

    # ----- step 4: optional L3 critique-conditioned forward -------------
    turn_q: Optional[torch.Tensor] = None
    if trajectories and response_length is not None:
        turn_q = torch.zeros(bs, int(response_length), dtype=per_turn_q.dtype, device=device)
        for traj in trajectories:
            row = int(traj.traj_offset_in_batch)
            if row < 0 or row >= bs:
                raise IndexError(
                    f"trajectory offset {row} out of range for batch size {bs}"
                )
            for turn in traj.turns:
                start = int(turn.response_start)
                end = int(turn.response_end)
                if start < 0 or end > int(response_length) or end < start:
                    raise IndexError(
                        f"turn span [{start}, {end}) is outside response axis "
                        f"[0, {response_length})"
                    )
                if end > start:
                    turn_q[row, start:end] = float(turn.q_t)

    critique_delta: Optional[torch.Tensor] = batch.batch.get("critique_delta")
    if critique_delta is not None:
        if tuple(critique_delta.shape) != (bs, int(response_length)):
            raise RuntimeError(
                f"batch.batch['critique_delta'] shape {tuple(critique_delta.shape)}, "
                f"expected ({bs}, {int(response_length)})"
            )
        critique_delta = critique_delta.to(device=device)
    if l3_provider is not None and trajectories:
        critique_delta = l3_provider.compute_critique_delta(
            trajectories=trajectories,
            old_logprobs=old_logprobs,
        )
        if tuple(critique_delta.shape) != (bs, int(response_length)):
            raise RuntimeError(
                f"l3_provider returned critique_delta shape {tuple(critique_delta.shape)}, "
                f"expected ({bs}, {int(response_length)})"
            )
        batch.batch["critique_delta"] = critique_delta

    ctx = AdvantageContext(
        per_turn_q=per_turn_q,
        turn_q=turn_q,
        critique_delta=critique_delta,
        has_format_error=has_format_error,
    )
    set_advantage_context(ctx)
    return ctx
