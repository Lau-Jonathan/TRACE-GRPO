"""LLM-judge teacher: drop-in replacement for the env-score-delta teacher.

Same ``annotate(batch)`` signature as
:class:`~trace_grpo.self_supervised.env_score_delta_annotator.EnvScoreDeltaAnnotator`,
so the trainer hook can swap teachers via config.

Key differences from env-score-delta:
  - q_t is ``decision_quality * confidence`` from the v3 LLM schema.
    decision_quality is snapped to {-1, -0.7, -0.3, 0, +0.3, +0.7, +1};
    confidence is snapped to {0, 0.25, 0.5, 0.75, 1}.
  - v3 labels may be sparse or dense depending on the judge's evidence.
    Unlabeled turns keep q_t = 0, while labeled turns provide the L2 / L3
    trajectory-aligned critique signal.
  - The critique text is the LLM's own ``rationale`` field, which is
    much richer than the env-score templated one-liners.

Per-trajectory aggregation to ``per_sample_q``: same ``sum`` rule as
:class:`EnvScoreDeltaAnnotator` (preserves the literal L1 ``group_q_sum``
semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from trace_grpo.patches.critique_conditioned_provider import (
    TrajectoryAnnotation,
    TurnAnnotation,
)
from trace_grpo.self_supervised.env_score_delta_annotator import (
    TrajectoryRecord,
    wrap_teacher_note,
)

from ._annotator import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    JudgeRequest,
    JudgeResponse,
    LLMJudgeClient,
)
from .prompt import (
    SYSTEM_PROMPT,
    V3_SYSTEM_PROMPT,
    build_user_message,
    build_v3_user_message,
    parse_judgment,
    parse_v3_trajectory_judgment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration / construction.
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgeAnnotator:
    """Teacher backed by a remote OpenAI-compatible LLM.

    The trainer hook calls ``annotate(batch)`` once per training step
    *after* rollout has populated ``batch.non_tensor_batch["trajectory_records"]``
    with full :class:`TrajectoryRecord` objects (including raw text fields
    on each :class:`TurnRecord`).

    Args:
        tokenizer: HF-compatible tokenizer (``encode`` is the only method
            used). Required if ``l3_enable=True``.
        l3_enable: whether to populate ``critique_token_ids`` for the L3
            forward.
        base_url, model, api_key, max_workers, max_retries, timeout_s:
            forwarded to :class:`LLMJudgeClient`.
        max_judge_tokens: upper bound on judge output tokens. v3 uses one
            trajectory-level JSON response, so long-horizon ScienceWorld runs
            need a much larger budget than the legacy one-turn schema.
        temperature: judge temperature; 0 is deterministic.
        client: optional pre-built :class:`LLMJudgeClient` (used by
            tests to inject a mock).
        schema_version: ``"v3"`` uses the tracegrpo trajectory-level JSON
            schema. ``"legacy"`` keeps the old one-request-per-turn XML path.
    """

    tokenizer: Optional[object] = None
    l3_enable: bool = True
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    max_workers: int = DEFAULT_MAX_WORKERS
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_judge_tokens: int = 256
    temperature: float = 0.0
    client: Optional[LLMJudgeClient] = None
    emit_turn_annotations: bool = False
    schema_version: str = "v3"
    per_sample_q_aggregator: str = "sum"

    # diagnostics — populated after each annotate() call
    last_total_requests: int = field(default=0, init=False)
    last_failed_requests: int = field(default=0, init=False)
    last_valid_parsed_requests: int = field(default=0, init=False)
    last_invalid_parsed_requests: int = field(default=0, init=False)
    last_outcome_compared_requests: int = field(default=0, init=False)
    last_outcome_inconsistent_requests: int = field(default=0, init=False)
    last_turns_total: int = field(default=0, init=False)
    last_turns_nonzero_q: int = field(default=0, init=False)
    last_turns_pos_q: int = field(default=0, init=False)
    last_turns_neg_q: int = field(default=0, init=False)
    last_turns_zero_q: int = field(default=0, init=False)
    last_turns_outcome_gated: int = field(default=0, init=False)
    last_trajectory_count: int = field(default=0, init=False)
    last_labels_per_trajectory_mean: float = field(default=0.0, init=False)
    last_labels_per_trajectory_std: float = field(default=0.0, init=False)
    last_labels_per_trajectory_p50: float = field(default=0.0, init=False)
    last_labels_per_trajectory_p95: float = field(default=0.0, init=False)
    last_judgment_tag_counts: Dict[str, int] = field(default_factory=dict, init=False)
    last_judgment_tag_total: int = field(default=0, init=False)
    last_mean_q: float = field(default=0.0, init=False)
    last_mean_abs_q: float = field(default=0.0, init=False)
    last_latency_mean_s: float = field(default=0.0, init=False)
    last_latency_p95_s: float = field(default=0.0, init=False)
    last_latency_max_s: float = field(default=0.0, init=False)
    _invalid_response_debug_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = LLMJudgeClient(
                base_url=self.base_url,
                model=self.model,
                api_key=self.api_key,
                max_workers=self.max_workers,
                max_retries=self.max_retries,
                timeout_s=self.timeout_s,
            )
        # Startup self-check line for judge config replay/debug.
        client_mode = (
            "max_completion_tokens"
            if getattr(self.client, "uses_reasoning_token_param", lambda: False)()
            else "max_tokens"
        )
        msg = (
            "[annotator_patch] LLMJudgeAnnotator init | "
            f"schema={self.schema_version} model={self.model} base_url={self.base_url} "
            f"workers={self.max_workers} retries={self.max_retries} timeout_s={self.timeout_s:.1f} "
            f"max_judge_tokens={self.max_judge_tokens} temperature={self.temperature:.2f} "
            f"token_budget_key={client_mode}"
        )
        print(msg, flush=True)
        logger.warning(msg)

    def _reset_diagnostics(self) -> None:
        self.last_total_requests = 0
        self.last_failed_requests = 0
        self.last_valid_parsed_requests = 0
        self.last_invalid_parsed_requests = 0
        self.last_outcome_compared_requests = 0
        self.last_outcome_inconsistent_requests = 0
        self.last_turns_total = 0
        self.last_turns_nonzero_q = 0
        self.last_turns_pos_q = 0
        self.last_turns_neg_q = 0
        self.last_turns_zero_q = 0
        self.last_turns_outcome_gated = 0
        self.last_trajectory_count = 0
        self.last_labels_per_trajectory_mean = 0.0
        self.last_labels_per_trajectory_std = 0.0
        self.last_labels_per_trajectory_p50 = 0.0
        self.last_labels_per_trajectory_p95 = 0.0
        self.last_judgment_tag_counts.clear()
        self.last_judgment_tag_total = 0
        self.last_mean_q = 0.0
        self.last_mean_abs_q = 0.0
        self.last_latency_mean_s = 0.0
        self.last_latency_p95_s = 0.0
        self.last_latency_max_s = 0.0
        self._invalid_response_debug_count = 0

    def _log_invalid_response_sample(self, *, request_id: str, text: str) -> None:
        limit = int(os.environ.get("TEXT_FEEDBACK_LOG_INVALID_SAMPLES", "3"))
        if limit <= 0 or self._invalid_response_debug_count >= limit:
            return
        self._invalid_response_debug_count += 1
        snippet = " ".join((text or "").strip().split())[:800]
        logger.warning(
            "[annotator_patch] invalid judge response sample #%d request_id=%s len=%d text=%r",
            self._invalid_response_debug_count,
            request_id,
            len(text or ""),
            snippet,
        )

    def _accumulate_judgment_tags(self, tags: Sequence[str]) -> None:
        for raw_tag in tags:
            tag = str(raw_tag).strip().lower().replace(" ", "_").replace("-", "_")
            if not tag:
                continue
            self.last_judgment_tag_counts[tag] = self.last_judgment_tag_counts.get(tag, 0) + 1
            self.last_judgment_tag_total += 1

    def _record_response_latencies(self, responses: Sequence[JudgeResponse]) -> None:
        if not responses:
            self.last_latency_mean_s = 0.0
            self.last_latency_p95_s = 0.0
            self.last_latency_max_s = 0.0
            return
        latency_np = np.asarray([max(float(r.latency_s), 0.0) for r in responses], dtype=np.float32)
        self.last_latency_mean_s = float(latency_np.mean())
        self.last_latency_p95_s = float(np.percentile(latency_np, 95))
        self.last_latency_max_s = float(latency_np.max())

    # -- public API consumed by the trainer hook ---------------------------

    def annotate(
        self,
        batch,
    ) -> tuple[np.ndarray, Optional[List[TrajectoryAnnotation]]]:
        """Score every turn via the LLM judge.

        Steps:
          1. Walk each :class:`TrajectoryRecord` in
             ``batch.non_tensor_batch["trajectory_records"]``.
          2. Build one :class:`JudgeRequest` per turn (carrying a unique
             ``request_id`` like ``"{traj_offset}:{history_index}"``).
          3. Fire all requests through the async client.
          4. Parse responses; failed / malformed judgments are skipped
             (q_t = 0, no critique).
          5. Aggregate per-trajectory: ``per_sample_q[i] = Σ turn_q``.
          6. If ``l3_enable``, build :class:`TrajectoryAnnotation` with
             ``<teacher_note>{rationale}</teacher_note>`` token ids.
        """
        self._reset_diagnostics()
        records: Sequence[TrajectoryRecord] = batch.non_tensor_batch.get("trajectory_records")
        if records is None:
            raise KeyError(
                "LLMJudgeAnnotator: batch.non_tensor_batch is missing "
                "'trajectory_records' — make sure the agent loop populated it"
            )
        if self.schema_version == "legacy":
            return self._annotate_legacy(batch, records)
        if self.schema_version != "v3":
            raise ValueError(f"Unsupported LLM judge schema_version={self.schema_version!r}")

        return self._annotate_v3(batch, records)

    # -- v3 tracegrpo path -------------------------------------------------

    def _platform_success(self, batch, traj: TrajectoryRecord) -> Optional[bool]:
        row = int(traj.traj_offset_in_batch)
        # Spec literal (§2): platform_success := 1[R_i > 0], where R_i is the
        # trajectory-level scalar reward used by GRPO.
        batch_tensors = getattr(batch, "batch", None)
        if batch_tensors is not None:
            for key in ("token_level_scores", "token_level_rewards"):
                tensor = batch_tensors.get(key)
                if tensor is None:
                    continue
                try:
                    if 0 <= row < int(tensor.shape[0]):
                        reward_sum = float(tensor[row].sum().detach().cpu().item())
                        return bool(reward_sum > 0.0)
                except Exception:
                    continue

        # Compatibility fallback when called outside the trainer path.
        if traj.won is not None:
            return bool(traj.won)
        won = batch.non_tensor_batch.get("won")
        if won is not None:
            try:
                return bool(won[row])
            except Exception:
                pass
        if traj.turns:
            return bool(traj.turns[-1].score > 0)
        return None

    def _aggregate_per_sample_q(self, q_values: List[float]) -> float:
        if self.per_sample_q_aggregator == "sum":
            return float(sum(q_values))
        if self.per_sample_q_aggregator == "sign_sum":
            return float(np.sign(float(sum(q_values))))
        raise ValueError(
            f"Unsupported per_sample_q_aggregator={self.per_sample_q_aggregator!r}; "
            "expected 'sum' or 'sign_sum'."
        )

    def _annotate_v3(
        self,
        batch,
        records: Sequence[TrajectoryRecord],
    ) -> tuple[np.ndarray, Optional[List[TrajectoryAnnotation]]]:
        bs = len(records)

        requests: List[JudgeRequest] = []
        for traj in records:
            if traj.traj_offset_in_batch < 0 or traj.traj_offset_in_batch >= bs:
                raise IndexError(
                    f"trajectory offset {traj.traj_offset_in_batch} out of [0, {bs})"
                )
            show_score = os.environ.get("TEXT_FEEDBACK_SHOW_SCORE", "1").strip() not in ("0", "false", "False")
            turns_payload = []
            prev_score = float(traj.score_initial)
            for turn in traj.turns:
                curr_score = float(turn.score)
                delta = curr_score - prev_score
                entry = {
                    "turn_index": turn.history_index,
                    "action": turn.action_text,
                    "env_response": turn.env_response_text,
                }
                if show_score:
                    entry["score"] = curr_score
                    entry["score_delta"] = delta
                turns_payload.append(entry)
                prev_score = curr_score
            final_score = traj.turns[-1].score if traj.turns else 0.0
            platform_won = bool(traj.won) if traj.won is not None else (final_score > 0)
            user_msg = build_v3_user_message(
                task_description=traj.task_description,
                initial_observation=traj.initial_observation,
                turns=turns_payload,
                task_succeeded=platform_won if show_score else None,
                final_score=final_score if show_score else None,
            )
            requests.append(
                JudgeRequest(
                    system_prompt=V3_SYSTEM_PROMPT,
                    user_message=user_msg,
                    max_tokens=self.max_judge_tokens,
                    temperature=self.temperature,
                    request_id=str(traj.traj_offset_in_batch),
                )
            )

        responses: List[JudgeResponse] = (
            self.client.run_batch_sync(requests) if requests else []
        )
        self.last_total_requests = len(requests)
        self.last_failed_requests = sum(1 for r in responses if r.text is None)
        self._record_response_latencies(responses)

        parsed_by_row = {}
        for resp in responses:
            if resp.text is None:
                continue
            try:
                row = int(resp.request_id)
            except ValueError:
                continue
            parsed = parse_v3_trajectory_judgment(resp.text)
            parsed_by_row[row] = parsed
            if parsed.is_valid:
                self.last_valid_parsed_requests += 1
            else:
                self.last_invalid_parsed_requests += 1
                self._log_invalid_response_sample(request_id=resp.request_id, text=resp.text)

        self.last_invalid_parsed_requests += max(
            self.last_total_requests - self.last_failed_requests - len(parsed_by_row),
            0,
        )

        per_sample_q = np.zeros(bs, dtype=np.float32)
        trajectories: List[TrajectoryAnnotation] = []
        turn_q_sum = 0.0
        turn_abs_q_sum = 0.0
        nonzero_labels_per_traj: List[int] = []

        for traj in records:
            row = traj.traj_offset_in_batch
            parsed = parsed_by_row.get(row)
            turn_lookup = {}
            outcome_consistent = True
            if parsed is None or not parsed.is_valid:
                outcome_consistent = False
            else:
                for parsed_turn in parsed.turns:
                    if parsed_turn.is_valid:
                        self._accumulate_judgment_tags(parsed_turn.judgment_tags)
                platform_success = self._platform_success(batch, traj)
                if platform_success is not None and parsed.agent_succeeded is not None:
                    self.last_outcome_compared_requests += 1
                    outcome_consistent = bool(parsed.agent_succeeded) == bool(platform_success)
                    if not outcome_consistent:
                        self.last_outcome_inconsistent_requests += 1
                turn_lookup = {t.turn_index: t for t in parsed.turns}

            q_values: List[float] = []
            turn_anns: List[TurnAnnotation] = []
            traj_nonzero = 0
            for turn in traj.turns:
                self.last_turns_total += 1
                judg = turn_lookup.get(turn.history_index)
                if outcome_consistent and judg is not None and judg.is_valid:
                    q_t = float(judg.q)
                    rationale = judg.rationale
                else:
                    q_t = 0.0
                    rationale = ""
                    if not outcome_consistent:
                        self.last_turns_outcome_gated += 1
                if q_t != 0.0:
                    self.last_turns_nonzero_q += 1
                    traj_nonzero += 1
                    if q_t > 0.0:
                        self.last_turns_pos_q += 1
                    else:
                        self.last_turns_neg_q += 1
                else:
                    self.last_turns_zero_q += 1
                q_values.append(float(q_t))
                turn_q_sum += q_t
                turn_abs_q_sum += abs(q_t)
                if self.l3_enable and self.tokenizer is not None and rationale:
                    critique_ids = self.tokenizer.encode(
                        wrap_teacher_note(rationale),
                        add_special_tokens=False,
                    )
                else:
                    critique_ids = []
                turn_anns.append(
                    TurnAnnotation(
                        history_index=turn.history_index,
                        response_start=turn.response_start,
                        response_end=turn.response_end,
                        critique_token_ids=critique_ids,
                        q_t=float(q_t),
                        has_format_error=bool(turn.has_format_error),
                    )
                )
            per_sample_q[row] = self._aggregate_per_sample_q(q_values)

            if any(t.critique_token_ids for t in turn_anns) or (
                self.emit_turn_annotations and turn_anns
            ):
                trajectories.append(
                    TrajectoryAnnotation(
                        traj_token_ids=traj.traj_token_ids,
                        prompt_length=traj.prompt_length,
                        response_length=traj.response_length,
                        turns=turn_anns,
                        traj_offset_in_batch=row,
                    )
                )
            nonzero_labels_per_traj.append(traj_nonzero)

        if self.last_turns_total > 0:
            self.last_mean_q = float(turn_q_sum / self.last_turns_total)
            self.last_mean_abs_q = float(turn_abs_q_sum / self.last_turns_total)
        self.last_trajectory_count = bs
        if nonzero_labels_per_traj:
            labels_np = np.asarray(nonzero_labels_per_traj, dtype=np.float32)
            self.last_labels_per_trajectory_mean = float(np.mean(labels_np))
            self.last_labels_per_trajectory_std = float(np.std(labels_np))
            self.last_labels_per_trajectory_p50 = float(np.percentile(labels_np, 50))
            self.last_labels_per_trajectory_p95 = float(np.percentile(labels_np, 95))

        return per_sample_q, (trajectories if trajectories else None)

    # -- legacy per-turn XML path -----------------------------------------

    def _annotate_legacy(
        self,
        batch,
        records: Sequence[TrajectoryRecord],
    ) -> tuple[np.ndarray, Optional[List[TrajectoryAnnotation]]]:
        bs = len(records)

        # ----- step 2: build all judge requests upfront --------------------
        requests: List[JudgeRequest] = []
        for traj in records:
            if traj.traj_offset_in_batch < 0 or traj.traj_offset_in_batch >= bs:
                raise IndexError(
                    f"trajectory offset {traj.traj_offset_in_batch} out of [0, {bs})"
                )
            history_dialogue: list[dict] = []
            for turn in traj.turns:
                user_msg = build_user_message(
                    task_description=traj.task_description,
                    initial_observation=traj.initial_observation,
                    history=list(history_dialogue),  # snapshot before this turn
                    target_history_index=turn.history_index,
                    target_action=turn.action_text,
                    target_env_response=turn.env_response_text,
                )
                req_id = f"{traj.traj_offset_in_batch}:{turn.history_index}"
                requests.append(
                    JudgeRequest(
                        system_prompt=SYSTEM_PROMPT,
                        user_message=user_msg,
                        max_tokens=self.max_judge_tokens,
                        temperature=self.temperature,
                        request_id=req_id,
                    )
                )
                # Append this turn to history so subsequent turns see it.
                if turn.action_text:
                    history_dialogue.append({"role": "assistant", "content": turn.action_text})
                if turn.env_response_text:
                    history_dialogue.append({"role": "user", "content": turn.env_response_text})

        # ----- step 3: async-batch the calls -------------------------------
        responses: List[JudgeResponse] = (
            self.client.run_batch_sync(requests) if requests else []
        )
        self.last_total_requests = len(requests)
        self.last_failed_requests = sum(1 for r in responses if r.text is None)
        self._record_response_latencies(responses)

        # Build (traj_offset, history_index) → judgment lookup.
        judgments: dict[tuple[int, int], tuple[float, str, List[str]]] = {}
        for resp in responses:
            if resp.text is None:
                continue
            try:
                traj_off_str, hist_str = resp.request_id.split(":")
                traj_off = int(traj_off_str)
                hist = int(hist_str)
            except ValueError:
                continue
            judg = parse_judgment(resp.text)
            if not judg.is_valid:
                self.last_invalid_parsed_requests += 1
                continue
            self.last_valid_parsed_requests += 1
            judgments[(traj_off, hist)] = (judg.decision_quality, judg.rationale, judg.judgment_tags)

        self.last_invalid_parsed_requests += max(
            self.last_total_requests
            - self.last_failed_requests
            - self.last_valid_parsed_requests
            - self.last_invalid_parsed_requests,
            0,
        )

        # ----- step 5/6: aggregate + build TrajectoryAnnotation -----------
        per_sample_q = np.zeros(bs, dtype=np.float32)
        trajectories: List[TrajectoryAnnotation] = []
        turn_q_sum = 0.0
        turn_abs_q_sum = 0.0
        nonzero_labels_per_traj: List[int] = []

        for traj in records:
            row = traj.traj_offset_in_batch
            turn_anns: List[TurnAnnotation] = []
            q_values: List[float] = []
            traj_nonzero = 0
            for turn in traj.turns:
                self.last_turns_total += 1
                key = (row, turn.history_index)
                q_t, rationale, tags = judgments.get(key, (0.0, "", []))
                q_values.append(float(q_t))
                turn_q_sum += q_t
                turn_abs_q_sum += abs(q_t)
                if q_t != 0.0:
                    self.last_turns_nonzero_q += 1
                    traj_nonzero += 1
                    if q_t > 0.0:
                        self.last_turns_pos_q += 1
                    else:
                        self.last_turns_neg_q += 1
                else:
                    self.last_turns_zero_q += 1
                if tags:
                    self._accumulate_judgment_tags(tags)
                if self.l3_enable and self.tokenizer is not None and rationale:
                    critique_ids = self.tokenizer.encode(
                        wrap_teacher_note(rationale),
                        add_special_tokens=False,
                    )
                else:
                    critique_ids = []
                turn_anns.append(
                    TurnAnnotation(
                        history_index=turn.history_index,
                        response_start=turn.response_start,
                        response_end=turn.response_end,
                        critique_token_ids=critique_ids,
                        q_t=float(q_t),
                        has_format_error=bool(turn.has_format_error),
                    )
                )
            per_sample_q[row] = self._aggregate_per_sample_q(q_values)

            if any(t.critique_token_ids for t in turn_anns) or (
                self.emit_turn_annotations and turn_anns
            ):
                trajectories.append(
                    TrajectoryAnnotation(
                        traj_token_ids=traj.traj_token_ids,
                        prompt_length=traj.prompt_length,
                        response_length=traj.response_length,
                        turns=turn_anns,
                        traj_offset_in_batch=row,
                    )
                )
            nonzero_labels_per_traj.append(traj_nonzero)

        if self.last_turns_total > 0:
            self.last_mean_q = float(turn_q_sum / self.last_turns_total)
            self.last_mean_abs_q = float(turn_abs_q_sum / self.last_turns_total)
        self.last_trajectory_count = bs
        if nonzero_labels_per_traj:
            labels_np = np.asarray(nonzero_labels_per_traj, dtype=np.float32)
            self.last_labels_per_trajectory_mean = float(np.mean(labels_np))
            self.last_labels_per_trajectory_std = float(np.std(labels_np))
            self.last_labels_per_trajectory_p50 = float(np.percentile(labels_np, 50))
            self.last_labels_per_trajectory_p95 = float(np.percentile(labels_np, 95))

        return per_sample_q, (trajectories if trajectories else None)
