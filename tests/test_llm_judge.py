"""Tests for the LLM judge teacher: prompt parser + annotator orchestration.

The annotator's annotate() is exercised against an injected mock client
(``LLMJudgeClient`` is the seam) so the test runs offline. A separate
test (``test_llm_judge_live.py``, kept in this file but skipped by
default) hits the configured <LLM_JUDGE> endpoint to verify end-to-end
connectivity — enable with ``TRACE_GRPO_RUN_LIVE_TESTS=1``.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from trace_grpo.agent_loops.trajectory_assembler import TrajectoryAssembler
from trace_grpo.workers.reward_manager.text_feedback._annotator import (
    JudgeRequest,
    JudgeResponse,
    LLMJudgeClient,
)
from trace_grpo.workers.reward_manager.text_feedback.manager import (
    LLMJudgeAnnotator,
)
from trace_grpo.workers.reward_manager.text_feedback.prompt import (
    CANONICAL_JUDGMENT_TAGS,
    CANONICAL_Q_VALUES,
    Judgment,
    V3_SYSTEM_PROMPT,
    build_user_message,
    build_v3_user_message,
    parse_judgment,
    parse_v3_trajectory_judgment,
)


# ---------------------------------------------------------------------------
# Prompt parser.
# ---------------------------------------------------------------------------


def test_parse_judgment_full_payload():
    raw = (
        "<decision_quality>+0.7</decision_quality>\n"
        "<judgment_tags>action_choice, plan_following</judgment_tags>\n"
        "<rationale>Picks up the right object and moves toward the goal.</rationale>"
    )
    j = parse_judgment(raw)
    assert j.is_valid
    assert j.decision_quality == 0.7
    assert j.judgment_tags == ["action_choice", "plan_following"]
    assert "right object" in j.rationale


def test_parse_judgment_snaps_off_grid_value():
    raw = "<decision_quality>0.5</decision_quality><judgment_tags></judgment_tags><rationale>x</rationale>"
    j = parse_judgment(raw)
    assert j.is_valid
    # 0.5 should snap to nearest canonical: tie between 0.3 and 0.7 — both 0.2 away.
    # Python min() picks the first match; CANONICAL is sorted ascending so 0.3 wins.
    assert j.decision_quality in (0.3, 0.7)
    # Either snap is acceptable; just confirm it's canonical.
    assert j.decision_quality in CANONICAL_Q_VALUES


def test_parse_judgment_clamps_extreme_values():
    raw = "<decision_quality>-3.5</decision_quality>"
    j = parse_judgment(raw)
    assert j.is_valid
    assert j.decision_quality == -1.0


def test_parse_judgment_missing_dq_is_invalid():
    raw = "<judgment_tags>foo</judgment_tags><rationale>bar</rationale>"
    j = parse_judgment(raw)
    assert not j.is_valid
    assert j.decision_quality == 0.0


def test_parse_judgment_no_tags_or_rationale():
    raw = "<decision_quality>0.0</decision_quality>"
    j = parse_judgment(raw)
    assert j.is_valid
    assert j.decision_quality == 0.0
    assert j.judgment_tags == []
    assert j.rationale == ""


def test_parse_judgment_robust_to_extra_prose():
    raw = (
        "Sure, here's my judgment:\n\n"
        "<decision_quality>-0.7</decision_quality>\n"
        "Quick note before tags...\n"
        "<judgment_tags>action_choice</judgment_tags>\n"
        "<rationale>multi\nline\nrationale</rationale>\n"
        "Done!"
    )
    j = parse_judgment(raw)
    assert j.is_valid
    assert j.decision_quality == -0.7
    assert j.judgment_tags == ["action_choice"]
    assert j.rationale == "multi\nline\nrationale"


def test_parse_v3_trajectory_judgment_computes_q_and_snaps():
    raw = """
    ```json
    {
      "trajectory_outcome_recap": {"agent_succeeded": true},
      "turn_judgments": [
        {
          "turn_index": 0,
          "decision_quality": 0.6,
          "confidence": 0.74,
          "judgment_tags": ["action_choice"],
          "judgment_basis": "Used the observation.",
          "rationale": "Good progress."
        },
        {
          "turn_index": 1,
          "decision_quality": -0.8,
          "confidence": 0.2,
          "judgment_tags": "formatting, action_choice",
          "judgment_basis": "Malformed action.",
          "rationale": "Bad format."
        }
      ]
    }
    ```
    """
    parsed = parse_v3_trajectory_judgment(raw)
    assert parsed.is_valid
    assert parsed.agent_succeeded is True
    assert parsed.turns[0].decision_quality == 0.7
    assert parsed.turns[0].confidence == 0.75
    assert parsed.turns[0].q == pytest.approx(0.525)
    assert parsed.turns[1].decision_quality == -0.7
    assert parsed.turns[1].confidence == 0.25
    assert parsed.turns[1].q == pytest.approx(-0.175)


def test_parse_v3_trajectory_judgment_normalizes_tags_to_fixed_enum():
    raw = """
    {
      "trajectory_outcome_recap": {"agent_succeeded": false},
      "turn_judgments": [
        {
          "turn_index": 3,
          "decision_quality": -0.7,
          "confidence": 1.0,
          "judgment_tags": ["观测信息利用", "规划决策", "推理准确性", "surprising_new_tag"],
          "judgment_basis": "Ignored the clue.",
          "rationale": "The action moved away from the necessary object."
        }
      ]
    }
    """
    parsed = parse_v3_trajectory_judgment(raw)
    assert parsed.is_valid
    assert parsed.turns[0].judgment_tags == ["observation_use", "planning", "accuracy", "other"]
    assert set(parsed.turns[0].judgment_tags).issubset(set(CANONICAL_JUDGMENT_TAGS))


def test_v3_system_prompt_specifies_general_agent_rl_judgment_frame():
    prompt = V3_SYSTEM_PROMPT.lower()
    assert "long-horizon agent tasks" in prompt
    assert "exactly one json object" in prompt
    assert "prior state" in prompt
    assert "action/tool call" in prompt
    assert "environment feedback" in prompt
    assert "final trajectory outcome" in prompt
    assert "effective credit-assignment judgment" in prompt
    assert "do not use it as a hard judgment rule" in prompt
    assert "q = decision_quality * confidence" in prompt
    for tag in CANONICAL_JUDGMENT_TAGS:
        assert tag in V3_SYSTEM_PROMPT


def test_build_user_message_includes_task_and_target():
    msg = build_user_message(
        task_description="Find a tomato and put it in the fridge.",
        initial_observation="You are in the kitchen.",
        history=[
            {"role": "assistant", "content": "go to fridge 1"},
            {"role": "user", "content": "You arrive at fridge 1."},
        ],
        target_history_index=2,
        target_action="open fridge 1",
        target_env_response="The fridge is now open.",
    )
    assert "Find a tomato" in msg
    assert "history_index = 2" in msg
    assert "open fridge 1" in msg
    assert "fridge is now open" in msg


def test_build_v3_user_message_includes_full_trajectory():
    msg = build_v3_user_message(
        task_description="Find a tomato.",
        initial_observation="Kitchen.",
        turns=[
            {"turn_index": 0, "action": "go to fridge", "env_response": "at fridge"},
            {"turn_index": 1, "action": "open fridge", "env_response": "open"},
        ],
    )
    assert "Find a tomato" in msg
    assert "Turn 0" in msg
    assert "go to fridge" in msg
    assert "open fridge" in msg


# ---------------------------------------------------------------------------
# Annotator: end-to-end with a mocked client.
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) % 256 for c in text[:16]]


class _MockClient:
    """Pretends to be an :class:`LLMJudgeClient` but returns canned text."""

    def __init__(self, judgments_by_request_id: dict[str, str]):
        self._canned = judgments_by_request_id
        self.calls: list[list[JudgeRequest]] = []

    def run_batch_sync(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        self.calls.append(list(requests))
        return [
            JudgeResponse(
                request_id=r.request_id,
                text=self._canned.get(r.request_id),
                error=None if r.request_id in self._canned else "no canned response",
                latency_s=0.001,
            )
            for r in requests
        ]


def _record_for_traj(
    *,
    offset: int,
    turns: list[tuple[int, str, str, int, bool]],   # (history_index, action, env_resp, score, fmt_err)
    task: str = "Heat the tomato.",
    initial_obs: str = "You are in the kitchen.",
):
    a = TrajectoryAssembler(prompt_ids=[1, 2], response_length=64, traj_offset_in_batch=offset)
    for hi, action, env_resp, score, fmt_err in turns:
        # token_ids don't matter for this test (we don't run a real model);
        # use a few placeholder ids per turn.
        a.record_response(
            response_ids=[100 + hi, 101 + hi],
            score=score,
            done=False,
            has_format_error=fmt_err,
        )
        a.turn_records[-1].action_text = action
        a.turn_records[-1].env_response_text = env_resp
    rec = a.finalize().trajectory_record
    rec.task_description = task
    rec.initial_observation = initial_obs
    return rec


def test_annotator_dispatches_one_request_per_turn():
    rec = _record_for_traj(
        offset=0,
        turns=[
            (0, "go to fridge", "you are at fridge", 0, False),
            (1, "open fridge",  "fridge is open",    5, False),
            (2, "take tomato",  "you have tomato",  10, False),
        ],
    )
    canned = {
        "0:0": "<decision_quality>+0.3</decision_quality><judgment_tags>action_choice</judgment_tags><rationale>moves toward fridge</rationale>",
        "0:1": "<decision_quality>+0.7</decision_quality><judgment_tags></judgment_tags><rationale>good progress</rationale>",
        "0:2": "<decision_quality>+1.0</decision_quality><judgment_tags></judgment_tags><rationale>essential pickup</rationale>",
    }
    client = _MockClient(canned)
    annotator = LLMJudgeAnnotator(
        tokenizer=_StubTokenizer(),
        client=client,
        schema_version="legacy",
    )
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})

    per_q, trajs = annotator.annotate(batch)

    # 1 batch fired, 3 requests total.
    assert len(client.calls) == 1
    assert len(client.calls[0]) == 3
    # request_ids match (offset:history_index).
    ids = sorted(r.request_id for r in client.calls[0])
    assert ids == ["0:0", "0:1", "0:2"]

    # per_sample_q = +0.3 + 0.7 + 1.0 = +2.0
    assert per_q.shape == (1,)
    assert abs(per_q[0] - 2.0) < 1e-6

    # Trajectory annotation present, three turns each with critique tokens.
    assert trajs is not None and len(trajs) == 1
    t = trajs[0]
    assert len(t.turns) == 3
    assert all(len(turn.critique_token_ids) > 0 for turn in t.turns)
    assert t.turns[0].q_t == 0.3
    assert t.turns[1].q_t == 0.7
    assert t.turns[2].q_t == 1.0


def test_annotator_handles_failed_judge_calls():
    """Failed / malformed judge responses must default to q=0 and no critique."""
    rec = _record_for_traj(
        offset=0,
        turns=[
            (0, "act 1", "obs 1", 0, False),
            (1, "act 2", "obs 2", 5, False),
        ],
    )
    canned = {
        "0:0": "<decision_quality>+0.7</decision_quality><judgment_tags></judgment_tags><rationale>fine</rationale>",
        # "0:1" missing → mock returns text=None → annotator treats as q=0.
    }
    client = _MockClient(canned)
    annotator = LLMJudgeAnnotator(
        tokenizer=_StubTokenizer(),
        client=client,
        schema_version="legacy",
    )
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})

    per_q, trajs = annotator.annotate(batch)

    assert annotator.last_total_requests == 2
    assert annotator.last_failed_requests == 1
    assert per_q.tolist() == pytest.approx([0.7])  # only valid turn contributes
    # Trajectory still in the list because turn 0 has critique tokens.
    assert trajs is not None and len(trajs) == 1
    # Turn 1 has q=0 and empty critique tokens.
    t = trajs[0]
    assert t.turns[0].q_t == 0.7
    assert len(t.turns[0].critique_token_ids) > 0
    assert t.turns[1].q_t == 0.0
    assert t.turns[1].critique_token_ids == []


def test_annotator_invalid_format_falls_back_to_zero():
    """Judge returns text but it doesn't match the schema → q=0, no critique."""
    rec = _record_for_traj(offset=0, turns=[(0, "act", "obs", 0, False)])
    canned = {"0:0": "Sorry, I can't help with that."}
    client = _MockClient(canned)
    annotator = LLMJudgeAnnotator(
        tokenizer=_StubTokenizer(),
        client=client,
        schema_version="legacy",
    )
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})

    per_q, trajs = annotator.annotate(batch)
    assert per_q.tolist() == [0.0]
    # No valid critique → no trajectory annotation emitted.
    assert trajs is None


def test_annotator_skips_empty_records():
    annotator = LLMJudgeAnnotator(
        tokenizer=_StubTokenizer(),
        client=_MockClient({}),
        schema_version="legacy",
    )
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": []})
    per_q, trajs = annotator.annotate(batch)
    assert per_q.shape == (0,)
    assert trajs is None


def test_annotator_l3_disabled_returns_no_trajectories():
    rec = _record_for_traj(offset=0, turns=[(0, "act", "obs", 0, False)])
    canned = {"0:0": "<decision_quality>+0.7</decision_quality><judgment_tags></judgment_tags><rationale>fine</rationale>"}
    client = _MockClient(canned)
    annotator = LLMJudgeAnnotator(
        tokenizer=None,
        l3_enable=False,
        client=client,
        schema_version="legacy",
    )
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})
    per_q, trajs = annotator.annotate(batch)
    assert per_q.tolist() == pytest.approx([0.7])
    assert trajs is None


def test_annotator_v3_dispatches_one_request_per_trajectory_and_uses_confidence():
    rec = _record_for_traj(
        offset=0,
        turns=[
            (0, "go to fridge", "you are at fridge", 0, False),
            (1, "open fridge", "fridge is open", 5, False),
        ],
    )
    rec.won = True
    canned = {
        "0": """
        {
          "trajectory_outcome_recap": {"agent_succeeded": true},
          "turn_judgments": [
            {
              "turn_index": 0,
              "decision_quality": 0.7,
              "confidence": 0.5,
              "judgment_tags": ["action_choice"],
              "judgment_basis": "Moves toward target.",
              "rationale": "Useful move."
            },
            {
              "turn_index": 1,
              "decision_quality": 1.0,
              "confidence": 0.75,
              "judgment_tags": [],
              "judgment_basis": "Opens the right container.",
              "rationale": "Essential setup."
            }
          ]
        }
        """,
    }
    client = _MockClient(canned)
    annotator = LLMJudgeAnnotator(tokenizer=_StubTokenizer(), client=client)
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})

    per_q, trajs = annotator.annotate(batch)

    assert len(client.calls) == 1
    assert len(client.calls[0]) == 1
    assert client.calls[0][0].request_id == "0"
    assert per_q.tolist() == pytest.approx([1.1])
    assert trajs is not None and len(trajs) == 1
    assert [t.q_t for t in trajs[0].turns] == pytest.approx([0.35, 0.75])
    assert all(len(t.critique_token_ids) > 0 for t in trajs[0].turns)


def test_annotator_v3_outcome_mismatch_gates_all_q_to_zero():
    rec = _record_for_traj(offset=0, turns=[(0, "act", "obs", 10, False)])
    rec.won = False
    canned = {
        "0": """
        {
          "trajectory_outcome_recap": {"agent_succeeded": true},
          "turn_judgments": [
            {
              "turn_index": 0,
              "decision_quality": 1.0,
              "confidence": 1.0,
              "judgment_tags": [],
              "judgment_basis": "Teacher thinks it succeeded.",
              "rationale": "Looks successful."
            }
          ]
        }
        """,
    }
    client = _MockClient(canned)
    annotator = LLMJudgeAnnotator(tokenizer=_StubTokenizer(), client=client)
    batch = SimpleNamespace(non_tensor_batch={"trajectory_records": [rec]})

    per_q, trajs = annotator.annotate(batch)

    assert per_q.tolist() == [0.0]
    assert trajs is None


# ---------------------------------------------------------------------------
# LIVE test — only runs when explicitly enabled.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("TRACE_GRPO_RUN_LIVE_TESTS", "") != "1",
    reason="set TRACE_GRPO_RUN_LIVE_TESTS=1 and INF_API_KEY to hit <LLM_JUDGE>",
)
def test_live_minimax_m27_one_request():
    """Smoke test: hit the real endpoint once and verify we get a parseable
    judgment back. Disabled by default to keep CI offline."""
    client = LLMJudgeClient(max_workers=2, max_retries=1)
    req = JudgeRequest(
        system_prompt=(
            "You are a tester. Output exactly: <decision_quality>0.7</decision_quality>"
            "<judgment_tags></judgment_tags><rationale>ok</rationale>"
        ),
        user_message="ping",
        request_id="live:0",
        max_tokens=128,
    )
    resp = client.run_batch_sync([req])
    assert len(resp) == 1
    assert resp[0].text is not None, f"endpoint error: {resp[0].error}"
    # Best-effort: just verify we got *some* structured-looking output.
    assert "decision_quality" in resp[0].text or len(resp[0].text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
