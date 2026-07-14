"""Smoke test of the ScienceWorld rollout driver.

Uses a deterministic mock env (no java, no scienceworld package) and a
canned generate function so we can exercise the full
:class:`ScienceWorldRunner` rollout in milliseconds.

What we're verifying:
  - The runner drives env.reset → loop(generate → parse → env.step →
    record) → finalize, in that exact order.
  - format-error responses (missing ``<think>`` or ``<action>``) flow
    through correctly: ``has_format_error=True`` lands on the matching
    turn record, the env still steps, and the trajectory does not silently
    advance.
  - The shaped reward matches BEACON Eq. 2.
  - The :class:`TrajectoryRecord` carries task_description /
    initial_observation / per-turn action_text / env_response_text — i.e.
    everything the LLM-judge teacher reads.
  - The trajectory stops on ``done=True`` early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

import pytest

from trace_grpo.agent_loops.sciworld_agent_loop import ScienceWorldAgentLoop
from trace_grpo.agent_loops.action_parser import parse_action
from trace_grpo.agent_loops.sciworld_env_adapter import (
    SciWorldEnvAdapter,
    SciWorldEnvProtocol,
)
from trace_grpo.agent_loops.sciworld_runner import (
    SCIWORLD_SYSTEM_PROMPT,
    ScienceWorldRunner,
)


# ---------------------------------------------------------------------------
# Mocks.
# ---------------------------------------------------------------------------


@dataclass
class _MockEnv:
    """Deterministic env: returns a scripted sequence of (obs, score, done).

    Each call to step() consumes one entry from ``timeline``; reset()
    seeds the initial observation. Score and done flags follow whatever
    the test scripts.
    """

    initial_obs: str
    timeline: List[tuple[str, int, bool]]
    task: str = "Mock task: heat the tomato."
    actions: List[str] = field(default_factory=lambda: ["look around", "open fridge", "take tomato"])
    _step_idx: int = 0

    def load(self, taskName: str, variationIdx: int = 0, simplificationStr: str = "", generateGoldPath: bool = False) -> None:
        self._step_idx = 0

    def reset(self) -> tuple[str, dict]:
        self._step_idx = 0
        return self.initial_obs, {"score": 0}

    def step(self, input_str: str) -> tuple[str, int, bool, dict]:
        if self._step_idx >= len(self.timeline):
            obs, score, done = "no more events", 0, True
        else:
            obs, score, done = self.timeline[self._step_idx]
            self._step_idx += 1
        return obs, 0, done, {"score": score}

    def get_task_description(self) -> str:
        return self.task

    def get_possible_actions(self) -> List[str]:
        return list(self.actions)

    def close(self) -> None:
        pass


class _StubTokenizer:
    """One token id per character — enough for the runner to compute
    response_mask offsets correctly. apply_chat_template returns a
    minimal serialized form.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(
        self,
        conversation: List[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ):
        # Render conversation as a flat string then encode.
        parts: List[str] = []
        for m in conversation:
            parts.append(f"<|{m.get('role', 'user')}|>")
            parts.append(m.get("content", ""))
            parts.append("<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        flat = "".join(parts)
        return self.encode(flat) if tokenize else flat


def _make_canned_generate_fn(responses: List[str], tokenizer):
    idx = {"i": 0}

    async def _gen(prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        text = responses[idx["i"]]
        idx["i"] += 1
        ids = tokenizer.encode(text)
        return ids[:max_new_tokens]

    return _gen


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_basic_three_turn_rollout():
    env = _MockEnv(
        initial_obs="You are in the kitchen.",
        timeline=[
            ("You opened the fridge.", 5, False),
            ("You picked up the tomato.", 10, False),
            ("Tomato is heated.", 50, True),
        ],
    )
    tokenizer = _StubTokenizer()
    responses = [
        "<think>open fridge</think><action>open fridge</action>",
        "<think>take tomato</think><action>take tomato</action>",
        "<think>heat it</think><action>heat tomato in microwave</action>",
    ]
    runner = ScienceWorldRunner(
        env_adapter=SciWorldEnvAdapter(env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=10,
        model_response_length=128,
        response_length_budget=8192,
        traj_offset_in_batch=3,
    )
    traj = await runner.run(is_train=True)

    # 3 turns recorded; trajectory ended on terminal done.
    assert traj.num_turns == 3
    assert traj.won is True
    # Reward: t1 0→5 (+1), t2 5→10 (+1), t3 10→50 (+1) + done&score>0 (+10) = 13
    assert traj.trajectory_reward == 13.0
    # No format errors anywhere.
    assert all(not t.has_format_error for t in traj.trajectory_record.turns)
    # Task description / initial obs propagated.
    assert traj.trajectory_record.task_description == env.task
    assert traj.trajectory_record.initial_observation == env.initial_obs
    # Per-turn action_text and env_response_text populated.
    assert traj.trajectory_record.turns[0].action_text == "open fridge"
    assert traj.trajectory_record.turns[0].env_response_text.startswith("You opened the fridge")
    assert traj.trajectory_record.turns[2].env_response_text == "Tomato is heated."
    # traj_offset propagated correctly.
    assert traj.trajectory_record.traj_offset_in_batch == 3


@pytest.mark.asyncio
async def test_runner_records_format_error_turns():
    """Response missing <think> or <action> tags must be flagged as
    format error and still progress the env loop."""
    env = _MockEnv(
        initial_obs="kitchen",
        timeline=[
            ("ok", 0, False),
            ("ok again", 0, False),
            ("done!", 5, True),
        ],
    )
    tokenizer = _StubTokenizer()
    responses = [
        # Missing <think> tag — flagged as format error, action still parsed.
        "<action>open fridge</action>",
        # Missing <action> tag — flagged, env_input=last 20 chars.
        "<think>I am thinking</think>",
        # Properly formatted final turn.
        "<think>finish</think><action>do task</action>",
    ]
    runner = ScienceWorldRunner(
        env_adapter=SciWorldEnvAdapter(env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=10,
        model_response_length=128,
        response_length_budget=8192,
    )
    traj = await runner.run(is_train=True)

    assert [t.has_format_error for t in traj.trajectory_record.turns] == [True, True, False]
    # Even with format errors, env loop did progress (3 step records).
    assert traj.num_turns == 3


@pytest.mark.asyncio
async def test_runner_stops_on_done():
    env = _MockEnv(
        initial_obs="start",
        timeline=[
            ("hit terminal early", 50, True),
            # These would be reached if the loop didn't stop on done.
            ("should not reach", 0, False),
            ("should not reach", 0, False),
        ],
    )
    tokenizer = _StubTokenizer()
    responses = [
        "<think>x</think><action>act1</action>",
        "<think>x</think><action>act2</action>",
        "<think>x</think><action>act3</action>",
    ]
    runner = ScienceWorldRunner(
        env_adapter=SciWorldEnvAdapter(env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=5,
        response_length_budget=8192,
    )
    traj = await runner.run()
    assert traj.num_turns == 1
    assert traj.won is True


@pytest.mark.asyncio
async def test_runner_truncates_when_response_length_exceeded():
    """Tight response budget → trajectory truncates and Assembler flags it."""
    env = _MockEnv(
        initial_obs="start",
        timeline=[
            ("step1", 0, False),
            ("step2", 0, False),
            ("step3", 0, False),
            ("step4", 0, False),
            ("step5", 0, False),
        ],
    )
    tokenizer = _StubTokenizer()
    # Each response decoded to ~50 chars → 50 tokens. With budget 64 we
    # should only fit one response plus part of the env-response of turn 1.
    responses = ["<think>aaa</think><action>do something useful</action>"] * 5
    runner = ScienceWorldRunner(
        env_adapter=SciWorldEnvAdapter(env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=10,
        model_response_length=200,
        response_length_budget=64,
    )
    traj = await runner.run()
    # Truncation happened — Assembler exposed the flag, trajectory still finishes.
    # Mostly we just want this not to crash and num_turns to be > 0.
    assert traj.num_turns >= 1
    assert len(traj.response_ids) == 64  # padded/truncated to budget


# ---------------------------------------------------------------------------
# Action parser direct tests.
# ---------------------------------------------------------------------------


def test_parse_action_well_formed():
    p = parse_action("<think>plan</think><action>look around</action>")
    assert p.action == "look around"
    assert p.has_format_error is False
    assert p.env_input == "look around"


def test_parse_action_missing_action_uses_last_20_chars():
    txt = "I cannot help with this. Sorry. The end of the message is here."
    p = parse_action(txt)
    assert p.action is None
    assert p.has_format_error is True
    assert p.env_input == txt[-20:]


def test_parse_action_missing_think_still_extracts_action():
    p = parse_action("<action>do thing</action>")
    assert p.action == "do thing"
    assert p.has_format_error is True       # missing <think>
    assert p.env_input == "do thing"


def test_parse_action_chinese_text_is_beacon_invalid():
    p = parse_action("<think>计划</think><action>look around</action>")
    assert p.action == "look around"
    assert p.has_format_error is True
    assert p.env_input == "look around"


def test_parse_action_tags_are_beacon_case_sensitive():
    txt = "<THINK>plan</THINK><ACTION>look around</ACTION>"
    p = parse_action(txt)
    assert p.action is None
    assert p.has_format_error is True
    assert p.env_input == txt[-20:]


def test_parse_action_handles_none_input():
    p = parse_action(None)
    assert p.action is None
    assert p.has_format_error is True
    assert p.env_input == ""


@pytest.mark.asyncio
async def test_agent_loop_generate_uses_verl_sampling_params():
    class _FakeServer:
        def __init__(self):
            self.sampling_params = None

        async def generate(self, *, request_id, prompt_ids, sampling_params):
            self.sampling_params = dict(sampling_params)
            return SimpleNamespace(token_ids=[11, 12])

    server = _FakeServer()
    loop = object.__new__(ScienceWorldAgentLoop)
    loop.server_manager = server
    loop._request_id = "req"
    loop.tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)

    ids = await loop._generate(
        [1, 2, 3],
        7,
        sampling_params={"temperature": 0.4, "top_p": 0.95, "top_k": -1, "n": 8},
    )

    assert ids == [11, 12]
    assert server.sampling_params["temperature"] == 0.4
    assert server.sampling_params["top_p"] == 0.95
    assert server.sampling_params["top_k"] == -1
    assert server.sampling_params["max_tokens"] == 7
    assert server.sampling_params["n"] == 1
    assert server.sampling_params["stop_token_ids"] == [151645, 151643]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
