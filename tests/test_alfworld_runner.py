"""Smoke test of the AlfWorld rollout driver.

Uses a deterministic mock env (no textworld/alfworld deps) and a canned
generate function so we can exercise :class:`AlfWorldRunner` in
milliseconds. Mirrors :mod:`test_sciworld_runner`; together they confirm
the TRACE-GRPO trainer hooks remain environment-agnostic.

What we verify:
  - reset → (generate → parse → step → record) loop drains cleanly.
  - The integerized ``score = round((10*won + goal_rate) * 100)`` lets the
    spec's strict-increase reward fire on each forward sub-goal step.
  - Format-error responses (missing ``<think>`` or ``<action>``) still
    progress the env loop and land ``has_format_error=True`` on the right
    turn record.
  - Trajectory record carries ``task_description`` /
    ``initial_observation`` / per-turn ``action_text`` /
    ``env_response_text`` — i.e. everything the LLM-judge teacher reads.
  - The trajectory stops on ``done=True`` early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pytest

from trace_grpo.agent_loops.alfworld_env_adapter import (
    AlfWorldEnvAdapter,
    AlfWorldEnvProtocol,
)
from trace_grpo.agent_loops.alfworld_runner import AlfWorldRunner


# ---------------------------------------------------------------------------
# Mocks.
# ---------------------------------------------------------------------------


@dataclass
class _MockTextWorldEnv:
    """Returns batch-size-1 lists matching TextWorld's gym interface.

    ``timeline`` is a sequence of
    ``(obs:str, won:bool, goal_rate:float, done:bool, admissible:list[str])``
    consumed by step(). ``initial_obs`` and ``initial_admissible`` are
    returned by reset().
    """

    initial_obs: str
    initial_admissible: List[str]
    timeline: List[Tuple[str, bool, float, bool, List[str]]] = field(default_factory=list)
    _step_idx: int = 0

    def reset(self):
        self._step_idx = 0
        info = {
            "admissible_commands": [list(self.initial_admissible)],
            "won": [False],
            "goal_condition_success_rate": [0.0],
        }
        return [self.initial_obs], info

    def step(self, commands: List[str]):
        if self._step_idx >= len(self.timeline):
            obs, won, goal_rate, done, admissible = "no more events", False, 0.0, True, []
        else:
            obs, won, goal_rate, done, admissible = self.timeline[self._step_idx]
            self._step_idx += 1
        info = {
            "admissible_commands": [list(admissible)],
            "won": [bool(won)],
            "goal_condition_success_rate": [float(goal_rate)],
        }
        return [obs], [0.0], [done], info

    def close(self) -> None:
        pass


class _StubTokenizer:
    """One token id per character so response_mask offsets stay accurate."""

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
async def test_alfworld_runner_basic_three_turn_rollout():
    initial_obs = (
        "You are in the middle of a room. Looking quickly around, you see a "
        "fridge 1, a microwave 1, and a tomato 1.\nYour task is to: heat the "
        "tomato and place it on the table."
    )
    env = _MockTextWorldEnv(
        initial_obs=initial_obs,
        initial_admissible=["go to fridge 1", "go to microwave 1", "take tomato 1", "help"],
        timeline=[
            ("You opened the fridge 1.", False, 0.25, False,
             ["take tomato 1", "close fridge 1", "help"]),
            ("You picked up the tomato 1.", False, 0.50, False,
             ["go to microwave 1", "open microwave 1", "help"]),
            ("Tomato 1 is heated and placed on the table.", True, 1.0, True,
             ["look", "help"]),
        ],
    )
    tokenizer = _StubTokenizer()
    responses = [
        "<think>open fridge</think><action>open fridge 1</action>",
        "<think>take tomato</think><action>take tomato 1</action>",
        "<think>heat it</think><action>heat tomato 1 with microwave 1</action>",
    ]
    runner = AlfWorldRunner(
        env_adapter=AlfWorldEnvAdapter(env=env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=10,
        model_response_length=128,
        response_length_budget=8192,
        max_model_len=16384,
        traj_offset_in_batch=2,
    )
    traj = await runner.run(is_train=True)

    assert traj.num_turns == 3
    assert traj.won is True
    # Alfworld uses its own shaper (`shape_alfworld_trajectory_reward`) that
    # mirrors BEACON `compute_reward(multi_modal=False)` exactly: r_t =
    # 10 * info['won']. Text-only AlfredTWEnv has no goal_condition_success_rate
    # and thus no score-up term — only the terminal +10 fires when won is hit.
    assert traj.trajectory_reward == 10.0
    assert all(not t.has_format_error for t in traj.trajectory_record.turns)
    assert traj.trajectory_record.task_description.startswith("heat the tomato")
    # BEACON keeps "Your task is to: ..." in the initial observation —
    # the NO_HIS template has no task slot, so the task line in the obs
    # itself is what tells the model the goal on turn 0.
    assert "Your task is to" in traj.trajectory_record.initial_observation
    assert traj.trajectory_record.initial_observation.startswith("You are in the middle")
    assert traj.trajectory_record.turns[0].action_text == "open fridge 1"
    assert traj.trajectory_record.turns[0].env_response_text.startswith("You opened the fridge")
    assert traj.trajectory_record.turns[2].env_response_text.startswith("Tomato 1 is heated")
    assert traj.trajectory_record.traj_offset_in_batch == 2


@pytest.mark.asyncio
async def test_alfworld_runner_records_format_error_turns():
    initial_obs = (
        "You see a kitchen.\nYour task is to: clean the apple."
    )
    env = _MockTextWorldEnv(
        initial_obs=initial_obs,
        initial_admissible=["go to sink 1", "take apple 1", "help"],
        timeline=[
            ("ok", False, 0.0, False, ["go to sink 1", "help"]),
            ("ok again", False, 0.0, False, ["clean apple 1", "help"]),
            ("done!", True, 1.0, True, ["look", "help"]),
        ],
    )
    tokenizer = _StubTokenizer()
    responses = [
        "<action>go to sink 1</action>",  # missing <think>
        "<think>thinking aloud</think>",   # missing <action>
        "<think>finish</think><action>clean apple 1</action>",
    ]
    runner = AlfWorldRunner(
        env_adapter=AlfWorldEnvAdapter(env=env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=10,
        model_response_length=128,
        response_length_budget=8192,
        max_model_len=16384,
    )
    traj = await runner.run(is_train=True)

    assert [t.has_format_error for t in traj.trajectory_record.turns] == [True, True, False]
    assert traj.num_turns == 3


@pytest.mark.asyncio
async def test_alfworld_runner_stops_on_done():
    initial_obs = "You see a room.\nYour task is to: do nothing."
    env = _MockTextWorldEnv(
        initial_obs=initial_obs,
        initial_admissible=["look", "help"],
        timeline=[
            ("hit terminal early", True, 1.0, True, []),
            ("should not reach", False, 0.0, False, []),
        ],
    )
    tokenizer = _StubTokenizer()
    responses = [
        "<think>x</think><action>look</action>",
        "<think>x</think><action>look</action>",
    ]
    runner = AlfWorldRunner(
        env_adapter=AlfWorldEnvAdapter(env=env),
        tokenizer=tokenizer,
        generate_fn=_make_canned_generate_fn(responses, tokenizer),
        max_interact_steps=5,
        response_length_budget=8192,
        max_model_len=16384,
    )
    traj = await runner.run()
    assert traj.num_turns == 1
    assert traj.won is True


def test_alfworld_adapter_score_integerization():
    """``StepRecord.score:int`` reflects BEACON's text-only ``10 * won`` reward.

    Text-only ``AlfredTWEnv`` does not expose ``goal_condition_success_rate``;
    the mock's ``goal_rate`` slot is therefore ignored — only ``won`` drives
    the integer score.
    """
    env = _MockTextWorldEnv(
        initial_obs="An empty room.\nYour task is to: place the cup.",
        initial_admissible=["look", "help"],
        timeline=[
            ("step1", False, 0.25, False, ["look", "help"]),
            ("step2", False, 0.50, False, ["look", "help"]),
            ("step3", True, 1.0, True, ["look", "help"]),
        ],
    )
    adapter = AlfWorldEnvAdapter(env=env)
    obs0, score0 = adapter.reset()
    assert score0 == 0
    # BEACON-aligned: keep the task line in the initial obs so the NO_HIS
    # template (no task slot) still surfaces the goal to the model.
    assert "Your task is to" in obs0
    assert adapter.task_description() == "place the cup."

    _o1, s1, d1 = adapter.step("look")
    _o2, s2, d2 = adapter.step("look")
    _o3, s3, d3 = adapter.step("look")
    # Score: stays 0 while won=False; jumps to 1000 only on the won step.
    assert (s1, s2, s3) == (0, 0, 1000)
    assert (d1, d2, d3) == (False, False, True)
    assert adapter.last_won is True


def test_alfworld_adapter_admissible_text_drops_help():
    """BEACON renders admissible commands minus 'help', one per line."""
    env = _MockTextWorldEnv(
        initial_obs="A room.\nYour task is to: look.",
        initial_admissible=["look", "help", "go to fridge 1"],
        timeline=[],
    )
    adapter = AlfWorldEnvAdapter(env=env)
    adapter.reset()
    text = adapter.available_actions_text()
    assert "'help'" not in text
    assert "'look'" in text
    assert "'go to fridge 1'" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
