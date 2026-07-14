"""Thin adapter over :class:`scienceworld.ScienceWorldEnv`.

The adapter exposes a minimal protocol (:class:`SciWorldEnvProtocol`) so:
  - the agent loop can be tested with a mock env (no java required);
  - the real ``ScienceWorldEnv`` import is lazy — import errors and the
    java sub-process startup cost only hit production.

ScienceWorld API recap:
  - ``env.load(taskName, variationIdx, simplificationStr, generateGoldPath)``
  - ``obs, info = env.reset()``
  - ``obs, reward, done, info = env.step(action_str)``
  - ``info`` contains ``"score"`` (an int in roughly [-100, 100]).
  - ``env.get_task_description() -> str``
  - ``env.get_possible_actions() -> list[str]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SciWorldEnvProtocol(Protocol):
    """Subset of :class:`scienceworld.ScienceWorldEnv` we depend on.

    Implementing only this interface lets us swap the real env for a
    deterministic mock in unit tests.
    """

    def load(
        self,
        taskName: str,
        variationIdx: int = 0,
        simplificationStr: str = "",
        generateGoldPath: bool = False,
    ) -> None: ...
    def reset(self) -> tuple[str, dict[str, Any]]: ...
    def step(self, input_str: str) -> tuple[str, int, bool, dict[str, Any]]: ...
    def get_task_description(self) -> str: ...
    def get_possible_actions(self) -> list[str]: ...
    def get_possible_objects(self) -> list[str]: ...
    def close(self) -> None: ...


@dataclass
class SciWorldEnvAdapter:
    """Adapter that handles task loading and surfaces ``info["score"]`` as
    a top-level field on each step.

    Usage::

        env_proto = ScienceWorldEnv()
        adapter = SciWorldEnvAdapter(env_proto)
        adapter.load_task("measure-melting-point-known-substance", variation=0)
        first_obs = adapter.reset()
        for _ in range(N):
            obs, score, done, info = adapter.step("look around")
            ...
    """

    env: SciWorldEnvProtocol

    def load_task(
        self,
        task_name: str,
        variation: int = 0,
        simplification_str: str = "",
        generate_gold_path: bool = False,
    ) -> None:
        self.env.load(
            taskName=task_name,
            variationIdx=variation,
            simplificationStr=simplification_str,
            generateGoldPath=generate_gold_path,
        )

    def reset(self) -> tuple[str, float]:
        """Return ``(initial_observation, initial_score)``.

        Note: ``ScienceWorldEnv.reset()`` returns ``(obs, info_dict)``,
        not ``info`` with ``"score"``. Some scienceworld versions don't
        populate ``info`` until the first ``step``. We guard for that.

        Score is float — ScienceWorld emits fractional sub-goal progress
        (e.g. 0.07, 0.14, 0.5). Truncating to int erases the per-step
        signal and ``shape_trajectory_reward`` never fires ``r=1[score↑]``.
        BEACON ``envs.py:82-91`` uses float throughout.
        """
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            score = float(info.get("score", 0.0)) if isinstance(info, dict) else 0.0
        else:
            obs = out
            score = 0.0
        return obs, score

    def step(self, action: str) -> tuple[str, float, bool]:
        """Return ``(observation, score, done)``.

        Wraps ``env.step`` and pulls the score field from ``info``. We
        ignore the ``reward`` returned by scienceworld — TRACE-GRPO uses
        BEACON's shaped reward computed from score deltas, see
        :func:`trace_grpo.patches.sciworld_reward_manager.shape_trajectory_reward`.
        Score kept as float (BEACON convention); see :meth:`reset`.
        """
        obs, _reward, done, info = self.env.step(action)
        score = float(info.get("score", 0.0)) if isinstance(info, dict) else 0.0
        return obs, score, bool(done)

    def task_description(self) -> str:
        return self.env.get_task_description()

    def possible_actions(self, max_actions: int = 64) -> list[str]:
        actions = list(self.env.get_possible_actions())
        return actions[:max_actions]

    def available_actions_text(self) -> str:
        """Return BEACON's exact ScienceWorld admissible-action text."""
        valid_actions = list(self.env.get_possible_actions())
        get_objs = getattr(self.env, "get_possible_objects", None)
        valid_objs = list(get_objs()) if callable(get_objs) else []
        return (
            f"Valid_actions: {valid_actions}, "
            f"OBJ needs to be replaced with one of the following objects: {valid_objs}\n"
            f"example: <action>focus on door</action>"
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
