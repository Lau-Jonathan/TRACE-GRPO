"""Thin adapter over a single AlfWorld ``AlfredTWEnv`` instance.

We deliberately bypass :class:`BEACON.agent_system.environments.env_package.alfworld.envs.AlfworldEnvs`
because that class wraps every game in a Ray remote actor for BEACON's
synchronous batched trainer. verl's async agent loop already runs one
trajectory per :class:`AgentLoopWorker`, so a per-instance, non-Ray adapter
keeps the call stack short and avoids spawning nested Ray actors inside an
``AgentLoopWorker``.

Public API mirrors :class:`SciWorldEnvAdapter`:

  - ``load_task(game_index, *, split)`` — pin one ``game_file`` from the
    alfworld split and create a ``batch_size=1`` TextWorld gym env.
  - ``reset() -> (initial_obs:str, score_initial:int)``
  - ``step(action:str) -> (obs:str, score:int, done:bool)``
  - ``task_description() -> str``
  - ``available_actions_text() -> str``  — BEACON's exact admissible-action
    formatting (so prompts are bit-identical to the BEACON baseline).
  - ``close()``

The integer ``score`` returned to the TRACE-GRPO reward shaper is BEACON's
``10 * won + goal_condition_success_rate`` scaled by 100 and rounded, so
``StepRecord.score:int`` stays meaningful and the spec's
``score_t > score_{t-1}`` reward fires on each forward sub-goal step.

We do not import :mod:`alfworld` / :mod:`textworld` at module load — the
load is lazy inside :meth:`load_task` so unit tests using a mock adapter
don't need the heavy deps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol


_TASK_DESC_MARKER = "Your task is to: "


# ---------------------------------------------------------------------------
# AlfWorld-specific trajectory reward shaper (GiGPO-aligned).
# ---------------------------------------------------------------------------


def shape_alfworld_trajectory_reward(
    steps: list,
    *,
    score_initial: float = 0.0,
    is_train: bool = True,
):
    """BEACON ``AlfredTWEnv`` (text-only) shaped reward, GiGPO-aligned.

    BEACON's ``compute_reward`` for ``multi_modal=False`` is purely
    ``r_t = 10 * info['won']`` (envs.py:48-53). textworld's ``AlfredTWEnv``
    only registers ``EnvInfos(won=True, admissible_commands=True,
    extras=["gamefile"])`` (alfred_tw_env.py:254) — there is **no**
    ``goal_condition_success_rate`` and therefore **no** continuous
    sub-goal score, so BEACON's per-step reward has **no score-up term**.

    The shared :func:`shape_trajectory_reward` from
    ``patches/sciworld_reward_manager.py`` adds a ``+1`` whenever
    ``score_t > score_{t-1}``. On alfworld that fires once at the won
    step (when our integer ``score`` jumps 0 → 1000), giving
    ``trajectory_reward = +1 + +10 = 11`` for successful trajectories,
    while GiGPO baseline gives **10**. After GRPO mean-normalization the
    advantage is scaled by ~1.1× — small but non-zero, and a strict
    apples-to-apples comparison demands the alfworld trajectory reward be
    ``10 * won`` exactly.

    This shaper is identical in shape to ``shape_trajectory_reward`` (so
    it can be passed via :meth:`TrajectoryAssembler.finalize(shaper=...)`)
    but **drops** the ``score_t > prev`` term entirely.

    Args:
        steps: ordered :class:`StepRecord` list.
        score_initial: kept for signature compatibility; unused.
        is_train: kept for signature compatibility; unused.

    Returns:
        :class:`ShapedTrajectoryReward` with
        ``trajectory_reward = 10 * won_terminal``.
    """
    # Local import keeps this adapter env-agnostic at import time.
    from trace_grpo.patches.sciworld_reward_manager import (
        SHAPED_REWARD_TERMINAL_BONUS,
        ShapedTrajectoryReward,
    )

    del is_train, score_initial  # alfworld text-only shaper is stateless
    out = ShapedTrajectoryReward()
    for step in steps:
        r = 0.0
        if step.done and step.score > 0:
            r += SHAPED_REWARD_TERMINAL_BONUS
        out.per_step.append(r)
        out.trajectory_reward += r
        if step.has_format_error:
            out.num_invalid_turns += 1
    out.trajectory_reward_no_format_penalty = out.trajectory_reward
    if steps:
        last = steps[-1]
        out.won = bool(last.done and last.score > 0)
    return out


def _default_alfworld_config_path() -> Path:
    """Return the BEACON-bundled config_tw.yaml path."""
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "BEACON"
        / "agent_system"
        / "environments"
        / "env_package"
        / "alfworld"
        / "configs"
        / "config_tw.yaml"
    )


def _split_to_train_eval(split: str) -> str:
    """Map our split labels to alfworld's ``train_eval`` argument."""
    s = (split or "train").strip().lower()
    if s in {"train", "training"}:
        return "train"
    if s in {"valid_seen", "eval_in_distribution", "id", "in"}:
        return "eval_in_distribution"
    if s in {"valid_unseen", "eval_out_of_distribution", "ood", "out"}:
        return "eval_out_of_distribution"
    raise ValueError(f"Unknown alfworld split: {split!r}")


class AlfWorldEnvProtocol(Protocol):
    """Subset of the env interface used by :class:`AlfWorldEnvAdapter`.

    Implementing this protocol lets us swap a deterministic mock in for
    the real TextWorld env in unit tests.
    """

    def reset(self) -> tuple[Any, dict]: ...
    def step(self, commands: list[str]) -> tuple[Any, list[float], list[bool], dict]: ...
    def close(self) -> None: ...


@dataclass
class AlfWorldEnvAdapter:
    """Single-trajectory AlfWorld adapter.

    Use :meth:`build_for_split` to construct one bound to a specific
    alfworld split (it lazy-imports the alfworld lib). For unit tests,
    instantiate directly with a mock that implements
    :class:`AlfWorldEnvProtocol`.

    Attributes:
        env: a TextWorld gym env returned by ``AlfredTWEnv.init_env(1)``,
            or any object satisfying :class:`AlfWorldEnvProtocol`.
        admissible_commands: cached admissible_commands for the current
            step (used to format the next user turn).
        task_text: cached "Your task is to: ..." string extracted from
            the initial obs.
    """

    env: AlfWorldEnvProtocol
    admissible_commands: list[str] = field(default_factory=list)
    task_text: str = ""
    _last_obs: str = ""
    _last_won: bool = False
    _last_goal_rate: float = 0.0

    @staticmethod
    def _extract_task(obs: str) -> str:
        idx = obs.find(_TASK_DESC_MARKER)
        if idx == -1:
            return ""
        return obs[idx + len(_TASK_DESC_MARKER) :].strip()

    @staticmethod
    def _strip_task_from_obs(obs: str) -> str:
        """Drop the ``Your task is to: ...`` postfix from the initial obs.

        BEACON keeps the task line in ``self.tasks`` and feeds the rest
        of the observation into the prompt template separately. We do
        the same so prompts match.
        """
        idx = obs.find(_TASK_DESC_MARKER)
        if idx == -1:
            return obs.strip()
        return obs[:idx].strip()

    @staticmethod
    def _info_value(info: dict, key: str, default: Any) -> Any:
        """TextWorld returns lists-of-batch-1 for batched envs.

        We unwrap [x] → x but still tolerate a scalar in case the mock
        env shortcuts.
        """
        val = info.get(key, default)
        if isinstance(val, (list, tuple)) and len(val) == 1:
            return val[0]
        return val

    @staticmethod
    def _score_from_info(won: bool, goal_rate: float) -> int:
        """BEACON-aligned shaped reward, integerized for ``StepRecord.score:int``.

        ``AlfredTWEnv`` (text-only) registers ``EnvInfos(won=True,
        admissible_commands=True, extras=["gamefile"])`` — i.e. textworld
        does **not** expose ``goal_condition_success_rate`` at all; only
        the visual ``AlfredThorEnv`` produces it. BEACON's ``compute_reward``
        in envs.py:48-53 mirrors that: ``multi_modal=False`` ⇒ reward is
        purely ``10 * won``. We do the same here so the TRACE-GRPO step
        reward (``score_t > score_{t-1}``) reduces to a single +1 spike
        on the terminal ``won`` step, matching GiGPO's signal.
        """
        del goal_rate  # text-only AlfredTWEnv has no goal_condition_success_rate
        return int(round(10.0 * float(won) * 100.0))

    def reset(self) -> tuple[str, int]:
        obs, info = self.env.reset()
        if isinstance(obs, (list, tuple)) and len(obs) >= 1:
            obs_text = str(obs[0])
        else:
            obs_text = str(obs)
        self.admissible_commands = list(
            self._info_value(info, "admissible_commands", []) or []
        )
        self.task_text = self._extract_task(obs_text)
        # Match BEACON: keep the full obs (including "Your task is to: ...") as
        # the initial observation; the task line is what tells the model what
        # to do on turn 0, since ALFWORLD_TEMPLATE_NO_HIS has no task slot.
        self._last_obs = obs_text.strip()
        self._last_won = bool(self._info_value(info, "won", False))
        self._last_goal_rate = float(
            self._info_value(info, "goal_condition_success_rate", 0.0)
        )
        score = self._score_from_info(self._last_won, self._last_goal_rate)
        return self._last_obs, score

    def step(self, action: str) -> tuple[str, int, bool]:
        obs, _rewards, dones, info = self.env.step([action])
        if isinstance(obs, (list, tuple)) and len(obs) >= 1:
            obs_text = str(obs[0])
        else:
            obs_text = str(obs)
        if isinstance(dones, (list, tuple)) and len(dones) >= 1:
            done = bool(dones[0])
        else:
            done = bool(dones)
        self.admissible_commands = list(
            self._info_value(info, "admissible_commands", []) or []
        )
        self._last_won = bool(self._info_value(info, "won", False))
        self._last_goal_rate = float(
            self._info_value(info, "goal_condition_success_rate", 0.0)
        )
        score = self._score_from_info(self._last_won, self._last_goal_rate)
        self._last_obs = obs_text
        return obs_text, score, done

    def task_description(self) -> str:
        return self.task_text

    def available_actions_text(self) -> str:
        """BEACON-exact admissible-action formatting (env_manager.py:193).

        Drops the always-available ``help`` action and quotes each entry
        on its own line, matching the substring fed into
        ``ALFWORLD_TEMPLATE.{admissible_actions}``.
        """
        return "\n ".join(f"'{s}'" for s in self.admissible_commands if s != "help")

    @property
    def last_won(self) -> bool:
        return self._last_won

    @property
    def last_goal_condition_success_rate(self) -> float:
        return self._last_goal_rate

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def build_for_split(
    *,
    split: str,
    game_index: int,
    seed: int = 0,
    config_path: Optional[str | os.PathLike] = None,
    num_games: Optional[int] = None,
) -> AlfWorldEnvAdapter:
    """Construct an :class:`AlfWorldEnvAdapter` for a single game.

    Args:
        split: one of ``"train"``, ``"valid_seen"`` /
            ``"eval_in_distribution"``, ``"valid_unseen"`` /
            ``"eval_out_of_distribution"``.
        game_index: index into ``AlfredTWEnv.game_files`` for the chosen
            split (after the lib's optional ``num_train_games`` /
            ``num_eval_games`` truncation).
        seed: RNG seed forwarded to the underlying TextWorld env.
        config_path: override for ``config_tw.yaml``.
        num_games: optional override for ``dataset.num_train_games`` /
            ``dataset.num_eval_games``. ``None`` uses the YAML default
            (``-1``, i.e. full split).

    Returns:
        An adapter with the env loaded and ready for ``reset()``.
    """
    import yaml  # noqa: WPS433 - lazy

    # Lazy import: hits a heavy dep chain (textworld + alfworld native libs).
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (  # noqa: WPS433
        get_environment,
    )

    yaml_path = Path(config_path) if config_path else _default_alfworld_config_path()
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_eval = _split_to_train_eval(split)
    if num_games is not None:
        if train_eval == "train":
            config.setdefault("dataset", {})["num_train_games"] = int(num_games)
        else:
            config.setdefault("dataset", {})["num_eval_games"] = int(num_games)

    base_env = get_environment("AlfredTWEnv")(config, train_eval=train_eval)
    if not base_env.game_files:
        raise RuntimeError(
            f"AlfWorldEnvAdapter: no game files found for split={train_eval!r}. "
            f"Check $ALFWORLD_DATA in {yaml_path}."
        )

    # Pin the chosen game file by reordering so that init_env(batch_size=1)
    # registers exactly that game. AlfredTWEnv re-registers all
    # ``self.game_files`` with TextWorld; we keep the slot deterministic
    # by pre-shrinking to a single-element list.
    n_total = len(base_env.game_files)
    chosen_idx = int(game_index) % max(1, n_total)
    base_env.game_files = [base_env.game_files[chosen_idx]]
    base_env.num_games = 1

    env = base_env.init_env(batch_size=1)
    try:
        env.seed(int(seed))
    except Exception:  # pragma: no cover - older textworld lacks .seed
        pass
    return AlfWorldEnvAdapter(env=env)
