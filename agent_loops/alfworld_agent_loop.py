"""verl-facing :class:`AgentLoopBase` subclass for AlfWorld + TRACE-GRPO.

Mirror of :class:`trace_grpo.agent_loops.sciworld_agent_loop.ScienceWorldAgentLoop`.
The shape (Hydra knobs read, ``run()`` signature, ``AgentLoopOutput``
``extra_fields``) is intentionally identical so the TRACE-GRPO trainer
hook, reward manager, and L3 estimator stay environment-agnostic.

Wiring:

  1. Registered as ``trace_alfworld`` via verl's ``@register`` so
     ``actor_rollout_ref.rollout.agent.default_agent_loop=trace_alfworld``
     activates it.

  2. ``run()`` either uses ``alfworld_game_index`` from the dataset row
     (val) or samples from the alfworld split with a deterministic
     RNG over ``trajectory_info["step"]`` (train), the same way the
     sciworld loop replays BEACON's pool RNG.

  3. The :class:`AlfWorldRunner` returns an ``AssembledTrajectory`` with
     a populated :class:`TrajectoryRecord`; this loop translates it into
     ``AgentLoopOutput``.

We do not import :mod:`alfworld` / :mod:`textworld` at module load —
that path is hit lazily inside the runner factory.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, ClassVar, Optional
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
    _HAVE_VERL = True
except Exception:  # pragma: no cover  (verl absent in some test envs)
    _HAVE_VERL = False

    class AgentLoopBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "verl is not installed; AlfWorldAgentLoop cannot be instantiated"
            )

    def register(name: str):  # type: ignore[no-redef]
        def _decorator(cls):
            return cls
        return _decorator

    class AgentLoopOutput:  # type: ignore[no-redef]
        ...

    class AgentLoopMetrics:  # type: ignore[no-redef]
        ...


from trace_grpo.agent_loops.alfworld_runner import (
    AlfWorldRunner,
    build_runner_with_real_env,
)


_POOL_ENV_VAR = "TRACE_GRPO_ALFWORLD_POOL_PATH"
_POOL_SEED_ENV_VAR = "TRACE_GRPO_ALFWORLD_POOL_SEED"


@register("trace_alfworld")
class AlfWorldAgentLoop(AgentLoopBase):
    """Run one AlfWorld trajectory using TRACE-GRPO conventions.

    Required dataset fields (per row passed via ``run(**kwargs)``):
      - ``alfworld_game_index`` (int): index into the alfworld split.
        Required for val rows; train rows fall back to pool sampling.
      - ``alfworld_split`` (str, optional): alfworld split label
        (``"train"`` / ``"valid_seen"`` / ``"valid_unseen"``).
      - ``split`` (str, optional): ``"train"`` or ``"val"``; controls
        the format-error penalty (train-only per spec §10).

    Hydra config knobs read from ``self.config``:
      - ``actor_rollout_ref.rollout.multi_turn.max_interact_steps``
        (default 50 for AlfWorld vs. 30 for ScienceWorld)
      - ``actor_rollout_ref.rollout.multi_turn.model_response_length``
        (default 512)
      - ``actor_rollout_ref.rollout.max_model_len``
      - ``data.max_response_length``
      - ``env.history_length``
      - ``env.alfworld.eval_dataset`` (``eval_in_distribution`` |
        ``eval_out_of_distribution``)
      - ``actor_rollout_ref.rollout.agent.use_beacon_train_pool``
      - ``actor_rollout_ref.rollout.agent.use_beacon_val_pool``
    """

    _pool: ClassVar[dict[str, list[int]] | None] = None

    # -- helpers ------------------------------------------------------------

    def _read_int(self, path: list[str], default: int) -> int:
        cur: Any = self.config
        for p in path:
            try:
                cur = cur[p] if not hasattr(cur, p) else getattr(cur, p)
            except (KeyError, AttributeError, TypeError):
                return default
        try:
            return int(cur)
        except Exception:
            return default

    def _read_bool(self, path: list[str], default: bool) -> bool:
        cur: Any = self.config
        for p in path:
            try:
                cur = cur[p] if not hasattr(cur, p) else getattr(cur, p)
            except (KeyError, AttributeError, TypeError):
                return default
        if isinstance(cur, str):
            return cur.strip().lower() in {"1", "true", "yes", "on"}
        return bool(cur)

    def _read_str(self, path: list[str], default: str) -> str:
        cur: Any = self.config
        for p in path:
            try:
                cur = cur[p] if not hasattr(cur, p) else getattr(cur, p)
            except (KeyError, AttributeError, TypeError):
                return default
        return str(cur)

    @classmethod
    def _default_pool_path(cls) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "alfworld_beacon_prod" / "index.json"

    @classmethod
    def _load_pool(cls) -> dict[str, list[int]]:
        """Load the cached alfworld split pool produced by the parquet builder.

        File format::

            {
              "train":              [game_index, ...],
              "valid_seen":         [game_index, ...],
              "valid_unseen":       [game_index, ...],
              # optional aliases written by the builder:
              "eval_in_distribution":     [...],
              "eval_out_of_distribution": [...]
            }
        """
        if cls._pool is not None:
            return cls._pool
        idx_path = Path(os.environ.get(_POOL_ENV_VAR, str(cls._default_pool_path())))
        if not idx_path.exists():
            raise FileNotFoundError(
                f"AlfWorldAgentLoop: pool index not found at {idx_path}. "
                f"Run trace_grpo.scripts.build_alfworld_parquet_beacon first."
            )
        with idx_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(
                f"AlfWorldAgentLoop: pool {idx_path} must be a dict, got {type(payload).__name__}"
            )
        # Coerce values to int lists.
        out: dict[str, list[int]] = {}
        for split, idxs in payload.items():
            out[str(split)] = [int(i) for i in idxs]
        cls._pool = out
        return out

    @classmethod
    def _select_pool_game(
        cls,
        *,
        split: str,
        step: int,
        env_index: int,
        env_num: int,
        test_freq: int,
    ) -> int:
        pool = cls._load_pool()
        train_aliases = ("train",)
        val_aliases = ("valid_seen", "eval_in_distribution")
        ood_aliases = ("valid_unseen", "eval_out_of_distribution")

        norm = (split or "").strip().lower()
        if norm in train_aliases:
            order = train_aliases
        elif norm in val_aliases or norm in {"val", "valid", "validation", "test"}:
            order = val_aliases
        elif norm in ood_aliases:
            order = ood_aliases
        else:
            order = (norm,)

        chosen_split = None
        for cand in order:
            if cand in pool and pool[cand]:
                chosen_split = cand
                break
        if chosen_split is None:
            raise RuntimeError(
                f"AlfWorldAgentLoop: empty pool for split={split!r} "
                f"(available={list(pool)})"
            )
        rows = pool[chosen_split]
        env_num = max(1, min(int(env_num), len(rows)))

        seed = int(os.environ.get(_POOL_SEED_ENV_VAR, "0"))
        if chosen_split in train_aliases:
            rng = np.random.RandomState(seed)
            chosen = None
            for _ in range(max(0, int(step)) + 1):
                chosen = rng.choice(range(len(rows)), size=env_num, replace=False)
            assert chosen is not None
            return int(rows[int(chosen[int(env_index) % env_num])])

        # Validation: replay seed+1000 once per validation round to mirror
        # the sciworld pool's deterministic protocol.
        rng = np.random.RandomState(seed + 1000)
        test_freq = max(1, int(test_freq))
        validation_round = max(0, int(step) // test_freq)
        chosen = None
        for _ in range(validation_round + 1):
            chosen = rng.choice(range(len(rows)), size=env_num, replace=False)
        assert chosen is not None
        return int(rows[int(chosen[int(env_index) % env_num])])

    # -- generate ----------------------------------------------------------

    async def _generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[int]:
        sm = self.server_manager
        request_sampling_params = dict(sampling_params or {})
        request_sampling_params["max_tokens"] = int(max_new_tokens)
        request_sampling_params["n"] = 1
        if "stop_token_ids" not in request_sampling_params:
            stop_ids: list[int] = []
            tokenizer = getattr(self, "tokenizer", None)
            for token_id in (
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
            ):
                if isinstance(token_id, (list, tuple)):
                    candidates = token_id
                else:
                    candidates = [token_id]
                for candidate in candidates:
                    if candidate is None:
                        continue
                    candidate_int = int(candidate)
                    if candidate_int not in stop_ids:
                        stop_ids.append(candidate_int)
            if stop_ids:
                request_sampling_params["stop_token_ids"] = stop_ids
        if hasattr(sm, "generate"):
            out = await sm.generate(
                request_id=getattr(self, "_request_id", uuid4().hex),
                prompt_ids=list(prompt_ids),
                sampling_params=request_sampling_params,
            )
            response_ids: list[int] = list(getattr(out, "token_ids", out))
            return response_ids
        raise RuntimeError(
            "AlfWorldAgentLoop._generate: server_manager has no generate()"
        )

    # -- public ------------------------------------------------------------

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":
        if not _HAVE_VERL:
            raise RuntimeError("verl unavailable")

        self._request_id = uuid4().hex
        # Dataset row fields (val rows pin the game; train falls back to pool).
        explicit_game_index = kwargs.get("alfworld_game_index")
        explicit_alf_split = kwargs.get("alfworld_split")
        split = str(kwargs.get("split", "train"))
        split_lower = split.strip().lower()
        is_train = split_lower == "train"
        traj_offset = int(
            kwargs.get("__global_batch_index__", kwargs.get("traj_offset_in_batch", 0))
        )
        trajectory_info = kwargs.get("trajectory_info") or {}
        sample_index = int(
            kwargs.get("index", trajectory_info.get("sample_index", traj_offset))
        )
        global_step = int(trajectory_info.get("step", 0))

        max_interact_steps = self._read_int(
            ["actor_rollout_ref", "rollout", "multi_turn", "max_interact_steps"], 50
        )
        model_response_length = self._read_int(
            ["actor_rollout_ref", "rollout", "multi_turn", "model_response_length"], 512
        )
        response_length_budget = self._read_int(
            ["data", "max_response_length"], 24576
        )
        max_model_len = self._read_int(
            ["actor_rollout_ref", "rollout", "max_model_len"], 40960
        )
        history_length = self._read_int(["env", "history_length"], 2)
        train_batch_size = self._read_int(["data", "train_batch_size"], 16)
        val_batch_size = self._read_int(["data", "val_batch_size"], 128)
        group_size = self._read_int(["actor_rollout_ref", "rollout", "n"], 8)
        test_freq = self._read_int(["trainer", "test_freq"], 5)
        eval_dataset = self._read_str(
            ["env", "alfworld", "eval_dataset"], "eval_in_distribution"
        )
        use_beacon_train_pool = self._read_bool(
            ["actor_rollout_ref", "rollout", "agent", "use_beacon_train_pool"], True
        )
        use_beacon_val_pool = self._read_bool(
            ["actor_rollout_ref", "rollout", "agent", "use_beacon_val_pool"], True
        )
        is_validation = bool(trajectory_info.get("validate", False)) or (not is_train)
        env_index = traj_offset if is_validation else traj_offset // max(1, group_size)
        env_num = val_batch_size if is_validation else train_batch_size

        # Resolve (alfworld_split, game_index).
        if is_train and use_beacon_train_pool:
            alf_split = "train"
            game_index = self._select_pool_game(
                split="train",
                step=global_step,
                env_index=env_index,
                env_num=env_num,
                test_freq=test_freq,
            )
        elif (not is_train) and use_beacon_val_pool:
            alf_split = explicit_alf_split or eval_dataset
            game_index = self._select_pool_game(
                split=alf_split,
                step=global_step,
                env_index=env_index,
                env_num=env_num,
                test_freq=test_freq,
            )
        else:
            alf_split = explicit_alf_split or ("train" if is_train else eval_dataset)
            if explicit_game_index is None:
                raise KeyError(
                    "AlfWorldAgentLoop: dataset row must provide 'alfworld_game_index' "
                    "when use_beacon_*_pool=false."
                )
            game_index = int(explicit_game_index)

        seed = int(os.environ.get(_POOL_SEED_ENV_VAR, "0")) + (global_step * 31 + env_index)

        async def generate_with_current_sampling(prompt_ids: list[int], max_new_tokens: int) -> list[int]:
            return await self._generate(
                prompt_ids,
                max_new_tokens,
                sampling_params=sampling_params,
            )

        runner: AlfWorldRunner = build_runner_with_real_env(
            split=alf_split,
            game_index=int(game_index),
            tokenizer=self.tokenizer,
            generate_fn=generate_with_current_sampling,
            max_interact_steps=max_interact_steps,
            model_response_length=model_response_length,
            response_length_budget=response_length_budget,
            max_model_len=max_model_len,
            traj_offset_in_batch=traj_offset,
            history_length=history_length,
            seed=int(seed),
        )
        try:
            traj = await runner.run(is_train=is_train)
        finally:
            try:
                runner.env_adapter.close()
            except Exception:  # noqa: BLE001
                logger.warning("AlfWorldAgentLoop: env close failed", exc_info=True)

        won_int = int(runner.env_adapter.last_won)
        last_goal_rate = float(getattr(runner.env_adapter, "last_goal_condition_success_rate", 0.0))
        return AgentLoopOutput(
            prompt_ids=list(traj.prompt_ids),
            response_ids=list(traj.response_ids),
            response_mask=list(traj.response_mask),
            response_logprobs=None,
            num_turns=int(traj.num_turns),
            metrics=AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0),
            reward_score=float(traj.trajectory_reward),
            extra_fields={
                "trajectory_record": traj.trajectory_record,
                "trajectory_records": traj.trajectory_record,
                "has_format_error": any(
                    bool(t.has_format_error) for t in traj.trajectory_record.turns
                ),
                "won": won_int,
                "alfworld_won": won_int,
                "alfworld_reward": float(traj.trajectory_reward),
                "alfworld_goal_condition_success_rate": last_goal_rate,
                "trajectory_reward_no_format_penalty": float(
                    traj.trajectory_reward_no_format_penalty
                ),
                "num_invalid_turns": int(traj.num_invalid_turns),
            },
        )
