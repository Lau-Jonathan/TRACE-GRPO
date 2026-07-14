"""verl-facing :class:`AgentLoopBase` subclass for ScienceWorld + TRACE-GRPO.

Thin adapter around :class:`ScienceWorldRunner`. The runner does all the
domain logic (env loop, prompt construction, format-error handling,
trajectory bookkeeping); this class only translates verl's IO conventions
to / from the runner.

Wiring:

  1. The class is registered as ``trace_sciworld`` via verl's
     ``@register`` decorator in
     :mod:`verl.experimental.agent_loop`. To activate it, set
     ``actor_rollout_ref.rollout.agent.agent_class``
     to ``trace_grpo.agent_loops.sciworld_agent_loop.ScienceWorldAgentLoop``
     in your hydra config (or refer to it by registry name).

  2. The ``run()`` async method:
       - Loads the ScienceWorld task identified by the dataset row's
         ``sciworld_task`` / ``sciworld_var`` fields (or, if those
         aren't present, falls back to the fields the spec uses for
         BEACON parquet rows).
       - Wraps verl's :class:`AsyncLLMServerManager` in a
         :data:`GenerateFn` callable.
       - Hands everything to :class:`ScienceWorldRunner.run`.
       - Translates the resulting :class:`AssembledTrajectory` into a
         verl :class:`AgentLoopOutput`. The full
         :class:`TrajectoryRecord` is stashed into
         ``extra_fields["trajectory_record"]`` so the trainer hook can
         hand it to the TRACE-GRPO teacher.

  3. We deliberately do *not* import :mod:`scienceworld` at module load
     time — it spawns a Java sub-process on instantiation. The import is
     lazy (inside :meth:`_load_env`) so unit tests that don't need the
     real env aren't blocked.
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


# ---------------------------------------------------------------------------
# Lazy verl imports.
# ---------------------------------------------------------------------------


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
        """Fallback shim so ``import`` succeeds without verl."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "verl is not installed; ScienceWorldAgentLoop cannot be instantiated"
            )

    def register(name: str):  # type: ignore[no-redef]
        def _decorator(cls):
            return cls
        return _decorator

    class AgentLoopOutput:  # type: ignore[no-redef]
        ...

    class AgentLoopMetrics:  # type: ignore[no-redef]
        ...


from trace_grpo.agent_loops.sciworld_runner import (
    ScienceWorldRunner,
    build_runner_with_real_env,
)


# ---------------------------------------------------------------------------
# Subclass.
# ---------------------------------------------------------------------------


@register("trace_sciworld")
class ScienceWorldAgentLoop(AgentLoopBase):
    """Run one ScienceWorld trajectory using TRACE-GRPO conventions.

    Required dataset fields (per row passed via ``run(**kwargs)``):
      - ``sciworld_task`` (str): ScienceWorld task name.
      - ``sciworld_var`` (int): variation index for the task.
      - ``split`` (str, optional): ``"train"`` or ``"val"``. Defaults to
        ``"train"``; only affects format-error penalty.

    Hydra config knobs read from ``self.config``:
      - ``actor_rollout_ref.rollout.multi_turn.max_interact_steps``
      - ``actor_rollout_ref.rollout.multi_turn.model_response_length``
      - ``actor_rollout_ref.rollout.response_length`` (per-step generation cap)
      - ``data.max_response_length`` (response axis length)

    Defaults match BEACON's ScienceWorld setup (spec §11):
      - max_interact_steps = 30
      - model_response_length = 512
      - response_length_budget = 16384
    """

    _beacon_pools: ClassVar[dict[str, list[tuple[int, int]]] | None] = None
    _beacon_train_pool: ClassVar[list[tuple[int, int]] | None] = None
    _beacon_pool_rng: ClassVar[random.Random | None] = None
    _task_names: ClassVar[list[str] | None] = None

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
    def _default_beacon_idx_path(cls) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return (
            repo_root
            / "BEACON"
            / "agent_system"
            / "environments"
            / "env_package"
            / "sciworld"
            / "variations_idx"
            / "L0_idx.json"
        )

    @classmethod
    def _load_beacon_pools(cls) -> dict[str, list[tuple[int, int]]]:
        if cls._beacon_pools is not None:
            return cls._beacon_pools

        idx_path = Path(
            os.environ.get("TRACE_GRPO_SCIWORLD_L0_IDX", str(cls._default_beacon_idx_path()))
        )
        with idx_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            cls._beacon_pools = {
                split: [(int(t), int(v)) for t, v in rows]
                for split, rows in payload.items()
            }
        else:
            cls._beacon_pools = {"train": [(int(t), int(v)) for t, v in payload]}
        cls._beacon_train_pool = cls._beacon_pools.get("train", [])

        seed = int(os.environ.get("TRACE_GRPO_SCIWORLD_TRAIN_POOL_SEED", "0"))
        cls._beacon_pool_rng = random.Random(seed)
        return cls._beacon_pools

    @classmethod
    def _load_beacon_train_pool(cls) -> list[tuple[int, int]]:
        return cls._load_beacon_pools().get("train", [])

    @classmethod
    def _load_task_names(cls) -> list[str]:
        if cls._task_names is not None:
            return cls._task_names

        from scienceworld import ScienceWorldEnv  # noqa: WPS433

        env = ScienceWorldEnv()
        try:
            cls._task_names = list(env.get_task_names())
        finally:
            try:
                env.close()
            except Exception:
                pass
        return cls._task_names

    @classmethod
    def _sample_beacon_train_task(cls) -> tuple[str, int]:
        pool = cls._load_beacon_train_pool()
        rng = cls._beacon_pool_rng or random.Random(0)
        task_id, variation = rng.choice(pool)
        task_names = cls._load_task_names()
        if task_id < 0 or task_id >= len(task_names):
            raise IndexError(
                f"BEACON ScienceWorld task id {task_id} is outside task_names[0:{len(task_names)}]"
        )
        return task_names[task_id], variation

    @classmethod
    def _select_beacon_task(
        cls,
        *,
        split: str,
        sample_index: int,
        step: int,
        env_index: int,
        env_num: int,
        test_freq: int,
    ) -> tuple[str, int]:
        pools = cls._load_beacon_pools()
        split_key = "test" if split in {"val", "valid", "validation", "test"} else "train"
        pool = pools.get(split_key) or pools.get("train") or []
        if not pool:
            raise RuntimeError(f"ScienceWorldAgentLoop: empty BEACON {split_key!r} task pool")

        env_num = max(1, min(int(env_num), len(pool)))
        if split_key == "train":
            # BEACON's SciWorldMultiProcessEnv owns a persistent
            # np.random.RandomState(seed) and calls choice(..., size=env_num,
            # replace=False) once per training reset. Replaying the RNG to
            # ``step`` keeps this stateless AgentLoop aligned with that
            # protocol.
            seed = int(os.environ.get("TRACE_GRPO_SCIWORLD_TRAIN_POOL_SEED", "0"))
            rng = np.random.RandomState(seed)
            chosen = None
            for _ in range(max(0, int(step)) + 1):
                chosen = rng.choice(range(len(pool)), size=env_num, replace=False)
            assert chosen is not None
            task_id, variation = pool[int(chosen[int(env_index) % env_num])]
        else:
            # Validation envs use seed + 1000 in BEACON. They reset only at
            # validation points, so approximate the persistent RNG state by
            # replaying the number of validations observed up to this step.
            seed = int(os.environ.get("TRACE_GRPO_SCIWORLD_TRAIN_POOL_SEED", "0")) + 1000
            rng = np.random.RandomState(seed)
            test_freq = max(1, int(test_freq))
            validation_round = max(0, int(step) // test_freq)
            chosen = None
            for _ in range(validation_round + 1):
                chosen = rng.choice(range(len(pool)), size=env_num, replace=False)
            assert chosen is not None
            task_id, variation = pool[int(chosen[int(env_index) % env_num])]

        task_names = cls._load_task_names()
        if task_id < 0 or task_id >= len(task_names):
            raise IndexError(
                f"BEACON ScienceWorld task id {task_id} is outside task_names[0:{len(task_names)}]"
            )
        return task_names[task_id], variation

    async def _generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[int]:
        """Adapt verl's AsyncLLMServerManager to our :data:`GenerateFn`."""
        # Different verl versions expose generate() with slightly different
        # signatures. We try the most common ones in order.
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
                prompt_ids=list(prompt_ids), sampling_params=request_sampling_params
            )
            # AsyncLLMServerManager returns a TokenOutput-like; pull the ids.
            response_ids: list[int] = list(getattr(out, "token_ids", out))
            return response_ids
        raise RuntimeError(
            "ScienceWorldAgentLoop._generate: server_manager has no generate()"
        )

    # -- public -------------------------------------------------------------

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":
        if not _HAVE_VERL:
            raise RuntimeError("verl unavailable")

        self._request_id = uuid4().hex
        task = kwargs.get("sciworld_task") or kwargs.get("task_name")
        has_variation_field = ("sciworld_var" in kwargs) or ("variation" in kwargs)
        variation = int(kwargs.get("sciworld_var", kwargs.get("variation", 0)))
        split = str(kwargs.get("split", "train"))
        split_lower = split.strip().lower()
        is_train = split_lower == "train"
        traj_offset = int(kwargs.get("__global_batch_index__", kwargs.get("traj_offset_in_batch", 0)))
        trajectory_info = kwargs.get("trajectory_info") or {}
        sample_index = int(kwargs.get("index", trajectory_info.get("sample_index", traj_offset)))
        global_step = int(trajectory_info.get("step", 0))

        max_interact_steps = self._read_int(
            ["actor_rollout_ref", "rollout", "multi_turn", "max_interact_steps"], 30
        )
        model_response_length = self._read_int(
            ["actor_rollout_ref", "rollout", "multi_turn", "model_response_length"], 512
        )
        response_length_budget = self._read_int(
            ["data", "max_response_length"], 16384
        )
        max_model_len = self._read_int(
            ["actor_rollout_ref", "rollout", "max_model_len"], 32768
        )
        train_batch_size = self._read_int(["data", "train_batch_size"], 16)
        val_batch_size = self._read_int(["data", "val_batch_size"], 128)
        group_size = self._read_int(["actor_rollout_ref", "rollout", "n"], 8)
        test_freq = self._read_int(["trainer", "test_freq"], 5)
        simplification_str = self._read_str(
            ["env", "sciworld", "simplifications_preset"], "easy"
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

        if is_train and use_beacon_train_pool:
            # Spec §4.3: training always overrides parquet task/variation with a
            # fresh sample from BEACON's train pool.
            task, variation = self._select_beacon_task(
                split="train",
                sample_index=sample_index,
                step=global_step,
                env_index=env_index,
                env_num=env_num,
                test_freq=test_freq,
            )
        elif (not is_train) and use_beacon_val_pool:
            # Strict BEACON parity: validation/test also draws from the
            # BEACON L0 test pool using the same seeded reset protocol.
            task, variation = self._select_beacon_task(
                split="test",
                sample_index=sample_index,
                step=global_step,
                env_index=env_index,
                env_num=env_num,
                test_freq=test_freq,
            )
        elif task is None:
            # Fallback path when BEACON pool sampling is disabled.
            raise KeyError(
                "ScienceWorldAgentLoop: dataset row must provide 'sciworld_task' "
                "and 'sciworld_var' for non-train splits."
            )
        elif (not is_train) and (not has_variation_field):
            raise KeyError(
                "ScienceWorldAgentLoop: dataset row must provide 'sciworld_var' "
                "for non-train splits."
            )

        async def generate_with_current_sampling(prompt_ids: list[int], max_new_tokens: int) -> list[int]:
            return await self._generate(
                prompt_ids,
                max_new_tokens,
                sampling_params=sampling_params,
            )

        runner: ScienceWorldRunner = build_runner_with_real_env(
            task_name=task,
            variation=variation,
            tokenizer=self.tokenizer,
            generate_fn=generate_with_current_sampling,
            max_interact_steps=max_interact_steps,
            model_response_length=model_response_length,
            response_length_budget=response_length_budget,
            max_model_len=max_model_len,
            traj_offset_in_batch=traj_offset,
            simplification_str=simplification_str,
        )
        try:
            traj = await runner.run(is_train=is_train)
        finally:
            # Free the Java sub-process before this AgentLoopWorker moves on.
            try:
                runner.env_adapter.close()
            except Exception:  # noqa: BLE001
                logger.warning("ScienceWorldAgentLoop: env close failed", exc_info=True)

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
                "won": int(traj.won),
                "scienceworld_won": int(traj.won),
                "scienceworld_reward": float(traj.trajectory_reward),
                "scienceworld_goal_score": float(
                    traj.trajectory_record.turns[-1].score if traj.trajectory_record.turns else 0.0
                ),
                "trajectory_reward_no_format_penalty": float(
                    traj.trajectory_reward_no_format_penalty
                ),
                # BEACON's invalid-action penalty consumes this count.
                # Surface it as a non-tensor batch field so the trainer's
                # apply_invalid_action_penalty hook can subtract
                # ``coef * num_invalid_turns`` from the packed
                # trajectory's last response token.
                "num_invalid_turns": int(traj.num_invalid_turns),
            },
        )
