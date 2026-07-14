"""AlfWorld rollout driver.

Mirrors :class:`trace_grpo.agent_loops.sciworld_runner.ScienceWorldRunner`
1:1, swapping in BEACON's exact AlfWorld prompt templates and the AlfWorld
env adapter. Keeping the shape identical lets the TRACE-GRPO trainer hooks,
reward shaping, and L3 estimator stay environment-agnostic.

Templates are copied verbatim from
``BEACON/agent_system/environments/prompts/alfworld.py`` so the in-context
distribution that the actor sees here is bit-identical to the BEACON
baseline (otherwise we cannot fairly compare the two recipes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Awaitable, Callable, List, Optional, Protocol

from trace_grpo.agent_loops.action_parser import parse_action
from trace_grpo.agent_loops.alfworld_env_adapter import (
    AlfWorldEnvAdapter,
    shape_alfworld_trajectory_reward,
)
from trace_grpo.agent_loops.trajectory_assembler import (
    AssembledTrajectory,
    TrajectoryAssembler,
)


class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...
    def apply_chat_template(
        self,
        conversation: list[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]: ...


GenerateFn = Callable[[list[int], int], Awaitable[list[int]]]
"""Async ``(prompt_ids, max_new_tokens) -> response_ids``.

Production: vLLM via verl ``AsyncLLMServerManager``. Tests: deterministic
canned-response stub.
"""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BEACON-aligned prompt templates (verbatim from
# BEACON/agent_system/environments/prompts/alfworld.py).
# ---------------------------------------------------------------------------


ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def _format_action_history(
    history: list[tuple[str, str]],
    history_length: int,
) -> tuple[str, int]:
    """Render the last ``history_length`` (obs, action) pairs.

    Layout matches BEACON's :class:`SimpleMemory` output, which is what
    the env_manager templates render alongside the admissible_actions.
    """
    recent = history[-history_length:] if history_length > 0 else []
    start_idx = len(history) - len(recent)
    lines = []
    for j, (obs, action) in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{action}']")
    return "\n".join(lines), len(recent)


def _format_user_turn(
    *,
    task_description: str,
    observation: str,
    admissible_actions: str,
    history: list[tuple[str, str]],
    history_length: int,
) -> str:
    if not history or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=observation,
            admissible_actions=admissible_actions,
        )
    action_history, valid_len = _format_action_history(history, history_length)
    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_length=valid_len,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=observation,
        admissible_actions=admissible_actions,
    )


@dataclass
class AlfWorldRunner:
    """Drive one AlfWorld trajectory end-to-end.

    Args mirror :class:`ScienceWorldRunner` so swapping environments only
    means swapping the runner class and the adapter.
    """

    env_adapter: AlfWorldEnvAdapter
    tokenizer: TokenizerProtocol
    generate_fn: GenerateFn
    max_interact_steps: int = 50
    model_response_length: int = 512
    response_length_budget: int = 24576
    max_model_len: int = 40960
    traj_offset_in_batch: int = 0
    history_length: int = 2

    last_num_turns: int = field(default=0, init=False)
    last_prompt_overflow_steps: int = field(default=0, init=False)

    async def run(self, *, is_train: bool = True) -> AssembledTrajectory:
        env = self.env_adapter

        # ---- initial reset + first user turn ------------------------------
        initial_obs, score_initial = env.reset()
        # ``task_description()`` is populated by reset(), so call after.
        task_description = env.task_description()
        history: list[tuple[str, str]] = []
        initial_actions = env.available_actions_text()
        # Track obs *before* the action that produced it. BEACON stores
        # ``(self.pre_text_obs, action)`` — i.e. the (obs_{t-1}, action_t)
        # pair — in SimpleMemory; our history must follow the same
        # convention or the formatted action_history shows obs / action
        # off-by-one.
        pre_obs = initial_obs

        conversation: list[dict] = [
            {
                "role": "user",
                "content": _format_user_turn(
                    task_description=task_description,
                    observation=initial_obs,
                    admissible_actions=initial_actions,
                    history=history,
                    history_length=self.history_length,
                ),
            },
        ]
        prompt_ids = list(
            self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
            )
        )

        assembler = TrajectoryAssembler(
            prompt_ids=prompt_ids,
            response_length=self.response_length_budget,
            traj_offset_in_batch=self.traj_offset_in_batch,
            score_initial=score_initial,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", 0) or 0,
        )
        prev_score = score_initial
        last_done = False

        base_prompt_ids = list(prompt_ids)

        for step_idx in range(self.max_interact_steps):
            current_prompt = self._compose_current_prompt(
                base_prompt_ids, assembler.response_ids
            )
            prompt_budget = max(1, int(self.max_model_len) - int(self.model_response_length))
            if len(current_prompt) > prompt_budget:
                # Preserve base_prompt_ids (system + task + initial obs +
                # chat-template markers) at the front; drop oldest history
                # from the middle. Previously ``current_prompt[overflow:]``
                # silently deleted the system + task description, which
                # destroyed grounding on long trajectories.
                base_len = len(base_prompt_ids)
                overflow = len(current_prompt) - prompt_budget
                if base_len + overflow < len(current_prompt):
                    current_prompt = (
                        list(base_prompt_ids)
                        + list(current_prompt[base_len + overflow:])
                    )
                else:
                    current_prompt = current_prompt[len(current_prompt) - prompt_budget:]
                self.last_prompt_overflow_steps += 1
                logger.warning(
                    "[alfworld_runner] prompt overflow at step=%s: budget=%s "
                    "(dropped_mid=%s, overflow_steps=%s)",
                    step_idx + 1,
                    prompt_budget,
                    overflow,
                    self.last_prompt_overflow_steps,
                )
            response_ids = await self.generate_fn(current_prompt, self.model_response_length)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # BEACON's alfworld_projection (projection.py:30) lowercases the
            # whole response before tag-finding, so ``<ACTION>`` / ``<Action>``
            # tags are still recognized. The shared ``parse_action`` is
            # case-sensitive (matching BEACON sciworld_projection). To stay
            # bit-aligned with BEACON alfworld without touching the shared
            # parser, we first try case-sensitive; on a tag-miss we retry
            # against the lowercased response. We keep ``parsed.action`` from
            # the case-sensitive pass so the LLM judge / trajectory record
            # still see the model's original casing.
            parsed = parse_action(response_text)
            if parsed.action is None and response_text:
                parsed_lower = parse_action(response_text.lower())
                if parsed_lower.action is not None:
                    parsed = parsed_lower

            # Feed the env the lowercased command (admissible_commands are
            # all lowercase; "Go to fridge 1" otherwise misses
            # "go to fridge 1"). Always step the env so the step counter
            # advances — BEACON's "Nothing happens." feedback then surfaces
            # naturally on parse failures.
            env_input = parsed.env_input.lower() if parsed.env_input else "(empty action)"
            new_obs, score, done = env.step(env_input)

            recorded = assembler.record_response(
                response_ids=list(response_ids),
                score=score,
                done=done,
                has_format_error=parsed.has_format_error,
            )
            if not recorded:
                # Response budget exhausted; stop without polluting the
                # baseline. ``assembler.truncated`` is True.
                break
            assembler.turn_records[-1].action_text = parsed.action or ""
            assembler.turn_records[-1].env_response_text = new_obs

            if not done and step_idx + 1 < self.max_interact_steps:
                # BEACON memory stores (obs_before_action, action_taken).
                history.append((pre_obs, env_input or ""))
                pre_obs = new_obs
                next_actions = env.available_actions_text()
                next_user = {
                    "role": "user",
                    "content": _format_user_turn(
                        task_description=task_description,
                        observation=new_obs,
                        admissible_actions=next_actions,
                        history=history,
                        history_length=self.history_length,
                    ),
                }
                assistant_turn = {"role": "assistant", "content": response_text}
                obs_ids = self._encode_user_turn(
                    next_user,
                    prefix_conversation=[*conversation, assistant_turn],
                )
                assembler.record_observation(obs_ids)
                conversation.append(assistant_turn)
                conversation.append(next_user)

            prev_score = score
            last_done = done
            if done or assembler.truncated:
                break

        # Use the alfworld-specific shaper (10*won only, GiGPO-aligned) instead
        # of the sciworld/alfworld-shared shaper which would also fire +1 for
        # ``score_t > score_{t-1}``. Text-only AlfredTWEnv only emits binary
        # info['won'] (no continuous goal_condition signal), so the shared
        # +1 term has no semantic basis here and would inflate
        # trajectory_reward to 11 vs GiGPO's 10 on success.
        traj = assembler.finalize(
            is_train=is_train,
            shaper=shape_alfworld_trajectory_reward,
        )
        traj.trajectory_record.task_description = task_description
        traj.trajectory_record.initial_observation = initial_obs
        self.last_num_turns = traj.num_turns
        return traj

    def _compose_current_prompt(
        self,
        base_prompt_ids: list[int],
        already_emitted_response_ids: list[int],
    ) -> list[int]:
        return list(base_prompt_ids) + list(already_emitted_response_ids)

    def _encode_user_turn(
        self,
        message: dict,
        *,
        prefix_conversation: list[dict] | None = None,
    ) -> list[int]:
        """Encode ONE user turn and return only the *new* tokens it adds.

        We compute the diff between chat-template tokenization before vs.
        after appending the message to the running conversation.
        """
        if prefix_conversation is None:
            after = list(
                self.tokenizer.apply_chat_template(
                    [message],
                    add_generation_prompt=False,
                    tokenize=True,
                )
            )
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None and len(after) > 0 and after[0] == bos_id:
                return after[1:]
            return after

        before = list(
            self.tokenizer.apply_chat_template(
                prefix_conversation,
                add_generation_prompt=False,
                tokenize=True,
            )
        )
        after = list(
            self.tokenizer.apply_chat_template(
                [*prefix_conversation, message],
                add_generation_prompt=False,
                tokenize=True,
            )
        )
        common = 0
        for a, b in zip(before, after):
            if a == b:
                common += 1
            else:
                break
        return after[common:]


def build_runner_with_real_env(
    *,
    split: str,
    game_index: int,
    tokenizer: TokenizerProtocol,
    generate_fn: GenerateFn,
    max_interact_steps: int = 50,
    model_response_length: int = 512,
    response_length_budget: int = 24576,
    max_model_len: int = 40960,
    traj_offset_in_batch: int = 0,
    history_length: int = 2,
    seed: int = 0,
) -> AlfWorldRunner:
    """Build a runner backed by a real :class:`AlfWorldEnvAdapter`.

    Lazy-imports the alfworld + textworld stack via
    :func:`alfworld_env_adapter.build_for_split`. Tests should construct
    :class:`AlfWorldRunner` directly with a mock env adapter.
    """
    from trace_grpo.agent_loops.alfworld_env_adapter import build_for_split

    adapter = build_for_split(
        split=split,
        game_index=game_index,
        seed=seed,
    )
    return AlfWorldRunner(
        env_adapter=adapter,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        max_interact_steps=max_interact_steps,
        model_response_length=model_response_length,
        response_length_budget=response_length_budget,
        max_model_len=max_model_len,
        traj_offset_in_batch=traj_offset_in_batch,
        history_length=history_length,
    )
