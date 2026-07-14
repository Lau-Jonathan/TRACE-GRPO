"""ScienceWorld rollout driver.

Decouples the rollout *algorithm* (env loop + prompt construction +
TrajectoryAssembler) from the production framework wrappers (verl
AgentLoopBase, vLLM server manager, async LLM serving). The verl-facing
:class:`ScienceWorldAgentLoop` (in ``sciworld_agent_loop.py``) instantiates
a :class:`ScienceWorldRunner` and just translates verl's IO conventions to
this driver's plain async API.

Tested inputs / outputs:

  - Input: env adapter, tokenizer, ``GenerateFn`` callable that returns
    response token ids for a given prompt token list.
  - Output: :class:`AssembledTrajectory` populated with everything the
    teacher (LLMJudge or env_score) and the L3 provider need.

Per-turn user content: the first turn carries task description +
observation + admissible actions; follow-up turns carry only the
current observation + admissible actions (full history is already
in the packed token stream).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Awaitable, Callable, List, Optional, Protocol

from trace_grpo.agent_loops.action_parser import parse_action
from trace_grpo.agent_loops.sciworld_env_adapter import SciWorldEnvAdapter
from trace_grpo.agent_loops.trajectory_assembler import (
    AssembledTrajectory,
    TrajectoryAssembler,
)


# ---------------------------------------------------------------------------
# Tokenizer protocol — minimum surface we need.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Generate-fn contract.
# ---------------------------------------------------------------------------


GenerateFn = Callable[[list[int], int], Awaitable[list[int]]]
"""Async function: ``(prompt_ids, max_new_tokens) -> response_ids``.

Real implementations call vLLM / sglang via the verl
``AsyncLLMServerManager``. Tests can pass a closure that returns canned
responses keyed by turn number.
"""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template (BEACON-aligned).
# ---------------------------------------------------------------------------


# Kept for backward-compatible imports in older tests. BEACON ScienceWorld
# uses no separate system prompt; the instruction lives in the user template.
SCIWORLD_SYSTEM_PROMPT = ""


SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the task goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


SCIWORLD_TEMPLATE_FOLLOWUP = """Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].
"""


def _format_user_turn(
    *,
    task_description: str,
    observation: str,
    available_actions: str,
    is_first_turn: bool,
    **_kwargs,
) -> str:
    if is_first_turn:
        return SCIWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=observation,
            available_actions=available_actions,
        )
    return SCIWORLD_TEMPLATE_FOLLOWUP.format(
        current_observation=observation,
        available_actions=available_actions,
    )


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


@dataclass
class ScienceWorldRunner:
    """Drive one ScienceWorld trajectory end-to-end.

    Args:
        env_adapter: pre-loaded :class:`SciWorldEnvAdapter`. The caller is
            responsible for calling ``env_adapter.load_task(...)`` before
            handing us the adapter.
        tokenizer: HF-compatible tokenizer.
        generate_fn: async callable that turns a prompt id list into a
            response id list. Production: vLLM via
            ``AsyncLLMServerManager``. Tests: a deterministic stub.
        max_interact_steps: BEACON default 30 for ScienceWorld.
        model_response_length: per-turn cap on assistant response (spec
            §11 reference: 512 tokens for ScienceWorld).
        response_length_budget: response axis length in the verl batch
            tensor (spec §9.4: 16384).
        traj_offset_in_batch: which row in the verl batch this trajectory
            occupies.
    """

    env_adapter: SciWorldEnvAdapter
    tokenizer: TokenizerProtocol
    generate_fn: GenerateFn
    max_interact_steps: int = 30
    model_response_length: int = 512
    response_length_budget: int = 16384
    max_model_len: int = 32768
    traj_offset_in_batch: int = 0

    # diagnostics populated after run()
    last_num_turns: int = field(default=0, init=False)
    last_prompt_overflow_steps: int = field(default=0, init=False)

    async def run(self, *, is_train: bool = True) -> AssembledTrajectory:
        """Run the env loop until done or ``max_interact_steps``.

        Returns an :class:`AssembledTrajectory` populated with full
        token stream, response_mask, shaped reward, and the
        :class:`TrajectoryRecord` (with raw text fields for the LLM
        judge).
        """
        env = self.env_adapter

        # ---- initial reset + system / first user turn ---------------------
        # BEACON ``envs.py:129-134`` order: ``load() -> reset() -> get_task_description()``.
        # Some scienceworld versions return empty / stale text from
        # ``get_task_description()`` if called before ``reset()``, which
        # then feeds the LLM judge an empty goal — silently corrupts q_t.
        initial_obs, score_initial = env.reset()
        task_description = env.task_description()
        initial_actions = env.available_actions_text()
        pre_obs = initial_obs

        conversation: list[dict] = [
            {
                "role": "user",
                "content": _format_user_turn(
                    task_description=task_description,
                    observation=initial_obs,
                    available_actions=initial_actions,
                    is_first_turn=True,
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
        # Stash task / initial obs onto the record (read by LLM judge).
        # We attach it to the assembler so finalize() will copy it through.
        # Rather than complicating Assembler, we carry separately and patch
        # at the end.
        prev_score = score_initial
        last_done = False

        # Generation prompt should mirror training-time packed trajectory:
        #   initial_prompt + emitted_response_obs_stream.
        # Do not re-tokenize the full growing conversation here, otherwise
        # history is double-counted (once in conversation text, once in
        # assembler.response_ids) and prompt length can explode.
        base_prompt_ids = list(prompt_ids)

        for step_idx in range(self.max_interact_steps):
            # ---- 1. generate assistant response ---------------------------
            current_prompt = self._compose_current_prompt(
                base_prompt_ids, assembler.response_ids
            )
            prompt_budget = max(1, int(self.max_model_len) - int(self.model_response_length))
            if len(current_prompt) > prompt_budget:
                # Preserve ``base_prompt_ids`` (system + task description +
                # initial obs + chat-template role markers) at the front
                # AND keep the most recent context at the tail. Drop from
                # the middle (oldest history). The previous behavior
                # ``current_prompt[overflow:]`` deleted the system message
                # and task line, which silently broke the model's grounding
                # on long trajectories.
                base_len = len(base_prompt_ids)
                overflow = len(current_prompt) - prompt_budget
                if base_len + overflow < len(current_prompt):
                    current_prompt = (
                        list(base_prompt_ids)
                        + list(current_prompt[base_len + overflow:])
                    )
                else:
                    # Pathological case: even base_prompt_ids alone exceeds
                    # the budget. Fall back to left-trim and accept
                    # degradation.
                    current_prompt = current_prompt[len(current_prompt) - prompt_budget:]
                self.last_prompt_overflow_steps += 1
                logger.warning(
                    "[scienceworld_runner] prompt overflow at step=%s: budget=%s "
                    "(dropped_mid=%s, overflow_steps=%s)",
                    step_idx + 1,
                    prompt_budget,
                    overflow,
                    self.last_prompt_overflow_steps,
                )
            response_ids = await self.generate_fn(current_prompt, self.model_response_length)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # ---- 2. parse <action> ----------------------------------------
            parsed = parse_action(response_text)

            # ---- 3. env step ---------------------------------------------
            # Always step the env so the step counter advances even on
            # parse failures — BEACON's "no known action" feedback then
            # surfaces in the next observation. Empty env_input gets a
            # placeholder so scienceworld doesn't crash on "".
            env_input = parsed.env_input or "(empty action)"
            new_obs, score, done = env.step(env_input)

            # ---- 4. record into assembler --------------------------------
            recorded = assembler.record_response(
                response_ids=list(response_ids),
                score=score,
                done=done,
                has_format_error=parsed.has_format_error,
            )
            if not recorded:
                # Response budget exhausted; the env step already ran but
                # its tokens couldn't fit. Stop the loop without polluting
                # the score baseline. ``assembler.truncated`` is True.
                break
            assembler.turn_records[-1].action_text = parsed.action or ""
            assembler.turn_records[-1].env_response_text = new_obs

            # ---- 5. append env response as next user turn ----------------
            if not done and step_idx + 1 < self.max_interact_steps:
                pre_obs = new_obs
                next_actions = env.available_actions_text()
                next_user = {
                    "role": "user",
                    "content": _format_user_turn(
                        task_description=task_description,
                        observation=new_obs,
                        available_actions=next_actions,
                        is_first_turn=False,
                    ),
                }
                assistant_turn = {"role": "assistant", "content": response_text}
                # Encode the env response as user-turn tokens, then push
                # onto the trajectory stream with response_mask=0.
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

        traj = assembler.finalize(is_train=is_train)
        # Patch the trajectory record's metadata (LLM judge reads these).
        traj.trajectory_record.task_description = task_description
        traj.trajectory_record.initial_observation = initial_obs
        self.last_num_turns = traj.num_turns
        return traj

    # ------------------------------------------------------------------ utils

    def _compose_current_prompt(
        self,
        base_prompt_ids: list[int],
        already_emitted_response_ids: list[int],
    ) -> list[int]:
        """Produce the prompt id stream for the next generation call.

        We keep the initial chat-template prompt fixed and only append
        the emitted response/observation stream, matching the packed
        trajectory representation used for training.
        """
        return list(base_prompt_ids) + list(already_emitted_response_ids)

    def _encode_user_turn(
        self,
        message: dict,
        *,
        prefix_conversation: list[dict] | None = None,
    ) -> list[int]:
        """Encode ONE user turn and return only the *new* tokens it adds.

        We compute the diff between chat-template tokenization before vs.
        after appending the message to the current conversation. This
        avoids encoding role/turn delimiters manually and stays compatible
        with tokenizer-specific templates.
        """
        if prefix_conversation is None:
            # Fallback path for tests/util callers that don't provide the
            # running conversation. Avoid empty-conversation template calls
            # (some transformers versions raise IndexError on []).
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
        # Most templates either:
        #   - emit only the per-message segment (after starts with the
        #     same prefix as before); we strip the prefix.
        #   - or emit a fresh leading BOS each time; in that case we need
        #     to find the longest common prefix and strip it.
        common = 0
        for a, b in zip(before, after):
            if a == b:
                common += 1
            else:
                break
        return after[common:]


# ---------------------------------------------------------------------------
# Convenience: build a runner from a real ScienceWorldEnv.
# ---------------------------------------------------------------------------


def build_runner_with_real_env(
    *,
    task_name: str,
    variation: int,
    tokenizer: TokenizerProtocol,
    generate_fn: GenerateFn,
    max_interact_steps: int = 30,
    model_response_length: int = 512,
    response_length_budget: int = 16384,
    max_model_len: int = 32768,
    traj_offset_in_batch: int = 0,
    simplification_str: str = "easy",
) -> ScienceWorldRunner:
    """Lazy import of scienceworld; raises if java is missing.

    Used by the production AgentLoopBase path. Tests should construct a
    :class:`ScienceWorldRunner` manually with a mock env adapter.
    """
    from scienceworld import ScienceWorldEnv  # noqa: WPS433 — intentional lazy import

    env = ScienceWorldEnv()
    adapter = SciWorldEnvAdapter(env)
    adapter.load_task(task_name=task_name, variation=variation, simplification_str=simplification_str)
    return ScienceWorldRunner(
        env_adapter=adapter,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        max_interact_steps=max_interact_steps,
        model_response_length=model_response_length,
        response_length_budget=response_length_budget,
        max_model_len=max_model_len,
        traj_offset_in_batch=traj_offset_in_batch,
    )
