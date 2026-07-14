#!/usr/bin/env bash
# One-click vanilla-GRPO baseline for ScienceWorld (Qwen2.5-1.5B).
#
# This is the framework sanity check: same agent loop, same shaped
# reward, same packed-trajectory rollout — but ``adv_estimator=grpo``
# instead of ``trace_l3_mask``, so no LLM judge / L2 / L3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STEPS="${1:-200}"
if [ "$#" -gt 0 ]; then
  shift
fi

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs/sciworld_vanilla_grpo_beacon_prod_1p5b}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-trace_grpo_sciworld_vanilla_grpo_1p5b}"
# vLLM 0.10.2 V1 rejects --disable-async-output-proc; eager-mode mitigates
# the 1.5B H200 V1 CUDA crash without changing data / rollout semantics.
export ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC=False
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# H100 80GB: reduce micro-batch to fit backward activation in memory.
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"

exec bash "$SCRIPT_DIR/run_sciworld_vanilla_grpo_beacon_prod_llmjudge.sh" "$STEPS" "$@"
