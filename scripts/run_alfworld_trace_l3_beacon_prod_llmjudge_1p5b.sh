#!/usr/bin/env bash
# One-click production launch for TRACE-GRPO AlfWorld (Qwen2.5-1.5B) on 8×H100.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STEPS="${1:-150}"
if [ "$#" -gt 0 ]; then
  shift
fi

export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/.cache/alfworld}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export INF_API_KEY="${INF_API_KEY:-}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs/alfworld_beacon_prod_1p5b}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-trace_grpo_alfworld_beacon_prod_1p5b}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-['console','wandb']}"
export ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC=False
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"

exec bash "$SCRIPT_DIR/run_alfworld_trace_l3_beacon_prod_llmjudge.sh" "$STEPS" "$@"
