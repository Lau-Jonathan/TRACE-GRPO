#!/usr/bin/env bash
# One-click production launch for TRACE-GRPO ScienceWorld (Qwen2.5-7B).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STEPS="${1:-200}"
if [ "$#" -gt 0 ]; then
  shift
fi

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-7B-Instruct}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs/sciworld_beacon_prod_7b}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-trace_grpo_sciworld_beacon_prod_7b}"

exec bash "$SCRIPT_DIR/run_sciworld_trace_l3_beacon_prod_llmjudge.sh" "$STEPS" "$@"
