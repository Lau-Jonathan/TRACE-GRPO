#!/usr/bin/env bash
# One-click production launch for TRACE-GRPO ScienceWorld (Qwen2.5-1.5B).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STEPS="${1:-200}"
if [ "$#" -gt 0 ]; then
  shift
fi

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs/sciworld_beacon_prod_1p5b}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-trace_grpo_sciworld_beacon_prod_1p5b}"
# vLLM 0.10.2 V1 rejects --disable-async-output-proc, so keep that flag off.
# The 1.5B H200 crash happens inside the V1 CUDA execution path; eager mode is
# the V1-compatible mitigation because it disables CUDA graph capture without
# changing BEACON/TRACE-GRPO data, rollout, teacher, or optimization semantics.
export ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC=False
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# H100 80GB: reduce micro-batch to fit backward activation in memory.
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"

exec bash "$SCRIPT_DIR/run_sciworld_trace_l3_beacon_prod_llmjudge.sh" "$STEPS" "$@"
