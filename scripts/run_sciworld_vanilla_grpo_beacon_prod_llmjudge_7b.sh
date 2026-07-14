#!/usr/bin/env bash
# One-click vanilla-GRPO baseline for ScienceWorld (Qwen2.5-7B) on 8×H100.
#
# Same agent loop, same shaped reward, same packed-trajectory rollout —
# but ``adv_estimator=grpo`` instead of ``trace_l3_mask``, so no LLM
# judge / L2 / L3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STEPS="${1:-200}"
if [ "$#" -gt 0 ]; then
  shift
fi

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-7B-http}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs/sciworld_vanilla_grpo_beacon_prod_7b}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-trace_grpo_sciworld_vanilla_grpo_7b}"
# 4×H200 default. H200 has 141GB VRAM, so 7B FSDP fits comfortably.
# 4 GPUs × ppo_mini=16 prompts × group_size=8 = 128 rows ⇒ 32 rows/GPU per
# PPO update; with ppo_micro=2 that's 16 micro-batches per update.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
# H200 has more headroom than H100 80GB; bump rollout memory util.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"

exec bash "$SCRIPT_DIR/run_sciworld_vanilla_grpo_beacon_prod_llmjudge.sh" "$STEPS" "$@"
