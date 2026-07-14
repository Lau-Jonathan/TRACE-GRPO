#!/usr/bin/env bash
# Ablation: L1+L2 only (L3 token-level re-weighting disabled), 8x H100.
#
# How L3 is disabled:
#   - ++algorithm.trace_grpo.trace_l3_enable=false : bypass the L3 pathway.
#   - ++algorithm.trace_grpo.trace_l3_alpha=0.0    : belt-and-braces --
#     alpha_3 = 0 makes w_delta identically 1, so A_3 = A_2 even if L3
#     is left on.
set -euo pipefail

cd ${TRACE_GRPO_ROOT:?set TRACE_GRPO_ROOT to the repo root}

# ---- 8x H100 resource configuration ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
export ROLLOUT_FREE_CACHE_ENGINE=True

# ---- Isolated output directory and W&B project name for this ablation ----
export LOG_ROOT="outputs/sciworld_ablation_l1l2"
export TRAINER_PROJECT_NAME="trace_grpo_sciworld_ablation_l1l2"

# ---- Teacher API key: allow "none" so hydra doesn't reject an empty string ----
export TEXT_FEEDBACK_API_KEY="${TEXT_FEEDBACK_API_KEY:-none}"

# ---- Allow additional hydra overrides (unlocks strict-mode) ----
export TRACE_GRPO_ALLOW_EXTRA_OVERRIDES=1

# ---- Match main-run teacher input format: show_score=1 lets the teacher
# see the current step score + outcome flag, matching what the main
# non-ablation TRACE-GRPO run uses. ----
export TEXT_FEEDBACK_SHOW_SCORE=1

# ---- We ask for 201 steps rather than 200. The verl trainer evaluates
# `is_last_step` *before* executing the step (global_steps=199 < total=200
# at that point), so requesting exactly 200 leaves step 200 without a
# validation or a checkpoint. Requesting 201 puts global_steps=200 at the
# evaluation site so 200 % save_freq == 0 and 200 % test_freq == 0 both
# fire normally. ----
exec bash scripts/run_sciworld_trace_l3_beacon_prod_llmjudge_7b.sh 201 \
  ++algorithm.trace_grpo.trace_l3_enable=false \
  ++algorithm.trace_grpo.trace_l3_alpha=0.0 \
  ++trainer.experiment_name=trace_grpo_l1l2_ablation_Qwen2_5_7B_Instruct
