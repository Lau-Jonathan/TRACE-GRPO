#!/usr/bin/env bash
# One-command TRACE-GRPO ScienceWorld launch with <LLM_JUDGE> judge.
# Defaults:
#   API_BASE=${API_BASE:?set API_BASE}
#   TEXT_FEEDBACK_MODEL=${TEXT_FEEDBACK_MODEL:?set TEXT_FEEDBACK_MODEL}
#   TEXT_FEEDBACK_API_KEY="" (optional)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

STEPS=${1:-200}
API_BASE=${API_BASE:?set API_BASE, e.g. https://your-endpoint.example.com/v1}
DEFAULT_CHAT_COMPLETIONS_URL="${API_BASE%/}/chat/completions"
JUDGE_BASE_URL=${TEXT_FEEDBACK_BASE_URL:-$DEFAULT_CHAT_COMPLETIONS_URL}
REQUESTED_PROFILE=${TRACE_GRPO_PROFILE_MODE:-beacon_strict}
BATCH_PROFILE=${TRACE_GRPO_BATCH_PROFILE:-beacon}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/outputs/h200_sciworld}
RUN_DIR="${LOG_ROOT}/${RUN_TAG}"
LOG_FILE="${RUN_DIR}/train.log"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  N_GPUS=4
fi

mkdir -p "$RUN_DIR"
touch "$LOG_FILE"
ln -sfn "$RUN_DIR" "$LOG_ROOT/latest"

log() {
  echo "$1" | tee -a "$LOG_FILE"
}

export TRACE_GRPO_TEACHER_KIND=llm
if [ "$REQUESTED_PROFILE" != "beacon_strict" ] && [ "${TRACE_GRPO_ALLOW_NON_STRICT_PROFILE:-0}" != "1" ]; then
  log "[LLM judge judge] refusing non-production profile: TRACE_GRPO_PROFILE_MODE=$REQUESTED_PROFILE"
  log "[LLM judge judge] unset TRACE_GRPO_PROFILE_MODE for production, or set TRACE_GRPO_ALLOW_NON_STRICT_PROFILE=1 for debug-only runs."
  exit 2
fi
export TRACE_GRPO_PROFILE_MODE="$REQUESTED_PROFILE"
if [ "$TRACE_GRPO_PROFILE_MODE" = "beacon_strict" ] && [ "${TRACE_GRPO_ALLOW_ENGINE_OVERRIDE:-0}" != "1" ]; then
  export ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
  export ROLLOUT_FREE_CACHE_ENGINE=True
fi

case "$BATCH_PROFILE" in
  min)
    export TRAIN_DATA_SIZE=$N_GPUS
    export VAL_DATA_SIZE=1
    export GROUP_SIZE=1
    export PPO_MINI_BATCH_SIZE=$N_GPUS
    export PPO_MICRO_BATCH_SIZE_PER_GPU=1
    export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
    export AGENT_NUM_WORKERS=$N_GPUS
    export DATALOADER_NUM_WORKERS=0
    ;;
  beacon)
    export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
    export VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
    export GROUP_SIZE=${GROUP_SIZE:-8}
    export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
    export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
    export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
    export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
    export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-7000}
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
    export PACKED_TOKEN_BUDGET=${PACKED_TOKEN_BUDGET:-32768}
    export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$PACKED_TOKEN_BUDGET}
    export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PACKED_TOKEN_BUDGET}
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$PACKED_TOKEN_BUDGET}
    export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$PACKED_TOKEN_BUDGET}
    ;;
  *)
    log "[LLM judge judge] unknown TRACE_GRPO_BATCH_PROFILE=$BATCH_PROFILE; expected min or beacon."
    exit 2
    ;;
esac

export TEXT_FEEDBACK_BASE_URL="$JUDGE_BASE_URL"
export TEXT_FEEDBACK_MODEL="${TEXT_FEEDBACK_MODEL:?set TEXT_FEEDBACK_MODEL}"
export TEXT_FEEDBACK_API_KEY="${TEXT_FEEDBACK_API_KEY:-${INF_API_KEY:-}}"
export TEXT_FEEDBACK_MAX_WORKERS="${TEXT_FEEDBACK_MAX_WORKERS:-64}"
export TEXT_FEEDBACK_MAX_RETRIES="${TEXT_FEEDBACK_MAX_RETRIES:-1}"
export TEXT_FEEDBACK_TIMEOUT_S="${TEXT_FEEDBACK_TIMEOUT_S:-240.0}"
export TEXT_FEEDBACK_MAX_JUDGE_TOKENS="${TEXT_FEEDBACK_MAX_JUDGE_TOKENS:-4096}"
export TEXT_FEEDBACK_TEMPERATURE="${TEXT_FEEDBACK_TEMPERATURE:-0.0}"
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

log "[LLM judge judge] model=$TEXT_FEEDBACK_MODEL"
log "[LLM judge judge] api_base=$API_BASE"
log "[LLM judge judge] base_url=$TEXT_FEEDBACK_BASE_URL"
log "[LLM judge judge] max_workers=$TEXT_FEEDBACK_MAX_WORKERS"
log "[LLM judge judge] api_key_present=$([ -n \"$TEXT_FEEDBACK_API_KEY\" ] && echo 1 || echo 0)"
log "[LLM judge judge] profile=$TRACE_GRPO_PROFILE_MODE"
log "[LLM judge judge] batch_profile=$BATCH_PROFILE"
log "[LLM judge judge] n_gpus=$N_GPUS"
log "[LLM judge judge] train_data_size=${TRAIN_DATA_SIZE:-<inner-default>} val_data_size=${VAL_DATA_SIZE:-<inner-default>} group_size=${GROUP_SIZE:-<inner-default>}"
log "[LLM judge judge] agent_num_workers=${AGENT_NUM_WORKERS:-<inner-default>}"
log "[LLM judge judge] ppo_mini=${PPO_MINI_BATCH_SIZE:-<inner-default>} ppo_micro=${PPO_MICRO_BATCH_SIZE_PER_GPU:-<inner-default>} logprob_micro=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-<inner-default>}"
log "[LLM judge judge] max_prompt_length=${MAX_PROMPT_LENGTH:-<inner-default>} max_response_length=${MAX_RESPONSE_LENGTH:-<inner-default>} packed_token_budget=${PACKED_TOKEN_BUDGET:-<inner-default>}"
log "[LLM judge judge] rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-<inner-default>}"
log "[LLM judge judge] rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-<inner-default>}"
log "[LLM judge judge] steps=$STEPS"
log "[LLM judge judge] run_dir=$RUN_DIR"
log "[LLM judge judge] log_file=$LOG_FILE"

set +e
bash "$REPO_ROOT/trace_grpo/scripts/run_sciworld_trace_l3_8gpu_beacon.sh" "$STEPS" "${@:2}" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

log "[LLM judge judge] exit_code=$status"
exit "$status"
