#!/usr/bin/env bash
# BEACON-production comparison launch for TRACE-GRPO on AlfWorld.
#
# This script mirrors run_sciworld_trace_l3_beacon_prod_llmjudge.sh: it
# fixes task sampling (BEACON-aligned alfworld split pool), env, rollout
# count, temperatures, evaluation cadence, and reward/metric semantics.
# Parameters whose meaning changes under packed-trajectory training are
# adapted below and logged explicitly.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$REPO_ROOT"

STEPS="${1:-150}"
if [ "$#" -gt 0 ]; then
  shift
fi
if [ "$#" -gt 0 ] && [ "${TRACE_GRPO_ALLOW_EXTRA_OVERRIDES:-0}" != "1" ]; then
  echo "[beacon-prod] Refusing extra Hydra overrides in strict mode: $*" >&2
  echo "[beacon-prod] Set TRACE_GRPO_ALLOW_EXTRA_OVERRIDES=1 only for intentional ablations." >&2
  exit 2
fi
EXTRA_ARGS=("$@")

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/verl:$REPO_ROOT/BEACON"
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

# Match BEACON ScienceWorld launch environment where possible. Same vLLM /
# FA2 settings as the validated 7B sciworld smoke.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export VERL_MASTER_PORT_RANGE=${VERL_MASTER_PORT_RANGE:-45000:47000}
export VERL_DIST_INIT_MAX_RETRIES=${VERL_DIST_INIT_MAX_RETRIES:-4}
export VERL_DIST_INIT_RETRY_SLEEP_S=${VERL_DIST_INIT_RETRY_SLEEP_S:-1.0}

if [ -z "${ALFWORLD_DATA:-}" ]; then
  echo "[beacon-prod] ERROR: ALFWORLD_DATA env var is required (alfworld json + game files dir)." >&2
  exit 2
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# BEACON AlfWorld production task/eval knobs.
ENGINE=vllm
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
PPO_MICRO_BATCH_SIZE_PER_GPU=2
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
AGENT_NUM_WORKERS=16
DATALOADER_NUM_WORKERS=8
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC=${ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC:-False}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}

# Packed-trajectory adaptation:
# verl counts ppo_mini_batch_size in PROMPTS, not rows. The validation
# `train_batch_size >= ppo_mini_batch_size` (workers/config/actor.py:214)
# is checked BEFORE the internal `ppo_mini_batch_size *= rollout.n` scaling
# (fsdp_workers.py:326). So to reproduce GiGPO's 1-update-per-batch cadence
# (train_batch=16 prompts × group_size=8 = 128 rows consumed in a single
# PPO update), we set ppo_mini_batch_size=16 — which verl then internally
# expands to 16*8=128 rows, then divides by n_gpus to get the per-GPU
# mini-batch.
PPO_MINI_BATCH_SIZE=16

# BEACON uses 512 response tokens per step. AlfWorld trajectories run up to
# 50 steps (vs. sciworld's 30), so the packed budget is bumped accordingly.
MAX_PROMPT_LENGTH=8192
MODEL_RESPONSE_LENGTH=512
MAX_RESPONSE_LENGTH=24576
PACKED_TOKEN_BUDGET=${PACKED_TOKEN_BUDGET:-32768}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$PACKED_TOKEN_BUDGET
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$PACKED_TOKEN_BUDGET
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$PACKED_TOKEN_BUDGET}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$PACKED_TOKEN_BUDGET}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=4096

if [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_MODEL_LEN" ]; then
  echo "[beacon-prod] ERROR: MAX_NUM_BATCHED_TOKENS($MAX_NUM_BATCHED_TOKENS) must be >= MAX_MODEL_LEN($MAX_MODEL_LEN)." >&2
  exit 2
fi

if [ -z "${MODEL_PATH:-}" ]; then
  for candidate in \
    "$REPO_ROOT/models/Qwen2.5-7B-http" \
    "$REPO_ROOT/models/Qwen2.5-7B-Instruct" \
    "$REPO_ROOT/models/Qwen2.5-7B" \
    "$REPO_ROOT/models/Qwen2.5-1.5B-Instruct"
  do
    if [ -d "$candidate" ]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi
if [ -z "${MODEL_PATH:-}" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[beacon-prod] ERROR: MODEL_PATH does not exist: ${MODEL_PATH:-<empty>}" >&2
  exit 2
fi
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
MODEL_TAG=${MODEL_TAG//./_}
MODEL_TAG=${MODEL_TAG//-/_}
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/data/alfworld_beacon_prod}
ALFWORLD_VAL_SPLIT=${ALFWORLD_VAL_SPLIT:-valid_seen}
ALFWORLD_EVAL_DATASET=${ALFWORLD_EVAL_DATASET:-eval_in_distribution}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/outputs/alfworld_beacon_prod}
RUN_DIR="$LOG_ROOT/$RUN_TAG"
if [ -d "$RUN_DIR" ]; then
  _suffix=1
  while [ -d "${RUN_DIR}_${_suffix}" ]; do _suffix=$((_suffix + 1)); done
  RUN_DIR="${RUN_DIR}_${_suffix}"
fi
LOG_FILE="$RUN_DIR/train.log"
ROLLOUT_DATA_DIR="$RUN_DIR/rollout_jsonl"
VALIDATION_DATA_DIR="$RUN_DIR/validation_jsonl"
TRAIN_LOGGER=${TRAIN_LOGGER:-"['console']"}
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-trace_grpo_alfworld}
ALLOW_REUSE_RUN_DIR=${ALLOW_REUSE_RUN_DIR:-0}

API_BASE=${API_BASE:?set API_BASE, e.g. https://your-endpoint.example.com/v1}
TEXT_FEEDBACK_BASE_URL=${TEXT_FEEDBACK_BASE_URL:-${API_BASE%/}/chat/completions}
TEXT_FEEDBACK_MODEL=${TEXT_FEEDBACK_MODEL:?set TEXT_FEEDBACK_MODEL, e.g. your-judge-model-name}
TEXT_FEEDBACK_API_KEY=${TEXT_FEEDBACK_API_KEY:-${INF_API_KEY:-}}
TEXT_FEEDBACK_MAX_WORKERS=64
TEXT_FEEDBACK_MAX_RETRIES=1
TEXT_FEEDBACK_TIMEOUT_S=240.0
TEXT_FEEDBACK_MAX_JUDGE_TOKENS=4096
TEXT_FEEDBACK_TEMPERATURE=0.0

if [ -d "$RUN_DIR" ] && [ "$ALLOW_REUSE_RUN_DIR" != "1" ]; then
  if find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "[beacon-prod] ERROR: RUN_DIR already exists and is non-empty: $RUN_DIR" >&2
    echo "[beacon-prod] Refusing to reuse run directory to avoid mixed artifacts/logs." >&2
    echo "[beacon-prod] Set ALLOW_REUSE_RUN_DIR=1 only if you intentionally resume/debug in place." >&2
    exit 2
  fi
fi

mkdir -p "$RUN_DIR" "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"
ln -sfn "$RUN_DIR" "$LOG_ROOT/latest"

log() {
  echo "$1" | tee -a "$LOG_FILE"
}

python -m trace_grpo.scripts.build_alfworld_parquet_beacon \
  --local_dir "$DATASET_DIR" \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_DATA_SIZE" \
  --val_split "$ALFWORLD_VAL_SPLIT" | tee -a "$LOG_FILE"

export TRACE_GRPO_ALFWORLD_POOL_PATH="${TRACE_GRPO_ALFWORLD_POOL_PATH:-$DATASET_DIR/index.json}"

log "[beacon-prod] script=run_alfworld_trace_l3_beacon_prod_llmjudge.sh"
log "[beacon-prod] steps=$STEPS gpus=$N_GPUS cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
log "[beacon-prod] model_path=$MODEL_PATH"
log "[beacon-prod] alfworld_data=$ALFWORLD_DATA val_split=$ALFWORLD_VAL_SPLIT eval_dataset=$ALFWORLD_EVAL_DATASET"
log "[beacon-prod] pool_index=$TRACE_GRPO_ALFWORLD_POOL_PATH"
log "[beacon-prod] teacher=llm model=$TEXT_FEEDBACK_MODEL base_url=$TEXT_FEEDBACK_BASE_URL api_key_present=$([ -n "$TEXT_FEEDBACK_API_KEY" ] && echo 1 || echo 0)"
log "[beacon-prod] beacon_semantics: train_batch=$TRAIN_DATA_SIZE val_batch=$VAL_DATA_SIZE group_size=$GROUP_SIZE"
log "[beacon-prod] packed_adapted: ppo_mini=$PPO_MINI_BATCH_SIZE prompts (verl scales ×rollout.n=8 → 128 rows = 1 update/batch, matches GiGPO) ppo_micro=$PPO_MICRO_BATCH_SIZE_PER_GPU logprob_micro=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
log "[beacon-prod] packed_adapted: max_prompt=$MAX_PROMPT_LENGTH model_response_per_turn=$MODEL_RESPONSE_LENGTH packed_response=$MAX_RESPONSE_LENGTH token_budget=$PACKED_TOKEN_BUDGET"
log "[beacon-prod] rollout_gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE attention_backend=$VLLM_ATTENTION_BACKEND"
log "[beacon-prod] max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
log "[beacon-prod] rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER rollout.disable_async_output_proc=$ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC rollout.update_weights_bucket_megabytes=$ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB"
log "[beacon-prod] master_port_range=$VERL_MASTER_PORT_RANGE dist_init_retries=$VERL_DIST_INIT_MAX_RETRIES"
log "[beacon-prod] train/val task pools: alfworld split via use_beacon_train_pool=true,use_beacon_val_pool=true"
log "[beacon-prod] trainer.total_epochs=$STEPS because train dataloader has one alfworld batch per epoch"
log "[beacon-prod] run_dir=$RUN_DIR"

set +e
python -m trace_grpo.launchers.main_ppo \
  algorithm.adv_estimator=trace_l3_mask \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_penalty=low_var_kl \
  +algorithm.trace_grpo.teacher_kind=llm \
  +algorithm.trace_grpo.text_feedback_lambda=0.2 \
  +algorithm.trace_grpo.text_feedback_sigma_eps=1.0e-3 \
  +algorithm.trace_grpo.text_feedback_alpha=0.5 \
  +algorithm.trace_grpo.trace_l3_enable=true \
  +algorithm.trace_grpo.trace_l3_alpha=0.3 \
  +algorithm.trace_grpo.trace_l3_kappa=1.0 \
  +algorithm.trace_grpo.trace_l3_negative_only=true \
  actor_rollout_ref.hybrid_engine=true \
  data.train_files="$DATASET_DIR/train.parquet" \
  data.val_files="$DATASET_DIR/val.parquet" \
  data.train_batch_size=$TRAIN_DATA_SIZE \
  data.val_batch_size=$VAL_DATA_SIZE \
  data.train_max_samples=$TRAIN_DATA_SIZE \
  data.val_max_samples=$VAL_DATA_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.filter_overlong_prompts=True \
  data.dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  +actor_rollout_ref.actor.use_invalid_action_penalty=True \
  +actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.rollout.n=$GROUP_SIZE \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB \
  actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL \
  actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_async_output_proc=$ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC \
  actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  +actor_rollout_ref.rollout.multi_turn.max_interact_steps=50 \
  +actor_rollout_ref.rollout.multi_turn.model_response_length=$MODEL_RESPONSE_LENGTH \
  actor_rollout_ref.rollout.agent.default_agent_loop=trace_alfworld \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=trace_grpo/configs/agent_loops.yaml \
  actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
  +actor_rollout_ref.rollout.agent.use_beacon_train_pool=true \
  +actor_rollout_ref.rollout.agent.use_beacon_val_pool=true \
  +env.env_name=AlfWorld \
  +env.seed=0 \
  +env.max_steps=50 \
  +env.history_length=2 \
  +env.alfworld.eval_dataset=$ALFWORLD_EVAL_DATASET \
  trainer.total_training_steps=$STEPS \
  trainer.total_epochs=$STEPS \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.resume_mode=disable \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.logger="$TRAIN_LOGGER" \
  trainer.val_before_train=True \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name=trace_l3_alfworld_beacon_prod_${MODEL_TAG} \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="'$NCCL_P2P_DISABLE'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="'$NCCL_IB_DISABLE'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PXN_DISABLE="'$NCCL_PXN_DISABLE'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE="'$NCCL_CUMEM_ENABLE'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="'$NCCL_NVLS_ENABLE'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MASTER_PORT_RANGE="'$VERL_MASTER_PORT_RANGE'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DIST_INIT_MAX_RETRIES="'$VERL_DIST_INIT_MAX_RETRIES'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DIST_INIT_RETRY_SLEEP_S="'$VERL_DIST_INIT_RETRY_SLEEP_S'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'$VLLM_USE_V1'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND="'$VLLM_ATTENTION_BACKEND'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_DATA="'$ALFWORLD_DATA'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRACE_GRPO_ALFWORLD_POOL_PATH="'$TRACE_GRPO_ALFWORLD_POOL_PATH'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_BASE_URL="'$TEXT_FEEDBACK_BASE_URL'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MODEL="'$TEXT_FEEDBACK_MODEL'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_API_KEY="'$TEXT_FEEDBACK_API_KEY'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MAX_WORKERS="'$TEXT_FEEDBACK_MAX_WORKERS'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MAX_RETRIES="'$TEXT_FEEDBACK_MAX_RETRIES'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_TIMEOUT_S="'$TEXT_FEEDBACK_TIMEOUT_S'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MAX_JUDGE_TOKENS="'$TEXT_FEEDBACK_MAX_JUDGE_TOKENS'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_TEMPERATURE="'$TEXT_FEEDBACK_TEMPERATURE'" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

log "[beacon-prod] exit_code=$status"

if [ "$status" -eq 0 ]; then
  max_rollout_step=0
  if find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' | grep -q .; then
    max_rollout_step=$(find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' | sed 's/\.jsonl$//' | sort -n | tail -1)
  fi

  max_validation_step=0
  if find "$VALIDATION_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' | grep -q .; then
    max_validation_step=$(find "$VALIDATION_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' | sed 's/\.jsonl$//' | sort -n | tail -1)
  fi

  log "[beacon-prod] postcheck rollout_max_step=$max_rollout_step validation_max_step=$max_validation_step expected_steps=$STEPS"
  if [ "${max_rollout_step:-0}" -ne "$STEPS" ] || [ "${max_validation_step:-0}" -ne "$STEPS" ]; then
    log "[beacon-prod] ERROR: run terminated before expected steps (or artifacts were mixed)."
    log "[beacon-prod] ERROR: expected rollout/validation max step = $STEPS."
    exit 3
  fi
fi

exit "$status"
