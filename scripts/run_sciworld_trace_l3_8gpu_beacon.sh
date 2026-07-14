#!/usr/bin/env bash
# TRACE-GRPO ScienceWorld launch script (BEACON-aligned reference config).
#
# Reproduction target (spec PDF §11):
#   train_batch_size=16, group_size=8 -> 128 trajectories per step
#   max_response_length=16384, model_response_length=512
#   30 max env interact steps, 200 total training steps
#
# Required env:
#   TRACE_GRPO_TEACHER_KIND  default "env_score"; can be "llm" or "counterfactual"
#   TEXT_FEEDBACK_BASE_URL optional; defaults to <LLM_JUDGE> endpoint
#   TRACE_GRPO_L3_USE_CAPA   default 1 (CAPA path; falls back to reference if not wired)
#   TRACE_GRPO_L3_NEGATIVE_ONLY  default 1 (clamp w³ ≤ 1)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/verl"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export VERL_MASTER_PORT_RANGE=${VERL_MASTER_PORT_RANGE:-45000:47000}
export VERL_DIST_INIT_MAX_RETRIES=${VERL_DIST_INIT_MAX_RETRIES:-4}
export VERL_DIST_INIT_RETRY_SLEEP_S=${VERL_DIST_INIT_RETRY_SLEEP_S:-1.0}
export TRACE_GRPO_TEACHER_KIND=${TRACE_GRPO_TEACHER_KIND:-env_score}
export TRACE_GRPO_L3_USE_CAPA=${TRACE_GRPO_L3_USE_CAPA:-1}
export TRACE_GRPO_L3_NEGATIVE_ONLY=${TRACE_GRPO_L3_NEGATIVE_ONLY:-1}
export TEXT_FEEDBACK_BASE_URL=${TEXT_FEEDBACK_BASE_URL:?set TEXT_FEEDBACK_BASE_URL, e.g. https://your-endpoint.example.com/v1/chat/completions}
export TEXT_FEEDBACK_MODEL=${TEXT_FEEDBACK_MODEL:?set TEXT_FEEDBACK_MODEL, e.g. your-judge-model-name}
export TEXT_FEEDBACK_API_KEY=${TEXT_FEEDBACK_API_KEY:-${INF_API_KEY:-}}
export TEXT_FEEDBACK_MAX_WORKERS=${TEXT_FEEDBACK_MAX_WORKERS:-64}
export TEXT_FEEDBACK_MAX_RETRIES=${TEXT_FEEDBACK_MAX_RETRIES:-1}
export TEXT_FEEDBACK_TIMEOUT_S=${TEXT_FEEDBACK_TIMEOUT_S:-240.0}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

ENGINE=${ENGINE:-vllm}
# Auto-pick a local model path when MODEL_PATH is not explicitly set.
# Prefer 7B checkpoints; fall back to 1.5B only if 7B is unavailable.
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
  echo "[TRACE-GRPO run] ERROR: MODEL_PATH does not exist: ${MODEL_PATH:-<empty>}" >&2
  echo "[TRACE-GRPO run] Set MODEL_PATH to a local HF directory (with config/tokenizer/weights)." >&2
  exit 2
fi
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH")}
MODEL_TAG=${MODEL_TAG//./_}
MODEL_TAG=${MODEL_TAG//-/_}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-trace_${TRACE_GRPO_TEACHER_KIND}_${MODEL_TAG}}
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/data/sciworld_beacon}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
TRAIN_LOGGER=${TRAIN_LOGGER:-"['console']"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$REPO_ROOT/outputs/sciworld_rollout_jsonl/trace_${TRACE_GRPO_TEACHER_KIND}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$REPO_ROOT/outputs/sciworld_validation_jsonl/trace_${TRACE_GRPO_TEACHER_KIND}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-7000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
PACKED_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PACKED_TOKEN_BUDGET=${PACKED_TOKEN_BUDGET:-32768}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$PACKED_TOKEN_BUDGET}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PACKED_TOKEN_BUDGET}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-4096}

PROFILE_MODE=${TRACE_GRPO_PROFILE_MODE:-beacon_strict}
if [ "${PROFILE_MODE}" = "safe_debug" ]; then
  # Optional fallback profile for smoke/debug on memory-constrained runs.
  if [ -z "${PPO_MICRO_BATCH_SIZE_PER_GPU+x}" ]; then PPO_MICRO_BATCH_SIZE_PER_GPU=4; fi
  if [ -z "${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU+x}" ]; then LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4; fi
  if [ -z "${ROLLOUT_GPU_MEMORY_UTILIZATION+x}" ]; then ROLLOUT_GPU_MEMORY_UTILIZATION=0.35; fi
  if [ -z "${MAX_MODEL_LEN+x}" ]; then MAX_MODEL_LEN=${PACKED_TOKEN_BUDGET}; fi
  if [ -z "${MAX_NUM_BATCHED_TOKENS+x}" ]; then MAX_NUM_BATCHED_TOKENS=${PACKED_TOKEN_BUDGET}; fi
  if [ -z "${MAX_NUM_SEQS+x}" ]; then MAX_NUM_SEQS=64; fi
  if [ -z "${ROLLOUT_ENABLE_CHUNKED_PREFILL+x}" ]; then ROLLOUT_ENABLE_CHUNKED_PREFILL=True; fi
  if [ -z "${ROLLOUT_ENFORCE_EAGER+x}" ]; then ROLLOUT_ENFORCE_EAGER=True; fi
  if [ -z "${ROLLOUT_FREE_CACHE_ENGINE+x}" ]; then ROLLOUT_FREE_CACHE_ENGINE=True; fi
  # vLLM V1 rejects --disable-async-output-proc. Keep async output enabled
  # and use eager execution/max_num_seqs to stabilize debug rollouts.
  if [ -z "${ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC+x}" ]; then ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC=False; fi
  if [ -z "${DATALOADER_NUM_WORKERS+x}" ]; then DATALOADER_NUM_WORKERS=0; fi
else
  # Default profile for paper runs under TraceGRPO packed-trajectory layout.
  if [ -z "${PPO_MICRO_BATCH_SIZE_PER_GPU+x}" ]; then PPO_MICRO_BATCH_SIZE_PER_GPU=4; fi
  if [ -z "${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU+x}" ]; then LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4; fi
  if [ -z "${ROLLOUT_GPU_MEMORY_UTILIZATION+x}" ]; then ROLLOUT_GPU_MEMORY_UTILIZATION=0.35; fi
  if [ -z "${MAX_MODEL_LEN+x}" ]; then MAX_MODEL_LEN=${PACKED_TOKEN_BUDGET}; fi
  if [ -z "${MAX_NUM_BATCHED_TOKENS+x}" ]; then MAX_NUM_BATCHED_TOKENS=${PACKED_TOKEN_BUDGET}; fi
  if [ -z "${MAX_NUM_SEQS+x}" ]; then MAX_NUM_SEQS=1024; fi
  if [ -z "${ROLLOUT_ENABLE_CHUNKED_PREFILL+x}" ]; then ROLLOUT_ENABLE_CHUNKED_PREFILL=False; fi
  if [ -z "${ROLLOUT_ENFORCE_EAGER+x}" ]; then ROLLOUT_ENFORCE_EAGER=False; fi
  if [ -z "${ROLLOUT_FREE_CACHE_ENGINE+x}" ]; then ROLLOUT_FREE_CACHE_ENGINE=True; fi
  if [ -z "${ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC+x}" ]; then ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC=False; fi
  if [ -z "${DATALOADER_NUM_WORKERS+x}" ]; then DATALOADER_NUM_WORKERS=8; fi
fi

if [ ! -f "${DATASET_DIR}/train.parquet" ] || [ ! -f "${DATASET_DIR}/val.parquet" ]; then
  python -m trace_grpo.scripts.build_sciworld_parquet_beacon \
    --local_dir "${DATASET_DIR}" \
    --train_data_size "${TRAIN_DATA_SIZE}" \
    --val_data_size "${VAL_DATA_SIZE}"
fi

echo "[TRACE-GRPO run] profile_mode=${PROFILE_MODE} gpus=${N_GPUS} gpu_mem_util=${ROLLOUT_GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} max_num_seqs=${MAX_NUM_SEQS}"
echo "[TRACE-GRPO run] train_batch=${TRAIN_DATA_SIZE} val_batch=${VAL_DATA_SIZE}"
echo "[TRACE-GRPO run] train_max_samples=${TRAIN_DATA_SIZE} val_max_samples=${VAL_DATA_SIZE}"
echo "[TRACE-GRPO run] packed_sequence_length=${PACKED_SEQUENCE_LENGTH} packed_token_budget=${PACKED_TOKEN_BUDGET}"
echo "[TRACE-GRPO run] ppo_micro=${PPO_MICRO_BATCH_SIZE_PER_GPU} logprob_micro=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} ppo_mini=${PPO_MINI_BATCH_SIZE} rollout_n=${GROUP_SIZE}"
echo "[TRACE-GRPO run] hybrid_engine=True actor.strategy=fsdp rollout.name=${ENGINE} rollout.mode=async rollout.tp=1 ref.param_offload=True"
echo "[TRACE-GRPO run] rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL} rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE} rollout.disable_async_output_proc=${ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC}"
echo "[TRACE-GRPO run] rollout.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}"
echo "[TRACE-GRPO run] dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
echo "[TRACE-GRPO run] agent_num_workers=${AGENT_NUM_WORKERS}"
echo "[TRACE-GRPO run] master_port_range=${VERL_MASTER_PORT_RANGE} dist_init_retries=${VERL_DIST_INIT_MAX_RETRIES} dist_init_retry_sleep_s=${VERL_DIST_INIT_RETRY_SLEEP_S}"
echo "[TRACE-GRPO run] model_path=${MODEL_PATH}"
echo "[TRACE-GRPO run] experiment_name=${EXPERIMENT_NAME}"
echo "[TRACE-GRPO run] rollout_data_dir=${ROLLOUT_DATA_DIR}"
echo "[TRACE-GRPO run] validation_data_dir=${VALIDATION_DATA_DIR}"

python -m trace_grpo.launchers.main_ppo \
  algorithm.adv_estimator=trace_l3_mask \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_penalty=low_var_kl \
  +algorithm.trace_grpo.teacher_kind="${TRACE_GRPO_TEACHER_KIND}" \
  +algorithm.trace_grpo.text_feedback_lambda=0.2 \
  +algorithm.trace_grpo.text_feedback_sigma_eps=1.0e-3 \
  +algorithm.trace_grpo.text_feedback_alpha=0.5 \
  +algorithm.trace_grpo.trace_l3_enable=true \
  +algorithm.trace_grpo.trace_l3_alpha=0.3 \
  +algorithm.trace_grpo.trace_l3_kappa=1.0 \
  +algorithm.trace_grpo.trace_l3_negative_only=true \
  actor_rollout_ref.hybrid_engine=true \
  data.train_files="${DATASET_DIR}/train.parquet" \
  data.val_files="${DATASET_DIR}/val.parquet" \
  data.train_batch_size=${TRAIN_DATA_SIZE} \
  data.val_batch_size=${VAL_DATA_SIZE} \
  data.train_max_samples=${TRAIN_DATA_SIZE} \
  data.val_max_samples=${VAL_DATA_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
  data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.n=${GROUP_SIZE} \
  actor_rollout_ref.rollout.name=${ENGINE} \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB} \
  actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL} \
  actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_async_output_proc=${ROLLOUT_DISABLE_ASYNC_OUTPUT_PROC} \
  actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  +actor_rollout_ref.rollout.multi_turn.max_interact_steps=30 \
  +actor_rollout_ref.rollout.multi_turn.model_response_length=512 \
  actor_rollout_ref.rollout.agent.default_agent_loop=trace_sciworld \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=trace_grpo/configs/agent_loops.yaml \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
  +actor_rollout_ref.rollout.agent.use_beacon_train_pool=true \
  +actor_rollout_ref.rollout.agent.use_beacon_val_pool=true \
  +env.env_name=SciWorld \
  +env.seed=0 \
  +env.max_steps=30 \
  +env.history_length=2 \
  +env.sciworld.simplifications_preset=easy \
  trainer.total_training_steps=${1:-200} \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.resume_mode=disable \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.logger="${TRAIN_LOGGER}" \
  trainer.val_before_train=False \
  trainer.project_name=trace_grpo_sciworld \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="'${NCCL_P2P_DISABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="'${NCCL_IB_DISABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PXN_DISABLE="'${NCCL_PXN_DISABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE="'${NCCL_CUMEM_ENABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="'${NCCL_NVLS_ENABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MASTER_PORT_RANGE="'${VERL_MASTER_PORT_RANGE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DIST_INIT_MAX_RETRIES="'${VERL_DIST_INIT_MAX_RETRIES}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DIST_INIT_RETRY_SLEEP_S="'${VERL_DIST_INIT_RETRY_SLEEP_S}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'${VLLM_USE_V1}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND="'${VLLM_ATTENTION_BACKEND}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_BASE_URL="'${TEXT_FEEDBACK_BASE_URL}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MODEL="'${TEXT_FEEDBACK_MODEL:?set TEXT_FEEDBACK_MODEL}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_API_KEY="'${TEXT_FEEDBACK_API_KEY:-${INF_API_KEY:-}}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MAX_WORKERS="'${TEXT_FEEDBACK_MAX_WORKERS:-64}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MAX_RETRIES="'${TEXT_FEEDBACK_MAX_RETRIES:-1}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_TIMEOUT_S="'${TEXT_FEEDBACK_TIMEOUT_S:-240.0}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_MAX_JUDGE_TOKENS="'${TEXT_FEEDBACK_MAX_JUDGE_TOKENS:-4096}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEXT_FEEDBACK_TEMPERATURE="'${TEXT_FEEDBACK_TEMPERATURE:-0.0}'" \
  "${@:2}"
