#!/usr/bin/env bash
# Vanilla GRPO baseline on the same data + agent loop, used to verify
# that our env loop / reward shaping match BEACON's published numbers
# before we put TRACE-GRPO on top.
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

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

ENGINE=${ENGINE:-vllm}
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/data/sciworld_beacon}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
TRAIN_LOGGER=${TRAIN_LOGGER:-"['console']"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$REPO_ROOT/outputs/sciworld_rollout_jsonl/grpo_baseline}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$REPO_ROOT/outputs/sciworld_validation_jsonl/grpo_baseline}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-7000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
PACKED_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PACKED_TOKEN_BUDGET=${PACKED_TOKEN_BUDGET:-32768}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$PACKED_TOKEN_BUDGET}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PACKED_TOKEN_BUDGET}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$PACKED_TOKEN_BUDGET}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$PACKED_TOKEN_BUDGET}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}

if [ ! -f "${DATASET_DIR}/train.parquet" ] || [ ! -f "${DATASET_DIR}/val.parquet" ]; then
  python -m trace_grpo.scripts.build_sciworld_parquet_beacon \
    --local_dir "${DATASET_DIR}" \
    --train_data_size "${TRAIN_DATA_SIZE}" \
    --val_data_size "${VAL_DATA_SIZE}"
fi

python -m trace_grpo.launchers.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.hybrid_engine=true \
  data.train_files="${DATASET_DIR}/train.parquet" \
  data.val_files="${DATASET_DIR}/val.parquet" \
  data.train_batch_size=${TRAIN_DATA_SIZE} \
  data.val_batch_size=${VAL_DATA_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
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
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
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
  trainer.experiment_name=grpo_baseline_qwen2.5_1.5b \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="'${NCCL_P2P_DISABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="'${NCCL_IB_DISABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PXN_DISABLE="'${NCCL_PXN_DISABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE="'${NCCL_CUMEM_ENABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="'${NCCL_NVLS_ENABLE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'${VLLM_USE_V1}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND="'${VLLM_ATTENTION_BACKEND}'" \
  "${@:2}"
