#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen3-8b

set -x
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}

SEED=${SEED:-1}

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
filter_overlong_prompts=True
truncation='left'

train_prompt_bsz=${train_prompt_bsz:-128}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-32}
n_resp_per_prompt=${n_resp_per_prompt:-16}
ppo_epochs=${ppo_epochs:-1}

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.2
clip_ratio_c=3.0
loss_agg_mode="token-mean"
entropy_coeff=${entropy_coeff:-0}
calculate_entropy=True

lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-0}
weight_decay=${weight_decay:-0.01}
grad_clip=1.0

train_tp=${train_tp:-1}
train_pp=${train_pp:-1}
train_cp=${train_cp:-1}
sequence_parallel=False
precision_dtype=bfloat16
use_remove_padding=True

rollout_name=vllm
rollout_mode=async
gpu_memory_utilization=${gpu_memory_utilization:-0.5}
rollout_tp=1
enable_chunked_prefill=True
max_num_batched_tokens=$((1024 * 10))
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=${val_temperature:-0.8}
val_top_p=${val_top_p:-0.7}
calculate_log_probs=True

test_freq=${test_freq:-2}
save_freq=${save_freq:-2}
total_epochs=${total_epochs:-3}
max_updates=${max_updates:-null}
updates_per_step=$(( train_prompt_bsz / train_prompt_mini_bsz * ppo_epochs ))
(( updates_per_step >= 1 )) || { echo "updates_per_step must be >= 1 (train_prompt_bsz / train_prompt_mini_bsz * ppo_epochs)" >&2; exit 2; }
if [[ "${max_updates}" == "null" ]]; then
    total_training_steps=null
else
    [[ "${max_updates}" =~ ^[1-9][0-9]*$ ]] || { echo "max_updates must be a positive integer or null, got '${max_updates}'" >&2; exit 2; }
    total_training_steps=$(( (max_updates + updates_per_step - 1) / updates_per_step ))
fi
val_before_train=${val_before_train:-True}
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null}
resume_mode=${resume_mode:-disable}

NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn${n_resp_per_prompt} mini-${train_prompt_mini_bsz} ppo-epochs-${ppo_epochs} DAPO17K-AIME24-25 Qwen3-8B tp${train_tp}dp${n_gpus_per_node} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd seed-${SEED}"}
exp_name_safe=${exp_name//\//_}
log_dir="logs/${exp_name_safe}"
CKPTS_DIR="${log_dir}"
mkdir -p -- "${log_dir}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation="${truncation}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.seed=${SEED} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.filter_overlong_prompts_workers=8 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.megatron.seed=${SEED} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.actor.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.actor.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.main_params_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.dtype=${precision_dtype} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.megatron.seed=${SEED} \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=vcpo \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}" \
    trainer.resume_mode=${resume_mode} \
    trainer.rollout_data_dir=null \
    trainer.log_val_generations=0 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=${total_epochs} "$@"
