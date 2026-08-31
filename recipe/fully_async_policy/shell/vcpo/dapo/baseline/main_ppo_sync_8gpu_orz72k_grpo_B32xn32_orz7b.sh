#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-orz72k-grpo-orz7b

# SYNCHRONOUS ORZ-continuation arm: verl.trainer.main_ppo (the plain colocated
# hybrid-engine trainer, NOT recipe/fully_async_policy), GRPO without a critic,
# continuing Open-Reasoner-Zero-7B on the ORZ-72k collection with ORZ prompts.
#
# WHY SYNC. Every async arm on this model either collapsed (is-pg: no trust
# region, entropy collapse at ~step 90 after briefly reaching AIME-24 0.201;
# replay trig=0 arms: off-policy divergence) or stalled (sqrt-braked replay:
# stable, flat val for 600+ updates). ORZ's own regime — fully synchronous,
# strictly on-policy, one optimizer step per generation batch
# (playground/orz_7b_ppo.py in the reference repo) — is the setting proven to
# train this checkpoint for 700+ steps. This arm reproduces that regime's
# schedule with GRPO in place of their PPO+critic:
#   * train_batch_size = ppo_mini_batch_size = 32 prompts, rollout.n = 32
#     -> 1024 sequences per optimizer step, exactly ONE optimizer step per
#     generation batch (the PPO ratio is identically 1, so the clip is inert
#     insurance; the loss degenerates to on-policy policy gradient — the same
#     effective loss ORZ's policy update has in its strictly on-policy limit).
#   * ppo_epochs=1 (ORZ parity; the collapsed is-pg arm ran 2).
#   * weight_decay=0 (ORZ parity; async arms ran 0.1).
#   * lr 1e-6 constant, NO warmup (deliberate deviation from ORZ's 50 steps).
#   * no KL in reward, no KL loss, no entropy bonus, T=1.0/top_p=1.0 (parity).
#   * clip 0.2/0.28: clip-higher is a second deliberate deviation — inert at
#     ratio==1, but if batch geometry is ever changed to multi-minibatch it
#     bounds drift asymmetrically (DAPO-style), the right default for a model
#     whose entropy starts at ~0.06.
#   * GRPO differences vs ORZ kept deliberately: group advantages instead of
#     GAE+critic (user choice; all-correct/all-wrong groups become zero-
#     advantage no-ops, there is no group filter here), and the scorer returns
#     +/-1 instead of ORZ's 1/0 — equivalent under GRPO group normalization.
#
# DATA. Training: orz-math-72k parquet (47,981 deduplicated problems, ORZ
# inner instruction verbatim; built by scripts/convert_orz72k_to_verl_parquet
# .py on the replay_buffer_vcpo_ess_threshold_final branch). Validation:
# aime-2024/2025 ORZ-prompt parquets exactly as that branch's ORZ-72k arms
# (x32 duplication, data_source aime2024_orz/aime2025_orz -> the same
# val-core/.../acc/mean@1 metric keys, so curves are directly comparable).
#
# REWARD. recipe/fully_async_policy/reward/orz_tag_aware_math.py (the tiered
# version ported from the _final branch): tag-aware extraction, then
# math_dapo string equality -> vendored ORZ is_equiv -> sympy parse_latex in
# a forked child under a hard ORZ_MATH_SYMPY_TIMEOUT=1.0s kill deadline.
# ~24% of ORZ-72k ground truths are LaTeX expressions; the pre-tier scorer
# would score them as false negatives.
#
# ENTROPY WATCH. No stabilizer by design (matching ORZ). actor/entropy is
# logged every step (calculate_entropy=True): this model starts at ~0.06 —
# watch the first ~50 steps; a monotone actor/grad_norm ramp (healthy: flat
# 0.15-0.2) is the earliest divergence signal known from the async post-
# mortems. Sync on-policy is the regime where ORZ never saw a collapse.

set -x
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Open-Reasoner-Zero/Open-Reasoner-Zero-7B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/orz/orz-math-72k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/orz/aime-2024-orz.parquet','/home/jovyan/datasets/math_datasets/orz/aime-2025-orz.parquet']"}
REWARD_FILE=${REWARD_FILE:-"recipe/fully_async_policy/reward/orz_tag_aware_math.py"}

# ================= Data =================
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
filter_overlong_prompts=True
truncation='left'

# ================= Batch geometry (strictly on-policy) =================
# ONE optimizer step per generation batch: train_batch_size == ppo_mini_batch
# _size, ppo_epochs=1. 32 prompts x 32 rollouts = 1024 seqs per step.
train_prompt_bsz=${train_prompt_bsz:-32}
n_resp_per_prompt=${n_resp_per_prompt:-32}
ppo_epochs=${ppo_epochs:-1}

# ================= Algorithm =================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=3.0
# ORZ parity: their packed PolicyLoss degenerates to a global token mean
# (action_mask is None under packing, so masked_mean(...).mean() is a plain
# mean over ~equal-length packed rows) — verl's token-mean, NOT the house
# seq-mean-token-mean. Every token weighs equally: long CoT gets
# proportionally more gradient, short degenerate responses get diluted —
# the direction that would have damped the AceReason 1-token spiral.
loss_agg_mode="token-mean"
entropy_coeff=${entropy_coeff:-0}
calculate_entropy=True

# ================= Optimizer =================
lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-0}
weight_decay=${weight_decay:-0}
grad_clip=1.0

# ================= Parallelism / precision =================
train_tp=${train_tp:-1}
train_pp=${train_pp:-1}
train_cp=${train_cp:-1}
sequence_parallel=False
precision_dtype=bfloat16
use_remove_padding=True

# ================= Rollout =================
rollout_name=vllm
rollout_mode=async
gpu_memory_utilization=${gpu_memory_utilization:-0.6}
rollout_tp=1
enable_chunked_prefill=True
max_num_batched_tokens=$((1024 * 10))
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=${val_temperature:-1.0}
# Cache vLLM's per-token log-probs so the driver computes the rollout_corr/*
# diagnostics (KL, pearson, IS tails — the early-warning surface every async
# post-mortem relied on). NO correction is applied: algorithm.rollout_
# correction.rollout_is stays null, so the helper adds metrics but no weights
# ("Metrics can be monitored before enabling IS weight correction"); the loss
# stays the uncorrected ORZ-parity objective. Flip rollout_is=token t=2.0
# later if the engine gap ever grows.
calculate_log_probs=True

# ================= Trainer =================
test_freq=${test_freq:-10}
save_freq=${save_freq:-50}
total_epochs=${total_epochs:-1}
val_before_train=${val_before_train:-True}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null}
ckpt_save_contents="['hf_model']"
resume_mode=disable

NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

# ================= Logging =================
exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn${n_resp_per_prompt} ppo-epochs-${ppo_epochs} ORZ72K-AIME24ORZ ORZ-7B tp${train_tp}dp${n_gpus_per_node} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.filter_overlong_prompts_workers=8 \
    custom_reward_function.path="${REWARD_FILE}" \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
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
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=vcpo \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    actor_rollout_ref.actor.checkpoint.save_contents="${ckpt_save_contents}" \
    trainer.resume_mode=${resume_mode} \
    trainer.rollout_data_dir=null \
    trainer.log_val_generations=0 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.total_epochs=${total_epochs} "$@"
