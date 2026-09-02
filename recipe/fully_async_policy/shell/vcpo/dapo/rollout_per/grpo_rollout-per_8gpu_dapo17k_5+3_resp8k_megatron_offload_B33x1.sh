#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-rollout-per

# grpo_rollout-per_8gpu_dapo17k_5+3_resp8k_megatron_offload_B33x1.sh
#
# Rollout-level |A_i|-prioritized experience replay for GRPO -- the method of
# arXiv:2606.04560 (Yoo et al., "Rollout-Level Advantage-Prioritized Experience
# Replay for GRPO") on the fully-async 5+3 pipeline. Per trainer step:
#   1. pull ONE fresh 33-group batch (n=16 -> 528 rollouts), at most k=1 param
#      version stale (staleness_threshold=1, sync every update);
#   2. compute GRPO advantages, drop zero-variance groups (all-correct /
#      all-wrong) -> B'_fresh surviving rollouts;
#   3. draw replay_ratio*B'_fresh INDIVIDUAL past rollouts from an age-gated
#      buffer with PER probability ~ (|A_i|+eps)^alpha (without replacement);
#   4. ONE PPO-clip update on the fresh survivors + replay draw concatenated
#      (fresh-anchored composition; the whole variable-size batch is a single
#      mini-batch via meta_info.mini_batch_size -- see make_minibatch_iterator);
#   5. insert the survivors into the buffer (advantages, group stats and
#      behavior log-probs FROZEN at birth), evict rollouts older than tau_max
#      updates BEFORE each draw (capacity 30000 is a never-binding backstop);
#   6. replay is disabled for the first warmup_steps updates while the buffer
#      populates.
#
# LOSS (paper Table 7): pure token-level PPO clip -- use_rollout_log_probs=True
# aliases old_log_probs := the cached vLLM generation-time log-probs (the
# paper's pi_t) for fresh AND replayed rollouts, and rollout_correction stays
# at its null default so no IS machinery touches the loss (with old==rollout
# the correction pass is arithmetically inert: every ratio is exactly 1).
# DAPO Clip-Higher 0.2/0.28 with verl dual-clip c=10.0, token-mean
# aggregation, KL fully off, lr 1e-6 constant, weight decay 0.01, grad clip
# 1.0, ppo_epochs=1. calculate_entropy=True logs actor/entropy from inside the
# update (one extra logits clone; entropy_coeff stays 0).
#
# BATCH GEOMETRY: the gradient batch is VARIABLE-SIZE (survivors + draw). The
# driver nudges the draw by <=2 rollouts (or, during warmup, trims <=2 random
# fresh rows from the gradient batch only) so the total divides trainer DP=3 --
# verl's nd dispatch and DataProto.make_iterator both require equal chunks.
#
# HORIZON CAVEAT: rollout.total_rollout_steps counts FED prompts, and the
# zero-variance filter drops a large fraction of generated groups (the paper
# saw ~50-65%) -- the number of *trained* steps per fed prompt is unchanged,
# but each update carries fewer fresh rollouts than the 528 fed.
#
# STOP-THE-WORLD ACCOUNTING (serialize_validation /
# pause_generation_during_save, both True): the pipeline freezes for the whole
# validation sweep and the whole checkpoint save, so both are pure time
# translations and fully_async/timing/cumulative_training_time + the
# trajectory match a no-validation-no-save run exactly (the stalls are
# excluded from the virtual clock).
#
# CHECKPOINTS (save_contents=['hf_model'], resume_mode=auto,
# max_actor_ckpt_to_keep=null): each save writes global_step_N/actor/
# huggingface/ - config, tokenizer and bf16 safetensors - directly loadable by
# vLLM / from_pretrained, with NO optimizer state and NO dist_ckpt/ directory
# at all; timing_state.json is still written per save. 'hf_model' is written
# but never read back, so with the default contents resume_mode=auto behaves
# like disable on a fresh run dir (nothing to resume, trains from scratch),
# and a restart ON TOP OF existing checkpoints stops with the trainer's loud
# restores_model_weights refusal instead of silently training from pretrained
# under the old exp_name - archive the run dir before relaunching, as usual.
# To make auto resume real, override
# save_contents="['model','optimizer','extra','hf_model']" at launch (the
# replay buffer is still not persisted: a resumed run refills it within
# ~tau_max updates).
#
# Trainer layout: tp=1/dp=3 (sequence_parallel needs TP>1), HDO full CPU
# offload with bf16 master weights (do NOT swap for
# use_precision_aware_optimizer without optimizer_cpu_offload: silent stall,
# probe 2026-07-30).

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0

export PYTHONUNBUFFERED=1

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}

project_name='vcpo'

# ================= GPU Layout =================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=${n_gpus_rollout:-5}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# ================= Rollout =================
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
gen_tp=1
n_resp_per_prompt=${n_resp_per_prompt:-16}
gpu_memory_utilization=0.8
enable_chunked_prefill=True
# Mandatory for this arm: the cached generation-time log-probs are the PPO
# ratio's behavior anchor for fresh AND replayed rollouts.
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
max_response_length=${max_response_length:-8192}
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= Megatron Parallelism =================
train_tp=1 # only valid TP for 3 trainer GPUs (pure DP, no TP comm)
train_pp=1
train_cp=1
sequence_parallel=False # requires TP>1
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33} # 33*16=528 seqs; must divide by trainer DP=3 (528/3=176)
micro_bsz_per_gpu=1
use_dynamic_bsz=False
log_prob_micro_bsz_per_gpu=1

# ================= Algorithm (paper Table 7) =================
adv_estimator=grpo
loss_agg_mode="token-mean"
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
use_kl_loss=False
kl_loss_coef=0.0
use_kl_in_reward=False
kl_coef=0.0
entropy_coeff=${entropy_coeff:-0}
# Log actor/entropy from inside the update (there is no old-log-prob forward
# in alias mode to get it from); costs one logits clone.
calculate_entropy=True
grad_clip=1.0

# ================= Optimizer (paper Table 7) =================
lr=${lr:-1e-6}
lr_warmup_steps=0
weight_decay=0.01

# ================= Rollout-level PER replay (paper defaults) =================
replay_ratio=${replay_ratio:-0.5}         # r: draw r*B'_fresh replayed rollouts
priority_alpha=${priority_alpha:-0.5}     # PER exponent on p_i = |A_i| + eps
priority_eps=${priority_eps:-1.0e-6}
replay_tau_max=${replay_tau_max:-10}      # age eviction horizon (model updates)
replay_warmup_steps=${replay_warmup_steps:-20}
replay_capacity=${replay_capacity:-30000}
replay_sampling_seed=${replay_sampling_seed:-1234}
replay_with_replacement=${replay_with_replacement:-False}

compute_prox_log_prob=False

# ================= Async Training =================
# k=1: samples are at most one parameter version stale when consumed -- the
# async approximation of the paper's fresh on-policy anchor.
staleness_threshold=${staleness_threshold:-1.0}
updates_per_param_sync=1
num_minibatches_per_update=1 # require_batches=1: ONE 33-group pull per trainer step (required by replay mode)
partial_rollout=True
# True => old_log_probs := cached rollout log-probs (the paper's pi_t); the
# replay buffer freezes exactly these values for replayed rollouts.
use_rollout_log_probs=True

# ================= PPO epochs =================
ppo_epochs=${ppo_epochs:-1} # paper: one pass per gradient batch

# ================= Stop-the-world accounting =================
# Freeze the pipeline during validation / checkpoint saves so
# fully_async/timing/cumulative_training_time and the trajectory match a
# no-validation-no-save run exactly.
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

# ================= Training/Rollout Steps =================
# Counts FED prompts (see horizon caveat in the header): 66000 licenses up to
# ~2000 trainer steps of 33 fed groups.
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
# test/save freq are in param-version units; versions tick per 33-group step,
# so 20 = validate and checkpoint every 660 fed groups.
test_freq=${test_freq:-20}
save_freq=${save_freq:-20}
# Weights only, in huggingface format: no optimizer state (fp32 master + adam
# moments are ~6x the bf16 weights) and no dist_ckpt/ directory at all -
# global_step_N/actor/huggingface/ loads in vLLM as is. ~16.4 GB per save for
# Qwen3-8B and nothing is rotated away: raise save_freq at launch if the disk
# is tighter than ~100 saves' worth. Override save_contents to
# "['model','optimizer','extra','hf_model']" for a run that must be resumable.
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
resume_mode=${resume_mode:-auto}

# ================= Logging =================
exp_name=${exp_name:-"GRPO rollout-PER r-${replay_ratio} a-${priority_alpha} tau-${replay_tau_max} warmup-${replay_warmup_steps} k-${staleness_threshold} clip-0.2-0.28-c-10 DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz}x${num_minibatches_per_update} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
exp_name_safe=${exp_name//\//_}
log_dir="logs/${exp_name_safe}"
CKPTS_DIR="${log_dir}"
mkdir -p -- "${log_dir}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

trainer_logger="['console','tensorboard']"
log_val_generations=0
val_before_train=${val_before_train:-True}

# ================= LR decay =================
lr_decay_style="constant"
lr_decay_steps=${total_rollout_steps}

# ================= Run =================
python -m recipe.fully_async_policy.fully_async_main \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
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
    actor_rollout_ref.actor.optim.lr_decay_style=${lr_decay_style} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.main_params_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.ref.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.ref.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=${precision_dtype} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n:-1} \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    critic.megatron.tensor_model_parallel_size=${train_tp} \
    critic.megatron.pipeline_model_parallel_size=${train_pp} \
    critic.megatron.context_parallel_size=${train_cp} \
    critic.megatron.sequence_parallel=${sequence_parallel} \
    critic.megatron.dtype=${precision_dtype} \
    trainer.logger=${trainer_logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    trainer.resume_mode=${resume_mode} \
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}" \
    trainer.rollout_data_dir="${log_dir}" \
    trainer.log_val_generations=${log_val_generations} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs="${epochs}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${updates_per_param_sync}" \
    async_training.require_batches="${num_minibatches_per_update}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.compute_prox_log_prob="${compute_prox_log_prob}" \
    async_training.use_rollout_log_probs="${use_rollout_log_probs}" \
    async_training.serialize_validation="${serialize_validation}" \
    async_training.pause_generation_during_save="${pause_generation_during_save}" \
    async_training.rollout_replay.enable=True \
    async_training.rollout_replay.replay_ratio="${replay_ratio}" \
    async_training.rollout_replay.priority_alpha="${priority_alpha}" \
    async_training.rollout_replay.priority_eps="${priority_eps}" \
    async_training.rollout_replay.tau_max="${replay_tau_max}" \
    async_training.rollout_replay.warmup_steps="${replay_warmup_steps}" \
    async_training.rollout_replay.capacity="${replay_capacity}" \
    async_training.rollout_replay.sampling_seed="${replay_sampling_seed}" \
    async_training.rollout_replay.with_replacement="${replay_with_replacement}" "$@"
