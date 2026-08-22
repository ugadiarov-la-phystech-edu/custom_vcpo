#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo

# grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh
#
# ARM A of the pair that reproduces the same-named arm of branch 'rollout-dapo' on
# this (pristine-verl + cumulative_training_time) branch. Both arms share every
# schedule/optimizer setting below and differ ONLY in how the off-policy gap is
# corrected:
#   * A = this file: IS-weighted policy gradient, no trust region -- what the
#     source arm EXECUTED.
#   * B = ..._decoupled.sh: 3-policy decoupled PPO, live clip + IS -- what the
#     source arm DECLARED (bypass_mode=False).
#
# WHY THIS IS THE SOURCE ARM'S OBJECTIVE. The source sets
# async_training.skip_recompute_old_log_prob=True (absent on this branch): the actor
# uses old_log_prob = log_prob.detach(), so the PPO ratio is exactly 1, clipping and
# dual-clip never bind, and the entire correction is trunc(pi_theta/pi_rollout, 2.0)
# recomputed per micro-batch. Base verl has the identical objective natively:
#   actor.policy_loss.loss_mode=rollout_correction
#     -> compute_policy_loss_with_rollout_correction (verl/trainer/ppo/core_algos.py):
#        L = -E[w * log pi * A], w = trunc(pi_theta/pi_rollout, rollout_is_threshold),
#        computed on the fly per micro-batch, NO PPO clipping, no extra forward pass.
# use_rollout_log_probs=True keeps old_log_probs := rollout_log_probs, i.e. the
# 2-policy ("bypass") substitution that loss mode expects.
#
# LOG SURFACE (differs from arm B -- do not plot the two together blindly):
#   * actor/pg_clipfrac and pg_clipfrac_lower do NOT exist here: the loss never
#     clips. (The source arm logs them as identically 0.)
#   * actor/ppo_kl is KL(current || rollout) here, but KL(current || old) in arm B.
#   * rollout_corr/* come from the actor, per micro-batch, and are the real IS
#     statistics; the driver's own correction pass is degenerate (old == rollout)
#     and its values are overwritten by the actor's.
#   * actor/entropy is measured INSIDE the update here (calculate_entropy=True);
#     arm B measures it at pull time instead.
#
# Schedule (identical in both arms, unchanged from the source):
#   * require_batches=1: the pull is ONE 33-group mini-batch per trainer step
#     (B-33x1). Model versions tick per 33-group step.
#   * TWO AdamW updates per step. The source used driver-side
#     async_training.ppo_epochs=2; here actor_rollout_ref.actor.ppo_epochs=2 does the
#     same, because megatron_workers.py scales ppo_mini_batch_size by rollout.n
#     (33*16=528 = the whole pull), so make_iterator(epochs=2) runs the same 2 steps.
#   * staleness_threshold=2, total_rollout_steps=66000 explicit, test_freq=save_freq=10.
#   * serialize_validation / pause_generation_during_save: stop-the-world validation
#     and saves, excluded from fully_async/timing/cumulative_training_time.
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
#   * each save writes global_step_N/actor/huggingface/ - config, tokenizer and bf16
#     safetensors - directly loadable by vLLM / from_pretrained, no merge step. No
#     optimizer state, no sharded dist_ckpt/ directory at all.
#   * nothing is rotated away: ~16.4 GB per save for Qwen3-8B, ~200 saves over the
#     full run is ~3.3 TB. Raise save_freq at launch if the disk is tighter.
#   * the run is NOT resumable: 'hf_model' is written but never read back, so
#     load_contents would restore nothing. resume_mode=disable makes that explicit,
#     and the trainer raises rather than resuming from such a checkpoint.
#
# ACCEPTED DIVERGENCES from the 'rollout-dapo' script (dropped, not emulated):
#   * math500 validation set -> the math500_dapo scorer is not in this branch's
#     reward registry, so validation is AIME-2024 only.
#   * +async_training.bsz_per_dp_rank=33 -> not ported: max_concurrent_samples stays
#     min(5 servers * 16, 99) = 80 instead of 99 (the full staleness budget).
#   * algorithm.rollout_correction.log_probs_pearson_corr -> not ported; the same
#     policy-vs-rollout pair is already covered by rollout_corr/* (KL, ESS).
#   * async_training.{dynamic_filtering,opportunistic_epochs} -> both were OFF in the
#     source arm, so dropping them changes nothing.
#
# Base-script notes that still apply: trainer tp=1/dp=3 (sequence_parallel needs
# TP>1), 33*16=528 seqs divide by DP=3, HDO full CPU offload with bf16 master weights
# (do NOT swap for use_precision_aware_optimizer without optimizer_cpu_offload:
# silent stall, probe 2026-07-30). calculate_entropy=True clones the logits on the
# non-fused megatron path (~3 GB at 10k tokens x 152k vocab) inside the training
# forward; if the trainer OOMs, set use_fused_kernels=True or calculate_entropy=False.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# AIME-2024 only (data_source=math_dapo) -> val-core/math_dapo/acc/mean@1.
# The rollout-dapo arm also validated on math500.parquet; that set needs the
# math500_dapo scorer, which this branch does not carry.
TEST_FILE=${TEST_FILE:-"/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet"}

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
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
max_response_length=8192
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
train_prompt_mini_bsz=33 # 33*16=528 seqs; must divide by trainer DP=3 (528/3=176)
micro_bsz_per_gpu=1
use_dynamic_bsz=False
log_prob_micro_bsz_per_gpu=1

# ================= Algorithm =================
adv_estimator=grpo
loss_agg_mode="seq-mean-token-mean"
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.2
clip_ratio_c=3.0
use_kl_loss=False
kl_loss_coef=0.0
use_kl_in_reward=False
kl_coef=0.0
entropy_coeff=0
# Log actor/entropy even with entropy_coeff=0 (honoured via should_calculate_entropy
# in verl/workers/actor/megatron_actor.py). This arm has no old-log-prob forward, so
# it is the only source of entropy here -- see the memory note in the header.
calculate_entropy=True
grad_clip=1.0

# ================= Optimizer =================
lr=1e-6
lr_warmup_steps=0
weight_decay=0.1

# ================= IS / Rollout Correction =================
# Token-level truncated IS at 2.0, applied as a pure policy-gradient correction with
# no PPO clipping (the source arm's skip_recompute behaviour, expressed natively).
# bypass_mode/use_policy_gradient describe exactly this mode, but note they are INERT
# on the fully-async path: their only consumer, apply_rollout_correction(), is called
# from verl/trainer/ppo/ray_trainer.py, never from this recipe. What actually selects
# the behaviour is policy_loss.loss_mode below; the recipe performs the bypass
# substitution itself via async_training.use_rollout_log_probs=True.
rollout_is="token"
rollout_is_threshold="2.0"
rollout_rs=null
rollout_rs_threshold=null
bypass_mode=True
use_policy_gradient=True
policy_loss_mode="rollout_correction"

compute_prox_log_prob=False

# ================= Async Training =================
# k=2 matches the VCPO baseline's staleness gating: the rollouter is licensed
# to generate up to (2+1) trainer batches ahead — at B-33x1 that is 99
# in-flight/queued groups.
staleness_threshold=${staleness_threshold:-2.0}
updates_per_param_sync=1
num_minibatches_per_update=1 # require_batches=1: ONE 33-group mini-batch per trainer step (B-33x1)
partial_rollout=True
use_rollout_log_probs=True

# ================= PPO epochs =================
# Stock worker-internal loop: 2 passes over the mini-batches of the pull. With
# require_batches=1 the pull IS one 33-group mini-batch, so this is exactly the
# 2 AdamW updates per trainer step the rollout-dapo arm ran.
ppo_epochs=${ppo_epochs:-2}

# ================= Stop-the-world accounting =================
# Freeze the pipeline during validation / checkpoint saves so
# fully_async/timing/cumulative_training_time and the trajectory match a
# no-validation-no-save run exactly.
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

# ================= Training/Rollout Steps =================
# Explicit 66000 (NOT the base arms' 500-step formula, which at B-33x1 would
# shrink to 500*1*1*33 = 16500): same generation budget as the B-33x4 arms,
# licensing up to ~2000 trainer steps of 33 groups.
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
# test/save freq are in param-version units; versions tick per 33-group step
# here, so 10 = every 330 groups.
test_freq=${test_freq:-10}
save_freq=${save_freq:-10}
# Weights only, in huggingface format: no optimizer state (fp32 master + 2 adam moments is ~6x
# the bf16 weights on the megatron distributed optimizer, and every save here is stop-the-world)
# and no merge step before offline eval - global_step_N/actor/huggingface/ loads in vLLM as is.
# ~16.4 GB per save for Qwen3-8B, and nothing is rotated away: ~200 saves over the run is ~3.3 TB,
# so raise save_freq at launch if the disk is tighter than that.
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
# Mandatory, not cosmetic: load_contents mirrors save_contents, and 'hf_model' is written but never
# read back, so a resume would restore nothing. The trainer refuses that combination outright.
resume_mode=${resume_mode:-disable}

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO is-pg k-${staleness_threshold} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz}x${num_minibatches_per_update} ppo-epochs-${ppo_epochs} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.use_policy_gradient=${use_policy_gradient} \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
    "+actor_rollout_ref.actor.policy_loss.rollout_correction={rollout_is:${rollout_is},rollout_is_threshold:${rollout_is_threshold},rollout_rs:${rollout_rs},rollout_rs_threshold:${rollout_rs_threshold}}" \
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
    async_training.pause_generation_during_save="${pause_generation_during_save}" "$@"
