#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo

# grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_ppo-epochs=2_B33x1_is-pg.sh
#
# FSDP2 twin of ..._megatron_offload_ppo-epochs=2_B33x1_is-pg.sh: same data, same
# objective, same schedule, same 5+3 GPU split - only the training backend differs,
# so the two can be compared step for step. Everything below that is not about the
# backend is copied verbatim from that script; see BACKEND MAPPING at the end of
# this header for what changed and why.
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
#   * staleness_threshold=1 (see the k=1 note below; the source arm ran k=2),
#     total_rollout_steps=66000 explicit, test_freq=save_freq=10.
#   * serialize_validation / pause_generation_during_save: stop-the-world validation
#     and saves, excluded from fully_async/timing/cumulative_training_time.
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
#   * each save writes global_step_N/actor/huggingface/ - config, tokenizer and bf16
#     safetensors - directly loadable by vLLM / from_pretrained, no merge step. No
#     optimizer state, no sharded dist_ckpt/ directory at all.
#   * nothing is rotated away: ~33 GB per save for Qwen3-8B (fp32, see the accepted
#     divergences below), ~200 saves over the full run is ~6.6 TB. Raise save_freq at
#     launch if the disk is tighter.
#   * the run is NOT resumable: 'hf_model' is written but never read back, so
#     load_contents would restore nothing. resume_mode=disable makes that explicit,
#     and the trainer raises rather than resuming from such a checkpoint.
#
# ACCEPTED DIVERGENCES from the 'rollout-dapo' script (dropped, not emulated):
#   * math500 validation set -> the math500_dapo scorer is not in this branch's
#     reward registry, so validation is AIME-2024 only.
#   * +async_training.bsz_per_dp_rank=33 -> not ported. At k=1 this no longer costs
#     anything: max_concurrent_samples = min(5 servers * 16, 66) = 66, i.e. the full
#     staleness budget. (At k=2 the 80-sample server cap bound it below the 99 allowed.)
#   * algorithm.rollout_correction.log_probs_pearson_corr -> not ported; the same
#     policy-vs-rollout pair is already covered by rollout_corr/* (KL, ESS).
#   * async_training.{dynamic_filtering,opportunistic_epochs} -> both were OFF in the
#     source arm, so dropping them changes nothing.
#
# BACKEND MAPPING (megatron arm -> here). No production code change was needed:
#   * --config-name is dropped: fully_async_main.py defaults to
#     config/fully_async_ppo_trainer.yaml, the FSDP entry config.
#   * strategy=fsdp2 for actor and critic; the recipe's own FSDP worker implements the
#     trainer->rollouter transfer (recipe/fully_async_policy/fsdp_workers.py).
#   * megatron tp/pp/cp/sequence_parallel -> pure DP over the 3 trainer GPUs
#     (fsdp_size=-1, ulysses_sequence_parallel_size=1).
#   * recompute_granularity=full/uniform/1 -> model.enable_gradient_checkpointing=True.
#   * the HDO block (optimizer_cpu_offload + main_params_dtype=bfloat16) is dropped:
#     torch AdamW with fp32 master states, sharded by FSDP.
#   * optim.clip_grad -> actor.grad_clip; lr_decay_style=constant is already the
#     FSDPOptimizerConfig default (lr_scheduler_type=constant).
#   * B-33x1 and ppo_epochs=2 mean exactly what they mean on megatron:
#     fsdp_workers.py also scales ppo_mini_batch_size by rollout.n (33*16=528 = the
#     whole pull), so make_iterator(epochs=2) is 2 AdamW updates per trainer step.
#   * calculate_entropy=True is honoured natively by the FSDP actor
#     (dp_actor.py: calculate_entropy = config.calculate_entropy or entropy_coeff != 0).
#
# MEMORY, 8.19B params on 3xH100-80GB with fsdp_size=-1 (fp32 params, bf16 compute):
# params 32.8/3 ~ 10.9 GB + grads ~ 10.9 GB + AdamW moments 65.5/3 ~ 21.8 GB ~ 44 GB
# steady before activations. entropy chunking + checkpointing keep the entropy
# computation (vocab 151k x up to 10k tokens, inside the update) off the peak. If the
# trainer still OOMs: fsdp_config.optimizer_offload=True frees ~22 GB, then
# param_offload=True, then model.use_fused_kernels=True.
#
# This is the EXACT-NUMERICS FSDP2 arm, mirroring the 5+3 fsdp2 script of branch
# rollout-dapo: verl's default fp32 params (= fp32 master) + plain torch AdamW with
# fp32 states, no CPU offload, no stochastic rounding. The 6+2 twin
# (..._6+2_resp8k_fsdp2_dynbsz_sr-adamw_...) trades that for bf16 params + torchao
# _AdamW with stochastic rounding and dynamic batching; use this one when the
# comparison against the megatron arm has to be numerical rather than system-level.
#
# ACCEPTED DIVERGENCES FROM THE MEGATRON ARM (same objective, different numerics and
# artifacts):
#   * optimizer master params are fp32 here vs main_params_dtype=bfloat16 under HDO.
#   * each hf_model checkpoint is fp32, ~33 GB, DOUBLE the megatron arm's bf16 16.4 GB:
#     get_fsdp_full_state_dict does not cast (verl/utils/fsdp_utils.py) and
#     save_pretrained writes the state dict as it is. Budget save_freq accordingly.

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
# AIME-2024 only (data_source=math_dapo) -> val-core/math_dapo/acc/mean@1.
# The rollout-dapo arm also validated on math500.parquet; that set needs the
# math500_dapo scorer, which this branch does not carry.
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
# 0.8, not 0.9: on the FSDP worker path the vLLM engine process sits ~7-10 GB
# above the configured pool target, and 0.9 OOMs at sampler warmup / weight sync
# (fork smoke campaign, 2026-07-30). The megatron path runs ~4 GB over.
gpu_memory_utilization=0.8
# vLLM v1 warms the sampler with max_num_seqs dummy requests AFTER filling the KV
# pool, so the transient scales with it; 512 is far above the ~50-60 concurrent
# sequences these engines actually run at this length.
max_num_seqs=512
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
max_response_length=${max_response_length:-8192}
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= FSDP2 Parallelism =================
# Pure data parallelism over the 3 trainer GPUs: FSDP2 shards params, grads and
# optimizer state across them (fsdp_size=-1 = the whole trainer group).
fsdp_size=-1
sp_size=1 # ulysses sequence parallelism
reshard_after_forward=True
offload_policy=False   # FSDP2 CPUOffloadPolicy
param_offload=False
optimizer_offload=False # first fallback if the trainer OOMs (frees ~22 GB/GPU)
ref_param_offload=True
enable_gradient_checkpointing=True # the megatron arm's recompute_granularity=full
# Both are needed on this path: non-chunked entropy materializes fp32
# logits-sized intermediates (vocab 151936) inside the training backward.
entropy_from_logits_with_chunking=True
entropy_checkpointing=True
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33} # 33*16=528 seqs; must divide by trainer DP=3 (528/3=176)
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
entropy_coeff=${entropy_coeff:-0}
# Log actor/entropy even with entropy_coeff=0 (honoured via should_calculate_entropy
# in verl/workers/actor/megatron_actor.py). This arm has no old-log-prob forward, so
# it is the only source of entropy here -- see the memory note in the header.
calculate_entropy=True
grad_clip=1.0

# ================= Optimizer =================
lr=${lr:-1e-6}
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
# k=1: the rollouter is licensed to generate up to (1+1) trainer batches ahead
# — at B-33x1 that is 66 in-flight/queued groups (max_required_samples =
# 33 * (k+1) * trigger_parameter_sync_step). Tighter than the VCPO baseline's k=2:
# samples are at most one parameter version stale when the trainer consumes them.
staleness_threshold=${staleness_threshold:-1.0}
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
# ~33 GB per save for Qwen3-8B (fp32 on FSDP), nothing is rotated away: ~200 saves is ~6.6 TB,
# so raise save_freq at launch if the disk is tighter than that.
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
# Mandatory, not cosmetic: load_contents mirrors save_contents, and 'hf_model' is written but never
# read back, so a resume would restore nothing. The trainer refuses that combination outright.
resume_mode=${resume_mode:-disable}

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO is-pg k-${staleness_threshold} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} fsdp2 B-${train_prompt_mini_bsz}x${num_minibatches_per_update} ppo-epochs-${ppo_epochs} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
    "+actor_rollout_ref.actor.policy_loss.rollout_correction={rollout_is:${rollout_is},rollout_is_threshold:${rollout_is_threshold},rollout_rs:${rollout_rs},rollout_rs_threshold:${rollout_rs_threshold}}" \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${offload_policy} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${reshard_after_forward} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=${entropy_from_logits_with_chunking} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${lr_decay_style} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
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
