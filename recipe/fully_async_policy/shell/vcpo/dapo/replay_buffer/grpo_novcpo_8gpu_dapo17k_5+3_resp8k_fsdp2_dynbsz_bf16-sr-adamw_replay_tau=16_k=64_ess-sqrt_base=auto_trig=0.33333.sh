#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-fsdp2

# bf16 + torchao stochastic-rounding AdamW variant of the FSDP2 dynbsz
# triggered ESS-braked replay arm
# (grpo_novcpo_..._fsdp2_dynbsz_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh),
# swapping in the trainer precision recipe of the bf16-sr-adamw B33x1 script:
#   * fsdp_config.model_dtype=bf16 + torchao _AdamW with bf16 stochastic
#     rounding, fully GPU-resident (no CPU offload, no fp32 master). Sharded
#     over dp=3: ~5.5 (bf16 params) + 5.5 (grads) + 10.9 (bf16 moments)
#     ~= 22 GB static per trainer GPU — HALF the fp32 arm's ~44 GB, so the
#     20480-token dynbsz budget drops from ~65-70 GiB estimated peak to
#     ~45-50 GiB: the fp32 dynbsz arm's OOM caveat does not apply here.
#   * model_dtype=bf16 is REQUIRED for this: verl builds the FSDP actor in
#     fp32 by default, which doubles static memory AND makes
#     bf16_stochastic_round silently inert (SR only acts on bf16 params).
#     Stochastic rounding makes master-less bf16 updates unbiased — at
#     lr=1e-6 a deterministic bf16 update rounds to zero for ~98% of weights.
#   * NUMERICS: this arm gives up the base FSDP2 arm's "fp32 masters, higher
#     optimizer fidelity than HDO" property. Its bf16-master character is
#     CLOSER to the winning Megatron HDO trigger-arm
#     (main_params_dtype=bfloat16 there), but the mechanisms differ — SR with
#     bf16 moments here vs deterministic rounding with fp32 moments there.
#     Treat all cross-recipe comparisons as system-level, not ablations.
#     Checkpoints are NOT interchangeable with the fp32 FSDP2 arms.
#   * The ESS brake composes unchanged: _ess_scaled_optimizer_step scales the
#     optimizer param-group LRs, which torchao _AdamW honors like any torch
#     optimizer. Requires torchao in the environment.
# Everything else is identical to the fp32 dynbsz base: trainer-side replay
# buffer (tau=16, eviction k=64, rmb=1, sync after every update, DAPO
# insertion gate, frozen advantages / behavior log-probs), token-IS 2.0
# against cached behavior log-probs, ESS brake sqrt/base=auto/trigger=1/3
# attached to the ordinary mini-batch update (dp_actor port),
# seq_adv_post_scale=True for Megatron per-traj loss parity,
# use_dynamic_bsz=True at the 20480-token budget (parity weighting exact,
# per-sequence ESS sums packing-independent), B=33 prompts x 16 responses,
# lr 1e-6 constant, 8K responses, two validation sets, stop-the-world
# validation/saves.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0
# Unbuffered worker stdout: Ray block-buffers prints otherwise, lagging the
# live log by minutes exactly when print volume is lowest (startup/stalls)
export PYTHONUNBUFFERED=1
# NOTE: do NOT export PYTORCH_CUDA_ALLOC_CONF=expandable_segments here —
# vLLM's sleep-mode allocator asserts against it; the trainer-only
# DetachActorWorker enables it in-process (recipe fsdp_workers.py).

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# Two validation sets, reported separately by data_source:
#   aime-2024.parquet (data_source=math_dapo) -> val-core/math_dapo/acc/mean@1
#   math500.parquet   (data_source=math500_dapo) -> val-core/math500_dapo/acc/mean@1
# Built by examples/data_preprocess/math500.py with the SAME prompt template
# and "Answer:"-line scorer (math_dapo) as the training set, so validation
# measures math ability rather than answer-format transfer.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/math500.parquet']"}

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
gpu_memory_utilization=0.9
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=${max_prompt_length:-2048}
max_response_length=${max_response_length:-8192}
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= FSDP2 Trainer =================
fsdp_size=${fsdp_size:-${n_gpus_training}} # full sharding over the trainer GPUs
sp_size=1                                  # no ulysses sequence parallel
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33} # 33*16=528 seqs; mini*n must divide by trainer DP=3 (528/3=176)
micro_bsz_per_gpu=1 # ignored under use_dynamic_bsz=True
use_dynamic_bsz=True
# token budget per micro-batch: 2x max_model_len. At the bf16 recipe's ~22 GB
# static the estimated peak is ~45-50 GiB — comfortable; kept at 2x (not
# raised) so the only change vs the fp32 dynbsz arm is the precision recipe.
ppo_max_token_len=${ppo_max_token_len:-$((2 * (max_prompt_length + max_response_length)))} # 20480
log_prob_micro_bsz_per_gpu=1

bsz_per_dp_rank=${bsz_per_dp_rank:-${train_prompt_mini_bsz}} # Rollout Bsz

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
calculate_entropy=True # log actor/entropy even with entropy_coeff=0
grad_clip=1.0

# ================= Optimizer =================
lr=1e-6
lr_warmup_steps=0
weight_decay=0.1

# ================= ESS-guided LR scaling (VCPO, dp_actor port) =================
# The ESS brake lives in the standard mini-batch update; seq_adv_post_scale
# reproduces the Megatron arm's per-traj loss semantics (see header).
seq_adv_post_scale=True
ess_enable=${ess_enable:-True}
ess_rule=${ess_rule:-sqrt}  # sqrt | linear
# rho_on reference; null = auto-calibrate from the first update's measured
# ESS (fresh runs only), or set explicitly (1.0 = paper value for math)
ess_base=${ess_base:-null}
ess_use_clipped=False # ESS from unclipped ratios (paper): the brake must see what truncation hides
# Intervention threshold on ess_ratio/base: scaling engages only for
# mini-batches where the ratio falls BELOW this value; at or above it the
# update runs at full nominal lr. Same trigger geometry as the winning
# Megatron arm (base/3 deadband on the auto-calibrated reference).
ess_trigger=${ess_trigger:-0.33333}
ess_base_tag=${ess_base}
[ "${ess_base_tag}" = "null" ] && ess_base_tag="auto"
[ "${ess_trigger}" != "null" ] && ess_base_tag="${ess_base_tag}-trig-${ess_trigger}"

# ================= IS / Rollout Correction =================
# Token-level truncated IS with PPO-clip loss against the *cached* behavior
# log-probs (frozen at insertion). use_policy_gradient=False is the
# equivalent of loss_type=ppo_clip.
rollout_is="token"
rollout_is_threshold="2.0"
rollout_rs=null
rollout_rs_threshold=null
bypass_mode=False
use_policy_gradient=False
# Log training/rollout_actor_probs_pearson_corr (exp of policy vs rollout
# log-probs over response tokens) from the deferred correction path
log_probs_pearson_corr=${log_probs_pearson_corr:-True}

skip_recompute_old_log_prob=True # REQUIRED by replay mode (cached behavior log-probs)
compute_prox_log_prob=False

# ================= Async Training =================
staleness_threshold=${staleness_threshold:-64.0}
updates_per_param_sync=1     # REQUIRED by replay mode: sync after every update
num_minibatches_per_update=1 # REQUIRED by replay mode: one mini-batch per update
partial_rollout=True
use_rollout_log_probs=True

# ================= Replay buffer =================
replay_enable=${replay_enable:-True}
replay_tau=${replay_tau:-16}
replay_staleness_threshold=${replay_staleness_threshold:-64}
replay_requires_mini_batches=${replay_requires_mini_batches:-1}
replay_sampling_seed=${replay_sampling_seed:-1234}

# ================= Elastic mechanisms OFF / stop-the-world accounting =================
dynamic_filtering_enable=False
min_buffered_batches=1.0
opportunistic_enable=False
opportunistic_max_extra_epochs=0
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

# ================= Training/Rollout Steps =================
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
test_freq=${test_freq:-20}
save_freq=${save_freq:-20}
max_actor_ckpt_to_keep=1 # keep only the most recent checkpoint

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches} ess-${ess_rule}-base-${ess_base_tag} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} fsdp2-noofl dynbsz sr-adamw B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
exp_name_safe=${exp_name//\//_}
log_dir="logs/${exp_name_safe}"
CKPTS_DIR="${log_dir}"
mkdir -p -- "${log_dir}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

trainer_logger="['console','tensorboard']"
log_val_generations=0
val_before_train=${val_before_train:-True}

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
    algorithm.rollout_correction.log_probs_pearson_corr=${log_probs_pearson_corr} \
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.seq_adv_post_scale=${seq_adv_post_scale} \
    actor_rollout_ref.actor.ess_scaling.enable=${ess_enable} \
    actor_rollout_ref.actor.ess_scaling.scaling_rule=${ess_rule} \
    actor_rollout_ref.actor.ess_scaling.base_ess_ratio=${ess_base} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_use_clipped} \
    actor_rollout_ref.actor.ess_scaling.trigger_ratio=${ess_trigger} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip} \
    actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim \
    actor_rollout_ref.actor.optim.optimizer=_AdamW \
    "actor_rollout_ref.actor.optim.override_optimizer_config={bf16_stochastic_round:true}" \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
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
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    trainer.logger=${trainer_logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
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
    async_training.skip_recompute_old_log_prob="${skip_recompute_old_log_prob}" \
    async_training.dynamic_filtering.enable="${dynamic_filtering_enable}" \
    async_training.dynamic_filtering.min_buffered_batches="${min_buffered_batches}" \
    async_training.opportunistic_epochs.enable="${opportunistic_enable}" \
    async_training.opportunistic_epochs.max_extra_epochs="${opportunistic_max_extra_epochs}" \
    async_training.ppo_epochs=null \
    async_training.serialize_validation="${serialize_validation}" \
    async_training.pause_generation_during_save="${pause_generation_during_save}" \
    async_training.replay_buffer.enable="${replay_enable}" \
    async_training.replay_buffer.tau="${replay_tau}" \
    async_training.replay_buffer.staleness_threshold="${replay_staleness_threshold}" \
    async_training.replay_buffer.requires_mini_batches="${replay_requires_mini_batches}" \
    async_training.replay_buffer.sampling_seed="${replay_sampling_seed}" \
    +async_training.bsz_per_dp_rank="${bsz_per_dp_rank}" "$@"
