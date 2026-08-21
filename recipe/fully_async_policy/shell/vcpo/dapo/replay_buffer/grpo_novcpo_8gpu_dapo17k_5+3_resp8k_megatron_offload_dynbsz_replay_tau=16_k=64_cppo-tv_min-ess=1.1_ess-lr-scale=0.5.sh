#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess

# CPPO-MASKED dynamic-batch-size Megatron replay arm: the unmasked dynbsz
# min-ess arm collapsed at step ~214 via reuse-driven uniform pi-mu drift
# (each fresh group trained 1.9-2.7x on average; with the PPO ratio anchored
# at 1 nothing bounded repeated passes against a group's frozen behavior
# policy — see INSTABILITY_DYNBSZ_MIN-ESS_VS_MBS1_TRIG_DISCUSSION.md). This
# arm adds the CPPO prefix-budgeted drift mask (paper/cppo.pdf, Binary-TV
# variant, ported from the authors' verl fork) as the ONE deliberate change:
#   * loss_mode=cppo: token gradients that push pi AWAY from the cached mu
#     are kept only while the position-weighted divergence w_t*D_t <= delta
#     (delta = clip_ratio = 0.15, D_t = |pi(y_t)-mu(y_t)|) AND the running
#     prefix budget S_{t-1} <= delta + delta_b*W_{t-1} holds; toward-mu
#     gradients always pass. Over-optimized replayed groups therefore
#     self-terminate instead of drifting into the KL/entropy runaway.
#   * The loss's old_log_prob is the CACHED behavior log-probs (mu), so the
#     mask measures true cumulative drift under reuse; the loss's truncated
#     pi/mu ratio (cap clip_ratio_c=20) replaces the former token-IS weights.
#   * delta_b is calibrated per sequence: clamp(k * P90(D_t), 0.02, 0.1).
#   * Health metrics: actor/pg_clipfrac = masked-token fraction (expected to
#     rise on drifted replayed groups — the new early-warning signal),
#     actor/cppo_toward_mu_frac = corrective-pressure fraction.
# The min-ESS lr brake is KEPT (orthogonal: token-level in-loss mask vs
# step-level lr; the brake covers the dominated-update ESS-collapse mode the
# mask does not). Brake: lr * ess_lr_scale when global ESS <= min_ess (1.1).
# Dynbsz mechanics (unchanged from the former dynbsz arm):
#   * actor.use_dynamic_bsz=True + ppo_max_token_len_per_gpu=15360: the
#     buffer-free per-traj update dispatches to the PACKED path
#     (_update_policy_per_traj_packed) — sequences packed into token-budget
#     micro-batches instead of 176 single-sequence micro-batches per DP rank.
#     Gradient parity with the mbs=1 path is EXACT (skip_recompute anchors
#     the PPO ratio at 1, and the n_rows*M/N rescale makes the loss the
#     global per-sequence mean, invariant to packing); the ESS brake consumes
#     per-sequence log-IS sums via the max-shifted (log-space) ESS — no
#     fp32-exp censoring, ESS floored at 1, so the brake multiplier is
#     exactly ess_lr_scale on degenerate mini-batches, never 0. The mbs=1
#     arm now reduces the same way (both go through
#     verl/workers/utils/ess.py), so all three arms' ESS traces are directly
#     comparable. Per-traj grad-norm diagnostics and per-token
#     record lists are mbs=1-only and stay empty here. OPOB remains
#     incompatible with dynbsz. Token budget 15360 = 1.5x max seq len
#     (measured envelope on this arm, tp1/dp3 HDO 8B: 20480 OOMs at
#     update 1 — NCCL calloc failure, ~83 GB predicted; 15360 fits at
#     63.4 GB allocated).
# Inherited replay-arm mechanics:
#   * update_policy_per_traj=True: every mini-batch's sequence-level IS
#     ratios against the cached behavior log-probs are DP-all-reduced into
#     ess_ratio = (sum w)^2 / (B * sum w^2), logged as staleness/ess_ratio
#     (ESS in effective samples = ess_ratio * B, B = 528 here).
#   * ess_scaling (min-ESS rule): the optimizer step's LR is multiplied by
#     the CONSTANT ess_lr_scale for that step only when global ESS <=
#     min_ess; the effective lr therefore takes exactly two values,
#     {lr, ess_lr_scale * lr}, logged as replay/ess_scaled_lr. No measured
#     reference, no base capture, nothing persisted in replay_buffer.pt.
#   * Costs vs the unbraked arm: slower updates from the per-traj path's
#     micro-batch-size-1 scheduling (the earlier ~20% figure included
#     per-traj buffer accumulation + grad norms, both gone now). With
#     grad_baselining.enable=False (set below) the per-traj path is
#     BUFFER-FREE since 2026-08-15: no extra grad-sized GPU buffer — the
#     former ~15.3 GB bf16 per trainer GPU on top of the ~58 GB HDO
#     footprint is reclaimed.
#   * The effective LR is logged as replay/ess_scaled_lr (and
#     actor/ess_scaled_lr + staleness/ess_ratio via structured metrics)
#     every update.
# Replay-arm notes that still apply:
#   * Groups staler than replay_buffer.staleness_threshold=64 updates are
#     evicted after each update; scores are recomputed each update. With
#     tau=16 a staleness-64 group still carries sampling weight 2^-4 = 1/16 —
#     deep replay is intended. The buffer retains every kept group of the
#     last 64 updates (~1000-1600 groups, roughly 7-12 GB driver RAM and the
#     same for replay_buffer.pt in checkpoints).
#   * Warm-up/watermark: requires_mini_batches=1 — the first update consumes
#     a fresh mini-batch of unseen groups; afterwards training pauses only
#     while the buffer holds < 1*33 = 33 groups.
#   * async_training.staleness_threshold=64 aligns the rollouter's generation
#     quota with the eviction horizon (33*(64+1)=2145 groups licensed; in
#     practice a stall backstop — concurrency caps at 165 in-flight).
#   * Model versions tick once per UPDATE: test/save freq are in update units.
#   * serialize_validation=True / pause_generation_during_save=True kept:
#     stop-the-world validation and checkpoint saves — pure time translations
#     excluded from cumulative_training_time.
# Base-script notes that still apply: trainer tp=1/dp=3 (sequence_parallel
# needs TP>1), 33*16=528 seqs divide by DP=3, HDO full CPU offload with bf16
# master weights (do NOT swap for use_precision_aware_optimizer without
# optimizer_cpu_offload: silent stall, probe 2026-07-30). OPOB stays off.

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

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# Two validation sets, reported separately by data_source:
#   aime-2024.parquet (data_source=math_dapo) -> val-core/math_dapo/acc/mean@1
#   aime-2025.parquet (data_source=aime2025_dapo) -> val-core/aime2025_dapo/acc/mean@1
# aime-2025 is built from MathArena/aime_2025 in the exact aime-2024 format
# (30 problems x 32 copies, same DAPO prompt template and "Answer:"-line
# scorer via the aime* dispatch), so both metrics measure the same objective;
# the distinct data_source stamp keeps the 2025 curve separate.
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
gpu_memory_utilization=0.9
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=${max_prompt_length:-2048}
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
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33} # 33*16=528 seqs; mini*n must divide by trainer DP=3 (528/3=176)
micro_bsz_per_gpu=1 # ignored by the packed path under use_dynamic_bsz=True
use_dynamic_bsz=True
# token budget per packed micro-batch (per GPU); 15360 = 1.5x max response
# len — the measured fit on this arm (20480 OOMs at update 1, see header).
ppo_max_token_len=${ppo_max_token_len:-15360}
log_prob_micro_bsz_per_gpu=1

bsz_per_dp_rank=${bsz_per_dp_rank:-${train_prompt_mini_bsz}} # Rollout Bsz

# ================= Algorithm =================
adv_estimator=grpo
loss_agg_mode="seq-mean-token-mean"
# Under loss_mode=cppo, clip_ratio is REPURPOSED as the token-level DIVERGENCE
# threshold delta of the CPPO mask (Binary-TV units, |pi(y_t)-mu(y_t)|; paper
# default 0.15 for dense models, 0.20 for MoE). It is not a PPO ratio clip here.
clip_ratio=${cppo_delta:-0.15}
# Unused by the cppo loss (no PPO clip branches); kept aligned with clip_ratio
clip_ratio_low=${cppo_delta:-0.15}
clip_ratio_high=${cppo_delta:-0.15}
# Truncated-IS cap on the detached pi/mu ratio weight inside the cppo loss
# (replaces the former rollout_is token cap in the loss; reference default 20)
clip_ratio_c=${clip_ratio_c:-20.0}
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

# ================= ESS-guided LR scaling (VCPO) =================
update_policy_per_traj=True
# OPOB off -> the per-traj path runs BUFFER-FREE: no grad accum buffers are
# allocated (the advantage is folded into the per-microbatch loss scale and
# gradients accumulate in Megatron's main buffer), saving 15.26 GiB bf16 of
# peak memory per trainer GPU. Set explicitly so a future default flip
# cannot silently re-enable the buffers. Per-traj grad-norm diagnostics
# (traj_record.grad_norm) are OPOB-only and stay empty in this mode.
grad_baselining=False
ess_enable=${ess_enable:-True}
# Min-ESS rule (replaces the auto-captured-base + trigger + sqrt logic): a
# mini-batch whose global ESS carries <= min_ess effective samples steps at
# lr * ess_lr_scale; above it the update runs at full nominal lr. Equivalent
# to ess_ratio <= min_ess/B (B = 528 here). The log-space ESS floors ESS at
# exactly 1, so degenerate (single-dominant-sequence) mini-batches always
# brake — at ess_lr_scale, never 0. No measured reference, no base capture:
# the threshold is backend-independent, unlike the auto-base (11x apart
# between the fsdp2 and Megatron arms for near-identical raw ESS traces).
min_ess=${min_ess:-1.1}
ess_lr_scale=${ess_lr_scale:-0.5}
ess_use_clipped=False # ESS from unclipped ratios (paper): the brake must see what truncation hides
ess_tag="min-ess-${min_ess}-lrscale-${ess_lr_scale}"

# ================= CPPO drift mask (paper/cppo.pdf, Binary-TV) =================
# loss_mode=cppo swaps the loss for the prefix-budgeted drift-masked truncated-IS
# REINFORCE surrogate: a token's away-from-mu gradient is kept only while its
# position-weighted divergence w_t*D_t <= delta AND the cumulative prefix budget
# S_{t-1} <= delta + delta_b*W_{t-1} holds (toward-mu gradients always pass).
# D_t = |pi(y_t)-mu(y_t)| against the CACHED behavior log-probs — in loss_func
# the cppo loss receives mu as old_log_prob (the ratio anchor does not apply),
# so over-optimized replayed groups self-terminate: exactly the reuse-drift
# runaway that collapsed the unmasked dynbsz arm at step ~214 (see
# INSTABILITY_DYNBSZ_MIN-ESS_VS_MBS1_TRIG_DISCUSSION.md). Watch
# actor/pg_clipfrac (masked fraction) and actor/cppo_toward_mu_frac.
loss_mode=cppo
cppo_w_min=${cppo_w_min:-0.8}       # position-weight floor, w_t in [w_min, 1]
cppo_delta_b=${cppo_delta_b:-0.02}  # prefix-average budget floor delta_b_min
cppo_delta_b_q=${cppo_delta_b_q:-0.9}  # per-seq calibration quantile (paper P90)
cppo_delta_b_k=${cppo_delta_b_k:-1.0}  # scale on that quantile
cppo_tag="cppo-tv-d${cppo_delta:-0.15}-db${cppo_delta_b}"

# ================= IS / Rollout Correction =================
# rollout_correction stays configured for its METRICS (rollout_corr/kl,
# token-IS stats, pearson) and the ESS brake inputs; under loss_mode=cppo the
# token-IS weights are NOT applied to the loss (the cppo loss's own truncated
# pi/mu ratio, capped at clip_ratio_c, takes that role).
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
# Generation quota aligned with the replay eviction horizon: groups older
# than replay_staleness_threshold updates are deleted anyway, so licensing
# generation beyond it would only produce evicted-unseen waste.
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
replay_save_state=False # no replay_buffer.pt in checkpoints: resume is disabled

# ================= Elastic mechanisms OFF / stop-the-world accounting =================
# Replay mode subsumes DAPO filtering (insertion gate always on) and replaces
# opportunistic/fractional epochs with score-weighted replay.
dynamic_filtering_enable=False
min_buffered_batches=1.0
opportunistic_enable=False
opportunistic_max_extra_epochs=0
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}
save_queue_state=False # no queue snapshots in checkpoints: resume is disabled

# ================= Training/Rollout Steps =================
# Same 66000-prompt generation budget as the B-33x4 arms (500 steps * 132
# groups). Fed prompts, not kept groups: filtering shortens the effective
# trained horizon proportionally.
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
# Model versions now tick once per UPDATE (not per 132-group step): validate /
# checkpoint every 20 updates (=660 groups consumed, matching the 5-step
# cadence of the B-33x4 arms in group units).
test_freq=${test_freq:-20}
# Model checkpointing is OFF: save_freq<=0 disables _check_save_checkpoint's
# save gate entirely (fully_async_trainer.py), so no global_step_N/ directory
# — not even an hf_model — is ever written; zero checkpoint disk footprint.
# resume_mode=disable is kept as a safety net: with no checkpoints of its own
# to resume from, this only matters if a prior run left one under the same
# exp_name/log_dir, which would otherwise be picked up by resume_mode=auto.
# replay_buffer.save_state / save_queue_state are moot with saving off
# (nothing ever calls the code path they gate) but left False for when
# save_freq is overridden back on. Re-enable saving with save_freq=N>0 and
# set ckpt_save_contents/max_actor_ckpt_to_keep as needed.
save_freq=${save_freq:-20}
max_actor_ckpt_to_keep=null
ckpt_save_contents="['hf_model']"
resume_mode=disable

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches} ${cppo_tag} ess-${ess_tag} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo dynbsz B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    algorithm.rollout_correction.log_probs_pearson_corr=${log_probs_pearson_corr} \
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.policy_loss.cppo.cppo_w_min=${cppo_w_min} \
    actor_rollout_ref.actor.policy_loss.cppo.cppo_delta_b=${cppo_delta_b} \
    actor_rollout_ref.actor.policy_loss.cppo.cppo_delta_b_q=${cppo_delta_b_q} \
    actor_rollout_ref.actor.policy_loss.cppo.cppo_delta_b_k=${cppo_delta_b_k} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.update_policy_per_traj=${update_policy_per_traj} \
    actor_rollout_ref.actor.grad_baselining.enable=${grad_baselining} \
    actor_rollout_ref.actor.ess_scaling.enable=${ess_enable} \
    actor_rollout_ref.actor.ess_scaling.min_ess=${min_ess} \
    actor_rollout_ref.actor.ess_scaling.lr_scale=${ess_lr_scale} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_use_clipped} \
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
    "actor_rollout_ref.actor.checkpoint.save_contents=${ckpt_save_contents}" \
    trainer.resume_mode=${resume_mode} \
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
    async_training.save_queue_state="${save_queue_state}" \
    async_training.replay_buffer.enable="${replay_enable}" \
    async_training.replay_buffer.tau="${replay_tau}" \
    async_training.replay_buffer.staleness_threshold="${replay_staleness_threshold}" \
    async_training.replay_buffer.requires_mini_batches="${replay_requires_mini_batches}" \
    async_training.replay_buffer.sampling_seed="${replay_sampling_seed}" \
    async_training.replay_buffer.save_state="${replay_save_state}" \
    +async_training.bsz_per_dp_rank="${bsz_per_dp_rank}" "$@"
