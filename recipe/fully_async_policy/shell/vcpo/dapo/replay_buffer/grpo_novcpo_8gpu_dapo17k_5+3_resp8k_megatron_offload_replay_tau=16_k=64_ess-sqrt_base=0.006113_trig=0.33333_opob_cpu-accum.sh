#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-opob-cpuaccum-bf16m

# OPOB (VCPO optimal off-policy baseline) replay arm, 5+3 layout, the twin's
# optimizer (HDO + bf16 master weights), host-resident OPOB accumulators:
# identical to
# grpo_novcpo_..._5+3_..._replay_tau=16_k=64_ess-sqrt_base=0.006113_trig=0.33333.sh
# (trainer-side replay buffer, tau=16, eviction k=64, rmb=1, sync after
# every update, DAPO insertion gate, frozen advantages / behavior log-probs,
# token TIS 2.0, fixed-base sqrt ESS brake with the trigger_ratio=0.33333
# deadband, 5 rollout + 3 trainer GPUs, B=33x16, HDO with
# main_params_dtype=bfloat16) except for OPOB and where its accumulators live:
#   * grad_baselining.enable=True — VCPO's OPOB (arXiv:2602.17616 §3.3,
#     Eq. 7/13). Per prompt-group the scalar baseline
#         b* = sum_i w_i^2 ||g_i||^2 R_i / sum_i w_i^2 ||g_i||^2
#     (w_i = UNCLIPPED sequence IS ratio pi/mu against the cached behavior
#     log-probs, g_i = the trajectory's score gradient, R_i = raw reward)
#     replaces GRPO's group-mean baseline. It is the variance-minimizing
#     baseline of the IS-weighted policy gradient: trajectories that are both
#     strongly up-weighted off-policy AND move the parameters most dominate
#     it, so the effective advantage R_i - b* of each group's dominant
#     up-drifted trajectory goes to ~0 — exactly the runaway quadrant
#     (A>0, pi/mu >> 1) that token-TIS caps per step but never in cumulative
#     movement. Paper Fig. 9: TIS+OPOB stays stable where TIS alone
#     collapses. norm_by_std=True divides (R_i - b*) by the group reward std
#     so the arm keeps GRPO's advantage scale (lr-comparable with the twin);
#     scope=group, agg_mode=mean, use_is_weights=True,
#     use_clipped_is_ratios=False, normalize_by_length=False are the paper
#     defaults. OPOB is exact in this loss: skip_recompute_old_log_prob=True
#     makes the PPO ratio == 1 (clip inert), so the per-trajectory loss is
#     linear in the advantage and the baseline is applied after the single
#     backward (Algorithm 1: G_R = sum w_i R_i g_i, G_S = sum w_i g_i,
#     G = G_R - b* G_S). Incompatible with the mu-anchored clip blend
#     (asserted in code); this arm has no blend.
#   * Memory: single-backward OPOB holds THREE grad-buffer copies (Megatron's
#     main buffer for the isolated g_i + accum G_R + score G_S). On the twin's
#     5+3 / TP=1 layout two extra 15.26 GiB GPU buffers do not fit (buffer-free
#     ~58 GB + 30.5 -> ~88 GB). With the twin's optimizer (HDO: Adam moments
#     on the CPU, bf16 master shard 3.8 GiB on the GPU, no fp32 gradient copy
#     thanks to use_precision_aware_optimizer) the GPU already holds only what
#     it must, so the ONE thing to offload is OPOB's two accumulators:
#       - grad_baselining.accum_device=cpu keeps G_R and G_S in pinned host
#         memory, in fp32 (accum_dtype=float32; 2 x 30.5 GiB per trainer
#         rank). Each trajectory's gradient (15.3 GiB bf16) is copied d2h ONCE
#         into a pinned staging buffer and added into both accumulators on the
#         CPU (accum_cpu_threads torch threads); the -b* G_S move at the group
#         close and the copy of the final G_R - b* G_S back into Megatron's
#         buffer at the step happen there too. Zero extra GPU memory for OPOB,
#         and fp32 accumulation over all 176 trajectories (the GPU variant
#         accumulates in bf16).
#       - Nothing else changes vs the twin: same HDO + bf16-master optimizer,
#         same bf16 grad buffers, same activations. (The per-trajectory
#         prepare_grads() in _compute_grad_norms is skipped when there is no
#         grad scaler; with the precision-aware optimizer it was a no-op alias
#         anyway.)
#     Estimate per trainer GPU: the buffer-free twin's ~58 GB (15.3 params +
#     15.3 main grad buffer + 3.8 bf16 master shard + ~16 activations at 10K
#     tokens with full recompute + ~6 non-torch), unchanged. Host RAM per
#     rank: 61 (accumulators) + 15.3 (staging) + ~20 (Adam moments) ~= 97 GB
#     -> ~290 GB for the 3 ranks (plus the replay buffer on the driver); the
#     remote_h100 job is capped at 768 GiB.
#     Cost: ~0.6 s d2h + two fp32 host adds per trajectory (~1-2 s with 16
#     threads) on top of ~5 s of GPU compute -> expect +30-50% update time.
#     Set grad_baselining_accum_dtype=auto to accumulate in bf16 (halves host
#     RAM, same numerics as the GPU variant).
#   * train_prompt_mini_bsz=33 (528 seqs) as in the twin: group-scope OPOB
#     needs WHOLE groups on each DP rank (compute_grad_info asserts it and
#     make_opportunistic_minibatch_indices splits whole groups), so the
#     prompt count must divide by DP=3 -> 11 groups / 176 seqs per rank.
#   * ess_scaling.base_ess_ratio=0.006113 kept from the twin (the value its
#     base=auto reference auto-calibrated on the first on-policy update: ESS
#     3.23 of B=528; same B here). ess_base=null re-enables auto-calibration.
#   * OPOB diagnostics every update (structured actor/opob_records -> opob/*
#     scalars): baseline_mean, baseline_abs_mean (how often b* sits at +-1),
#     weight_conc_mean (max W_i / sum W_i: how argmax-like the baseline is),
#     dominant_pos_frac (share of groups whose dominant trajectory has R=+1,
#     i.e. the runaway quadrant being neutralized), zeroed_frac
#     (share of trajectories with |R_i - b*| < 0.1), groups.
# Inherited mechanics of the fixed-base trig=0.33333 arm:
#   * ess_scaling.trigger_ratio=0.33333 — at or above ratio 1/3 of base the
#     update runs at FULL nominal lr (hard knee: the multiplier jumps from 1
#     to sqrt(ratio) at the threshold); the deadband is a RATIO of the
#     reference (base/3), not a hand-picked absolute value.
#   * update_policy_per_traj=True: every mini-batch's sequence-level IS
#     ratios against the cached behavior log-probs are DP-all-reduced into
#     ess_ratio = (sum w)^2 / (B * sum w^2), logged as staleness/ess_ratio.
#   * ess_scaling: the optimizer step's LR is scaled by
#         min(1, ess_ratio / base_ess_ratio) ^ (1/2)   (sqrt rule, unclipped)
#     for that step only. This is the brake the unbraked tau-16/k-64 run
#     lacked: in its collapse precursors (token-IS mean 24-250, pearson<0.9)
#     the ESS ratio plummets, so the LR shrinks exactly when the off-policy
#     runaway starts, turning collapses into slowdowns.
#   * base_ess_ratio=0.006113 (fixed): auto-calibration is skipped — the
#     explicit config value wins unconditionally (resolve_ess_base in
#     megatron_actor.py), the first update is braked normally, and the value
#     is logged as replay/ess_base every update. Override ess_base to change
#     it (null would re-enable auto-calibration from the first update).
#   * Costs vs the unbraked arm: slower updates from the per-traj path's
#     micro-batch-size-1 scheduling PLUS, with grad_baselining.enable=True,
#     the per-trajectory buffer accumulation and grad norms that the
#     buffer-free twin dropped, here with the accumulation done on the host
#     (d2h copy + CPU adds per trajectory, see the memory note above).
#   * The effective LR is logged as replay/ess_scaled_lr (and
#     actor/ess_scaled_lr + staleness/ess_ratio via structured metrics)
#     every update.
# Replay-arm notes that still apply:
#   * Groups staler than replay_buffer.staleness_threshold=64 updates are
#     evicted after each update; scores are recomputed each update. With
#     tau=16 a staleness-64 group still carries sampling weight 2^-4 = 1/16 —
#     deep replay is intended. The buffer retains every kept group of the
#     last 64 updates (~1000-1600 groups, roughly 7-12 GB driver RAM;
#     replay_buffer.pt is NOT saved — checkpoints are hf_model-only,
#     resume disabled, see the checkpoint section below).
#   * Warm-up/watermark: requires_mini_batches=1 — the first update consumes
#     a fresh mini-batch of unseen groups; afterwards training pauses only
#     while the buffer holds < 1*33 = 33 groups.
#   * async_training.staleness_threshold=64 aligns the rollouter's generation
#     quota with the eviction horizon (33*(64+1)=2145 groups licensed; in
#     practice a stall backstop — concurrency caps at 5*33=165 in-flight).
#   * Model versions tick once per UPDATE: test/save freq are in update units.
#   * serialize_validation=True / pause_generation_during_save=True kept:
#     stop-the-world validation and checkpoint saves — pure time translations
#     excluded from cumulative_training_time.
# Base-script notes that still apply: HDO full CPU offload with bf16 master
# weights (do NOT swap for use_precision_aware_optimizer without
# optimizer_cpu_offload: silent stall, probe 2026-07-30); bf16 grad buffers
# (grad_reduce_in_fp32=False). Layout: trainer tp=1/dp=3 (pure DP, no TP
# comm; sequence_parallel needs TP>1), 33*16=528 seqs divide by DP=3. OPOB ON
# with host-resident fp32 accumulators.

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
# 5 rollout + 3 trainer GPUs (the twin's layout): OPOB's two extra grad-buffer
# copies live on the host (grad_baselining.accum_device=cpu, see header).
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
# 33*16=528 seqs (the twin's batch). Group-scope OPOB needs whole prompt-groups per DP
# rank, so the PROMPT count (not just mini*n) must divide by trainer DP=3: 11 groups = 176 seqs per rank.
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33}
micro_bsz_per_gpu=1 # per-traj path REQUIRES micro batch size 1 and use_dynamic_bsz=False
use_dynamic_bsz=False
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

# ================= Seeds =================
# ONE master seed for every RNG this configuration actually uses. Each can
# still be overridden separately via its env var; by default all take SEED.
# (Seeds that are inert in this config are NOT pinned: actor.data_loader_seed
# — actor.shuffle=False and replay mode composes one mini-batch per update;
# async_training.ppo_epochs_shuffle_seed — ppo_epochs=null;
# opportunistic_epochs.shuffle_seed — opportunistic epochs disabled.)
SEED=${SEED:-1}
# Train-prompt order: data.shuffle=True (yaml default) draws prompts via a
# RandomSampler whose torch.Generator is seeded from data.seed. The yaml
# default is null = UNSEEDED — prompt order would differ between runs.
data_seed=${data_seed:-${SEED}}
# Megatron model/parallel RNG (init, dropout, TP rng state). ref and critic
# resolve their megatron.seed from this value via oc.select automatically.
megatron_seed=${megatron_seed:-${SEED}}
# Replay-buffer weighted group sampling (defined here; used in the replay
# section's override below)
replay_sampling_seed=${replay_sampling_seed:-${SEED}}
# NOT covered: the vLLM engine seed. RolloutConfig has no `seed` field
# (a `+actor_rollout_ref.rollout.seed=` override crashes worker init with
# ConfigKeyError), so vllm_async_server falls back to seed=0. Note vLLM
# continuous batching is not bitwise-reproducible at temperature=1.0 anyway.

# ================= ESS-guided LR scaling (VCPO) =================
update_policy_per_traj=True
# ================= OPOB (VCPO optimal off-policy baseline) =================
# grad_baselining.enable=True allocates two extra grad-buffer copies per
# trainer GPU (accum G_R + score G_S, 7.63 GiB bf16 each at TP=2) on top of
# Megatron's main buffer, isolates every trajectory's gradient to measure
# ||g_i|| and applies b* = sum W_i R_i / sum W_i, W_i = ||g_i||^2 * w_i^2, on
# the last trajectory of each group (see header). Every knob is passed
# explicitly so a yaml default flip cannot silently change the arm.
grad_baselining=True
grad_baselining_scope="group"        # b* per prompt-group over RAW rewards (minibatch: over GRPO advantages)
grad_baselining_agg_mode="mean"      # weighted mean (paper); median | winsorized_mean soften the argmax behavior
grad_baselining_use_is_weights=True  # W_i includes the sequence IS ratio squared (paper Eq. 7)
grad_baselining_use_clipped_is_ratios=False # UNCLIPPED w_i: the baseline must see what truncation hides
grad_baselining_normalize_by_length=False   # no 1/L_i^2 factor: seq-mean-token-mean already length-normalizes g_i
grad_baselining_norm_by_std=True     # (R_i - b*) / std_group(R): keeps GRPO's advantage scale
# Host-resident accumulators (see header): G_R and G_S in pinned CPU memory, fp32,
# one d2h copy of the gradient per trajectory, CPU adds with this many torch threads.
grad_baselining_accum_device=${grad_baselining_accum_device:-cpu}
grad_baselining_accum_dtype=${grad_baselining_accum_dtype:-float32}
grad_baselining_accum_cpu_threads=${grad_baselining_accum_cpu_threads:-16}
ess_enable=${ess_enable:-True}
ess_rule=${ess_rule:-sqrt}  # sqrt | linear
# rho_on reference, FIXED to the value the base=auto_trig=0.33333 reference
# run auto-calibrated on its first on-policy update (ESS 3.23 / B=528):
# deterministic brake, active from update 1. null = re-enable auto-calibration
ess_base=${ess_base:-0.006113}
ess_use_clipped=False # ESS from unclipped ratios (paper): the brake must see what truncation hides
# Intervention threshold on ess_ratio/base: scaling engages only for
# mini-batches where the ratio falls BELOW this value; at or above it the
# update runs at full nominal lr (the multiplier jumps from 1 to
# sqrt(ratio) at the threshold). 0.33333 = a deadband at base/3: with
# base=0.006113 braking engages only below ess_ratio 0.002038, i.e.
# ESS < ~1.08 of the 528-sequence mini-batch — essentially only on
# ESS-floor mini-batches. null = legacy (engage whenever ratio < 1).
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
# replay_sampling_seed is defined in the Seeds section (defaults to SEED)
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
# Checkpoints are hf_model-only: every save_freq updates the trainer writes
# global_step_N/actor/huggingface/ (sharded safetensors + config + tokenizer)
# and skips optimizer/dist_ckpt state entirely (ckpt_save_contents=['hf_model'],
# handled by megatron_checkpoint_manager's hf-only path). Resume state is also
# off: no replay_buffer.pt (replay_buffer.save_state=False), no queue
# snapshots (save_queue_state=False), and resume_mode=disable so a leftover
# checkpoint under the same exp_name/log_dir is never picked up. These
# checkpoints are for warm-starting/eval, not resuming. save_freq<=0 disables
# saving entirely (zero checkpoint disk footprint).
save_freq=${save_freq:-20}
max_actor_ckpt_to_keep=null
ckpt_save_contents="['hf_model']"
resume_mode=disable

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches} ess-${ess_rule}-base-${ess_base_tag} opob-${grad_baselining_scope}-w2-normstd-${grad_baselining_accum_device}accum-${grad_baselining_accum_dtype} bf16-masters DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} tp${train_tp}dp$((n_gpus_training / train_tp)) hdo B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd seed-${SEED}"}
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
    data.seed=${data_seed} \
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
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.update_policy_per_traj=${update_policy_per_traj} \
    actor_rollout_ref.actor.grad_baselining.enable=${grad_baselining} \
    actor_rollout_ref.actor.grad_baselining.scope=${grad_baselining_scope} \
    actor_rollout_ref.actor.grad_baselining.agg_mode=${grad_baselining_agg_mode} \
    actor_rollout_ref.actor.grad_baselining.use_is_weights=${grad_baselining_use_is_weights} \
    actor_rollout_ref.actor.grad_baselining.use_clipped_is_ratios=${grad_baselining_use_clipped_is_ratios} \
    actor_rollout_ref.actor.grad_baselining.normalize_by_length=${grad_baselining_normalize_by_length} \
    actor_rollout_ref.actor.grad_baselining.norm_by_std=${grad_baselining_norm_by_std} \
    actor_rollout_ref.actor.grad_baselining.accum_device=${grad_baselining_accum_device} \
    actor_rollout_ref.actor.grad_baselining.accum_dtype=${grad_baselining_accum_dtype} \
    actor_rollout_ref.actor.grad_baselining.accum_cpu_threads=${grad_baselining_accum_cpu_threads} \
    actor_rollout_ref.actor.ess_scaling.enable=${ess_enable} \
    actor_rollout_ref.actor.ess_scaling.scaling_rule=${ess_rule} \
    actor_rollout_ref.actor.ess_scaling.base_ess_ratio=${ess_base} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_use_clipped} \
    actor_rollout_ref.actor.ess_scaling.trigger_ratio=${ess_trigger} \
    actor_rollout_ref.actor.megatron.seed=${megatron_seed} \
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
