#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-fresh0.5-openpangu7b

# FRESH-SHARE-GATED variant of the openPangu-Embedded-7B MEGATRON replay / min-ESS arm
#   ..._replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_openpangu7b.sh
# (the base). The ONLY difference is replay_buffer.min_fresh_ratio=0.5 (base: 0,
# never wait) and the ` fresh-0.5` tag in the experiment name; everything else —
# model, BOS, trust flags, data, objective, layout, replay depth, brake, reuse
# decay, lr, validation — is the base's, so the two runs differ in exactly one knob.
#
# FRESH-SHARE GATE. With f=0.5 and mini-batch 33 the trainer does not run an update
# until ceil(0.5 x 33) = 17 groups have ARRIVED FROM THE ROLLOUTER since the previous
# mini-batch was composed (the fresh prefix of the next mini-batch; untrained groups
# already in the buffer do not count — they lost their one-shot freshness). Why:
# the FSDP2 openPangu replay arm's fresh share sank from ~0.4 to 0 as arrivals fell
# to ~2 per update, and the unbraked Qwen replay arm recovered from its blow-up only
# during runs of fresh-dominated updates (share 0.7-1.0) while stalling at 0.2-0.3.
# The gate is a throttle: mean trainings per group become ~1/f = 2 instead of M/A,
# and the trainer idles ~f x M/A - 1 update-times per update (at the FSDP2 arm's
# mid-run 9 arrivals/update: ~0.9 update-times of idle per update; at its late 2.2:
# ~7). MEASURED at f=0.7 on this geometry (2026-09-06, updates 1-10): floor 24, gate
# waits of 85-276 s against 175-215 s of actor work, trainer idle ratio alternating
# between ~0 and 0.4-0.55 (mean ~0.3, ~40-45 % more wall time per update than the
# base) for mean reuse ~1.3; f=0.5 here trades part of that idle for reuse ~2, the
# regime of the no-knee Qwen k-2 ppo-epochs-2 arms. Waiting does not age the fresh
# groups (no version is produced meanwhile) and it shortens their arrival lag (fresh
# groups arrived 1-3 versions old under the gate vs 4-6 without). The wait is capped
# by replay_buffer.min_fresh_wait_timeout_s (yaml default 3600 s, more than the
# ~25 min a late-run wait would take here); on the cap the update runs anyway and
# replay/fresh_floor_waived logs 1. Watch replay/minibatch_new_ratio (>= 0.52 by
# construction), replay/minibatch_fresh_staleness_mean, replay/fresh_wait_s and
# fully_async/timing/cumulative_training_time (the idle is charged to it, correctly).
# Compare against the base at equal cumulative_training_time, not equal update count.
#
# ---- everything below this line is inherited from the base ----
#
# openPangu-Embedded-7B variant of the MEGATRON replay / min-ESS arm
#   ..._5+3_resp8k_megatron_offload_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh
# (the Qwen3-8B twin). Same data, objective, layout, HDO backend recipe, IS
# correction, lr and validation sampling; the differences are THE MODEL (and the
# two trust_remote_code flags + the BOS switch it needs), THE REPLAY DEPTH and
# the brake floor (see REPLAY DEPTH), the reuse decay, and the experiment name.
#
# THE MODEL. MODEL_PATH points at a LOCAL, RE-ALIASED checkpoint, not the hub id.
# openPangu ships as a trust_remote_code PanguEmbeddedForCausalLM whose modeling
# file does not import under transformers 4.57.6, and vLLM 0.11.0 has no
# PanguEmbeddedForCausalLM; the generated modeling code is a Llama derivative
# whose math is byte-for-byte Llama, so scripts/realias_openpangu_to_llama.py
# rewrites config.json to LlamaForCausalLM with attention_bias=true,
# mlp_bias=false and no modeling auto_map (weights need no remapping). The
# tokenizer stays the custom PanguTokenizer (tokenizer_config.json keeps its
# auto_map), hence BOTH trust_remote_code keys below: data.trust_remote_code
# (dataset-side tokenizer, fully_async_main.py) and
# actor_rollout_ref.model.trust_remote_code (agent-loop tokenizer, Megatron
# weight load, vLLM engine). They are independent; setting one and not the
# other crashes the other half of the system. Ray workers unpickle that
# tokenizer BY REFERENCE (transformers_modules.<hash>.tokenization_openpangu),
# which only resolves if the HF modules cache is on PYTHONPATH — exported below.
#   Build the checkpoint with: python scripts/realias_openpangu_to_llama.py --out <MODEL_PATH>
#
# MEGATRON AND THE ATTENTION BIAS. attention_bias=true gives q/k/v AND o_proj a
# bias. Megatron-Core allocates o_proj's only through add_bias_linear, one flag
# for linear_proj plus BOTH MLP projections. This branch (ported from
# baselines_main-ppo_openpangu, GPU-validated by the openPangu sync seed runs):
#   * derives add_bias_linear from the HF attention_bias/mlp_bias flags
#     (verl/models/mcore/config_converter.py::hf_to_mcore_config_dense);
#   * FREEZES the MLP biases add_bias_linear also creates, at their zero TE init,
#     inside the model provider — before DDP/optimizer construction — so the
#     forward equals HF Llama with mlp_bias=false
#     (model_initializer.py::freeze_absent_mlp_biases; the log line
#     "[DenseModel] add_bias_linear=True with mlp_bias=False: froze N MLP bias
#     tensors" confirms it at startup);
#   * loads o_proj.bias (loader.py), syncs it to vLLM every param version
#     (weight_converter.py) and EXPORTS it in hf_model checkpoints (saver.py) —
#     without the export a Llama checkpoint with attention_bias=true is refused
#     by vLLM. Qwen arms are untouched (attention_bias absent -> False).
#
# BOS (data.add_bos_token_to_prompt=True, the ONE prompt-level difference from the
# twin). The Pangu tokenizer has add_bos_token=true and a chat template that never
# emits <s>; the official recipe tokenizes the rendered template with a plain
# tokenizer(text) call, which PREPENDS <s> (id 1). verl's default path drops it
# (measured: 13 vs 14 tokens). With the flag on, RLHFDataset (input_ids,
# raw_prompt_ids, the overlong filter) and the single-turn agent loops all go
# through verl/utils/dataset/prompt_utils.py::maybe_prepend_bos, so training,
# validation and prompt-length accounting agree. A no-op for Qwen (no BOS token);
# guarded against templates that already start with the BOS string.
#   * Offline eval of these checkpoints must reproduce the BOS (vLLM's
#     chat-completions endpoint defaults add_special_tokens=False).
#   * NOT comparable with the FSDP2 openPangu replay arm, which trained WITHOUT
#     BOS at val 1.0/0.8. Comparable with the openPangu sync arms (BOS, 0.8/0.7)
#     and with the Qwen twin up to the model and the replay depth.
#
# REWARD. Stock math_dapo through the default reward manager, exactly as the twin
# (no custom_reward_function): the DAPO prompts ask for an "Answer:" line,
# math_dapo takes solution_str[-300:] and the LAST (?i)Answer\s*:\s*(...) match,
# and the Pangu thinking delimiters [unused16]/[unused17] are not special tokens,
# so they survive the decode and need no handling. The one live risk is a
# verbose epilogue pushing the answer out of the 300-char window (rollout dumps
# show it as [INVALID] preds). Validation keys: val-core/math_dapo/acc/mean@1
# (aime-2024), val-core/aime2025_dapo/acc/mean@1.
#
# REPLAY DEPTH. tau=8 / k=32 / min_ess=1.07 / reuse_halflife=1 (the twin: 16 / 64
# / 1.1 / none), the settings chosen for the low-entropy ORZ-7B arm. The FSDP2
# openPangu replay arm under the twin's 16/64 collapsed at update 57 (arrivals
# fell to 2 per update once all-correct groups were gated out, reuse 15x, entropy
# 0.36 -> 7.2). Halving the depth bounds the reuse window; the async generation
# quota (async_training.staleness_threshold) follows k so no group is generated
# only to be evicted unseen; a staleness-32 group carries 2^-4 at tau=8, the
# twin's terminal weight at half the depth. The reuse decay halves a group's
# draw weight per training (REPLAY_REUSE_PENALTY_DISCUSSION.md): mean reuse is
# unchanged (M/A), the over-trained tail and the trained-once-then-evicted tail
# go away, replayed picks get younger. min_ess=1.07 brakes only within 7% of
# the structural ESS=1 floor. Every knob is env-overridable.
#
# MEMORY — NOT MEASURED ON THIS MODEL. The twin's envelope is assumed to carry over
# (openPangu-7B: 34 layers x 8 KV heads x 128 head_dim, 8.0B params, vocab
# 153,376 — the same shape budget as Qwen3-8B); gpu_memory_utilization is
# env-overridable for the first launch. Never set
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments. Run
# smoke_test_openpangu7b_replay_3+3.sh first: it is the first end-to-end run of
# the Megatron openPangu path through the replay trainer.
#
# ---- everything below this line is inherited from the Qwen3-8B twin ----
#
# MIN-ESS-braked replay arm (mbs=1 per-traj path): the ESS brake is a floor
# detector — brake (lr * ess_lr_scale) only when the mini-batch's global ESS
# is <= min_ess effective samples, i.e. within a few percent of the structural
# ESS = 1 floor a single dominant sequence produces; all other steps run at
# FULL nominal lr. Replaces the auto-captured on-policy base + base/3
# trigger + sqrt rule of the former
# ..._replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh (renamed to this
# script): the captured base was a one-mini-batch lottery draw (CV 63%
# across seeds on the fsdp2 backend) while the raw ESS trace is
# backend-independent. NOTE: the removed ess_scaling keys (scaling_rule,
# base_ess_ratio, trigger_ratio) no longer exist in the dataclass — sibling
# historical scripts that still set them fail fast at Hydra instantiation.
# Inherited replay-arm mechanics (trainer-side replay buffer, tau=8,
# eviction k=32 — the twin runs 16/64, see REPLAY DEPTH — rmb=1, sync after
# every update, DAPO insertion gate, frozen advantages / behavior log-probs):
#   * update_policy_per_traj=True: every mini-batch's sequence-level IS
#     ratios against the cached behavior log-probs are DP-all-reduced into
#     ess_ratio = (sum w)^2 / (B * sum w^2), logged as staleness/ess_ratio
#     (ESS in effective samples = ess_ratio * B, B = 528 here). This mbs=1
#     path and the dynbsz arm now share the same max-shifted log-space
#     computation (verl/workers/utils/ess.py), so ESS is exact at any drift
#     and floored at 1 — the brake multiplier is exactly ess_lr_scale on
#     degenerate mini-batches, never 0 and never silently off. It used to
#     read the fp32 torch.exp of the log-IS sum: sums below ~-87 flushed to
#     0 (ESS 0) and above ~88.7 to inf (ESS NaN), and both ran the step at
#     FULL lr — observed at steps 345/346/348 of the 2026-08 replay run.
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
#   * Groups staler than replay_buffer.staleness_threshold=32 updates are
#     evicted after each update; scores are recomputed each update. With
#     tau=8 a staleness-32 group still carries sampling weight 2^-4 = 1/16 —
#     the twin's terminal weight at half the depth (see REPLAY DEPTH). The
#     buffer retains every kept group of the last 32 updates (~500-800
#     groups, roughly 4-6 GB driver RAM and the same for replay_buffer.pt in
#     checkpoints).
#   * Warm-up/watermark: requires_mini_batches=1 — the first update consumes
#     a fresh mini-batch of unseen groups; afterwards training pauses only
#     while the buffer holds < 1*33 = 33 groups.
#   * async_training.staleness_threshold=32 aligns the rollouter's generation
#     quota with the eviction horizon (33*(32+1)=1089 groups licensed; in
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
# Ray workers deserialize the trust_remote_code tokenizer BY REFERENCE, as
# transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer; that dynamic
# package is only on sys.path in a process that has itself loaded remote code.
# Exporting the HF modules cache on PYTHONPATH makes the reference resolvable in
# every Ray worker (verified on remote_smoke, 2026-08-23).
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;  # already there (e.g. the wrapper set it)
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# A LOCAL, RE-ALIASED copy — not the hub id (scripts/realias_openpangu_to_llama.py).
MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
# Both keys, see THE MODEL in the header.
trust_remote_code=${trust_remote_code:-True}
# Prepend <s> to every prompt, as the official openPangu recipe does. See BOS in the header.
add_bos_token_to_prompt=${add_bos_token_to_prompt:-True}
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
gpu_memory_utilization=${gpu_memory_utilization:-0.9} # the twin's value; env-overridable for the first launch on this model
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
entropy_coeff=${entropy_coeff:-0} # env-overridable for the smoke wrapper (reward-independent gradient); the arm runs 0
calculate_entropy=True # log actor/entropy even with entropy_coeff=0
grad_clip=1.0

# ================= Optimizer =================
lr=${lr:-1e-6} # env-overridable for the smoke wrapper; the arm runs 1e-6
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
# to ess_ratio <= min_ess/B (B = 528 here). No measured reference, no base
# capture: the threshold is backend-independent, unlike the auto-base.
min_ess=${min_ess:-1.07} # the twin: 1.1 (see REPLAY DEPTH)
ess_lr_scale=${ess_lr_scale:-0.5}
ess_use_clipped=False # ESS from unclipped ratios (paper): the brake must see what truncation hides
ess_tag="min-ess-${min_ess}-lrscale-${ess_lr_scale}"

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
staleness_threshold=${staleness_threshold:-32.0} # follows replay k (the twin: 64)
updates_per_param_sync=1     # REQUIRED by replay mode: sync after every update
num_minibatches_per_update=1 # REQUIRED by replay mode: one mini-batch per update
partial_rollout=True
use_rollout_log_probs=True

# ================= Replay buffer =================
replay_enable=${replay_enable:-True}
replay_tau=${replay_tau:-8} # the twin: 16 (see REPLAY DEPTH)
replay_staleness_threshold=${replay_staleness_threshold:-32} # the twin: 64
replay_requires_mini_batches=${replay_requires_mini_batches:-1}
replay_sampling_seed=${replay_sampling_seed:-1234}
# Reuse-decay half-life in trainings (2^(-times_trained/nu) on the replay draw
# weight; REPLAY_REUSE_PENALTY_DISCUSSION.md, REPLAY DEPTH above). The twin runs
# null (staleness-only draw). Tagged into exp_name when set.
replay_reuse_halflife=${replay_reuse_halflife:-1}
replay_reuse_tag=""
if [[ "${replay_reuse_halflife}" != "null" ]]; then replay_reuse_tag=" nu-${replay_reuse_halflife}"; fi
# Fresh-share gate (replay_buffer.min_fresh_ratio): with a value f > 0 the trainer
# does not run an update until ceil(f x mini_bsz) groups have ARRIVED FROM THE
# ROLLOUTER since the previous mini-batch was composed (the fresh prefix of the
# next mini-batch; untrained groups already in the buffer do not count). Mean
# trainings per group become ~1/f instead of M/A (mini-batch groups / arrivals per
# update) at the cost of trainer idle time (~f x M/A - 1 update-times per update);
# waiting does not age the fresh groups (no version is produced meanwhile). 0 keeps
# the arm bit-for-bit. The wait is capped by replay_buffer.min_fresh_wait_timeout_s
# (yaml default 3600 s; on the cap the update runs anyway and
# replay/fresh_floor_waived logs 1). Fresh groups are NOT on-policy — a long
# rollout spans several updates — so watch replay/minibatch_fresh_staleness_mean.
replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.5} # the base: 0 (see FRESH-SHARE GATE)
replay_fresh_tag=""
if [[ "${replay_min_fresh_ratio}" != "0" ]]; then replay_fresh_tag=" fresh-${replay_min_fresh_ratio}"; fi
replay_save_state=${replay_save_state:-True} # replay_buffer.pt is part of the resumable state (see CHECKPOINTS)

# ================= Elastic mechanisms OFF / stop-the-world accounting =================
# Replay mode subsumes DAPO filtering (insertion gate always on) and replaces
# opportunistic/fractional epochs with score-weighted replay.
dynamic_filtering_enable=False
min_buffered_batches=1.0
opportunistic_enable=False
opportunistic_max_extra_epochs=0
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}
save_queue_state=${save_queue_state:-True} # rollout_queue.pt / message_queue.pt are part of the resumable state (see CHECKPOINTS)

# ================= Training/Rollout Steps =================
# Same 66000-prompt generation budget as the B-33x4 arms (500 steps * 132
# groups). Fed prompts, not kept groups: filtering shortens the effective
# trained horizon proportionally.
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
# Model versions now tick once per UPDATE (not per 132-group step): validate /
# checkpoint every 20 updates (=660 groups consumed, matching the 5-step
# cadence of the B-33x4 arms in group units).
test_freq=${test_freq:-5} # the twin: 20
# CHECKPOINTS — two tiers. Every save_freq updates (parameter-version units, one
# version per replay update) the trainer writes a FULL checkpoint:
#   global_step_N/actor/huggingface/       hf_model export (bf16 safetensors + config +
#                                          tokenizer; loadable by vLLM as is) — KEPT AT EVERY SAVE
#   global_step_N/actor/dist_ckpt/         Megatron dist-checkpoint: model + optimizer + extra
#   global_step_N/replay_buffer.pt         the replay buffer (groups, scores, RNG, counters)
#   global_step_N/rollout_queue.pt, message_queue.pt   the rollouter's in-flight / queued groups
#   global_step_N/timing_state.json, data.pt, actor/transformer_config.json   (small)
# and then, with async_training.resumable_ckpts_to_keep=1, deletes dist_ckpt/ + replay_buffer.pt +
# rollout_queue.pt + message_queue.pt from every OLDER global_step_* directory, so exactly one
# resumable checkpoint (the newest) exists at any time while the hf_model of every save stays
# for evaluation. trainer.max_actor_ckpt_to_keep must stay null: that knob rmtree's whole actor/
# directories, hf_model included. The base arm and the Qwen twin keep the hf-only policy.
#
# Disk (remote_h100, ~600 GB free): hf ~16 GB per save (the accumulating part: ~320 GB per 100
# updates at save_freq=5 — this fills the disk first); the resumable state ~65-100 GB
# (bf16 weights + bf16 master + Adam moments under the precision-aware CPU-offload optimizer;
# NOT YET MEASURED — record it after the first save) + replay_buffer.pt 4-6 GB + queue snapshots,
# present in the newest directory only, ~2x that at the instant of a save.
#
# resume_mode=auto: a relaunch under the SAME exp_name resumes from the newest full checkpoint
# (actor, optimizer, replay buffer, queues, timing offsets all restored; a directory whose resume
# state was pruned is refused with a clear error). A fresh start needs a new exp_name or removing
# the run directory. The stop-the-world save pause (pause_generation_during_save) now brackets a
# multi-minute dist-checkpoint write to NFS instead of the ~45 s hf export; it is excluded from
# cumulative_training_time by design. This is the first arm on this stack that saves the
# Megatron optimizer under optimizer_cpu_offload — run the 3+3 smoke with
# ARM_SCRIPT=<this script> and verify_checkpoints.py --resumable-last 1 before a long run.
save_freq=${save_freq:-5} # the twin: 20
max_actor_ckpt_to_keep=null # MUST stay null (see CHECKPOINTS)
ckpt_save_contents=${ckpt_save_contents:-"['model','optimizer','extra','hf_model']"} # the base: ['hf_model']
resumable_ckpts_to_keep=${resumable_ckpts_to_keep:-1} # the base: null (nothing to prune)
resume_mode=${resume_mode:-auto} # the base: disable

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches}${replay_reuse_tag}${replay_fresh_tag} ess-${ess_tag} DAPO17K-AIME24 openPangu-7B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd bos"}
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
    data.trust_remote_code=${trust_remote_code} \
    data.add_bos_token_to_prompt=${add_bos_token_to_prompt} \
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
    actor_rollout_ref.model.trust_remote_code=${trust_remote_code} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
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
    async_training.resumable_ckpts_to_keep="${resumable_ckpts_to_keep}" \
    async_training.replay_buffer.enable="${replay_enable}" \
    async_training.replay_buffer.tau="${replay_tau}" \
    async_training.replay_buffer.staleness_threshold="${replay_staleness_threshold}" \
    async_training.replay_buffer.requires_mini_batches="${replay_requires_mini_batches}" \
    async_training.replay_buffer.sampling_seed="${replay_sampling_seed}" \
    async_training.replay_buffer.reuse_halflife="${replay_reuse_halflife}" \
    async_training.replay_buffer.min_fresh_ratio="${replay_min_fresh_ratio}" \
    async_training.replay_buffer.save_state="${replay_save_state}" \
    +async_training.bsz_per_dp_rank="${bsz_per_dp_rank}" "$@"
