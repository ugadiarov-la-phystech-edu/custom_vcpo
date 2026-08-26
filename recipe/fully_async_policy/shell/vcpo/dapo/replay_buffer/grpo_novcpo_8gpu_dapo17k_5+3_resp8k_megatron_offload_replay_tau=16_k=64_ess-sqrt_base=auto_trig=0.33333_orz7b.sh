#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess

# Open-Reasoner-Zero-7B variant of
#   ..._megatron_offload_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh
# Every replay, ESS, optimizer, parallelism and schedule setting below is
# byte-identical to that arm; the only differences are the model, the experiment
# name, the validation sampling, and THE REWARD FUNCTION -- see "ANSWER FORMAT",
# which is the whole reason this file exists.
#
# BRAKE CHOICE (lesson of the 2026-08-25 ORZ collapse on the min-ess branch).
# The ORZ replay arm with the binary min-ESS brake (min_ess=1.1, lr_scale=0.5)
# collapsed: the mini-batch ESS pinned to its structural floor (1.0 of 528) by
# step 95 -- ~30-40 steps before entropy or grad_norm moved -- and a fixed 2x
# lr cut against a 100x ESS deficit let the off-policy runaway through
# (KL 0.001 -> 3.2 by step 140; entropy exploded 0.06 -> 3; val 0.18 -> 0.0).
# This arm uses the continuous sqrt rule with auto-calibrated base instead:
# rho_on measured on ORZ's own first update (~0.2), so at the ESS floor the lr
# scales by sqrt(0.0019/0.2) ~ 0.1 -- a 10x brake engaging progressively from
# rho/base < trigger_ratio=0.33333 (ESS ~35 of 528), exactly the window where
# the min-ess run was still healthy.
# If the balance still tips (watch replay/minibatch_staleness_mean and
# rollout_corr/kl in the first 50 updates), n_gpus_rollout=4 flips the
# 5+3 layout to 4+4: arrivals/update drop from ~45 to ~27 < 33 consumed, so
# no privileged backlog of aging groups can form at all.
#
# MODEL. Open-Reasoner-Zero/Open-Reasoner-Zero-7B is Qwen2.5-7B RL-tuned with PPO +
# critic, no KL and no entropy bonus:
#   architectures=['Qwen2ForCausalLM'], model_type=qwen2, 28 layers, hidden 3584,
#   28 q / 4 kv heads (GQA 7:1), vocab 152064, rope_theta 1e6, untied embeddings,
#   max_position_embeddings 131072, bf16, eos=bos=<|endoftext|>=151643.
# 'Qwen2ForCausalLM' is registered in verl's mcore registry (verl/models/mcore/registry.py:59,
# marked "tested"), so strategy=megatron loads it natively -- no re-alias, unlike openPangu,
# and no trust_remote_code. tp=1/pp=1/dp=3 is unchanged: 28 layers and 4 KV heads divide
# trivially at TP=1. Tokenizer is Qwen2TokenizerFast with len(tokenizer)=151665 < vocab
# 152064, so vLLM's logits[..., len(tokenizer):] mask is well-formed.
#
# PROMPTS. The parquets are byte-identical to the Qwen3-8B arm -- the DAPO wrapper is kept.
# ORZ's own chat template (shipped in tokenizer_config.json) wraps every prompt in
#   <preamble demanding <think> ... </think> <answer> ... </answer>>
#   User: {problem}
#   Assistant: <think>
# so add_generation_prompt=True primes the model mid-<think>. Prompt lengths are 160-913
# tokens against max_prompt_length=2048; nothing is dropped.
#
# ANSWER FORMAT -- MEASURED, NOT ASSUMED. 30 deduplicated AIME-2024 problems, ORZ-7B on
# one H100 at T=1.0/top_p=1.0, 8192 max tokens, run twice:
#
#   prompt form              stock math_dapo   tag-aware scorer   'Answer:' line present
#   DAPO wrapper (as-is)          0/30              5/30                  5/30
#   wrapper stripped              0/30              5/30                  1/30
#
#   30/30 responses carried BOTH <answer>...</answer> and \boxed{}; 28-29/30 closed the
#   tag on the same line as the content; 0/30 hit the token cap.
#
# Two things follow. First, the wrapper is INERT: stripping DAPO's "put your answer on its
# own line after 'Answer:'" changes neither accuracy nor shape, because ORZ obeys its own
# template. So the datasets stay identical to the Qwen arm. Second, stock math_dapo is
# BROKEN on this model: it matches (?i)Answer\s*:\s*([^\n]+) over the last 300 chars,
# capturing to end of LINE, so ' Answer: 42 </answer>' yields pred '42</answer>' and a bare
# ' \boxed{42} ' block yields '[INVALID]'. It reports 0/30 for a model whose published
# AIME-2024 pass@1 is ~15-18% -- with the same-line cases looking like WRONG MATHS rather
# than a broken parser. Left uncorrected the advantage signal would be pure noise: every
# rollout scores -1, GRPO's group-relative advantage is identically 0, and nothing learns.
#
# Hence custom_reward_function -> recipe/fully_async_policy/reward/orz_tag_aware_math.py,
# which scores the LAST <answer> block (\boxed{} > 'Answer:' line > bare text), degrades to
# math_dapo's 300-char window when no tag is present, and refuses to let a \boxed{} from
# mid-reasoning win. It returns math_dapo's exact {score, acc, pred} dict, so the reward
# managers and the rollout dumps are unchanged. Confirmed end to end through the real verl
# pipeline: the UNTRAINED checkpoint validated at 0.2333 (7/30) on these 30 problems,
# against the offline vLLM probe's 5/30 at the same sampling.
#
# VALIDATION SAMPLING. val_kwargs is 1.0/1.0 here, not the twin's 0.8/0.7: ORZ was trained
# and published at its "most basic sampling strategy", and 1.0/1.0 is what the probe used,
# so the step-0 validation point should land near 5/30 ~ 0.167 on AIME-2024. That makes the
# first validation a free check that the scorer is wired -- a near-zero reading means it is
# not. This is the only schedule knob that deliberately differs from the twin; training
# rollouts stay 1.0/1.0 in both arms, so the ESS brake sees identical sampling.
#
# CHECKPOINTS. ckpt_save_contents=['hf_model'] at bf16 is ~15.2 GB per save for ORZ-7B
# (7.6B params; the HF snapshot measures 15 GB), nothing rotated -- raise save_freq at
# launch if the disk is tight.
#
# ENTROPY WATCH. ORZ was RL-trained without a KL penalty or entropy bonus, so it starts
# lower-entropy than an instruct base. entropy_coeff=0 here, matching the twin; watch
# actor/entropy over the first ~30 steps and be ready to raise rollout temperature or add a
# small entropy_coeff if it collapses. Note the min-ESS brake reacts to the ESS floor, not
# to entropy, so it will not catch a collapse for you.
#
# ---- everything below this line is inherited from the Qwen3-8B arm ----

# MIN-ESS-braked replay arm (mbs=1 per-traj path): the ESS brake is a floor
# detector — brake (lr * ess_lr_scale) only when the mini-batch's global ESS
# is <= min_ess (1.1) effective samples, i.e. within 10% of the structural
# ESS = 1 floor a single dominant sequence produces; all other steps run at
# FULL nominal lr. Replaces the auto-captured on-policy base + base/3
# trigger + sqrt rule of the former
# ..._replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh (renamed to this
# script): the captured base was a one-mini-batch lottery draw (CV 63%
# across seeds on the fsdp2 backend) while the raw ESS trace is
# backend-independent. NOTE: the removed ess_scaling keys (scaling_rule,
# base_ess_ratio, trigger_ratio) no longer exist in the dataclass — sibling
# historical scripts that still set them fail fast at Hydra instantiation.
# Inherited replay-arm mechanics (trainer-side replay buffer, tau=16,
# eviction k=64, rmb=1, sync after every update, DAPO insertion gate, frozen
# advantages / behavior log-probs):
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
MODEL_PATH=${MODEL_PATH:-"Open-Reasoner-Zero/Open-Reasoner-Zero-7B"}
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
ess_rule=${ess_rule:-sqrt}  # sqrt | linear
# rho_on reference; null = auto-calibrate from the first update's measured
# ESS (fresh runs only), or set explicitly (1.0 = paper value for math)
ess_base=${ess_base:-null}
ess_use_clipped=False # ESS from unclipped ratios (paper): the brake must see what truncation hides
# Intervention threshold on ess_ratio/base: scaling engages only for
# mini-batches where the ratio falls BELOW this value; at or above it the
# update runs at full nominal lr (the multiplier jumps from 1 to
# sqrt(ratio) at the threshold). 0.33333 ~= a deadband at base/3: with the
# auto-calibrated base (rho_on ~= 0.033 on this setup) braking engages only
# below ESS ~0.011 — between the base=auto arm (brakes below 0.033,
# over-braked healthy steps) and the fixed 0.016 deadband arm. null = legacy
# (engage whenever ratio < 1).
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

# ================= Reward =================
# Tag-aware scorer for ORZ's <think>/<answer> format -- see the ANSWER FORMAT block in the
# header for the measurement that forces it. Same {score, acc, pred} contract as
# verl.utils.reward_score.math_dapo.compute_score, so the reward manager, the rollout dumps
# and the val-core metric names are all unchanged.
#
# Absolute, resolved from this script's own location: get_custom_reward_fn() does a plain
# os.path.exists() and the loader runs inside the rollouter AND trainer Ray actors, whose cwd
# is not guaranteed to be the repo root.
_recipe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
reward_fn_path=${reward_fn_path:-"${_recipe_root}/reward/orz_tag_aware_math.py"}
reward_fn_name=${reward_fn_name:-"compute_score"}

# ORZ-native validation sampling; the twin validates at 0.8/0.7. See the header.
val_temperature=${val_temperature:-1.0}
val_top_p=${val_top_p:-1.0}

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches} ess-${ess_rule}-base-${ess_base_tag} DAPO17K-AIME24 ORZ-7B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    custom_reward_function.path="${reward_fn_path}" \
    custom_reward_function.name="${reward_fn_name}" \
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
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.update_policy_per_traj=${update_policy_per_traj} \
    actor_rollout_ref.actor.grad_baselining.enable=${grad_baselining} \
    actor_rollout_ref.actor.ess_scaling.enable=${ess_enable} \
    actor_rollout_ref.actor.ess_scaling.scaling_rule=${ess_rule} \
    actor_rollout_ref.actor.ess_scaling.base_ess_ratio=${ess_base} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_use_clipped} \
    actor_rollout_ref.actor.ess_scaling.trigger_ratio=${ess_trigger} \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
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
