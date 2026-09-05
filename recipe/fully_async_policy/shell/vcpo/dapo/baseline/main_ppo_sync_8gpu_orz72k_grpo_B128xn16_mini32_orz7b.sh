#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-orz72k-grpo-B128-mini32-orz7b

# SYNCHRONOUS ORZ-continuation arm, QWEN-GEOMETRY VARIANT: verl.trainer.main_ppo
# (the plain colocated hybrid-engine trainer, NOT recipe/fully_async_policy),
# GRPO without a critic, continuing Open-Reasoner-Zero-7B on ITS OWN RL
# training set (orz-math-72k) with ORZ prompts.
#
# This is main_ppo_sync_8gpu_deepmath_grpo_B32xn16_orz7b.sh with the training
# parquet swapped back to orz-math-72k and FOUR knobs taken from
# main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh (the Qwen3-8B
# synchronous reference arm), everything else unchanged:
#   * train_prompt_bsz      32   -> 128   (prompts generated per rollout step)
#   * train_prompt_mini_bsz 32   -> 32    (NEW knob: prompts per optimizer step;
#                                          the base script tied it to
#                                          train_prompt_bsz)
#   * clip_ratio*           0.2/0.28/3.0 -> 0.2/0.2/0.2/3.0 (verl actor.yaml
#                                          defaults: symmetric band [0.8, 1.2],
#                                          dual-clip c=3.0; NOT DAPO's clip-
#                                          higher 0.28)
#   * weight_decay          0    -> 0.01  (verl actor.optim default)
#
# GEOMETRY (the deliberate deviation from the base script's strictly on-policy
# 32/32). train_batch_size = 128 prompts x 16 samples = 2048 sequences generated
# per rollout step; ppo_mini_batch_size = 32 prompts = 512 sequences per
# optimizer step; ppo_epochs = 1 -> FOUR gradient updates per rollout step.
# Consequences:
#   * update 1 of each step sees ratio == 1 (weights unchanged since the
#     old_log_probs pass); updates 2-4 see the drift of the previous 1-3
#     optimizer steps, so the clip band is a WORKING trust region here and
#     actor/pg_clipfrac is non-zero — unlike the 32/32 base arm where the clip
#     is present but never binds. The symmetric 0.2 band is therefore no longer
#     inert insurance: it bounds the per-step drift of a model whose entropy
#     starts at ~0.06.
#   * one weight sync + one old_log_prob pass (2048 seqs) per 4 updates.
#   * 47,981 prompts / 128 = 374 rollout steps = 1,496 optimizer updates per
#     epoch. trainer.test_freq / save_freq count ROLLOUT steps. 512 seqs per
#     update / 8 DP ranks = 64 per rank (divisible).
#   * old_log_probs are RECOMPUTED by the trainer before each update
#     (algorithm.rollout_correction.bypass_mode=false, the default), so the
#     PPO ratio compares pi_theta to the trainer's own pre-update policy, never
#     to vLLM. The rollout log-probs are cached only for the rollout_corr/*
#     diagnostic metrics; no importance correction is applied.
#
# WHAT IS KEPT FROM THE BASE ARM (ORZ parity where it still applies):
#   * ppo_epochs=1, lr 1e-6 constant, NO warmup, grad clip 1.0.
#   * no KL in reward, no KL loss, no entropy bonus, T=1.0/top_p=1.0 training
#     sampling; validation sampling T=1.0/top_p=1.0/n=1 (ORZ-style, NOT the
#     Qwen arm's T=0.8/top_p=0.7 — keeps the val curves comparable with the
#     other ORZ-7B arms).
#   * loss_agg_mode=token-mean (ORZ's packed PolicyLoss degenerates to a global
#     token mean; also the Qwen arm's and verl's default).
#   * GRPO group advantages (R - mean) / std over the 16 samples; all-correct/
#     all-wrong groups become zero-advantage no-ops, no group filter.
#
# DATA. Training: orz-math-72k parquet (47,981 rows after conversion,
# prompt_key=prompt, data_source=math_dapo, ORZ inner instruction with
# <answer></answer> tags) — Open-Reasoner-Zero's own RL training data, i.e. the
# set ORZ-7B was already trained on for 700+ steps (the earlier async ORZ-72k
# arms trained flat on it; the DeepMath base arm was the attempt to move off
# it). Validation: aime-2024/2025 ORZ-prompt parquets exactly as the ORZ-72k
# async arms (x32 duplication, data_source aime2024_orz/aime2025_orz -> the
# same val-core/.../acc/mean@1 metric keys, so curves are directly
# comparable).
#
# REWARD. recipe/fully_async_policy/reward/orz_tag_aware_math.py (the tiered
# version ported from the _final branch): tag-aware extraction, then
# math_dapo string equality -> vendored ORZ is_equiv -> sympy parse_latex in
# a forked child under a hard ORZ_MATH_SYMPY_TIMEOUT=1.0s kill deadline.
# ~24% of ORZ-72k ground truths are LaTeX expressions; the pre-tier scorer
# would score them as false negatives.
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
#   * each save writes global_step_N/actor/huggingface/ - config, tokenizer and bf16
#     safetensors - directly loadable by vLLM / from_pretrained, no merge step. No
#     optimizer state, no sharded dist_ckpt/ directory at all.
#   * nothing is rotated away: ~15.2 GB per save for ORZ-7B; at save_freq=5 over the
#     374-step epoch that is 74 saves, ~1.1 TB per epoch. Check free disk before
#     launch (the shared cloud.ru volume was at 96% on 2026-09-05) and raise
#     save_freq if it is tight. test_freq=5 (every 20 optimizer updates, ~22 min
#     of training at the measured ~4.4 min/step) — the first launch at 2/2 spent
#     ~25% of wall time on validation+saves and trained flat for 200 updates.
#   * the run is NOT resumable: 'hf_model' is written but never read back, so
#     load_contents would restore nothing. resume_mode=disable makes that explicit,
#     and the trainer refuses the resume combination outright.
#
# MEMORY. Unchanged from the base arm: gpu_memory_utilization=0.5 is the
# ceiling with the resident (non-offloaded) Megatron trainer (~33 GiB held
# before vLLM claims its share; 0.6 aborted the 2026-08-31 launch before
# step 1). The larger rollout step only changes how many KV fills generation
# takes (2048 / 8 = 256 sequences per GPU per step vs 64), not the peak: the
# mini-batch is unchanged at 512 sequences and micro-batch 1 with gradient
# accumulation keeps the trainer footprint the same. Expect the generation
# phase to take ~4x longer per step than the base arm.
#
# ENTROPY WATCH. No stabilizer besides the PPO clip. actor/entropy is logged
# every step (calculate_entropy=True): this model starts at ~0.06 — watch the
# first ~50 steps; a monotone actor/grad_norm ramp (healthy: flat 0.15-0.2) is
# the earliest divergence signal known from the async post-mortems. With 4
# updates per step, actor/pg_clipfrac and actor/ppo_kl are now informative
# too (they are identically 0 in the 32/32 arm).

set -x
export VLLM_USE_V1=1
# vLLM 0.11 auto-selects the FlashInfer top-k/top-p sampler when flashinfer is
# importable, and flashinfer 0.3.x JIT-compiles it with nvcc on first use — during
# vLLM's memory-profiling dummy run, i.e. at engine init. The cloud.ru jobs have no
# nvcc (2026-09-05 launch of this script on remote_2: "FileNotFoundError: 'nvcc'"
# from flashinfer/jit/cpp_ext.py inside determine_available_memory, before step 1;
# same failure as the 2026-09-03 Qwen3-8B smoke). The Qwen3-8B sync arm carries the
# same export; it selects vLLM's native torch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
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

# ================= Batch geometry (4 optimizer steps per rollout step) =================
# 128 prompts x 16 rollouts = 2048 seqs generated per step; 32 prompts x 16 =
# 512 seqs per optimizer step; ppo_epochs=1 -> 4 gradient updates per step.
# (Values from the Qwen3-8B B128xn16 mini32 reference arm.)
train_prompt_bsz=${train_prompt_bsz:-128}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-32}
n_resp_per_prompt=${n_resp_per_prompt:-16}
ppo_epochs=${ppo_epochs:-1}

# ================= Algorithm =================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
# verl actor.yaml defaults (from the Qwen3-8B reference arm): symmetric PPO
# band 0.2/0.2, dual-clip c=3.0. NOT the base ORZ arm's clip-higher 0.28.
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.2
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
# verl actor.optim default (from the Qwen3-8B reference arm); the base ORZ arm
# ran 0 (ORZ parity), the async ORZ arms 0.1.
weight_decay=${weight_decay:-0.01}
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
# 0.5, not the async arms' 0.6: in the colocated hybrid engine the Megatron
# trainer initializes FIRST and holds ~33 GiB (bf16 params + bf16 grad buffer
# + overhead), and vLLM validates its claim against FREE memory at init —
# 0.6*79.2 = 47.5 GiB > the ~46.3 GiB left, which aborted the 2026-08-31
# launch before step 1. 0.5 claims 39.6 GiB (~25 GiB KV after weights, enough
# for 128 concurrent seqs at our lengths) with ~7 GiB headroom. To go higher,
# flip megatron param_offload/grad_offload=True first.
gpu_memory_utilization=${gpu_memory_utilization:-0.5}
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
# stays the uncorrected objective. Flip rollout_is=token t=2.0 later if the
# engine gap ever grows.
calculate_log_probs=True

# ================= Trainer =================
test_freq=${test_freq:-5}    # rollout steps (= 20 optimizer updates)
save_freq=${save_freq:-5}    # rollout steps
total_epochs=${total_epochs:-3}
val_before_train=${val_before_train:-True}
# Weights only, in huggingface format: no optimizer state (fp32 master + 2 adam
# moments is ~6x the bf16 weights on the megatron distributed optimizer) and no
# merge step before offline eval - global_step_N/actor/huggingface/ loads in
# vLLM as is. See the CHECKPOINTS header block for sizes.
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
# Mandatory, not cosmetic: 'hf_model' is written but never read back, so a
# resume would restore nothing. The trainer refuses that combination outright.
resume_mode=${resume_mode:-disable}

NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

# ================= Logging =================
exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn${n_resp_per_prompt} mini-${train_prompt_mini_bsz} ppo-epochs-${ppo_epochs} ORZ72K-AIME24ORZ ORZ-7B tp${train_tp}dp${n_gpus_per_node} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}" \
    trainer.resume_mode=${resume_mode} \
    trainer.rollout_data_dir=null \
    trainer.log_val_generations=0 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.total_epochs=${total_epochs} "$@"
