#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen3-8b

# SYNCHRONOUS Qwen3-8B reference arm: verl.trainer.main_ppo (the plain colocated
# hybrid-engine trainer, NOT recipe/fully_async_policy), GRPO without a critic,
# the STOCK PPO LOSS WITH VERL'S DEFAULT PARAMETERS, on DAPO-Math-17k.
#
# WHY. Every Qwen3-8B async/replay arm on DAPO-17k (TIS-only, mu-clip blends,
# OPOB) eventually diverges with the same signature — a ~100-update
# actor/grad_norm ramp, then an in-loss KL knee and an entropy explosion (OPOB
# at update ~220, the others at ~300-440). None of them has a synchronous,
# never-stale reference on this model and data. This arm is that reference.
#
# LOSS. verl's default `policy_loss.loss_mode=vanilla`, i.e. the PPO clipped
# surrogate with dual-clip (verl/trainer/ppo/core_algos.py::
# compute_policy_loss_vanilla), all parameters at their actor.yaml defaults:
#   * clip_ratio = clip_ratio_low = clip_ratio_high = 0.2 (symmetric band
#     [0.8, 1.2]; NOT DAPO's clip-higher 0.28 and NOT the ORZ arm's 0.28),
#   * clip_ratio_c = 3.0 (dual-clip lower bound, Ye et al. 2019),
#   * loss_agg_mode = token-mean (verl default; DAPO's token-level loss),
#   * no KL in reward, no KL loss, no entropy bonus (all verl defaults),
#   * advantages: GRPO group normalisation (R - mean) / std over the 16 samples.
#   * old_log_probs are RECOMPUTED by the trainer before each update
#     (algorithm.rollout_correction.bypass_mode=false, the default), so the
#     PPO ratio compares pi_theta to the trainer's own pre-update policy, never
#     to vLLM. The rollout log-probs are cached only for the rollout_corr/*
#     diagnostic metrics; no importance correction is applied.
#
# GEOMETRY (the one deliberate deviation from the ORZ base script's 32/32).
#   train_batch_size = 128 prompts x 16 samples = 2048 sequences generated per
#   rollout step; ppo_mini_batch_size = 32 prompts = 512 sequences per optimizer
#   step; ppo_epochs = 1  ->  FOUR gradient updates per rollout step (verl's own
#   defaults, 1024/256, are also 4 updates per step). Consequences:
#   * update 1 of each step sees ratio == 1 (weights unchanged since the
#     old_log_probs pass); updates 2-4 see the drift of the previous 1-3
#     optimizer steps, so the clip band is a WORKING trust region here and
#     actor/pg_clipfrac is non-zero — unlike the 32/32 ORZ arm where the clip
#     is present but never binds.
#   * one weight sync + one old_log_prob pass (2048 seqs) per 4 updates.
#   * 17,398 prompts / 128 = 135 rollout steps = 540 optimizer updates per
#     epoch. trainer.test_freq / save_freq count ROLLOUT steps: 2 = every 8
#     updates. 512 seqs per update / 8 DP ranks = 64 per rank (divisible).
#
# OTHER SCHEDULE. AdamW lr 1e-6 constant, no warmup, weight_decay 0.01 (verl's
# actor.optim default; the async Qwen3-8B arms ran 0.1, the ORZ arm 0),
# grad clip 1.0, T=1.0 / top_p=1.0 training sampling, prompt 2048 / response
# 8192. Validation sampling T=0.8 / top_p=0.7 / n=1 — the SAME settings as
# every Qwen3-8B async arm, so val-core/math_dapo (aime-2024) and
# val-core/aime2025_dapo curves are directly comparable with them.
#
# DATA. Training: DAPO-Math-17k parquet (17,398 prompts, prompt_key=prompt,
# data_source=math_dapo) — the same file the async arms train on. Validation:
# the dapo-format aime-2024.parquet (data_source math_dapo) and
# aime-2025.parquet (data_source aime2025_dapo). No custom reward function:
# verl/utils/reward_score/__init__.py routes math_dapo and aime* to the
# built-in math_dapo scorer (+1 correct / -1 otherwise), exactly what the
# async arms score with.
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
#   * each save writes global_step_N/actor/huggingface/ - config, tokenizer and bf16
#     safetensors - directly loadable by vLLM / from_pretrained, no merge step. No
#     optimizer state, no sharded dist_ckpt/ directory at all.
#   * nothing is rotated away: ~16.4 GB per save for Qwen3-8B; at save_freq=2 over the
#     135-step epoch that is 67 saves, ~1.1 TB per epoch. Check free disk before launch
#     (remote_h100 was at 95% on 2026-08-23) and raise save_freq if it is tight.
#   * the run is NOT resumable: 'hf_model' is written but never read back, so
#     load_contents would restore nothing. resume_mode=disable makes that explicit,
#     and the trainer refuses the resume combination outright.
#
# MEMORY (why gpu_memory_utilization stays 0.5 and max_num_batched_tokens 10240).
#   Qwen3-8B: 36 layers x 8 KV heads x 128 head_dim = 147,456 B of KV per token,
#   2.57x ORZ-7B's (28 x 4 x 128 = 57,344 B); weights 16.4 GB vs 15.2 GB.
#   * In the colocated hybrid engine the Megatron trainer initialises FIRST and,
#     with param_offload=False (kept, as in the Qwen3-8B megatron async scripts),
#     stays resident: bf16 params 15.3 GiB + bf16 grad buffer 15.3 GiB + context
#     ~= 34-36 GiB, leaving ~44 GiB of the 79.65 GiB H100 free. vLLM validates
#     its claim against FREE memory at init: 0.5 * 79.65 = 39.8 GiB passes with
#     ~4 GiB headroom, 0.55 is already too tight, 0.8 aborts before step 1. So
#     0.5 is the ceiling in this configuration, not a tunable.
#   * Generation budget at 0.5: 39.8 - 15.3 (weights) - ~3 (activations, CUDA
#     graphs) ~= 21 GiB KV ~= 150k tokens per GPU: ~14 concurrent full-length
#     (10,240-token) sequences, ~25 at a ~6k mean response. Each rollout step
#     puts 2048 / 8 = 256 sequences on every GPU, i.e. ~10+ KV fills per step —
#     EXPECT THE GENERATION PHASE TO DOMINATE THE STEP (ORZ-7B at the same 0.5
#     has ~420k tokens of KV, ~1-1.5 fills). To trade that for throughput,
#     flip actor.megatron.param_offload=True and raise gpu_memory_utilization
#     to ~0.8 (~44 GiB KV): rollout_mode resumes the KV cache only after the
#     params are offloaded again, so it is memory-safe, but param offload
#     together with the CPU-offloaded precision-aware optimizer is untested
#     in this repo.
#   * max_num_batched_tokens=10240 is the chunked-prefill budget per scheduler
#     step (= max_model_len 2048+8192); it bounds prefill activations (~2-3 GiB
#     for 8B), not capacity — unchanged.
#   * MEASURED (smoke_test_oom_qwen3-8b_sync.sh, 2026-09-04, 2 steps at 8192
#     tokens, nvidia-smi every 2 s): trainer resident after init 35.9 GB; peak
#     78.1 GB of 81.6 GB during vLLM's profiling pass at init (3.5 GB headroom,
#     the tightest point); generation 73.5-74.5 GB; old_log_prob + updates
#     75-76 GB (vLLM's sleep returns less than the KV+weights it holds). Every
#     phase fits with 3-5 GB to spare; there is NO room to raise the
#     utilization without param offload. The mini-batch size does not change
#     the peak (micro-batch 1, gradient accumulation). Never set
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
#
# SEEDS. One SEED variable (default 1, the async arms' seed) feeds every seed
# knob verl.trainer.main_ppo exposes:
#   * data.seed                              -> torch.Generator of the training
#     RandomSampler (prompt order; main_ppo.py create_rl_sampler) and the
#     max_samples subset rng of RLHFDataset;
#   * actor_rollout_ref.actor.megatron.seed  -> set_random_seed on every trainer
#     worker: torch / numpy / random + Megatron's model-parallel CUDA rng
#     (megatron_workers.py); ref.megatron.seed resolves from it via oc.select
#     and follows automatically, critic.megatron.seed does NOT (literal 42 in
#     the generated config) and is set explicitly — inert here (no critic
#     under GRPO) but keeps "one SEED everywhere" true;
#   * actor_rollout_ref.actor.data_loader_seed -> the Megatron actor's mini-batch
#     shuffle seed (megatron_actor.py; only used if actor.shuffle=True, set for
#     completeness).
#   NOT covered: vLLM's own sampling seed. vllm_async_server.py passes
#   `self.config.get("seed", 0)`, but RolloutConfig has no `seed` field, so an
#   extra `+actor_rollout_ref.rollout.seed` key raises at worker init
#   ("unexpected keyword argument 'seed'") and the engine seed is always 0.
#   Making it settable needs a dataclass field (verl/workers/config/rollout.py
#   + rollout.yaml + regenerated _generated_*.yaml), not a script change.
#   The seed is part of exp_name ("seed-N") so repeats get their own log dir.
#
# ENTROPY WATCH. No stabilizer by design. actor/entropy is logged every step
# (calculate_entropy=True); Qwen3-8B starts at ~0.28. The async post-mortems'
# earliest divergence signal is a monotone actor/grad_norm ramp (healthy: flat
# ~0.08-0.15 at this batch size), then rollout_corr/kl and actor/ppo_kl
# rising; entropy direction alone is not reliable (it exploded in some arms
# and imploded in others).

set -x
export VLLM_USE_V1=1
# vLLM 0.11 auto-selects the FlashInfer top-k/top-p sampler when flashinfer is
# importable, and flashinfer 0.3.x JIT-compiles it with nvcc on first use — during
# vLLM's memory-profiling dummy run, i.e. at engine init. remote_h100 has no nvcc
# (2026-09-03 smoke: "FileNotFoundError: 'nvcc'" from flashinfer/jit/cpp_ext.py
# inside determine_available_memory, before step 1). The async Qwen3-8B scripts
# carry the same export; it selects vLLM's native torch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# aime-2024 (data_source=math_dapo -> val-core/math_dapo/acc/mean@1) and
# aime-2025 (data_source=aime2025_dapo -> val-core/aime2025_dapo/acc/mean@1),
# the same validation files and metric keys as the Qwen3-8B async arms.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}

# ================= Seeds =================
# Every seed knob main_ppo exposes (see the SEEDS header block). vLLM's sampling
# seed is not among them: it is hard-wired to 0 by RolloutConfig.
SEED=${SEED:-1}

# ================= Data =================
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
filter_overlong_prompts=True
truncation='left'

# ================= Batch geometry (4 optimizer steps per rollout step) =================
# 128 prompts x 16 rollouts = 2048 seqs generated per step; 32 prompts x 16 =
# 512 seqs per optimizer step; ppo_epochs=1 -> 4 gradient updates per step.
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
# verl actor.yaml defaults: symmetric PPO band 0.2/0.2, dual-clip c=3.0.
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.2
clip_ratio_c=3.0
# verl default (and DAPO's token-level loss): every token weighs equally.
loss_agg_mode="token-mean"
entropy_coeff=${entropy_coeff:-0}
calculate_entropy=True

# ================= Optimizer =================
lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-0}
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
# 0.5 is the ceiling with the resident (non-offloaded) trainer — see the MEMORY
# block in the header. Raise only together with param_offload=True.
gpu_memory_utilization=${gpu_memory_utilization:-0.5}
rollout_tp=1
enable_chunked_prefill=True
max_num_batched_tokens=$((1024 * 10))
temperature=1.0
top_p=1.0
top_k=-1
# Validation sampling identical to the Qwen3-8B async arms.
val_temperature=${val_temperature:-0.8}
val_top_p=${val_top_p:-0.7}
# Cache vLLM's per-token log-probs so the driver computes the rollout_corr/*
# diagnostics (KL, pearson, IS tails). NO correction is applied: algorithm.
# rollout_correction.rollout_is stays null (metrics only, no weights) and
# bypass_mode stays false (old_log_probs recomputed by the trainer).
calculate_log_probs=True

# ================= Trainer =================
test_freq=${test_freq:-2}    # rollout steps (= 8 optimizer updates)
save_freq=${save_freq:-2}    # rollout steps
total_epochs=${total_epochs:-3}
val_before_train=${val_before_train:-True}
# Weights only, in huggingface format: no optimizer state and no merge step
# before offline eval - global_step_N/actor/huggingface/ loads in vLLM as is.
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
# Mandatory, not cosmetic: 'hf_model' is written but never read back, so a
# resume would restore nothing. The trainer refuses that combination outright.
resume_mode=${resume_mode:-disable}

NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

# ================= Logging =================
exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn${n_resp_per_prompt} mini-${train_prompt_mini_bsz} ppo-epochs-${ppo_epochs} DAPO17K-AIME24-25 Qwen3-8B tp${train_tp}dp${n_gpus_per_node} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd seed-${SEED}"}
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
    data.seed=${SEED} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.filter_overlong_prompts_workers=8 \
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
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.megatron.seed=${SEED} \
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
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.megatron.seed=${SEED} \
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
