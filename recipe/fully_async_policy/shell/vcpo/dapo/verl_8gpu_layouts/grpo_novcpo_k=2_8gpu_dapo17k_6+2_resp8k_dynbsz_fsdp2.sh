#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo

# Vanilla fully-async GRPO on the fork's pipeline (recipe.fully_async_policy) —
# throughput-oriented variant of grpo_novcpo_k=2_8gpu_dapo17k_4+4_resp8k.sh:
#   * 6 rollout + 2 trainer GPUs (rollout-bound at 4+4: idle_ratio 0.56 at step 5)
#   * FSDP2 trainer backend, pure DP=2 (no TP/megatron)
#   * use_dynamic_bsz=True (packs ~4-5 seqs per micro-batch, 1.5-3x faster updates)
#   * torchao _AdamW with bf16 stochastic rounding, fully GPU-resident (no CPU
#     offload, no fp32 master): per trainer GPU ~8.2 (sharded bf16 params)
#     + 8.2 (sharded grads) + 16.4 (bf16 moments) ~= 33 GB static on 80 GB.
#     fsdp_config.model_dtype=bf16 is REQUIRED for this: verl builds the FSDP
#     actor in fp32 by default (fsdp_workers.py:311), which quadruples static
#     memory to ~66 GB (OOMed at the first packed micro-batch, smoke attempt 4)
#     AND makes bf16_stochastic_round silently inert (SR only acts on bf16 params).
#     entropy chunking+checkpointing are also required at the 20480-token
#     micro-batch budget: non-chunked entropy materializes logits-sized fp32
#     intermediates (~18 GB per micro-batch at vocab 151936) and OOMed the
#     backward pass in smoke attempt 5.
#     Stochastic rounding makes master-less bf16 updates unbiased — at lr=1e-6
#     a deterministic bf16 update rounds to zero for ~98% of weights.
#     NOTE: numerics differ from the Megatron runs (exact fp32 masters there);
#     treat cross-backend comparisons as system-level, not ablations.
#   * async_training.skip_recompute_old_log_prob=True works on FSDP via the
#     dp_actor port of the Megatron deferred old-log-prob path
#     (tests/workers/actor/test_skip_recompute_old_log_prob_on_cpu.py).
# Requires torchao in the environment.
# VCPO mechanisms stay at their defaults (off): actor.update_policy_per_traj=False,
# actor.ess_scaling.enable=False, actor.grad_baselining.enable=False.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0
# Do NOT export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: Ray
# workers inherit this shell's env, and vLLM's sleep-mode allocator hard-asserts
# against expandable segments (vllm/device_allocator/cumem.py:149) — every
# rollout engine would crash at init. If trainer-side fragmentation needs it,
# use verl's set_expandable_segments(True) inside the trainer worker instead.

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
n_gpus_rollout=${n_gpus_rollout:-6}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# ================= Rollout =================
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
gen_tp=1
n_resp_per_prompt=${n_resp_per_prompt:-16}
# 0.8, not 0.9: on the FSDP worker path the vLLM engine process sits ~7-10 GB
# above the configured pool target (smoke tests, 2026-07-30; the Megatron path
# runs ~4 GB over). The rollout GPUs must additionally keep a free margin for
# (a) the vLLM sampler warmup transient (~0.9 GB at max_num_seqs=512) and
# (b) sync_rollout_weights: a per-tensor broadcast buffer (largest = embedding,
# ~1.25 GB bf16) + NCCL communicator buffers on first use. 0.9 OOMed at (a),
# 0.85 OOMed at (b); 0.8 leaves ~5 GB free. TODO: find the extra resident
# memory in the FSDP DetachAsyncRolloutWorker — it costs ~10% KV capacity.
gpu_memory_utilization=0.8
# vLLM v1 warms the sampler with max_num_seqs dummy requests AFTER filling the KV
# pool; the transient scales with it. 512 is far above the ~50-60 seqs the
# engines actually run concurrently at these lengths.
max_num_seqs=512
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
max_response_length=8192
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= Trainer Backend (FSDP2) =================
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=128 # 128*16=2048 seqs; divisible by trainer DP=2
micro_bsz_per_gpu=1
use_dynamic_bsz=True
# token budget per micro-batch: 2x max_model_len (conservative; the length
# distribution drifts during training, and FSDP materializes full-vocab logits)
ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))  # 20480
log_prob_micro_bsz_per_gpu=1

bsz_per_dp_rank=32 # Rollout Bsz

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

# ================= IS / Rollout Correction =================
# Token-level truncated IS with PPO-clip loss, matching the installed-verl
# reference script. This fork has no rollout_correction.loss_type key:
# use_policy_gradient=False is the equivalent of loss_type=ppo_clip.
rollout_is="token"
rollout_is_threshold="2.0"
rollout_rs=null
rollout_rs_threshold=null
bypass_mode=False
use_policy_gradient=False

skip_recompute_old_log_prob=True
compute_prox_log_prob=False

# ================= Async Training =================
# k=2 matches the VCPO baseline's staleness gating (timing-neutral in this
# layout: concurrency saturates at 128 prompt-groups for any k >= 0).
staleness_threshold=${staleness_threshold:-2.0}
updates_per_param_sync=1
num_minibatches_per_update=1
partial_rollout=True
use_rollout_log_probs=True

# ================= Training/Rollout Steps =================
total_rollout_steps=${total_rollout_steps:-$((500 * num_minibatches_per_update * updates_per_param_sync * train_prompt_mini_bsz))}
epochs=10000000
test_freq=${test_freq:-5}
save_freq=5              # checkpoint every 5 param versions (= every 5 steps at trigger_parameter_sync_step=1)
max_actor_ckpt_to_keep=1 # keep only the most recent checkpoint

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO k-${staleness_threshold} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} fsdp2 dynbsz sr-adamw B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    --config-name=fully_async_ppo_trainer.yaml \
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
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim \
    actor_rollout_ref.actor.optim.optimizer=_AdamW \
    "actor_rollout_ref.actor.optim.override_optimizer_config={bf16_stochastic_round:true}" \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=${precision_dtype} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
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
    +async_training.bsz_per_dp_rank="${bsz_per_dp_rank}" "$@"
