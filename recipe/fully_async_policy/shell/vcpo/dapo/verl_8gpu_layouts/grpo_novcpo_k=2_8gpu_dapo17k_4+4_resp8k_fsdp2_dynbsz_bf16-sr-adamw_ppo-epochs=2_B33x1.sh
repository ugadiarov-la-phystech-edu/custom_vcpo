#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-fsdp2

# grpo_novcpo_k=2_8gpu_dapo17k_4+4_resp8k_fsdp2_dynbsz_bf16-sr-adamw_ppo-epochs=2_B33x1.sh
# 4+4-layout variant of the 5+3 bf16-SR FSDP2 dynbsz B-33x1 arm
# (grpo_novcpo_k=2_..._5+3_resp8k_fsdp2_dynbsz_bf16-sr-adamw_ppo-epochs=2_B33x1.sh)
# with ONE change:
#   * 4 rollout + 4 trainer GPUs (was 5+3): one engine less generating, one
#     DP rank more training. 33*16=528 seqs divide by DP=4 (528/4=132), so
#     the mini-batch size is unchanged and the arm stays directly comparable.
#     The bf16 static shards down further: ~4.1 (bf16 params) + 4.1 (grads)
#     + 8.2 (bf16 moments) ~= 16 GB per trainer GPU (vs ~22 GB at DP=3); the
#     freed headroom is spent on a 40960-token dynbsz budget (4x, up from the
#     5+3 arm's 3x) — estimated peak ~62-72 GB, UNTESTED at this budget (see
#     the Batch Sizes section for fallbacks).
# Everything else inherited from the 5+3 bf16-SR arm:
#   * fsdp_config.model_dtype=bf16 + torchao _AdamW with bf16 stochastic
#     rounding, fully GPU-resident (no CPU offload, no fp32 master).
#     model_dtype=bf16 is REQUIRED: verl builds the FSDP actor in fp32 by
#     default, which quadruples static memory AND makes bf16_stochastic_round
#     silently inert (SR only acts on bf16 params). Stochastic rounding makes
#     master-less bf16 updates unbiased — at lr=1e-6 a deterministic bf16
#     update rounds to zero for ~98% of weights.
#   * NUMERICS: bf16-master character matches the Megatron HDO B33x1 arm
#     (main_params_dtype=bfloat16 there), but the mechanisms differ — SR with
#     bf16 moments here vs deterministic rounding with fp32 moments there,
#     and the fp32 FSDP2 siblings are a third regime (fp32 master). Treat all
#     cross-recipe comparisons as system-level, not ablations.
#   * Requires torchao in the environment.
#   * use_dynamic_bsz=True: micro-batches packed to a 40960-token budget
#     (4x max_model_len) — 1.5-3x faster updates than 1 seq per micro-batch,
#     plus ~10-20% over the 2x budget from fewer FSDP2 param all-gather
#     rounds (the 4x-over-3x increment is small, ~3-7%). Entropy
#     chunking+checkpointing and gradient checkpointing kept.
#   * ppo_epochs=2 (async_training.ppo_epochs): ONE 33-group mini-batch per
#     trainer step (require_batches=1, B-33x1), round(2*1)=2 AdamW updates —
#     two shuffled passes over the same 33 groups; second-epoch IS ratios
#     recomputed against the current policy per update (skip_recompute path —
#     works on FSDP via the dp_actor port,
#     tests/workers/actor/test_skip_recompute_old_log_prob_on_cpu.py).
#     Model versions tick per 33-group step.
#   * staleness_threshold=2, two validation sets, pearson logging.
#   * total_rollout_steps=66000 explicit (same generation budget as the
#     B-33x4 arms, up to ~2000 trainer steps of 33 groups); test_freq=10 /
#     save_freq=10 in param-version units = every 330 groups.
#   * OPPORTUNISTIC PPO EPOCHS OFF, DAPO FILTERING OFF.
#   * serialize_validation=True / pause_generation_during_save=True kept:
#     stop-the-world validation and checkpoint saves — pure time translations
#     excluded from cumulative_training_time.
#   * FSDP rollout-worker constraints (smoke-validated on the 6+2 FSDP2
#     config): gpu_memory_utilization=0.8 (the FSDP-path engine process sits
#     ~7-10 GB above the pool target; 0.9 OOMs at sampler warmup / weight
#     sync) and max_num_seqs=512.
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
# Do NOT export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: vLLM's
# sleep-mode allocator hard-asserts against it (vllm/device_allocator/cumem.py).
# The trainer worker enables it per-process in fsdp_workers.DetachActorWorker.

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
n_gpus_rollout=${n_gpus_rollout:-4}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# ================= Rollout =================
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
gen_tp=1
n_resp_per_prompt=${n_resp_per_prompt:-16}
# 0.8, not 0.9: FSDP-path engines sit ~7-10 GB above the pool target
# (6+2 FSDP2 smoke campaign, 2026-07-30); 0.9 OOMs at warmup/weight-sync.
gpu_memory_utilization=0.8
max_num_seqs=512
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
max_response_length=8192
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=33 # 33*16=528 seqs; must divide by trainer DP=4 (528/4=132)
micro_bsz_per_gpu=1      # ignored under use_dynamic_bsz=True
use_dynamic_bsz=True
# token budget per micro-batch: 4x max_model_len — the bf16 recipe's ~16 GB
# static at DP=4 (+~6 GB overhead) leaves ~58 GB for transients; at 40960 the
# estimated peak is ~62-72 GB. UNTESTED at this budget and sensitive to
# packing variance — relies on the trainer-scoped expandable segments. Fewer
# micro-batches = fewer FSDP2 param all-gather rounds; the gain over 3x is
# small (~3-7%). Fall back to 30720 (3x) or 20480 (2x, the smoke-validated
# budget) if it OOMs.
ppo_max_token_len=${ppo_max_token_len:-$((4 * (max_prompt_length + max_response_length)))} # 40960
log_prob_micro_bsz_per_gpu=1

bsz_per_dp_rank=33 # Rollout Bsz: in-flight prompt-groups per rollout replica (33*4 replicas = 132 concurrent)

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
# Log training/rollout_actor_probs_pearson_corr (exp of policy vs rollout
# log-probs over response tokens) from the deferred correction path
log_probs_pearson_corr=${log_probs_pearson_corr:-True}

skip_recompute_old_log_prob=True
compute_prox_log_prob=False

# ================= Async Training =================
# k=2 matches the VCPO baseline's staleness gating: the rollouter is
# licensed to generate up to (2+1) trainer batches ahead — at B-33x1 that is
# 99 in-flight/queued groups.
staleness_threshold=${staleness_threshold:-2.0}
updates_per_param_sync=1
num_minibatches_per_update=1 # require_batches=1: ONE 33-group mini-batch per trainer step (B-33x1); with ppo_epochs=2 that is round(2*1)=2 AdamW updates per step
partial_rollout=True
use_rollout_log_probs=True

# ================= Fractional scheduled PPO epochs =================
# round(ppo_epochs * require_batches) = 2 driver-side AdamW updates per trainer
# step: two shuffled passes over the single 33-group mini-batch of the pull.
ppo_epochs=${ppo_epochs:-2}
ppo_epochs_shuffle_seed=${ppo_epochs_shuffle_seed:-1234}

# ================= Elastic mechanisms OFF / stop-the-world accounting =================
# Opportunistic epochs and DAPO filtering are disabled in this ablation arm.
# serialize_validation / pause_generation_during_save freeze the pipeline
# during validation / checkpoint saves so cumulative_training_time and the
# trajectory match a no-validation-no-save run exactly.
dynamic_filtering_enable=${dynamic_filtering_enable:-False}
min_buffered_batches=${min_buffered_batches:-1.0}
opportunistic_enable=${opportunistic_enable:-False}
opportunistic_max_extra_epochs=${opportunistic_max_extra_epochs:-0}
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

# ================= Training/Rollout Steps =================
# Explicit 66000 (NOT the base arms' 500-step formula, which at B-33x1 would
# shrink to 500*1*1*33 = 16500): same generation budget as the B-33x4 arms,
# licensing up to ~2000 trainer steps of 33 groups.
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
# test/save freq are in param-version units; versions tick per 33-group step
# here, so 10 = every 330 groups — 2x the B-33x4 arms' 660-group cadence
# (their test_freq=5 at 132 groups/version).
test_freq=${test_freq:-10}
save_freq=${save_freq:-10}
max_actor_ckpt_to_keep=1 # keep only the most recent checkpoint

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO k-${staleness_threshold} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} fsdp2 dynbsz sr-adamw B-${train_prompt_mini_bsz}x${num_minibatches_per_update} ppo-epochs-${ppo_epochs} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
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
    algorithm.rollout_correction.log_probs_pearson_corr=${log_probs_pearson_corr} \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim \
    actor_rollout_ref.actor.optim.optimizer=_AdamW \
    "actor_rollout_ref.actor.optim.override_optimizer_config={bf16_stochastic_round:true}" \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
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
    async_training.ppo_epochs="${ppo_epochs}" \
    async_training.ppo_epochs_shuffle_seed="${ppo_epochs_shuffle_seed}" \
    async_training.serialize_validation="${serialize_validation}" \
    async_training.pause_generation_during_save="${pause_generation_during_save}" \
    +async_training.bsz_per_dp_rank="${bsz_per_dp_rank}" "$@"
