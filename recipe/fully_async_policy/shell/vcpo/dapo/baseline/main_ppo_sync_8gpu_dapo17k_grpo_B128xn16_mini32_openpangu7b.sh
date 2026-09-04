#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-openpangu7b

# SYNCHRONOUS openPangu-Embedded-7B reference arm: verl.trainer.main_ppo (the plain
# colocated hybrid-engine MEGATRON trainer, NOT recipe/fully_async_policy), GRPO
# without a critic, the STOCK PPO LOSS WITH VERL'S DEFAULT PARAMETERS, on
# DAPO-Math-17k. Byte-identical in every schedule/optimizer/geometry setting to the
# Qwen3-8B twin main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh; the
# differences are the model, the two trust_remote_code flags it needs, the BOS
# switch, and the experiment name. Read the twin's header for LOSS / GEOMETRY /
# SEEDS / ENTROPY WATCH; only what differs is documented here.
#
# THE MODEL. MODEL_PATH points at a LOCAL, RE-ALIASED checkpoint, not the hub id:
# openPangu's remote modeling code does not import under transformers 4.57.6 and
# vLLM 0.11.0 has no PanguEmbeddedForCausalLM, so scripts/realias_openpangu_to_llama.py
# rewrites config.json to LlamaForCausalLM with attention_bias=true, mlp_bias=false
# and no modeling auto_map (weights need no remapping). The tokenizer stays the
# custom PanguTokenizer, hence BOTH trust_remote_code keys below: data.trust_remote_code
# (dataset tokenizer) and actor_rollout_ref.model.trust_remote_code (agent-loop
# tokenizer, Megatron weight load, vLLM engine). They are independent; setting one and
# not the other crashes the other half of the system.
#
# MEGATRON AND THE ATTENTION BIAS. attention_bias=true gives o_proj a bias, which
# Megatron only allocates through add_bias_linear (one flag for o_proj AND both MLP
# projections). This branch derives add_bias_linear from the HF config
# (config_converter.py), freezes the extra MLP biases at their zero init before DDP /
# optimizer construction (model_initializer.py::freeze_absent_mlp_biases), loads and
# syncs o_proj.bias to vLLM (loader.py, weight_converter.py) and exports it in hf_model
# checkpoints (saver.py). Forward == HF Llama with mlp_bias=false; checkpoints load in
# vLLM as-is. See the Megatron is-pg openPangu arm's header for the long version.
#
# BOS (data.add_bos_token_to_prompt=True). The Pangu tokenizer has add_bos_token=true
# and a template that never emits <s>; the official recipe tokenizes the rendered
# template with a plain tokenizer(text) call, so real prompts start with <s> (id 1).
# verl's default drops it (13 vs 14 tokens measured). The flag routes the dataset and
# the agent loop through prompt_utils.maybe_prepend_bos so training, validation and the
# overlong filter agree. Offline eval must reproduce it (vLLM chat-completions defaults
# add_special_tokens=False; use completions with the rendered template, or token ids).
# Not comparable with the FSDP2 openPangu arm, which trained without BOS.
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
# ~16 GB bf16 per save (8.0B params) including the 34 o_proj.bias tensors; at
# save_freq=2 over the 135-step epoch that is 67 saves, ~1.1 TB per epoch. Not
# resumable (hf_model is never read back).
#
# MEMORY - NOT YET MEASURED FOR THIS MODEL. The Qwen3-8B twin's envelope is assumed to
# carry over and must be confirmed by an OOM smoke (2 steps at 8192 tokens with an
# nvidia-smi sampler, as smoke_test_oom_qwen3-8b_sync.sh does) BEFORE a full run:
#   * openPangu-7B: 34 layers x 8 KV heads x 128 head_dim = 139,264 B of KV per token
#     (Qwen3-8B: 147,456), 8.0B params (Qwen3-8B 8.2B), vocab 153,376 (151,936). Same
#     shape budget, so the resident trainer (~35 GB after init) and the 0.5 vLLM
#     ceiling without param offload are expected to match the twin's measurements
#     (peak 78.1 GB during vLLM profiling, 3.5 GB headroom - the tightest point).
#   * max_num_batched_tokens=10240 (= max_model_len) bounds prefill activations only.
#   * Never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
#
# VALIDATION SAMPLING is the twin's 0.8/0.7/n=1, so val-core/math_dapo (aime-2024) and
# val-core/aime2025_dapo curves are comparable with the Qwen3-8B sync arm and with the
# Megatron is-pg openPangu arm (also 0.8/0.7); NOT with the FSDP2 openPangu arm (1.0/0.8,
# and no BOS).

set -x
export VLLM_USE_V1=1
# vLLM 0.11 auto-selects the FlashInfer sampler when flashinfer is importable and
# JIT-compiles it with nvcc at engine init; remote_h100 has no nvcc. Native sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

# ================= Paths =================
# Ray workers deserialize the trust_remote_code tokenizer BY REFERENCE, as
# transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer; that dynamic package
# is only on sys.path in a process that has itself loaded remote code. Exporting the HF
# modules cache on PYTHONPATH makes the reference resolvable in every Ray worker
# (verified on remote_smoke, 2026-08-23, for the fully-async arm; main_ppo's workers
# unpickle the same tokenizer).
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# A LOCAL, RE-ALIASED copy - not the hub id (scripts/realias_openpangu_to_llama.py).
MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
# Both keys, see THE MODEL in the header.
trust_remote_code=${trust_remote_code:-True}
# Prepend <s> to every prompt, as the official openPangu recipe does. See BOS in the header.
add_bos_token_to_prompt=${add_bos_token_to_prompt:-True}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# aime-2024 (data_source=math_dapo -> val-core/math_dapo/acc/mean@1) and
# aime-2025 (data_source=aime2025_dapo -> val-core/aime2025_dapo/acc/mean@1).
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}

# ================= Seeds =================
# Every seed knob main_ppo exposes (see the Qwen twin's SEEDS block). vLLM's sampling
# seed is not among them: it is hard-wired to 0 by RolloutConfig.
SEED=${SEED:-1}

# ================= Data =================
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
filter_overlong_prompts=True
truncation='left'

# ================= Batch geometry (4 optimizer steps per rollout step) =================
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
# 0.5 is the ceiling with the resident (non-offloaded) trainer on the Qwen3-8B twin;
# assumed to carry over (see MEMORY). Raise only together with param_offload=True.
gpu_memory_utilization=${gpu_memory_utilization:-0.5}
rollout_tp=1
enable_chunked_prefill=True
max_num_batched_tokens=$((1024 * 10))
temperature=1.0
top_p=1.0
top_k=-1
# Validation sampling identical to the Qwen3-8B twin and the Megatron is-pg openPangu arm.
val_temperature=${val_temperature:-0.8}
val_top_p=${val_top_p:-0.7}
# Cache vLLM's per-token log-probs for the rollout_corr/* diagnostics; no correction
# is applied (rollout_is null, bypass_mode false: old_log_probs recomputed by the trainer).
calculate_log_probs=True

# ================= Trainer =================
test_freq=${test_freq:-2}    # rollout steps (= 8 optimizer updates)
save_freq=${save_freq:-2}    # rollout steps
total_epochs=${total_epochs:-3}
val_before_train=${val_before_train:-True}
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
# Mandatory, not cosmetic: 'hf_model' is written but never read back.
resume_mode=${resume_mode:-disable}

NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

# ================= Logging =================
exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn${n_resp_per_prompt} mini-${train_prompt_mini_bsz} ppo-epochs-${ppo_epochs} DAPO17K-AIME24-25 openPangu-7B tp${train_tp}dp${n_gpus_per_node} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd bos seed-${SEED}"}
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
    data.trust_remote_code=${trust_remote_code} \
    data.add_bos_token_to_prompt=${add_bos_token_to_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=${trust_remote_code} \
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
