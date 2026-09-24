#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-openpangu7b-fsdp2

# SYNCHRONOUS openPangu-Embedded-7B reference arm on FSDP2: verl.trainer.main_ppo (the
# plain colocated hybrid-engine trainer, NOT recipe/fully_async_policy), GRPO without a
# critic, the STOCK PPO LOSS WITH VERL'S DEFAULT PARAMETERS, on DAPO-Math-17k. The FSDP2
# twin of main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh (Megatron):
# identical data, schedule, batch geometry, optimizer hyper-parameters, loss, BOS,
# rollout and validation settings, seeds and cadence; only the training backend and its
# precision/offload knobs differ, plus the experiment name (" fsdp2-dp8" instead of
# " tp1dp8"). Read the Megatron twin's header (and the Qwen3-8B twin's for LOSS /
# GEOMETRY / SEEDS / ENTROPY WATCH); only what differs is documented here.
#
# THE MODEL. Same LOCAL, RE-ALIASED checkpoint as the Megatron twin
# (scripts/realias_openpangu_to_llama.py: config.json -> LlamaForCausalLM with
# attention_bias=true, mlp_bias=false, no modeling auto_map; the tokenizer stays the
# custom PanguTokenizer, hence BOTH trust_remote_code keys below).
#
# FSDP2 AND THE ATTENTION BIAS. Nothing to do on this backend: FSDP2 trains the HF model
# itself, and HF LlamaForCausalLM with attention_bias=true already gives q/k/v/o their
# biases and creates no MLP biases, so the trained parameter set is exactly the
# checkpoint's (the Megatron-only work - add_bias_linear, freezing the extra MLP biases,
# o_proj.bias in loader / weight_converter / saver - has no FSDP counterpart). The
# remove-padding monkey patch is the generic Llama one, the FSDP2 wrap unit is
# LlamaDecoderLayer, and weight sync to vLLM goes through vLLM's own
# LlamaForCausalLM.load_weights (which reads attention_bias for o_proj).
#
# PRECISION: bf16 compute, fp32 master weights and fp32 AdamW state - verl's standard FSDP
# mixed precision, set explicitly: fsdp_config.model_dtype=fp32 (the parameters the
# optimizer updates) + fsdp_config.mixed_precision param_dtype=bf16 (forward/backward),
# reduce_dtype=fp32 (gradient reduce-scatter), buffer_dtype=fp32. Rollout (vLLM) is bf16 as
# on the twin. The optimizer math matches the Megatron twin's: that arm keeps bf16 weights
# on the GPU and runs Adam on an fp32 CPU copy with fp32 moments (its hybrid CPU optimizer
# is built with param_update_in_fp32=True); here the fp32 shard IS the master. The one
# numeric difference left is the gradient reduction: fp32 here, bf16 on the twin
# (grad_reduce_in_fp32=False). model_dtype=bf16 is deliberately NOT used: with plain AdamW
# it makes the moments and the update itself bf16 (why the sr-adamw arm needs stochastic
# rounding).
#
# CPU UPDATE (fsdp_config.offload_policy=True, the default): FSDP2's CPUOffloadPolicy keeps
# the actor's sharded fp32 parameters and gradients in pinned CPU memory, so the AdamW
# optimizer is built on CPU tensors and the step - fp32 master, fp32 moments - runs on the
# CPU, as on the Megatron twin. The GPU then holds no persistent fp32 training state, only
# the bf16 layer being computed and the activations. The price differs from Megatron's:
# Megatron keeps bf16 weights resident on the GPU and transfers once per optimizer step,
# while FSDP2 copies each layer's fp32 shard host-to-device at EVERY forward and backward
# (every micro-batch of every update and of every log-prob pass) and moves every
# micro-batch's gradient shard back to the CPU. Expect a slower trainer phase; measure it
# in the smoke. offload_policy=False keeps everything on the GPU (the fallbacks under
# MEMORY then apply). verl disables param_offload / optimizer_offload under this policy
# (the parameters and optimizer state already live on the CPU).
# Costs of the fp32 master: hf_model checkpoints are fp32 (~32 GB per save, vs ~16 GB on
# the twin; get_fsdp_full_state_dict does not cast) and weight sync to vLLM ships fp32
# tensors (vLLM casts them on load).
#
# BOS (data.add_bos_token_to_prompt=True). The Pangu tokenizer has add_bos_token=true
# and a template that never emits <s>; the official recipe tokenizes the rendered
# template with a plain tokenizer(text) call, so real prompts start with <s> (id 1).
# verl's default drops it (13 vs 14 tokens measured). The flag routes the dataset and
# the agent loop through prompt_utils.maybe_prepend_bos so training, validation and the
# overlong filter agree. Offline eval must reproduce it (vLLM chat-completions defaults
# add_special_tokens=False; use completions with the rendered template, or token ids).
# The fully-async FSDP2 openPangu arm now trains with BOS too (its earlier runs did not).
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
# ~32 GB fp32 per save (8.0B params, see PRECISION) including the 34 o_proj.bias tensors;
# at save_freq=2 over the 135-step epoch that is 67 saves, ~2.1 TB per epoch - check the
# disk, or raise save_freq. Not resumable (hf_model is never read back). Verify a save with
# verify_checkpoints.py --dtype F32.
#
# MEMORY - NOT YET MEASURED. 8.0B params sharded over 8 GPUs (fsdp_size=-1): fp32 master
# + fp32 grads + 2 fp32 AdamW moments = 16 B/param = ~16 GB per GPU. With the default CPU
# UPDATE all of it sits in pinned host memory instead (~16 GB per trainer process, ~128 GB
# for the node), and the GPU holds the bf16 all-gathered layer and activations (gradient
# checkpointing on). vLLM keeps the twin's 0.5 fraction. Confirm with a short smoke
# (max_updates=4 test_freq=1 save_freq=1) before a full run. With offload_policy=False the
# fallbacks if the trainer OOMs are, in order: optimizer_offload=True (moments to CPU
# between phases, frees ~8 GB/GPU), then param_offload=True, then
# model.use_fused_kernels=True.
# Never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
#
# VALIDATION SAMPLING is the twin's 0.8/0.7/n=1, so val-core/math_dapo (aime-2024) and
# val-core/aime2025_dapo curves are comparable with the Megatron twin, the Qwen3-8B sync
# arm, and both fully-async openPangu arms (all 0.8/0.7).

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

# ================= Parallelism / precision (FSDP2) =================
# Shard params / grads / optimizer state across all 8 GPUs; no sequence parallelism.
fsdp_size=-1
sp_size=1
reshard_after_forward=True
# fp32 master weights and AdamW state, bf16 compute (see PRECISION in the header).
# Seeds: fsdp_config.seed takes the place of the twin's megatron.seed; the twin's
# actor.data_loader_seed has no FSDP counterpart (the FSDP actor splits mini-batches in
# order, actor.shuffle=False), so data.seed is the only data-order seed here.
fsdp_model_dtype=fp32
mp_param_dtype=bf16
mp_reduce_dtype=fp32
mp_buffer_dtype=fp32
# CPU update: FSDP2 CPUOffloadPolicy for the actor (see CPU UPDATE in the header).
offload_policy=${offload_policy:-True}
param_offload=${param_offload:-False}         # ignored by verl while offload_policy=True
optimizer_offload=${optimizer_offload:-False} # ignored by verl while offload_policy=True
ref_param_offload=True # no reference model in use (use_kl_loss=False); kept offloaded
enable_gradient_checkpointing=True # the twin's full / uniform / 1-layer recompute
precision_dtype=bfloat16 # rollout (vLLM) dtype
use_remove_padding=True

# ================= Rollout =================
rollout_name=vllm
rollout_mode=async
# The Megatron twin's value; not yet measured on FSDP2 (see MEMORY). Raise only together
# with param_offload=True.
gpu_memory_utilization=${gpu_memory_utilization:-0.5}
# H100 emulation on bigger cards (verl/utils/gpu_memory_cap.py): exporting VERL_GPU_MEM_CAP_GB=80
# in the launching shell caps the TRAINER worker allocator at 80 GiB (actor role only; the vLLM
# servers are separate processes bounded by gpu_memory_utilization). Pair it with the fraction
# that gives vLLM the same absolute budget as on an 80 GiB H100: gpu_memory_utilization =
# 0.5*80/<device GiB> (0.28 on a 143.8 GiB H200). The knob is only READ here, never set. Tagged
# in exp_name so emulated runs never share a log dir with real ones.
emu_tag=""
if [[ -n "${VERL_GPU_MEM_CAP_GB:-}" ]]; then emu_tag=" h100-emu-${VERL_GPU_MEM_CAP_GB}gb-gmu${gpu_memory_utilization}"; fi
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
# Cap on OPTIMIZER UPDATES (null = none: total_epochs decides). One rollout step runs
# updates_per_step = train_prompt_bsz / ppo_mini_batch_size * ppo_epochs updates, so the
# cap is rounded UP to whole rollout steps and passed as verl's trainer.total_training_steps:
# that step runs validation + a checkpoint save regardless of test_freq / save_freq and ends
# the run (RayPPOTrainer.is_last_step). total_epochs still bounds it from above (a cap beyond
# the epoch budget is never reached). LR is constant with no warmup: unaffected. A trailing
# trainer.total_training_steps=N on the command line (the OOM smoke) still overrides this.
#   export max_updates=200; bash <this script>
max_updates=${max_updates:-null}
updates_per_step=$(( train_prompt_bsz / train_prompt_mini_bsz * ppo_epochs ))
(( updates_per_step >= 1 )) || { echo "updates_per_step must be >= 1 (train_prompt_bsz / train_prompt_mini_bsz * ppo_epochs)" >&2; exit 2; }
if [[ "${max_updates}" == "null" ]]; then
    total_training_steps=null
else
    [[ "${max_updates}" =~ ^[1-9][0-9]*$ ]] || { echo "max_updates must be a positive integer or null, got '${max_updates}'" >&2; exit 2; }
    total_training_steps=$(( (max_updates + updates_per_step - 1) / updates_per_step ))
fi
val_before_train=${val_before_train:-True}
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
# Mandatory, not cosmetic: 'hf_model' is written but never read back.
resume_mode=${resume_mode:-disable}

NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

# ================= Logging =================
exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn${n_resp_per_prompt} mini-${train_prompt_mini_bsz} ppo-epochs-${ppo_epochs} DAPO17K-AIME24-25 openPangu-7B fsdp2-dp${n_gpus_per_node} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd bos seed-${SEED}${emu_tag}"}
exp_name_safe=${exp_name//\//_}
log_dir="logs/${exp_name_safe}"
CKPTS_DIR="${log_dir}"
mkdir -p -- "${log_dir}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer.yaml \
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
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
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
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.seed=${SEED} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${reshard_after_forward} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${offload_policy} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${fsdp_model_dtype} \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=${mp_param_dtype} \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=${mp_reduce_dtype} \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=${mp_buffer_dtype} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
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
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=${total_epochs} "$@"
