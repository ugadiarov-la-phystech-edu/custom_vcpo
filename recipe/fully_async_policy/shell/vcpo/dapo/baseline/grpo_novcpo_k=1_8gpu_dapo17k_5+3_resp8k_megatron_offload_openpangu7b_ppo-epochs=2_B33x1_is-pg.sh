#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-openpangu-megatron

# grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh
#
# openPangu-Embedded-7B arm of the MEGATRON 5+3 is-pg baseline: identical to
# ..._5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh (same data, objective,
# schedule, layout, HDO backend recipe) except for the model, the two trust_remote_code
# flags it needs, the BOS switch, and the experiment name. Its FSDP2 twin is
# ..._5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh; see BOS below for why the
# two openPangu arms are NOT a same-prompt comparison.
#
# THE MODEL. MODEL_PATH points at a LOCAL, RE-ALIASED checkpoint, not the hub id.
# openPangu ships as a trust_remote_code PanguEmbeddedForCausalLM whose
# modeling_openpangu_dense.py imports LossKwargs, removed in transformers >= 4.54 (this
# env runs 4.57.6), so AutoModelForCausalLM cannot build it; vLLM 0.11.0 also has no
# PanguEmbeddedForCausalLM. The generated modeling file is a `modular` derivative of
# Llama whose math-carrying functions are byte-for-byte Llama - the only substantive
# difference is Pangu folding attention bias into a single `bias` flag - so
# scripts/realias_openpangu_to_llama.py rewrites config.json to LlamaForCausalLM /
# model_type=llama with attention_bias=true, mlp_bias=false and no auto_map. Weight
# names need no remapping. transformers and vLLM then use their native Llama.
#   Build it with:
#     python scripts/realias_openpangu_to_llama.py --out <MODEL_PATH>
#
# MEGATRON AND THE ATTENTION BIAS (why this arm needed production code, unlike the FSDP2
# one). HF attention_bias=true means q/k/v AND o_proj carry a bias. Megatron-Core has no
# o_proj-only switch: add_qkv_bias covers the fused qkv, and add_bias_linear is one flag
# for linear_proj (o_proj) plus BOTH MLP projections. The stock converter hard-coded
# add_bias_linear=False and the legacy loader/converter/saver knew nothing about
# o_proj.bias, so strategy=megatron silently trained a DIFFERENT model (34 zeroed bias
# vectors) and synced it to vLLM as such. This branch now:
#   * derives add_bias_linear from the HF attention_bias/mlp_bias flags
#     (verl/models/mcore/config_converter.py::hf_to_mcore_config_dense);
#   * FREEZES the MLP biases add_bias_linear also creates, at their exact-zero TE init
#     (model_initializer.py::freeze_absent_mlp_biases, inside the model provider, i.e.
#     before DDP/optimizer construction so they never get a grad buffer or an optimizer
#     slot): the forward is then identical to HF Llama with mlp_bias=false. Left
#     trainable they would drift from update 1 and never reach vLLM or the checkpoint;
#   * loads o_proj.bias (loader.py), syncs it to vLLM every param version
#     (weight_converter.py) and EXPORTS it in hf_model checkpoints (saver.py) - the last
#     one is what makes global_step_N/actor/huggingface/ loadable by vLLM at all: a
#     Llama checkpoint with attention_bias=true and no o_proj.bias tensors is refused.
#   The MLP biases are skipped by the converter and the saver. Qwen2/Qwen3 arms are
#   untouched (attention_bias absent -> add_bias_linear stays False).
#
# BOS (data.add_bos_token_to_prompt=True, the ONE prompt-level difference from every
# other arm on this branch). The Pangu tokenizer has add_bos_token=true and a chat
# template that never emits <s>. The official inference recipe (README) is
# apply_chat_template(tokenize=False) followed by a plain tokenizer(text) call, which
# PREPENDS <s> (id 1): every prompt the model was built for starts with BOS. verl's
# default path tokenizes with add_special_tokens=False / apply_chat_template(tokenize=True)
# and drops it (measured on the checkpoint: 13 vs 14 tokens for a one-line user turn).
# With the flag on, RLHFDataset (input_ids, raw_prompt_ids, the overlong filter) and the
# single-turn agent loops all go through verl/utils/dataset/prompt_utils.py::
# maybe_prepend_bos, so training, validation and prompt-length accounting agree.
#   * NOT comparable with the FSDP2 openPangu arm, which trained WITHOUT BOS.
#   * Offline eval of these checkpoints must reproduce the BOS: vLLM's chat-completions
#     endpoint defaults add_special_tokens=False (no BOS); use the completions endpoint
#     with the rendered template, or pass token ids.
#   * The flag is a no-op for Qwen (no BOS token) and guards against templates that
#     already start with the BOS string (Llama-3/Mistral), so it cannot double a BOS.
#
# WHY trust_remote_code IS STILL REQUIRED: only the MODELING entries left config.json's
# auto_map. tokenizer_config.json keeps its own, so the tokenizer is still PanguTokenizer
# from tokenization_openpangu.py (a slow tokenizer; verl's default use_fast resolves to it
# because no fast variant exists). The two keys are INDEPENDENT and nothing links them:
# data.trust_remote_code feeds the dataset-side tokenizer (fully_async_main.py), while
# actor_rollout_ref.model.trust_remote_code feeds HFModelConfig - the agent-loop
# tokenizer, the actor/ref weight load AND the vLLM engine. Setting one and not the other
# crashes the other half of the system.
#
# VERIFIED ON THE REAL CHECKPOINT (CPU probes, 2026-08-23; the model-dependent half lives
# in tests/models/test_openpangu_tokenizer_contract_on_cpu.py, run it with
# OPENPANGU_MODEL_PATH=<MODEL_PATH>):
#   * chat template renders [unused9]系统：[unused10][unused9]用户：...[unused10][unused9]助手：
#     SLOW THINK IS THE DEFAULT - no /no_think or /auto_think suffix is injected - and
#     add_generation_prompt=True only appends, which is what the agent loop assumes.
#   * eos = [unused10] = 45892 < len(tokenizer) = 153376, so verl's vLLM logit mask
#     (vllm_rollout/utils.py masks logits[..., len(tokenizer):]) cannot make EOS
#     unsamplable. verl passes no stop/eos to vLLM; termination is the model's own eos.
#   * prompt token lengths under THIS tokenizer: dapo-math-17k 75/135/500,
#     aime-2024 106/158/261, aime-2025 91/172/795 (min/median/max), +1 for BOS here.
#     Nothing approaches max_prompt_length=2048, so filter_overlong_prompts drops no rows.
#   * ANSWER EXTRACTION: math_dapo is format-agnostic - it takes solution_str[-300:] and
#     the LAST (?i)Answer\s*:\s*(...) match - and needs no thinking-delimiter handling.
#     [unused16]/[unused17] are not special tokens and DO survive the reward decode.
#   * THE ONE LIVE RISK: a verbose epilogue AFTER the answer pushes it out of the 300-char
#     window and scores -1.0 with pred='[INVALID]'. WATCH THE [INVALID] RATE in the first
#     validation.
#
# ARM A of the pair that reproduces the same-named arm of branch 'rollout-dapo' on
# this (pristine-verl + cumulative_training_time) branch: IS-weighted policy gradient
# with token-level truncated IS at 2.0 and no trust region -
#   actor.policy_loss.loss_mode=rollout_correction
#     -> compute_policy_loss_with_rollout_correction (verl/trainer/ppo/core_algos.py):
#        L = -E[w * log pi * A], w = trunc(pi_theta/pi_rollout, rollout_is_threshold),
#        computed on the fly per micro-batch, NO PPO clipping, no extra forward pass.
# use_rollout_log_probs=True keeps old_log_probs := rollout_log_probs. See the Qwen
# twin's header for the full objective/log-surface discussion; it applies verbatim.
#
# Schedule (identical to the Qwen twin):
#   * require_batches=1: the pull is ONE 33-group mini-batch per trainer step (B-33x1).
#   * TWO AdamW updates per step (actor.ppo_epochs=2; megatron_workers.py scales
#     ppo_mini_batch_size by rollout.n, 33*16=528 = the whole pull).
#   * staleness_threshold=1, total_rollout_steps=66000, test_freq=save_freq=10,
#     stop-the-world validation and saves.
#   * validation sampling 0.8/0.7, as in the Qwen twin (NOT the FSDP2 openPangu arm's
#     1.0/0.8), so val-core/math_dapo is comparable with the Qwen Megatron arm's.
#
# CHECKPOINTS (save_contents=['hf_model'], max_actor_ckpt_to_keep=null, resume_mode=disable):
#   * each save writes global_step_N/actor/huggingface/ - config, tokenizer and bf16
#     safetensors INCLUDING the 34 o_proj.bias tensors - directly loadable by vLLM /
#     from_pretrained. ~16 GB per save (8.0B params in bf16; half the FSDP2 arm's fp32
#     32 GB); nothing is rotated away, ~200 saves is ~3.2 TB. Raise save_freq if tight.
#   * the run is NOT resumable ('hf_model' is written but never read back);
#     resume_mode=disable makes that explicit.
#
# Base-script notes that still apply: trainer tp=1/dp=3 (sequence_parallel needs
# TP>1), 33*16=528 seqs divide by DP=3, HDO full CPU offload with bf16 master weights
# (do NOT swap for use_precision_aware_optimizer without optimizer_cpu_offload:
# silent stall, probe 2026-07-30). openPangu-7B: 34 layers, hidden 4096, FFN 12800,
# 32 q / 8 kv heads, vocab 153376 - the same shape budget as Qwen3-8B, so the Qwen
# twin's memory envelope carries over. calculate_entropy=True clones the logits on the
# non-fused megatron path (~3 GB at 10k tokens x 153k vocab); if the trainer OOMs, set
# use_fused_kernels=True or calculate_entropy=False.
#
# GPU CHECKS STILL OWED before a full run (CPU tests cannot cover them):
#   bash smoke_test_openpangu_megatron_3+3.sh  -> 2 steps, 2 checkpoints, and
#   verify_checkpoints.py diffing parameter NAMES against MODEL_PATH (o_proj.bias must be
#   present, bf16 throughout); then confirm rollout_corr/* KL in the first steps sits at
#   the Qwen arms' level - a wrong or missing o_proj bias shows up as a large
#   trainer-vs-vLLM KL immediately.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0

export PYTHONUNBUFFERED=1

# ================= Paths =================
# Ray workers deserialize the trust_remote_code tokenizer BY REFERENCE, as
# transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer. That dynamic package
# only lands on sys.path in a process that has itself loaded remote code: the driver has,
# the actors have not, so FullyAsyncRollouter.__init__ dies unpickling its own constructor
# arguments with "ModuleNotFoundError: No module named 'transformers_modules'" - before any
# GPU work, and with no hint that the tokenizer is at fault (verified on remote_smoke,
# 2026-08-23; reproduced with a 20-line ray script and fixed by exactly this line). Ray
# workers inherit this environment, which makes the reference resolvable everywhere.
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;  # already there (e.g. the wrapper set it)
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# A LOCAL, RE-ALIASED copy - not the hub id. See THE MODEL in the header and
# scripts/realias_openpangu_to_llama.py.
MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
# Still required after the re-alias, because the TOKENIZER stays custom code. Both keys
# are needed; see WHY trust_remote_code IS STILL REQUIRED in the header.
trust_remote_code=${trust_remote_code:-True}
# Prepend <s> to every prompt, as the official openPangu recipe does. See BOS in the header.
add_bos_token_to_prompt=${add_bos_token_to_prompt:-True}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# aime-2024 (data_source=math_dapo -> val-core/math_dapo/acc/mean@1) and
# aime-2025 (data_source=aime2025_dapo -> val-core/aime2025_dapo/acc/mean@1).
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
# Same as the Qwen Megatron twin: the trainer lives on its own GPUs and the bf16 weight
# sync is half the FSDP2 arm's fp32 broadcast, so the FSDP2 arm's 0.75 is not needed.
gpu_memory_utilization=${gpu_memory_utilization:-0.8}
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
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
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33} # 33*16=528 seqs; must divide by trainer DP=3 (528/3=176)
micro_bsz_per_gpu=1
use_dynamic_bsz=False
log_prob_micro_bsz_per_gpu=1

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
entropy_coeff=${entropy_coeff:-0}
# Log actor/entropy even with entropy_coeff=0 (honoured via should_calculate_entropy
# in verl/workers/actor/megatron_actor.py). This arm has no old-log-prob forward, so
# it is the only source of entropy here.
calculate_entropy=True
grad_clip=1.0

# ================= Optimizer =================
lr=${lr:-1e-6}
lr_warmup_steps=0
weight_decay=0.1

# ================= IS / Rollout Correction =================
# Token-level truncated IS at 2.0, applied as a pure policy-gradient correction with
# no PPO clipping. bypass_mode/use_policy_gradient describe exactly this mode, but note
# they are INERT on the fully-async path: their only consumer, apply_rollout_correction(),
# is called from verl/trainer/ppo/ray_trainer.py, never from this recipe. What actually
# selects the behaviour is policy_loss.loss_mode below; the recipe performs the bypass
# substitution itself via async_training.use_rollout_log_probs=True.
rollout_is="token"
rollout_is_threshold="2.0"
rollout_rs=null
rollout_rs_threshold=null
bypass_mode=True
use_policy_gradient=True
policy_loss_mode="rollout_correction"

compute_prox_log_prob=False

# ================= Async Training =================
# k=1: the rollouter is licensed to generate up to (1+1) trainer batches ahead
# — at B-33x1 that is 66 in-flight/queued groups.
staleness_threshold=${staleness_threshold:-1.0}
updates_per_param_sync=1
num_minibatches_per_update=1 # require_batches=1: ONE 33-group mini-batch per trainer step (B-33x1)
partial_rollout=True
use_rollout_log_probs=True

# ================= PPO epochs =================
ppo_epochs=${ppo_epochs:-2}

# ================= Stop-the-world accounting =================
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

# ================= Training/Rollout Steps =================
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
test_freq=${test_freq:-10}
save_freq=${save_freq:-10}
# Weights only, in huggingface format, ~16 GB bf16 per save, nothing rotated away.
save_contents=${save_contents:-"['hf_model']"}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-null} # keep every checkpoint
resume_mode=${resume_mode:-disable}

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO is-pg k-${staleness_threshold} DAPO17K-AIME24 openPangu-7B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz}x${num_minibatches_per_update} ppo-epochs-${ppo_epochs} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd bos"}
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
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
    "+actor_rollout_ref.actor.policy_loss.rollout_correction={rollout_is:${rollout_is},rollout_is_threshold:${rollout_is_threshold},rollout_rs:${rollout_rs},rollout_rs_threshold:${rollout_rs_threshold}}" \
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
    trainer.resume_mode=${resume_mode} \
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}" \
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
    async_training.serialize_validation="${serialize_validation}" \
    async_training.pause_generation_during_save="${pause_generation_during_save}" "$@"
