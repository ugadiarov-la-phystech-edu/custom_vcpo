#!/usr/bin/env bash
# =============================================================================
# smoke_test_oom_openpangu7b_sync.sh
#
# Fastest end-to-end check of the synchronous openPangu-7B arm
#   main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh
# on the real 8-GPU colocated Megatron layout. A thin wrapper over
# smoke_test_oom_qwen3-8b_sync.sh (read its header for the protocol): the real arm
# script runs for 2 rollout steps with a 16-prompt batch, 8-prompt mini-batch (2
# optimizer updates per step), the arm's own 8192-token responses and memory
# settings, no validation before training, a 2-row final validation, one final
# hf_model save, an nvidia-smi sampler and an OOM-signature scan of the log.
#
# What this one adds, because the Megatron openPangu path carries production code
# the Qwen arm never touches (add_bias_linear from attention_bias, frozen MLP
# biases, o_proj.bias in the loader / vLLM sync / hf_model saver, BOS prepending):
#   * the final checkpoint is verified with verify_checkpoints.py --base-model:
#     parameter NAMES and shapes are diffed against the re-aliased checkpoint, so a
#     saver that lost the 34 o_proj.bias tensors fails here instead of at the first
#     offline eval ("parameter missing vs the base model"); dtype must be bf16 and
#     the tokenizer must round-trip through save_pretrained (custom PanguTokenizer).
#   * the driver log must show the frozen-MLP-bias message from DenseModel.initialize
#     (proof that add_bias_linear was derived and the freeze ran on the trainers) and
#     must NOT show a vLLM weight-loading complaint about unexpected/missing keys.
#   * HF_MODULES_CACHE is put on PYTHONPATH before anything starts: Ray workers
#     unpickle the trust_remote_code tokenizer by reference (see the arm header).
#
# Expected duration on 8xH100: ~10-20 min after model load, like the Qwen smoke.
#
# Usage:  bash smoke_test_oom_openpangu7b_sync.sh
#         gpu_memory_utilization=0.55 bash smoke_test_oom_openpangu7b_sync.sh   # probe the ceiling
# Env:    MODEL_PATH, TRAIN_FILE, FULL_TEST_FILE, SMOKE_DIR (default logs/smoke_oom),
#         gpu_memory_utilization, train_prompt_bsz, train_prompt_mini_bsz, and
#         everything the arm itself reads.
# =============================================================================

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

BASE_SMOKE="${HERE}/smoke_test_oom_qwen3-8b_sync.sh"
[[ -f "${BASE_SMOKE}" ]] || { echo "no such base smoke: ${BASE_SMOKE}" >&2; exit 2; }

# Ray workers deserialize the trust_remote_code tokenizer BY REFERENCE
# (transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer); the arm exports
# this too, but the base smoke writes its parquet before the arm runs, so do it here.
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

export ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh"}
export MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
export exp_name=${exp_name:-"SMOKE-OOM-openpangu7b-sync-B16xn16-mini8"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
export SMOKE_DIR=${SMOKE_DIR:-"logs/smoke_oom"}

# `--cfg job` and friends: the base smoke composes and exits, nothing to verify.
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exec bash "${BASE_SMOKE}" "$@" ;; esac
done

rc=0
bash "${BASE_SMOKE}" "$@" || rc=$?

RUN_LOG="${SMOKE_DIR}/${exp_name}.log"
CKPTS_DIR="logs/${exp_name}"

echo "==================== openPangu / Megatron specifics ===================="
fail=0
if [[ -f "${RUN_LOG}" ]]; then
    if grep -q "froze .* MLP bias tensors at zero" "${RUN_LOG}"; then
        echo "frozen MLP biases: $(grep -m1 -o 'froze [0-9]* MLP bias tensors at zero' "${RUN_LOG}")"
    else
        echo "FAIL: DenseModel.initialize never reported frozen MLP biases - add_bias_linear was not derived"
        fail=1
    fi
    if grep -n -E "Following weights were not initialized|unexpected keyword|KeyError: 'model\.layers\.[0-9]+\.(self_attn\.o_proj|mlp\.[a-z_]+)\.bias'" "${RUN_LOG}" | head -5; then
        echo "FAIL: vLLM/loader key mismatch on a bias tensor"
        fail=1
    fi
else
    echo "FAIL: no driver log at ${RUN_LOG}"
    fail=1
fi

if [[ -d "${CKPTS_DIR}" ]]; then
    # --expect 1: the base smoke saves only at the last step (global_step_2).
    # --no-timing-state: main_ppo writes no timing_state.json (fully-async trainer only).
    if python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 1 --dtype BF16 --base-model "${MODEL_PATH}" --no-timing-state; then
        echo "checkpoint verified against ${MODEL_PATH} (o_proj.bias present, bf16, tokenizer round-trips)"
    else
        echo "FAIL: verify_checkpoints.py rejected ${CKPTS_DIR}"
        fail=1
    fi
else
    echo "FAIL: no checkpoint dir ${CKPTS_DIR}"
    fail=1
fi

if [[ "${rc}" != 0 || "${fail}" != 0 ]]; then
    echo "FAIL: base smoke rc=${rc}, openPangu checks fail=${fail}; see ${RUN_LOG}"
    exit 1
fi
echo "PASS: openPangu-7B sync arm - 2 steps at 8192 tokens without a memory failure, o_proj.bias saved, MLP biases frozen"
