#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

BASE_SMOKE="${HERE}/smoke_test_oom_qwen3-8b_sync.sh"
[[ -f "${BASE_SMOKE}" ]] || { echo "no such base smoke: ${BASE_SMOKE}" >&2; exit 2; }

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
