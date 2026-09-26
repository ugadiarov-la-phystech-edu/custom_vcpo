#!/usr/bin/env bash

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_ppo-default.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024_smoke.parquet']"}

exp_name=${exp_name:-"SMOKE-openpangu7b-ppo-default-5+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export n_gpus_rollout=${n_gpus_rollout:-5}

export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-4}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}
export total_rollout_steps=${total_rollout_steps:-6}
export test_freq=${test_freq:-1}
export save_freq=${save_freq:--1}
export val_before_train=${val_before_train:-False}

export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-80}
export gpu_memory_utilization=${gpu_memory_utilization:-0.43}

for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exec bash "${ARM_SCRIPT}" "$@" ;; esac
done

LOG_DIR="logs/${exp_name//\//_}"
mkdir -p -- "${LOG_DIR}"
CONSOLE_LOG="${LOG_DIR}/smoke_console.log"

start_time=$(date +%s)
set +e
bash "${ARM_SCRIPT}" "$@" 2>&1 | tee -- "${CONSOLE_LOG}"
arm_rc=${PIPESTATUS[0]}
set -e
set +x
elapsed=$(( $(date +%s) - start_time ))
echo "==================== smoke summary (${elapsed}s) ===================="

fail=0
check() {
    if [[ "$2" == 0 ]]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi
}
strip() { sed 's/\x1b\[[0-9;]*m//g' -- "${CONSOLE_LOG}"; }

check "arm exited 0 (rc=${arm_rc})" "$([[ ${arm_rc} -eq 0 ]]; echo $?)"
n_tb=$(strip | grep -c "Traceback (most recent call last)" || true)
check "no Python traceback (${n_tb} found)" "$([[ ${n_tb} -eq 0 ]]; echo $?)"
last_step=$(strip | grep -o "step:[0-9]* - " | grep -o "[0-9]*" | sort -n | tail -1 || true)
check "trainer reached step 2 (last logged step: ${last_step:-none})" "$([[ -n "${last_step}" && ${last_step} -ge 2 ]]; echo $?)"
n_clip=$(strip | grep -c "actor/pg_clipfrac:" || true)
check "PPO clip metric actor/pg_clipfrac logged (${n_clip} lines)" "$([[ ${n_clip} -ge 1 ]]; echo $?)"
n_val=$(strip | grep -o "val-core/math_dapo/acc/mean@1:[0-9][0-9.e-]*" | wc -l)
check "validation results logged (${n_val} x val-core/math_dapo/acc/mean@1, expect >= 2)" "$([[ ${n_val} -ge 2 ]]; echo $?)"
echo "  validation values:"
strip | grep "val-core/math_dapo/acc/mean@1:" \
    | sed -n 's/.*\(step:[0-9]*\) - .*\(val-core\/math_dapo\/acc\/mean@1:[0-9.e-]*\).*/    \1 \2/p' || true

if [[ "${save_freq}" -gt 0 ]]; then
    echo "==================== checkpoint verification ===================="
    if ! python "${HERE}/verify_checkpoints.py" "${LOG_DIR}" --expect 2 --dtype F32 --base-model "${MODEL_PATH}"; then
        echo "  FAIL  verify_checkpoints.py"; fail=1
    fi
fi

echo "console log: ${CONSOLE_LOG}"
if [[ ${fail} -eq 0 ]]; then echo "SMOKE PASS"; else echo "SMOKE FAIL"; exit 1; fi
