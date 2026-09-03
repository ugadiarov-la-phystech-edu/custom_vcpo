#!/usr/bin/env bash

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
FULL_TEST_FILE=${FULL_TEST_FILE:-"/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet"}

exp_name=${exp_name:-"SMOKE-OOM-qwen3-8b-sync-B16xn16-mini8"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
SMOKE_DIR=${SMOKE_DIR:-"logs/smoke_oom"}
mkdir -p -- "${SMOKE_DIR}"

compose_only=0
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) compose_only=1 ;; esac
done

TINY_TEST_FILE="${SMOKE_DIR}/tiny_val.parquet"
if [[ "${compose_only}" == 0 ]]; then
    python - "${FULL_TEST_FILE}" "${TINY_TEST_FILE}" <<'PY'
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
pd.read_parquet(src).head(2).to_parquet(dst)
print(f"[smoke] wrote {dst} from {src}", file=sys.stderr)
PY
fi

export MODEL_PATH TRAIN_FILE exp_name
export TEST_FILE="['${TINY_TEST_FILE}']"
export train_prompt_bsz=${train_prompt_bsz:-16}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-8}
export n_resp_per_prompt=${n_resp_per_prompt:-16}
export max_response_length=${max_response_length:-8192}
export test_freq=${test_freq:-1000000}
export save_freq=${save_freq:-1000000}
export val_before_train=False
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}

RUN_LOG="${SMOKE_DIR}/${exp_name}.log"
GPU_CSV="${SMOKE_DIR}/${exp_name}_nvidia-smi.csv"

if [[ "${compose_only}" == 1 ]]; then
    bash "${ARM_SCRIPT}" trainer.total_training_steps=2 "$@"
    exit 0
fi

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits -l 2 \
    > "${GPU_CSV}" 2>/dev/null &
sampler_pid=$!
trap 'kill "${sampler_pid}" 2>/dev/null || true' EXIT

start_time=$(date +%s)
run_rc=0
bash "${ARM_SCRIPT}" trainer.total_training_steps=2 "$@" 2>&1 | tee "${RUN_LOG}" || run_rc=${PIPESTATUS[0]}
echo "[smoke] arm exited rc=${run_rc} after $(( $(date +%s) - start_time ))s" >&2

kill "${sampler_pid}" 2>/dev/null || true
wait "${sampler_pid}" 2>/dev/null || true

set +x
echo "==================== peak GPU memory (nvidia-smi, 2 s samples) ===================="
python - "${GPU_CSV}" <<'PY'
import collections
import sys

peak, total = collections.defaultdict(int), {}
n = 0
for line in open(sys.argv[1]):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        continue
    _, idx, used, tot = parts
    try:
        idx, used, tot = int(idx), int(used), int(tot)
    except ValueError:
        continue
    n += 1
    peak[idx] = max(peak[idx], used)
    total[idx] = tot
if not peak:
    print("WARN: no samples collected - was nvidia-smi available?")
    sys.exit(0)
print(f"{n} samples")
print("gpu  peak_used_MiB  total_MiB  headroom_MiB")
worst = None
for idx in sorted(peak):
    head = total[idx] - peak[idx]
    worst = head if worst is None else min(worst, head)
    print(f"{idx:>3}  {peak[idx]:>13}  {total[idx]:>9}  {head:>12}")
print(f"minimum headroom across GPUs: {worst} MiB")
if worst < 2048:
    print("WARN: less than 2 GiB of headroom on at least one GPU - the real run is at risk of OOM")
PY

echo "==================== OOM / memory-abort signatures in the driver log ===================="
if grep -n -E "OutOfMemoryError|CUDA out of memory|out of memory|less than desired GPU memory utilization|NCCL WARN Cuda failure|cudaErrorMemoryAllocation|calloc failed" "${RUN_LOG}" | head -20; then
    echo "FAIL: memory failure signature found in ${RUN_LOG}"
    exit 1
fi
echo "none found"

echo "==================== final checkpoint ===================="
CKPTS_DIR="logs/${exp_name//\//_}"
HF_DIR="${CKPTS_DIR}/global_step_2/actor/huggingface"
if [[ -d "${HF_DIR}" ]] && ls "${HF_DIR}"/*.safetensors >/dev/null 2>&1 && [[ -f "${HF_DIR}/config.json" ]]; then
    du -sh "${HF_DIR}"
    ls "${HF_DIR}"
    echo "hf_model checkpoint present: ${HF_DIR}"
else
    echo "FAIL: no complete hf_model checkpoint under ${HF_DIR} (the last-step save did not happen or died)"
    run_rc=${run_rc:-1}; [[ "${run_rc}" == 0 ]] && run_rc=1
fi

if [[ "${run_rc}" != 0 ]]; then
    echo "FAIL: arm exited with rc=${run_rc}; see ${RUN_LOG}"
    exit "${run_rc}"
fi
echo "PASS: 2 rollout steps (2 updates each) at 8192-token responses, final validation and hf_model save completed without a memory failure"
