#!/usr/bin/env bash

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }
arm_tag=$(basename -- "${ARM_SCRIPT}" .sh)
arm_tag=${arm_tag//[^A-Za-z0-9-]/-}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
FULL_TEST_FILE=${FULL_TEST_FILE:-"/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet"}

exp_name=${exp_name:-"SMOKE-${arm_tag}"}
SMOKE_DIR=${SMOKE_DIR:-"logs/smoke_ckpt"}
mkdir -p -- "${SMOKE_DIR}"

TINY_TEST_FILE="${SMOKE_DIR}/tiny_val.parquet"
python - "${FULL_TEST_FILE}" "${TINY_TEST_FILE}" <<'PY'
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
pd.read_parquet(src).head(2).to_parquet(dst)
print(f"[smoke] wrote {dst} from {src}")
PY

export MODEL_PATH TRAIN_FILE
export TEST_FILE="['${TINY_TEST_FILE}']"
export n_resp_per_prompt=2
export train_prompt_mini_bsz=3
export max_response_length=512
export total_rollout_steps=6
export save_freq=1
export test_freq=1000000
export val_before_train=False
export ppo_epochs=2
export entropy_coeff=${entropy_coeff:-0.01}
export lr=${lr:-1e-4}
export exp_name

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s"

CKPTS_DIR="logs/${exp_name//\//_}"
set +x
echo "==================== checkpoint verification ===================="
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 3 --base-model "${MODEL_PATH}"
