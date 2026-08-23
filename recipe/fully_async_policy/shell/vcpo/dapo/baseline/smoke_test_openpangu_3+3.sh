#!/usr/bin/env bash

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet']"}

exp_name=${exp_name:-"SMOKE-openpangu7b-3+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
export n_gpus_rollout=${n_gpus_rollout:-3}

export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}
export total_rollout_steps=${total_rollout_steps:-6}
export test_freq=${test_freq:-1}
export save_freq=${save_freq:-1}
export val_before_train=${val_before_train:-False}
export ppo_epochs=${ppo_epochs:-2}
export entropy_coeff=${entropy_coeff:-0.01}
export lr=${lr:-1e-4}

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exit 0 ;; esac
done

CKPTS_DIR="logs/${exp_name//\//_}"
set +x
echo "==================== checkpoint verification ===================="
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 2 --dtype F32 --base-model "${MODEL_PATH}"
