#!/usr/bin/env bash
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

UV_ENV_ACTIVATE=${UV_ENV_ACTIVATE:-/data/homes/ugadiarov_la/ugadiarov.la/uv-envs/vcpo-env/bin/activate}
if [[ -f "${REPO_ROOT}/activate.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/activate.sh"
elif [[ -f "${UV_ENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1091
    source "${UV_ENV_ACTIVATE}"
fi
cd -- "${REPO_ROOT}"
python -c "import ray, hydra" 2>/dev/null || {
    echo "vcpo uv environment is not active: no activate.sh in ${REPO_ROOT} and no ${UV_ENV_ACTIVATE}" >&2
    echo "(python: $(command -v python || echo none))" >&2
    exit 2
}

ARM_SCRIPT="${HERE}/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh"
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT} (not a baselines_main-ppo_openpangu checkout?)" >&2; exit 2; }
grep -q 'max_updates=' "${ARM_SCRIPT}" || { echo "arm script lacks the max_updates knob" >&2; exit 2; }
grep -q '"math500_dapo"' "${REPO_ROOT}/verl/utils/reward_score/__init__.py" || {
    echo "verl/utils/reward_score/__init__.py has no math500_dapo route: validation on ${TEST_FILE:-math500.parquet} would raise NotImplementedError." >&2
    echo "port commit 2ea0444 (Add math-500 dataset for validation) first." >&2
    exit 2
}

export MODEL_PATH="/data/homes/ugadiarov_la/ugadiarov.la/models/openPangu-Embedded-7B-llama"
export TRAIN_FILE="/data2/datasets/dapo/dapo-math-17k.parquet"
export TEST_FILE="/data2/datasets/math_datasets/math500.parquet"
export save_freq=1
export test_freq=1
export SEED=1
export train_prompt_bsz=32
export gpu_memory_utilization=0.7
export max_updates=20
unset VERL_GPU_MEM_CAP_GB
export exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn16 mini-32 ppo-epochs-1 DAPO17K-AIME24-25 openPangu-7B tp1dp8 token-mean 8192-len 0.01-wd bos seed-${SEED} gmu-${gpu_memory_utilization} h200"}
for p in "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    [[ -e "${p}" ]] || { echo "missing on this machine: ${p}" >&2; exit 2; }
done

LOG_FILE=${LOG_FILE:-"logs/run_openpangu7b_sync_B128_gmu0.7_math500_h200_$(date +%Y%m%d_%H%M%S).log"}
mkdir -p -- "$(dirname -- "${LOG_FILE}")"
echo "[run_openpangu7b_sync_B128_gmu0.7_math500_h200] repo: ${REPO_ROOT}"
echo "[run_openpangu7b_sync_B128_gmu0.7_math500_h200] console log: ${LOG_FILE}"

exec bash "${ARM_SCRIPT}" actor_rollout_ref.rollout.val_kwargs.n=3 "$@" > >(tee -- "${LOG_FILE}") 2>&1
