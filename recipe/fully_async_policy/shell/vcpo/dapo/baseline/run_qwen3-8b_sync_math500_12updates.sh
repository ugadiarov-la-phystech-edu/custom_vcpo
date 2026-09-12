#!/usr/bin/env bash
# =============================================================================
# run_qwen3-8b_sync_math500_12updates.sh
#
# Short synchronous Qwen3-8B run for remote_h100: launches the real arm
#   main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh
# (no second copy of its configuration) and pins only:
#   TEST_FILE=/home/jovyan/datasets/math_datasets/math500.parquet   MATH-500 instead of AIME-2024/2025
#   test_freq=1 save_freq=1        validate and write an hf_model checkpoint after EVERY rollout step
#   max_updates=12                 12 OPTIMIZER updates = 3 rollout steps at B128 / mini32 / 1 epoch
#                                  (the arm rounds up and passes trainer.total_training_steps=3; the
#                                  last step validates, saves and ends the run)
#   val_before_train=False         no validation of the base model before step 1
# Everything else is the arm's own and stays env-overridable from the caller (SEED, exp_name,
# train_prompt_mini_bsz, ...); extra hydra overrides pass through "$@".
#
# Disk: 3 hf exports x ~16 GB under logs/<exp_name> (this arm has no log_dir / CKPTS_DIR override:
# relabel with exp_name=... to move the run inside logs/). Validation metrics land under
# val-core/<data_source of math500.parquet>/acc/mean@1. The arm's default exp_name still says
# DAPO17K-AIME24-25; pass exp_name=... to relabel the run.
# =============================================================================
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

# uv environment. The remote checkouts keep their activation in the repo root (activgate.sh:
# uv + the shared vcpo-env + HF_HOME); without it, fall back to the shared env's own activate
# (UV_ENV_ACTIVATE overrides the path). Either way the interpreter is verified before anything
# is launched, so a missing environment fails here instead of deep inside Ray.
UV_ENV_ACTIVATE=${UV_ENV_ACTIVATE:-/home/jovyan/ugadiarov/uv-envs/vcpo-env/bin/activate}
if [[ -f "${REPO_ROOT}/activate.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/activate.sh"
elif [[ -f "${UV_ENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1091
    source "${UV_ENV_ACTIVATE}"
fi
python -c "import ray, hydra" 2>/dev/null || {
    echo "vcpo uv environment is not active: no activate.sh in ${REPO_ROOT} and no ${UV_ENV_ACTIVATE}" >&2
    echo "(python: $(command -v python || echo none))" >&2
    exit 2
}

ARM_SCRIPT="${HERE}/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh"
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

export TEST_FILE="/home/jovyan/datasets/math_datasets/math500.parquet"
export test_freq=1
export save_freq=1
export max_updates=12
export val_before_train=False

exec bash "${ARM_SCRIPT}" "$@"
