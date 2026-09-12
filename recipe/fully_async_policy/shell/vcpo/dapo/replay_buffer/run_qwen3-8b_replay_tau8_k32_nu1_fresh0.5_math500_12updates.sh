#!/usr/bin/env bash
# =============================================================================
# run_qwen3-8b_replay_tau8_k32_nu1_fresh0.5_math500_12updates.sh
#
# Short Qwen3-8B replay run for remote_h100: launches the real arm
#   grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh
# (no second copy of its configuration) and pins only:
#   TEST_FILE=/home/jovyan/datasets/math_datasets/math500.parquet   MATH-500 instead of AIME-2024/2025
#   test_freq=1 save_freq=1        validate and write an hf_model checkpoint after EVERY update
#   max_updates=12                 stop after 12 optimizer updates (= parameter versions): final
#                                  validation, forced final checkpoint, rollouter cancelled
#   val_before_train=False         no validation of the base model before update 1
# Everything else is the arm's own and stays env-overridable from the caller (SEED, exp_name,
# replay_min_fresh_ratio, ...); extra hydra overrides pass through "$@".
#
# Disk: 12 hf exports x ~16 GB = ~190 GB under logs/<exp_name> (this arm has no log_dir / CKPTS_DIR
# override: relabel with exp_name=... to move the run inside logs/). Validation metrics land under
# val-core/<data_source of math500.parquet>/acc/mean@1. The arm's default exp_name still says
# DAPO17K-AIME24; pass exp_name=... to relabel the run.
# =============================================================================
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

# uv environment. The remote checkouts keep their activation in the repo root (activate.sh:
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

ARM_SCRIPT="${HERE}/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh"
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

export TEST_FILE="/home/jovyan/datasets/math_datasets/math500.parquet"
export test_freq=1
export save_freq=1
export max_updates=12
export val_before_train=False

exec bash "${ARM_SCRIPT}" "$@"
