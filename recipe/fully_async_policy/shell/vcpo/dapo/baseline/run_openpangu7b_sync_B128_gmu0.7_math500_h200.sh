#!/usr/bin/env bash
# =============================================================================
# run_openpangu7b_sync_B128_gmu0.7_math500_h200.sh  (remote_h200 launcher, SYNC arm, batch 32, NO H100 emulation)
#
# Launches the real arm
#   main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh
# (verl.trainer.main_ppo, colocated hybrid engine, 8 GPUs; no second copy of its configuration)
# with the remote_h200 paths, a per-step cadence and an H200-sized vLLM budget:
#   MODEL_PATH  /data/homes/ugadiarov_la/ugadiarov.la/models/openPangu-Embedded-7B-llama
#   TRAIN_FILE  /data2/datasets/dapo/dapo-math-17k.parquet
#   TEST_FILE   /data2/datasets/math_datasets/math500.parquet  (MATH-500: 500 problems, 1 row each,
#               data_source "math500_dapo" -> metric val-core/math500_dapo/acc/mean@3 and the
#               LaTeX-normalising MATH-500 scorer; REQUIRES the math500_dapo reward route in
#               verl/utils/reward_score/__init__.py -- guarded below)
#   save_freq=1 test_freq=1  validate and write an hf_model checkpoint after EVERY rollout step
#   SEED=1
#   actor_rollout_ref.rollout.val_kwargs.n=3   passed as a trailing hydra override (the arm
#                            hard-codes n=1 and has no env knob; the trailing override wins):
#                            1500 generations per validation
#   train_prompt_bsz=32      32 prompts x 16 = 512 sequences per rollout step (64 per engine)
#                            instead of the arm's native 128; with train_prompt_mini_bsz=32
#                            (unchanged) and ppo_epochs=1 that is ONE optimizer update per rollout
#                            step, fully on-policy (ppo_kl and clipfrac are exactly 0)
#   max_updates=20           20 OPTIMIZER updates = trainer.total_training_steps=20: twenty rollout
#                            steps, each validated and checkpointed, then verl's is_last_step ends
#                            the run.
#   gpu_memory_utilization=0.7   vLLM gets 0.7 x 139.8 GiB = 97.9 GiB per GPU (the arm's H100 default
#                            is 0.5 = 40 GB). The fraction is a share of the WHOLE card while the
#                            resident trainer (bf16 params + grad buffers, ~36.5 GB) is a fixed
#                            amount, so the like-for-like ceiling grows with the card: on the H100,
#                            0.5 leaves ~2.3 GB of the 81.6 GB; on the H200, 0.7 leaves ~4.6 GB of
#                            the 143.8 GB (measured 2026-09-12: trainer resident 36.5 GB, first-wake
#                            spike ~2 GB above steady rollout). ~83 GiB of KV cache (~620k tokens)
#                            vs ~15 GiB at 0.283: the 64 sequences per engine fit resident even at
#                            the 10k-token limit, so generation runs single-wave at the latency of
#                            the longest response (~100-130 s) instead of the 110-240 s measured at
#                            0.283. If the first step OOMs in generation (a vLLM KV-allocation
#                            message), step down to 0.65.
# This is NOT an H100 emulation: no trainer cap (this branch has none), and 0.7 of an H200 is a
# budget no H100 can offer. Compare with the emulated runs on the cumulative_training_time axis.
# exp_name: the arm's default name does not record the fraction and would COLLIDE with a real-H100
# run of the same arm, so the launcher sets it explicitly (the arm's default text + " gmu-0.7 h200").
# Pass exp_name=... to override. Everything else stays env-overridable (val_before_train,
# log_dir / CKPTS_DIR, CUDA_VISIBLE_DEVICES, ...); extra hydra overrides pass through "$@" and,
# being later on the command line, override even the val_kwargs.n=3 set here.
#
# Disk: one hf export (~16 GB) per rollout step under CKPTS_DIR (default logs/<exp_name>): 20 steps =
# ~320 GB -- check `df -h /data` first or point CKPTS_DIR at a volume with room. Console output is
# teed into LOG_FILE (default
# logs/run_openpangu7b_sync_B128_gmu0.7_math500_h200_<timestamp>.log) and still shown.
# =============================================================================
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
cd -- "${REPO_ROOT}" # activate.sh must not move us off the repo root (the fork's verl shadows the installed one)
python -c "import ray, hydra" 2>/dev/null || {
    echo "vcpo uv environment is not active: no activate.sh in ${REPO_ROOT} and no ${UV_ENV_ACTIVATE}" >&2
    echo "(python: $(command -v python || echo none))" >&2
    exit 2
}

ARM_SCRIPT="${HERE}/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh"
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT} (not a baselines_main-ppo_openpangu checkout?)" >&2; exit 2; }
grep -q 'max_updates=' "${ARM_SCRIPT}" || { echo "arm script lacks the max_updates knob" >&2; exit 2; }
# The MATH-500 file is useless without its reward route: the run would die at the first validation.
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
export train_prompt_bsz=32 # -> exp_name "B-32xn16 mini-32", 1 update per rollout step
export gpu_memory_utilization=0.7 # H200-sized vLLM budget, see the header
export max_updates=20 # -> trainer.total_training_steps=20 (1 update per rollout step at B32/mini32)
# No cap on this branch anyway; discard one inherited from the calling shell so the intent is explicit.
unset VERL_GPU_MEM_CAP_GB
# The arm's default exp_name at these knobs, plus a suffix that keeps this run apart from H100 runs.
export exp_name=${exp_name:-"MAIN-PPO-SYNC grpo B-${train_prompt_bsz}xn16 mini-32 ppo-epochs-1 DAPO17K-AIME24-25 openPangu-7B tp1dp8 token-mean 8192-len 0.01-wd bos seed-${SEED} gmu-${gpu_memory_utilization} h200"}
for p in "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    [[ -e "${p}" ]] || { echo "missing on this machine: ${p}" >&2; exit 2; }
done

LOG_FILE=${LOG_FILE:-"logs/run_openpangu7b_sync_B128_gmu0.7_math500_h200_$(date +%Y%m%d_%H%M%S).log"}
mkdir -p -- "$(dirname -- "${LOG_FILE}")"
echo "[run_openpangu7b_sync_B128_gmu0.7_math500_h200] repo: ${REPO_ROOT}"
echo "[run_openpangu7b_sync_B128_gmu0.7_math500_h200] console log: ${LOG_FILE}"

# exec keeps the arm as this PID (kill / $! act on the run itself); tee captures stdout+stderr.
# val_kwargs.n=3 goes BEFORE "$@" so a caller's own override of it still wins.
exec bash "${ARM_SCRIPT}" actor_rollout_ref.rollout.val_kwargs.n=3 "$@" > >(tee -- "${LOG_FILE}") 2>&1
