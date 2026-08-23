#!/usr/bin/env bash
# =============================================================================
# smoke_test_openpangu_3+3.sh
#
# Fast end-to-end exercise of the openPangu-Embedded-7B arm
#   grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh
# on a 3+3 layout (3 vLLM engines + 3 FSDP2 trainer GPUs, 6 GPUs total). It runs the real
# arm script - there is no second copy of the configuration to drift - and only overrides
# what makes it short:
#
#   * 2 TRAINER STEPS: total_rollout_steps = 6 = 2 x (mini_bsz 3 x require_batches 1),
#     so total_train_steps = 2 and the run stops on its own.
#   * n_resp_per_prompt=2 instead of 16 (the requested speed lever) and mini_bsz 3 instead
#     of 33: 3*2 = 6 sequences, divisible by trainer DP=3.
#   * max_response_length stays at the arm's 8192, so generation is exercised at the
#     real length. This is the dominant cost: with validation on the full 960-row
#     aime-2024 after each of the 2 steps, expect the two validation sweeps to take
#     longer than the training itself.
#   * VALIDATION ON AIME-2024 ONLY, EVERY STEP (test_freq=1) and NO validation before
#     training (val_before_train=False), so the two validations that run are both of a
#     trained model. Note this is the full 960-row file at 8192 tokens - point TEST_FILE
#     at a subset, or lower max_response_length, if you need it faster.
#   * A CHECKPOINT EVERY STEP (save_freq=1) -> exactly 2 checkpoints. Unlike the 8-GPU
#     smoke test there is no third one: the trainer's end-of-fit block only forces a final
#     sync when `param_version % test_freq != 0 or local_trigger_step > 1`, and at
#     test_freq=1 both are false.
#   * a real gradient regardless of rewards: with 3 prompts x n=2 a group can easily tie
#     (both responses wrong, or both truncated), and a tied group has GRPO advantage
#     identically 0 - pg_loss and grad_norm are then exactly 0, the weights cannot move and
#     "the checkpoints differ" becomes unverifiable. entropy_coeff gives a gradient that
#     does not depend on the rewards, and the large lr makes 4 updates visible in the
#     weights. This is a plumbing test, not a learning test: neither value relates to the
#     arm's own settings.
#
# The model is NOT shrunk: the point is to exercise the real openPangu path - the custom
# tokenizer through trust_remote_code, the re-aliased Llama config in vLLM and FSDP2, and
# the hf_model save. FSDP2 keeps params in fp32 (fsdp_config.model_dtype default), and
# get_fsdp_full_state_dict does not cast, so each checkpoint is ~32 GB: budget ~64 GB.
#
# Afterwards it runs verify_checkpoints.py, which fails loudly if a checkpoint is
# incomplete: weights + tokenizer + config present and loadable (the tokenizer needs
# trust_remote_code - openPangu keeps a custom tokenizer class), no sharded leftovers,
# parameter names/shapes/dtype matching the base checkpoint, weights actually changed
# between the two checkpoints, timing_state.json complete, tracker up to date.
#
# Usage:  bash smoke_test_openpangu_3+3.sh
# Env:    MODEL_PATH, TRAIN_FILE, TEST_FILE, SMOKE_DIR, n_resp_per_prompt, test_freq, ...
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# AIME-2024 only, as a single-element list: the arm validates on 2024+2025 by default.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet']"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-openpangu7b-3+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}

# ---- 3 + 3 layout on the first six GPUs --------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
export n_gpus_rollout=${n_gpus_rollout:-3}

# ---- 2 steps, cheap ones -----------------------------------------------------
export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}   # the arm's own length, not a shortened one
export total_rollout_steps=${total_rollout_steps:-6}
export test_freq=${test_freq:-1}          # validate after every param version
export save_freq=${save_freq:-1}          # and checkpoint every one
export val_before_train=${val_before_train:-False}
export ppo_epochs=${ppo_epochs:-2}        # unchanged from the arm: cheap at 6 sequences
export entropy_coeff=${entropy_coeff:-0.01}  # see the header: the only non-zero gradient here
export lr=${lr:-1e-4}                     # 100x the arm's, so 4 updates clear rounding

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
# stderr, so that `bash smoke_test_openpangu_3+3.sh --cfg job --resolve` yields clean YAML
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

# `--cfg job` and friends make the arm print its config and exit: nothing to verify then.
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exit 0 ;; esac
done

CKPTS_DIR="logs/${exp_name//\//_}"
set +x
echo "==================== checkpoint verification ===================="
# --dtype F32: FSDP2 at the default model_dtype=fp32 writes fp32 weights.
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 2 --dtype F32 --base-model "${MODEL_PATH}"
