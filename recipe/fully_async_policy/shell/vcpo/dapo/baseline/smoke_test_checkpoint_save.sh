#!/usr/bin/env bash
# =============================================================================
# smoke_test_checkpoint_save.sh
#
# Fastest possible end-to-end exercise of the is-pg baseline arm, for checking
# that CHECKPOINTS ARE SAVED CORRECTLY. It runs the real script
#   grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh
# with everything expensive turned down, so there is no second copy of the
# configuration to drift out of sync - only the knobs below differ:
#
#   * 2 trainer iterations: total_rollout_steps = 6 = 2 x (mini_bsz 3 x 1 batch),
#     so total_train_steps = 2 and the run stops on its own;
#   * mini batch 3 groups (3*2 = 6 seqs, divisible by trainer DP=3) instead of 33;
#   * n_resp_per_prompt=2 instead of 16, max_response_length=512 instead of 8192;
#   * save_freq=1 -> one checkpoint per param version, i.e. 2 checkpoints, and
#     max_actor_ckpt_to_keep=null (the baseline default) keeps both;
#   * a LEARNING SIGNAL that does not depend on the rewards: at 512 response tokens
#     every DAPO answer is truncated, so all responses in a group score the same, the
#     GRPO advantage is identically 0 and pg_loss/grad_norm are exactly 0 - the weights
#     could not change and "the checkpoints differ" would be unverifiable. entropy_coeff
#     and a large lr give a real, visible update. This is a plumbing test, not a
#     learning test: neither value has anything to do with the arm's own settings.
#   * NO VALIDATION: val_before_train=False and test_freq is set beyond the run,
#     so update_param_version never validates. (test_freq=0 is NOT usable: the
#     end-of-fit block in fully_async_trainer.py does `param_version % test_freq`.
#     The trainer's final sync still forces one validation pass after the loop,
#     after both checkpoints exist, so TEST_FILE is pointed at a 2-row parquet
#     this script generates to keep it near-free.)
#
# The model is deliberately NOT shrunk: the point is to exercise the real
# hf_model save path (Qwen3-8B, untied embeddings, ~16 GB per checkpoint written
# by rank 0 after an all-ranks gather). Expect ~35 GB of checkpoints.
#
# ARM_SCRIPT selects which arm to smoke: the megatron is-pg arm by default, or the
# FSDP2 twin (whose checkpoints are fp32, so ~66 GB for the two).
#
# After the run it calls verify_checkpoints.py, which fails loudly if a
# checkpoint is incomplete: weights, tokenizer and config present and readable,
# no dist_ckpt/ left behind, parameter names/shapes/dtype matching the base
# model, weights actually changed between the two checkpoints, timing_state.json
# complete with timezone-aware datetimes, and the tracker file pointing at the
# newest checkpoint.
#
# Usage:  bash smoke_test_checkpoint_save.sh
#         ARM_SCRIPT=<...>_fsdp2_<...>_is-pg.sh bash smoke_test_checkpoint_save.sh
# Env:    ARM_SCRIPT, SMOKE_DIR (default logs/smoke_ckpt), MODEL_PATH, TRAIN_FILE
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }
# keep the tag to [A-Za-z0-9-]: it becomes a hydra override value and a directory name
arm_tag=$(basename -- "${ARM_SCRIPT}" .sh)
arm_tag=${arm_tag//[^A-Za-z0-9-]/-}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
FULL_TEST_FILE=${FULL_TEST_FILE:-"/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}, so this stays
# byte-identical on both sides and the verifier can find the checkpoints.
exp_name=${exp_name:-"SMOKE-${arm_tag}"}
SMOKE_DIR=${SMOKE_DIR:-"logs/smoke_ckpt"}
mkdir -p -- "${SMOKE_DIR}"

# A 2-row validation set, so the trainer's unavoidable final validation pass is
# near-free instead of a full AIME sweep.
TINY_TEST_FILE="${SMOKE_DIR}/tiny_val.parquet"
python - "${FULL_TEST_FILE}" "${TINY_TEST_FILE}" <<'PY'
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
pd.read_parquet(src).head(2).to_parquet(dst)
print(f"[smoke] wrote {dst} from {src}")
PY

# 6 prompts fed -> required_samples = mini_bsz * require_batches = 3 -> 2 trainer steps
export MODEL_PATH TRAIN_FILE
export TEST_FILE="['${TINY_TEST_FILE}']"
export n_resp_per_prompt=2
export train_prompt_mini_bsz=3
export max_response_length=512
export total_rollout_steps=6
export save_freq=1
export test_freq=1000000        # never inside the run; 0 would divide by zero at the end
export val_before_train=False
export ppo_epochs=2             # unchanged from the arm: cheap at 6 sequences
export entropy_coeff=${entropy_coeff:-0.01}  # see the header: the only non-zero gradient here
export lr=${lr:-1e-4}           # 100x the arm's, so 4 updates clear bf16 rounding
export exp_name

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s"

CKPTS_DIR="logs/${exp_name//\//_}"
set +x
echo "==================== checkpoint verification ===================="
# 3 checkpoints, not 2: after the 2 training iterations the trainer's end-of-fit block
# performs one final parameter sync (fully_async_trainer.py), which creates a third param
# version, and save_freq=1 checkpoints it.
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 3 --base-model "${MODEL_PATH}"
