#!/usr/bin/env bash
# =============================================================================
# smoke_test_openpangu_ppo-default_5+3.sh
#
# Fast end-to-end check that the openPangu-Embedded-7B arm with the recipe's DEFAULT objective
#   grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_ppo-default.sh
# TRAINS AND VALIDATES WITHOUT ERRORS on its own 5+3 layout (5 vLLM engines + 3 FSDP2 trainer
# GPUs, all 8 GPUs). It runs the real arm script - there is no second copy of the configuration
# to drift - and only overrides what makes the run short:
#
#   * 2 TRAINER STEPS: total_rollout_steps = 6 = 2 x (mini_bsz 3 x require_batches 1), so the
#     trainer runs 2 steps (4 AdamW updates at ppo_epochs=2) and the run stops on its own.
#   * n_resp_per_prompt=4 instead of 16 and mini_bsz 3 instead of 33: 3*4 = 12 sequences per
#     step, divisible by trainer DP=3. n=4 (not 2) makes an all-tied group - GRPO advantage 0,
#     no policy-gradient signal - less likely, so the PPO clip path usually sees real ratios.
#   * max_response_length stays at the arm's 8192: generation, the rollout log-probs the PPO
#     ratio is built on, and the trainer's memory are exercised at the real length.
#   * VALIDATION AFTER EVERY STEP (test_freq=1), NOT before training (val_before_train=False):
#     exactly 2 validations, both of a trained model. AIME-2024 only, from the deduplicated
#     aime-2024_smoke.parquet (30 problems instead of 960 rows) - at 8192 tokens the full file
#     would dominate the runtime.
#   * NO CHECKPOINTS by default (save_freq=-1): each FSDP2 hf_model save is ~32 GB fp32, and
#     saving is not what this test checks. save_freq=1 turns them on, and the wrapper then also
#     runs verify_checkpoints.py on the 2 saves.
#
# THE OBJECTIVE IS NOT TOUCHED: loss_mode=vanilla (PPO clip 0.2/0.2, dual-clip 3.0), no
# importance weights, token-mean, no KL, entropy_coeff=0, lr as in the arm. So this checks the
# arm's real loss path, not a proxy for it. A consequence: if every group happens to tie, the
# policy gradient is exactly 0 and the weights do not move - still a pass for "trains without
# errors", which is all this test claims. (smoke_test_openpangu_3+3.sh adds an entropy bonus and
# a 100x lr to force visible weight changes; that is a different question.)
#
# The model is NOT shrunk: the point is the real openPangu path - the custom tokenizer through
# trust_remote_code, the re-aliased Llama checkpoint in vLLM and FSDP2, BOS, weight sync.
#
# PASS/FAIL: the console output is tee'd to logs/<exp_name>/smoke_console.log and checked at the
# end: the arm must exit 0, log no Python traceback, reach trainer step 2, log the PPO clip
# metric actor/pg_clipfrac (proof the vanilla loss ran), and log 2 validation results for
# val-core/math_dapo/acc/mean@1. The last line is SMOKE PASS or SMOKE FAIL (exit 1).
#
# Usage:  bash smoke_test_openpangu_ppo-default_5+3.sh
# Env:    MODEL_PATH, TRAIN_FILE, TEST_FILE, exp_name, n_resp_per_prompt, save_freq, ...
#         (defaults are the cloud.ru paths of the arm; override them on other machines)
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_ppo-default.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

# Ray workers deserialize the trust_remote_code tokenizer BY REFERENCE, as
# transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer; that dynamic package is
# only on sys.path in a process that has itself loaded remote code. Exporting the HF modules
# cache makes the reference resolvable in every Ray worker (see the arm's header). The arm
# does the same; the guard keeps the entry single.
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# AIME-2024 only, deduplicated (30 distinct problems): see the header.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024_smoke.parquet']"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-openpangu7b-ppo-default-5+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}

# ---- the arm's own 5 + 3 layout on all eight GPUs ---------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export n_gpus_rollout=${n_gpus_rollout:-5}

# ---- 2 steps, cheap ones; objective untouched ------------------------------------
export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-4}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}   # the arm's own length
export total_rollout_steps=${total_rollout_steps:-6}       # = 2 trainer steps of 3 groups
export test_freq=${test_freq:-1}                           # validate after every step
export save_freq=${save_freq:--1}                          # no checkpoints (see header)
export val_before_train=${val_before_train:-False}

# `--cfg job` and friends make the arm print its config and exit: pass straight through, so
# the output stays clean YAML and nothing is checked.
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
check() {  # check <description> <condition-result 0|1>
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
# the trainer's per-step metric line carries "val-core/<source>/acc/mean@1:<value>" once per
# validation (the pretty-printed dict form may wrap the value onto the next line, so it is not used)
n_val=$(strip | grep -o "val-core/math_dapo/acc/mean@1:[0-9][0-9.e-]*" | wc -l)
check "validation results logged (${n_val} x val-core/math_dapo/acc/mean@1, expect >= 2)" "$([[ ${n_val} -ge 2 ]]; echo $?)"
echo "  validation values:"
strip | grep "val-core/math_dapo/acc/mean@1:" \
    | sed -n 's/.*\(step:[0-9]*\) - .*\(val-core\/math_dapo\/acc\/mean@1:[0-9.e-]*\).*/    \1 \2/p' || true

if [[ "${save_freq}" -gt 0 ]]; then
    echo "==================== checkpoint verification ===================="
    # FSDP2 at model_dtype=fp32 writes fp32 hf_model weights.
    if ! python "${HERE}/verify_checkpoints.py" "${LOG_DIR}" --expect 2 --dtype F32 --base-model "${MODEL_PATH}"; then
        echo "  FAIL  verify_checkpoints.py"; fail=1
    fi
fi

echo "console log: ${CONSOLE_LOG}"
if [[ ${fail} -eq 0 ]]; then echo "SMOKE PASS"; else echo "SMOKE FAIL"; exit 1; fi
