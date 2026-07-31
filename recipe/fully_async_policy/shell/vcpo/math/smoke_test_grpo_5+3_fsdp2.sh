#!/usr/bin/env bash
# Smoke test for grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_fsdp2.sh (FSDP2 dp=3
# trainer with verl's standard precision recipe — fp32 params + plain AdamW —
# fixed micro-batching, FSDP-path rollout at util 0.8 / max_num_seqs 512).
#
# Runs the real launch script end-to-end for only 2 optimizer steps with
# validation disabled, then asserts on the log:
#   1. the deferred old-log-prob path engaged (trainer announcements) and the
#      dp_actor computed IS weights in the backward pass (rollout_corr/ metrics);
#   2. both training steps completed — first update exercises the dp=3 trainer
#      memory profile (~44 GB static fp32 params/grads/AdamW + transients) and
#      the 129-prompt (divisible-by-3) mini-batch plumbing;
#   3. no Traceback / CUDA OOM anywhere in the log.
# Also samples nvidia-smi and reports per-GPU peaks (expected: 5 rollout GPUs
# ~74-75 GB at util 0.8; 3 trainer GPUs ~55-65 GB peak).
#
# Usage (needs the training environment activated; all 8 GPUs free):
#   bash recipe/fully_async_policy/shell/vcpo/math/smoke_test_grpo_5+3_fsdp2.sh
# Env knobs: SMOKE_TIMEOUT (s, default 5400), SMOKE_LOG (log path),
#            SMOKE_N (responses/prompt, default 4; 16 = full fidelity).

set -uo pipefail

cd "$(dirname "$0")/../../../../.."  # repo root, so the fork's verl shadows the installed one

SCRIPT="recipe/fully_async_policy/shell/vcpo/math/grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_fsdp2.sh"
LOG=${SMOKE_LOG:-logs/smoke_5+3_fsdp2.log}
MEMLOG="${LOG%.log}.gpumem.csv"
mkdir -p "$(dirname "${LOG}")"

# The launch script honors these env overrides.
export val_before_train=False
export test_freq=-1                     # no validation at all
export total_rollout_steps=$((2 * 129)) # exactly 2 optimizer steps (mini-batch is 129 prompts)
export exp_name="SMOKE-5+3-fsdp2"       # keep TB/ckpt junk out of real run dirs
# 4 responses/prompt instead of 16: ~4x faster generation. 129*4=516 seqs still
# divides by trainer DP=3. Fixed 1-seq micro-batches mean the per-micro-batch
# memory profile is n-independent; batch-tensor residency and full-batch update
# time are NOT measured at n=4 — run once with SMOKE_N=16 after a pass.
export n_resp_per_prompt=${SMOKE_N:-4}

# Sample per-GPU memory every 15 s for the peak report.
if command -v nvidia-smi >/dev/null; then
    ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 15; done ) \
        > "${MEMLOG}" 2>/dev/null &
    MEM_PID=$!
    trap '[ -n "${MEM_PID:-}" ] && kill "${MEM_PID}" 2>/dev/null' EXIT
fi

echo "[smoke] launching 2-step run (timeout ${SMOKE_TIMEOUT:-5400}s); log: ${LOG}"
timeout "${SMOKE_TIMEOUT:-5400}" bash "${SCRIPT}" > "${LOG}" 2>&1
rc=$?
[ -n "${MEM_PID:-}" ] && kill "${MEM_PID}" 2>/dev/null && MEM_PID=""

fail=0
note() { echo "[smoke] $*"; }
bad()  { echo "[smoke][FAIL] $*"; fail=1; }

# --- 1. process outcome -----------------------------------------------------
if [ "${rc}" -eq 124 ]; then
    bad "run hit the ${SMOKE_TIMEOUT:-5400}s timeout before finishing 2 steps"
elif [ "${rc}" -ne 0 ]; then
    bad "launch script exited with code ${rc}"
fi

# --- 2. crashes -------------------------------------------------------------
if grep -aqE "Traceback|CUDA out of memory" "${LOG}"; then
    bad "Traceback / CUDA OOM found in log:"
    grep -anE "Traceback|CUDA out of memory" "${LOG}" | head -3
fi

# --- 3. deferred old-log-prob path engaged ----------------------------------
if grep -aq "Skipping old_log_prob recomputation" "${LOG}"; then
    note "deferred old-log-prob path engaged (trainer skipped the recompute pass)"
else
    bad "trainer never announced 'Skipping old_log_prob recomputation'"
fi
if grep -aq "Deferring rollout correction to backward pass" "${LOG}"; then
    note "rollout correction deferred to the actor backward pass"
else
    bad "trainer never announced 'Deferring rollout correction to backward pass'"
fi

# --- 4. both steps completed, dp_actor computed IS weights -------------------
if grep -aq "step:2 " "${LOG}"; then
    note "reached step 2; per-step metrics (last metrics line):"
    grep -a "rollout_corr/rollout_is_mean" "${LOG}" | tail -1 | tr " " "\n" \
        | grep -aE "^step:|time_per_step|timing_s/update_actor|timing_s/gen:|max_memory|response_length/mean|rollout_corr/rollout_is_mean|actor/pg_loss" || true
    if grep -aq "rollout_corr/rollout_is_mean" "${LOG}"; then
        note "rollout_corr/ metrics present (dp_actor deferred path computed IS weights)"
    else
        bad "no rollout_corr/ metrics anywhere in the log — deferred IS-weight computation did not run"
    fi
else
    bad "never reached step 2 (last step lines below)"
    grep -a "step:" "${LOG}" | tail -2 | cut -c1-200
fi

# --- 5. GPU memory peaks ----------------------------------------------------
if [ -s "${MEMLOG}" ]; then
    note "peak GPU memory during run (MiB, per GPU):"
    awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "  GPU %s: %d\n", g, m[g] }' \
        "${MEMLOG}" | sort -V
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS — FSDP2 5+3 (fp32 AdamW, fixed micro-bsz) + deferred old-log-prob: 2 steps trained without OOM"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
