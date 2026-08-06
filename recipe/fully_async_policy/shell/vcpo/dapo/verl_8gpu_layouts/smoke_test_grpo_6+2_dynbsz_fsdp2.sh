#!/usr/bin/env bash
# Smoke test for grpo_novcpo_k=2_8gpu_dapo17k_6+2_resp8k_dynbsz_fsdp2.sh (FSDP2 backend,
# torchao _AdamW with bf16 stochastic rounding, dynamic bsz, deferred old-log-prob).
#
# Runs the real launch script end-to-end but for only 2 optimizer steps with
# validation disabled, then asserts on the log:
#   1. torchao is importable BEFORE launching (fails fast with a clear message —
#      the optimizer import otherwise dies deep inside worker init);
#   2. the deferred old-log-prob path engaged (trainer prints "Skipping
#      old_log_prob recomputation" / "Deferring rollout correction"), and the
#      dp_actor port actually computed IS weights (rollout_corr/ metrics appear
#      in the step lines — they are produced inside the actor on this path);
#   3. both training steps completed — the first update exercises FSDP2 +
#      dynamic-bsz packing + the SR optimizer together for the first time;
#   4. no Traceback / CUDA OOM anywhere in the log.
# It also samples nvidia-smi during the run and reports per-GPU peak memory so
# the OOM headroom is visible even when nothing crashes (expected: trainer GPUs
# ~33 GB static + activation peaks).
#
# Usage (from anywhere; needs the training environment activated):
#   bash recipe/fully_async_policy/shell/vcpo/math/smoke_test_grpo_6+2_dynbsz_fsdp2.sh
# Env knobs: SMOKE_TIMEOUT (s, default 5400), SMOKE_LOG (log path).

set -uo pipefail

cd "$(dirname "$0")/../../../../.."  # repo root, so the fork's verl shadows the installed one

SCRIPT="recipe/fully_async_policy/shell/vcpo/math/grpo_novcpo_k=2_8gpu_dapo17k_6+2_resp8k_dynbsz_fsdp2.sh"
LOG=${SMOKE_LOG:-logs/smoke_6+2_dynbsz.log}
MEMLOG="${LOG%.log}.gpumem.csv"
mkdir -p "$(dirname "${LOG}")"

# --- 0. fail fast on missing torchao ----------------------------------------
if ! python -c "from torchao.optim import _AdamW" 2>/dev/null; then
    echo "[smoke][FAIL] torchao (with torchao.optim._AdamW) is not importable in this environment;"
    echo "               the launch script's optimizer_impl=torchao.optim cannot work. Install torchao first."
    exit 1
fi
echo "[smoke] torchao import OK"

# The launch script honors these env overrides.
export val_before_train=False
export test_freq=-1                     # no validation at all
export total_rollout_steps=$((2 * 128)) # exactly 2 optimizer steps
export exp_name="SMOKE-6+2-dynbsz"      # keep TB/ckpt junk out of real run dirs
# 4 responses/prompt instead of 16: ~4x faster generation. Trainer micro-batches
# still pack to the same token budget (the OOM-relevant peak is unchanged);
# engines just run below full KV load. SMOKE_N=16 restores full fidelity.
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
    bad "trainer never announced 'Skipping old_log_prob recomputation' (skip_recompute path not taken?)"
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
    echo "[smoke] PASS — FSDP2 + torchao SR-AdamW + dynamic bsz + deferred old-log-prob: 2 steps trained without OOM"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
