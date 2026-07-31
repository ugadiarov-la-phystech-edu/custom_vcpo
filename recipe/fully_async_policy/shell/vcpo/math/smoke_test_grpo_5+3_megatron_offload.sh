#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/../../../../.."

SCRIPT="recipe/fully_async_policy/shell/vcpo/math/grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload.sh"
LOG=${SMOKE_LOG:-logs/smoke_5+3_megatron_offload.log}
MEMLOG="${LOG%.log}.gpumem.csv"
mkdir -p "$(dirname "${LOG}")"

export val_before_train=False
export test_freq=-1
export total_rollout_steps=$((2 * 129))
export exp_name="SMOKE-5+3-megatron-offload"
export n_resp_per_prompt=${SMOKE_N:-4}

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

if [ "${rc}" -eq 124 ]; then
    bad "run hit the ${SMOKE_TIMEOUT:-5400}s timeout before finishing 2 steps"
elif [ "${rc}" -ne 0 ]; then
    bad "launch script exited with code ${rc}"
fi

if grep -aqE "Traceback|CUDA out of memory" "${LOG}"; then
    bad "Traceback / CUDA OOM found in log:"
    grep -anE "Traceback|CUDA out of memory" "${LOG}" | head -3
fi

if grep -aq "optimizer config after override" "${LOG}"; then
    if grep -aq "'optimizer_cpu_offload': True" "${LOG}"; then
        note "HybridDeviceOptimizer (CPU offload) confirmed in Megatron OptimizerConfig"
    else
        bad "optimizer override printed but optimizer_cpu_offload is not True"
    fi
else
    bad "optimizer config printout not found (trainer may not have reached optimizer init)"
fi

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

if grep -aq "step:2 " "${LOG}"; then
    note "reached step 2; per-step metrics (last metrics line):"
    grep -a "rollout_corr/rollout_is_mean" "${LOG}" | tail -1 | tr " " "\n" \
        | grep -aE "^step:|time_per_step|timing_s/update_actor|timing_s/gen:|max_memory|response_length/mean|rollout_corr/rollout_is_mean|actor/pg_loss" || true
    if grep -aq "rollout_corr/rollout_is_mean" "${LOG}"; then
        note "rollout_corr/ metrics present (deferred IS-weight computation ran in the actor)"
    else
        bad "no rollout_corr/ metrics anywhere in the log — deferred IS-weight computation did not run"
    fi
else
    bad "never reached step 2 (last step lines below)"
    grep -a "step:" "${LOG}" | tail -2 | cut -c1-200
fi

if [ -s "${MEMLOG}" ]; then
    note "peak GPU memory during run (MiB, per GPU):"
    awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "  GPU %s: %d\n", g, m[g] }' \
        "${MEMLOG}" | sort -V
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS — Megatron 5+3 (tp1/dp3, HDO offload, fixed micro-bsz) + deferred old-log-prob: 2 steps trained without OOM"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
