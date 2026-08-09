#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/../../../../../.."

SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64.sh"
LOG=${SMOKE_LOG:-logs/smoke_replay_5+3.log}
MEMLOG="${LOG%.log}.gpumem.csv"
mkdir -p "$(dirname "${LOG}")"

export MODEL_PATH=${SMOKE_MODEL:-"Qwen/Qwen3-1.7B"}
export exp_name="SMOKE-replay-5+3"
export val_before_train=False
export test_freq=-1
export max_prompt_length=1024
export max_response_length=1024
export n_resp_per_prompt=8
export train_prompt_mini_bsz=6
export total_rollout_steps=${SMOKE_PROMPTS:-400}
export save_freq=5
export replay_tau=2
export replay_staleness_threshold=3
export replay_requires_mini_batches=2
export staleness_threshold=3.0

CKPT_DIR="logs/${exp_name}"

if command -v nvidia-smi >/dev/null; then
    ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 15; done ) \
        > "${MEMLOG}" 2>/dev/null &
    MEM_PID=$!
    trap '[ -n "${MEM_PID:-}" ] && kill "${MEM_PID}" 2>/dev/null' EXIT
fi

echo "[smoke] launching replay-buffer run (timeout ${SMOKE_TIMEOUT:-3600}s); log: ${LOG}"
timeout "${SMOKE_TIMEOUT:-3600}" bash "${SCRIPT}" > "${LOG}" 2>&1
rc=$?
[ -n "${MEM_PID:-}" ] && kill "${MEM_PID}" 2>/dev/null && MEM_PID=""

fail=0
note() { echo "[smoke] $*"; }
bad()  { echo "[smoke][FAIL] $*"; fail=1; }

if [ "${rc}" -eq 124 ]; then
    bad "run hit the ${SMOKE_TIMEOUT:-3600}s timeout before finishing"
elif [ "${rc}" -ne 0 ]; then
    bad "launch script exited with code ${rc}"
fi

if grep -aqE "Traceback|CUDA out of memory" "${LOG}"; then
    bad "Traceback / CUDA OOM found in log:"
    grep -anE "Traceback|CUDA out of memory" "${LOG}" | head -3
fi

updates=$(grep -ac "\[FullyAsyncTrainer\]\[Replay\] global_steps" "${LOG}")
if [ "${updates}" -ge 5 ]; then
    note "replay loop ran ${updates} model updates"
else
    bad "only ${updates} replay updates ran (expected >= 5)"
    note "kept-group supply diagnostics (all-wrong ratio near 1.0 = starved by the insertion gate):"
    grep -aoE "fully_async/groups/(all_wrong|all_correct)_ratio_total:[0-9.]+" "${LOG}" | tail -4
fi
if grep -aq "\[FullyAsyncTrainer\]\[Replay\] rollout finished" "${LOG}"; then
    note "trainer terminated on the data sentinel"
else
    bad "no sentinel termination message from the replay loop"
fi

syncs=$(grep -ac "Parameter version updated from" "${LOG}")
if [ "${syncs}" -ge 5 ]; then
    note "parameter version updated ${syncs} times (sync-per-update active)"
else
    bad "only ${syncs} parameter version updates (expected one per model update)"
fi

if grep -aq "fully_async/groups/all_wrong_ratio" "${LOG}"; then
    note "insertion-gate group ratios logged (all-correct/all-wrong counters active)"
else
    bad "group all-correct/all-wrong ratio metrics never logged"
fi

if grep -aqE "replay/minibatch_replayed:[1-9]" "${LOG}"; then
    note "score-weighted replay engaged (some mini-batch reused old groups):"
    grep -aoE "replay/minibatch_new_ratio:[0-9.]+" "${LOG}" | tail -3
else
    bad "no mini-batch ever contained replayed groups (replay/minibatch_replayed always 0)"
fi

if grep -aqE "replay/evicted_cum:[1-9]" "${LOG}"; then
    note "staleness eviction engaged (replay/evicted_cum > 0)"
else
    bad "no evictions recorded despite k=3 (replay/evicted_cum stayed 0)"
fi

last_ckpt=$(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | sort -V | tail -1)
if [ -n "${last_ckpt}" ] && [ -f "${last_ckpt}/replay_buffer.pt" ]; then
    note "checkpoint ${last_ckpt} contains replay_buffer.pt"
else
    bad "no replay_buffer.pt in the latest checkpoint (${last_ckpt:-none found})"
fi

if [ -s "${MEMLOG}" ]; then
    note "peak GPU memory during run (MiB, per GPU):"
    awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "  GPU %s: %d\n", g, m[g] }' \
        "${MEMLOG}" | sort -V
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS — replay buffer: warm-up, sync-per-update, insertion gate, weighted replay, eviction, checkpointing all exercised"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
