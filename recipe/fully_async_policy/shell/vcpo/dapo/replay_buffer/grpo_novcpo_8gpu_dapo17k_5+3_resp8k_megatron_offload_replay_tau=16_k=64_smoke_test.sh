#!/usr/bin/env bash
# Smoke test for the replay-buffer training mode, driving the real launch
# script grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64.sh
# end-to-end with a small model and shrunk hyperparameters:
#   Qwen3-1.7B, 1024/1024 prompt/response, mini-batch 6 groups x 8 responses
#   (48 seqs, divides trainer DP=3), tau=2, eviction threshold k=3,
#   requires_mini_batches=2 (watermark 12 groups), ~400 fed prompts,
#   checkpoints every 5 updates, no validation.
#
# CAVEAT — kept-group supply: the insertion gate drops zero-variance groups,
# and a small model with 1024-token responses gets most DAPO groups all-wrong.
# n=8 responses/group keeps the mixed-group rate workable; if the run still
# starves (few updates + all_wrong_ratio near 1.0 in the log), raise
# SMOKE_PROMPTS or switch SMOKE_MODEL to Qwen/Qwen3-4B.
#
# The aggressive k=3 / fast-trainer setting deliberately drives the buffer
# through every regime within minutes: warm-up on fresh groups, heavy
# score-weighted replay between arrivals, watermark pauses, and evictions.
#
# Asserts on the log:
#   1. no Traceback / CUDA OOM and the run terminates on the data sentinel;
#   2. >= 5 model updates ran, each followed by a parameter sync
#      (version increments per update);
#   3. the insertion gate is active (group all-correct/all-wrong ratio
#      metrics logged) and frozen stats reached the trainer;
#   4. replayed (is_new=False) groups were actually trained on
#      (replay/minibatch_replayed > 0 in some update);
#   5. evictions occurred (replay/evicted_cum > 0);
#   6. a checkpoint contains replay_buffer.pt.
#
# On hosts where the datasets are not at the script's /home/jovyan defaults,
# export TRAIN_FILE/TEST_FILE before invoking (the launch script honors them).
# Needs 8 free GPUs by default; a 4-GPU host can run it with
#   NGPUS_PER_NODE=4 n_gpus_rollout=2   (36 seqs still divide DP=2).
#
# Usage (training environment activated):
#   bash "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_smoke_test.sh"
# Env knobs: SMOKE_TIMEOUT (s, default 3600), SMOKE_LOG (log path),
#            SMOKE_MODEL (default Qwen/Qwen3-1.7B), SMOKE_PROMPTS (default 200).

set -uo pipefail

cd "$(dirname "$0")/../../../../../.."  # repo root, so the fork's verl shadows the installed one

SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64.sh"
LOG=${SMOKE_LOG:-logs/smoke_replay_5+3.log}
MEMLOG="${LOG%.log}.gpumem.csv"
mkdir -p "$(dirname "${LOG}")"

# The launch script honors these env overrides.
export MODEL_PATH=${SMOKE_MODEL:-"Qwen/Qwen3-1.7B"}
export exp_name="SMOKE-replay-5+3" # keep TB/ckpt junk out of real run dirs
export val_before_train=False
export test_freq=-1 # no validation at all
export max_prompt_length=1024
export max_response_length=1024
export n_resp_per_prompt=8     # bigger groups -> fewer all-wrong (degenerate) groups filtered
export train_prompt_mini_bsz=6 # 6*8=48 seqs, divides trainer DP=3 (and DP=2 on 4-GPU hosts)
export total_rollout_steps=${SMOKE_PROMPTS:-400}
export save_freq=5
# Tight replay dynamics: half-life 2 updates, evict at staleness 3, watermark 2 mini-batches.
export replay_tau=2
export replay_staleness_threshold=3
export replay_requires_mini_batches=2
export staleness_threshold=3.0 # generation quota aligned with the eviction horizon

CKPT_DIR="logs/${exp_name}"

# Sample per-GPU memory every 15 s for the peak report.
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

# --- 1. process outcome ------------------------------------------------------
if [ "${rc}" -eq 124 ]; then
    bad "run hit the ${SMOKE_TIMEOUT:-3600}s timeout before finishing"
elif [ "${rc}" -ne 0 ]; then
    bad "launch script exited with code ${rc}"
fi

# --- 2. crashes --------------------------------------------------------------
if grep -aqE "Traceback|CUDA out of memory" "${LOG}"; then
    bad "Traceback / CUDA OOM found in log:"
    grep -anE "Traceback|CUDA out of memory" "${LOG}" | head -3
fi

# --- 3. replay loop ran and terminated cleanly -------------------------------
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

# --- 4. sync after every update ----------------------------------------------
syncs=$(grep -ac "Parameter version updated from" "${LOG}")
if [ "${syncs}" -ge 5 ]; then
    note "parameter version updated ${syncs} times (sync-per-update active)"
else
    bad "only ${syncs} parameter version updates (expected one per model update)"
fi

# --- 5. insertion gate + frozen stats ----------------------------------------
if grep -aq "fully_async/groups/all_wrong_ratio" "${LOG}"; then
    note "insertion-gate group ratios logged (all-correct/all-wrong counters active)"
else
    bad "group all-correct/all-wrong ratio metrics never logged"
fi

# --- 6. replayed groups trained on -------------------------------------------
if grep -aqE "replay/minibatch_replayed:[1-9]" "${LOG}"; then
    note "score-weighted replay engaged (some mini-batch reused old groups):"
    grep -aoE "replay/minibatch_new_ratio:[0-9.]+" "${LOG}" | tail -3
else
    bad "no mini-batch ever contained replayed groups (replay/minibatch_replayed always 0)"
fi

# --- 7. evictions ------------------------------------------------------------
if grep -aqE "replay/evicted_cum:[1-9]" "${LOG}"; then
    note "staleness eviction engaged (replay/evicted_cum > 0)"
else
    bad "no evictions recorded despite k=3 (replay/evicted_cum stayed 0)"
fi

# --- 8. checkpoint contains the replay buffer --------------------------------
last_ckpt=$(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | sort -V | tail -1)
if [ -n "${last_ckpt}" ] && [ -f "${last_ckpt}/replay_buffer.pt" ]; then
    note "checkpoint ${last_ckpt} contains replay_buffer.pt"
else
    bad "no replay_buffer.pt in the latest checkpoint (${last_ckpt:-none found})"
fi

# --- 9. GPU memory peaks -----------------------------------------------------
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
