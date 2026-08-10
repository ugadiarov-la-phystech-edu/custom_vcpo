#!/usr/bin/env bash
# Fast memory smoke for the ESS-braked replay arm
# (grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=auto.sh).
#
# Purpose: verify the per-trajectory update path SURVIVES ON THE REAL MODEL
# with its extra grad-sized accumulation buffer (~16 GiB bf16 on top of the
# ~58 GB HDO trainer footprint -> ~74 GB peak on 80 GB cards). Memory peak is
# set by the model size, the single-sequence 10240-token micro-batches and
# the buffers — NOT by the mini-batch size — so the smoke keeps Qwen3-8B and
# the full 2048/8192 sequence lengths but shrinks the mini-batch to 6 groups
# (6*16=96 seqs, divides DP=3) to reach the first updates fast, and tightens
# the replay horizon (irrelevant to memory) for quick turnover.
#
# The run is launched in the background, watched until SMOKE_UPDATES (default
# 2) model updates complete — update 1 captures the auto base_ess_ratio,
# update 2 is the first actually-braked step — then torn down with the
# bracketed pkill patterns. Asserts on the log:
#   1. no Traceback / CUDA OOM anywhere (the main check);
#   2. the per-traj accumulation buffer was actually allocated
#      ("[vcpo] allocated grad accum buffers: ~16 GiB");
#   3. >= SMOKE_UPDATES updates completed;
#   4. base_ess_ratio was auto-calibrated from update 1 (value printed);
#   5. replay/ess_base and replay/ess_scaled_lr metrics logged.
# Reports per-GPU memory peaks and warns if a trainer GPU exceeded 79000 MiB
# (thin headroom on 80 GB cards).
#
# WARNING: tears down ray/vllm/fully_async processes on this host at the end —
# run only on a box this smoke owns (8 free GPUs).
#
# On hosts where the datasets are not at the script's /home/jovyan defaults,
# export TRAIN_FILE/TEST_FILE before invoking.
#
# Usage (training environment activated):
#   bash "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=auto_smoke_test_memory.sh"
# Env knobs: SMOKE_TIMEOUT (s, default 3600), SMOKE_UPDATES (default 2),
#            SMOKE_LOG (log path)

set -uo pipefail

cd "$(dirname "$0")/../../../../../.."  # repo root, so the fork's verl shadows the installed one

SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=auto.sh"
LOG=${SMOKE_LOG:-logs/smoke_replay_ess_memory_5+3.log}
MEMLOG="${LOG%.log}.gpumem.csv"
WANT_UPDATES=${SMOKE_UPDATES:-2}
DEADLINE=$(($(date +%s) + ${SMOKE_TIMEOUT:-3600}))
mkdir -p "$(dirname "${LOG}")"

# Ray block-buffers worker stdout: without this the milestone prints
# ([vcpo] alloc, auto-calibrated base, [Replay] global_steps) sit in worker
# pipes for many minutes and the log-based assertions below misfire even
# though the run is healthy (observed on the first remote execution).
export PYTHONUNBUFFERED=1

# The launch script honors these env overrides. Full-size model and sequence
# lengths (they set the memory peak); small mini-batch + short replay horizon
# (they don't).
export exp_name="SMOKE-replay-ess-memory-5+3"
export val_before_train=False
export test_freq=-1
export save_freq=-1
export train_prompt_mini_bsz=6 # 6*16=96 seqs, divides trainer DP=3; memory-neutral (1-seq micro-batches)
export replay_tau=2
export replay_staleness_threshold=2
export replay_requires_mini_batches=1
export staleness_threshold=2.0

# Sample per-GPU memory every 10 s for the peak report.
if command -v nvidia-smi >/dev/null; then
    ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 10; done ) \
        > "${MEMLOG}" 2>/dev/null &
    MEM_PID=$!
fi

cleanup() {
    [ -n "${MEM_PID:-}" ] && kill "${MEM_PID}" 2>/dev/null && MEM_PID=""
    # Bracketed patterns: never match this script or the ssh session.
    pkill -f "fully_async_ma[i]n" 2>/dev/null
    sleep 5
    pkill -f "ray:[:]" 2>/dev/null
    pkill -f "rayle[t]" 2>/dev/null
    pkill -f "vll[m]" 2>/dev/null
}
trap cleanup EXIT

echo "[smoke] launching ESS-braked replay run in background; waiting for ${WANT_UPDATES} updates (timeout ${SMOKE_TIMEOUT:-3600}s); log: ${LOG}"
bash "${SCRIPT}" > "${LOG}" 2>&1 &
RUN_PID=$!

updates=0
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    updates=$(grep -ac "\[FullyAsyncTrainer\]\[Replay\] global_steps" "${LOG}" 2>/dev/null)
    if grep -aqE "Traceback|CUDA out of memory" "${LOG}"; then
        break # fail fast on crash
    fi
    if [ "${updates}" -ge "${WANT_UPDATES}" ]; then
        sleep 30 # let the last update's metrics flush into the log
        break
    fi
    if ! kill -0 "${RUN_PID}" 2>/dev/null; then
        break # launch script exited on its own
    fi
    sleep 20
done
cleanup
trap - EXIT

fail=0
note() { echo "[smoke] $*"; }
bad()  { echo "[smoke][FAIL] $*"; fail=1; }

# --- 1. crashes (the main check: does the 3rd grad-sized region fit?) -------
if grep -aqE "CUDA out of memory" "${LOG}"; then
    bad "CUDA OOM — the per-traj buffer does NOT fit this configuration:"
    grep -anE "CUDA out of memory" "${LOG}" | head -2
elif grep -aqE "Traceback" "${LOG}"; then
    bad "Traceback found in log:"
    grep -anE "Traceback" "${LOG}" | head -3
else
    note "no OOM / Traceback"
fi

# --- 2. the extra buffer was actually allocated ------------------------------
alloc_line=$(grep -ao "\[vcpo\] allocated grad accum buffers: [0-9.]* GiB[^\"]*" "${LOG}" | head -1)
if [ -n "${alloc_line}" ]; then
    note "per-traj accumulation buffer allocated: ${alloc_line}"
else
    bad "no '[vcpo] allocated grad accum buffers' line — per-traj path (and its buffer) never engaged"
fi

# --- 3. enough updates completed ---------------------------------------------
updates=$(grep -ac "\[FullyAsyncTrainer\]\[Replay\] global_steps" "${LOG}")
if [ "${updates}" -ge "${WANT_UPDATES}" ]; then
    note "${updates} model updates completed (wanted ${WANT_UPDATES}: capture + first braked step)"
else
    bad "only ${updates}/${WANT_UPDATES} updates completed before timeout/exit"
fi

# --- 4. auto base_ess_ratio captured -----------------------------------------
capture_line=$(grep -ao "auto-calibrated ess_scaling.base_ess_ratio=[0-9.]*" "${LOG}" | head -1)
if [ -n "${capture_line}" ]; then
    note "${capture_line} (from the staleness-0 first update)"
else
    bad "no auto-calibration line — base_ess_ratio was never captured"
fi

# --- 5. brake metrics logged --------------------------------------------------
if grep -aq "replay/ess_base:" "${LOG}"; then
    note "replay/ess_base logged"
else
    bad "replay/ess_base never appeared in step metrics"
fi
if grep -aq "replay/ess_scaled_lr:" "${LOG}"; then
    note "effective lr logged: $(grep -ao 'replay/ess_scaled_lr:[0-9.e-]*' "${LOG}" | tail -1)"
else
    bad "replay/ess_scaled_lr never appeared in step metrics"
fi

# --- 6. GPU memory peaks ------------------------------------------------------
if [ -s "${MEMLOG}" ]; then
    note "peak GPU memory during run (MiB, per GPU):"
    awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "  GPU %s: %d\n", g, m[g] }' \
        "${MEMLOG}" | sort -V
    trainer_peak=$(awk -F', ' '$1 >= 5 { if ($2 > p) p = $2 } END { print p+0 }' "${MEMLOG}")
    if [ "${trainer_peak}" -gt 79000 ]; then
        note "WARNING: trainer-GPU peak ${trainer_peak} MiB > 79000 — headroom is thin, expect OOM risk under longer sequences"
    else
        note "trainer-GPU peak ${trainer_peak} MiB — headroom OK"
    fi
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS — per-traj path + extra grad buffer survived on the real model; auto base captured, brake metrics flowing"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
