#!/usr/bin/env bash
# Fast OOM/sanity smoke for the bf16-sr-adamw FSDP2 dynbsz replay arm
# (grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_dynbsz_bf16-sr-adamw_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh).
#
# Purpose: verify the UNTESTED memory envelope of that arm on the real model —
# bf16 params + torchao _AdamW (bf16 stochastic rounding, fully GPU-resident,
# no fp32 master) sharded over trainer DP=3, with dynamic batching at the
# raised 30720-token micro-batch budget (estimated peak ~52-60 GiB on 80 GB) —
# plus the wiring: torchao import/construction, the dp_actor ESS brake with
# auto-calibrated base + trigger, and the replay loop.
#
# The memory peak is set by the model size, the 30720-token packed
# micro-batches and the sharded optimizer states — NOT by the mini-batch
# size — so the smoke keeps Qwen3-8B and the full 2048/8192 sequence lengths
# and the full token budget, but shrinks the mini-batch to 6 groups
# (6*16=96 seqs, divides DP=3) to reach the first updates fast, and tightens
# the replay horizon (irrelevant to memory) for quick turnover. Validation
# and checkpointing are disabled: fastest possible path to the risky updates.
#
# The run is launched in the background, watched until SMOKE_UPDATES (default
# 2) model updates complete — update 1 captures the auto base_ess_ratio,
# update 2 is the first potentially-braked step and the first with replayed
# (stale) groups in the mix — then torn down with the bracketed pkill
# patterns. Asserts on the log:
#   1. no Traceback / CUDA OOM anywhere (the main check; a missing torchao
#      or a model_dtype/SR misconfiguration also surfaces here);
#   2. >= SMOKE_UPDATES updates completed;
#   3. base_ess_ratio was auto-calibrated from update 1 (value printed);
#   4. replay/ess_base and replay/ess_scaled_lr metrics logged (the dp_actor
#      brake emitted its structured entries and the trainer consumed them).
# Reports per-GPU memory peaks; warns if a trainer GPU (index >= 5 in the
# 5+3 layout) exceeded 70000 MiB (above the expected ~52-60 GiB band) and
# flags > 79000 MiB as OOM-risk territory.
#
# WARNING: tears down ray/vllm/fully_async processes on this host at the end —
# run only on a box this smoke owns (8 free GPUs).
#
# On hosts where the datasets are not at the script's /home/jovyan defaults,
# export TRAIN_FILE/TEST_FILE before invoking.
#
# Usage (training environment activated):
#   bash "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_dynbsz_bf16-sr-adamw_replay_smoke_test.sh"
# Env knobs: SMOKE_TIMEOUT (s, default 3600), SMOKE_UPDATES (default 2),
#            SMOKE_LOG (log path)

set -uo pipefail

cd "$(dirname "$0")/../../../../../.."  # repo root, so the fork's verl shadows the installed one

SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_dynbsz_bf16-sr-adamw_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh"
LOG=${SMOKE_LOG:-logs/smoke_replay_ess_fsdp2_bf16sr_dynbsz_5+3.log}
MEMLOG="${LOG%.log}.gpumem.csv"
WANT_UPDATES=${SMOKE_UPDATES:-2}
DEADLINE=$(($(date +%s) + ${SMOKE_TIMEOUT:-3600}))
mkdir -p "$(dirname "${LOG}")"

# Ray block-buffers worker stdout: without this the milestone prints
# (auto-calibrated base, [Replay] global_steps) sit in worker pipes for many
# minutes and the log-based assertions below misfire even though the run is
# healthy (observed on the Megatron smoke's first remote execution).
export PYTHONUNBUFFERED=1

# The launch script honors these env overrides. Full-size model, sequence
# lengths and 30720-token dynbsz budget (they set the memory peak); small
# mini-batch + short replay horizon + no validation/saves (they don't, and
# they get us to the first updates fastest).
export exp_name="SMOKE-replay-ess-fsdp2-bf16sr-dynbsz-5+3"
export val_before_train=False
export test_freq=-1
export save_freq=-1
export train_prompt_mini_bsz=6 # 6*16=96 seqs, divides trainer DP=3; memory-neutral under dynbsz (budget-sized micro-batches)
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

echo "[smoke] launching bf16-sr dynbsz replay run in background; waiting for ${WANT_UPDATES} updates (timeout ${SMOKE_TIMEOUT:-3600}s); log: ${LOG}"
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

# --- 1. crashes (the main check: does the 30720 budget fit the bf16 recipe?) -
if grep -aqE "CUDA out of memory" "${LOG}"; then
    bad "CUDA OOM — the 30720-token dynbsz budget does NOT fit the bf16-sr recipe:"
    grep -anE "CUDA out of memory" "${LOG}" | head -2
elif grep -aqE "Traceback" "${LOG}"; then
    bad "Traceback found in log (torchao import? model_dtype? dp_actor guard?):"
    grep -anE "Traceback" "${LOG}" | head -3
else
    note "no OOM / Traceback"
fi

# --- 2. enough updates completed ---------------------------------------------
updates=$(grep -ac "\[FullyAsyncTrainer\]\[Replay\] global_steps" "${LOG}")
if [ "${updates}" -ge "${WANT_UPDATES}" ]; then
    note "${updates} model updates completed (wanted ${WANT_UPDATES}: capture + first replayed/braked step)"
else
    bad "only ${updates}/${WANT_UPDATES} updates completed before timeout/exit"
fi

# --- 3. auto base_ess_ratio captured -----------------------------------------
capture_line=$(grep -ao "auto-calibrated ess_scaling.base_ess_ratio=[0-9.]*" "${LOG}" | head -1)
if [ -n "${capture_line}" ]; then
    note "${capture_line} (from the staleness-0 first update)"
else
    bad "no auto-calibration line — base_ess_ratio was never captured (dp_actor staleness/ess entries missing?)"
fi

# --- 4. brake metrics logged --------------------------------------------------
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

# --- 5. GPU memory peaks ------------------------------------------------------
if [ -s "${MEMLOG}" ]; then
    note "peak GPU memory during run (MiB, per GPU):"
    awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "  GPU %s: %d\n", g, m[g] }' \
        "${MEMLOG}" | sort -V
    trainer_peak=$(awk -F', ' '$1 >= 5 { if ($2 > p) p = $2 } END { print p+0 }' "${MEMLOG}")
    if [ "${trainer_peak}" -gt 79000 ]; then
        bad "trainer-GPU peak ${trainer_peak} MiB > 79000 — effectively no headroom, expect OOM under real packing variance"
    elif [ "${trainer_peak}" -gt 70000 ]; then
        note "WARNING: trainer-GPU peak ${trainer_peak} MiB — above the expected ~52-60 GiB band; consider ppo_max_token_len=20480"
    else
        note "trainer-GPU peak ${trainer_peak} MiB — within the expected band, headroom OK"
    fi
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS — bf16-sr + torchao + 30720-token dynbsz survived on the real model; auto base captured, brake metrics flowing"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
