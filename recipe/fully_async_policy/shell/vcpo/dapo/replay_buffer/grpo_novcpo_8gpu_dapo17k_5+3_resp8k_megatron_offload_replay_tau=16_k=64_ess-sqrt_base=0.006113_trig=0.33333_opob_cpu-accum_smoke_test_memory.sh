#!/usr/bin/env bash
# Fast memory/error smoke for the 5+3 OPOB replay arm with the twin's HDO + bf16-master optimizer
# and host-resident OPOB accumulators
# (grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=0.006113_trig=0.33333_opob_cpu-accum.sh).
#
# Purpose: verify that single-backward OPOB with G_R / G_S in pinned host
# memory SURVIVES ON THE REAL MODEL in the 5+3 / TP=1 layout with the plain
# HDO + bf16-master optimizer: per trainer GPU the params, Megatron's main grad
# buffer, the bf16 master shard, the 10240-token activations and NCCL context
# (~58 GB estimated, the buffer-free twin's footprint; no fp32 gradient
# copy with the precision-aware optimizer); per trainer rank ~97 GB of host RAM (two fp32 accumulators,
# the bf16 staging copy, HDO's Adam moments). Memory peak is set by the model
# size, the single-sequence micro-batches and the buffers — NOT by the
# mini-batch size — so the smoke keeps Qwen3-8B and the full 2048/8192
# sequence lengths but shrinks the mini-batch to 3 groups (3*16=48 seqs: one
# whole group per DP rank, the minimum group-scope OPOB allows at DP=3) to
# reach the first updates fast, and tightens the replay horizon (irrelevant
# to memory) for quick turnover.
#
# The run is launched in the background, watched until SMOKE_UPDATES (default
# 2) model updates complete — update 1 exercises the OPOB group close on
# fresh groups, update 2 the replay draw + weight sync + eviction on top —
# then torn down with the bracketed pkill patterns. Asserts on the log:
#   1. no Traceback / CUDA OOM anywhere (the main check);
#   2. BOTH OPOB accumulators were allocated ON THE HOST in fp32 (two
#      "[vcpo] allocated grad accum buffers" lines with devices={'cpu'} and
#      dtypes={'torch.float32'}, ~30.5 GiB each at TP=1) plus the pinned
#      staging copy ("[vcpo] allocated pinned staging buffers", ~15.3 GiB);
#   3. >= SMOKE_UPDATES updates completed;
#   4. the opob/* diagnostics (baseline_mean, weight_conc_mean, ...) and the
#      brake metrics (replay/ess_base, replay/ess_scaled_lr) are logged.
# Reports per-GPU memory peaks (trainer GPUs are indices 5-7: Ray places the
# rollout pool on the first n_gpus_rollout devices) and warns above 79000 MiB,
# plus the per-update actor/grad_norm (a healthy value is O(0.1-1)).
#
# WARNING: tears down ray/vllm/fully_async processes on this host at the end —
# run only on a box this smoke owns (8 free GPUs).
#
# On hosts where the datasets are not at the script's /home/jovyan defaults,
# export TRAIN_FILE/TEST_FILE before invoking.
#
# Usage (training environment activated, from the repo root or anywhere):
#   bash "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=0.006113_trig=0.33333_opob_cpu-accum_smoke_test_memory.sh"
# Env knobs: SMOKE_UPDATES (default 2), SMOKE_LOG (log path),
#            SMOKE_MINI_BSZ (default 3; must divide by 3)
# No timeout: the watch loop runs until the updates complete, the run
# crashes, or the launch script exits; Ctrl-C triggers the same teardown.

set -uo pipefail

cd "$(dirname "$0")/../../../../../.."  # repo root, so the fork's verl shadows the installed one

SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=0.006113_trig=0.33333_opob_cpu-accum.sh"
LOG=${SMOKE_LOG:-logs/smoke_replay_opob_cpu-accum-bf16m_5+3.log}
MEMLOG="${LOG%.log}.gpumem.csv"
WANT_UPDATES=${SMOKE_UPDATES:-2}
mkdir -p "$(dirname "${LOG}")"

# Ray block-buffers worker stdout: without this the milestone prints
# ([vcpo] alloc, [Replay] global_steps) sit in worker pipes for many minutes
# and the log-based assertions below misfire even though the run is healthy.
export PYTHONUNBUFFERED=1

# The launch script honors these env overrides. Full-size model, 5+3/TP=1
# layout, optimizer and sequence lengths stay (they set the memory peak);
# small mini-batch + short replay horizon (they don't). No validation, no
# checkpoints.
export exp_name="SMOKE-replay-opob-cpu-accum-bf16m-5+3"
export val_before_train=False
export test_freq=-1
export save_freq=-1
export train_prompt_mini_bsz=${SMOKE_MINI_BSZ:-3} # 3*16=48 seqs: one whole group per DP rank (DP=3)
export replay_tau=2
export replay_staleness_threshold=2
export replay_requires_mini_batches=1
export staleness_threshold=2.0

if [ $(( train_prompt_mini_bsz % 3 )) -ne 0 ]; then
    echo "[smoke][FAIL] SMOKE_MINI_BSZ=${train_prompt_mini_bsz} must divide by trainer DP=3 (group-scope OPOB keeps whole groups per rank)"
    exit 2
fi

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

echo "[smoke] launching OPOB 5+3/TP=1 cpu-accum bf16-masters replay run in background; waiting for ${WANT_UPDATES} updates (no timeout); log: ${LOG}"
bash "${SCRIPT}" > "${LOG}" 2>&1 &
RUN_PID=$!

updates=0
while true; do
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

# --- 1. crashes (the main check: the twin's footprint + nothing extra on the GPU) ---
if grep -aqE "CUDA out of memory" "${LOG}"; then
    bad "CUDA OOM — the 5+3 cpu-accum layout does NOT fit this configuration:"
    grep -anE "CUDA out of memory" "${LOG}" | head -2
elif grep -aqE "Traceback" "${LOG}"; then
    bad "Traceback found in log:"
    grep -anE "Traceback" "${LOG}" | head -3
else
    note "no OOM / Traceback"
fi

# --- 2. both OPOB accumulators allocated ON THE HOST in fp32, plus staging ----
# Every trainer rank prints once per allocation: 2 buffers x 3 ranks = 6 lines
# expected; require at least 2 (one accum + one score) to tolerate Ray's
# per-worker stdout dedup.
alloc_lines=$(grep -ao "\[vcpo\] allocated grad accum buffers: [0-9.]* GiB[^\"]*" "${LOG}")
alloc_count=$(printf '%s\n' "${alloc_lines}" | grep -c "allocated")
if [ "${alloc_count}" -ge 2 ]; then
    first=$(printf '%s\n' "${alloc_lines}" | head -1)
    note "OPOB accumulators allocated (${alloc_count} lines): ${first}"
    if ! printf '%s\n' "${first}" | grep -q "devices={'cpu'}"; then
        bad "accumulators are not on the host — grad_baselining.accum_device=cpu did not reach the actor"
    fi
    if ! printf '%s\n' "${first}" | grep -q "torch.float32"; then
        bad "accumulators are not fp32 — grad_baselining.accum_dtype=float32 did not reach the actor"
    fi
elif [ "${alloc_count}" -eq 1 ]; then
    bad "only one '[vcpo] allocated grad accum buffers' line — the score buffer was not allocated (grad_baselining off?)"
else
    bad "no '[vcpo] allocated grad accum buffers' line — per-traj OPOB path never engaged"
fi
staging_line=$(grep -ao "\[vcpo\] allocated pinned staging buffers: [0-9.]* GiB" "${LOG}" | head -1)
if [ -n "${staging_line}" ]; then
    note "${staging_line}"
else
    bad "no pinned staging buffer allocation — the d2h accumulation path never engaged"
fi
# Sanity on the optimizer: this arm keeps the twin's HDO + bf16-master optimizer.
if grep -aq "main_params_dtype': torch.bfloat16" "${LOG}"; then
    note "optimizer runs with bf16 master weights (main_params_dtype=bfloat16 in the resolved optimizer config), as in the twin"
else
    bad "optimizer config lacks main_params_dtype=bfloat16 — the precision-aware overrides did not land"
fi

# --- 3. enough updates completed ---------------------------------------------
updates=$(grep -ac "\[FullyAsyncTrainer\]\[Replay\] global_steps" "${LOG}")
if [ "${updates}" -ge "${WANT_UPDATES}" ]; then
    note "${updates} model updates completed (wanted ${WANT_UPDATES}: OPOB close on fresh groups + replay draw/sync/eviction)"
else
    bad "only ${updates}/${WANT_UPDATES} updates completed before the run crashed/exited"
fi

# --- 4. OPOB diagnostics + brake metrics logged ------------------------------
for key in opob/baseline_mean opob/baseline_abs_mean opob/weight_conc_mean opob/dominant_pos_frac opob/zeroed_frac opob/groups; do
    if grep -aq "${key}:" "${LOG}"; then
        note "${key} logged: $(grep -ao "${key}:[0-9.e-]*" "${LOG}" | tail -1)"
    else
        bad "${key} never appeared in step metrics — actor/opob_records did not reach the trainer"
    fi
done
if grep -aq "replay/ess_base:" "${LOG}"; then
    note "replay/ess_base logged: $(grep -ao 'replay/ess_base:[0-9.e-]*' "${LOG}" | tail -1)"
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
    note "peak GPU memory during run (MiB, per GPU; 0-4 rollout, 5-7 trainer):"
    awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "  GPU %s: %d\n", g, m[g] }' \
        "${MEMLOG}" | sort -V
    trainer_peak=$(awk -F', ' '$1 >= 5 { if ($2 > p) p = $2 } END { print p+0 }' "${MEMLOG}")
    if [ "${trainer_peak}" -gt 79000 ]; then
        note "WARNING: trainer-GPU peak ${trainer_peak} MiB > 79000 — headroom is thin, expect OOM risk under longer sequences"
    else
        note "trainer-GPU peak ${trainer_peak} MiB — headroom OK (estimate was ~58 GB, the twin's)"
    fi
fi

# --- 6. per-update gradient norms and update timing --------------------------
grep -a "training/global_step" "${LOG}" | grep -a "critic/score/mean" | while read -r line; do
    f() { echo "${line}" | grep -oE "$1:[-0-9.e+]+" | head -1 | cut -d: -f2; }
    note "update $(f 'training/global_step'): actor/grad_norm=$(f 'actor/grad_norm') update_actor=$(f 'timing_s/update_actor')s step=$(f 'timing_s/step')s max_mem_alloc=$(f 'perf/max_memory_allocated_gb')GB"
done

echo
if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS — single-backward OPOB with host-resident fp32 accumulators + the twin's bf16-master HDO optimizer survived on the real model at 5+3/TP=1; opob/* and brake metrics flowing"
else
    echo "[smoke] FAIL — see messages above; full log: ${LOG}"
fi
exit "${fail}"
