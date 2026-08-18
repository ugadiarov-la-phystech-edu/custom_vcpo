#!/usr/bin/env bash

# One-mini-batch estimator of the on-policy ESS base ratio (rho_on) for the
# Megatron DYNBSZ (packed per-traj, HDO) replay arm. The base is a property
# of the ENTIRE numerical configuration — backend, precision recipe,
# batching — so measure it under the exact production script it will serve;
# point BASE_SCRIPT at a different arm (e.g. the mbs=1 trigger arm) to
# estimate for that one. NOTE: the default dynbsz script requires the
# dyn-batch branch's actor code (_update_policy_per_traj_packed); on the
# base branch it dies at the mbs=1 assert.
#
# Launches the production script (default)
#   grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_dynbsz_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh
# with all controllable seeds set from ${SEED}, lets it generate the first
# (staleness-0, warm-up) mini-batch of B=33 groups and run update 1 — the
# exact measurement ess_scaling.base_ess_ratio=null auto-calibrates from —
# then extracts staleness/ess_ratio (and the clipped variant) from the step:1
# console metrics line, appends "seed,ess_ratio,ess_ratio_clipped" to
# ${RESULTS_FILE}, and tears the run down. Validation and checkpointing are
# disabled; nothing is trained beyond the single update. Requires 'console'
# in trainer.logger (the production script hardcodes it).
#
# Seeds set: data.seed (dataloader prompt shuffle — the knob that actually
# varies the sampled mini-batch across runs) and
# replay_buffer.sampling_seed (inert at update 1, set for hygiene).
# The vLLM sampling seed is
# NOT settable in this fork: RolloutConfig is a strict dataclass without a
# 'seed' field, so a +actor_rollout_ref.rollout.seed override crashes worker
# init (verified 2026-08-18); the engine runs at its default seed, and
# continuous-batching nondeterminism varies generations anyway. NOTE: even a
# fixed seed does not make the first mini-batch reproducible — its membership
# is the first 33 non-degenerate groups to COMPLETE, which depends on
# wall-clock scheduling. Different seeds simply guarantee different prompts,
# so repeated runs sample the distribution of the auto-captured base.
#
# The rollouter's prompt budget (total_rollout_steps) counts prompts BEFORE
# the degenerate-group filter, so a budget of exactly 33 can starve the first
# mini-batch. We leave a generous budget (330) and stop by watching the log
# for the step:1 metrics instead.
#
# SAFETY: refuses to start while any fully_async_main / raylet process is
# alive — the teardown sweep would kill it.
#
# Usage (env activated, GPUs free; cwd-independent):
#   SEED=1000 bash recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/estimate_ess_base_1mb.sh
# Env knobs: SEED (default 42), RESULTS_FILE (default
# logs/ess_base_estimates_megatron_dynbsz.csv, relative to the repo root),
# TIMEOUT_S (default 3600), BASE_SCRIPT (production script to wrap, default
# the dynbsz Megatron trigger arm), EXP_TAG (log-dir name component, default
# megatron-dynbsz).

set -uo pipefail

SEED=${SEED:-42}
RESULTS_FILE=${RESULTS_FILE:-"logs/ess_base_estimates_megatron_dynbsz.csv"}
TIMEOUT_S=${TIMEOUT_S:-3600}
POLL_S=5
STARTUP_GRACE_S=30

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_script="${BASE_SCRIPT:-${script_dir}/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_dynbsz_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh}"
# .../recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer -> repo root
repo_root="$(cd "${script_dir}/../../../../../.." && pwd)"

if [ ! -f "${base_script}" ]; then
    echo "[estimate_ess_base] FATAL: base script not found: ${base_script}" >&2
    exit 2
fi
if [ ! -f "${repo_root}/recipe/fully_async_policy/fully_async_main.py" ]; then
    echo "[estimate_ess_base] FATAL: repo root resolution failed: ${repo_root}" >&2
    exit 2
fi
cd "${repo_root}"

# Pre-flight: never launch (or later pkill) over an existing training run
if pgrep -f '[f]ully_async_main' >/dev/null 2>&1 || pgrep -x 'raylet' >/dev/null 2>&1; then
    echo "[estimate_ess_base] FATAL: a fully_async_main/raylet process is already running —" >&2
    echo "    refusing to launch: the estimator's teardown would kill it." >&2
    exit 2
fi

exp_tag="${EXP_TAG:-megatron-dynbsz}"
exp_name="ESS-base-est seed-${SEED} ${exp_tag} B-33"
exp_name_safe=${exp_name//\//_}
run_log="logs/${exp_name_safe}/launch.log"
mkdir -p "logs/${exp_name_safe}"
mkdir -p "$(dirname "${RESULTS_FILE}")"

# Ensures the header exists exactly once, then appends one row — used by the
# success AND failure paths so a leading NA row cannot produce a headerless
# file (whose later mid-file header would corrupt the driver's summary).
write_result() {
    if [ ! -s "${RESULTS_FILE}" ]; then
        echo "seed,ess_ratio,ess_ratio_clipped" > "${RESULTS_FILE}"
    fi
    echo "${SEED},$1,$2" >> "${RESULTS_FILE}"
}

echo "[estimate_ess_base] seed=${SEED} log=${run_log}"

# setsid: new session/process group so the ray tree dies with one group kill.
# CAVEAT handled below: some setsid builds fork when the child is already a
# group leader, making $! exit immediately — liveness and teardown therefore
# never rely on ${run_pid} alone.
setsid env \
    exp_name="${exp_name}" \
    val_before_train=False \
    test_freq=1000000 \
    save_freq=-1 \
    total_rollout_steps=330 \
    replay_sampling_seed="${SEED}" \
    bash "${base_script}" \
    data.seed="${SEED}" \
    > "${run_log}" 2>&1 &
run_pid=$!

run_alive() {
    kill -0 "${run_pid}" 2>/dev/null && return 0
    pgrep -f '[f]ully_async_main' >/dev/null 2>&1
}

# Kill by the process groups of every process we can attribute to the run
# (launcher pid + any live fully_async_main), then sweep by name for anything
# ray double-forked out of those groups.
cleanup() {
    local pids pgids p g
    pids="${run_pid} $(pgrep -f '[f]ully_async_main' 2>/dev/null || true)"
    pgids=""
    for p in ${pids}; do
        g=$(ps -o pgid= -p "${p}" 2>/dev/null | tr -d ' ' || true)
        [ -n "${g}" ] && pgids="${pgids} ${g}"
    done
    for g in ${pgids}; do kill -TERM -- "-${g}" 2>/dev/null; done
    kill -TERM "${run_pid}" 2>/dev/null
    sleep 20
    for g in ${pgids}; do kill -KILL -- "-${g}" 2>/dev/null; done
    pkill -KILL -f '[f]ully_async_main' 2>/dev/null
    pkill -KILL -f '[r]ay::' 2>/dev/null
    pkill -KILL -x 'raylet' 2>/dev/null
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

elapsed=0
step_line=""
while [ "${elapsed}" -lt "${TIMEOUT_S}" ]; do
    if [ "${elapsed}" -ge "${STARTUP_GRACE_S}" ] && ! run_alive; then
        # One last look at the log: the run may have finished printing step 1
        # and exited/died in the same poll interval.
        step_line=$(grep -m1 -E 'step:1 .*staleness/ess_ratio:' "${run_log}" 2>/dev/null || true)
        if [ -z "${step_line}" ]; then
            echo "[estimate_ess_base] run died before producing step 1 — see ${run_log}"
            write_result NA NA
            exit 1
        fi
        break
    fi
    # The step-1 console metrics line carries the full-precision measurement
    step_line=$(grep -m1 -E 'step:1 .*staleness/ess_ratio:' "${run_log}" 2>/dev/null || true)
    if [ -n "${step_line}" ]; then
        break
    fi
    sleep "${POLL_S}"
    elapsed=$((elapsed + POLL_S))
done

if [ -z "${step_line}" ]; then
    echo "[estimate_ess_base] TIMEOUT after ${TIMEOUT_S}s — see ${run_log}"
    write_result NA NA
    exit 1
fi

ess_ratio=$(printf '%s\n' "${step_line}" | grep -o 'staleness/ess_ratio:[0-9.eE+-]*' | head -1 | cut -d: -f2)
ess_ratio_clipped=$(printf '%s\n' "${step_line}" | grep -o 'staleness/ess_ratio_clipped:[0-9.eE+-]*' | head -1 | cut -d: -f2)
ess_ratio=${ess_ratio:-NA}
ess_ratio_clipped=${ess_ratio_clipped:-NA}

write_result "${ess_ratio}" "${ess_ratio_clipped}"
echo "[estimate_ess_base] seed=${SEED} ess_ratio=${ess_ratio} ess_ratio_clipped=${ess_ratio_clipped} -> ${RESULTS_FILE}"
# cleanup runs via the EXIT trap
exit 0
