#!/usr/bin/env bash

set -uo pipefail

SEED=${SEED:-42}
RESULTS_FILE=${RESULTS_FILE:-"logs/ess_base_estimates_megatron_dynbsz.csv"}
TIMEOUT_S=${TIMEOUT_S:-3600}
POLL_S=5
STARTUP_GRACE_S=30

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_script="${BASE_SCRIPT:-${script_dir}/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_dynbsz_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh}"
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

write_result() {
    if [ ! -s "${RESULTS_FILE}" ]; then
        echo "seed,ess_ratio,ess_ratio_clipped" > "${RESULTS_FILE}"
    fi
    echo "${SEED},$1,$2" >> "${RESULTS_FILE}"
}

echo "[estimate_ess_base] seed=${SEED} log=${run_log}"

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
        step_line=$(grep -m1 -E 'step:1 .*staleness/ess_ratio:' "${run_log}" 2>/dev/null || true)
        if [ -z "${step_line}" ]; then
            echo "[estimate_ess_base] run died before producing step 1 — see ${run_log}"
            write_result NA NA
            exit 1
        fi
        break
    fi
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
exit 0
