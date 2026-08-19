#!/usr/bin/env bash

# Multi-seed driver for estimate_ess_base_1mb.sh: runs the one-mini-batch
# on-policy ESS ratio (rho_on) estimator — a diagnostic; the min-ESS brake
# uses no measured reference (Megatron dynbsz arm by default; BASE_SCRIPT /
# EXP_TAG / RESULTS_FILE pass through to the inner script) N times with distinct seeds (sequentially —
# each run needs all 8 GPUs) and prints mean/std/min/max of the collected
# ess_ratio samples at the end. Results accumulate in ${RESULTS_FILE}
# (seed,ess_ratio,ess_ratio_clipped; failed runs record NA and are excluded
# from the summary, as is any non-numeric junk).
#
# Between runs it waits until nvidia-smi reports no compute processes (up to
# ${GPU_WAIT_S}) so the next vLLM engine init never races the previous run's
# teardown for GPU memory.
#
# Usage (env activated, GPUs free; cwd-independent):
#   bash recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/estimate_ess_base_multiseed.sh [N]
# Env knobs: N_RUNS (default 5; positional arg wins), SEED_BASE (default
# 1000; run i uses SEED_BASE+i), RESULTS_FILE (default
# logs/ess_base_estimates_megatron_dynbsz.csv, relative to the repo root),
# COOLDOWN_S floor
# pause between runs (default 30), GPU_WAIT_S max wait for GPU release
# (default 300), TIMEOUT_S per run (inner default 3600).

set -uo pipefail

N_RUNS=${1:-${N_RUNS:-7}}
SEED_BASE=${SEED_BASE:-1000}
RESULTS_FILE=${RESULTS_FILE:-"logs/ess_base_estimates_megatron_dynbsz.csv"}
COOLDOWN_S=${COOLDOWN_S:-30}
GPU_WAIT_S=${GPU_WAIT_S:-300}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
inner="${script_dir}/estimate_ess_base_1mb.sh"

if [ ! -f "${inner}" ]; then
    echo "[multiseed] FATAL: inner script not found: ${inner}" >&2
    exit 2
fi

wait_for_gpu_release() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local waited=0
    while [ "${waited}" -lt "${GPU_WAIT_S}" ]; do
        if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "[multiseed] WARNING: GPUs still busy after ${GPU_WAIT_S}s — launching anyway" >&2
}

for i in $(seq 0 $((N_RUNS - 1))); do
    seed=$((SEED_BASE + i))
    echo "===== [multiseed] run $((i + 1))/${N_RUNS} seed=${seed} ====="
    SEED="${seed}" RESULTS_FILE="${RESULTS_FILE}" \
        BASE_SCRIPT="${BASE_SCRIPT:-}" EXP_TAG="${EXP_TAG:-}" bash "${inner}" \
        || echo "[multiseed] run with seed=${seed} failed — see its output above"
    if [ "$i" -lt $((N_RUNS - 1)) ]; then
        sleep "${COOLDOWN_S}"
        wait_for_gpu_release
    fi
done

# The inner script resolves RESULTS_FILE relative to the repo root; do the
# same here for the summary.
repo_root="$(cd "${script_dir}/../../../../../.." && pwd)"
case "${RESULTS_FILE}" in
    /*) results_path="${RESULTS_FILE}" ;;
    *) results_path="${repo_root}/${RESULTS_FILE}" ;;
esac

echo "===== [multiseed] results (${results_path}) ====="
if [ ! -s "${results_path}" ]; then
    echo "no results file produced"
    exit 1
fi
cat "${results_path}"
# Guard: only rows whose ess_ratio field parses as a number enter the
# summary (excludes NA rows and any stray header/junk lines). LC_ALL=C:
# under a decimal-comma locale awk would parse "0.066" as 0.
LC_ALL=C awk -F, '$2 ~ /^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/ {
    n += 1; s += $2; ss += $2 * $2
    if (min == "" || $2 + 0 < min + 0) min = $2
    if (max == "" || $2 + 0 > max + 0) max = $2
}
END {
    if (n == 0) { print "no successful runs"; exit 1 }
    mean = s / n
    std = (n > 1) ? sqrt((ss - n * mean * mean) / (n - 1)) : 0
    printf "n=%d mean=%.6g std=%.6g min=%.6g max=%.6g\n", n, mean, std, min, max
}' "${results_path}"
