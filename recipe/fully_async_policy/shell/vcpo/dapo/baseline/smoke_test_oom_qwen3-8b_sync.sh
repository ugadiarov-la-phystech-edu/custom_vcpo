#!/usr/bin/env bash
# =============================================================================
# smoke_test_oom_qwen3-8b_sync.sh
#
# Fastest possible OUT-OF-MEMORY check of the synchronous Qwen3-8B arm
#   main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh
# on the real 8-GPU colocated layout. It runs the real arm script - there is no
# second copy of the configuration to drift - and overrides only what makes it
# short. Every memory-relevant setting is left EXACTLY as the arm has it:
#
#   * Qwen3-8B, tp1 x 8 DP ranks, hybrid engine, NO param/optimizer/grad offload
#     (the resident trainer is what caps vLLM at 0.5 - see the arm header);
#   * gpu_memory_utilization=0.5, max_num_batched_tokens=10240, prompt 2048 /
#     RESPONSE 8192 (the arm's length, NOT shortened: the training-phase peak is
#     set by the longest micro-batch, so a capped 10,240-token sequence must
#     occur - at Qwen3-8B's ~25% cap rate over 256 sequences that is certain);
#   * ppo_micro_batch_size_per_gpu=1, full recompute, HDO + bf16 masters,
#     calculate_log_probs / calculate_entropy on, token-mean, lr and loss untouched.
#
# What is shortened:
#   * 2 ROLLOUT STEPS (trainer.total_training_steps=2, a hydra override the arm
#     forwards via "$@"), so the run stops on its own;
#   * train_batch_size 16 prompts, mini-batch 8 prompts (instead of 128 / 32):
#     16 x 16 = 256 sequences generated per step (32 per GPU, one KV fill at
#     ~150k tokens per GPU), 8 x 16 = 128 per optimizer step (16 per DP rank),
#     TWO optimizer updates per step so the multi-mini-batch path (old_log_prob
#     pass + ratio != 1 on update 2) is exercised. Per-GPU peak memory does not
#     depend on the batch size (micro-batch 1 + gradient accumulation), so the
#     OOM verdict transfers to the real 128/32 geometry.
#   * NO validation before training and test_freq beyond the run. The trainer
#     still validates and saves once at the LAST step regardless of the
#     frequencies (ray_trainer.py: `is_last_step or global_steps % freq == 0`),
#     which is wanted: the final hf_model save (all-rank gather + rank-0 write of
#     ~16.4 GB) is the last memory spike of a real step. The unavoidable final
#     validation runs on a 2-row parquet this script writes, so it is near-free.
#
# Every phase that can OOM in the real arm is therefore hit once or twice:
#   vLLM init (free-memory check against 0.5 x 79.65 GiB with the trainer
#   resident) -> generation at full length -> old_log_prob forward over 256
#   sequences -> 2 x (backward at up to 10,240 tokens + optimizer step) ->
#   weight re-sync into vLLM -> 2-row validation -> hf_model save.
#
# While the arm runs, nvidia-smi is sampled every 2 s into a CSV; afterwards the
# PEAK used memory per GPU and the headroom against the card are printed, the
# driver log is scanned for OOM signatures (torch OutOfMemoryError, NCCL alloc
# failures, vLLM's "less than desired GPU memory utilization" abort), and the
# final checkpoint directory is checked for a complete hf_model.
#
# Expected duration on 8xH100: ~10-20 min after model load (two ~3-5 min
# generations at 8k tokens, two short update phases, one ~1-2 min save).
#
# Usage:  bash smoke_test_oom_qwen3-8b_sync.sh
#         gpu_memory_utilization=0.55 bash smoke_test_oom_qwen3-8b_sync.sh   # probe the ceiling
# Env:    ARM_SCRIPT, SMOKE_DIR (default logs/smoke_oom), MODEL_PATH, TRAIN_FILE,
#         FULL_TEST_FILE, gpu_memory_utilization, train_prompt_bsz, train_prompt_mini_bsz
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
FULL_TEST_FILE=${FULL_TEST_FILE:-"/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-OOM-qwen3-8b-sync-B16xn16-mini8"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
SMOKE_DIR=${SMOKE_DIR:-"logs/smoke_oom"}
mkdir -p -- "${SMOKE_DIR}"

# `--cfg job` and friends make the arm print its config and exit: skip the
# parquet, the sampler and the checks, just compose.
compose_only=0
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) compose_only=1 ;; esac
done

# A 2-row validation set, so the trainer's unavoidable final validation pass is
# near-free instead of a 2x AIME sweep.
TINY_TEST_FILE="${SMOKE_DIR}/tiny_val.parquet"
if [[ "${compose_only}" == 0 ]]; then
    python - "${FULL_TEST_FILE}" "${TINY_TEST_FILE}" <<'PY'
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
pd.read_parquet(src).head(2).to_parquet(dst)
print(f"[smoke] wrote {dst} from {src}", file=sys.stderr)
PY
fi

# ---- 2 steps, real lengths, real memory settings -----------------------------
export MODEL_PATH TRAIN_FILE exp_name
export TEST_FILE="['${TINY_TEST_FILE}']"
export train_prompt_bsz=${train_prompt_bsz:-16}        # 16 x 16 = 256 seqs generated per step
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-8}  # 8 x 16 = 128 seqs per update -> 2 updates/step
export n_resp_per_prompt=${n_resp_per_prompt:-16}      # the arm's own
export max_response_length=${max_response_length:-8192}  # the arm's own - do NOT shorten (see header)
export test_freq=${test_freq:-1000000}                 # never inside the run; the last step validates anyway
export save_freq=${save_freq:-1000000}                 # never inside the run; the last step saves anyway
export val_before_train=False
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}  # the arm's default; raise to probe the ceiling

RUN_LOG="${SMOKE_DIR}/${exp_name}.log"
GPU_CSV="${SMOKE_DIR}/${exp_name}_nvidia-smi.csv"

if [[ "${compose_only}" == 1 ]]; then
    bash "${ARM_SCRIPT}" trainer.total_training_steps=2 "$@"
    exit 0
fi

# ---- GPU memory sampler ------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits -l 2 \
    > "${GPU_CSV}" 2>/dev/null &
sampler_pid=$!
trap 'kill "${sampler_pid}" 2>/dev/null || true' EXIT

start_time=$(date +%s)
run_rc=0
bash "${ARM_SCRIPT}" trainer.total_training_steps=2 "$@" 2>&1 | tee "${RUN_LOG}" || run_rc=${PIPESTATUS[0]}
echo "[smoke] arm exited rc=${run_rc} after $(( $(date +%s) - start_time ))s" >&2

kill "${sampler_pid}" 2>/dev/null || true
wait "${sampler_pid}" 2>/dev/null || true

set +x
echo "==================== peak GPU memory (nvidia-smi, 2 s samples) ===================="
python - "${GPU_CSV}" <<'PY'
import collections
import sys

peak, total = collections.defaultdict(int), {}
n = 0
for line in open(sys.argv[1]):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        continue
    _, idx, used, tot = parts
    try:
        idx, used, tot = int(idx), int(used), int(tot)
    except ValueError:
        continue
    n += 1
    peak[idx] = max(peak[idx], used)
    total[idx] = tot
if not peak:
    print("WARN: no samples collected - was nvidia-smi available?")
    sys.exit(0)
print(f"{n} samples")
print("gpu  peak_used_MiB  total_MiB  headroom_MiB")
worst = None
for idx in sorted(peak):
    head = total[idx] - peak[idx]
    worst = head if worst is None else min(worst, head)
    print(f"{idx:>3}  {peak[idx]:>13}  {total[idx]:>9}  {head:>12}")
print(f"minimum headroom across GPUs: {worst} MiB")
if worst < 2048:
    print("WARN: less than 2 GiB of headroom on at least one GPU - the real run is at risk of OOM")
PY

echo "==================== OOM / memory-abort signatures in the driver log ===================="
if grep -n -E "OutOfMemoryError|CUDA out of memory|out of memory|less than desired GPU memory utilization|NCCL WARN Cuda failure|cudaErrorMemoryAllocation|calloc failed" "${RUN_LOG}" | head -20; then
    echo "FAIL: memory failure signature found in ${RUN_LOG}"
    exit 1
fi
echo "none found"

echo "==================== final checkpoint ===================="
CKPTS_DIR="logs/${exp_name//\//_}"
HF_DIR="${CKPTS_DIR}/global_step_2/actor/huggingface"
if [[ -d "${HF_DIR}" ]] && ls "${HF_DIR}"/*.safetensors >/dev/null 2>&1 && [[ -f "${HF_DIR}/config.json" ]]; then
    du -sh "${HF_DIR}"
    ls "${HF_DIR}"
    echo "hf_model checkpoint present: ${HF_DIR}"
else
    echo "FAIL: no complete hf_model checkpoint under ${HF_DIR} (the last-step save did not happen or died)"
    run_rc=${run_rc:-1}; [[ "${run_rc}" == 0 ]] && run_rc=1
fi

if [[ "${run_rc}" != 0 ]]; then
    echo "FAIL: arm exited with rc=${run_rc}; see ${RUN_LOG}"
    exit "${run_rc}"
fi
echo "PASS: 2 rollout steps (2 updates each) at 8192-token responses, final validation and hf_model save completed without a memory failure"
