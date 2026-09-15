#!/usr/bin/env bash

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg_orz7b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-"Open-Reasoner-Zero/Open-Reasoner-Zero-7B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024_smoke.parquet']"}

exp_name=${exp_name:-"SMOKE-orz7b-2+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-5}
export n_gpus_rollout=${n_gpus_rollout:-2}

export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export SEED=${SEED:-1}
export n_resp_per_prompt=${n_resp_per_prompt:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}
export total_rollout_steps=${total_rollout_steps:-6}
export test_freq=${test_freq:-1}
export save_freq=${save_freq:-1}
export val_before_train=${val_before_train:-False}
export ppo_epochs=${ppo_epochs:-2}
export entropy_coeff=${entropy_coeff:-0.01}
export lr=${lr:-1e-4}

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exit 0 ;; esac
done

CKPTS_DIR="logs/${exp_name//\//_}"
set +x
echo "==================== checkpoint verification ===================="
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 2 --base-model "${MODEL_PATH}"

echo "==================== validation accuracy ===================="
python - "${CKPTS_DIR}" <<'PY'
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROBE_REFERENCE = 5 / 30

ckpt_dir = sys.argv[1]
events = sorted(glob.glob(os.path.join(ckpt_dir, "tensorboard", "events.out.tfevents.*")),
                key=os.path.getmtime)
if not events:
    print(f"WARN: no TensorBoard events under {ckpt_dir}/tensorboard - cannot report accuracy")
    sys.exit(0)

acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = [t for t in acc.Tags().get("scalars", []) if t.startswith("val-core/") and t.endswith("acc/mean@1")]
if not tags:
    print("WARN: the run logged no val-core/*/acc/mean@1 - did validation run at all?")
    sys.exit(0)

for tag in sorted(tags):
    points = [(e.step, e.value) for e in acc.Scalars(tag)]
    print(f"\n{tag}")
    for step, value in points:
        note = ""
        if step == 0:
            delta = value - PROBE_REFERENCE
            note = f"   <- untrained reference (offline probe: {PROBE_REFERENCE:.3f}, delta {delta:+.3f})"
        print(f"  param_version {step:>3}: {value:.4f}{note}")
    if points and points[0][0] == 0 and points[0][1] < 0.05:
        print("  WARN: the untrained model scored ~0 here but 0.167 in the offline probe -")
        print("        suspect the chat template, the prompts or the scorer, not the model.")
PY

echo "==================== reward extraction check ===================="
python - "${CKPTS_DIR}" <<'PY'
import collections
import glob
import json
import os
import re
import sys

ckpt_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(ckpt_dir, "*.jsonl")),
               key=lambda f: int(re.search(r"(\d+)\.jsonl$", f).group(1)))
if not files:
    sys.exit(f"FAIL: no rollout dumps in {ckpt_dir} (is trainer.rollout_data_dir set?)")

rows = [json.loads(line) for f in files for line in open(f)]
scores = collections.Counter(r.get("score") for r in rows)
preds = [r.get("pred") for r in rows if "pred" in r]
invalid = sum(p == "[INVALID]" for p in preds)
tagged = sum("<answer>" in (r.get("output") or "") for r in rows)

print(f"dumps      : {len(files)} files ({', '.join(os.path.basename(f) for f in files)})")
print(f"samples    : {len(rows)}")
print(f"scores     : {dict(scores)}")
print(f"emitted <answer> : {tagged}/{len(rows)}")
if preds:
    print(f"[INVALID]  : {invalid}/{len(preds)}")
    print(f"sample preds     : {preds[:8]}")

failures = []
if rows and "pred" not in rows[0]:
    failures.append("no 'pred' field in the dumps: the custom reward function did not run")
if preds and invalid == len(preds):
    failures.append("every prediction is [INVALID]: extraction is broken")
if preds and invalid > 0.5 * len(preds):
    failures.append(f"{invalid}/{len(preds)} predictions are [INVALID]: extraction is mostly failing")
if tagged == 0:
    failures.append("no response contained <answer>: ORZ's chat template did not apply")

if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)

if len(scores) == 1 and next(iter(scores)) is not None and next(iter(scores)) < 0:
    print("WARN: every sample scored -1. At this sample size that is ordinary - these are")
    print("      AIME-difficulty problems and a 12-rollout smoke tells you nothing about")
    print("      accuracy. Extraction is what this check verifies, and it worked.")
print("OK: answers were extracted from ORZ's <answer> blocks")
PY
