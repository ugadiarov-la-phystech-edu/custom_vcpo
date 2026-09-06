#!/usr/bin/env bash

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_8gpu_orz72k_3+5_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_orz7b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

MODEL_PATH=${MODEL_PATH:-"Open-Reasoner-Zero/Open-Reasoner-Zero-7B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/orz/orz-math-72k.parquet"}
VAL_SOURCE_FILE=${VAL_SOURCE_FILE:-"/home/jovyan/datasets/math_datasets/orz/aime-2024-orz.parquet"}

exp_name=${exp_name:-"SMOKE-orz7b-replay-3+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
CKPTS_DIR="logs/${exp_name//\//_}"
mkdir -p -- "${CKPTS_DIR}"

config_only=0
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) config_only=1 ;; esac
done

SMOKE_VAL_FILE="${REPO_ROOT}/${CKPTS_DIR}/aime-2024-orz_smoke.parquet"
if [[ -z "${TEST_FILE:-}" && "${config_only}" -eq 0 ]]; then
    python - "${VAL_SOURCE_FILE}" "${SMOKE_VAL_FILE}" <<'PY'
import json
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
key = df["prompt"].map(lambda p: json.dumps([dict(m) for m in p], sort_keys=True))
gt = df["reward_model"].map(lambda r: str(dict(r).get("ground_truth")))
conflicts = df.assign(_key=key, _gt=gt).groupby("_key")["_gt"].nunique()
assert (conflicts == 1).all(), f"{int((conflicts != 1).sum())} prompts carry conflicting ground truths"
dedup = df.loc[~key.duplicated()].reset_index(drop=True)
assert len(dedup) == 30, f"expected 30 distinct problems, got {len(dedup)}"
dedup.to_parquet(dst, index=False)
print(f"[smoke] wrote {len(dedup)} rows ({sorted(dedup['data_source'].unique())}) to {dst}")
PY
fi
TEST_FILE=${TEST_FILE:-"['${SMOKE_VAL_FILE}']"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
export n_gpus_rollout=${n_gpus_rollout:-3}

export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}
export total_rollout_steps=${total_rollout_steps:-3}
export replay_staleness_threshold=${replay_staleness_threshold:-1}
export replay_requires_mini_batches=${replay_requires_mini_batches:-1}
export test_freq=${test_freq:-1}
export save_freq=${save_freq:-1}
export val_before_train=${val_before_train:-False}
export entropy_coeff=${entropy_coeff:-0.01}
export lr=${lr:-1e-4}

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

[[ "${config_only}" -eq 0 ]] || exit 0

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
events = sorted(glob.glob(os.path.join(ckpt_dir, "tensorboard", "events.out.tfevents.*")), key=os.path.getmtime)
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
            note = f"   <- untrained reference (offline probe: {PROBE_REFERENCE:.3f}, delta {value - PROBE_REFERENCE:+.3f})"
        print(f"  param_version {step:>3}: {value:.4f}{note}")
    if points and points[0][0] == 0 and points[0][1] < 0.05:
        print("  WARN: the untrained model scored ~0 here but 0.167 in the offline probe -")
        print("        suspect the chat template, the ORZ prompts or the scorer, not the model.")
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
files = sorted(glob.glob(os.path.join(ckpt_dir, "*.jsonl")), key=lambda f: int(re.search(r"(\d+)\.jsonl$", f).group(1)))
if not files:
    sys.exit(f"FAIL: no rollout dumps in {ckpt_dir} (is trainer.rollout_data_dir set?)")

rows = [json.loads(line) for f in files for line in open(f)]
scores = collections.Counter(r.get("score") for r in rows)
preds = [r.get("pred") for r in rows if "pred" in r]
invalid = sum(p == "[INVALID]" for p in preds)
tagged = sum("<answer>" in (r.get("output") or "") for r in rows)
gts = [str(r.get("gts", r.get("ground_truth", ""))) for r in rows]
non_integer_gts = sum(1 for g in gts if g and not re.fullmatch(r"-?\d+", g.strip()))

print(f"dumps      : {len(files)} files ({', '.join(os.path.basename(f) for f in files)})")
print(f"samples    : {len(rows)}")
print(f"scores     : {dict(scores)}")
print(f"emitted <answer> : {tagged}/{len(rows)}")
if preds:
    print(f"[INVALID]  : {invalid}/{len(preds)}")
    print(f"sample preds     : {preds[:8]}")
print(f"non-integer ground truths: {non_integer_gts}/{len(gts)} (the LaTeX equality tiers apply to these)")

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
    print("      AIME-difficulty problems and a 6-rollout smoke tells you nothing about")
    print("      accuracy. Extraction is what this check verifies, and it worked.")
print("OK: answers were extracted from ORZ's <answer> blocks")
PY

echo "==================== replay path check ===================="
python - "${CKPTS_DIR}" "${train_prompt_mini_bsz}" <<'PY'
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ckpt_dir, mini_bsz = sys.argv[1], int(sys.argv[2])
events = sorted(glob.glob(os.path.join(ckpt_dir, "tensorboard", "events.out.tfevents.*")), key=os.path.getmtime)
if not events:
    sys.exit(f"FAIL: no TensorBoard events under {ckpt_dir}/tensorboard")
acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = set(acc.Tags().get("scalars", []))


def series(tag):
    return {e.step: e.value for e in acc.Scalars(tag)} if tag in tags else {}


new, replayed, ess = series("replay/minibatch_new"), series("replay/minibatch_replayed"), series("staleness/ess_ratio")
scaled_lr = series("replay/ess_scaled_lr")
for step in sorted(set(new) | set(replayed)):
    print(
        f"  update -> version {step}: fresh={new.get(step)} replayed={replayed.get(step)} "
        f"ess_ratio={ess.get(step)} ess_scaled_lr={scaled_lr.get(step)}"
    )

failures = []
if not new or not replayed:
    failures.append("replay/minibatch_new|replayed not logged: the trainer did not run the replay fit loop")
else:
    steps = sorted(new)
    if len(steps) != 2:
        failures.append(f"expected exactly 2 updates, the events show {len(steps)}: {steps}")
    else:
        first, last = steps
        if new[first] != mini_bsz or replayed.get(first, 0) != 0:
            failures.append(f"update 1 should be all-fresh ({mini_bsz}), got fresh={new[first]} replayed={replayed.get(first)}")
        if replayed.get(last) != mini_bsz or new[last] != 0:
            failures.append(f"update 2 should be pure replay ({mini_bsz}), got fresh={new[last]} replayed={replayed.get(last)}")
if not ess:
    failures.append("staleness/ess_ratio not logged: the min-ESS brake was never consulted")
elif len(ess) < 2:
    failures.append(f"staleness/ess_ratio logged for {len(ess)} update(s), expected 2")

if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("OK: update 1 was fresh, update 2 was pure replay, and the ESS brake saw both")
PY
