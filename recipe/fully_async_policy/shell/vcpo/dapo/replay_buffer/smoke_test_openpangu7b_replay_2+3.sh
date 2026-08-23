#!/usr/bin/env bash
# =============================================================================
# smoke_test_openpangu7b_replay_2+3.sh
#
# Fast end-to-end exercise of the openPangu-Embedded-7B replay arm
#   grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh
# on a 2+3 layout. It runs the REAL arm script - there is no second copy of the
# configuration to drift - and overrides ONLY the knobs below. Everything else
# (n=16 responses per prompt, 8192-token responses, entropy_coeff=0, lr=1e-6,
# save_freq=20, the replay/min-ESS settings, the sampling) is the arm's own.
#
#   * 2 + 3 LAYOUT: 2 vLLM engines + 3 FSDP2 trainer GPUs, 5 GPUs total.
#   * mini_batch_size = 3 (the arm's own is 33). With the arm's n=16 that is 48
#     sequences per step, and 48 % 3 trainer GPUs == 0, so the batch still divides
#     across DP. The replay arm hardcodes require_batches=1, so required_samples = 3.
#   * 2 TRAINER STEPS: the rollouter derives
#         total_train_steps = total_rollout_steps / (required_samples * trigger_parameter_sync_step)
#     (fully_async_rollouter.py:229-232), which with required_samples=3 and
#     trigger_parameter_sync_step=1 makes total_rollout_steps=6 give exactly 2 steps.
#   * VALIDATION AFTER EVERY STEP: test_freq=1 (the arm's own is 20).
#   * NO VALIDATION BEFORE TRAINING: val_before_train=False (the arm's own is True),
#     so both validations that run are of a trained model.
#
# One further thing outside that list is set: exp_name, so the run writes to
# logs/SMOKE-openpangu7b-replay-2+3 instead of colliding with the real arm's log and
# checkpoint directory. It is a label, not a training parameter.
#
#   * VALIDATION ON AIME-2024 ONLY, deduplicated. The arm validates on the FULL
#     aime-2024 AND aime-2025 parquets - ~960 rows each after their 32x duplication,
#     ~1920 problems per sweep at 8192 response tokens - and test_freq=1 would sweep
#     both TWICE. On the openPangu probe 30 problems took ~2 minutes on one GPU, so
#     that would put this test on the order of an hour per sweep and make it dominated
#     by validation rather than by the 2 training steps it exists to exercise.
#     aime-2024_smoke.parquet is the same 30 AIME-2024 problems with the duplication
#     removed (all ground truths agreeing within each duplicate group, checked before
#     it was written): 32x cheaper, and val-core/math_dapo/acc becomes mean@1 over 30
#     problems - fine for a plumbing check, far too noisy to compare arms with.
#
# ---------------------------------------------------------------------------
# NO CHECKPOINTS ARE WRITTEN. save_freq stays at the arm's 20, and this run reaches
# step 2, so nothing is saved and there is nothing to verify. Pass save_freq=1 if you
# want the checkpoint path exercised too.
# ---------------------------------------------------------------------------
#
# Afterwards it reports what the run produced, without altering it: the validation
# accuracy per param version read from the run's own TensorBoard events, and a sanity
# pass over the rollout dumps (score distribution, [INVALID] rate, response shapes).
# The openPangu arm scores with stock math_dapo, and the measured failure mode there
# is TRUNCATION, not extraction: 19/30 AIME-2024 responses hit the 8192-token cap
# still inside the [unused16] think block and scored -1 regardless of ability
# (measured 2026-08-23). The dump report surfaces that rate so a smoke run says
# whether the training rollouts look the same way.
#
# Usage:  bash smoke_test_openpangu7b_replay_2+3.sh
# Env:    ARM_SCRIPT, TEST_FILE, MODEL_PATH, TRAIN_FILE, save_freq, ...
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

# Ray workers deserialize the trust_remote_code tokenizer BY REFERENCE, as
# transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer. That dynamic package
# only lands on sys.path in a process that has itself loaded remote code: the driver has,
# the actors have not, so FullyAsyncRollouter.__init__ dies unpickling its own constructor
# arguments with "ModuleNotFoundError: No module named 'transformers_modules'". Ray workers
# inherit this environment, which makes the reference resolvable everywhere. (The arm sets
# the same thing; kept here so the wrapper works even if invoked with a different ARM_SCRIPT.)
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-openpangu7b-replay-2+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
export exp_name

# AIME-2024 only, as a single-element list: the arm validates on 2024+2025 by default.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024_smoke.parquet']"}
export TEST_FILE

# ---- the overrides, and nothing else --------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-5}
export n_gpus_rollout=${n_gpus_rollout:-2}          # -> n_gpus_training = 5 - 2 = 3
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export total_rollout_steps=${total_rollout_steps:-6}   # 6 / (3 * 1) = 2 trainer steps
export test_freq=${test_freq:-1}                    # validate after every param version
export val_before_train=${val_before_train:-False}  # and not before training

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
# stderr, so that `bash smoke_test_openpangu7b_replay_2+3.sh --cfg job --resolve` yields clean YAML
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

# `--cfg job` and friends make the arm print its config and exit: nothing to report then.
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exit 0 ;; esac
done

RUN_DIR="logs/${exp_name//\//_}"
set +x

echo "==================== validation accuracy ===================="
python - "${RUN_DIR}" <<'PY'
"""Read val-core/*/acc/mean@1 out of the run's own TensorBoard events.

Both points are of a trained model (val_before_train=False), two steps apart, so they say
the validation path works end to end - they are far too few to say anything about learning.
"""

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = sys.argv[1]
events = sorted(glob.glob(os.path.join(run_dir, "tensorboard", "events.out.tfevents.*")),
                key=os.path.getmtime)
if not events:
    print(f"WARN: no TensorBoard events under {run_dir}/tensorboard - cannot report accuracy")
    sys.exit(0)

acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = [t for t in acc.Tags().get("scalars", []) if t.startswith("val-core/") and t.endswith("acc/mean@1")]
if not tags:
    print("WARN: the run logged no val-core/*/acc/mean@1 - did validation run at all?")
    sys.exit(0)

for tag in sorted(tags):
    print(f"\n{tag}")
    for e in acc.Scalars(tag):
        print(f"  param_version {e.step:>3}: {e.value:.4f}")
PY

echo "==================== rollout sanity ===================="
python - "${RUN_DIR}" <<'PY'
"""Score distribution, extraction rate and response shapes over the training rollouts.

The openPangu arm scores with stock math_dapo and answers in slow-think mode: the model
reasons inside [unused16] ... [unused17] and then writes the answer on the last line, which
is what math_dapo's last-300-characters window reads. The failure mode measured on
AIME-2024 (2026-08-23) was TRUNCATION rather than extraction - 19/30 responses reached the
8192-token cap still inside the think block, emitted no answer and scored -1 whatever the
model knew - so the truncation rate is the number to look at here.
"""

import collections
import glob
import json
import os
import re
import sys

run_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")),
               key=lambda f: int(re.search(r"(\d+)\.jsonl$", f).group(1)))
if not files:
    sys.exit(f"FAIL: no rollout dumps in {run_dir} (is trainer.rollout_data_dir set?)")

rows = [json.loads(line) for f in files for line in open(f)]
scores = collections.Counter(r.get("score") for r in rows)
preds = [r.get("pred") for r in rows if "pred" in r]
invalid = sum(p == "[INVALID]" for p in preds)
outs = [r.get("output") or "" for r in rows]
closed = sum("[unused17]" in o for o in outs)
answer = sum(bool(re.search(r"(?i)answer\s*:", o)) for o in outs)

print(f"dumps          : {len(files)} ({', '.join(os.path.basename(f) for f in files)})")
print(f"samples        : {len(rows)}")
print(f"scores         : {dict(scores)}")
if preds:
    print(f"[INVALID]      : {invalid}/{len(preds)}")
print(f"left the think block ([unused17]) : {closed}/{len(rows)}")
print(f"emitted an 'Answer:' line         : {answer}/{len(rows)}")
if closed < len(rows):
    print(f"NOTE: {len(rows) - closed}/{len(rows)} responses never closed [unused16]; at 8192 tokens")
    print("      those are truncated mid-reasoning and score -1 regardless of ability.")

failures = []
if rows and "pred" not in rows[0]:
    failures.append("no 'pred' field in the dumps: the reward function did not run")
if preds and invalid == len(preds):
    failures.append("every prediction is [INVALID]: extraction is broken")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("OK: rollouts were scored")
PY
