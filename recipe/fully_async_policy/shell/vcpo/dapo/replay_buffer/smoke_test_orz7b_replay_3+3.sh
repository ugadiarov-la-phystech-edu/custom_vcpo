#!/usr/bin/env bash
# =============================================================================
# smoke_test_orz7b_replay_3+3.sh
#
# Fast end-to-end exercise of the Open-Reasoner-Zero-7B replay / min-ESS arm
#   grpo_novcpo_8gpu_orz72k_3+5_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_orz7b.sh
# on a 3+3 layout (3 vLLM engines + 3 Megatron trainer GPUs, 6 GPUs total). It runs the real
# arm script - there is no second copy of the configuration to drift - and only overrides what
# makes it short, then verifies checkpoints, validation, the reward path and the replay path.
#
# WHY THE RUN LENGTH IS CONTROLLED THROUGH THE REPLAY BUFFER, NOT THE PROMPT BUDGET.
# In replay mode the trainer does NOT stop when rollout.total_rollout_steps prompts are spent:
# FullyAsyncTrainer._acquire_replay_minibatch keeps composing pure-replay mini-batches for as
# long as the buffer holds >= requires_mini_batches x mini_bsz groups, and only stops once the
# rollouter is done AND eviction (staleness > replay_buffer.staleness_threshold, staleness =
# current_version - group_version) has drained the buffer below that watermark. With the arm's
# k=32 a 3-prompt smoke would therefore run ~33 updates. So:
#
#   * ONE fresh mini-batch: total_rollout_steps = 3 = mini_bsz 3 x require_batches 1, n=2
#     -> 6 sequences per update, divisible by trainer DP=3 (tp=pp=1).
#   * replay_staleness_threshold=1 (the ONLY replay knob changed; tau stays 8, reuse_halflife stays 1
#     — inert here, no group is ever drawn with times_trained > 1):
#       update 1 trains the 3 fresh version-0 groups         -> version 1, staleness 1, kept
#       update 2 is a PURE-REPLAY mini-batch of those groups  -> version 2, staleness 2 > 1, evicted
#       buffer 0 < watermark 3 and the rollouter is finished  -> the fit loop exits.
#     Exactly 2 updates, deterministically, and update 2 exercises what this arm is about: the
#     cached behavior log-probs, the token-IS correction and the min-ESS brake on genuinely
#     stale data (after the lr=1e-4 update the IS ratios are extreme, so expect the brake to
#     fire: replay/ess_scaled_lr == lr * ess_lr_scale at update 2 is a pass, not a failure).
#   * test_freq=save_freq=1 (param-version units; one version per update here) -> 2 validations
#     and 2 checkpoints. No end-of-fit extras: the tail block only forces a final sync when
#     `version % test_freq != 0 or local_trigger_step > 1`, and the final save is skipped
#     because the last update already saved that version.
#   * max_response_length stays at the arm's 8192: ORZ's answers reach ~6.9k tokens and a
#     shortened cap would truncate the <answer> block and make the reward check meaningless.
#   * a real gradient regardless of rewards: with 3 prompts x n=2 a group easily ties (both
#     right or both wrong), and a tied group has GRPO advantage identically 0 - the weights
#     could not move and "the checkpoints differ" would be unverifiable. entropy_coeff=0.01
#     gives a reward-independent gradient and lr=1e-4 (100x the arm) makes 2 updates visible in
#     bf16. Plumbing values, not the arm's.
#
# VALIDATION SET. The arm validates on aime-2024-orz + aime-2025-orz (30 problems x 32 copies
# each, ORZ's own prompt instruction). At 8192 tokens on 3 engines those 1920 rows would
# dominate the runtime, and no 30-row ORZ-format file exists on the cluster, so this wrapper
# BUILDS one at launch: aime-2024-orz.parquet deduplicated on the prompt content (30 rows, the
# ground truths agreeing within each duplicate group - asserted) written to
# logs/<exp_name>/aime-2024-orz_smoke.parquet. TEST_FILE overrides it. The metric is then
# val-core/aime2024_orz/acc/mean@1 over 30 problems: a plumbing check, far too noisy to compare
# arms with. Both points are of a model driven hard by lr=1e-4; for a number that IS comparable
# run with val_before_train=True, which adds a sweep of the UNTRAINED checkpoint that should land
# near the offline probe's 5/30 = 0.167 (same 30 problems, same T=1.0/top_p=1.0 sampling); a
# value near 0 there means the chat template, the ORZ prompts or the scorer is not wired.
#
# The model is NOT shrunk: the point is the real ORZ path end to end - its own chat template,
# Qwen2ForCausalLM through the mcore registry, the tiered tag-aware reward function on
# ORZ-72k-style prompts, the replay buffer, the min-ESS brake and the hf_model save. Megatron
# writes bf16 weights, ~15.2 GB per checkpoint: budget ~31 GB.
#
# Afterwards it runs four checks (all skipped when the arm is only asked for --cfg/--help):
#   1. verify_checkpoints.py: weights + tokenizer + config present and loadable, no sharded
#      leftovers, parameter names/shapes/dtype matching the base checkpoint, weights actually
#      changed between the two checkpoints, timing_state.json complete, tracker up to date.
#   2. a VALIDATION ACCURACY report from the run's TensorBoard events (val-core/*/acc/mean@1
#      per param version, step-0 reference annotated when present).
#   3. THE REWARD CHECK: the rollout dumps are scanned for the [INVALID] rate and the presence
#      of <answer>. The FAILURE condition is extraction, not the sign of the rewards - 6 rollouts
#      over AIME-difficulty problems scoring all -1 is ordinary. It also reports how many ground
#      truths were non-integer, i.e. whether the LaTeX equality tiers were even reachable.
#   4. THE REPLAY CHECK: from the events, update 1 must be all-fresh (replay/minibatch_new == 3)
#      and update 2 all-replayed (replay/minibatch_replayed == 3), and staleness/ess_ratio must
#      be logged for both - that is the path a DAPO-17k smoke of the twin never touches.
#
# Usage:  bash smoke_test_orz7b_replay_3+3.sh
# Env:    MODEL_PATH, TRAIN_FILE, TEST_FILE, VAL_SOURCE_FILE, val_before_train, n_resp_per_prompt, ...
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_8gpu_orz72k_3+5_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_orz7b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

# The hub id works as-is: ORZ is Qwen2ForCausalLM, which is in verl's mcore registry, and its
# tokenizer is a stock Qwen2TokenizerFast - no trust_remote_code.
MODEL_PATH=${MODEL_PATH:-"Open-Reasoner-Zero/Open-Reasoner-Zero-7B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/orz/orz-math-72k.parquet"}
# Source of the 30-row validation file built below (the arm's own aime-2024-orz set).
VAL_SOURCE_FILE=${VAL_SOURCE_FILE:-"/home/jovyan/datasets/math_datasets/orz/aime-2024-orz.parquet"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-orz7b-replay-3+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
CKPTS_DIR="logs/${exp_name//\//_}"
mkdir -p -- "${CKPTS_DIR}"

# ---- config-only invocations (--cfg job, --help): compose and exit, build nothing ----------
config_only=0
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) config_only=1 ;; esac
done

# ---- the 30-row ORZ-format validation set -----------------------------------------------
SMOKE_VAL_FILE="${REPO_ROOT}/${CKPTS_DIR}/aime-2024-orz_smoke.parquet"
if [[ -z "${TEST_FILE:-}" && "${config_only}" -eq 0 ]]; then
    python - "${VAL_SOURCE_FILE}" "${SMOKE_VAL_FILE}" <<'PY'
"""Deduplicate the ORZ-prompt AIME-2024 parquet (30 problems x 32 copies) to one row per problem."""

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

# ---- 3 + 3 layout on the first six GPUs ------------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
export n_gpus_rollout=${n_gpus_rollout:-3}   # -> n_gpus_training = 6 - 3 = 3

# ---- 2 updates, cheap ones (see the header for why these values) -----------------------
export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}   # the arm's own length, not a shortened one
export total_rollout_steps=${total_rollout_steps:-3}      # ONE fresh mini-batch (3 x require_batches 1)
export replay_staleness_threshold=${replay_staleness_threshold:-1}  # drains the buffer after update 2
export replay_requires_mini_batches=${replay_requires_mini_batches:-1}
export test_freq=${test_freq:-1}          # validate after every param version
export save_freq=${save_freq:-1}          # and checkpoint every one
export val_before_train=${val_before_train:-False}  # override to True for the 0.167 reference; see the header
export entropy_coeff=${entropy_coeff:-0.01}  # the only reward-independent gradient
export lr=${lr:-1e-4}                     # 100x the arm's, so 2 updates clear bf16 rounding

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
# stderr, so that `bash smoke_test_orz7b_replay_3+3.sh --cfg job --resolve` yields clean YAML
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

[[ "${config_only}" -eq 0 ]] || exit 0

set +x
echo "==================== checkpoint verification ===================="
# no --dtype: megatron's hf_model save writes bf16 (actor.megatron.dtype=bfloat16), the default.
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 2 --base-model "${MODEL_PATH}"

echo "==================== validation accuracy ===================="
python - "${CKPTS_DIR}" <<'PY'
"""Print val-core/*/acc/mean@1 from the run's own TensorBoard events.

By default there are two points, one after each update, and both are of a model that lr=1e-4 has
deliberately driven hard - they say the validation path works, not that the model is good. Under
val_before_train=True a step-0 point appears as well; that one IS comparable, because the offline
vLLM probe scored 5/30 = 0.167 on exactly these 30 problems at the same sampling, and a value near
0 there is a wiring failure rather than a bad model. It is annotated when present.
"""

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROBE_REFERENCE = 5 / 30  # ORZ-7B, the 30 AIME-2024 problems, T=1.0/top_p=1.0, tag-aware scorer

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
"""Did the tiered tag-aware scorer actually extract answers from ORZ's <answer> blocks?

A uniformly -1 batch is the silent failure this arm exists to prevent: with stock math_dapo every
ORZ rollout scores -1, the GRPO advantage is identically 0, and the run looks healthy while
learning nothing. Exit non-zero if that is what the dumps show.
"""

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

# What can actually be concluded at this sample size. The run produces 6 rollouts over 3
# AIME-difficulty problems, so "every sample scored -1" is ORDINARY and a WARNING, not a failure.
# What IS diagnostic is whether extraction worked at all: stock math_dapo returned [INVALID] on
# 12/12 ORZ responses in the 2026-08-23 probe while this scorer recovered a clean value every time.
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
"""Update 1 must be all-fresh and update 2 all-replayed, and the ESS brake must have been consulted.

The replay metrics are logged at the param version the update produced (1 and 2 here).
"""

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
