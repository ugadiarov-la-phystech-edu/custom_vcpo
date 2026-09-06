#!/usr/bin/env bash
# =============================================================================
# smoke_test_openpangu7b_replay_3+3.sh
#
# Fast end-to-end exercise of the openPangu-Embedded-7B replay / min-ESS arm
#   grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_openpangu7b.sh
# on a 3+3 layout (3 vLLM engines + 3 Megatron trainer GPUs, 6 GPUs total). It runs the real
# arm script - there is no second copy of the configuration to drift - and only overrides what
# makes it short, then verifies checkpoints, validation, answer extraction, the replay path and
# the o_proj-bias sync. This is the FIRST end-to-end run of the Megatron openPangu path (ported
# from baselines_main-ppo_openpangu) through the replay trainer: run it before the arm.
#
# WHY THE RUN LENGTH IS CONTROLLED THROUGH THE REPLAY BUFFER, NOT THE PROMPT BUDGET.
# In replay mode the trainer does NOT stop when rollout.total_rollout_steps prompts are spent:
# FullyAsyncTrainer._acquire_replay_minibatch keeps composing pure-replay mini-batches for as
# long as the buffer holds >= requires_mini_batches x mini_bsz groups, and only stops once the
# rollouter is done AND eviction (staleness > replay_buffer.staleness_threshold) has drained the
# buffer below that watermark. With the arm's k=32 a 3-prompt smoke would run ~33 updates. So:
#
#   * ONE fresh mini-batch: total_rollout_steps = 3 = mini_bsz 3 x require_batches 1, n=2
#     -> 6 sequences per update, divisible by trainer DP=3 (tp=pp=1).
#   * replay_staleness_threshold=1 (the ONLY replay knob changed; tau stays 8, reuse_halflife
#     stays 1 - inert here, no group is ever drawn with times_trained > 1):
#       update 1 trains the 3 fresh version-0 groups         -> version 1, staleness 1, kept
#       update 2 is a PURE-REPLAY mini-batch of those groups  -> version 2, staleness 2 > 1, evicted
#       buffer 0 < watermark 3 and the rollouter is finished  -> the fit loop exits.
#     Exactly 2 updates, deterministically; update 2 exercises the cached behavior log-probs,
#     the token-IS correction and the min-ESS brake on genuinely stale data (after the lr=1e-4
#     update the IS ratios are extreme, so expect the brake to fire).
#   * test_freq=save_freq=1 (param-version units; one version per update here) -> 2 validations
#     and 2 checkpoints, no end-of-fit extras.
#   * max_response_length stays at the arm's 8192 so generation runs at the real length.
#   * a real gradient regardless of rewards: with 3 prompts x n=2 a group easily ties (GRPO
#     advantage identically 0), so entropy_coeff=0.01 and lr=1e-4 (100x the arm) make 2 updates
#     visible in bf16. Plumbing values, not the arm's.
#
# VALIDATION SET: aime-2024_smoke.parquet (the DAPO-format aime-2024 with its 32x duplication
# removed, 30 rows; exists on the cluster next to the arm's data). At 8192 tokens the full 960-row
# files would dominate the runtime. Both points are of a model driven hard by lr=1e-4; run with
# val_before_train=True for a sweep of the UNTRAINED checkpoint, comparable with the openPangu
# sync arms' step-0 point (0.222 on aime-2024 at T=0.8/top_p=0.7 with BOS).
#
# The model is NOT shrunk: the point is the real openPangu path end to end - the re-aliased
# Llama checkpoint with attention_bias=true, the trust_remote_code PanguTokenizer resolved in
# every Ray worker via HF_MODULES_CACHE on PYTHONPATH, the BOS prepend, add_bias_linear with
# frozen MLP biases, o_proj.bias loaded / synced to vLLM / exported, the replay buffer, the
# min-ESS brake and the hf_model save. Megatron writes bf16 weights, ~16 GB per checkpoint
# (8.0B params incl. the 34 o_proj.bias tensors): budget ~32 GB.
#
# Afterwards it runs five checks (all skipped when the arm is only asked for --cfg/--help):
#   1. verify_checkpoints.py --dtype BF16 --base-model: weights + tokenizer + config present and
#      loadable, no sharded leftovers, parameter NAMES/shapes/dtype matching the re-aliased
#      checkpoint - a saver that forgot o_proj.bias fails with 34 "missing" tensors - weights
#      changed between the two checkpoints (which the frozen MLP biases must not prevent),
#      timing_state.json complete, tracker up to date.
#   2. a VALIDATION ACCURACY report from the run's TensorBoard events.
#   3. THE EXTRACTION CHECK on the rollout dumps: the math_dapo scorer must recover an
#      "Answer:" from the Pangu output ([unused16]/[unused17] thinking delimiters survive the
#      decode and math_dapo is format-agnostic); the FAILURE condition is the [INVALID] rate,
#      not the sign of the rewards.
#   4. THE REPLAY CHECK: update 1 all-fresh, update 2 all-replayed, staleness/ess_ratio logged.
#   5. THE BIAS-SYNC CHECK: rollout_corr/kl at update 1 must be small (the Qwen and ORZ arms sit
#      at 0.0003-0.0008; the openPangu sync arm at 0.0008). A wrong or missing o_proj bias in
#      the vLLM sync shows up as a trainer-vs-vLLM KL far above that from the first update.
#
# Usage:  bash smoke_test_openpangu7b_replay_3+3.sh
# Env:    MODEL_PATH, TRAIN_FILE, TEST_FILE, val_before_train, n_resp_per_prompt, ...
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_openpangu7b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

# The trust_remote_code PanguTokenizer is unpickled by reference in every Ray worker; the HF
# modules cache must be importable there (the arm exports it too; set it here so the verifier
# below can load the tokenizer from the checkpoints as well).
HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/modules}
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# A LOCAL, RE-ALIASED copy (scripts/realias_openpangu_to_llama.py), the arm's default.
MODEL_PATH=${MODEL_PATH:-"/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# The 30-row DAPO-format aime-2024 (32x duplication removed); the arm validates on 2024+2025.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024_smoke.parquet']"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-openpangu7b-replay-3+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}
CKPTS_DIR="logs/${exp_name//\//_}"
mkdir -p -- "${CKPTS_DIR}"

# ---- config-only invocations (--cfg job, --help): compose and exit, verify nothing ---------
config_only=0
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) config_only=1 ;; esac
done

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
# --dtype BF16: megatron's hf_model save writes bf16; --base-model diffs parameter names against the
# re-aliased checkpoint, which is what catches a missing o_proj.bias export.
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 2 --dtype BF16 --base-model "${MODEL_PATH}"

echo "==================== validation accuracy ===================="
python - "${CKPTS_DIR}" <<'PY'
"""Print val-core/*/acc/mean@1 from the run's own TensorBoard events.

By default there are two points, one after each update, and both are of a model that lr=1e-4 has
deliberately driven hard - they say the validation path works, not that the model is good. Under
val_before_train=True a step-0 point appears as well; that one IS comparable, because the offline
openPangu sync arms scored 0.222 on aime-2024 at the same sampling before training, and a value near
0 there is a wiring failure rather than a bad model. It is annotated when present.
"""

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROBE_REFERENCE = 0.222  # openPangu-7B sync arm step-0 aime-2024 (full 960-row set, T=0.8/top_p=0.7, BOS)

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
        print("  WARN: the untrained model scored ~0 here but ~0.22 in the sync arms -")
        print("        suspect the BOS/tokenizer wiring or the bias sync, not the model.")
PY

echo "==================== reward extraction check ===================="
python - "${CKPTS_DIR}" <<'PY'
"""Did the stock math_dapo scorer extract answers from openPangu's outputs?

math_dapo takes solution_str[-300:] and the LAST "Answer:" match; the Pangu thinking delimiters
are ordinary tokens and survive the decode. The FAILURE condition is the [INVALID] rate: at 6
rollouts over 3 AIME-difficulty problems an all -1 batch is ordinary, a batch of [INVALID]s is a
prompt/template/decode problem. Exit non-zero on the latter.
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
with_answer_line = sum(bool(re.search(r"(?i)answer\s*:", (r.get("output") or "")[-300:])) for r in rows)
print(f"dumps      : {len(files)} files ({', '.join(os.path.basename(f) for f in files)})")
print(f"samples    : {len(rows)}")
print(f"scores     : {dict(scores)}")
print(f"'Answer:' in the last 300 chars : {with_answer_line}/{len(rows)}")
if preds:
    print(f"[INVALID]  : {invalid}/{len(preds)}")
    print(f"sample preds     : {preds[:8]}")

failures = []
if rows and "pred" not in rows[0]:
    failures.append("no 'pred' field in the dumps: the reward function did not run")
if preds and invalid == len(preds):
    failures.append("every prediction is [INVALID]: extraction is broken (template / decode / 300-char window)")
if preds and invalid > 0.5 * len(preds):
    failures.append(f"{invalid}/{len(preds)} predictions are [INVALID]: extraction is mostly failing")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
if len(scores) == 1 and next(iter(scores)) is not None and next(iter(scores)) < 0:
    print("WARN: every sample scored -1. At this sample size that is ordinary; extraction worked.")
print("OK: math_dapo extracted answers from openPangu's outputs")
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

echo "==================== bias-sync check (rollout_corr/kl at update 1) ===================="
python - "${CKPTS_DIR}" <<'PY'
"""A wrong or missing o_proj bias in the Megatron -> vLLM sync shows up as a large trainer-vs-vLLM
token KL from the very first update (the Qwen/ORZ arms and the openPangu sync arm sit at
0.0003-0.0008). Hard-fail above 0.02; the replayed update 2 is expected to be large (lr=1e-4
between the two passes) and is only printed.
"""

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ckpt_dir = sys.argv[1]
events = sorted(glob.glob(os.path.join(ckpt_dir, "tensorboard", "events.out.tfevents.*")), key=os.path.getmtime)
if not events:
    sys.exit(f"FAIL: no TensorBoard events under {ckpt_dir}/tensorboard")
acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = set(acc.Tags().get("scalars", []))
if "rollout_corr/kl" not in tags:
    sys.exit("FAIL: rollout_corr/kl not logged")
kl = {e.step: e.value for e in acc.Scalars("rollout_corr/kl")}
for step in sorted(kl):
    print(f"  version {step}: rollout_corr/kl = {kl[step]:.5f}")
first = kl[min(kl)]
if first > 0.02:
    print(f"FAIL: rollout_corr/kl at update 1 is {first:.4f} (expected ~1e-3): the vLLM weights do not match the trainer -")
    print("      suspect the o_proj.bias sync (weight_converter.py) or the frozen MLP biases (model_initializer.py)")
    sys.exit(1)
print("OK: trainer-vs-vLLM KL at update 1 is at the healthy level; the bias sync is consistent")
PY
