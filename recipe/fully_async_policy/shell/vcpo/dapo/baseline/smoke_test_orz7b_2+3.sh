#!/usr/bin/env bash
# =============================================================================
# smoke_test_orz7b_2+3.sh
#
# Fast end-to-end exercise of the Open-Reasoner-Zero-7B arm
#   grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg_orz7b.sh
# on a 2+3 layout (2 vLLM engines + 3 Megatron trainer GPUs, 5 GPUs total). It runs the real
# arm script - there is no second copy of the configuration to drift - and only overrides
# what makes it short:
#
#   * 2 TRAINER STEPS: total_rollout_steps = 6 = 2 x (mini_bsz 3 x require_batches 1),
#     so total_train_steps = 2 and the run stops on its own.
#   * n_resp_per_prompt=2 instead of 16 and mini_bsz 3 instead of 33: 3*2 = 6 sequences,
#     divisible by trainer DP=3 (tp=pp=1, so DP is just the 3 trainer GPUs).
#   * max_response_length stays at the arm's 8192, so generation is exercised at the real
#     length. That matters more here than for the Qwen arm: ORZ's responses on AIME-2024
#     measured 1084 / 2818 / 6892 tokens (min/median/max) with 0/30 hitting the cap, so a
#     shortened cap would truncate the <answer> block and make the reward check meaningless.
#   * VALIDATION ON AIME-2024 ONLY, AFTER EVERY STEP (test_freq=1), and NOT before training
#     (val_before_train=False), so the two validations that run are both of a trained model.
#     It uses aime-2024_smoke.parquet - the same problems with the 32x duplication removed,
#     30 rows instead of 960 - because at 8192 tokens the full file dominates the runtime.
#
#     Neither point is a quality measurement: at lr=1e-4 with entropy_coeff=0.01 - see below -
#     the weights are driven hard on purpose, so accuracy is expected to move, quite possibly
#     downwards. To get a number that IS comparable to something, run
#
#         val_before_train=True bash smoke_test_orz7b_2+3.sh
#
#     which adds one 30-problem sweep of the UNTRAINED checkpoint through the real pipeline.
#     That point should land near the 5/30 = 0.167 the same 30 problems scored in the offline
#     vLLM probe at the same sampling (T=1.0/top_p=1.0); agreement means the chat template,
#     the DAPO prompts and the tag-aware scorer are wired correctly end to end, and a value
#     near 0 means one of them is not. The report below annotates it automatically.
#   * A CHECKPOINT AFTER EVERY STEP (save_freq=1) -> exactly 2 checkpoints. Unlike the 8-GPU
#     smoke test there is no third one: the trainer's end-of-fit block only forces a final
#     sync when `param_version % test_freq != 0 or local_trigger_step > 1`
#     (fully_async_trainer.py:341), and at test_freq=1 both are false.
#   * a real gradient regardless of rewards: with 3 prompts x n=2 a group can easily tie
#     (both responses right, or both wrong), and a tied group has GRPO advantage identically
#     0 - pg_loss and grad_norm are then exactly 0, the weights cannot move and "the
#     checkpoints differ" becomes unverifiable. entropy_coeff gives a gradient that does not
#     depend on the rewards, and the large lr makes 4 updates visible in the weights. This is
#     a plumbing test, not a learning test: neither value relates to the arm's own settings.
#
# The model is NOT shrunk: the point is to exercise the real ORZ path end to end - its own
# chat template (preamble + `User: ...` + `Assistant: <think>`, no ChatML), Qwen2ForCausalLM
# through the mcore registry, the tag-aware custom reward function, and the hf_model save.
# Megatron writes bf16 weights (actor.megatron.dtype=bfloat16), so each checkpoint is
# ~15.2 GB: budget ~31 GB.
#
# Afterwards it runs two checks:
#   1. verify_checkpoints.py, which fails loudly if a checkpoint is incomplete: weights +
#      tokenizer + config present and loadable, no sharded leftovers, parameter
#      names/shapes/dtype matching the base checkpoint, weights actually changed between the
#      two checkpoints, timing_state.json complete, tracker up to date.
#   2. a VALIDATION ACCURACY report, read from the run's own TensorBoard events, printing
#      val-core/*/acc/mean@1 per param version with the step-0 reference called out.
#   3. THE REWARD CHECK, which is why this arm exists. Stock math_dapo scores ORZ 0/30 on
#      AIME-2024 (measured, both with and without the DAPO wrapper): it captures
#      `Answer:\s*([^\n]+)` to end of LINE, so a same-line `</answer>` leaks into the
#      prediction, and a bare `\boxed{}` answer block yields [INVALID]. Under GRPO every
#      rollout would then score -1, the group-relative advantage would be identically 0, and
#      nothing would learn - a failure that looks exactly like a healthy run that is not
#      improving. So the rollout dumps are scanned for the score distribution and the
#      [INVALID] rate, and a uniformly -1 batch is reported as a FAILURE.
#
# Usage:  bash smoke_test_orz7b_2+3.sh
# Env:    MODEL_PATH, TRAIN_FILE, TEST_FILE, n_resp_per_prompt, test_freq, save_freq, ...
# =============================================================================

set -xeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../../../../.." && pwd)"
cd -- "${REPO_ROOT}"

ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg_orz7b.sh"}
[[ -f "${ARM_SCRIPT}" ]] || { echo "no such arm script: ${ARM_SCRIPT}" >&2; exit 2; }

# The hub id works as-is: ORZ is Qwen2ForCausalLM, which is in verl's mcore registry, and its
# tokenizer is a stock Qwen2TokenizerFast. Unlike the openPangu smoke there is no
# trust_remote_code and no transformers_modules dance on PYTHONPATH.
MODEL_PATH=${MODEL_PATH:-"Open-Reasoner-Zero/Open-Reasoner-Zero-7B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# AIME-2024 only, as a single-element list: the arm validates on 2024+2025 by default.
# The _smoke file is aime-2024 with its 32x duplication removed - 30 distinct problems
# instead of 960 rows, all ground truths agreeing within each duplicate group (checked
# before it was written). Validation is the dominant cost of this test at 8192 tokens, and
# this makes it 32x cheaper. The price is that val-core/math_dapo/acc is mean@1 over 30
# problems: fine for a plumbing check, far too noisy to compare arms with. For reference,
# ORZ-7B scored 5/30 there under the arm's own scorer at T=1.0/top_p=1.0.
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024_smoke.parquet']"}

# no spaces or slashes: the arm's log dir is logs/${exp_name//\//_}
exp_name=${exp_name:-"SMOKE-orz7b-2+3"}
exp_name=${exp_name//[^A-Za-z0-9+-]/-}

# ---- 2 + 3 layout on the first five GPUs -------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-5}
export n_gpus_rollout=${n_gpus_rollout:-2}   # -> n_gpus_training = 5 - 2 = 3

# ---- 2 steps, cheap ones -----------------------------------------------------
export MODEL_PATH TRAIN_FILE TEST_FILE exp_name
export n_resp_per_prompt=${n_resp_per_prompt:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-3}
export max_response_length=${max_response_length:-8192}   # the arm's own length, not a shortened one
export total_rollout_steps=${total_rollout_steps:-6}
export test_freq=${test_freq:-1}          # validate after every param version
export save_freq=${save_freq:-1}          # and checkpoint every one
export val_before_train=${val_before_train:-False}  # override to True for the 0.167 reference; see the header
export ppo_epochs=${ppo_epochs:-2}        # unchanged from the arm: cheap at 6 sequences
export entropy_coeff=${entropy_coeff:-0.01}  # see the header: the only reward-independent gradient
export lr=${lr:-1e-4}                     # 100x the arm's, so 4 updates clear bf16 rounding

start_time=$(date +%s)
bash "${ARM_SCRIPT}" "$@"
# stderr, so that `bash smoke_test_orz7b_2+3.sh --cfg job --resolve` yields clean YAML
echo "[smoke] training finished in $(( $(date +%s) - start_time ))s" >&2

# `--cfg job` and friends make the arm print its config and exit: nothing to verify then.
for arg in "$@"; do
    case "${arg}" in --cfg|--help|--hydra-help|-h) exit 0 ;; esac
done

CKPTS_DIR="logs/${exp_name//\//_}"
set +x
echo "==================== checkpoint verification ===================="
# no --dtype: megatron's hf_model save writes bf16 (actor.megatron.dtype=bfloat16), which is
# verify_checkpoints.py's default.
python "${HERE}/verify_checkpoints.py" "${CKPTS_DIR}" --expect 2 --base-model "${MODEL_PATH}"

echo "==================== validation accuracy ===================="
python - "${CKPTS_DIR}" <<'PY'
"""Print val-core/*/acc/mean@1 from the run's own TensorBoard events.

By default there are two points, one after each trainer step, and both are of a model that
lr=1e-4 has deliberately driven hard - they say the validation path works, not that the model is
good. Under val_before_train=True a step-0 point appears as well; that one IS comparable, because
the offline vLLM probe scored 5/30 = 0.167 on exactly these 30 problems at the same sampling, and
a value near 0 there is a wiring failure rather than a bad model. It is annotated when present.
"""

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROBE_REFERENCE = 5 / 30  # ORZ-7B, aime-2024_smoke, T=1.0/top_p=1.0, tag-aware scorer

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
"""Did the tag-aware scorer actually extract answers from ORZ's <answer> blocks?

A uniformly -1 batch is the silent failure this arm exists to prevent: with stock math_dapo
every ORZ rollout scores -1, the GRPO advantage is identically 0, and the run looks healthy
while learning nothing. Exit non-zero if that is what the dumps show.
"""

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
if "pred" not in (rows[0] if rows else {}):
    failures.append("no 'pred' field in the dumps: the custom reward function did not run")
if len(scores) == 1 and next(iter(scores)) is not None and next(iter(scores)) < 0:
    failures.append("every sample scored -1: the advantage is identically 0 and nothing can learn")
if preds and invalid == len(preds):
    failures.append("every prediction is [INVALID]: extraction is broken")
if tagged == 0:
    failures.append("no response contained <answer>: ORZ's chat template did not apply")

if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("OK: answers were extracted and the reward is not degenerate")
PY
