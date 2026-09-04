# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify the checkpoints a fully_async_policy run wrote.

Checks what `save_contents=['hf_model']` is supposed to produce, without loading a full model:

  global_step_N/
    actor/huggingface/   config.json + tokenizer + bf16 safetensors, loadable by vLLM as is
    timing_state.json    run/checkpoint datetimes + the cumulative timing totals
    data.pt              the rollouter's dataloader state
  latest_checkpointed_iteration.txt

Usage: python verify_checkpoints.py <ckpt_dir> [--expect N] [--base-model Qwen/Qwen3-8B]
Exits non-zero and prints every failure it found.
"""

import argparse
import json
import os
import sys
from datetime import datetime

TIMING_KEYS = (
    "run_start_datetime",
    "checkpoint_datetime",
    "wall_time_since_first_sample",
    "cumulative_validation_time",
    "cumulative_save_time",
    "cumulative_training_time",
)
# A checkpoint needs the tokenizer config plus at least one vocabulary artifact. Which artifact
# depends on the model: fast tokenizers ship tokenizer.json, slow/sentencepiece ones (openPangu's
# PanguTokenizer, most Llama-family originals) ship tokenizer.model and no tokenizer.json at all.
TOKENIZER_CONFIG = "tokenizer_config.json"
VOCAB_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "spiece.model")

# Buffers transformers recomputes at load time from the config and deliberately does NOT save.
# A base checkpoint exported by an older transformers may still contain them (openPangu ships 34
# rotary_emb.inv_freq, one per layer), so comparing raw key sets reports them as "missing" from a
# perfectly good save.
NON_PERSISTENT_BUFFER_SUFFIXES = ("rotary_emb.inv_freq", "masked_bias", "attention.bias")


def tokenizer_files_present(files):
    """Returns (ok, missing-description) for the tokenizer artifacts in a checkpoint directory."""
    if TOKENIZER_CONFIG not in files:
        return False, TOKENIZER_CONFIG
    if not any(v in files for v in VOCAB_FILES):
        return False, f"one of {list(VOCAB_FILES)}"
    return True, ""


def is_non_persistent_buffer(key):
    return key.endswith(NON_PERSISTENT_BUFFER_SUFFIXES)


class Report:
    def __init__(self):
        self.failures = []
        self.notes = []

    def check(self, ok, message):
        (self.notes if ok else self.failures).append(("PASS" if ok else "FAIL") + ": " + message)
        return ok


def _safetensors_state(hf_dir):
    """Map parameter name -> (dtype, shape) across every shard, plus the shard list."""
    from safetensors import safe_open

    shards = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
    state = {}
    for shard in shards:
        with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                state[key] = (tensor_slice.get_dtype(), tuple(tensor_slice.get_shape()))
    return state, shards


def _first_tensor(hf_dir, key):
    from safetensors import safe_open

    for shard in sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def verify_checkpoint(step_dir, report, base_state=None, expect_dtype=None, check_timing=True):
    """Everything one global_step_N directory must contain. Returns its parsed timing state.

    expect_dtype pins the weight dtype ("BF16"/"F32"); the megatron arms save bf16, while an FSDP2
    arm at the default fsdp_config.model_dtype=fp32 writes fp32 (get_fsdp_full_state_dict does not
    cast). None only requires that one dtype is used throughout.

    check_timing=False is for verl.trainer.main_ppo checkpoints (the synchronous arms): only the
    fully-async trainer writes timing_state.json, so those checks are skipped and {} is returned.
    """
    name = os.path.basename(step_dir)
    hf_dir = os.path.join(step_dir, "actor", "huggingface")

    if not report.check(os.path.isdir(hf_dir), f"{name}: actor/huggingface/ exists"):
        return None

    files = set(os.listdir(hf_dir))
    report.check("config.json" in files, f"{name}: config.json written")
    tokenizer_ok, tokenizer_missing = tokenizer_files_present(files)
    report.check(tokenizer_ok, f"{name}: tokenizer written (missing: {tokenizer_missing})")
    report.check(
        any(f.endswith(".safetensors") for f in files),
        f"{name}: weights written (files: {sorted(files)})",
    )
    # save_contents=['hf_model'] must not leave a sharded checkpoint behind, on either backend
    report.check(
        not os.path.exists(os.path.join(step_dir, "actor", "dist_ckpt")),
        f"{name}: no megatron dist_ckpt/ (hf_model-only save)",
    )
    # the FSDP manager writes per-rank shards under the same actor/ directory; fsdp_config.json
    # is written unconditionally by rank 0 and is expected
    shard_prefixes = ("model_world_size_", "optim_world_size_", "extra_state_world_size_")
    actor_files = os.listdir(os.path.join(step_dir, "actor"))
    fsdp_shards = [f for f in actor_files if f.startswith(shard_prefixes)]
    report.check(not fsdp_shards, f"{name}: no FSDP per-rank shards (found: {fsdp_shards[:4]})")
    report.check(os.path.exists(os.path.join(step_dir, "data.pt")), f"{name}: rollouter dataloader state saved")

    # the weights themselves
    state, shards = _safetensors_state(hf_dir)
    report.check(bool(state), f"{name}: safetensors readable ({len(state)} tensors in {len(shards)} shard(s))")
    dtypes = {dtype for dtype, _ in state.values()}
    report.check(len(dtypes) == 1, f"{name}: one dtype across all tensors (found {sorted(dtypes)})")
    if expect_dtype:
        report.check(dtypes == {expect_dtype}, f"{name}: all tensors {expect_dtype} (found {sorted(dtypes)})")
    if len(shards) > 1:
        report.check("model.safetensors.index.json" in files, f"{name}: sharded weights have an index")

    if base_state is not None:
        missing = sorted(k for k in set(base_state) - set(state) if not is_non_persistent_buffer(k))
        skipped = sorted(k for k in set(base_state) - set(state) if is_non_persistent_buffer(k))
        if skipped:
            report.check(True, f"{name}: {len(skipped)} non-persistent buffers not saved, as expected")
        extra = sorted(set(state) - set(base_state))
        report.check(not missing, f"{name}: no parameter missing vs the base model (missing: {missing[:5]})")
        report.check(not extra, f"{name}: no unexpected parameter vs the base model (extra: {extra[:5]})")
        mismatched = [k for k in set(state) & set(base_state) if state[k][1] != base_state[k][1]]
        report.check(not mismatched, f"{name}: shapes match the base model (mismatched: {mismatched[:5]})")

    # the config must describe these weights
    config_path = os.path.join(hf_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        report.check("architectures" in config, f"{name}: config carries architectures={config.get('architectures')}")

    # tokenizer must actually load from the checkpoint (this is what vLLM does)
    try:
        from transformers import AutoTokenizer

        # trust_remote_code: openPangu keeps a custom tokenizer class after the config re-alias
        # (tokenizer_config.json carries its own auto_map), and save_pretrained copies the module
        # into the checkpoint - without the flag this fails on a perfectly good checkpoint.
        tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
        report.check(tokenizer.vocab_size > 0, f"{name}: AutoTokenizer.from_pretrained works")
    except Exception as exc:  # noqa: BLE001
        report.check(False, f"{name}: AutoTokenizer.from_pretrained failed: {exc}")

    # timing state (fully-async trainer only)
    if not check_timing:
        return {}
    timing_path = os.path.join(step_dir, "timing_state.json")
    if not report.check(os.path.exists(timing_path), f"{name}: timing_state.json written"):
        return None
    with open(timing_path) as f:
        timing = json.load(f)
    missing_keys = [k for k in TIMING_KEYS if k not in timing]
    report.check(not missing_keys, f"{name}: timing_state.json complete (missing: {missing_keys})")
    for key in ("run_start_datetime", "checkpoint_datetime"):
        if key in timing:
            try:
                parsed = datetime.fromisoformat(timing[key])
                report.check(parsed.tzinfo is not None, f"{name}: {key}={timing[key]} is timezone-aware")
            except ValueError as exc:
                report.check(False, f"{name}: {key} unparseable: {exc}")
    return timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="trainer.default_local_dir of the run")
    parser.add_argument("--expect", type=int, default=None, help="expected number of checkpoints")
    parser.add_argument("--base-model", default=None, help="base model to compare parameter names/shapes against")
    parser.add_argument(
        "--dtype",
        default="BF16",
        help="expected weight dtype: BF16 (megatron / bf16 FSDP arms), F32 (FSDP at model_dtype=fp32), "
        'or "any" to only require that one dtype is used throughout',
    )
    parser.add_argument(
        "--no-timing-state",
        action="store_true",
        help="checkpoints from verl.trainer.main_ppo (sync arms): skip the timing_state.json checks, which only "
        "the fully-async trainer writes",
    )
    args = parser.parse_args()

    report = Report()
    steps = sorted(
        (d for d in os.listdir(args.ckpt_dir) if d.startswith("global_step_")),
        key=lambda d: int(d.rsplit("_", 1)[1]),
    )
    report.check(bool(steps), f"checkpoints found in {args.ckpt_dir}: {steps}")
    if args.expect is not None:
        report.check(len(steps) == args.expect, f"expected {args.expect} checkpoints, found {len(steps)}: {steps}")

    base_state = None
    if args.base_model:
        try:
            from transformers.utils import cached_file

            # any file resolves the snapshot directory; the index is present for sharded models
            base_dir = os.path.dirname(cached_file(args.base_model, "config.json"))
            base_state, _ = _safetensors_state(base_dir)
            print(f"base model {args.base_model}: {len(base_state)} tensors from {base_dir}")
        except Exception as exc:  # noqa: BLE001
            report.check(False, f"could not read the base model {args.base_model}: {exc}")

    timings = []
    for step in steps:
        timing = verify_checkpoint(
            os.path.join(args.ckpt_dir, step),
            report,
            base_state=base_state,
            expect_dtype=None if args.dtype.lower() == "any" else args.dtype.upper(),
            check_timing=not args.no_timing_state,
        )
        timings.append((step, timing))

    # cross-checkpoint invariants
    complete = [(s, t) for s, t in timings if t]
    if complete:
        # The trainer learns the timing anchor (the rollouter's first training sample) from the
        # validation stream, which it drains at the NEXT parameter sync - so the very first
        # checkpoint of a run legitimately carries zeros. Every later one must be live.
        for step, timing in complete[1:]:
            report.check(
                timing["cumulative_training_time"] > 0,
                f"{step}: cumulative_training_time={timing['cumulative_training_time']} > 0",
            )
            report.check(
                timing["wall_time_since_first_sample"] > 0,
                f"{step}: wall_time_since_first_sample={timing['wall_time_since_first_sample']} > 0",
            )
    if len(complete) > 1:
        starts = {t["run_start_datetime"] for _, t in complete}
        report.check(len(starts) == 1, f"all checkpoints share one run_start_datetime: {starts}")
        stamps = [datetime.fromisoformat(t["checkpoint_datetime"]) for _, t in complete]
        report.check(stamps == sorted(stamps), f"checkpoint_datetime increases with the step: {stamps}")
        totals = [t["cumulative_training_time"] for _, t in complete]
        report.check(
            all(b >= a for a, b in zip(totals, totals[1:], strict=False)),
            f"cumulative_training_time never goes backwards: {totals}",
        )

    # the weights must actually differ between checkpoints (training moved them)
    if len(steps) > 1:
        key = "model.layers.0.self_attn.q_proj.weight"
        tensors = [_first_tensor(os.path.join(args.ckpt_dir, s, "actor", "huggingface"), key) for s in steps[:2]]
        if all(t is not None for t in tensors):
            report.check(not tensors[0].equal(tensors[1]), f"{key} differs between {steps[0]} and {steps[1]}")
        else:
            report.check(False, f"{key} not found in the first two checkpoints")

    tracker = os.path.join(args.ckpt_dir, "latest_checkpointed_iteration.txt")
    if report.check(os.path.exists(tracker), "latest_checkpointed_iteration.txt written"):
        with open(tracker) as f:
            latest = f.read().strip()
        expected = steps[-1].rsplit("_", 1)[1] if steps else None
        report.check(latest == expected, f"tracker points at the newest checkpoint ({latest} vs {expected})")

    for line in report.notes:
        print(line)
    for line in report.failures:
        print(line)
    print(f"\n{len(report.notes)} checks passed, {len(report.failures)} failed")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
