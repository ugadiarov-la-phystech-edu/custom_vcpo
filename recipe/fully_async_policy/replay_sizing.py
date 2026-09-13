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
"""Sizing helpers shared by the fully-async TRAINER and ROLLOUTER (no ray/torch imports).

- trainer_dp_size / first_minibatch_groups: the trainer's data-parallel size and the size of the
  first replay mini-batch when replay_buffer.requires_mini_batches < 1 (see fully_async_trainer.py).
- parse_concurrency_ramp / concurrency_cap: the rollouter's staged in-flight cap
  (async_training.concurrency_ramp), keyed on the number of groups DELIVERED to the message queue.
"""

import math

__all__ = ["trainer_dp_size", "first_minibatch_groups", "parse_concurrency_ramp", "concurrency_cap"]


def trainer_dp_size(config) -> int:
    """Data-parallel size of the TRAINER worker group, from the config alone (the trainer
    never queries the workers' parallel state): trainer GPUs divided by the model-parallel
    degree — tp x pp x cp for Megatron, ulysses_sequence_parallel_size for fsdp/fsdp2."""
    gpus = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    actor_cfg = config.actor_rollout_ref.actor
    strategy = str(actor_cfg.get("strategy", "megatron"))
    if strategy == "megatron":
        mp = int(actor_cfg.megatron.get("tensor_model_parallel_size", 1))
        mp *= int(actor_cfg.megatron.get("pipeline_model_parallel_size", 1))
        mp *= int(actor_cfg.megatron.get("context_parallel_size", 1))
    else:
        mp = int(actor_cfg.get("ulysses_sequence_parallel_size", 1))
    if mp < 1 or gpus % mp != 0:
        raise ValueError(f"trainer GPUs ({gpus}) are not divisible by the model-parallel degree ({mp})")
    return gpus // mp


def first_minibatch_groups(requires_mini_batches: float, mini_size: int, n: int, dp: int) -> int | None:
    """replay_buffer.requires_mini_batches in (0, 1): the size, in GROUPS, of the FIRST
    replay mini-batch — the smallest g >= requires_mini_batches x mini_size whose g x n
    sequences split evenly over the dp trainer ranks, capped at mini_size. None (no special
    first mini-batch) for values >= 1, which keep the watermark semantics. E.g. 0.5 x 33
    = 16.5 -> 17; 17 x 16 = 272 does not divide by dp=3, 18 x 16 = 288 does -> 18."""
    rmb = float(requires_mini_batches)
    if rmb <= 0:
        raise ValueError(f"replay_buffer.requires_mini_batches must be > 0, got {requires_mini_batches!r}")
    if rmb >= 1:
        return None
    if mini_size < 1 or n < 1 or dp < 1:
        raise ValueError(f"invalid mini_size={mini_size}, n={n}, dp={dp}")
    g = max(1, math.ceil(rmb * mini_size - 1e-9))
    while g < mini_size and (g * n) % dp != 0:
        g += 1
    return min(g, mini_size)


def parse_concurrency_ramp(value) -> list[int]:
    """async_training.concurrency_ramp -> list of per-engine caps for the warm-up stages.

    Accepts None / "" / "null" / [] (ramp OFF -> []), a list (or OmegaConf ListConfig) of positive
    ints, or the string form "[4,8,16]" / "4,8,16". Stages must be non-decreasing."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "null", "none", "[]"):
            return []
        items = [t for t in text.strip("[]() ").replace(";", ",").split(",") if t.strip()]
    else:
        try:
            items = list(value)
        except TypeError as e:
            raise ValueError(f"async_training.concurrency_ramp must be a list of positive ints, got {value!r}") from e
    ramp: list[int] = []
    for item in items:
        try:
            f = float(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"async_training.concurrency_ramp entries must be ints, got {item!r}") from e
        if f != int(f) or int(f) < 1:
            raise ValueError(f"async_training.concurrency_ramp entries must be positive ints, got {item!r}")
        ramp.append(int(f))
    for a, b in zip(ramp, ramp[1:], strict=False):
        if b < a:
            raise ValueError(f"async_training.concurrency_ramp must be non-decreasing, got {ramp}")
    return ramp


def concurrency_cap(
    delivered: int,
    ramp: list[int],
    n_engines: int,
    first_size: int,
    mini_size: int,
    full_cap: int,
    hard_cap: int | None = None,
) -> int:
    """In-flight group cap for the rollouter given `delivered` groups pushed to the queue so far.

    Stage i (cap ramp[i] x n_engines) is active while delivered < T_i, with T_0 = first_size (the
    trainer's first mini-batch) and T_i = first_size + i x mini_size; after the last stage the cap is
    `full_cap` (n_engines x bsz_per_dp_rank, today's value). Every stage is bounded by `hard_cap`
    (the staleness quota max_required_samples) and never below 1. An empty ramp returns full_cap."""
    cap = full_cap
    for i, per_engine in enumerate(ramp):
        threshold = first_size + i * mini_size
        if delivered < threshold:
            cap = per_engine * n_engines
            break
    if hard_cap is not None:
        cap = min(cap, hard_cap)
    return max(1, int(cap))
