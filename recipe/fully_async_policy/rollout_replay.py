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
"""Rollout-level |A_i|-prioritized experience replay for GRPO.

Implements the replay design of "Rollout-Level Advantage-Prioritized
Experience Replay for GRPO" (arXiv:2606.04560): a driver-side buffer of
*individual* rollouts (not prompt-groups) with

- fresh-anchored composition: every update trains on the current step's
  surviving fresh rollouts in full, plus a separately drawn replay portion of
  ``replay_ratio * B'_fresh`` rollouts concatenated on top;
- rollout-level PER priority: each rollout is drawn with probability
  proportional to ``(|A_i| + eps)^alpha`` (Schaul et al., 2016), where
  ``|A_i|`` is its GRPO group-relative advantage magnitude frozen at birth;
- age eviction: any rollout older than ``tau_max`` model updates is removed,
  independently of buffer fullness (a ``capacity`` FIFO backstop exists but
  rarely binds);
- frozen birth-step values: advantages, group statistics and the behavior
  log-probs (``rollout_log_probs``, aliased into ``old_log_probs``) are cached
  at insertion and never recomputed, so the PPO ratio of a replayed rollout
  measures drift from its birth step to the current policy.

Everything in this module is pure CPU / numpy / DataProto — no ray, no GPU —
so it is unit-testable with ``pytest ... -on_cpu.py``.
"""

from collections import deque
from typing import Optional

import numpy as np
import torch

from verl import DataProto

__all__ = [
    "AdvantagePrioritizedReplayBuffer",
    "resolve_replay_draw",
    "rollout_priorities",
    "zero_variance_group_mask",
]


def zero_variance_group_mask(scores: np.ndarray, uids: np.ndarray) -> np.ndarray:
    """Per-row keep mask for the DAPO-style zero-variance filter.

    A prompt-group (rows sharing a ``uid``) is kept iff its sequence-level
    scores are not all identical: all-correct / all-wrong groups produce
    exactly zero GRPO advantage for every member and carry no learning signal.
    A single-row group has zero variance by construction and is dropped.

    Args:
        scores: shape (bs,) sequence-level scores.
        uids: shape (bs,) group ids (object array).

    Returns:
        np.ndarray of bool, shape (bs,): True for rows of surviving groups.
    """
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.zeros(len(uids), dtype=bool)
    uid_to_rows: dict = {}
    for i, uid in enumerate(uids):
        uid_to_rows.setdefault(uid, []).append(i)
    for rows in uid_to_rows.values():
        group_scores = scores[rows]
        if len(rows) > 1 and (group_scores != group_scores[0]).any():
            keep[rows] = True
    return keep


def rollout_priorities(advantages: torch.Tensor, response_mask: torch.Tensor, eps: float) -> np.ndarray:
    """Per-rollout PER priority ``p_i = |A_i| + eps``.

    The GRPO outcome advantage is constant over a rollout's response tokens, so
    the masked mean of ``|advantages|`` recovers ``|A_i|`` exactly (and stays
    well-defined for an empty response, where the priority degrades to eps).
    """
    mask = response_mask.to(advantages.dtype)
    token_count = mask.sum(dim=-1).clamp(min=1.0)
    abs_adv = (advantages.abs() * mask).sum(dim=-1) / token_count
    return (abs_adv.detach().cpu().numpy() + eps).astype(np.float64)


def resolve_replay_draw(
    n_fresh: int,
    replay_ratio: float,
    buffer_size: int,
    dp_size: int,
    warmup_active: bool = False,
) -> tuple[int, int]:
    """Pick the replay draw size and (rarely) a fresh-row trim count.

    The target draw is ``round(replay_ratio * n_fresh)`` (Algorithm 1 step 3),
    bounded by what the buffer holds and forced to 0 during warmup. The
    gradient batch ``n_fresh - trim + draw`` must divide the trainer DP size
    (verl's nd dispatch and DataProto.make_iterator both assert equal chunks),
    so the draw is nudged by the minimal amount that lands on a multiple of
    ``dp_size``; only when no legal draw adjustment exists (warmup, empty
    buffer) are ``trim`` fresh rows dropped from the gradient batch instead
    (they still enter the buffer — the trim never loses data, only up to
    ``dp_size - 1`` rows of one update's signal).

    Returns:
        (draw_n, trim_n) with ``(n_fresh - trim_n + draw_n) % dp_size == 0``,
        ``0 <= draw_n <= buffer_size`` and ``0 <= trim_n < dp_size <= n_fresh``
        (trim_n == 0 whenever a draw adjustment suffices). n_fresh == 0 returns
        (0, 0).
    """
    assert dp_size >= 1
    if n_fresh <= 0:
        return 0, 0
    base = 0 if warmup_active else min(int(round(replay_ratio * n_fresh)), buffer_size)
    if (n_fresh + base) % dp_size == 0:
        return base, 0
    if not warmup_active:
        # Try draw adjustments by growing |delta|, preferring the draw closer to base.
        for delta in range(1, dp_size):
            for candidate in (base + delta, base - delta):
                if 0 <= candidate <= buffer_size and (n_fresh + candidate) % dp_size == 0:
                    return candidate, 0
    trim = (n_fresh + base) % dp_size
    if trim >= n_fresh:
        # Cannot trim the whole batch away; caller must handle the degenerate step.
        return base, 0
    return base, trim


class AdvantagePrioritizedReplayBuffer:
    """Rollout-level replay buffer with |A_i| PER priority and age eviction.

    Rows are stored in per-step blocks (all rollouts inserted together share a
    ``birth_version``), which makes age eviction O(blocks) and keeps the
    per-row tensors exactly as frozen at insertion. Sampling never removes
    entries; only ``evict_older_than`` and the capacity backstop do.
    """

    def __init__(
        self,
        replay_ratio: float = 0.5,
        priority_alpha: float = 0.5,
        priority_eps: float = 1e-6,
        tau_max: int = 10,
        warmup_steps: int = 20,
        capacity: int = 30000,
        seed: int = 1234,
        with_replacement: bool = False,
    ):
        assert replay_ratio >= 0.0
        assert 0.0 <= priority_alpha
        assert priority_eps > 0.0
        assert tau_max >= 0
        assert warmup_steps >= 0
        assert capacity >= 1
        self.replay_ratio = float(replay_ratio)
        self.priority_alpha = float(priority_alpha)
        self.priority_eps = float(priority_eps)
        self.tau_max = int(tau_max)
        self.warmup_steps = int(warmup_steps)
        self.capacity = int(capacity)
        self.with_replacement = bool(with_replacement)
        self._rng = np.random.default_rng(seed)
        # blocks of {"data": DataProto, "birth": int, "priorities": np.ndarray}
        self._blocks: deque = deque()
        self._size = 0

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def warmup_active(self, current_version: int) -> bool:
        """Replay stays disabled while fewer than warmup_steps updates happened."""
        return current_version < self.warmup_steps

    def add(self, rows: DataProto, priorities: np.ndarray, birth_version: int) -> None:
        """Insert one step's surviving fresh rollouts as a block."""
        priorities = np.asarray(priorities, dtype=np.float64)
        assert len(priorities) == len(rows), f"{len(priorities)=} != {len(rows)=}"
        if len(rows) == 0:
            return
        assert (priorities > 0.0).all(), "PER priorities must be positive (use |A_i| + eps)"
        self._blocks.append({"data": rows, "birth": int(birth_version), "priorities": priorities})
        self._size += len(rows)
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        """FIFO backstop: drop the oldest rows until size <= capacity."""
        overflow = self._size - self.capacity
        while overflow > 0 and self._blocks:
            oldest = self._blocks[0]
            n = len(oldest["priorities"])
            if n <= overflow:
                self._blocks.popleft()
                self._size -= n
                overflow -= n
            else:
                keep = np.arange(overflow, n)
                oldest["data"] = oldest["data"].select_idxs(keep.tolist())
                oldest["priorities"] = oldest["priorities"][keep]
                self._size -= overflow
                overflow = 0

    def evict_older_than(self, current_version: int) -> int:
        """Drop every block with age ``current_version - birth > tau_max``.

        Returns the number of evicted rows. Blocks are inserted in birth order,
        so eviction only ever pops from the left.
        """
        evicted = 0
        while self._blocks and current_version - self._blocks[0]["birth"] > self.tau_max:
            evicted += len(self._blocks[0]["priorities"])
            self._blocks.popleft()
        self._size -= evicted
        return evicted

    def sample(self, k: int, current_version: int) -> tuple[Optional[DataProto], dict]:
        """Draw ``k`` rollouts with probability ``p_i^alpha / sum_j p_j^alpha``.

        Without replacement by default (duplicate-free gradient batches); the
        canonical Schaul Eq.-3 with-replacement draw is available via
        ``with_replacement=True``. ``k >= size`` returns the whole buffer.
        Sampling does not consume entries.

        Returns:
            (rows, info): ``rows`` is a DataProto of the drawn rollouts (None
            when k <= 0 or the buffer is empty); ``info`` carries the drawn
            ages and priorities for metrics.
        """
        if k <= 0 or self._size == 0:
            return None, {"ages": np.array([], dtype=np.int64), "priorities": np.array([], dtype=np.float64)}
        k = min(int(k), self._size)
        priorities = np.concatenate([b["priorities"] for b in self._blocks])
        weights = priorities**self.priority_alpha
        probs = weights / weights.sum()
        flat_idxs = self._rng.choice(self._size, size=k, replace=self.with_replacement, p=probs)

        # Map flat indices back to (block, row) and slice per block, preserving
        # the drawn multiplicity (with replacement may repeat a row).
        block_starts = np.cumsum([0] + [len(b["priorities"]) for b in self._blocks])
        parts = []
        ages = np.empty(len(flat_idxs), dtype=np.int64)
        for order, flat in enumerate(flat_idxs):
            block_i = int(np.searchsorted(block_starts, flat, side="right")) - 1
            block = self._blocks[block_i]
            row_i = int(flat - block_starts[block_i])
            parts.append(block["data"].select_idxs([row_i]))
            ages[order] = current_version - block["birth"]
        rows = DataProto.concat(parts)
        info = {"ages": ages, "priorities": priorities[flat_idxs]}
        return rows, info

    def stats(self, current_version: int) -> dict:
        """Buffer-wide metrics (prefixed by the caller)."""
        if self._size == 0:
            return {"buffer_size": 0}
        ages = np.concatenate(
            [np.full(len(b["priorities"]), current_version - b["birth"], dtype=np.int64) for b in self._blocks]
        )
        priorities = np.concatenate([b["priorities"] for b in self._blocks])
        return {
            "buffer_size": self._size,
            "buffer_age_mean": float(ages.mean()),
            "buffer_age_max": int(ages.max()),
            "buffer_priority_mean": float(priorities.mean()),
            "buffer_priority_max": float(priorities.max()),
        }
