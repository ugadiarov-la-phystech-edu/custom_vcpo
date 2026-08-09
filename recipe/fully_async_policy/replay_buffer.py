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
"""Trainer-local replay buffer of prompt-groups for the fully-async pipeline.

Decouples generation from updating: the rollouter streams completed
(non-degenerate) groups through the MessageQueue as before, but the trainer
keeps them in this buffer and composes each optimizer mini-batch as
   1) every not-yet-trained-on group (``is_new``), oldest-inserted first,
      capped at the mini-batch size, then
   2) a without-replacement sample of already-used groups with probability
      proportional to the staleness-decayed score ``2^(-staleness/tau)``.
Groups staler than ``staleness_threshold`` model updates are evicted after
every update; scores are recomputed against the new model version.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class GroupEntry:
    """One prompt-group (a `RolloutSample`) with replay bookkeeping."""

    sample: Any
    group_version: int
    is_new: bool
    score: float
    insert_seq: int

    def staleness(self, current_version: int) -> int:
        return int(current_version) - int(self.group_version)


def staleness_score(staleness: int, tau: float) -> float:
    """score = exp(-staleness * ln2 / tau) = 2^(-staleness / tau)."""
    return float(2.0 ** (-float(staleness) / float(tau)))


class ReplayBuffer:
    """Version-aware group buffer: add / compose / evict / rescore / mark-used.

    Lives in the trainer driver process (plain Python object, no Ray). The
    MessageQueue remains the rollouter->trainer transport; this class only
    stores what the trainer has already drained.
    """

    def __init__(self, tau: float, staleness_threshold: int, seed: int = 1234):
        assert tau > 0, f"replay_buffer.tau must be positive, got {tau}"
        assert staleness_threshold >= 0, (
            f"replay_buffer.staleness_threshold must be >= 0, got {staleness_threshold}"
        )
        self.tau = float(tau)
        self.staleness_threshold = int(staleness_threshold)
        self.rng = np.random.default_rng(seed)
        self.entries: list[GroupEntry] = []
        self._next_insert_seq = 0
        # Lifetime counters (survive checkpointing).
        self.total_added = 0
        self.evicted_total = 0
        self.evicted_unseen_total = 0

    # ---------------- mutation ----------------

    def add(self, sample: Any, current_version: int) -> GroupEntry:
        group_version = int(getattr(sample, "group_version", 0))
        entry = GroupEntry(
            sample=sample,
            group_version=group_version,
            is_new=True,
            score=staleness_score(int(current_version) - group_version, self.tau),
            insert_seq=self._next_insert_seq,
        )
        self._next_insert_seq += 1
        self.entries.append(entry)
        self.total_added += 1
        return entry

    def evict(self, current_version: int) -> tuple[int, int]:
        """Remove groups with staleness > staleness_threshold.

        Returns (evicted, evicted_unseen); unseen evictions are generated-but-
        never-trained-on groups, i.e. wasted rollout compute worth monitoring.
        """
        kept: list[GroupEntry] = []
        evicted = 0
        evicted_unseen = 0
        for entry in self.entries:
            if entry.staleness(current_version) > self.staleness_threshold:
                evicted += 1
                if entry.is_new:
                    evicted_unseen += 1
            else:
                kept.append(entry)
        self.entries = kept
        self.evicted_total += evicted
        self.evicted_unseen_total += evicted_unseen
        return evicted, evicted_unseen

    def recompute_scores(self, current_version: int) -> None:
        for entry in self.entries:
            entry.score = staleness_score(entry.staleness(current_version), self.tau)

    def mark_used(self, entries: list[GroupEntry]) -> None:
        for entry in entries:
            entry.is_new = False

    # ---------------- composition ----------------

    def compose_minibatch(self, mini_size: int, current_version: int) -> tuple[list[GroupEntry], dict]:
        """Compose one mini-batch of ``mini_size`` groups.

        Priority: all ``is_new`` groups, oldest-inserted first (so no unseen
        group waits indefinitely toward eviction), capped at ``mini_size``.
        The remainder is sampled without replacement from the other groups
        with probability proportional to their staleness-decayed score.
        """
        if len(self.entries) < mini_size:
            raise ValueError(
                f"Replay buffer holds {len(self.entries)} groups < mini_size {mini_size}; "
                "caller must enforce the pause watermark before composing"
            )
        new_entries = sorted(
            (e for e in self.entries if e.is_new), key=lambda e: e.insert_seq
        )
        selected = new_entries[:mini_size]
        n_new = len(selected)
        n_fill = mini_size - n_new
        if n_fill > 0:
            selected_set = set(id(e) for e in selected)
            pool = [e for e in self.entries if id(e) not in selected_set]
            weights = np.asarray([e.score for e in pool], dtype=np.float64)
            total = weights.sum()
            if not np.isfinite(total) or total <= 0.0:
                # All scores underflowed (extreme staleness): fall back to uniform.
                probs = np.full(len(pool), 1.0 / len(pool))
            else:
                probs = weights / total
            fill_idx = self.rng.choice(len(pool), size=n_fill, replace=False, p=probs)
            selected = selected + [pool[i] for i in fill_idx]
        staleness = [e.staleness(current_version) for e in selected]
        info = {
            "n_new": n_new,
            "n_replayed": mini_size - n_new,
            "staleness": staleness,
        }
        return selected, info

    def take_oldest_new(self, mini_size: int) -> list[GroupEntry]:
        """Warm-up composition: the oldest-inserted ``mini_size`` unseen groups."""
        new_entries = sorted(
            (e for e in self.entries if e.is_new), key=lambda e: e.insert_seq
        )
        if len(new_entries) < mini_size:
            raise ValueError(
                f"Only {len(new_entries)} unseen groups available < mini_size {mini_size}"
            )
        return new_entries[:mini_size]

    # ---------------- introspection ----------------

    def size(self) -> int:
        return len(self.entries)

    def new_count(self) -> int:
        return sum(1 for e in self.entries if e.is_new)

    def staleness_list(self, current_version: int) -> list[int]:
        return [e.staleness(current_version) for e in self.entries]

    def max_staleness(self, current_version: int) -> Optional[int]:
        if not self.entries:
            return None
        return max(e.staleness(current_version) for e in self.entries)

    # ---------------- checkpointing ----------------

    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "staleness_threshold": self.staleness_threshold,
            "next_insert_seq": self._next_insert_seq,
            "total_added": self.total_added,
            "evicted_total": self.evicted_total,
            "evicted_unseen_total": self.evicted_unseen_total,
            "rng_state": self.rng.bit_generator.state,
            "entries": [
                {
                    "sample": e.sample,
                    "group_version": e.group_version,
                    "is_new": e.is_new,
                    "score": e.score,
                    "insert_seq": e.insert_seq,
                }
                for e in self.entries
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        # tau / staleness_threshold stay config-driven; only dynamic state is restored.
        self._next_insert_seq = int(state.get("next_insert_seq", 0))
        self.total_added = int(state.get("total_added", 0))
        self.evicted_total = int(state.get("evicted_total", 0))
        self.evicted_unseen_total = int(state.get("evicted_unseen_total", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.entries = [
            GroupEntry(
                sample=d["sample"],
                group_version=int(d["group_version"]),
                is_new=bool(d["is_new"]),
                score=float(d["score"]),
                insert_seq=int(d["insert_seq"]),
            )
            for d in state.get("entries", [])
        ]
