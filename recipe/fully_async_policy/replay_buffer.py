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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class GroupEntry:
    sample: Any
    group_version: int
    score: float
    insert_seq: int
    times_trained: int = 0

    def staleness(self, current_version: int) -> int:
        return int(current_version) - int(self.group_version)


def staleness_score(staleness: int, tau: float) -> float:
    return float(2.0 ** (-float(staleness) / float(tau)))


def reuse_score(times_trained: int, reuse_halflife: Optional[float]) -> float:
    if reuse_halflife is None:
        return 1.0
    return float(2.0 ** (-float(times_trained) / float(reuse_halflife)))


def sampling_weight(entry: "GroupEntry", reuse_halflife: Optional[float]) -> float:
    return float(entry.score) * reuse_score(entry.times_trained, reuse_halflife)


class ReplayBuffer:
    def __init__(
        self,
        tau: float,
        staleness_threshold: int,
        seed: int = 1234,
        reuse_halflife: Optional[float] = None,
    ):
        assert tau > 0, f"replay_buffer.tau must be positive, got {tau}"
        assert staleness_threshold >= 0, f"replay_buffer.staleness_threshold must be >= 0, got {staleness_threshold}"
        self.tau = float(tau)
        self.staleness_threshold = int(staleness_threshold)
        if reuse_halflife is not None and float(reuse_halflife) > 0:
            assert np.isfinite(float(reuse_halflife)), (
                f"replay_buffer.reuse_halflife must be finite, got {reuse_halflife}"
            )
            self.reuse_halflife: Optional[float] = float(reuse_halflife)
        else:
            self.reuse_halflife = None
        self.rng = np.random.default_rng(seed)
        self.entries: list[GroupEntry] = []
        self.pending_fresh: list[GroupEntry] = []
        self._next_insert_seq = 0
        self.total_added = 0
        self.evicted_total = 0
        self.evicted_unseen_total = 0
        self.evicted_trained_once_total = 0

    def add(self, sample: Any, current_version: int) -> GroupEntry:
        group_version = int(getattr(sample, "group_version", 0))
        entry = GroupEntry(
            sample=sample,
            group_version=group_version,
            score=staleness_score(int(current_version) - group_version, self.tau),
            insert_seq=self._next_insert_seq,
        )
        self._next_insert_seq += 1
        self.entries.append(entry)
        self.pending_fresh.append(entry)
        self.total_added += 1
        return entry

    def evict(self, current_version: int) -> tuple[int, int]:
        kept: list[GroupEntry] = []
        evicted = 0
        evicted_unseen = 0
        evicted_trained_once = 0
        for entry in self.entries:
            if entry.staleness(current_version) > self.staleness_threshold:
                evicted += 1
                if entry.times_trained == 0:
                    evicted_unseen += 1
                elif entry.times_trained == 1:
                    evicted_trained_once += 1
            else:
                kept.append(entry)
        self.entries = kept
        if evicted:
            kept_ids = set(id(e) for e in kept)
            self.pending_fresh = [e for e in self.pending_fresh if id(e) in kept_ids]
        self.evicted_total += evicted
        self.evicted_unseen_total += evicted_unseen
        self.evicted_trained_once_total += evicted_trained_once
        return evicted, evicted_unseen

    def recompute_scores(self, current_version: int) -> None:
        for entry in self.entries:
            entry.score = staleness_score(entry.staleness(current_version), self.tau)

    def mark_trained(self, entries: list[GroupEntry]) -> None:
        for entry in entries:
            entry.times_trained += 1

    def compose_minibatch(self, mini_size: int, current_version: int) -> tuple[list[GroupEntry], dict]:
        if len(self.entries) < mini_size:
            raise ValueError(
                f"Replay buffer holds {len(self.entries)} groups < mini_size {mini_size}; "
                "caller must enforce the pause watermark before composing"
            )
        fresh = sorted(self.pending_fresh, key=lambda e: e.insert_seq, reverse=True)
        self.pending_fresh = []
        selected = fresh[:mini_size]
        n_fresh = len(selected)
        n_fill = mini_size - n_fresh
        if n_fill > 0:
            selected_set = set(id(e) for e in selected)
            pool = [e for e in self.entries if id(e) not in selected_set]
            weights = np.asarray([sampling_weight(e, self.reuse_halflife) for e in pool], dtype=np.float64)
            total = weights.sum()
            if not np.isfinite(total) or total <= 0.0:
                probs = np.full(len(pool), 1.0 / len(pool))
            else:
                probs = weights / total
            fill_idx = self.rng.choice(len(pool), size=n_fill, replace=False, p=probs)
            selected = selected + [pool[i] for i in fill_idx]
        staleness = [e.staleness(current_version) for e in selected]
        info = {
            "n_new": n_fresh,
            "n_replayed": mini_size - n_fresh,
            "staleness": staleness,
            "fresh_staleness": staleness[:n_fresh],
            "times_trained": [e.times_trained for e in selected],
        }
        return selected, info

    def size(self) -> int:
        return len(self.entries)

    def pending_fresh_count(self) -> int:
        return len(self.pending_fresh)

    def untrained_count(self) -> int:
        return sum(1 for e in self.entries if e.times_trained == 0)

    def staleness_list(self, current_version: int) -> list[int]:
        return [e.staleness(current_version) for e in self.entries]

    def times_trained_list(self) -> list[int]:
        return [e.times_trained for e in self.entries]

    def max_staleness(self, current_version: int) -> Optional[int]:
        if not self.entries:
            return None
        return max(e.staleness(current_version) for e in self.entries)

    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "staleness_threshold": self.staleness_threshold,
            "next_insert_seq": self._next_insert_seq,
            "total_added": self.total_added,
            "evicted_total": self.evicted_total,
            "evicted_unseen_total": self.evicted_unseen_total,
            "evicted_trained_once_total": self.evicted_trained_once_total,
            "rng_state": self.rng.bit_generator.state,
            "pending_seqs": [e.insert_seq for e in self.pending_fresh],
            "entries": [
                {
                    "sample": e.sample,
                    "group_version": e.group_version,
                    "score": e.score,
                    "insert_seq": e.insert_seq,
                    "times_trained": e.times_trained,
                }
                for e in self.entries
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._next_insert_seq = int(state.get("next_insert_seq", 0))
        self.total_added = int(state.get("total_added", 0))
        self.evicted_total = int(state.get("evicted_total", 0))
        self.evicted_unseen_total = int(state.get("evicted_unseen_total", 0))
        self.evicted_trained_once_total = int(state.get("evicted_trained_once_total", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.entries = [
            GroupEntry(
                sample=d["sample"],
                group_version=int(d["group_version"]),
                score=float(d["score"]),
                insert_seq=int(d["insert_seq"]),
                times_trained=int(d.get("times_trained", 0 if d.get("is_new", True) else 1)),
            )
            for d in state.get("entries", [])
        ]
        pending_seqs = set(int(s) for s in state.get("pending_seqs", []))
        self.pending_fresh = [e for e in self.entries if e.insert_seq in pending_seqs]
