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
   1) the *fresh* groups — those added since the previous composition —
      newest-arrived first, capped at the mini-batch size; fresh groups that
      do not fit (the earliest arrivals of the window) simply join the replay
      pool (freshness is one-shot: it never carries over), then
   2) a without-replacement sample of the remaining groups with probability
      proportional to the staleness-decayed score ``2^(-staleness/tau)``.
Groups staler than ``staleness_threshold`` model updates are evicted after
every update; scores are recomputed against the new model version.

Freshness is deliberately NOT "never trained on": an absolute priority for
unseen groups lets a producer/consumer rate imbalance build an unbounded
unseen FIFO backlog whose head is many updates old — nominally "new"
mini-batches were ~36 updates stale in the ORZ-7B run, which drove the
off-policy collapse. With one-shot freshness the queue delay of prioritized
data is bounded by a single update.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class GroupEntry:
    """One prompt-group (a `RolloutSample`) with replay bookkeeping."""

    sample: Any
    group_version: int
    score: float
    insert_seq: int
    # Metrics only — never consulted by composition. Counts optimizer updates
    # this group participated in; 0 means generated-but-never-trained-on.
    times_trained: int = 0

    def staleness(self, current_version: int) -> int:
        return int(current_version) - int(self.group_version)


def staleness_score(staleness: int, tau: float) -> float:
    """score = exp(-staleness * ln2 / tau) = 2^(-staleness / tau)."""
    return float(2.0 ** (-float(staleness) / float(tau)))


class ReplayBuffer:
    """Version-aware group buffer: add / compose / evict / rescore / mark-trained.

    Lives in the trainer driver process (plain Python object, no Ray). The
    MessageQueue remains the rollouter->trainer transport; this class only
    stores what the trainer has already drained.
    """

    def __init__(self, tau: float, staleness_threshold: int, seed: int = 1234):
        assert tau > 0, f"replay_buffer.tau must be positive, got {tau}"
        assert staleness_threshold >= 0, f"replay_buffer.staleness_threshold must be >= 0, got {staleness_threshold}"
        self.tau = float(tau)
        self.staleness_threshold = int(staleness_threshold)
        self.rng = np.random.default_rng(seed)
        self.entries: list[GroupEntry] = []
        # Groups added since the last composition, in arrival order. Cleared
        # wholesale by compose_minibatch: freshness is one-shot.
        self.pending_fresh: list[GroupEntry] = []
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
            score=staleness_score(int(current_version) - group_version, self.tau),
            insert_seq=self._next_insert_seq,
        )
        self._next_insert_seq += 1
        self.entries.append(entry)
        self.pending_fresh.append(entry)
        self.total_added += 1
        return entry

    def evict(self, current_version: int) -> tuple[int, int]:
        """Remove groups with staleness > staleness_threshold.

        Returns (evicted, evicted_unseen); unseen evictions are groups with
        ``times_trained == 0`` — generated-but-never-trained-on, i.e. wasted
        rollout compute worth monitoring. Evicted groups also leave the
        pending-fresh list so they cannot resurface at composition.
        """
        kept: list[GroupEntry] = []
        evicted = 0
        evicted_unseen = 0
        for entry in self.entries:
            if entry.staleness(current_version) > self.staleness_threshold:
                evicted += 1
                if entry.times_trained == 0:
                    evicted_unseen += 1
            else:
                kept.append(entry)
        self.entries = kept
        if evicted:
            kept_ids = set(id(e) for e in kept)
            self.pending_fresh = [e for e in self.pending_fresh if id(e) in kept_ids]
        self.evicted_total += evicted
        self.evicted_unseen_total += evicted_unseen
        return evicted, evicted_unseen

    def recompute_scores(self, current_version: int) -> None:
        for entry in self.entries:
            entry.score = staleness_score(entry.staleness(current_version), self.tau)

    def mark_trained(self, entries: list[GroupEntry]) -> None:
        for entry in entries:
            entry.times_trained += 1

    # ---------------- composition ----------------

    def compose_minibatch(self, mini_size: int, current_version: int) -> tuple[list[GroupEntry], dict]:
        """Compose one mini-batch of ``mini_size`` groups, fresh-first.

        Fresh groups (added since the previous composition) go in first,
        newest-arrived first, capped at ``mini_size`` — the most on-policy
        data available trains immediately; the overflow (the window's earliest
        arrivals) stays in the buffer as ordinary replay candidates. The
        pending list is cleared either way — an unselected fresh group holds
        no priority next time. Note the starvation trade-off vs oldest-first:
        under a sustained arrival surplus the displaced earliest arrivals
        enter the pool at the low end of the weight distribution and may
        reach the eviction horizon untrained (visible as evicted_unseen).
        The remainder is sampled without replacement from the non-selected
        groups with probability proportional to their staleness-decayed
        score. Returned entries are ordered fresh-first, so
        ``selected[:info["n_new"]]`` is exactly the fresh prefix.
        """
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
            "n_new": n_fresh,
            "n_replayed": mini_size - n_fresh,
            "staleness": staleness,
        }
        return selected, info

    # ---------------- introspection ----------------

    def size(self) -> int:
        return len(self.entries)

    def untrained_count(self) -> int:
        return sum(1 for e in self.entries if e.times_trained == 0)

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
                score=float(d["score"]),
                insert_seq=int(d["insert_seq"]),
                # Legacy payloads (pre-freshness) carried is_new instead:
                # map seen -> trained once so the eviction-waste counter keeps
                # its meaning.
                times_trained=int(d.get("times_trained", 0 if d.get("is_new", True) else 1)),
            )
            for d in state.get("entries", [])
        ]
        pending_seqs = set(int(s) for s in state.get("pending_seqs", []))
        self.pending_fresh = [e for e in self.entries if e.insert_seq in pending_seqs]
