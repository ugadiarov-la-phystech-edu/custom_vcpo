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
"""Unit tests for the replay-buffer training mode:
- ReplayBuffer: score formula, add/evict/rescore, one-shot freshness, mini-batch
  composition (one-shot fresh priority, newest first + score-weighted fill),
  checkpoint round-trip incl. RNG state
- Rollouter insertion gate: frozen GRPO advantages match
  compute_grpo_outcome_advantage, group_version stamping, all-correct /
  all-wrong classification counters and the per-sync ratio metrics
- Trainer acquire loop: warm-up sequencing, steady-state watermark pause,
  sentinel termination, batch building from frozen statistics
- MessageQueue.get_available_samples non-blocking drain

Run: pytest recipe/fully_async_policy/unittest/test_replay_buffer_on_cpu.py
"""

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle
import torch
from omegaconf import OmegaConf

from recipe.fully_async_policy.detach_utils import RolloutSample, ValidateMetrics
from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from recipe.fully_async_policy.message_queue import MessageQueue as _MessageQueueActor
from recipe.fully_async_policy.replay_buffer import GroupEntry, ReplayBuffer, staleness_score
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _unwrap_ray_actor_class(actor_cls):
    """The classes are @ray.remote ActorClass wrappers; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap_ray_actor_class(_RollouterActor)
FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)
MessageQueue = _unwrap_ray_actor_class(_MessageQueueActor)


def _sample(group_version=0):
    """Minimal stand-in for a RolloutSample as the buffer sees it."""
    return SimpleNamespace(group_version=group_version)


def _make_buffer(tau=4.0, staleness_threshold=8, seed=1234):
    return ReplayBuffer(tau=tau, staleness_threshold=staleness_threshold, seed=seed)


# ---------------------------------------------------------------- score & add


def test_staleness_score_halves_every_tau():
    assert staleness_score(0, tau=4.0) == pytest.approx(1.0)
    assert staleness_score(4, tau=4.0) == pytest.approx(0.5)
    assert staleness_score(8, tau=4.0) == pytest.approx(0.25)
    assert staleness_score(2, tau=2.0) == pytest.approx(0.5)


def test_add_stamps_entry_fields():
    buf = _make_buffer(tau=4.0)
    e0 = buf.add(_sample(group_version=6), current_version=10)
    e1 = buf.add(_sample(group_version=10), current_version=10)
    assert e0.times_trained == 0 and e1.times_trained == 0
    assert (e0.insert_seq, e1.insert_seq) == (0, 1)
    assert e0.group_version == 6
    assert e0.score == pytest.approx(staleness_score(4, 4.0))
    assert e1.score == pytest.approx(1.0)
    assert buf.total_added == 2
    assert buf.size() == 2 and buf.untrained_count() == 2
    assert buf.pending_fresh == [e0, e1]


# ---------------------------------------------------------------- evict / rescore / mark_used


def test_evict_boundary_and_unseen_counting():
    buf = _make_buffer(staleness_threshold=2)
    at_threshold = buf.add(_sample(group_version=8), current_version=8)  # staleness 2 at v=10
    over_trained = buf.add(_sample(group_version=7), current_version=8)  # staleness 3 at v=10
    buf.add(_sample(group_version=6), current_version=8)  # over_untrained: staleness 4 at v=10
    fresh = buf.add(_sample(group_version=10), current_version=10)
    buf.mark_trained([over_trained])
    evicted, evicted_unseen = buf.evict(current_version=10)
    assert (evicted, evicted_unseen) == (2, 1)  # over_trained + over_untrained; only the latter unseen
    remaining = {e.insert_seq for e in buf.entries}
    assert remaining == {at_threshold.insert_seq, fresh.insert_seq}
    assert buf.evicted_total == 2 and buf.evicted_unseen_total == 1


def test_evict_purges_pending_fresh():
    buf = _make_buffer(staleness_threshold=2)
    stale_pending = buf.add(_sample(group_version=0), current_version=0)  # staleness 10 at v=10
    kept_pending = buf.add(_sample(group_version=10), current_version=10)
    buf.evict(current_version=10)
    assert buf.pending_fresh == [kept_pending]
    assert stale_pending not in buf.entries
    selected, info = buf.compose_minibatch(1, current_version=10)
    assert selected == [kept_pending] and info["n_new"] == 1


def test_recompute_scores_tracks_current_version():
    buf = _make_buffer(tau=4.0, staleness_threshold=100)
    entry = buf.add(_sample(group_version=0), current_version=0)
    assert entry.score == pytest.approx(1.0)
    buf.recompute_scores(current_version=4)
    assert entry.score == pytest.approx(0.5)
    buf.recompute_scores(current_version=8)
    assert entry.score == pytest.approx(0.25)


def test_mark_trained_increments_and_keeps_entry():
    buf = _make_buffer()
    entry = buf.add(_sample(), current_version=0)
    buf.mark_trained([entry])
    buf.mark_trained([entry])
    assert entry.times_trained == 2
    assert buf.size() == 1 and buf.untrained_count() == 0


# ---------------------------------------------------------------- composition


def test_compose_fresh_priority_newest_arrived_first():
    buf = _make_buffer()
    entries = [buf.add(_sample(group_version=i), current_version=5) for i in range(5)]
    selected, info = buf.compose_minibatch(3, current_version=5)
    assert [e.insert_seq for e in selected] == [4, 3, 2]
    assert info["n_new"] == 3 and info["n_replayed"] == 0
    assert info["staleness"] == [e.staleness(5) for e in entries[4:1:-1]]
    # freshness is one-shot: the pending list is cleared wholesale
    assert buf.pending_fresh == []


def test_compose_fresh_overflow_goes_to_replay_pool():
    buf = _make_buffer()
    entries = [buf.add(_sample(group_version=5), current_version=5) for _ in range(7)]
    selected, info = buf.compose_minibatch(5, current_version=5)
    # the 5 newest-arrived fresh groups fill the mini-batch...
    assert selected == entries[:1:-1] and info["n_new"] == 5
    # ...the overflow (the window's 2 earliest arrivals) stays buffered as
    # plain replay candidates
    assert buf.pending_fresh == []
    assert entries[0] in buf.entries and entries[1] in buf.entries
    # next composition with no new arrivals is pure replay
    selected2, info2 = buf.compose_minibatch(5, current_version=5)
    assert info2["n_new"] == 0 and info2["n_replayed"] == 5


def test_freshness_is_one_shot():
    buf = _make_buffer(tau=1.0, staleness_threshold=100)
    weak_overflow = buf.add(_sample(group_version=0), current_version=10)  # score 2^-10
    strong = buf.add(_sample(group_version=10), current_version=10)  # score 1
    selected, info = buf.compose_minibatch(1, current_version=10)
    # weak_overflow (the earlier arrival) is fresh overflow: unselected,
    # never trained, freshness spent
    assert selected == [strong] and info["n_new"] == 1
    picks = 0
    for trial in range(100):
        buf.rng = np.random.default_rng(trial)
        fresh = buf.add(_sample(group_version=10), current_version=10)
        selected, info = buf.compose_minibatch(2, current_version=10)
        # only this round's arrival counts as fresh...
        assert info["n_new"] == 1 and selected[0] is fresh
        picks += int(weak_overflow in selected)
        buf.entries.remove(fresh)
    # ...and the never-trained overflow holds no priority: it competes on its
    # ~2^-10 weight against strong's 1.0 (the old is_new rule would have
    # selected it deterministically every round)
    assert weak_overflow.times_trained == 0
    assert picks < 10


def test_compose_mixes_fresh_and_weighted_replay_without_duplicates():
    buf = _make_buffer()
    old_entries = [buf.add(_sample(group_version=0), current_version=0) for _ in range(4)]
    buf.compose_minibatch(4, current_version=0)  # consume their freshness
    buf.mark_trained(old_entries)
    new = [buf.add(_sample(group_version=3), current_version=3) for _ in range(2)]
    selected, info = buf.compose_minibatch(4, current_version=3)
    assert info["n_new"] == 2 and info["n_replayed"] == 2
    assert selected[:2] == new[::-1]  # fresh-first, newest-arrived first
    assert len({id(e) for e in selected}) == 4  # no duplicates
    assert all(e in old_entries for e in selected[2:])


def test_compose_pure_replay_when_no_new_groups():
    buf = _make_buffer()
    buf.add(_sample(), current_version=0)
    buf.add(_sample(), current_version=0)
    buf.add(_sample(), current_version=0)
    buf.compose_minibatch(3, current_version=0)  # consume freshness
    selected, info = buf.compose_minibatch(2, current_version=0)
    assert info["n_new"] == 0 and info["n_replayed"] == 2
    assert len({id(e) for e in selected}) == 2


def test_compose_is_seed_deterministic():
    def build():
        buf = _make_buffer(seed=42)
        used = [buf.add(_sample(group_version=i), current_version=6) for i in range(6)]
        buf.compose_minibatch(6, current_version=6)  # consume freshness
        buf.mark_trained(used)
        return buf

    sel_a, _ = build().compose_minibatch(3, current_version=6)
    sel_b, _ = build().compose_minibatch(3, current_version=6)
    assert [e.insert_seq for e in sel_a] == [e.insert_seq for e in sel_b]


def test_compose_sampling_prefers_high_scores():
    picks = {0: 0, 1: 0}
    for trial in range(200):
        buf = _make_buffer(tau=1.0, staleness_threshold=100, seed=trial)
        recent = buf.add(_sample(group_version=10), current_version=10)  # staleness 0, score 1
        stale = buf.add(_sample(group_version=0), current_version=10)  # staleness 10, score 2^-10
        buf.compose_minibatch(2, current_version=10)  # consume freshness
        buf.mark_trained([recent, stale])
        selected, _ = buf.compose_minibatch(1, current_version=10)
        picks[selected[0].insert_seq] += 1
    assert picks[0] > 190  # ~1000:1 odds per draw


def test_compose_uniform_fallback_when_scores_underflow():
    buf = _make_buffer(tau=1.0, staleness_threshold=10**6)
    used = [buf.add(_sample(group_version=0), current_version=0) for _ in range(3)]
    buf.compose_minibatch(3, current_version=0)  # consume freshness
    buf.mark_trained(used)
    buf.recompute_scores(current_version=5000)  # 2^-5000 underflows to 0.0
    assert all(e.score == 0.0 for e in buf.entries)
    selected, info = buf.compose_minibatch(2, current_version=5000)
    assert len(selected) == 2 and info["n_replayed"] == 2


def test_compose_raises_below_mini_size():
    buf = _make_buffer()
    buf.add(_sample(), current_version=0)
    with pytest.raises(ValueError, match="watermark"):
        buf.compose_minibatch(2, current_version=0)


# ---------------------------------------------------------------- checkpoint round-trip


def test_state_dict_roundtrip_restores_entries_counters_and_rng():
    buf = _make_buffer(seed=7)
    entries = [buf.add(_sample(group_version=i), current_version=8) for i in range(8)]
    buf.compose_minibatch(6, current_version=8)  # 6 fresh consumed, 2 overflow
    buf.mark_trained(entries[:6])
    buf.add(_sample(group_version=8), current_version=8)  # pending at save time
    buf.evict(current_version=8)
    state = ray.cloudpickle.loads(ray.cloudpickle.dumps(buf.state_dict()))

    restored = _make_buffer(seed=999)  # seed overwritten by the restored RNG state
    restored.load_state_dict(state)
    assert restored.size() == buf.size()
    assert restored.untrained_count() == buf.untrained_count()
    assert restored.total_added == buf.total_added
    assert restored.evicted_total == buf.evicted_total
    assert [e.insert_seq for e in restored.entries] == [e.insert_seq for e in buf.entries]
    assert [e.score for e in restored.entries] == [e.score for e in buf.entries]
    assert [e.times_trained for e in restored.entries] == [e.times_trained for e in buf.entries]
    assert [e.insert_seq for e in restored.pending_fresh] == [e.insert_seq for e in buf.pending_fresh]
    # identical RNG continuation and identical fresh set: the next draw matches
    sel_orig, info_orig = buf.compose_minibatch(4, current_version=8)
    sel_rest, info_rest = restored.compose_minibatch(4, current_version=8)
    assert [e.insert_seq for e in sel_orig] == [e.insert_seq for e in sel_rest]
    assert info_orig["n_new"] == info_rest["n_new"] == 1


def test_load_state_dict_tolerates_legacy_is_new_payload():
    # Pre-freshness checkpoints carried is_new per entry and no pending_seqs.
    state = {
        "next_insert_seq": 2,
        "total_added": 2,
        "evicted_total": 0,
        "evicted_unseen_total": 0,
        "rng_state": None,
        "entries": [
            {"sample": _sample(0), "group_version": 0, "is_new": True, "score": 1.0, "insert_seq": 0},
            {"sample": _sample(0), "group_version": 0, "is_new": False, "score": 1.0, "insert_seq": 1},
        ],
    }
    buf = _make_buffer()
    buf.load_state_dict(state)
    assert buf.pending_fresh == []  # legacy unseen backlog gets no fresh priority
    assert [e.times_trained for e in buf.entries] == [0, 1]  # waste counter keeps meaning
    selected, info = buf.compose_minibatch(2, current_version=0)
    assert info["n_new"] == 0


# ---------------------------------------------------------------- rollouter insertion gate


def _make_replay_rollouter(norm_adv_by_std_in_grpo=True):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.replay_mode = True
    r.norm_adv_by_std_in_grpo = norm_adv_by_std_in_grpo
    r.groups_completed_total = 0
    r.all_correct_groups_total = 0
    r.all_wrong_groups_total = 0
    r.groups_completed_window = 0
    r.all_correct_groups_window = 0
    r.all_wrong_groups_window = 0
    return r


def _group_sample(param_version_start):
    return SimpleNamespace(
        sample_id="sample_0_1",
        group_version=0,
        full_batch=SimpleNamespace(non_tensor_batch={"param_version_start": list(param_version_start)}),
    )


@pytest.mark.parametrize("norm", [True, False])
def test_frozen_advantages_match_compute_grpo_outcome_advantage(norm):
    rewards = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
    rollouter = _make_replay_rollouter(norm_adv_by_std_in_grpo=norm)
    rollouter._score_group = lambda rs: torch.tensor(rewards)
    sample = _group_sample(param_version_start=[4, 2, 3, 5, 2, 6])

    assert rollouter._prepare_replay_group(sample) is True
    non_tensor = sample.full_batch.non_tensor_batch
    assert sample.group_version == 2
    np.testing.assert_allclose(non_tensor["reward_scalar"], rewards, rtol=1e-6)

    # Reference: the fork's GRPO estimator on the same single-group batch.
    response_len = 3
    token_level_rewards = torch.zeros(len(rewards), response_len)
    token_level_rewards[:, -1] = torch.tensor(rewards)
    response_mask = torch.ones(len(rewards), response_len)
    index = np.array(["g"] * len(rewards))
    _, _, adv_scalars = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=norm
    )
    np.testing.assert_allclose(non_tensor["advantage_scalar"], adv_scalars, rtol=1e-5)


def test_insertion_gate_classifies_and_counts_groups():
    rollouter = _make_replay_rollouter()
    outcomes = []
    for rewards, kept in [
        ([1.0, 1.0, 1.0], False),  # all-correct -> dropped
        ([-1.0, -1.0, -1.0], False),  # all-wrong -> dropped
        ([1.0, -1.0, 1.0], True),  # mixed -> kept
        (None, False),  # unscorable -> dropped
    ]:
        rollouter._score_group = lambda rs, rewards=rewards: torch.tensor(rewards) if rewards is not None else None
        outcomes.append(rollouter._prepare_replay_group(_group_sample([0, 0, 0])))
    assert outcomes == [False, False, True, False]
    assert rollouter.groups_completed_total == 4
    assert rollouter.all_correct_groups_total == 1
    assert rollouter.all_wrong_groups_total == 1
    assert rollouter.groups_completed_window == 4


def test_zero_reward_all_equal_group_counts_as_all_wrong():
    # 0/1-style rewards: an all-zero group is all-wrong (score not > 0).
    rollouter = _make_replay_rollouter()
    rollouter._score_group = lambda rs: torch.tensor([0.0, 0.0, 0.0])
    assert rollouter._prepare_replay_group(_group_sample([0])) is False
    assert rollouter.all_wrong_groups_total == 1 and rollouter.all_correct_groups_total == 0


class _StubMessageQueueClient:
    def __init__(self):
        self.put_payloads = []

    async def get_queue_size(self):
        return 0

    async def put_validate(self, data):
        self.put_payloads.append(data)

    def last_validate_metrics(self) -> ValidateMetrics:
        return ray.cloudpickle.loads(self.put_payloads[-1])


def test_update_param_version_logs_and_resets_group_ratio_window():
    r = _make_replay_rollouter()
    r.lock = asyncio.Lock()
    r.current_param_version = 0
    r.active_tasks = set()
    r.cancel_queue = asyncio.Queue()
    r.message_queue_client = _StubMessageQueueClient()
    r.idle_start_time = None
    r.version_start_time = None
    r.val_reward_fn = object()
    r.config = OmegaConf.create({"rollout": {"test_freq": 1}})
    r.first_sample_time = time.time()
    r.cumulative_validation_time = 0.0
    r._validate = lambda: {"val-core/acc": 1.0}
    r.groups_completed_total = 10
    r.all_correct_groups_total = 4
    r.all_wrong_groups_total = 2
    r.groups_completed_window = 5
    r.all_correct_groups_window = 2
    r.all_wrong_groups_window = 1

    asyncio.run(r.update_param_version(1))

    timing_raw = r.message_queue_client.last_validate_metrics().timing_raw
    assert timing_raw["fully_async/groups/all_correct_ratio_total"] == pytest.approx(0.4)
    assert timing_raw["fully_async/groups/all_wrong_ratio_total"] == pytest.approx(0.2)
    assert timing_raw["fully_async/groups/all_correct_ratio"] == pytest.approx(0.4)
    assert timing_raw["fully_async/groups/all_wrong_ratio"] == pytest.approx(0.2)
    assert timing_raw["fully_async/groups/completed_total"] == 10
    # window reset, totals kept
    assert r.groups_completed_window == 0
    assert r.all_correct_groups_window == 0
    assert r.all_wrong_groups_window == 0
    assert r.groups_completed_total == 10


# ---------------------------------------------------------------- trainer acquire loop


class _QueueStub:
    """Fake MessageQueueClient: one-shot non-blocking drain + a scripted
    sequence for the blocking get_sample calls (empty script -> sentinel)."""

    def __init__(self, available=None, blocking=None):
        self.available = [ray.cloudpickle.dumps(s) if s is not None else None for s in (available or [])]
        self.blocking = [ray.cloudpickle.dumps(s) if s is not None else None for s in (blocking or [])]
        self.blocking_calls = 0

    def get_available_samples_sync(self):
        out, self.available = self.available, []
        return out

    def get_sample_sync(self):
        self.blocking_calls += 1
        if not self.blocking:
            return (None, 0)
        return (self.blocking.pop(0), 0)


def _make_replay_trainer(mini_size, requires_mini_batches, available=None, blocking=None):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.replay_buffer = ReplayBuffer(tau=4.0, staleness_threshold=100, seed=0)
    t.replay_updates_done = 0
    t.replay_requires_mini_batches = float(requires_mini_batches)
    t.required_samples = mini_size
    t.rollout_done = False
    t.current_param_version = 0
    t.message_queue_client = _QueueStub(available=available, blocking=blocking)
    # virtual-clock state touched by _open_virtual_step
    t.virtual_free_time = None
    t._step_virtual_start = None
    t._step_actual_start = None
    return t


def test_first_update_is_all_fresh_then_steady_state():
    s = [_sample(group_version=v) for v in range(5)]
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=2, available=[s[0]], blocking=[s[1], s[2], s[3]])

    # update 1: drain gives s0; watermark 4 forces 3 blocking pulls; the
    # all-pending buffer makes the first mini-batch all-fresh, newest first
    entries, info = trainer._acquire_replay_minibatch()
    assert trainer.message_queue_client.blocking_calls == 3
    assert [e.sample.group_version for e in entries] == [3, 2]
    assert info["n_new"] == 2
    trainer.replay_buffer.mark_trained(entries)

    # update 2: no new arrivals, buffer holds 4 >= watermark -> pure replay,
    # and the s0/s1 overflow of update 1 carries no fresh priority
    calls_before = trainer.message_queue_client.blocking_calls
    entries, info = trainer._acquire_replay_minibatch()
    assert trainer.message_queue_client.blocking_calls == calls_before
    assert info["n_new"] == 0 and info["n_replayed"] == 2


def test_steady_state_pauses_until_watermark():
    s = [_sample(group_version=v) for v in range(4)]
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=2, available=[s[0]], blocking=[s[1], s[2], s[3]])
    entries, info = trainer._acquire_replay_minibatch()
    # drain gave 1 group; watermark 4 forced 3 blocking pulls
    assert trainer.message_queue_client.blocking_calls == 3
    assert trainer.replay_buffer.size() == 4
    assert info["n_new"] == 2  # mini_size caps the fresh set; overflow to pool
    assert [e.sample.group_version for e in entries] == [3, 2]


def test_fractional_requires_mini_batches():
    s = [_sample(group_version=v) for v in range(6)]
    trainer = _make_replay_trainer(
        mini_size=2, requires_mini_batches=1.5, available=[s[0]], blocking=[s[1], s[2], s[3], s[4]]
    )
    # update 1: watermark 1.5*2=3 forces 2 blocking pulls; all-fresh compose
    entries, info = trainer._acquire_replay_minibatch()
    assert trainer.message_queue_client.blocking_calls == 2
    assert [e.sample.group_version for e in entries] == [2, 1]
    assert info["n_new"] == 2
    trainer.replay_buffer.mark_trained(entries)

    # buffer 3 >= watermark 3 -> no blocking pull
    calls_before = trainer.message_queue_client.blocking_calls
    entries, info = trainer._acquire_replay_minibatch()
    assert trainer.message_queue_client.blocking_calls == calls_before
    trainer.replay_buffer.mark_trained(entries)

    # drop below the fractional watermark: keep only 2 groups
    trainer.replay_buffer.entries = trainer.replay_buffer.entries[:2]
    trainer.replay_buffer.pending_fresh = []
    entries, info = trainer._acquire_replay_minibatch()
    # 2 < 3 forced exactly one blocking pull (s3), then size 3 >= 3 composes
    # with the pulled group as the only fresh one
    assert trainer.message_queue_client.blocking_calls == calls_before + 1
    assert trainer.replay_buffer.size() == 3
    assert info["n_new"] == 1


def test_post_update_maintenance_marks_trained_before_eviction():
    # Regression: _fit_replay used to evict BEFORE marking, so a group
    # trained on the very update that pushed it past the staleness threshold
    # was counted as evicted_unseen ("never trained on" waste). The trainer's
    # maintenance method must bump times_trained first: the just-trained
    # group counts as evicted-seen, only the genuinely untrained one as
    # unseen.
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    trainer.replay_buffer = ReplayBuffer(tau=4.0, staleness_threshold=8, seed=0)
    just_trained = trainer.replay_buffer.add(_sample(group_version=0), current_version=8)
    never_trained = trainer.replay_buffer.add(_sample(group_version=0), current_version=8)
    survivor = trainer.replay_buffer.add(_sample(group_version=5), current_version=8)

    # The update at version 8 trained on just_trained (staleness 8 <= 8);
    # at the post-update version 9 both version-0 groups cross the threshold.
    trainer._replay_post_update_maintenance([just_trained], new_version=9)

    assert just_trained.times_trained == 1 and never_trained.times_trained == 0
    assert trainer.replay_buffer.entries == [survivor]
    assert trainer.replay_buffer.evicted_total == 2
    assert trainer.replay_buffer.evicted_unseen_total == 1  # never_trained only
    # survivor rescored at the post-update version: staleness 4, tau 4 -> 0.5
    assert survivor.score == pytest.approx(0.5)


def test_virtual_step_gated_by_fresh_prefix_only():
    # Only the fresh entries' arrival stamps may gate the virtual step:
    # replayed groups were ready long ago, whatever their enqueue stamps say.
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    for _ in range(2):
        old_sample = _sample(group_version=0)
        old_sample.enqueue_time = 1e9  # would dominate if wrongly included
        old_sample.validation_pause_before = 0.0
        trainer.replay_buffer.add(old_sample, current_version=0)
    trainer.replay_buffer.compose_minibatch(2, current_version=0)  # consume freshness
    fresh_sample = _sample(group_version=0)
    fresh_sample.enqueue_time = 100.0
    fresh_sample.validation_pause_before = 0.0
    trainer.message_queue_client = _QueueStub(available=[fresh_sample])

    entries, info = trainer._acquire_replay_minibatch()
    assert info["n_new"] == 1 and info["n_replayed"] == 1
    assert entries[0].sample.enqueue_time == 100.0  # fresh-first ordering
    assert trainer._step_virtual_start == pytest.approx(100.0)


def test_acquire_terminates_on_sentinel_when_buffer_insufficient():
    # warm-up: sentinel in the drain, nothing else -> stop
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=2, available=[None])
    assert trainer._acquire_replay_minibatch() == (None, None)
    assert trainer.rollout_done

    # steady state: one buffered group, sentinel on the blocking path -> stop
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1, available=[_sample()])
    trainer.replay_updates_done = 3
    assert trainer._acquire_replay_minibatch() == (None, None)
    assert trainer.rollout_done
    assert trainer.replay_buffer.size() == 1


def test_drain_adds_groups_and_flags_sentinel():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1, available=[_sample(1), None, _sample(2)])
    added = trainer._drain_queue_into_buffer()
    assert added == 2
    assert trainer.rollout_done
    assert trainer.replay_buffer.size() == 2


def test_add_replay_metrics_reports_buffer_and_minibatch_stats():
    trainer = _make_replay_trainer(mini_size=4, requires_mini_batches=1)
    buf = trainer.replay_buffer
    for v in (10, 10, 8, 6):
        buf.add(_sample(group_version=v), current_version=10)
    metrics = {}
    info = {"n_new": 3, "n_replayed": 1, "staleness": [0, 0, 2, 4]}
    trainer._add_replay_metrics(metrics, info, new_version=10)
    assert metrics["replay/buffer_size"] == 4
    assert metrics["replay/minibatch_new"] == 3
    assert metrics["replay/minibatch_replayed"] == 1
    assert metrics["replay/minibatch_new_ratio"] == pytest.approx(0.75)
    assert metrics["replay/minibatch_staleness_mean"] == pytest.approx(1.5)
    assert metrics["replay/minibatch_staleness_max"] == 4.0
    assert metrics["replay/buffer_max_staleness"] == 4.0
    assert metrics["replay/minibatch_staleness_hist"] == [0, 0, 2, 4]
    assert metrics["replay/buffer_staleness_hist"] == [0, 0, 2, 4]
    assert "replay/ess_scaled_lr" not in metrics  # standard update path: no staleness/ess entries


def test_add_replay_metrics_surfaces_mean_ess_scaled_lr():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    buf = trainer.replay_buffer
    for v in (5, 5):
        buf.add(_sample(group_version=v), current_version=5)
    # per-traj actor output: structured staleness/ess entries carry the
    # effective (possibly ESS-scaled) lr per mini-batch
    metrics = {
        "staleness/ess": [
            {"minibatch_idx": 0, "ess_scaled_lr": 1e-6},
            {"minibatch_idx": 1, "ess_scaled_lr": 2e-6},
            {"minibatch_idx": 2, "ess_scaled_lr": None},  # ignored
        ]
    }
    info = {"n_new": 2, "n_replayed": 0, "staleness": [0, 0]}
    trainer._add_replay_metrics(metrics, info, new_version=5)
    assert metrics["replay/ess_scaled_lr"] == pytest.approx(1.5e-6)


# ---------------------------------------------------------------- batch building


def _rollout_sample_with_batch(uid, rewards, advantages, response_mask_rows, param_version=3):
    n = len(rewards)
    response_len = len(response_mask_rows[0])
    response_mask = torch.tensor(response_mask_rows, dtype=torch.long)
    seq_len = response_len + 2  # fake 2-token prompt
    attention_mask = torch.cat([torch.ones(n, 2, dtype=torch.long), response_mask], dim=-1)
    full_batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "attention_mask": attention_mask,
            "responses": torch.zeros(n, response_len, dtype=torch.long),
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "position_ids": torch.zeros(n, seq_len, dtype=torch.long),
        },
        non_tensors={
            "uid": np.array([uid] * n, dtype=object),
            "reward_scalar": np.asarray(rewards, dtype=np.float32),
            "advantage_scalar": np.asarray(advantages, dtype=np.float32),
            "param_version_start": np.array([param_version] * n),
            "param_version_end": np.array([param_version] * n),
            "processing_times": np.array([0.1] * n),
            "tool_calls_times": np.array([0.0] * n),
        },
    )
    return RolloutSample(
        full_batch=full_batch,
        agent_loop_output_list=[],
        sample_id=f"sample_0_{uid}",
        epoch=0,
        processing_times=[],
        tool_calls=[],
        param_version=param_version,
        param_version_start=[param_version],
        param_version_end=[param_version],
        rollout_status={"count/current_param_version": param_version},
        group_version=param_version,
    )


def _minimal_trainer_config():
    return OmegaConf.create(
        {
            "trainer": {"balance_batch": False},
            "algorithm": {"rollout_correction": {"rollout_is": "token", "rollout_is_threshold": 2.0}},
            "actor_rollout_ref": {
                "rollout": {"n": 2, "temperature": 1.0, "multi_turn": {"enable": False}},
                "actor": {"grad_baselining": {"enable": False}, "update_policy_per_traj": False},
            },
        }
    )


def test_build_replay_batch_uses_frozen_statistics():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    trainer.config = _minimal_trainer_config()
    trainer.tokenizer = None
    rs_a = _rollout_sample_with_batch(
        "uid_a", rewards=[1.0, -1.0], advantages=[1.0, -1.0], response_mask_rows=[[1, 1, 0], [1, 1, 1]]
    )
    rs_b = _rollout_sample_with_batch(
        "uid_b", rewards=[-1.0, 1.0], advantages=[-0.5, 0.5], response_mask_rows=[[1, 0, 0], [1, 1, 0]]
    )
    entries = [
        GroupEntry(sample=rs_a, group_version=3, score=1.0, insert_seq=0),
        GroupEntry(sample=rs_b, group_version=3, score=0.5, insert_seq=1, times_trained=1),
    ]
    batch = trainer._build_replay_batch(entries)

    # advantages broadcast over the response mask
    expected_adv = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    torch.testing.assert_close(batch.batch["advantages"], expected_adv)
    torch.testing.assert_close(batch.batch["returns"], expected_adv)

    # reward scalar sits on the last valid response token
    expected_scores = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(batch.batch["token_level_scores"], expected_scores)
    torch.testing.assert_close(batch.batch["token_level_rewards"], expected_scores)

    meta = batch.meta_info
    assert meta["skip_recompute_old_log_prob"] is True
    assert meta["temperature"] == 1.0
    assert meta["n_resp_per_rollout"] == 2
    assert meta["multi_turn"] is False
    assert "dp_group_key" not in meta
    assert len(batch.non_tensor_batch["traj_uid"]) == 4
    assert len(meta["global_token_num"]) == 4


def test_build_replay_batch_pins_dp_groups_for_opob():
    trainer = _make_replay_trainer(mini_size=1, requires_mini_batches=1)
    config = _minimal_trainer_config()
    config.actor_rollout_ref.actor.grad_baselining.enable = True
    config.actor_rollout_ref.actor.update_policy_per_traj = True
    trainer.config = config
    trainer.tokenizer = None
    rs = _rollout_sample_with_batch(
        "uid_a", rewards=[1.0, -1.0], advantages=[1.0, -1.0], response_mask_rows=[[1, 1, 0], [1, 1, 1]]
    )
    entries = [GroupEntry(sample=rs, group_version=3, score=1.0, insert_seq=0)]
    batch = trainer._build_replay_batch(entries)
    assert batch.meta_info["dp_group_key"] == "uid"
    assert batch.meta_info["dp_group_size"] == 2


# ---------------------------------------------------------------- ESS metrics & checkpoint state


def test_build_replay_batch_never_stamps_ess_base_override():
    # The min-ESS brake needs no measured reference: the removed auto-base
    # override key must not reappear in the batch meta.
    trainer = _make_replay_trainer(mini_size=1, requires_mini_batches=1)
    trainer.config = _minimal_trainer_config()
    trainer.tokenizer = None
    rs = _rollout_sample_with_batch(
        "uid_a", rewards=[1.0, -1.0], advantages=[1.0, -1.0], response_mask_rows=[[1, 1, 0], [1, 1, 1]]
    )
    entries = [GroupEntry(sample=rs, group_version=3, score=1.0, insert_seq=0)]
    batch = trainer._build_replay_batch(entries)
    assert "ess_base_override" not in batch.meta_info


def test_replay_checkpoint_state_roundtrips():
    saver = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    saver.replay_buffer.add(_sample(group_version=3), current_version=4)
    saver.replay_updates_done = 7
    state = ray.cloudpickle.loads(ray.cloudpickle.dumps(saver._replay_checkpoint_state()))
    assert "ess_base" not in state  # removed auto-base field is no longer persisted

    restored = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    restored._load_replay_checkpoint_state(state)
    assert restored.replay_updates_done == 7
    assert restored.replay_buffer.size() == 1


def test_load_replay_checkpoint_state_tolerates_old_ess_base_key():
    # Checkpoints from the removed auto-base mechanism carry an "ess_base"
    # key; loading must ignore it without failing.
    saver = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    saver.replay_buffer.add(_sample(), current_version=0)
    old_state = {"buffer": saver.replay_buffer.state_dict(), "updates_done": 3, "ess_base": 0.66}

    restored = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    restored._load_replay_checkpoint_state(old_state)
    assert restored.replay_updates_done == 3
    assert restored.replay_buffer.size() == 1
    assert not hasattr(restored, "replay_ess_base")


def test_save_replay_state_gated_by_flag(tmp_path):
    # replay_buffer.save_state=False skips the 7-12 GB replay_buffer.pt
    # (resume state — dead weight for never-resumed runs)
    import os

    for flag, expect_file in ((True, True), (False, False)):
        trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
        trainer.replay_enable = True
        trainer.replay_save_state = flag
        trainer.replay_buffer.add(_sample(group_version=0), current_version=0)
        folder = tmp_path / f"global_step_{int(flag)}"
        folder.mkdir()
        trainer._save_replay_state(str(folder))
        assert os.path.exists(folder / "replay_buffer.pt") is expect_file


def test_replay_save_state_defaults_to_true_in_recipe_configs():
    # The yaml default keeps old behavior (persist replay_buffer.pt); the
    # checkpoint-off scripts opt out explicitly with save_state=False.
    import os

    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    for name in ("fully_async_ppo_trainer.yaml", "fully_async_ppo_megatron_trainer.yaml"):
        cfg = OmegaConf.load(os.path.join(cfg_dir, name))
        assert cfg.async_training.replay_buffer.save_state is True, name


def test_add_replay_metrics_reports_scaled_lr_not_base():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    for _ in range(2):
        trainer.replay_buffer.add(_sample(group_version=0), current_version=0)
    metrics = {
        "staleness/ess": [
            {"ess_scaled_lr": 1e-6},
            {"ess_scaled_lr": 2e-6},
            {"ess_scaled_lr": None},  # ignored
        ]
    }
    trainer._add_replay_metrics(metrics, {"n_new": 2, "n_replayed": 0, "staleness": [0, 0]}, new_version=0)
    assert metrics["replay/ess_scaled_lr"] == pytest.approx(1.5e-6)
    assert "replay/ess_base" not in metrics


def test_process_structured_metrics_emits_ess_scalars_without_base():
    from recipe.fully_async_policy.detach_utils import process_structured_metrics

    payload = process_structured_metrics(
        {
            "staleness/ess": [
                {"minibatch_ess_ratio": 0.5, "ess_scaled_lr": 1e-6},
                {"minibatch_ess_ratio": 0.7, "ess_scaled_lr": 2e-6},
                {"minibatch_ess_ratio": 0.6},
            ]
        },
        allow_media=False,
    )
    assert payload["staleness/ess_ratio"] == pytest.approx(0.6)
    assert payload["actor/ess_scaled_lr"] == pytest.approx(1.5e-6)
    assert "staleness/base_ess_ratio" not in payload


# ---------------------------------------------------------------- tensorboard histograms


class _FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.histograms = []

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))

    def add_histogram(self, key, values, step):
        self.histograms.append((key, list(np.asarray(values)), step))


def test_tensorboard_adapter_routes_lists_to_histograms():
    from verl.utils.tracking import _TensorboardAdapter

    adapter = object.__new__(_TensorboardAdapter)
    adapter.writer = _FakeSummaryWriter()
    adapter.log(
        data={
            "replay/buffer_size": 42,
            "replay/minibatch_staleness_hist": [0, 1, 1, 3],
            "replay/buffer_staleness_hist": np.array([2, 2, 5]),
            "replay/empty_hist": [],
        },
        step=7,
    )
    assert ("replay/buffer_size", 42, 7) in adapter.writer.scalars
    assert ("replay/minibatch_staleness_hist", [0, 1, 1, 3], 7) in adapter.writer.histograms
    assert ("replay/buffer_staleness_hist", [2, 2, 5], 7) in adapter.writer.histograms
    # empty lists are dropped, never sent to add_scalar
    logged_keys = {k for k, _, _ in adapter.writer.scalars} | {k for k, _, _ in adapter.writer.histograms}
    assert "replay/empty_hist" not in logged_keys


class _BackendRecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, data=None, step=None, backend=None):
        self.calls.append((data, step, backend))


def _make_hist_trainer(logger_backends, structured):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.config = OmegaConf.create({"trainer": {"logger": logger_backends}})
    t.structured_metrics = structured
    t.logger = _BackendRecordingLogger()
    return t


def test_tb_staleness_histograms_logged_to_tensorboard_backend_only():
    structured = {
        "replay/minibatch_staleness_hist": [0, 1, 2],
        "replay/buffer_staleness_hist": [1, 1, 4, 4],
        "staleness/ess": [{"minibatch_ess": 1.0}],  # unrelated structured key ignored
    }
    trainer = _make_hist_trainer(["console", "tensorboard"], structured)
    trainer._log_tb_staleness_histograms(step=9)
    assert len(trainer.logger.calls) == 1
    data, step, backend = trainer.logger.calls[0]
    assert step == 9 and backend == ["tensorboard"]
    assert data == {
        "replay/minibatch_staleness_hist": [0.0, 1.0, 2.0],
        "replay/buffer_staleness_hist": [1.0, 1.0, 4.0, 4.0],
    }


def test_tb_staleness_histograms_noop_without_tensorboard_or_data():
    trainer = _make_hist_trainer(["console"], {"replay/minibatch_staleness_hist": [0, 1]})
    trainer._log_tb_staleness_histograms(step=3)
    assert trainer.logger.calls == []

    trainer = _make_hist_trainer(["tensorboard"], {})
    trainer._log_tb_staleness_histograms(step=3)
    assert trainer.logger.calls == []


# ---------------------------------------------------------------- message queue drain


def test_message_queue_get_available_samples_drains_in_order():
    config = OmegaConf.create({"async_training": {"staleness_threshold": 1}})
    mq = MessageQueue(config, max_queue_size=16)

    async def run():
        await mq.put_sample("a", 0)
        await mq.put_sample("b", 0)
        await mq.put_sample(None, 0)  # termination sentinel passes through
        drained = await mq.get_available_samples()
        size_after = await mq.get_queue_size()
        empty = await mq.get_available_samples()
        return drained, size_after, empty

    drained, size_after, empty = asyncio.run(run())
    assert drained == ["a", "b", None]
    assert size_after == 0
    assert empty == []
