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
- ReplayBuffer: score formula, add/evict/rescore, is_new lifecycle, mini-batch
  composition (all-unseen-oldest-first priority + score-weighted fill),
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
import math
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
    assert e0.is_new and e1.is_new
    assert (e0.insert_seq, e1.insert_seq) == (0, 1)
    assert e0.group_version == 6
    assert e0.score == pytest.approx(staleness_score(4, 4.0))
    assert e1.score == pytest.approx(1.0)
    assert buf.total_added == 2
    assert buf.size() == 2 and buf.new_count() == 2


# ---------------------------------------------------------------- evict / rescore / mark_used


def test_evict_boundary_and_unseen_counting():
    buf = _make_buffer(staleness_threshold=2)
    at_threshold = buf.add(_sample(group_version=8), current_version=8)  # staleness 2 at v=10
    over_used = buf.add(_sample(group_version=7), current_version=8)  # staleness 3 at v=10
    buf.add(_sample(group_version=6), current_version=8)  # over_new: staleness 4 at v=10
    fresh = buf.add(_sample(group_version=10), current_version=10)
    buf.mark_used([over_used])
    evicted, evicted_unseen = buf.evict(current_version=10)
    assert (evicted, evicted_unseen) == (2, 1)  # over_used + over_new; only over_new unseen
    remaining = {e.insert_seq for e in buf.entries}
    assert remaining == {at_threshold.insert_seq, fresh.insert_seq}
    assert buf.evicted_total == 2 and buf.evicted_unseen_total == 1


def test_recompute_scores_tracks_current_version():
    buf = _make_buffer(tau=4.0, staleness_threshold=100)
    entry = buf.add(_sample(group_version=0), current_version=0)
    assert entry.score == pytest.approx(1.0)
    buf.recompute_scores(current_version=4)
    assert entry.score == pytest.approx(0.5)
    buf.recompute_scores(current_version=8)
    assert entry.score == pytest.approx(0.25)


def test_mark_used_flips_is_new_but_keeps_entry():
    buf = _make_buffer()
    entry = buf.add(_sample(), current_version=0)
    buf.mark_used([entry])
    assert not entry.is_new
    assert buf.size() == 1 and buf.new_count() == 0


# ---------------------------------------------------------------- composition


def test_compose_all_new_priority_oldest_inserted_first():
    buf = _make_buffer()
    entries = [buf.add(_sample(group_version=i), current_version=5) for i in range(5)]
    selected, info = buf.compose_minibatch(3, current_version=5)
    assert [e.insert_seq for e in selected] == [0, 1, 2]
    assert info["n_new"] == 3 and info["n_replayed"] == 0
    assert info["staleness"] == [e.staleness(5) for e in entries[:3]]


def test_compose_fills_with_used_groups_without_duplicates():
    buf = _make_buffer()
    used = [buf.add(_sample(group_version=0), current_version=0) for _ in range(4)]
    buf.mark_used(used)
    new = [buf.add(_sample(group_version=3), current_version=3) for _ in range(2)]
    selected, info = buf.compose_minibatch(4, current_version=3)
    assert info["n_new"] == 2 and info["n_replayed"] == 2
    assert selected[:2] == new
    assert len({id(e) for e in selected}) == 4  # no duplicates
    assert all(not e.is_new for e in selected[2:])


def test_compose_pure_replay_when_no_new_groups():
    buf = _make_buffer()
    used = [buf.add(_sample(), current_version=0) for _ in range(3)]
    buf.mark_used(used)
    selected, info = buf.compose_minibatch(2, current_version=0)
    assert info["n_new"] == 0 and info["n_replayed"] == 2
    assert len({id(e) for e in selected}) == 2


def test_compose_is_seed_deterministic():
    def build():
        buf = _make_buffer(seed=42)
        used = [buf.add(_sample(group_version=i), current_version=6) for i in range(6)]
        buf.mark_used(used)
        return buf

    sel_a, _ = build().compose_minibatch(3, current_version=6)
    sel_b, _ = build().compose_minibatch(3, current_version=6)
    assert [e.insert_seq for e in sel_a] == [e.insert_seq for e in sel_b]


def test_compose_sampling_prefers_high_scores():
    picks = {0: 0, 1: 0}
    for trial in range(200):
        buf = _make_buffer(tau=1.0, staleness_threshold=100, seed=trial)
        fresh = buf.add(_sample(group_version=10), current_version=10)  # staleness 0, score 1
        stale = buf.add(_sample(group_version=0), current_version=10)  # staleness 10, score 2^-10
        buf.mark_used([fresh, stale])
        selected, _ = buf.compose_minibatch(1, current_version=10)
        picks[selected[0].insert_seq] += 1
    assert picks[0] > 190  # ~1000:1 odds per draw


def test_compose_uniform_fallback_when_scores_underflow():
    buf = _make_buffer(tau=1.0, staleness_threshold=10**6)
    used = [buf.add(_sample(group_version=0), current_version=0) for _ in range(3)]
    buf.mark_used(used)
    buf.recompute_scores(current_version=5000)  # 2^-5000 underflows to 0.0
    assert all(e.score == 0.0 for e in buf.entries)
    selected, info = buf.compose_minibatch(2, current_version=5000)
    assert len(selected) == 2 and info["n_replayed"] == 2


def test_compose_raises_below_mini_size():
    buf = _make_buffer()
    buf.add(_sample(), current_version=0)
    with pytest.raises(ValueError, match="watermark"):
        buf.compose_minibatch(2, current_version=0)


def test_take_oldest_new_order_and_underflow():
    buf = _make_buffer()
    first = buf.add(_sample(), current_version=0)
    used = buf.add(_sample(), current_version=0)
    buf.mark_used([used])
    second = buf.add(_sample(), current_version=0)
    assert buf.take_oldest_new(2) == [first, second]
    with pytest.raises(ValueError):
        buf.take_oldest_new(3)


# ---------------------------------------------------------------- checkpoint round-trip


def test_state_dict_roundtrip_restores_entries_counters_and_rng():
    buf = _make_buffer(seed=7)
    used = [buf.add(_sample(group_version=i), current_version=8) for i in range(8)]
    buf.mark_used(used[:6])
    buf.evict(current_version=8)
    state = ray.cloudpickle.loads(ray.cloudpickle.dumps(buf.state_dict()))

    restored = _make_buffer(seed=999)  # seed overwritten by the restored RNG state
    restored.load_state_dict(state)
    assert restored.size() == buf.size()
    assert restored.new_count() == buf.new_count()
    assert restored.total_added == buf.total_added
    assert restored.evicted_total == buf.evicted_total
    assert [e.insert_seq for e in restored.entries] == [e.insert_seq for e in buf.entries]
    assert [e.score for e in restored.entries] == [e.score for e in buf.entries]
    # identical RNG continuation: the next weighted draw matches
    sel_orig, _ = buf.compose_minibatch(4, current_version=8)
    sel_rest, _ = restored.compose_minibatch(4, current_version=8)
    assert [e.insert_seq for e in sel_orig] == [e.insert_seq for e in sel_rest]


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
        rollouter._score_group = lambda rs, rewards=rewards: (
            torch.tensor(rewards) if rewards is not None else None
        )
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
        self.available = [
            ray.cloudpickle.dumps(s) if s is not None else None for s in (available or [])
        ]
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


def _make_replay_trainer(
    mini_size, requires_mini_batches, available=None, blocking=None, ess_auto_base=False, ess_use_clipped=False
):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.replay_buffer = ReplayBuffer(tau=4.0, staleness_threshold=100, seed=0)
    t.replay_updates_done = 0
    t.replay_requires_mini_batches = float(requires_mini_batches)
    t.replay_warmup_updates = math.ceil(requires_mini_batches)
    t.required_samples = mini_size
    t.rollout_done = False
    t.current_param_version = 0
    t.message_queue_client = _QueueStub(available=available, blocking=blocking)
    # auto-calibrated ESS base state
    t.replay_ess_auto_base = ess_auto_base
    t.replay_ess_use_clipped = ess_use_clipped
    t.replay_ess_base = None
    # virtual-clock state touched by _open_virtual_step
    t.virtual_free_time = None
    t._step_virtual_start = None
    t._step_actual_start = None
    return t


def test_warmup_consumes_fresh_chunks_oldest_first_then_steady_state():
    s = [_sample(group_version=v) for v in range(5)]
    trainer = _make_replay_trainer(
        mini_size=2, requires_mini_batches=2, available=[s[0]], blocking=[s[1], s[2], s[3]]
    )

    # warm-up update 1: drain gives s0, one blocking wait pulls s1
    entries, info = trainer._acquire_replay_minibatch()
    assert [e.sample.group_version for e in entries] == [0, 1]
    assert info["n_new"] == 2
    trainer.replay_buffer.mark_used(entries)
    trainer.replay_updates_done = 1

    # warm-up update 2: needs a fresh unseen mini-batch (s2, s3)
    entries, info = trainer._acquire_replay_minibatch()
    assert [e.sample.group_version for e in entries] == [2, 3]
    assert info["n_new"] == 2
    trainer.replay_buffer.mark_used(entries)
    trainer.replay_updates_done = 2

    # steady state: buffer holds 4 >= watermark 4 -> composes without blocking
    calls_before = trainer.message_queue_client.blocking_calls
    entries, info = trainer._acquire_replay_minibatch()
    assert trainer.message_queue_client.blocking_calls == calls_before
    assert info["n_new"] == 0 and info["n_replayed"] == 2


def test_steady_state_pauses_until_watermark():
    s = [_sample(group_version=v) for v in range(4)]
    trainer = _make_replay_trainer(
        mini_size=2, requires_mini_batches=2, available=[s[0]], blocking=[s[1], s[2], s[3]]
    )
    trainer.replay_updates_done = 5  # past warm-up
    entries, info = trainer._acquire_replay_minibatch()
    # drain gave 1 group; watermark 4 forced 3 blocking pulls
    assert trainer.message_queue_client.blocking_calls == 3
    assert trainer.replay_buffer.size() == 4
    assert info["n_new"] == 2  # mini_size caps the all-new priority set
    assert [e.sample.group_version for e in entries] == [0, 1]


def test_fractional_requires_mini_batches():
    s = [_sample(group_version=v) for v in range(6)]
    trainer = _make_replay_trainer(
        mini_size=2, requires_mini_batches=1.5, available=[s[0]], blocking=[s[1], s[2], s[3], s[4]]
    )
    # warm-up runs ceil(1.5) = 2 fresh-chunk updates
    assert trainer.replay_warmup_updates == 2
    for expected_versions in ([0, 1], [2, 3]):
        entries, info = trainer._acquire_replay_minibatch()
        assert [e.sample.group_version for e in entries] == expected_versions
        assert info["n_new"] == 2
        trainer.replay_buffer.mark_used(entries)
        trainer.replay_updates_done += 1
    # steady state: buffer holds 4 >= watermark 1.5*2=3 -> no blocking pull
    calls_before = trainer.message_queue_client.blocking_calls
    entries, info = trainer._acquire_replay_minibatch()
    assert trainer.message_queue_client.blocking_calls == calls_before
    trainer.replay_buffer.mark_used(entries)

    # drop below the fractional watermark: evict everything but 2 groups
    trainer.replay_buffer.entries = trainer.replay_buffer.entries[:2]
    entries, info = trainer._acquire_replay_minibatch()
    # 2 < 3 forced exactly one blocking pull (s4), then size 3 >= 3 composes
    assert trainer.message_queue_client.blocking_calls == calls_before + 1
    assert trainer.replay_buffer.size() == 3


def test_post_update_maintenance_marks_used_before_eviction():
    # Regression: _fit_replay used to evict BEFORE mark_used, so a group
    # trained on the very update that pushed it past the staleness threshold
    # was counted as evicted_unseen ("never trained on" waste). The trainer's
    # maintenance method must retire is_new first: the just-trained group
    # counts as evicted-seen, only the genuinely untrained one as unseen.
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    trainer.replay_buffer = ReplayBuffer(tau=4.0, staleness_threshold=8, seed=0)
    just_trained = trainer.replay_buffer.add(_sample(group_version=0), current_version=8)
    never_trained = trainer.replay_buffer.add(_sample(group_version=0), current_version=8)
    survivor = trainer.replay_buffer.add(_sample(group_version=5), current_version=8)

    # The update at version 8 trained on just_trained (staleness 8 <= 8);
    # at the post-update version 9 both version-0 groups cross the threshold.
    trainer._replay_post_update_maintenance([just_trained], new_version=9)

    assert not just_trained.is_new
    assert trainer.replay_buffer.entries == [survivor]
    assert trainer.replay_buffer.evicted_total == 2
    assert trainer.replay_buffer.evicted_unseen_total == 1  # never_trained only
    # survivor rescored at the post-update version: staleness 4, tau 4 -> 0.5
    assert survivor.score == pytest.approx(0.5)


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
    trainer = _make_replay_trainer(
        mini_size=2, requires_mini_batches=1, available=[_sample(1), None, _sample(2)]
    )
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
        GroupEntry(sample=rs_a, group_version=3, is_new=True, score=1.0, insert_seq=0),
        GroupEntry(sample=rs_b, group_version=3, is_new=False, score=0.5, insert_seq=1),
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
    entries = [GroupEntry(sample=rs, group_version=3, is_new=True, score=1.0, insert_seq=0)]
    batch = trainer._build_replay_batch(entries)
    assert batch.meta_info["dp_group_key"] == "uid"
    assert batch.meta_info["dp_group_size"] == 2


# ---------------------------------------------------------------- auto base_ess_ratio


def test_resolve_ess_base_explicit_config_wins():
    from verl.workers.actor.megatron_actor import resolve_ess_base

    assert resolve_ess_base(0.8, 0.5) == 0.8
    assert resolve_ess_base(None, 0.5) == 0.5
    assert resolve_ess_base(None, None) is None
    assert resolve_ess_base(0.0, 0.5) == 0.0  # explicit 0.0 is still explicit


def test_capture_ess_base_means_unclipped_entries_once():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    metrics = {
        "staleness/ess": [
            {"minibatch_ess_ratio": 0.8, "minibatch_ess_ratio_clipped": 0.9},
            {"minibatch_ess_ratio": 0.6, "minibatch_ess_ratio_clipped": 0.7},
            {"minibatch_ess_ratio": None},  # ignored
            "not-a-dict",  # ignored
        ]
    }
    trainer._capture_ess_base(metrics)
    assert trainer.replay_ess_base == pytest.approx(0.7)  # mean of unclipped values
    # capture is one-shot: later (different) entries must not overwrite it
    trainer._capture_ess_base({"staleness/ess": [{"minibatch_ess_ratio": 0.1}]})
    assert trainer.replay_ess_base == pytest.approx(0.7)


def test_capture_ess_base_uses_clipped_field_when_configured():
    trainer = _make_replay_trainer(
        mini_size=2, requires_mini_batches=1, ess_auto_base=True, ess_use_clipped=True
    )
    metrics = {"staleness/ess": [{"minibatch_ess_ratio": 0.6, "minibatch_ess_ratio_clipped": 0.9}]}
    trainer._capture_ess_base(metrics)
    assert trainer.replay_ess_base == pytest.approx(0.9)


def test_capture_ess_base_stays_none_without_entries():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    trainer._capture_ess_base({})
    trainer._capture_ess_base({"staleness/ess": []})
    trainer._capture_ess_base({"staleness/ess": [{"other_key": 1.0}]})
    assert trainer.replay_ess_base is None


def test_build_replay_batch_stamps_ess_base_override_in_auto_mode():
    def build(auto, base):
        trainer = _make_replay_trainer(mini_size=1, requires_mini_batches=1, ess_auto_base=auto)
        trainer.replay_ess_base = base
        trainer.config = _minimal_trainer_config()
        trainer.tokenizer = None
        rs = _rollout_sample_with_batch(
            "uid_a", rewards=[1.0, -1.0], advantages=[1.0, -1.0], response_mask_rows=[[1, 1, 0], [1, 1, 1]]
        )
        entries = [GroupEntry(sample=rs, group_version=3, is_new=True, score=1.0, insert_seq=0)]
        return trainer._build_replay_batch(entries)

    # auto mode, pre-capture: override present but None (actor skips scaling)
    batch = build(auto=True, base=None)
    assert "ess_base_override" in batch.meta_info and batch.meta_info["ess_base_override"] is None
    # auto mode, post-capture: captured value stamped
    batch = build(auto=True, base=0.73)
    assert batch.meta_info["ess_base_override"] == pytest.approx(0.73)
    # explicit-base (non-auto) runs never stamp the key
    batch = build(auto=False, base=None)
    assert "ess_base_override" not in batch.meta_info


def test_replay_checkpoint_state_roundtrips_ess_base():
    saver = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    saver.replay_buffer.add(_sample(group_version=3), current_version=4)
    saver.replay_updates_done = 7
    saver.replay_ess_base = 0.66
    state = ray.cloudpickle.loads(ray.cloudpickle.dumps(saver._replay_checkpoint_state()))

    restored = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    restored._load_replay_checkpoint_state(state)
    assert restored.replay_ess_base == pytest.approx(0.66)
    assert restored.replay_updates_done == 7
    assert restored.replay_buffer.size() == 1


def test_load_replay_checkpoint_state_tolerates_pre_feature_checkpoints():
    # Old-format state: no ess_base / updates_done keys beyond the buffer.
    saver = _make_replay_trainer(mini_size=2, requires_mini_batches=1)
    saver.replay_buffer.add(_sample(), current_version=0)
    old_state = {"buffer": saver.replay_buffer.state_dict(), "updates_done": 3}

    restored = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    restored.replay_ess_base = 0.9  # must be overwritten by the (missing) stored value
    restored._load_replay_checkpoint_state(old_state)
    assert restored.replay_ess_base is None  # falls back to first-post-resume capture (with warning)
    assert restored.replay_updates_done == 3


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


def test_add_replay_metrics_reports_ess_base_when_captured():
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    trainer.replay_buffer.add(_sample(group_version=0), current_version=0)
    trainer.replay_buffer.add(_sample(group_version=0), current_version=0)
    trainer.replay_ess_base = 0.71
    metrics = {}
    trainer._add_replay_metrics(metrics, {"n_new": 2, "n_replayed": 0, "staleness": [0, 0]}, new_version=0)
    assert metrics["replay/ess_base"] == pytest.approx(0.71)


def test_add_replay_metrics_prefers_actor_reported_dynamic_base():
    # When the actor reports the base it actually used (which may evolve during
    # training), that value wins over the trainer's captured auto-base.
    trainer = _make_replay_trainer(mini_size=2, requires_mini_batches=1, ess_auto_base=True)
    for _ in range(2):
        trainer.replay_buffer.add(_sample(group_version=0), current_version=0)
    trainer.replay_ess_base = 0.71  # stale trainer-side capture
    metrics = {
        "staleness/ess": [
            {"base_ess_ratio": 0.60, "ess_scaled_lr": 1e-6},
            {"base_ess_ratio": 0.64, "ess_scaled_lr": 1e-6},
            {"base_ess_ratio": None},  # unresolved entry ignored
        ]
    }
    trainer._add_replay_metrics(metrics, {"n_new": 2, "n_replayed": 0, "staleness": [0, 0]}, new_version=0)
    assert metrics["replay/ess_base"] == pytest.approx(0.62)  # mean of actor-used values


def test_process_structured_metrics_emits_base_ess_ratio_scalar():
    from recipe.fully_async_policy.detach_utils import process_structured_metrics

    payload = process_structured_metrics(
        {
            "staleness/ess": [
                {"minibatch_ess_ratio": 0.5, "base_ess_ratio": 0.8, "ess_scaled_lr": 1e-6},
                {"minibatch_ess_ratio": 0.7, "base_ess_ratio": 0.9, "ess_scaled_lr": 2e-6},
                {"minibatch_ess_ratio": 0.6, "base_ess_ratio": None},
            ]
        },
        allow_media=False,
    )
    assert payload["staleness/base_ess_ratio"] == pytest.approx(0.85)
    assert payload["staleness/ess_ratio"] == pytest.approx(0.6)
    assert payload["actor/ess_scaled_lr"] == pytest.approx(1.5e-6)


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
