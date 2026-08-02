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
"""Unit tests for elastic DAPO-style dynamic filtering in FullyAsyncRollouter:
- _group_is_degenerate: zero-variance detection via the reward fn, keep rules
  for singleton groups / missing reward_fn, fail-open on scoring errors
- _should_drop_degenerate: queue-depth elastic gate (drop only when at least
  min_buffered_batches full trainer batches are buffered)
- _process_single_sample_streaming: dropped groups decrement the staleness
  quota, increment the filtered counter, and never reach the queue; kept
  groups enqueue exactly as before

Run: pytest recipe/fully_async_policy/unittest/test_dynamic_filtering_on_cpu.py
"""

import asyncio

import numpy as np
import torch

import recipe.fully_async_policy.fully_async_rollouter as far_module
from recipe.fully_async_policy.detach_utils import RolloutSample
from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from verl.protocol import DataProto


def _unwrap_ray_actor_class(actor_cls):
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap_ray_actor_class(_RollouterActor)

N_RESP = 4
_DUMMY_REWARD_FN = object()


def _make_sample(n_resp=N_RESP):
    batch = DataProto.from_dict(
        tensors={"responses": torch.zeros(n_resp, 3, dtype=torch.long)},
        non_tensors={"uid": np.array(["g0"] * n_resp, dtype=object)},
    )
    return RolloutSample(
        full_batch=batch,
        agent_loop_output_list=[],
        sample_id="s0",
        epoch=0,
        processing_times=[],
        tool_calls=[],
        param_version=0,
        param_version_start=[],
        param_version_end=[],
        rollout_status={},
    )


class _StubQueueClient:
    def __init__(self, queue_size=0):
        self.queue_size = queue_size
        self.put_samples = []

    async def get_queue_size(self):
        return self.queue_size

    async def put_sample(self, sample, param_version):
        self.put_samples.append(param_version)
        return True


class _StubRolloutManager:
    def __init__(self, batch):
        self._batch = batch

    async def generate_single_sample_async(self, full_batch, agent_loop_output_list):
        return self._batch, False


def _make_rollouter(enable=True, queue_size=0, min_buffered=1.0, required_samples=4, reward_fn=_DUMMY_REWARD_FN):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.dynamic_filtering_enable = enable
    r.dynamic_filtering_min_buffered = min_buffered
    r.required_samples = required_samples
    r.reward_fn = reward_fn
    r.message_queue_client = _StubQueueClient(queue_size=queue_size)
    r.current_param_version = 7
    r.filtered_degenerate_groups = 0
    r.staleness_samples = 10
    r.processed_sample_count = 0
    r.total_generated_samples = 0
    r.dropped_stale_samples = 0
    r.cumulative_validation_time = 0.0
    r.cumulative_checkpoint_pause = 0.0

    async def get_statistics():
        return {}

    r.get_statistics = get_statistics
    return r


def _patch_rewards(monkeypatch, per_seq_scores):
    """compute_reward returns a token-level tensor whose per-seq sum is per_seq_scores."""

    def fake_compute_reward(batch, reward_fn):
        n = len(per_seq_scores)
        tensor = torch.zeros(n, 3)
        tensor[:, -1] = torch.tensor(per_seq_scores, dtype=torch.float32)
        return tensor, {}

    monkeypatch.setattr(far_module, "compute_reward", fake_compute_reward)


# ---------------------------------------------------------------------------
# _group_is_degenerate
# ---------------------------------------------------------------------------


def test_uniform_rewards_are_degenerate(monkeypatch):
    rollouter = _make_rollouter()
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    assert rollouter._group_is_degenerate(_make_sample()) is True
    _patch_rewards(monkeypatch, [-1.0, -1.0, -1.0, -1.0])
    assert rollouter._group_is_degenerate(_make_sample()) is True


def test_mixed_rewards_are_kept(monkeypatch):
    rollouter = _make_rollouter()
    _patch_rewards(monkeypatch, [1.0, -1.0, 1.0, 1.0])
    assert rollouter._group_is_degenerate(_make_sample()) is False


def test_singleton_group_and_missing_reward_fn_are_kept(monkeypatch):
    _patch_rewards(monkeypatch, [1.0])
    assert _make_rollouter()._group_is_degenerate(_make_sample(n_resp=1)) is False
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    assert _make_rollouter(reward_fn=None)._group_is_degenerate(_make_sample()) is False


def test_scoring_failure_fails_open(monkeypatch):
    rollouter = _make_rollouter()

    def broken(batch, reward_fn):
        raise RuntimeError("no ground truth")

    monkeypatch.setattr(far_module, "compute_reward", broken)
    assert rollouter._group_is_degenerate(_make_sample()) is False


# ---------------------------------------------------------------------------
# _should_drop_degenerate: the elastic gate
# ---------------------------------------------------------------------------


def test_gate_blocks_when_queue_shallow(monkeypatch):
    rollouter = _make_rollouter(queue_size=3, required_samples=4)  # < 1 full batch buffered
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    assert asyncio.run(rollouter._should_drop_degenerate(_make_sample())) is False


def test_gate_opens_when_full_batch_buffered(monkeypatch):
    rollouter = _make_rollouter(queue_size=4, required_samples=4)
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    assert asyncio.run(rollouter._should_drop_degenerate(_make_sample())) is True


def test_gate_respects_min_buffered_batches(monkeypatch):
    rollouter = _make_rollouter(queue_size=6, required_samples=4, min_buffered=2.0)  # needs 8
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    assert asyncio.run(rollouter._should_drop_degenerate(_make_sample())) is False


def test_gate_disabled_never_queries_queue(monkeypatch):
    rollouter = _make_rollouter(enable=False, queue_size=100)

    async def boom():
        raise AssertionError("queue must not be queried when disabled")

    rollouter.message_queue_client.get_queue_size = boom
    assert asyncio.run(rollouter._should_drop_degenerate(_make_sample())) is False


# ---------------------------------------------------------------------------
# _process_single_sample_streaming: drop vs enqueue bookkeeping
# ---------------------------------------------------------------------------


def _run_streaming(rollouter, sample):
    rollouter.async_rollout_manager = _StubRolloutManager(sample.full_batch)
    asyncio.run(rollouter._process_single_sample_streaming(sample))


def test_degenerate_group_dropped_with_quota_decrement(monkeypatch):
    rollouter = _make_rollouter(queue_size=4, required_samples=4)
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    _run_streaming(rollouter, _make_sample())
    assert rollouter.message_queue_client.put_samples == [], "dropped group must not reach the queue"
    assert rollouter.filtered_degenerate_groups == 1
    assert rollouter.staleness_samples == 9, "drop must license a replacement within the quota"
    assert rollouter.processed_sample_count == 1
    assert rollouter.total_generated_samples == 0


def test_informative_group_enqueued(monkeypatch):
    rollouter = _make_rollouter(queue_size=4, required_samples=4)
    _patch_rewards(monkeypatch, [1.0, -1.0, 1.0, 1.0])
    _run_streaming(rollouter, _make_sample())
    assert rollouter.message_queue_client.put_samples == [7]
    assert rollouter.filtered_degenerate_groups == 0
    assert rollouter.staleness_samples == 10
    assert rollouter.total_generated_samples == 1


def test_degenerate_group_kept_when_rollout_bound(monkeypatch):
    rollouter = _make_rollouter(queue_size=0, required_samples=4)
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    _run_streaming(rollouter, _make_sample())
    assert rollouter.message_queue_client.put_samples == [7], "shallow queue -> pass through"
    assert rollouter.filtered_degenerate_groups == 0


def test_filtering_disabled_is_transparent(monkeypatch):
    rollouter = _make_rollouter(enable=False, queue_size=100)
    _patch_rewards(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    _run_streaming(rollouter, _make_sample())
    assert rollouter.message_queue_client.put_samples == [7]
    assert rollouter.filtered_degenerate_groups == 0
