# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""KL to a reference policy in the fully-async recipe (actor.use_kl_loss).

Trainer side (FullyAsyncTrainer):
  * _ensure_ref_log_probs: the reference log-probs of a composed replay mini-batch are computed in
    ONE reference forward over the groups that lack them, padded to the trainer DP size, and cached
    in each group's full_batch, so a replayed group never costs a second forward;
  * _maybe_reset_reference: async_training.kl_ref_reset_interval re-anchors the reference at the
    current policy and drops that cache; null never resets.
Worker side: copy_actor_params_to_ref / DetachActorWorker.reset_ref_to_actor copy the actor's weights
into the reference living in the same process.

The actor-side half (the KL term must not be scaled by the per-trajectory advantage) is covered in
tests/workers/actor/test_update_policy_per_traj_on_cpu.py.

Run: pytest recipe/fully_async_policy/unittest/test_kl_reference_on_cpu.py
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import recipe.fully_async_policy.megatron_worker as megatron_worker_module
from recipe.fully_async_policy.detach_utils import RolloutSample
from recipe.fully_async_policy.fully_async_trainer import (
    REF_LOG_PROB_KEY,
    parse_kl_ref_reset_interval,
    resolve_kl_ref_reset_interval,
)
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from recipe.fully_async_policy.megatron_utils import copy_actor_params_to_ref
from recipe.fully_async_policy.replay_buffer import GroupEntry, ReplayBuffer
from verl import DataProto

# the class is a @ray.remote ActorClass wrapper; the tests need the plain class
FullyAsyncTrainer = getattr(_TrainerActor, "__ray_metadata__", None)
FullyAsyncTrainer = FullyAsyncTrainer.modified_class if FullyAsyncTrainer is not None else _TrainerActor

RESPONSE_LEN = 3


def _sample(marker: int, n: int = 3) -> RolloutSample:
    """A group of n sequences whose input_ids are filled with ``marker`` (so the fake reference can
    return values that identify the row they were computed from)."""
    seq_len = RESPONSE_LEN + 2
    full_batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(n, RESPONSE_LEN, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            "responses": torch.zeros(n, RESPONSE_LEN, dtype=torch.long),
            "input_ids": torch.full((n, seq_len), marker, dtype=torch.long),
            "position_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "rollout_log_probs": torch.zeros(n, RESPONSE_LEN),
        },
        non_tensors={
            "uid": np.array([f"uid_{marker}"] * n, dtype=object),
            "reward_scalar": np.asarray([1.0] + [-1.0] * (n - 1), dtype=np.float32),
            "advantage_scalar": np.asarray([1.0] + [-0.5] * (n - 1), dtype=np.float32),
            "param_version_start": np.array([0] * n),
            "param_version_end": np.array([0] * n),
            "processing_times": np.array([0.1] * n),
            "tool_calls_times": np.array([0.0] * n),
        },
    )
    return RolloutSample(
        full_batch=full_batch,
        agent_loop_output_list=[],
        sample_id=f"sample_{marker}",
        epoch=0,
        processing_times=[],
        tool_calls=[],
        param_version=0,
        param_version_start=[0],
        param_version_end=[0],
        rollout_status={"count/current_param_version": 0},
        group_version=0,
    )


class _FakeRefWorkerGroup:
    """compute_ref_log_prob returns marker + 0.5 for every token of a row; records what it saw."""

    def __init__(self):
        self.batch_sizes = []
        self.batch_keys = []
        self.resets = 0

    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        self.batch_sizes.append(len(data))
        self.batch_keys.append(sorted(data.batch.keys()))
        marker = data.batch["input_ids"][:, :1].float()
        return DataProto.from_dict(tensors={REF_LOG_PROB_KEY: (marker + 0.5).expand(-1, RESPONSE_LEN).clone()})

    def reset_ref_to_actor(self):
        self.resets += 1


def _config(n_gpus=5, use_kl_loss=True, interval=None, strategy="megatron"):
    return OmegaConf.create(
        {
            "trainer": {"nnodes": 1, "n_gpus_per_node": n_gpus, "balance_batch": False},
            "algorithm": {"rollout_correction": {"rollout_is": "token", "rollout_is_threshold": 2.0}},
            "async_training": {"kl_ref_reset_interval": interval},
            "actor_rollout_ref": {
                "rollout": {"n": 3, "temperature": 1.0, "multi_turn": {"enable": False}},
                "actor": {
                    "strategy": strategy,
                    "use_kl_loss": use_kl_loss,
                    "grad_baselining": {"enable": False},
                    "update_policy_per_traj": True,
                    "megatron": {
                        "tensor_model_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                        "context_parallel_size": 1,
                    },
                },
            },
        }
    )


def _trainer(config=None, replay=True, interval=None):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.config = config if config is not None else _config()
    t.tokenizer = None
    t.use_reference_policy = True
    t.ref_policy_wg = _FakeRefWorkerGroup()
    t.replay_enable = replay
    t.replay_buffer = ReplayBuffer(tau=4.0, staleness_threshold=100, seed=0)
    t.current_param_version = 0
    t.kl_ref_reset_interval = interval
    t.kl_ref_resets_total = 0
    t.kl_ref_last_reset_version = 0
    t._kl_ref_pending_metrics = {}
    t._replay_ref_computed_groups = 0
    return t


def _entries(*markers):
    return [GroupEntry(sample=_sample(m), group_version=0, score=1.0, insert_seq=i) for i, m in enumerate(markers)]


# ==================== reference log-probs of a replay mini-batch ====================


def test_reference_forward_runs_once_per_group_and_is_cached():
    t = _trainer()
    entries = _entries(7, 9)  # 2 groups x 3 sequences = 6 rows, trainer dp = 5
    assert t._ensure_ref_log_probs(entries) == 2

    # one call, padded from 6 rows up to the next multiple of the DP size, with the forward keys only
    assert t.ref_policy_wg.batch_sizes == [10]
    assert t.ref_policy_wg.batch_keys == [["attention_mask", "input_ids", "position_ids", "responses"]]
    for entry, marker in zip(entries, (7, 9), strict=True):
        cached = entry.sample.full_batch.batch[REF_LOG_PROB_KEY]
        assert cached.shape == (3, RESPONSE_LEN)
        assert torch.all(cached == marker + 0.5)  # each group got ITS rows, padding rows were dropped

    # a replayed group is never forwarded again
    assert t._ensure_ref_log_probs(entries) == 0
    assert t.ref_policy_wg.batch_sizes == [10]


def test_only_groups_without_cached_values_are_forwarded():
    t = _trainer(config=_config(n_gpus=3))
    replayed, fresh = _entries(2, 4)
    replayed.sample.full_batch.batch[REF_LOG_PROB_KEY] = torch.full((3, RESPONSE_LEN), -1.0)
    assert t._ensure_ref_log_probs([fresh, replayed]) == 1
    assert t.ref_policy_wg.batch_sizes == [3]  # 3 rows split over dp = 3: no padding
    assert torch.all(replayed.sample.full_batch.batch[REF_LOG_PROB_KEY] == -1.0)  # untouched
    assert torch.all(fresh.sample.full_batch.batch[REF_LOG_PROB_KEY] == 4.5)


def test_groups_with_the_rollouters_meta_info_are_forwarded_together():
    """Real groups carry the same meta_info keys (DataProto.concat asserts they agree); the
    reference forward still gets one batch, and each group's own meta_info is left as it was."""
    t = _trainer()
    entries = _entries(3, 6, 8)  # 9 rows, trainer dp = 5 -> padded to 10
    for e in entries:
        e.sample.full_batch.meta_info.update({"eos_token_id": 2, "pad_token_id": 0})
    assert t._ensure_ref_log_probs(entries) == 3
    assert t.ref_policy_wg.batch_sizes == [10]
    for entry, marker in zip(entries, (3, 6, 8), strict=True):
        assert entry.sample.full_batch.meta_info == {"eos_token_id": 2, "pad_token_id": 0}
        assert torch.all(entry.sample.full_batch.batch[REF_LOG_PROB_KEY] == marker + 0.5)


def test_replay_batch_carries_reference_log_probs_aligned_with_its_rows():
    t = _trainer()
    entries = _entries(1, 5)
    t._ensure_ref_log_probs(entries)
    batch = t._build_replay_batch(entries)
    assert list(batch.non_tensor_batch["uid"]) == ["uid_1"] * 3 + ["uid_5"] * 3
    expected = torch.tensor([1.5] * 3 + [5.5] * 3).unsqueeze(-1).expand(-1, RESPONSE_LEN)
    torch.testing.assert_close(batch.batch[REF_LOG_PROB_KEY], expected)


# ==================== reference reset ====================


def test_reset_never_happens_without_an_interval():
    t = _trainer(interval=None)
    for version in range(1, 10):
        t.current_param_version = version
        assert t._maybe_reset_reference() is False
    assert t.ref_policy_wg.resets == 0


def test_reset_fires_once_per_interval_and_drops_the_cache():
    t = _trainer(interval=4)
    entries = _entries(3, 6)
    t.replay_buffer.entries = list(entries)
    t._ensure_ref_log_probs(entries)

    fired = []
    for version in (0, 1, 2, 3, 4, 4, 5, 8):  # 4 twice: the version did not move (FIFO sync every k updates)
        t.current_param_version = version
        fired.append(t._maybe_reset_reference())
    assert fired == [False, False, False, False, True, False, False, True]
    assert t.ref_policy_wg.resets == 2
    assert (t.kl_ref_resets_total, t.kl_ref_last_reset_version) == (2, 8)
    # cached values were computed against the OLD reference: dropped, recomputed lazily
    assert all(REF_LOG_PROB_KEY not in e.sample.full_batch.batch.keys() for e in entries)
    assert t._ensure_ref_log_probs(entries) == 2


def test_reset_metrics_are_reported_with_the_next_step_only_once():
    t = _trainer(interval=2)
    t.replay_buffer.entries = _entries(1)
    t._ensure_ref_log_probs(t.replay_buffer.entries)
    t.current_param_version = 2
    assert t._maybe_reset_reference() is True

    metrics = {}
    t._add_kl_ref_metrics(metrics)
    assert metrics["kl_ref/resets_total"] == 1
    assert metrics["kl_ref/last_reset_version"] == 2
    assert metrics["kl_ref/dropped_cached_groups"] == 1
    assert metrics["timing_s/ref_reset"] >= 0.0
    assert "replay/ref_computed_groups" in metrics

    later = {}
    t._add_kl_ref_metrics(later)
    assert "timing_s/ref_reset" not in later and later["kl_ref/resets_total"] == 1


def test_reset_works_on_the_fifo_path_without_a_replay_buffer():
    t = _trainer(replay=False, interval=3)
    del t.replay_buffer
    t.current_param_version = 3
    assert t._maybe_reset_reference() is True
    assert t.ref_policy_wg.resets == 1
    metrics = {}
    t._add_kl_ref_metrics(metrics)
    assert "replay/ref_computed_groups" not in metrics


def test_nothing_happens_without_a_reference_policy():
    t = _trainer(interval=1)
    t.use_reference_policy = False
    t.current_param_version = 1
    assert t._maybe_reset_reference() is False
    metrics = {}
    t._add_kl_ref_metrics(metrics)
    assert metrics == {}


@pytest.mark.parametrize("value,expected", [(None, None), ("null", None), ("", None), (48, 48), ("48", 48), (48.0, 48)])
def test_interval_parsing(value, expected):
    assert parse_kl_ref_reset_interval(value) == expected


@pytest.mark.parametrize("value", [0, -4, 2.5, "2.5"])
def test_interval_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        parse_kl_ref_reset_interval(value)


def test_interval_needs_the_kl_loss_and_megatron():
    assert resolve_kl_ref_reset_interval(_config(interval=None, use_kl_loss=False)) is None
    assert resolve_kl_ref_reset_interval(_config(interval=24)) == 24
    with pytest.raises(AssertionError, match="use_kl_loss=True"):
        resolve_kl_ref_reset_interval(_config(interval=24, use_kl_loss=False))
    with pytest.raises(AssertionError, match="megatron strategy only"):
        resolve_kl_ref_reset_interval(_config(interval=24, strategy="fsdp2"))


# ==================== actor -> reference weight copy ====================


class _Wrapper(torch.nn.Module):
    """Stands in for DDP / Float16Module: the real model sits under ``.module``."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def _net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2, bias=False))


def test_copy_unwraps_the_actor_and_overwrites_the_reference():
    actor = _Wrapper(_Wrapper(_net(0)))  # DDP(Float16Module(model))
    ref = _Wrapper(_net(1))  # Float16Module(model), not DDP-wrapped
    copied = copy_actor_params_to_ref([actor], [ref], wrapper_types=(_Wrapper,))
    assert copied == 3
    for a, r in zip(actor.module.module.parameters(), ref.module.parameters(), strict=True):
        assert torch.equal(a, r)
        assert a.data_ptr() != r.data_ptr()  # a copy, not an alias: the actor keeps training


def test_copy_rejects_models_that_do_not_match():
    wrong_shape = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 5, bias=False))
    with pytest.raises(ValueError, match="in the actor"):
        copy_actor_params_to_ref([_net(0)], [wrong_shape], wrapper_types=(_Wrapper,))
    with pytest.raises(ValueError, match="parameter names differ"):
        copy_actor_params_to_ref([_net(0)], [torch.nn.Sequential(torch.nn.Linear(4, 3))], wrapper_types=(_Wrapper,))
    with pytest.raises(ValueError, match="model chunks"):
        copy_actor_params_to_ref([_net(0), _net(0)], [_net(1)], wrapper_types=(_Wrapper,))


def test_copy_refuses_an_offloaded_actor():
    actor, ref = _net(0), _net(1)
    before = [p.detach().clone() for p in ref.parameters()]
    next(actor.parameters()).data.untyped_storage().resize_(0)  # what offload_megatron_model_to_cpu does to DDP buffers
    with pytest.raises(RuntimeError, match="no storage"):
        copy_actor_params_to_ref([actor], [ref], wrapper_types=(_Wrapper,))
    assert torch.equal(next(ref.parameters()), before[0])


def _ref_worker(monkeypatch, ref_module):
    worker = megatron_worker_module.DetachActorWorker.__new__(megatron_worker_module.DetachActorWorker)
    worker._is_ref = True
    worker.ref_module = ref_module
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0)
    return worker


def test_reference_worker_copies_from_the_actor_of_its_process(monkeypatch):
    actor, ref = _net(0), _net(1)
    monkeypatch.setattr(megatron_worker_module, "_LOCAL_ACTOR_MODULES", {"actor": [actor]})
    assert _ref_worker(monkeypatch, [ref]).reset_ref_to_actor() == 3
    for a, r in zip(actor.parameters(), ref.parameters(), strict=True):
        assert torch.equal(a, r)


def test_reference_worker_without_a_colocated_actor_fails_loudly(monkeypatch):
    monkeypatch.setattr(megatron_worker_module, "_LOCAL_ACTOR_MODULES", {})
    with pytest.raises(RuntimeError, match="same process"):
        _ref_worker(monkeypatch, [_net(1)]).reset_ref_to_actor()


def test_only_the_actor_role_registers_its_model(monkeypatch):
    registry = {}
    monkeypatch.setattr(megatron_worker_module, "_LOCAL_ACTOR_MODULES", registry)
    monkeypatch.setattr(megatron_worker_module.AsyncActorRolloutRefWorker, "init_model", lambda self: None)
    for is_actor, expected in ((False, {}), (True, {"actor": ["chunk"]})):
        worker = megatron_worker_module.DetachActorWorker.__new__(megatron_worker_module.DetachActorWorker)
        worker._is_actor = is_actor
        worker.actor_module = ["chunk"]
        worker.init_model()
        assert registry == expected
