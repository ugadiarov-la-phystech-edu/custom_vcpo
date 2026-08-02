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
"""Unit tests for opportunistic PPO epochs in FullyAsyncTrainer:
- make_opportunistic_minibatch_indices yields group-complete, equal-size,
  seed-controlled shuffled mini-batches
- _run_opportunistic_epochs stops the moment the queue holds a full batch,
  honors the per-step cap, logs extra-update counts and per-epoch metrics,
  and leaves the original batch's meta_info untouched

Run: pytest recipe/fully_async_policy/unittest/test_opportunistic_epochs_on_cpu.py
"""

import numpy as np
import pytest
import torch

from recipe.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer as _TrainerActor,
)
from recipe.fully_async_policy.fully_async_trainer import (
    make_opportunistic_minibatch_indices,
)
from verl.protocol import DataProto


def _unwrap_ray_actor_class(actor_cls):
    """FullyAsyncTrainer is a @ray.remote ActorClass wrapper; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)

N_GROUPS = 4
N_RESP = 2  # responses per group


class _StubQueueClient:
    """get_queue_size_sync pops scripted sizes; the last one repeats."""

    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.calls = 0

    def get_queue_size_sync(self):
        self.calls += 1
        return self.sizes.pop(0) if len(self.sizes) > 1 else self.sizes[0]


class _StubActorWorkerGroup:
    def __init__(self):
        self.calls = []

    def update_actor(self, subset):
        self.calls.append(subset)
        # update_policy metrics arrive as lists (one entry per micro-batch)
        # and go through reduce_metrics on the driver.
        return DataProto(
            meta_info={
                "metrics": {
                    "actor/pg_clipfrac": [0.1, 0.3],
                    "rollout_corr/rollout_is_mean": [1.0],
                    "perf/mfu/actor": [0.5],
                }
            }
        )


def _make_batch(n_groups=N_GROUPS, n_resp=N_RESP, distinct_token_counts=False):
    n_seqs = n_groups * n_resp
    uids = np.array([f"g{i // n_resp}" for i in range(n_seqs)], dtype=object)
    if distinct_token_counts:
        # row i has i+1 valid tokens, so global_token_num identifies the row
        mask = torch.zeros(n_seqs, n_seqs + 1, dtype=torch.long)
        for i in range(n_seqs):
            mask[i, : i + 1] = 1
    else:
        mask = torch.ones(n_seqs, 5, dtype=torch.long)
    batch = DataProto.from_dict(tensors={"attention_mask": mask}, non_tensors={"uid": uids})
    batch.meta_info["skip_recompute_old_log_prob"] = True
    return batch


def _make_trainer(queue_sizes, enable=True, max_extra_epochs=3, require_batches=2, n_groups=N_GROUPS, global_steps=5):
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.opportunistic_enable = enable
    trainer.opportunistic_max_extra_epochs = max_extra_epochs
    trainer.opportunistic_shuffle_seed = 7
    trainer.require_batches = require_batches
    trainer.required_samples = n_groups  # in prompt-group units, as in __init__
    trainer.global_steps = global_steps
    trainer.message_queue_client = _StubQueueClient(queue_sizes)
    trainer.actor_rollout_wg = _StubActorWorkerGroup()
    return trainer


# ---------------------------------------------------------------------------
# make_opportunistic_minibatch_indices
# ---------------------------------------------------------------------------


def test_minibatch_indices_group_complete_and_equal_size():
    uids = np.array([f"g{i // 8}" for i in range(128 * 8)], dtype=object)
    chunks = make_opportunistic_minibatch_indices(uids, 4, np.random.default_rng(0))
    assert len(chunks) == 4
    seen = []
    for chunk in chunks:
        assert len(chunk) == 128 * 8 // 4
        # each group is entirely inside one chunk
        chunk_uids = uids[chunk]
        for uid, count in zip(*np.unique(chunk_uids, return_counts=True), strict=True):
            assert count == 8, f"group {uid} split across mini-batches"
        seen.extend(chunk.tolist())
    # a permutation of the whole batch
    assert sorted(seen) == list(range(128 * 8))


def test_minibatch_indices_shuffle_is_seeded():
    uids = np.array([f"g{i // 2}" for i in range(16)], dtype=object)
    a = make_opportunistic_minibatch_indices(uids, 2, np.random.default_rng(0))
    b = make_opportunistic_minibatch_indices(uids, 2, np.random.default_rng(0))
    c = make_opportunistic_minibatch_indices(uids, 2, np.random.default_rng(1))
    assert all((x == y).all() for x, y in zip(a, b, strict=True))
    assert any((x != y).any() for x, y in zip(a, c, strict=True))


def test_minibatch_indices_rejects_bad_shapes():
    with pytest.raises(ValueError, match="not divisible"):
        make_opportunistic_minibatch_indices(
            np.array(["g0", "g0", "g1", "g1", "g2", "g2"], dtype=object), 2, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="unequal sizes"):
        make_opportunistic_minibatch_indices(
            np.array(["g0", "g0", "g1", "g1", "g1", "g2", "g2", "g3"], dtype=object), 2, np.random.default_rng(0)
        )


# ---------------------------------------------------------------------------
# _run_opportunistic_epochs
# ---------------------------------------------------------------------------


def test_disabled_makes_no_calls_and_no_metrics():
    trainer = _make_trainer(queue_sizes=[0], enable=False)
    metrics, timing_raw = {}, {}
    trainer._run_opportunistic_epochs(_make_batch(), metrics, timing_raw)
    assert trainer.actor_rollout_wg.calls == []
    assert trainer.message_queue_client.calls == 0
    assert metrics == {}


def test_dormant_when_queue_already_full():
    trainer = _make_trainer(queue_sizes=[N_GROUPS])
    metrics, timing_raw = {}, {}
    trainer._run_opportunistic_epochs(_make_batch(), metrics, timing_raw)
    assert trainer.actor_rollout_wg.calls == []
    assert metrics["opportunistic/extra_updates"] == 0
    assert metrics["opportunistic/extra_epochs_completed"] == 0
    assert "opportunistic_extra" in timing_raw


def test_runs_to_cap_when_queue_never_fills():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=3, require_batches=2)
    metrics, timing_raw = {}, {}
    trainer._run_opportunistic_epochs(_make_batch(), metrics, timing_raw)
    assert metrics["opportunistic/extra_updates"] == 3 * 2
    assert metrics["opportunistic/extra_epochs_completed"] == 3
    assert metrics["opportunistic/extra_epochs"] == 3.0
    for epoch in (1, 2, 3):
        assert metrics[f"opportunistic/epoch_{epoch}/actor/pg_clipfrac"] == pytest.approx(0.2)
        assert metrics[f"opportunistic/epoch_{epoch}/rollout_corr/rollout_is_mean"] == pytest.approx(1.0)
        # perf/ keys are not re-keyed per epoch
        assert f"opportunistic/epoch_{epoch}/perf/mfu/actor" not in metrics


def test_stops_mid_epoch_when_queue_fills():
    # checks: epoch1 mb1 (0), epoch1 mb2 (0), epoch2 mb1 (0), epoch2 mb2 (full)
    trainer = _make_trainer(queue_sizes=[0, 0, 0, N_GROUPS], max_extra_epochs=3, require_batches=2)
    metrics, timing_raw = {}, {}
    trainer._run_opportunistic_epochs(_make_batch(), metrics, timing_raw)
    assert metrics["opportunistic/extra_updates"] == 3
    assert metrics["opportunistic/extra_epochs_completed"] == 1
    assert metrics["opportunistic/extra_epochs"] == pytest.approx(1.5)
    # partial epoch 2 still logs the metrics of the one update that ran
    assert "opportunistic/epoch_2/actor/pg_clipfrac" in metrics
    assert "opportunistic/epoch_3/actor/pg_clipfrac" not in metrics


def test_subsets_are_group_complete_with_fresh_meta_info():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=1, require_batches=2)
    batch = _make_batch()
    metrics, timing_raw = {}, {}
    trainer._run_opportunistic_epochs(batch, metrics, timing_raw)
    assert len(trainer.actor_rollout_wg.calls) == 2
    for subset in trainer.actor_rollout_wg.calls:
        assert len(subset) == N_GROUPS * N_RESP // 2
        counts = np.unique(subset.non_tensor_batch["uid"], return_counts=True)[1]
        assert (counts == N_RESP).all()
        assert subset.meta_info["opportunistic_extra_epoch"] == 1
        assert subset.meta_info["global_token_num"] == [5] * (N_GROUPS * N_RESP // 2)
    # stamping happened on copies, not on the shared original meta_info
    assert "global_token_num" not in batch.meta_info
    assert "opportunistic_extra_epoch" not in batch.meta_info


def test_each_epoch_partitions_all_groups():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=1, require_batches=2, n_groups=8)
    trainer._run_opportunistic_epochs(_make_batch(n_groups=8), {}, {})
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 2
    uid_sets = [set(subset.non_tensor_batch["uid"].tolist()) for subset in calls]
    assert uid_sets[0].isdisjoint(uid_sets[1])
    assert uid_sets[0] | uid_sets[1] == {f"g{i}" for i in range(8)}


def test_epochs_reshuffle_within_step():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=2, require_batches=2, n_groups=16)
    trainer._run_opportunistic_epochs(_make_batch(n_groups=16), {}, {})
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 4
    epoch1 = tuple(uid for subset in calls[:2] for uid in subset.non_tensor_batch["uid"].tolist())
    epoch2 = tuple(uid for subset in calls[2:] for uid in subset.non_tensor_batch["uid"].tolist())
    assert epoch1 != epoch2, "the rng must advance between extra epochs"


def test_shuffle_depends_on_global_step():
    def first_epoch_uids(global_steps):
        trainer = _make_trainer(
            queue_sizes=[0], max_extra_epochs=1, require_batches=2, n_groups=16, global_steps=global_steps
        )
        trainer._run_opportunistic_epochs(_make_batch(n_groups=16), {}, {})
        return tuple(
            uid for subset in trainer.actor_rollout_wg.calls for uid in subset.non_tensor_batch["uid"].tolist()
        )

    assert first_epoch_uids(5) != first_epoch_uids(6)
    assert first_epoch_uids(5) == first_epoch_uids(5), "same step must shuffle identically"


def test_queue_checked_before_every_minibatch():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=2, require_batches=2)
    trainer._run_opportunistic_epochs(_make_batch(), {}, {})
    # one check per attempted mini-batch: 2 epochs x 2 mini-batches
    assert trainer.message_queue_client.calls == 4
    assert len(trainer.actor_rollout_wg.calls) == 4


def test_global_token_num_matches_selected_rows():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=1, require_batches=2)
    trainer._run_opportunistic_epochs(_make_batch(distinct_token_counts=True), {}, {})
    for subset in trainer.actor_rollout_wg.calls:
        # group g_k covers original rows k*N_RESP..k*N_RESP+N_RESP-1; row i has i+1 tokens
        expected = []
        for uid in dict.fromkeys(subset.non_tensor_batch["uid"].tolist()):
            k = int(uid[1:])
            expected.extend(k * N_RESP + r + 1 for r in range(N_RESP))
        assert subset.meta_info["global_token_num"] == expected, "rows must stay aligned with uids"


def test_require_batches_one_uses_whole_batch_per_epoch():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=2, require_batches=1)
    metrics = {}
    trainer._run_opportunistic_epochs(_make_batch(), metrics, {})
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 2
    assert all(len(subset) == N_GROUPS * N_RESP for subset in calls)
    assert metrics["opportunistic/extra_updates"] == 2
    assert metrics["opportunistic/extra_epochs"] == 2.0


def test_malformed_groups_skip_gracefully():
    # one group has a missing response: unequal group sizes must not crash the
    # run, just disable the extra epochs for this step
    uids = np.array(["g0", "g0", "g1", "g1", "g1", "g2", "g2", "g3"], dtype=object)
    batch = DataProto.from_dict(
        tensors={"attention_mask": torch.ones(len(uids), 5, dtype=torch.long)},
        non_tensors={"uid": uids},
    )
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=3, require_batches=2)
    metrics = {}
    trainer._run_opportunistic_epochs(batch, metrics, {})
    assert trainer.actor_rollout_wg.calls == []
    assert trainer.message_queue_client.calls == 0
    assert metrics["opportunistic/extra_updates"] == 0


def test_base_metrics_not_overwritten():
    trainer = _make_trainer(queue_sizes=[0], max_extra_epochs=1, require_batches=2)
    metrics = {"actor/pg_clipfrac": 0.99}  # from the scheduled pass
    trainer._run_opportunistic_epochs(_make_batch(), metrics, {})
    assert metrics["actor/pg_clipfrac"] == 0.99
    assert metrics["opportunistic/epoch_1/actor/pg_clipfrac"] == pytest.approx(0.2)
