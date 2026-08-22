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
"""Unit tests for the non-integer async_training.ppo_epochs update path
(FullyAsyncRayPPOTrainer._update_actor_fractional_epochs):
- update count = max(1, round(ppo_epochs * require_batches))
- mini-batches are group-complete, cover the batch per epoch, and reshuffle
  across epochs; the shuffle is seeded per global step
- metrics are averaged under unprefixed keys plus update counters
- malformed groups fall back to a single whole-batch update

Run: pytest recipe/fully_async_policy/unittest/test_fractional_ppo_epochs_on_cpu.py
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.protocol import DataProto

N_GROUPS = 4
N_RESP = 2  # responses per group


class _StubActorWorkerGroup:
    def __init__(self):
        self.calls = []

    def update_actor(self, subset):
        self.calls.append(subset)
        return DataProto(
            meta_info={
                "metrics": {
                    "actor/pg_clipfrac": [0.1, 0.3],
                    "rollout_corr/rollout_is_mean": [1.0],
                }
            }
        )


class _StubActorWorkerGroupWithEss(_StubActorWorkerGroup):
    """update_actor also returns the structured staleness/ess entry the ESS
    brake emits: reduce_metrics passes list[dict] values through untouched."""

    def update_actor(self, subset):
        self.calls.append(subset)
        return DataProto(
            meta_info={
                "metrics": {
                    "actor/pg_clipfrac": [0.1, 0.3],
                    "staleness/ess": [
                        {"minibatch_idx": len(self.calls), "minibatch_ess": 4.0, "ess_scaled_lr": 1e-6}
                    ],
                }
            }
        )


def _make_batch(n_groups=N_GROUPS, n_resp=N_RESP, equal_groups=True):
    if equal_groups:
        uids = np.array([f"g{i // n_resp}" for i in range(n_groups * n_resp)], dtype=object)
    else:
        # one group with an extra response: unequal sizes, cannot split evenly
        uids = np.array(["g0", "g0", "g1", "g1", "g1", "g2", "g2", "g3"], dtype=object)
    mask = torch.ones(len(uids), 5, dtype=torch.long)
    batch = DataProto.from_dict(tensors={"attention_mask": mask}, non_tensors={"uid": uids})
    batch.meta_info["skip_recompute_old_log_prob"] = True
    return batch


def _make_trainer(require_batches=2, shuffle_seed=7, global_steps=5):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"async_training": {"require_batches": require_batches, "ppo_epochs_shuffle_seed": shuffle_seed}}
    )
    trainer.global_steps = global_steps
    trainer.actor_rollout_wg = _StubActorWorkerGroup()
    return trainer


def _run(trainer, batch, ppo_epochs):
    metrics, timing_raw = {}, {}
    trainer._update_actor_fractional_epochs(batch, ppo_epochs, metrics, timing_raw)
    return metrics, timing_raw


# ---------------------------------------------------------------------------
# update-count arithmetic
# ---------------------------------------------------------------------------


def test_quarter_epoch_runs_one_minibatch():
    trainer = _make_trainer(require_batches=4)
    metrics, timing_raw = _run(trainer, _make_batch(n_groups=8), 0.25)
    assert len(trainer.actor_rollout_wg.calls) == 1
    assert metrics["actor/ppo_epoch_updates"] == 1
    assert metrics["actor/ppo_epochs_effective"] == pytest.approx(0.25)
    assert "update_actor" in timing_raw
    # the single mini-batch is a quarter of the pull
    assert len(trainer.actor_rollout_wg.calls[0]) == 8 * N_RESP // 4


def test_full_epoch_covers_all_groups_once():
    trainer = _make_trainer(require_batches=2)
    metrics, _ = _run(trainer, _make_batch(), 1.0)
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 2
    uid_sets = [set(subset.non_tensor_batch["uid"].tolist()) for subset in calls]
    assert uid_sets[0].isdisjoint(uid_sets[1])
    assert uid_sets[0] | uid_sets[1] == {f"g{i}" for i in range(N_GROUPS)}
    assert metrics["actor/ppo_epochs_effective"] == pytest.approx(1.0)


def test_one_and_a_half_epochs():
    trainer = _make_trainer(require_batches=2)
    metrics, _ = _run(trainer, _make_batch(n_groups=16), 1.5)
    assert len(trainer.actor_rollout_wg.calls) == 3
    assert metrics["actor/ppo_epoch_updates"] == 3
    assert metrics["actor/ppo_epochs_effective"] == pytest.approx(1.5)


def test_tiny_fraction_still_runs_one_update():
    trainer = _make_trainer(require_batches=2)
    _run(trainer, _make_batch(), 0.05)  # round(0.1) == 0 -> clamped to 1
    assert len(trainer.actor_rollout_wg.calls) == 1


def test_two_epochs_with_require_batches_one():
    trainer = _make_trainer(require_batches=1)
    metrics, _ = _run(trainer, _make_batch(), 2.0)
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 2
    assert all(len(subset) == N_GROUPS * N_RESP for subset in calls)
    assert metrics["actor/ppo_epochs_effective"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# shuffling
# ---------------------------------------------------------------------------


def test_epochs_reshuffle_within_step():
    trainer = _make_trainer(require_batches=2)
    _run(trainer, _make_batch(n_groups=16), 2.0)
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 4
    epoch1 = tuple(uid for subset in calls[:2] for uid in subset.non_tensor_batch["uid"].tolist())
    epoch2 = tuple(uid for subset in calls[2:] for uid in subset.non_tensor_batch["uid"].tolist())
    assert epoch1 != epoch2, "the rng must advance between epochs"


def test_shuffle_depends_on_global_step():
    def order(global_steps):
        trainer = _make_trainer(require_batches=2, global_steps=global_steps)
        _run(trainer, _make_batch(n_groups=16), 1.0)
        return tuple(
            uid for subset in trainer.actor_rollout_wg.calls for uid in subset.non_tensor_batch["uid"].tolist()
        )

    assert order(5) == order(5), "same step must shuffle identically"
    assert order(5) != order(6)


def test_minibatches_are_group_complete_with_fresh_meta_info():
    trainer = _make_trainer(require_batches=2)
    batch = _make_batch()
    _run(trainer, batch, 1.0)
    for subset in trainer.actor_rollout_wg.calls:
        counts = np.unique(subset.non_tensor_batch["uid"], return_counts=True)[1]
        assert (counts == N_RESP).all()
        assert subset.meta_info["global_token_num"] == [5] * (N_GROUPS * N_RESP // 2)
        assert subset.meta_info["skip_recompute_old_log_prob"] is True
    # stamping happened on copies, not on the shared original meta_info
    assert "global_token_num" not in batch.meta_info


# ---------------------------------------------------------------------------
# metrics and fallback
# ---------------------------------------------------------------------------


def test_metrics_averaged_under_unprefixed_keys():
    trainer = _make_trainer(require_batches=2)
    metrics, _ = _run(trainer, _make_batch(), 1.0)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.2)
    assert metrics["rollout_corr/rollout_is_mean"] == pytest.approx(1.0)


def test_unequal_groups_fall_back_to_whole_batch():
    trainer = _make_trainer(require_batches=2)
    metrics, _ = _run(trainer, _make_batch(equal_groups=False), 1.0)
    calls = trainer.actor_rollout_wg.calls
    assert len(calls) == 1
    assert len(calls[0]) == 8  # the full malformed batch, untouched
    assert metrics["actor/ppo_epoch_updates"] == 2  # counted as one whole epoch
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# structured metrics (the ESS brake)
# ---------------------------------------------------------------------------


def test_structured_ess_entries_do_not_crash_the_metric_average():
    """staleness/ess is a list[dict] that reduce_metrics deliberately does not
    reduce; a blanket np.mean over every key raised TypeError as soon as
    actor.ess_scaling.enable=True was combined with async_training.ppo_epochs."""
    trainer = _make_trainer()
    trainer.actor_rollout_wg = _StubActorWorkerGroupWithEss()
    metrics, _ = _run(trainer, _make_batch(), ppo_epochs=1.0)

    assert isinstance(metrics["actor/pg_clipfrac"], float)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.2)


def test_structured_ess_entries_survive_as_a_flat_list():
    """They must still reach process_structured_metrics downstream, one entry
    per mini-batch update, not be dropped or averaged."""
    trainer = _make_trainer()
    trainer.actor_rollout_wg = _StubActorWorkerGroupWithEss()
    metrics, _ = _run(trainer, _make_batch(), ppo_epochs=1.0)

    entries = metrics["staleness/ess"]
    assert isinstance(entries, list)
    assert len(entries) == metrics["actor/ppo_epoch_updates"]
    assert all(isinstance(e, dict) and "minibatch_ess" in e for e in entries)
