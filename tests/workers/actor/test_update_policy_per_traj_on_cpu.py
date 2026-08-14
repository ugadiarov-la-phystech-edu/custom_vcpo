# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""CPU wiring tests for MegatronPPOActor.update_policy_per_traj.

The real path only runs on multi-GPU Megatron; these tests pin the control
flow with stubbed chunks/optimizer/forward so regressions surface locally:

- buffer-free mode (grad_baselining=False): no accum-buffer allocation, the
  advantage folded into the per-microbatch ``loss_multiplier``, no gradient
  zeroing between trajectories, the schedule-level finalize suppressed during
  the loop and restored after, and ``_optimizer_step_with_buffer`` receiving
  ``accum_buffers=None``;
- OPOB mode (grad_baselining=True): two buffer allocations, per-trajectory
  isolation (grad-norm capture + zeroing), and the finalize left untouched;
- ``_optimizer_step_with_buffer``: copies accum buffers when present, runs the
  one-shot ``finalize_model_grads_ignore_dp`` when not, and restores the
  nominal learning rate after an ESS-scaled step.

Run: pytest tests/workers/actor/test_update_policy_per_traj_on_cpu.py
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verl.workers.actor.megatron_actor as megatron_actor
from recipe.fully_async_policy.staleness_utils import TrajRecord, TrajRecordList
from verl import DataProto
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config.actor import ESSScalingConfig
from verl.workers.utils import vcpo

TRAJ_UIDS = ["t0", "t1", "t2"]
ADVANTAGES = [1.5, -0.5, 0.0]  # include an exactly-zero advantage
ORIGINAL_FINALIZE = object()  # sentinel installed as finalize_model_grads_func


class _FakeBuffer:
    def __init__(self, grad_data):
        self.grad_data = grad_data


class _FakeChunk:
    def __init__(self):
        self.buffers = [_FakeBuffer(torch.zeros(4))]
        self.config = SimpleNamespace(finalize_model_grads_func=ORIGINAL_FINALIZE)
        self.zero_grad_buffer_calls = 0

    def zero_grad_buffer(self):
        self.zero_grad_buffer_calls += 1

    def no_sync(self):
        return nullcontext()


class _FakeOptimizer:
    def __init__(self, lr=1e-6):
        self.param_groups = [{"lr": lr}]
        self.zero_grad_calls = 0
        self.stepped_lrs = []

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.stepped_lrs.append(float(self.param_groups[0]["lr"]))
        return True, 0.123, 0


def _make_records() -> TrajRecordList:
    records = TrajRecordList()
    for uid, adv in zip(TRAJ_UIDS, ADVANTAGES, strict=True):
        records.append(
            TrajRecord(
                uid=uid,
                group_uid="g0",
                epoch_idx=0,
                minibatch_idx=0,
                trainer_global_step=0,
                trainer_local_step=0,
                param_version_start=0,
                param_version_end=0,
                trainer_param_version=0,
                response_length=4,
                prompt_length=2,
                advantage_scalar=adv,
                reward_scalar=1.0,
            )
        )
    return records


def _make_minibatch() -> DataProto:
    batch_size, resp_len = len(TRAJ_UIDS), 4
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(batch_size, resp_len, dtype=torch.long),
            "advantages": torch.ones(batch_size, resp_len),
        },
        non_tensors={
            "uid": np.array(["g0"] * batch_size, dtype=object),
            "traj_uid": np.array(TRAJ_UIDS, dtype=object),
        },
    )
    data.meta_info["micro_batch_size"] = 1
    data.meta_info["skip_recompute_old_log_prob"] = False
    data.meta_info["rollout_corr_config"] = {"rollout_is_threshold": None}
    return data


def _make_actor(grad_baselining: bool) -> tuple[MegatronPPOActor, dict]:
    actor = MegatronPPOActor.__new__(MegatronPPOActor)
    actor.config = SimpleNamespace(
        use_dynamic_bsz=False,
        ppo_micro_batch_size_per_gpu=1,
        ppo_mini_batch_size=len(TRAJ_UIDS),
        calculate_entropy=False,
        entropy_coeff=0,
        policy_loss={},
        loss_agg_mode="seq-mean-token-mean",
        grad_baselining=SimpleNamespace(scope="group", norm_by_std=False),
        ess_scaling=ESSScalingConfig(),
    )
    actor.use_distributed_opt = False
    actor.actor_module = [_FakeChunk()]
    actor.actor_optimizer = _FakeOptimizer()

    calls = {
        "fbb_loss_multipliers": [],
        "fbb_finalize_funcs": [],
        "step_accum_buffers": [],
        "grad_norm_uids": [],
        "update_grad_buffer_advs": [],
    }

    def fake_forward_backward_batch(data, **kwargs):
        calls["fbb_loss_multipliers"].append(float(data.meta_info["loss_multiplier"]))
        calls["fbb_finalize_funcs"].append(actor.actor_module[0].config.finalize_model_grads_func)
        return {"output": [({}, None, None, None)]}

    def fake_optimizer_step(accum_buffers, local_traj_records, rollout_is_threshold, minibatch_idx=0, **kwargs):
        calls["step_accum_buffers"].append(accum_buffers)
        return True, {"actor/grad_norm": 0.123}

    def fake_compute_grad_norms(response_len, loss_agg_mode, adv_scalar, microbatch_loss_scale):
        calls["grad_norm_uids"].append(adv_scalar)
        return 2.5, 4.0  # (unscaled_grad_norm, grad_norm)

    def fake_update_grad_buffers(**kwargs):
        assert grad_baselining, "buffer update must never run in buffer-free mode"
        calls["update_grad_buffer_advs"].append(kwargs["adv_scalar"])

    actor.forward_backward_batch = fake_forward_backward_batch
    actor._optimizer_step_with_buffer = fake_optimizer_step
    actor._compute_grad_norms = fake_compute_grad_norms
    actor._update_grad_buffers = fake_update_grad_buffers
    return actor, calls


@pytest.fixture
def patched_env(monkeypatch):
    """Stub the module-level collaborators of update_policy_per_traj."""
    records = _make_records()

    def fake_compute_staleness_statistics(batch, minibatch_idx, rollout_is_threshold, use_old, epoch_idx=0):
        return records, {}

    def fake_compute_grad_info(minibatch, scope):
        minibatch.meta_info["is_last_traj_in_scope"] = {uid: uid == TRAJ_UIDS[-1] for uid in TRAJ_UIDS}
        minibatch.meta_info["reward_std_by_traj_uid"] = {uid: 0.5 for uid in TRAJ_UIDS}
        return minibatch

    allocate_calls = []
    real_allocate = megatron_actor.allocate_grad_accum_buffers

    def counting_allocate(modules):
        allocate_calls.append(modules)
        return real_allocate(modules)

    monkeypatch.setattr(megatron_actor, "compute_staleness_statistics", fake_compute_staleness_statistics)
    monkeypatch.setattr(megatron_actor, "compute_grad_info", fake_compute_grad_info)
    monkeypatch.setattr(megatron_actor, "allocate_grad_accum_buffers", counting_allocate)
    monkeypatch.setattr(
        megatron_actor.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=False: SimpleNamespace(size=lambda: 1),
    )
    monkeypatch.setattr(megatron_actor, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    return {"records": records, "allocate_calls": allocate_calls}


class TestBufferFreeMode:
    def test_no_buffer_allocation(self, patched_env):
        actor, _ = _make_actor(grad_baselining=False)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        assert patched_env["allocate_calls"] == []

    def test_advantage_is_folded_into_loss_multiplier(self, patched_env):
        actor, calls = _make_actor(grad_baselining=False)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        scale = 1.0 / len(TRAJ_UIDS)
        assert calls["fbb_loss_multipliers"] == pytest.approx([scale * adv for adv in ADVANTAGES])

    def test_no_gradient_zeroing_between_trajectories(self, patched_env):
        actor, _ = _make_actor(grad_baselining=False)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        # Once at minibatch start only — zeroing between trajectories would
        # destroy the accumulated gradient.
        assert actor.actor_module[0].zero_grad_buffer_calls == 1
        # Optimizer zero_grad: minibatch start + final cleanup.
        assert actor.actor_optimizer.zero_grad_calls == 2

    def test_finalize_suppressed_during_loop_and_restored_after(self, patched_env):
        actor, calls = _make_actor(grad_baselining=False)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        # Every backward ran with the noop finalize installed...
        assert all(f is vcpo._noop_finalize_model_grads for f in calls["fbb_finalize_funcs"])
        # ...and the original was restored on exit.
        assert actor.actor_module[0].config.finalize_model_grads_func is ORIGINAL_FINALIZE

    def test_optimizer_step_receives_no_buffers(self, patched_env):
        actor, calls = _make_actor(grad_baselining=False)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        assert calls["step_accum_buffers"] == [None]

    def test_traj_records_have_no_grad_norms(self, patched_env):
        actor, _ = _make_actor(grad_baselining=False)
        metrics = actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        for rec in metrics["actor/local_traj_records"]:
            assert rec["grad_norm"] is None
            assert rec["grad_norm_unscaled"] is None


class TestOpobMode:
    def test_two_buffer_allocations(self, patched_env):
        actor, _ = _make_actor(grad_baselining=True)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=True)
        assert len(patched_env["allocate_calls"]) == 2

    def test_per_trajectory_isolation(self, patched_env):
        actor, calls = _make_actor(grad_baselining=True)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=True)
        # Grads zeroed at minibatch start and after every trajectory.
        assert actor.actor_module[0].zero_grad_buffer_calls == 1 + len(TRAJ_UIDS)
        # Grad norms measured and buffers updated once per trajectory.
        assert calls["grad_norm_uids"] == pytest.approx(ADVANTAGES)
        assert calls["update_grad_buffer_advs"] == pytest.approx(ADVANTAGES)

    def test_loss_multiplier_stays_unscaled(self, patched_env):
        actor, calls = _make_actor(grad_baselining=True)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=True)
        scale = 1.0 / len(TRAJ_UIDS)
        assert calls["fbb_loss_multipliers"] == pytest.approx([scale] * len(TRAJ_UIDS))

    def test_finalize_left_untouched(self, patched_env):
        actor, calls = _make_actor(grad_baselining=True)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=True)
        assert all(f is ORIGINAL_FINALIZE for f in calls["fbb_finalize_funcs"])
        assert actor.actor_module[0].config.finalize_model_grads_func is ORIGINAL_FINALIZE

    def test_grad_norms_recorded_on_traj_records(self, patched_env):
        actor, _ = _make_actor(grad_baselining=True)
        metrics = actor.update_policy_per_traj([_make_minibatch()], grad_baselining=True)
        for rec in metrics["actor/local_traj_records"]:
            assert rec["grad_norm"] == 4.0
            assert rec["grad_norm_unscaled"] == 2.5


class TestOptimizerStepWithBuffer:
    @staticmethod
    def _make_step_actor(monkeypatch, ess_scaling=None):
        actor = MegatronPPOActor.__new__(MegatronPPOActor)
        actor.config = SimpleNamespace(ess_scaling=ess_scaling or ESSScalingConfig())
        actor.actor_module = [object()]
        actor.actor_optimizer = _FakeOptimizer(lr=1e-6)

        monkeypatch.setattr(
            megatron_actor,
            "compute_ess_info",
            lambda records, thr: {"ess": 4.0, "ess_ratio": 0.25, "ess_clipped": 4.0, "ess_ratio_clipped": 0.2},
        )
        copied, finalized = [], []
        monkeypatch.setattr(
            megatron_actor, "copy_accum_buffers_to_grad_buffers", lambda modules, bufs: copied.append(bufs)
        )
        monkeypatch.setattr(megatron_actor, "finalize_model_grads_ignore_dp", lambda modules: finalized.append(modules))
        return actor, copied, finalized

    def test_buffer_path_copies_and_skips_finalize(self, monkeypatch):
        actor, copied, finalized = self._make_step_actor(monkeypatch)
        sentinel_buffers = [torch.zeros(2)]
        ok, _ = actor._optimizer_step_with_buffer(sentinel_buffers, [], None, do_grad_sync=False)
        assert ok
        assert copied == [sentinel_buffers]
        assert finalized == []

    def test_buffer_free_path_finalizes_once_and_skips_copy(self, monkeypatch):
        actor, copied, finalized = self._make_step_actor(monkeypatch)
        ok, _ = actor._optimizer_step_with_buffer(None, [], None, do_grad_sync=False)
        assert ok
        assert copied == []
        assert finalized == [actor.actor_module]

    def test_metrics_report_ess_entry(self, monkeypatch):
        actor, _, _ = self._make_step_actor(monkeypatch)
        _, metrics = actor._optimizer_step_with_buffer(None, [], None, minibatch_idx=7, do_grad_sync=False)
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_idx"] == 7
        assert entry["minibatch_ess_ratio"] == 0.25
        assert entry["minibatch_ess_ratio_clipped"] == 0.2

    def test_ess_scaled_step_restores_nominal_lr(self, monkeypatch):
        scaling = ESSScalingConfig(enable=True, scaling_rule="sqrt", base_ess_ratio=0.5)
        actor, _, _ = self._make_step_actor(monkeypatch, ess_scaling=scaling)
        actor._optimizer_step_with_buffer(None, [], None, do_grad_sync=False)
        # ess_ratio 0.25 / base 0.5 = 0.5 -> sqrt rule steps at lr * sqrt(0.5)...
        assert actor.actor_optimizer.stepped_lrs == pytest.approx([1e-6 * 0.5**0.5])
        # ...and the nominal lr is restored afterwards.
        assert actor.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
