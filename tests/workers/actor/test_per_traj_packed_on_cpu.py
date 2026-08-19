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

"""CPU tests for the packed (dynamic-batch-size) per-traj update path.

The real path only runs on multi-GPU Megatron; these tests pin the dispatch,
control flow, ESS wiring, and the packing-invariance contract with stubbed
forward/optimizer (same harness style as test_update_policy_per_traj_on_cpu):

- dispatch: use_dynamic_bsz=True routes update_policy_per_traj to the packed
  path; combined with grad_baselining=True it must refuse (OPOB needs
  per-trajectory gradient isolation);
- packed control flow: ONE forward_backward_batch per mini-batch with
  use_dynamic_bsz=True, the parity meta keys stamped
  (global_seq_mean_count=N, collect_seq_log_is=True), the schedule-level
  finalize left untouched (no disable_grad_finalize), skip_recompute
  required;
- ESS wiring: per-sequence log-IS sums flow from the micro-batch metrics into
  the max-shifted ESS, the min-ESS brake multiplies the LR by the constant
  lr_scale for the step (restored after) exactly when ESS <= min_ess, and
  the staleness/ess entry carries the 6-key contract — including the
  overflow regression (a +60 log-weight reads ESS = 1, not 0, so the braked
  LR is exactly lr * lr_scale, never zero);
- the n_rows*M/N rescale contract: mean-of-means over an arbitrary unequal
  packing, rescaled per micro-batch, equals the global per-sequence mean.

Run: pytest tests/workers/actor/test_per_traj_packed_on_cpu.py
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verl.workers.actor.megatron_actor as megatron_actor
from recipe.fully_async_policy.staleness_utils import TrajRecord, TrajRecordList
from verl import DataProto
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config.actor import ESSScalingConfig

TRAJ_UIDS = ["t0", "t1", "t2"]
ADVANTAGES = [1.5, -0.5, 0.0]
NOMINAL_LR = 1e-6
ORIGINAL_FINALIZE = object()


class _FakeChunk:
    def __init__(self):
        self.config = SimpleNamespace(finalize_model_grads_func=ORIGINAL_FINALIZE)
        self.zero_grad_buffer_calls = 0

    def zero_grad_buffer(self):
        self.zero_grad_buffer_calls += 1


class _FakeOptimizer:
    def __init__(self, lr=NOMINAL_LR):
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


def _make_minibatch(skip_recompute: bool = True) -> DataProto:
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
    data.meta_info["skip_recompute_old_log_prob"] = skip_recompute
    data.meta_info["rollout_corr_config"] = {"rollout_is_threshold": 2.0}
    return data


def _make_actor(ess_scaling=None, seq_log_is_per_microbatch=None) -> tuple[MegatronPPOActor, dict]:
    """Actor with a stubbed forward: returns two 'micro-batches' of metrics
    carrying the given per-row log-IS sums (append_to_dict-wrapped, as
    loss_func produces them)."""
    if seq_log_is_per_microbatch is None:
        seq_log_is_per_microbatch = [[0.0, 0.0], [0.0]]
    actor = MegatronPPOActor.__new__(MegatronPPOActor)
    actor.config = SimpleNamespace(
        use_dynamic_bsz=True,
        ppo_max_token_len_per_gpu=2048,
        ppo_micro_batch_size_per_gpu=1,
        ppo_mini_batch_size=len(TRAJ_UIDS),
        calculate_entropy=False,
        entropy_coeff=0,
        policy_loss={},
        loss_agg_mode="seq-mean-token-mean",
        megatron=SimpleNamespace(context_parallel_size=1),
        grad_baselining=SimpleNamespace(scope="group", norm_by_std=False),
        ess_scaling=ess_scaling or ESSScalingConfig(),
    )
    actor.actor_module = [_FakeChunk()]
    actor.actor_optimizer = _FakeOptimizer()

    calls = {"fbb_kwargs": [], "fbb_meta": []}

    def fake_forward_backward_batch(data, **kwargs):
        calls["fbb_kwargs"].append(kwargs)
        calls["fbb_meta"].append(dict(data.meta_info))
        outputs = []
        for rows in seq_log_is_per_microbatch:
            # loss_func stores the row list in stats; append_to_dict EXTENDS
            # list values, so the per-micro-batch metrics carry flat floats
            outputs.append([{"_ess/seq_log_is": list(rows), "actor/pg_loss": [0.0]}, None, None, None])
        return {"output": outputs, "indices": [[0, 1], [2]]}

    actor.forward_backward_batch = fake_forward_backward_batch
    return actor, calls


@pytest.fixture
def patched_env(monkeypatch):
    records = _make_records()

    def fake_compute_staleness_statistics(batch, minibatch_idx, rollout_is_threshold, use_old, epoch_idx=0):
        assert use_old is False, "packed path must not request old-log-prob record fields"
        return records, {}

    monkeypatch.setattr(megatron_actor, "compute_staleness_statistics", fake_compute_staleness_statistics)
    monkeypatch.setattr(megatron_actor, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    return {"records": records}


class TestDispatch:
    def test_dynamic_bsz_routes_to_packed_path(self, patched_env, monkeypatch):
        actor, calls = _make_actor()
        sentinel = {"routed": False}

        def fake_packed(dataloader):
            sentinel["routed"] = True
            return {}

        monkeypatch.setattr(actor, "_update_policy_per_traj_packed", fake_packed)
        actor.update_policy_per_traj([_make_minibatch()], grad_baselining=False)
        assert sentinel["routed"]

    def test_dynamic_bsz_with_opob_refuses(self, patched_env):
        actor, _ = _make_actor()
        with pytest.raises(AssertionError, match="grad_baselining"):
            actor.update_policy_per_traj([_make_minibatch()], grad_baselining=True)

    def test_skip_recompute_required(self, patched_env):
        actor, _ = _make_actor()
        with pytest.raises(AssertionError, match="skip_recompute_old_log_prob"):
            actor._update_policy_per_traj_packed([_make_minibatch(skip_recompute=False)])

    def test_context_parallel_refused(self, patched_env):
        actor, _ = _make_actor()
        actor.config.megatron = SimpleNamespace(context_parallel_size=2)
        with pytest.raises(AssertionError, match="context parallelism"):
            actor._update_policy_per_traj_packed([_make_minibatch()])

    def test_loss_agg_mode_guard(self, patched_env):
        actor, _ = _make_actor()
        actor.config.loss_agg_mode = "token-mean"
        with pytest.raises(AssertionError, match="seq-mean-token-mean"):
            actor._update_policy_per_traj_packed([_make_minibatch()])


class TestPackedControlFlow:
    def test_single_forward_backward_with_dynamic_bsz(self, patched_env):
        actor, calls = _make_actor()
        actor._update_policy_per_traj_packed([_make_minibatch()])
        assert len(calls["fbb_kwargs"]) == 1
        kwargs = calls["fbb_kwargs"][0]
        assert kwargs["use_dynamic_bsz"] is True
        assert kwargs["max_token_len"] == 2048  # budget * cp_size(=1)

    def test_parity_meta_keys_stamped(self, patched_env):
        actor, calls = _make_actor()
        actor._update_policy_per_traj_packed([_make_minibatch()])
        meta = calls["fbb_meta"][0]
        assert meta["global_seq_mean_count"] == len(TRAJ_UIDS)
        assert meta["collect_seq_log_is"] is True

    def test_finalize_left_untouched(self, patched_env):
        """The packed path relies on the schedule's own finalize (DP sync
        included) — suppressing it would silently skip gradient sync."""
        actor, _ = _make_actor()
        actor._update_policy_per_traj_packed([_make_minibatch()])
        assert actor.actor_module[0].config.finalize_model_grads_func is ORIGINAL_FINALIZE

    def test_grad_zeroing_once_per_minibatch(self, patched_env):
        actor, _ = _make_actor()
        actor._update_policy_per_traj_packed([_make_minibatch()])
        assert actor.actor_module[0].zero_grad_buffer_calls == 1
        # minibatch start + final cleanup
        assert actor.actor_optimizer.zero_grad_calls == 2

    def test_metrics_carry_records_and_grad_info(self, patched_env):
        actor, _ = _make_actor()
        metrics = actor._update_policy_per_traj_packed([_make_minibatch()])
        assert len(metrics["actor/local_traj_records"]) == len(TRAJ_UIDS)
        for rec in metrics["actor/local_traj_records"]:
            assert rec["grad_norm"] is None  # per-traj grad norms are mbs=1/OPOB-only
        (grad_info,) = metrics["actor/minibatch_grad_info"]
        assert grad_info["minibatch_idx"] == 0

    def test_seq_log_is_key_consumed_not_leaked(self, patched_env):
        actor, _ = _make_actor()
        metrics = actor._update_policy_per_traj_packed([_make_minibatch()])
        assert "_ess/seq_log_is" not in metrics


class TestPackedEssStep:
    """ESS wiring end to end: the stubbed forward emits crafted log-weights;
    the step must scale by the max-shifted ESS and restore the LR."""

    def test_entry_contract_and_values(self, patched_env):
        # weights e^0=1 x3 -> ess = 3.0, ess_ratio = 1.0 exactly
        actor, _ = _make_actor(seq_log_is_per_microbatch=[[0.0, 0.0], [0.0]])
        metrics = actor._update_policy_per_traj_packed([_make_minibatch()])
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_idx"] == 0
        assert entry["minibatch_ess"] == pytest.approx(3.0)
        assert entry["minibatch_ess_ratio"] == pytest.approx(1.0)
        assert entry["minibatch_ess_ratio_clipped"] == pytest.approx(1.0)
        assert entry["ess_scaled_lr"] == pytest.approx(NOMINAL_LR)
        assert "base_ess_ratio" not in entry  # removed with the auto-base logic

    def test_braked_step_scales_and_restores_lr(self, patched_env):
        # log-weights [log(4), 0, 0] -> ESS = 36/18 = 2.0 <= min_ess 2.0
        # (inclusive boundary) -> lr * lr_scale, restored after the step
        scaling = ESSScalingConfig(enable=True, min_ess=2.0, lr_scale=0.5)
        actor, _ = _make_actor(ess_scaling=scaling, seq_log_is_per_microbatch=[[math.log(4.0), 0.0], [0.0]])
        actor._update_policy_per_traj_packed([_make_minibatch()])
        assert actor.actor_optimizer.stepped_lrs == pytest.approx([NOMINAL_LR * 0.5])
        assert actor.actor_optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)

    def test_above_threshold_runs_full_lr(self, patched_env):
        # ESS = 2.0 > min_ess 1.1 -> full nominal lr (no shaping by distance)
        scaling = ESSScalingConfig(enable=True, min_ess=1.1, lr_scale=0.5)
        actor, _ = _make_actor(ess_scaling=scaling, seq_log_is_per_microbatch=[[math.log(4.0), 0.0], [0.0]])
        actor._update_policy_per_traj_packed([_make_minibatch()])
        assert actor.actor_optimizer.stepped_lrs == pytest.approx([NOMINAL_LR])

    def test_overflow_dominant_weight_brakes_at_lr_scale_never_zero(self, patched_env):
        """Regression: a +60 log-weight made the raw-space ESS read 0.0 and
        the old brake step at lr=0. The max-shifted ESS reads exactly 1 (the
        structural floor), which the min-ESS rule brakes at exactly
        lr * lr_scale — never 0."""
        scaling = ESSScalingConfig(enable=True, min_ess=1.1, lr_scale=0.5)
        actor, _ = _make_actor(ess_scaling=scaling, seq_log_is_per_microbatch=[[60.0, 0.0], [0.0]])
        metrics = actor._update_policy_per_traj_packed([_make_minibatch()])
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_ess"] == pytest.approx(1.0, rel=1e-9)
        assert actor.actor_optimizer.stepped_lrs == pytest.approx([NOMINAL_LR * 0.5])
        assert actor.actor_optimizer.stepped_lrs[0] > 0.0

    def test_deep_underflow_runs_unbraked(self, patched_env):
        # all log-weights -200 but EQUAL: max-shifted ESS = 3 > min_ess
        scaling = ESSScalingConfig(enable=True, min_ess=1.1, lr_scale=0.5)
        actor, _ = _make_actor(ess_scaling=scaling, seq_log_is_per_microbatch=[[-200.0, -200.0], [-200.0]])
        actor._update_policy_per_traj_packed([_make_minibatch()])
        assert actor.actor_optimizer.stepped_lrs == pytest.approx([NOMINAL_LR])

    def test_use_clipped_selects_clipped_ess(self, patched_env):
        # rollout_is_threshold=2.0: log-weights [log(4), 0, 0] give
        # unclipped ESS = 2.0 (brakes at min_ess=2.0) but clipped weights
        # [2, 1, 1] give ESS = 16/6 ~= 2.667 > 2.0 -> full lr.
        scaling = ESSScalingConfig(enable=True, min_ess=2.0, lr_scale=0.5, use_clipped=True)
        actor, _ = _make_actor(ess_scaling=scaling, seq_log_is_per_microbatch=[[math.log(4.0), 0.0], [0.0]])
        metrics = actor._update_policy_per_traj_packed([_make_minibatch()])
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_ess_clipped"] == pytest.approx(16.0 / 6.0)
        assert actor.actor_optimizer.stepped_lrs == pytest.approx([NOMINAL_LR])


class TestPackingInvarianceContract:
    """The loss_func rescale (policy_loss * n_rows * M / N after the schedule's
    1/M) must make the summed loss the GLOBAL per-sequence mean for any
    packing. This pins the formula the implementation applies in
    megatron_actor's loss_func under meta_info['global_seq_mean_count']."""

    @staticmethod
    def _schedule_total(per_seq_losses, partition):
        """Emulate: per micro-batch, seq-mean -> * n_rows*M/N (loss_func) ->
        * 1/M (Megatron schedule) -> sum over micro-batches."""
        n_total = len(per_seq_losses)
        n_micro = len(partition)
        total = 0.0
        for rows in partition:
            mb = [per_seq_losses[i] for i in rows]
            seq_mean = sum(mb) / len(mb)
            rescaled = seq_mean * (len(mb) * n_micro / n_total)
            total += rescaled / n_micro
        return total

    def test_equal_and_unequal_packings_agree_with_global_mean(self):
        losses = [3.0, -1.0, 0.5, 2.5, -0.25, 4.0]
        global_mean = sum(losses) / len(losses)
        partitions = [
            [[0], [1], [2], [3], [4], [5]],  # mbs=1 reference
            [[0, 1, 2], [3, 4, 5]],  # equal packing
            [[0, 1, 2, 3, 4], [5]],  # extreme skew
            [[5, 0], [4, 1, 3], [2]],  # unequal + reordered
        ]
        for partition in partitions:
            assert self._schedule_total(losses, partition) == pytest.approx(global_mean, rel=1e-12)

    def test_without_rescale_unequal_packing_diverges(self):
        """Sanity: the naive mean-of-means (what plain seq-mean-token-mean
        would give) is NOT the global mean under unequal packing — the rescale
        is load-bearing, not decorative."""
        losses = [3.0, -1.0, 0.5, 2.5, -0.25, 4.0]
        global_mean = sum(losses) / len(losses)
        partition = [[0, 1, 2, 3, 4], [5]]
        naive = 0.0
        for rows in partition:
            mb = [losses[i] for i in rows]
            naive += (sum(mb) / len(mb)) / len(partition)
        assert naive != pytest.approx(global_mean, rel=1e-6)
