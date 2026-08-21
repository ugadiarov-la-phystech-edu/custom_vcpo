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
"""CPU wiring tests for the min-ESS brake tail shared by both per-traj paths.

_apply_ess_scale_and_step is where the measured ESS turns into an actual LR:
it scales the param groups for this step only, steps, and restores the nominal
LRs. Both entry points feed it — _optimizer_step_with_buffer (mbs=1, ESS from
compute_ess_info) and _ess_scaled_optimizer_step_packed (dynamic bsz, ESS from
compute_global_ess_from_log_weights) — and both must hand it the measurement's
sequence count, which is what separates "not measured" (no scaling) from
"measurement broke" (fail closed).

Run: pytest tests/workers/actor/test_ess_brake_wiring_on_cpu.py
"""

import math
from types import SimpleNamespace

import pytest

from verl.workers.actor import megatron_actor
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config.actor import ESSScalingConfig

NOMINAL_LR = 1e-6
LR_SCALE = 0.5
B = 528


class _FakeOptimizer:
    """Records the LR in force at the moment step() is called."""

    def __init__(self, lr=NOMINAL_LR, groups=1):
        self.param_groups = [{"lr": lr} for _ in range(groups)]
        self.stepped_lrs = []

    def step(self):
        self.stepped_lrs.append([float(pg["lr"]) for pg in self.param_groups])
        return True, 0.123, 0


def _actor(groups=1, **ess_kwargs):
    actor = MegatronPPOActor.__new__(MegatronPPOActor)
    actor.config = SimpleNamespace(
        ess_scaling=ESSScalingConfig(enable=True, min_ess=1.1, lr_scale=LR_SCALE, **ess_kwargs)
    )
    actor.actor_module = [object()]
    actor.actor_optimizer = _FakeOptimizer(groups=groups)
    return actor


def _step(actor, ess, count, ess_clipped=None):
    ok, metrics = actor._apply_ess_scale_and_step(
        ess, ess if ess_clipped is None else ess_clipped, 0.0, 0.0, minibatch_idx=0, ess_count=count
    )
    assert ok
    return actor.actor_optimizer.stepped_lrs[-1][0], metrics


class TestBrakeDecisions:
    def test_healthy_ess_steps_at_nominal_lr(self):
        actor = _actor()
        lr, _ = _step(actor, ess=400.0, count=B)
        assert lr == pytest.approx(NOMINAL_LR)

    def test_degenerate_ess_brakes(self):
        actor = _actor()
        lr, _ = _step(actor, ess=1.0, count=B)
        assert lr == pytest.approx(NOMINAL_LR * LR_SCALE)

    def test_unmeasured_ess_steps_at_nominal_lr(self):
        """count == 0: the path never filled the IS fields (e.g. records built
        without old log-probs). Nothing was measured, so nothing is braked."""
        actor = _actor()
        lr, _ = _step(actor, ess=0.0, count=0)
        assert lr == pytest.approx(NOMINAL_LR)

    @pytest.mark.parametrize("broken", [float("nan"), float("inf"), 0.0, -1.0])
    def test_broken_measurement_fails_closed(self, broken):
        """The old rule ran these at full lr — min(1.0, nan) == 1.0 and
        'ess == 0' was read as an empty batch — which is how the collapse
        steps of the 2026-08 replay run took unbraked updates."""
        actor = _actor()
        lr, _ = _step(actor, ess=broken, count=B)
        assert lr == pytest.approx(NOMINAL_LR * LR_SCALE)

    def test_missing_ess_is_treated_as_unmeasured(self):
        """compute_ess_info can return ess=None on a path that skipped it."""
        actor = _actor()
        lr, _ = _step(actor, ess=None, count=0)
        assert lr == pytest.approx(NOMINAL_LR)

    def test_use_clipped_selects_the_clipped_measurement(self):
        actor = _actor(use_clipped=True)
        lr, _ = _step(actor, ess=400.0, count=B, ess_clipped=1.0)
        assert lr == pytest.approx(NOMINAL_LR * LR_SCALE)

    def test_disabled_brake_never_scales(self):
        actor = MegatronPPOActor.__new__(MegatronPPOActor)
        actor.config = SimpleNamespace(ess_scaling=ESSScalingConfig(enable=False, min_ess=1.1, lr_scale=LR_SCALE))
        actor.actor_module = [object()]
        actor.actor_optimizer = _FakeOptimizer()
        lr, _ = _step(actor, ess=1.0, count=B)
        assert lr == pytest.approx(NOMINAL_LR)


class TestLrRestoration:
    @pytest.mark.parametrize("ess,count", [(1.0, B), (400.0, B), (float("nan"), B), (0.0, 0)])
    def test_nominal_lr_is_restored_after_the_step(self, ess, count):
        actor = _actor(groups=3)
        _step(actor, ess=ess, count=count)
        assert [pg["lr"] for pg in actor.actor_optimizer.param_groups] == [NOMINAL_LR] * 3

    def test_every_param_group_is_scaled(self):
        actor = _actor(groups=3)
        actor._apply_ess_scale_and_step(1.0, 1.0, 0.0, 0.0, minibatch_idx=0, ess_count=B)
        assert actor.actor_optimizer.stepped_lrs[-1] == [pytest.approx(NOMINAL_LR * LR_SCALE)] * 3

    def test_brake_applies_only_to_its_own_step(self):
        actor = _actor()
        _step(actor, ess=1.0, count=B)
        _step(actor, ess=400.0, count=B)
        first, second = (lrs[0] for lrs in actor.actor_optimizer.stepped_lrs)
        assert first == pytest.approx(NOMINAL_LR * LR_SCALE)
        assert second == pytest.approx(NOMINAL_LR)


class TestMetrics:
    def test_ess_entry_reports_the_scaled_lr(self):
        actor = _actor()
        lr, metrics = _step(actor, ess=1.0, count=B)
        entry = metrics["staleness/ess"][0]
        assert entry["minibatch_ess"] == 1.0
        assert entry["ess_scaled_lr"] == pytest.approx(lr)

    def test_nan_ess_is_reported_not_swallowed(self):
        actor = _actor()
        _, metrics = _step(actor, ess=float("nan"), count=B)
        assert math.isnan(metrics["staleness/ess"][0]["minibatch_ess"])


class TestEntryPointsPassCount:
    """Both callers must forward the count; without it a broken measurement
    falls back to the legacy 'ess == 0 means empty' reading and runs unbraked."""

    def test_mbs1_path_forwards_the_count(self, monkeypatch):
        actor = _actor()
        monkeypatch.setattr(
            megatron_actor,
            "compute_ess_info",
            lambda records, thr: {"ess": 0.0, "ess_ratio": 0.0, "ess_clipped": 0.0, "ess_ratio_clipped": 0.0,
                                  "count": B},
        )
        monkeypatch.setattr(megatron_actor, "finalize_model_grads_ignore_dp", lambda modules: None)

        ok, _ = actor._optimizer_step_with_buffer(None, [], None, do_grad_sync=False)
        assert ok
        assert actor.actor_optimizer.stepped_lrs[-1][0] == pytest.approx(NOMINAL_LR * LR_SCALE)

    def test_mbs1_path_without_a_measurement_steps_unbraked(self, monkeypatch):
        actor = _actor()
        monkeypatch.setattr(
            megatron_actor,
            "compute_ess_info",
            lambda records, thr: {"ess": 0.0, "ess_ratio": 0.0, "ess_clipped": 0.0, "ess_ratio_clipped": 0.0,
                                  "count": 0},
        )
        monkeypatch.setattr(megatron_actor, "finalize_model_grads_ignore_dp", lambda modules: None)

        actor._optimizer_step_with_buffer(None, [], None, do_grad_sync=False)
        assert actor.actor_optimizer.stepped_lrs[-1][0] == pytest.approx(NOMINAL_LR)

    def test_packed_path_forwards_the_count(self, monkeypatch):
        actor = _actor()
        monkeypatch.setattr(
            megatron_actor,
            "compute_global_ess_from_log_weights",
            lambda logs, thr, group=None: (1.0, 1.0 / B, 1.0, 1.0 / B, B),
        )
        ok, _ = actor._ess_scaled_optimizer_step_packed([-200.0] * B, 2.0, minibatch_idx=0)
        assert ok
        assert actor.actor_optimizer.stepped_lrs[-1][0] == pytest.approx(NOMINAL_LR * LR_SCALE)

    def test_packed_path_on_an_empty_batch_steps_unbraked(self, monkeypatch):
        actor = _actor()
        monkeypatch.setattr(
            megatron_actor,
            "compute_global_ess_from_log_weights",
            lambda logs, thr, group=None: (0.0, 0.0, 0.0, 0.0, 0),
        )
        actor._ess_scaled_optimizer_step_packed([], 2.0, minibatch_idx=0)
        assert actor.actor_optimizer.stepped_lrs[-1][0] == pytest.approx(NOMINAL_LR)
