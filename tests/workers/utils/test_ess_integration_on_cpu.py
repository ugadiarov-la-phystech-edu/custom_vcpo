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

"""CPU tests for the log-space ESS as wired into the per-traj brake.

Covers the production incident these changes fix: on 2026-08-21 the mbs=1
replay arm read ESS = 0.00 at steps 345/346/348 and the ESS brake scaled the
effective learning rate to 2.7e-18 / 1.3e-25 / 5.0e-11 — silent no-op updates.
The raw-space arithmetic formed w = exp(sum of masked token log-ratios) in
fp32, which flushes to 0 below ~-87 (ESS -> 0 -> lr -> 0) and to inf above
~88.7 (ESS -> NaN -> brake disabled).

Run: pytest tests/workers/utils/test_ess_integration_on_cpu.py
"""

import math

import numpy as np
import pytest

from verl.workers.actor.megatron_actor import compute_ess_lr_scale
from verl.workers.utils.ess import ess_from_log_weights

B = 528  # 33 prompts x 16 responses, the production mini-batch
BASE = 0.006113  # measured on-policy rho_on for that setup


def raw_space_ess(seq_log_is, eps: float = 1e-8):
    """The arithmetic this change replaces, incl. its fp32 exp()."""
    import torch

    weights = [float(torch.exp(torch.tensor(s, dtype=torch.float32))) for s in seq_log_is]
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    ess = sw**2 / (sw2 + eps)
    return ess, ess / len(weights)


def lr_multiplier(ess_ratio: float, trigger: float | None = 0.33333) -> float:
    """Effective lr / nominal lr under the production brake (sqrt rule)."""
    return compute_ess_lr_scale(ess_ratio, BASE, trigger) ** 0.5


class TestEquivalenceInHealthyRange:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_raw_space_when_nothing_saturates(self, seed):
        rng = np.random.default_rng(seed)
        logs = rng.uniform(-10.0, 2.0, size=64).tolist()
        ref_ess, ref_ratio = raw_space_ess(logs)
        ess, ratio, _, _, n = ess_from_log_weights(logs)
        assert n == 64
        assert ess == pytest.approx(ref_ess, rel=1e-5)
        assert ratio == pytest.approx(ref_ratio, rel=1e-5)

    def test_clipping_matches_raw_space(self):
        logs = [1.5, 0.4, -0.3, -2.0, 0.9]
        threshold = 2.0
        import torch

        clipped = [min(float(torch.exp(torch.tensor(s, dtype=torch.float32))), threshold) for s in logs]
        sw, sw2 = sum(clipped), sum(w * w for w in clipped)
        ref_ess = sw**2 / sw2
        _, _, ess_c, ratio_c, _ = ess_from_log_weights(logs, rollout_is_threshold=threshold)
        assert ess_c == pytest.approx(ref_ess, rel=1e-5)
        assert ratio_c == pytest.approx(ref_ess / len(logs), rel=1e-5)


class TestProductionIncident:
    def test_underflowing_batch_no_longer_zeroes_the_lr(self):
        """Every weight below the fp32 exp floor: the failure that produced
        lr = 1.3e-25 at step 346."""
        logs = [-200.0] * B

        old_ess, old_ratio = raw_space_ess(logs)
        assert old_ess == 0.0  # every exp() flushed to zero
        assert lr_multiplier(old_ratio) == pytest.approx(0.0, abs=1e-30)

        ess, ratio, _, _, n = ess_from_log_weights(logs)
        assert n == B
        assert ess == pytest.approx(float(B))  # identical weights -> ESS = B
        assert ratio == pytest.approx(1.0)
        assert lr_multiplier(ratio) == 1.0  # uniform weights are healthy

    def test_total_domination_floors_at_one_not_zero(self):
        """One sequence dominating by more than the fp range: ESS must read 1
        (the structural floor), not 0."""
        logs = [0.0] + [-300.0] * (B - 1)

        old_ess, old_ratio = raw_space_ess(logs)
        ess, ratio, _, _, _ = ess_from_log_weights(logs)

        assert ess == pytest.approx(1.0, abs=1e-12)
        assert ratio == pytest.approx(1.0 / B)
        # the brake engages hard, but to the floor value, never to zero
        assert lr_multiplier(ratio) == pytest.approx(math.sqrt((1.0 / B) / BASE), rel=1e-9)
        assert lr_multiplier(ratio) > 0.55
        assert old_ess == pytest.approx(ess, rel=1e-6)  # raw space happens to cope here

    def test_overflowing_batch_no_longer_disables_the_brake(self):
        """Mirror failure: a log-weight above ~88.7 makes fp32 exp() inf, so
        the raw formula returns NaN and min(1.0, nan) silently returns 1.0 —
        the brake switches itself off exactly when one sequence dominates."""
        logs = [200.0] + [0.0] * (B - 1)  # production batch size, one runaway weight

        old_ess, old_ratio = raw_space_ess(logs)
        assert math.isnan(old_ess)
        assert compute_ess_lr_scale(old_ratio, BASE, 0.33333) == 1.0  # brake off

        ess, ratio, _, _, _ = ess_from_log_weights(logs)
        assert math.isfinite(ess)
        assert ess == pytest.approx(1.0, abs=1e-9)  # one weight dominates -> floor
        assert ratio == pytest.approx(1.0 / B)
        # ratio/base = 0.31 < trigger 0.33333, so the brake engages as intended
        assert lr_multiplier(ratio) == pytest.approx(math.sqrt((1.0 / B) / BASE), rel=1e-9)
        assert lr_multiplier(ratio) < 1.0


class TestBrakeFloor:
    @pytest.mark.parametrize("seed", range(6))
    def test_multiplier_never_below_structural_floor(self, seed):
        """Whatever the weights, ESS >= 1 => the sqrt brake cannot go below
        sqrt(1/(B*base)). This is what makes lr ~ 1e-25 unreachable."""
        rng = np.random.default_rng(seed)
        floor = math.sqrt((1.0 / B) / BASE)
        for scale in (1.0, 50.0, 500.0):
            logs = (rng.normal(0.0, scale, size=B)).tolist()
            _, ratio, _, _, _ = ess_from_log_weights(logs)
            assert ratio >= 1.0 / B - 1e-12
            assert lr_multiplier(ratio) >= floor - 1e-12

    def test_non_finite_ratio_runs_at_nominal_lr(self):
        for bad in (float("nan"), float("inf")):
            assert compute_ess_lr_scale(bad, BASE, 0.33333) == 1.0
            assert compute_ess_lr_scale(bad, BASE, None) == 1.0


class TestStalenessUtilsWiring:
    """The two functions that feed the brake in production."""

    @staticmethod
    def _record():
        from recipe.fully_async_policy.staleness_utils import TrajRecord

        return TrajRecord(
            uid="t0",
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
            advantage_scalar=1.0,
            reward_scalar=1.0,
        )

    def test_compute_is_info_stores_the_log_sum(self):
        import torch

        from recipe.fully_async_policy.staleness_utils import compute_is_info

        old_lp = torch.tensor([-0.5, -1.0, -2.0, -0.25])
        rollout_lp = torch.tensor([-0.75, -0.5, -1.0, -0.25])
        mask = torch.tensor([1, 1, 1, 0])
        expected = float(((old_lp - rollout_lp) * mask).sum())

        rec = compute_is_info(self._record(), rollout_lp, old_lp, mask, 2.0)
        assert rec.rollout_seq_log_is == pytest.approx(expected, rel=1e-6)
        # consistent with the exp'd field wherever that one is representable
        assert math.exp(rec.rollout_seq_log_is) == pytest.approx(rec.rollout_seq_is, rel=1e-5)

    def test_compute_is_info_log_survives_where_exp_saturates(self):
        import torch

        from recipe.fully_async_policy.staleness_utils import compute_is_info

        old_lp = torch.tensor([-100.0, -100.0])
        rollout_lp = torch.tensor([0.0, 0.0])
        mask = torch.tensor([1, 1])

        rec = compute_is_info(self._record(), rollout_lp, old_lp, mask, 2.0)
        assert rec.rollout_seq_is == 0.0  # fp32 exp underflowed, as before
        assert rec.rollout_seq_log_is == pytest.approx(-200.0)  # log value intact

    def _ess_info(self, monkeypatch, records, threshold=2.0):
        from recipe.fully_async_policy import staleness_utils as su

        monkeypatch.setattr(su.mpu, "get_data_parallel_group", lambda **kw: None)
        monkeypatch.setattr(su.mpu, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(su.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(su, "allgather_dict_into_list", lambda local, group=None: local)
        return su.compute_ess_info(records, threshold)

    def test_compute_ess_info_uses_log_field(self, monkeypatch):
        """The end-to-end production path on the batch that broke it."""
        records = []
        for _ in range(B):
            r = self._record()
            r.rollout_seq_log_is = -200.0
            r.rollout_seq_is = 0.0  # what fp32 exp() produced
            records.append(r)

        info = self._ess_info(monkeypatch, records)
        assert info["ess"] == pytest.approx(float(B))  # identical weights
        assert info["ess_ratio"] == pytest.approx(1.0)
        assert compute_ess_lr_scale(info["ess_ratio"], BASE, 0.33333) == 1.0

    def test_compute_ess_info_falls_back_to_exp_field(self, monkeypatch):
        """Records predating the log field still measure correctly."""
        weights = [3.0, 0.2, 1.5, 0.7]
        records = []
        for w in weights:
            r = self._record()
            r.rollout_seq_is = w  # rollout_seq_log_is stays None
            records.append(r)

        info = self._ess_info(monkeypatch, records, threshold=None)
        sw, sw2 = sum(weights), sum(w * w for w in weights)
        assert info["ess"] == pytest.approx(sw**2 / sw2, rel=1e-9)

    def test_compute_ess_info_skips_unmeasured_records(self, monkeypatch):
        good = self._record()
        good.rollout_seq_log_is = 0.0
        missing = self._record()  # both fields None
        nonfinite = self._record()
        nonfinite.rollout_seq_log_is = float("nan")

        info = self._ess_info(monkeypatch, [good, missing, nonfinite])
        assert info["ess"] == pytest.approx(1.0)  # only one usable record
        assert info["ess_ratio"] == pytest.approx(1.0)

    def test_compute_ess_info_empty_batch(self, monkeypatch):
        info = self._ess_info(monkeypatch, [])
        assert info == {"ess": 0.0, "ess_ratio": 0.0, "ess_clipped": 0.0, "ess_ratio_clipped": 0.0}
