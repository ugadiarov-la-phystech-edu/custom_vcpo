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
"""Unit tests for the dynamic ESS-base estimator:
- EssBaseEstimator pooling gate, EMA folds, clamps, length correction,
  seed/warm-start semantics, and checkpoint state round-trip
- winsorization keeps a single heavy-tail outlier from crushing the reference
- the s->0 staleness regression recovers known sigma^2_num / delta^2 from
  synthetic bucket data
- staleness_utils._compute_base_estimator_payloads builds cohort moments and
  robust staleness buckets from TrajRecord dicts
- trainer wiring: the replay_is_new stamp survives batch reordering, the
  _update_ess_base_estimator glue seeds/refreshes/logs correctly (and
  degrades to the seed on payload-free entries), and the estimator state
  rides the replay checkpoint dict
- compute_ess_info end-to-end: TrajRecord -> asdict -> payloads (dist stubs)

Run: pytest recipe/fully_async_policy/unittest/test_ess_base_estimator_on_cpu.py
"""

import math
import random
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import ray.cloudpickle
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import recipe.fully_async_policy.fully_async_trainer as fully_async_trainer_module
import recipe.fully_async_policy.staleness_utils as staleness_utils
from recipe.fully_async_policy.ess_base_estimator import EssBaseEstimator, _ess_ratio
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from recipe.fully_async_policy.replay_buffer import ReplayBuffer
from verl.protocol import DataProto

# The trainer is a @ray.remote ActorClass wrapper; tests need the plain class.
FullyAsyncTrainer = (
    _TrainerActor.__ray_metadata__.modified_class
    if hasattr(_TrainerActor, "__ray_metadata__")
    else _TrainerActor
)


def _uniform_cohort(n, w=1.0, length=1000.0):
    """Cohort payload with n identical weights (ESS ratio exactly 1)."""
    return {
        "n": n,
        "w_sum": n * w,
        "w_sq_sum": n * w * w,
        "w_sum_raw": n * w,
        "w_sq_sum_raw": n * w * w,
        "len_sum": n * length,
    }


def _cohort_from_weights(weights, length=1000.0):
    raw_sum = sum(weights)
    raw_sq = sum(w * w for w in weights)
    winsorized = list(weights)
    if len(winsorized) >= 2:
        top_idx = max(range(len(winsorized)), key=winsorized.__getitem__)
        second = max(w for i, w in enumerate(winsorized) if i != top_idx)
        winsorized[top_idx] = min(winsorized[top_idx], second)
    return {
        "n": len(weights),
        "w_sum": sum(winsorized),
        "w_sq_sum": sum(w * w for w in winsorized),
        "w_sum_raw": raw_sum,
        "w_sq_sum_raw": raw_sq,
        "len_sum": len(weights) * length,
    }


class TestPoolingAndEma:
    def test_no_base_before_min_seqs(self):
        est = EssBaseEstimator(min_seqs=64)
        est.observe(_uniform_cohort(32), None, 1000.0)
        assert est.current_base() is None
        assert est.diagnostics()["replay/ess_base_from_estimator"] == 0.0
        assert est.diagnostics()["replay/ess_base_pending_n"] == 32.0

    def test_fold_at_min_seqs_and_pending_reset(self):
        est = EssBaseEstimator(min_seqs=64, clamp_min=0.002, clamp_max=1.0)
        est.observe(_uniform_cohort(32), None, 1000.0)
        est.observe(_uniform_cohort(32), None, 1000.0)
        assert est.pend_n == 0
        base = est.current_base()
        assert base == pytest.approx(1.0)
        assert est.diagnostics()["replay/ess_base_from_estimator"] == 1.0

    def test_seed_used_until_first_fold_and_set_once(self):
        est = EssBaseEstimator(min_seqs=64)
        est.seed(0.5)
        est.seed(0.9)  # set-once: ignored
        assert est.current_base() == pytest.approx(0.5)
        est.observe(_uniform_cohort(64), None, 1000.0)
        # after the fold the estimator value (1.0) is ceilinged by the seed
        assert est.current_base() == pytest.approx(0.5)
        assert est.diagnostics()["replay/ess_base_clamped_high"] == 1.0

    def test_ema_moves_toward_new_regime(self):
        est = EssBaseEstimator(min_seqs=64, beta=0.5, clamp_min=1e-6, clamp_max=1.0)
        est.observe(_uniform_cohort(64), None, 1000.0)
        first = est.current_base()
        # second fold: heavy-tailed cohort with much lower ESS
        weights = [1.0] * 63 + [50.0]
        est2_expected_ess = _ess_ratio(sum([1.0] * 63 + [1.0]), sum([1.0] * 63 + [1.0]), 64)
        assert est2_expected_ess == pytest.approx(1.0)
        est.observe(_cohort_from_weights(weights), None, 1000.0)
        second = est.current_base()
        # winsor caps the 50x outlier to 1.0 -> cohort ESS 1.0 -> base stays
        assert second == pytest.approx(first)

    def test_clamp_min_floors_the_base(self):
        est = EssBaseEstimator(min_seqs=4, clamp_min=0.01, clamp_max=1.0, winsor_top1=False)
        # one enormous weight among 200: raw ESS ratio collapses toward
        # 1/n = 0.005, below the clamp floor
        est.observe(_cohort_from_weights([1.0] * 199 + [1e6]), None, 1000.0)
        base = est.current_base()
        assert base == pytest.approx(0.01)
        d = est.diagnostics()
        assert d["replay/ess_base_clamped_low"] == 1.0
        assert d["replay/ess_base_unclamped"] < 0.01


class TestWinsorization:
    def test_winsor_protects_reference_from_single_outlier(self):
        weights = [1.0] * 63 + [4000.0]
        cohort = _cohort_from_weights(weights)
        winsorized = EssBaseEstimator(min_seqs=4, clamp_min=1e-6, clamp_max=1.0, winsor_top1=True)
        raw = EssBaseEstimator(min_seqs=4, clamp_min=1e-6, clamp_max=1.0, winsor_top1=False)
        winsorized.observe(cohort, None, 1000.0)
        raw.observe(cohort, None, 1000.0)
        assert winsorized.current_base() > 0.9  # outlier capped to 1.0
        assert raw.current_base() < 0.05  # ESS crushed toward 1/n
        d = winsorized.diagnostics()
        # both variants are always logged for the raw-vs-winsorized gap
        assert d["replay/ess_base_cohort_ess"] > 0.9
        assert d["replay/ess_base_cohort_ess_raw"] < 0.05


class TestLengthCorrection:
    def test_reference_decays_with_minibatch_length(self):
        sigma2 = 1e-4
        length_fresh = 2000.0
        ess_fresh = math.exp(-sigma2 * length_fresh)
        # cohort of identical weights cannot express intra-cohort variance, so
        # build moments with the target pooled ESS directly: choose w's with
        # (sum w)^2/(n sum w^2) = ess_fresh via two weight values
        n = 64
        # solve with half the weights at 1 and half at a: ess(a) is monotone
        lo, hi = 1.0, 100.0
        for _ in range(200):
            a = 0.5 * (lo + hi)
            ws = [1.0] * (n // 2) + [a] * (n // 2)
            e = _ess_ratio(sum(ws), sum(w * w for w in ws), n)
            if e > ess_fresh:
                lo = a
            else:
                hi = a
        est = EssBaseEstimator(
            min_seqs=n, clamp_min=1e-8, clamp_max=1.0, length_correction=True, winsor_top1=False
        )
        est.observe(_cohort_from_weights(ws, length=length_fresh), None, length_fresh)
        base_at_fresh_len = est.current_base()
        assert base_at_fresh_len == pytest.approx(ess_fresh, rel=1e-3)
        # now the minibatch length doubles: base must decay as exp(-sigma2*L)
        est.observe(None, None, 2 * length_fresh)
        base_at_double_len = est.current_base()
        assert base_at_double_len == pytest.approx(math.exp(-sigma2 * 2 * length_fresh), rel=1e-3)
        assert est.diagnostics()["replay/ess_base_len_correction"] == pytest.approx(
            base_at_double_len / ess_fresh, rel=1e-3
        )


class TestStalenessRegression:
    def test_recovers_sigma2_and_delta2(self):
        sigma2_num = 2e-5
        delta2 = 1e-5
        mean_len = 3000.0
        est = EssBaseEstimator(window=100)
        for _ in range(5):
            buckets = [
                {
                    "s": float(s),
                    "n": 50,
                    "mean_len": mean_len,
                    "var_log_w": (sigma2_num + delta2 * s) * mean_len,
                }
                for s in (1, 4, 8, 16, 32)
            ]
            est.observe(None, buckets, mean_len)
        d = est.diagnostics()
        assert d["replay/ess_base_sigma2_num"] == pytest.approx(sigma2_num, rel=1e-6)
        assert d["replay/ess_base_delta2"] == pytest.approx(delta2, rel=1e-6)
        assert d["replay/ess_base_reg_r2"] == pytest.approx(1.0, abs=1e-9)
        assert d["replay/ess_base_rho_on_est"] == pytest.approx(math.exp(-sigma2_num * mean_len), rel=1e-6)

    def test_insufficient_buckets_yield_no_regression(self):
        est = EssBaseEstimator()
        est.observe(None, [{"s": 1.0, "n": 50, "mean_len": 1000.0, "var_log_w": 0.1}], 1000.0)
        d = est.diagnostics()
        assert "replay/ess_base_sigma2_num" not in d


class TestStateRoundTrip:
    def test_checkpoint_round_trip(self):
        rng = random.Random(0)
        est = EssBaseEstimator(min_seqs=32, beta=0.1, clamp_min=0.001, clamp_max=0.8)
        est.seed(0.4)
        for _ in range(7):
            weights = [math.exp(rng.gauss(0, 0.5)) for _ in range(20)]
            buckets = [
                {"s": float(s), "n": 30, "mean_len": 2000.0, "var_log_w": 0.05 + 0.01 * s} for s in (1, 5, 9)
            ]
            est.observe(_cohort_from_weights(weights, length=2000.0), buckets, 2500.0)
        base_before = est.current_base()
        state = est.state_dict()
        restored = EssBaseEstimator(min_seqs=32, beta=0.1, clamp_min=0.001, clamp_max=0.8)
        restored.load_state_dict(state)
        # regression diagnostics come from observe(); compare via a no-op observe
        est.observe(None, None, None)
        restored.observe(None, None, None)
        assert restored.current_base() == pytest.approx(base_before)
        assert restored.diagnostics() == pytest.approx(est.diagnostics())

    def test_from_config_defaults_and_nulls(self):
        est = EssBaseEstimator.from_config({"mode": "new_cohort", "clamp_max": None, "beta": None})
        assert est.beta == 0.05
        assert est.clamp_max is None
        est2 = EssBaseEstimator.from_config({"clamp_max": 0.3, "min_seqs": 16})
        assert est2.clamp_max == 0.3
        assert est2.min_seqs == 16


class TestObserveEntries:
    def test_extracts_first_complete_payload(self):
        est = EssBaseEstimator(min_seqs=4, clamp_min=1e-6, clamp_max=1.0)
        entries = [
            {"minibatch_ess_ratio": 0.5},  # entry without payloads (older format)
            {
                "new_cohort": _uniform_cohort(8),
                "staleness_buckets": [],
                "minibatch_mean_len": 1234.0,
            },
        ]
        est.observe_entries(entries)
        assert est.last_mb_len == 1234.0
        assert est.current_base() == pytest.approx(1.0)


class TestPayloadComputation:
    """staleness_utils._compute_base_estimator_payloads from record dicts."""

    def _records(self):
        records = []
        rng = random.Random(1)
        for i in range(40):
            is_new = i < 12
            staleness = 1 if is_new else rng.choice([8, 16, 24])
            records.append(
                {
                    "rollout_seq_is": math.exp(rng.gauss(0.0, 0.3)),
                    "response_length": 1000 + 10 * i,
                    "is_new": is_new,
                    "trainer_param_version": 100,
                    "param_version_start": 100 - staleness,
                }
            )
        return records

    def test_payload_shapes_and_moments(self):
        from recipe.fully_async_policy.staleness_utils import _compute_base_estimator_payloads

        records = self._records()
        payloads = _compute_base_estimator_payloads(records)
        cohort = payloads["new_cohort"]
        assert cohort["n"] == 12
        fresh_w = [r["rollout_seq_is"] for r in records if r["is_new"]]
        assert cohort["w_sum_raw"] == pytest.approx(sum(fresh_w))
        assert cohort["w_sq_sum_raw"] == pytest.approx(sum(w * w for w in fresh_w))
        # winsorized sums cap only the top weight
        top = max(fresh_w)
        second = max(w for w in fresh_w if w != top)
        assert cohort["w_sum"] == pytest.approx(sum(fresh_w) - top + second)
        assert payloads["minibatch_mean_len"] == pytest.approx(
            sum(r["response_length"] for r in records) / len(records)
        )
        buckets = payloads["staleness_buckets"]
        assert {b["s"] for b in buckets} <= {1.0, 8.0, 16.0, 24.0}
        for b in buckets:
            assert b["n"] >= 4
            assert b["var_log_w"] >= 0.0

    def test_no_new_records_yield_none_cohort(self):
        from recipe.fully_async_policy.staleness_utils import _compute_base_estimator_payloads

        records = self._records()
        for r in records:
            r["is_new"] = None
        payloads = _compute_base_estimator_payloads(records)
        assert payloads["new_cohort"] is None

    def test_zero_and_missing_weights_skipped(self):
        from recipe.fully_async_policy.staleness_utils import _compute_base_estimator_payloads

        records = [
            {"rollout_seq_is": None, "response_length": 100, "is_new": True},
            {
                "rollout_seq_is": 0.0,
                "response_length": 100,
                "is_new": True,
                "trainer_param_version": 10,
                "param_version_start": 9,
            },
        ]
        payloads = _compute_base_estimator_payloads(records)
        # the zero-weight record contributes to cohort moments (w=0) but not
        # to log-weight buckets; the None-weight record is skipped entirely
        assert payloads["new_cohort"]["n"] == 1
        assert payloads["staleness_buckets"] == []


# --------------------------------------------------------------- trainer glue


def _trainer_config():
    return OmegaConf.create(
        {
            "trainer": {"balance_batch": False},
            "algorithm": {"rollout_correction": None},
            "actor_rollout_ref": {
                "rollout": {"temperature": 1.0, "n": 2, "multi_turn": {"enable": False}},
                "actor": {"grad_baselining": {"enable": False}, "update_policy_per_traj": True},
            },
        }
    )


def _bare_trainer(estimator=None, ess_base=None):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.config = _trainer_config()
    t.tokenizer = None
    t.current_param_version = 0
    t.replay_ess_auto_base = True
    t.replay_ess_base = ess_base
    t.ess_base_estimator = estimator
    return t


class TestReplayIsNewStamp:
    """_build_replay_batch marks exactly the rows of unseen groups, by group
    uid membership, regardless of how the assembler (re)orders rows."""

    def _entry(self, uid, is_new, n=2):
        full_batch = SimpleNamespace(non_tensor_batch={"uid": np.array([uid] * n, dtype=object)})
        return SimpleNamespace(is_new=is_new, sample=SimpleNamespace(full_batch=full_batch))

    def _assembled_batch(self, row_uids):
        n = len(row_uids)
        return DataProto(
            batch=TensorDict({"response_mask": torch.ones(n, 4)}, batch_size=[n]),
            non_tensor_batch={
                "uid": np.array(row_uids, dtype=object),
                "advantage_scalar": np.zeros(n, dtype=np.float32),
                "reward_scalar": np.ones(n, dtype=np.float32),
            },
            meta_info={},
        )

    def test_stamp_follows_uid_membership_under_reordering(self, monkeypatch):
        entries = [
            self._entry("gA", is_new=True),
            self._entry("gB", is_new=False),
            self._entry("gC", is_new=True),
        ]
        # assembler interleaves groups (balance_batch-style reordering)
        row_uids = ["gB", "gA", "gC", "gB", "gC", "gA"]
        monkeypatch.setattr(
            fully_async_trainer_module,
            "assemble_batch_from_rollout_samples",
            lambda samples, tokenizer, config, balance: self._assembled_batch(row_uids),
        )
        trainer = _bare_trainer(ess_base=0.02)
        batch = trainer._build_replay_batch(entries)
        stamped = batch.non_tensor_batch["replay_is_new"]
        assert stamped.tolist() == [uid in {"gA", "gC"} for uid in row_uids]
        # the override rides along for the actor
        assert batch.meta_info["ess_base_override"] == pytest.approx(0.02)

    def test_all_replayed_minibatch_stamps_nothing(self, monkeypatch):
        entries = [self._entry("gA", is_new=False), self._entry("gB", is_new=False)]
        row_uids = ["gA", "gA", "gB", "gB"]
        monkeypatch.setattr(
            fully_async_trainer_module,
            "assemble_batch_from_rollout_samples",
            lambda samples, tokenizer, config, balance: self._assembled_batch(row_uids),
        )
        batch = _bare_trainer()._build_replay_batch(entries)
        assert not batch.non_tensor_batch["replay_is_new"].any()


class TestUpdateEssBaseEstimatorGlue:
    def _entry_with_payloads(self, weights, mb_len=1000.0):
        return {
            "minibatch_ess_ratio": 0.5,
            "base_ess_ratio": None,
            "new_cohort": _cohort_from_weights(weights, length=mb_len),
            "staleness_buckets": [],
            "minibatch_mean_len": mb_len,
        }

    def test_seeds_refreshes_base_and_logs_diagnostics(self):
        estimator = EssBaseEstimator(min_seqs=4, clamp_min=1e-6, clamp_max=None)
        trainer = _bare_trainer(estimator=estimator, ess_base=0.03)  # first-update capture
        # heavy-tailed fresh cohort: TWO dominant weights (winsor caps only
        # one) among 100 -> ESS ratio ~2/100 = 0.02, below the 0.03 ceiling
        weights = [1.0] * 98 + [1e4, 1e4]
        metrics = {"staleness/ess": [self._entry_with_payloads(weights)]}
        trainer._update_ess_base_estimator(metrics)
        assert estimator.seed_base == pytest.approx(0.03)
        expected = _ess_ratio(sum(weights), sum(w * w for w in weights), len(weights))
        assert trainer.replay_ess_base == pytest.approx(expected)
        assert metrics["replay/ess_base_from_estimator"] == 1.0
        assert metrics["replay/ess_base_cohort_n"] == float(len(weights))
        assert "replay/ess_base_unclamped" in metrics

    def test_healthy_cohort_is_ceilinged_by_the_capture(self):
        estimator = EssBaseEstimator(min_seqs=4, clamp_min=1e-6, clamp_max=None)
        trainer = _bare_trainer(estimator=estimator, ess_base=0.03)
        metrics = {"staleness/ess": [self._entry_with_payloads([1.0] * 8)]}
        trainer._update_ess_base_estimator(metrics)
        assert trainer.replay_ess_base == pytest.approx(0.03)
        assert metrics["replay/ess_base_clamped_high"] == 1.0

    def test_payload_free_entries_degrade_to_the_seed(self):
        # Old-format entries (no new_cohort): the estimator never folds and
        # the base stays at the first-update capture — visible only via
        # replay/ess_base_from_estimator == 0.
        estimator = EssBaseEstimator(min_seqs=4)
        trainer = _bare_trainer(estimator=estimator, ess_base=0.03)
        metrics = {"staleness/ess": [{"minibatch_ess_ratio": 0.4, "base_ess_ratio": 0.03}]}
        trainer._update_ess_base_estimator(metrics)
        assert trainer.replay_ess_base == pytest.approx(0.03)
        assert metrics["replay/ess_base_from_estimator"] == 0.0

    def test_missing_metrics_key_is_a_noop(self):
        estimator = EssBaseEstimator(min_seqs=4)
        trainer = _bare_trainer(estimator=estimator, ess_base=None)
        metrics = {}
        trainer._update_ess_base_estimator(metrics)
        assert trainer.replay_ess_base is None
        assert metrics["replay/ess_base_from_estimator"] == 0.0


class TestTrainerCheckpointWithEstimator:
    def _replay_trainer(self, estimator):
        t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        t.replay_buffer = ReplayBuffer(tau=4.0, staleness_threshold=100, seed=0)
        t.replay_updates_done = 5
        t.replay_ess_base = 0.021
        t.replay_ess_auto_base = True
        t.ess_base_estimator = estimator
        return t

    def test_estimator_state_rides_the_replay_checkpoint(self):
        estimator = EssBaseEstimator(min_seqs=8, clamp_min=1e-6, clamp_max=0.9)
        estimator.seed(0.03)
        estimator.observe(_cohort_from_weights([1.0] * 4 + [3.0] * 4), None, 1500.0)
        saver = self._replay_trainer(estimator)
        state = ray.cloudpickle.loads(ray.cloudpickle.dumps(saver._replay_checkpoint_state()))

        restored_estimator = EssBaseEstimator(min_seqs=8, clamp_min=1e-6, clamp_max=0.9)
        restored = self._replay_trainer(restored_estimator)
        restored.replay_ess_base = None
        restored._load_replay_checkpoint_state(state)
        assert restored.replay_ess_base == pytest.approx(0.021)
        assert restored_estimator.current_base() == pytest.approx(estimator.current_base())
        assert restored_estimator.seed_base == pytest.approx(0.03)

    def test_pre_feature_checkpoint_leaves_estimator_fresh(self):
        saver = self._replay_trainer(estimator=None)
        old_state = {
            "buffer": saver.replay_buffer.state_dict(),
            "updates_done": 3,
            "ess_base": 0.5,
        }
        restored_estimator = EssBaseEstimator(min_seqs=8)
        restored = self._replay_trainer(restored_estimator)
        restored._load_replay_checkpoint_state(old_state)
        assert restored.replay_ess_base == pytest.approx(0.5)
        # fresh estimator: no folds, no seed (seeding happens on first update)
        assert restored_estimator.m1 is None
        assert restored_estimator.seed_base is None


class TestComputeEssInfoIntegration:
    """End-to-end actor-side path: TrajRecord dataclasses -> asdict ->
    payloads, with the DP machinery stubbed out."""

    def test_payloads_from_traj_records(self):
        rng = random.Random(0)
        records = []
        for i in range(64):
            is_new = i < 16
            staleness = 1 if is_new else rng.choice([8, 16, 32])
            records.append(
                staleness_utils.TrajRecord(
                    uid=f"t{i}",
                    group_uid=f"g{i // 16}",
                    epoch_idx=0,
                    minibatch_idx=0,
                    trainer_global_step=1,
                    trainer_local_step=1,
                    param_version_start=200 - staleness,
                    param_version_end=200,
                    trainer_param_version=200,
                    response_length=3000,
                    prompt_length=100,
                    advantage_scalar=0.0,
                    reward_scalar=1.0,
                    is_new=is_new,
                    rollout_seq_is=math.exp(rng.gauss(0.0, 0.3)),
                    rollout_seq_is_clipped=None,
                )
            )
        mpu_stub = mock.MagicMock()
        mpu_stub.get_tensor_model_parallel_rank.return_value = 0
        mpu_stub.get_pipeline_model_parallel_rank.return_value = 0
        with (
            mock.patch.object(staleness_utils, "mpu", mpu_stub),
            mock.patch.object(staleness_utils, "allgather_dict_into_list", lambda dicts, group=None: dicts),
        ):
            info = staleness_utils.compute_ess_info(records, rollout_is_threshold=2.0)
        assert 0.0 < info["ess_ratio"] <= 1.0
        cohort = info["new_cohort"]
        assert cohort["n"] == 16
        fresh = [r.rollout_seq_is for r in records if r.is_new]
        assert cohort["w_sum_raw"] == pytest.approx(sum(fresh))
        assert cohort["len_sum"] == pytest.approx(16 * 3000)
        assert info["minibatch_mean_len"] == pytest.approx(3000.0)
        assert {b["s"] for b in info["staleness_buckets"]} <= {1.0, 8.0, 16.0, 32.0}
        # the payload feeds the estimator without adaptation
        estimator = EssBaseEstimator(min_seqs=16, clamp_min=1e-6, clamp_max=1.0)
        estimator.observe(cohort, info["staleness_buckets"], info["minibatch_mean_len"])
        assert estimator.current_base() is not None
