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
"""The ESS measurement must be range-safe on BOTH per-traj paths.

The min-ESS brake sets the LR of every update from one number, and that number
used to be produced two different ways:

- mbs=1 (use_dynamic_bsz=False): compute_is_info stored exp(sum log-ratio) in
  fp32 and compute_ess_info reduced those raw weights. fp32 exp() flushes sums
  below ~-87 to 0 and turns sums above ~88.7 into inf, so a degenerate batch
  read ESS = 0 or NaN and compute_min_ess_lr_scale ran it at FULL lr — the
  brake disengaged exactly on the batches it exists to catch (observed at
  steps 345/346/348 of the 2026-08 replay run, ESS = 0.00).
- packed (use_dynamic_bsz=True): per-sequence log-IS sums straight into the
  max-shifted computation, exact at any range.

Both now share verl.workers.utils.ess. These tests pin the agreement between
the two paths, the range regressions, and the resulting brake decisions.

Run: pytest tests/workers/actor/test_ess_range_both_paths_on_cpu.py
"""

import math
import random
from types import SimpleNamespace

import pytest
import torch

from recipe.fully_async_policy import staleness_utils
from recipe.fully_async_policy.staleness_utils import TrajRecord, compute_ess_info, compute_is_info
from verl.workers.utils.ess import compute_global_ess_from_log_weights, compute_min_ess_lr_scale

B = 528  # production mini-batch: 33 prompts x 16 responses
MIN_ESS = 1.1
LR_SCALE = 0.5
THRESHOLD = 2.0  # algorithm.rollout_correction.rollout_is_threshold in both scripts


@pytest.fixture()
def single_rank_gather(monkeypatch):
    """compute_ess_info's DP all-gather, reduced to a single leader rank."""
    monkeypatch.setattr(
        staleness_utils,
        "mpu",
        SimpleNamespace(
            get_data_parallel_group=lambda with_context_parallel=False: None,
            get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0,
        ),
    )
    monkeypatch.setattr(staleness_utils, "allgather_dict_into_list", lambda local, group=None: list(local))


def _traj(uid="t0", **kwargs) -> TrajRecord:
    """TrajRecord with the bookkeeping fields filled in; only the IS fields matter here."""
    core = dict(
        uid=uid,
        group_uid="g0",
        epoch_idx=0,
        minibatch_idx=0,
        trainer_global_step=0,
        trainer_local_step=0,
        param_version_start=0,
        param_version_end=0,
        trainer_param_version=0,
        response_length=8,
        prompt_length=4,
        advantage_scalar=0.0,
        reward_scalar=0.0,
    )
    core.update(kwargs)
    return TrajRecord(**core)


def _records(log_weights, threshold=THRESHOLD):
    """TrajRecords as compute_is_info leaves them (both IS fields populated)."""
    records = []
    for i, s in enumerate(log_weights):
        seq_is = float(torch.exp(torch.tensor(float(s), dtype=torch.float32)))
        rec = _traj(uid=f"t{i}", rollout_seq_log_is=float(s), rollout_seq_is=seq_is)
        if threshold is not None and threshold > 0:
            rec.rollout_seq_is_clipped = min(seq_is, float(threshold))
        records.append(rec)
    return records


def raw_space_ess(log_weights):
    """The old mbs=1 formula, kept as the regression reference: fp32 exp() per
    sequence, float64 sums, ESS = (sum w)^2 / (sum w^2 + eps)."""
    w = [float(torch.exp(torch.tensor(float(s), dtype=torch.float32))) for s in log_weights]
    return sum(w) ** 2 / (sum(x * x for x in w) + 1e-8)


def brake(ess, count):
    return compute_min_ess_lr_scale(ess, MIN_ESS, LR_SCALE, count=count)


class TestPathsAgree:
    """Same log weights in, same ESS out — with dynamic batch size on or off."""

    @pytest.mark.parametrize(
        "log_weights",
        [
            [0.3 * (i % 7 - 3) for i in range(B)],  # healthy spread
            [-200.0] * B,  # every weight underflows fp32
            [200.0] + [0.0] * (B - 1),  # dominant weight overflows fp32
            [-200.0] + [-300.0] * (B - 1),  # degenerate AND fully underflowing
            [0.0] * B,  # perfectly on-policy
            [-5.0],  # single sequence
        ],
    )
    @pytest.mark.parametrize("threshold", [None, THRESHOLD])
    def test_mbs1_matches_packed(self, single_rank_gather, log_weights, threshold):
        mbs1 = compute_ess_info(_records(log_weights, threshold), threshold)
        packed = compute_global_ess_from_log_weights(log_weights, threshold)
        assert mbs1["ess"] == pytest.approx(packed[0], rel=1e-12)
        assert mbs1["ess_ratio"] == pytest.approx(packed[1], rel=1e-12)
        assert mbs1["ess_clipped"] == pytest.approx(packed[2], rel=1e-12)
        assert mbs1["ess_ratio_clipped"] == pytest.approx(packed[3], rel=1e-12)
        assert mbs1["count"] == packed[4] == len(log_weights)

    def test_both_paths_brake_identically(self, single_rank_gather):
        log_weights = [-200.0] + [-300.0] * (B - 1)
        mbs1 = compute_ess_info(_records(log_weights), THRESHOLD)
        packed = compute_global_ess_from_log_weights(log_weights, THRESHOLD)
        assert brake(mbs1["ess"], mbs1["count"]) == brake(packed[0], packed[4]) == LR_SCALE


class TestRangeRegressions:
    def test_uniform_underflow_no_longer_reads_as_empty(self, single_rank_gather):
        """The observed incident: every sequence's log-IS below -87. The batch
        is perfectly uniform, so the truth is ESS = B (fully healthy)."""
        log_weights = [-200.0] * B
        assert raw_space_ess(log_weights) == 0.0  # what the old path reported

        info = compute_ess_info(_records(log_weights), THRESHOLD)
        assert info["ess"] == pytest.approx(float(B))
        assert info["ess_ratio"] == pytest.approx(1.0)
        assert brake(info["ess"], info["count"]) == 1.0

    def test_underflowed_degenerate_batch_now_brakes(self, single_rank_gather):
        """The case the censoring actually hid: one sequence carries e^100 of
        the mass of the others — a textbook ESS = 1 batch — but every raw
        weight is below fp32's floor, so the old path read ESS = 0 and, under
        the min-ESS rule, stepped at full lr."""
        log_weights = [-200.0] + [-300.0] * (B - 1)
        assert raw_space_ess(log_weights) == 0.0
        # ...and 'ess == 0' used to mean 'empty batch: do not scale'
        assert compute_min_ess_lr_scale(0.0, MIN_ESS, LR_SCALE) == 1.0

        info = compute_ess_info(_records(log_weights), THRESHOLD)
        assert info["ess"] == pytest.approx(1.0, rel=1e-9)
        assert info["ess_ratio"] == pytest.approx(1.0 / B, rel=1e-9)
        assert brake(info["ess"], info["count"]) == LR_SCALE

    def test_overflow_no_longer_disables_the_brake(self, single_rank_gather):
        """Mirror failure: one log-IS above 88.7 -> inf -> ESS NaN -> the old
        brake returned 1.0 (min(1.0, nan)) and stepped at full lr."""
        log_weights = [200.0] + [0.0] * (B - 1)
        assert math.isnan(raw_space_ess(log_weights))

        info = compute_ess_info(_records(log_weights), THRESHOLD)
        assert info["ess"] == pytest.approx(1.0, rel=1e-9)
        assert brake(info["ess"], info["count"]) == LR_SCALE

    def test_healthy_range_matches_the_old_arithmetic(self, single_rank_gather):
        """Where fp32 was never in danger, the fix must not move the number."""
        rng = random.Random(0)
        log_weights = [rng.uniform(-10.0, 2.0) for _ in range(B)]
        info = compute_ess_info(_records(log_weights), None)
        assert info["ess"] == pytest.approx(raw_space_ess(log_weights), rel=1e-6)

    def test_clipping_still_bites_in_log_space(self, single_rank_gather):
        """Clipped ESS is evaluated on min(w, threshold) — equivalently on the
        clamped exponents — and must stay above the unclipped ESS when a heavy
        tail is what dragged the latter down."""
        log_weights = [8.0] + [0.0] * (B - 1)  # e^8 >> threshold 2.0
        info = compute_ess_info(_records(log_weights), THRESHOLD)
        assert info["ess"] < info["ess_clipped"]
        expected_clipped = compute_global_ess_from_log_weights(log_weights, THRESHOLD)[2]
        assert info["ess_clipped"] == pytest.approx(expected_clipped, rel=1e-12)


class TestRecordHandling:
    def test_legacy_records_fall_back_to_the_stored_weight(self, single_rank_gather):
        """Records written before rollout_seq_log_is existed still carry a
        usable exponent wherever the fp32 weight was not censored."""
        log_weights = [-2.0, -1.0, 0.0, 0.5]
        records = _records(log_weights)
        for rec in records:
            rec.rollout_seq_log_is = None  # legacy shape

        info = compute_ess_info(records, THRESHOLD)
        expected = compute_global_ess_from_log_weights(log_weights, THRESHOLD)
        assert info["ess"] == pytest.approx(expected[0], rel=1e-6)
        assert info["count"] == 4

    def test_legacy_censored_weights_are_skipped_not_trusted(self, single_rank_gather):
        """A legacy record whose weight underflowed to 0 carries no
        information: it must drop out of the measurement entirely rather than
        be read as a zero weight."""
        records = _records([-2.0, -200.0, 0.0])
        for rec in records:
            rec.rollout_seq_log_is = None
        assert records[1].rollout_seq_is == 0.0

        info = compute_ess_info(records, THRESHOLD)
        assert info["count"] == 2
        assert info["ess"] == pytest.approx(compute_global_ess_from_log_weights([-2.0, 0.0], THRESHOLD)[0], rel=1e-6)

    def test_records_without_is_fields_are_ignored(self, single_rank_gather):
        """The packed path builds records with no IS fields at all; a mixed
        list must not count them (count == 0 means 'not measured')."""
        records = [_traj(uid="a"), _traj(uid="b")]
        info = compute_ess_info(records, THRESHOLD)
        assert info["count"] == 0
        assert info["ess"] == 0.0
        assert brake(info["ess"], info["count"]) == 1.0  # nothing measured -> no scaling

    def test_compute_is_info_stores_the_masked_log_sum(self):
        """rollout_seq_log_is is the masked sum of token log-ratios, and it
        exponentiates back to rollout_seq_is wherever fp32 can hold it."""
        torch.manual_seed(0)
        old_log_prob = torch.randn(1, 12) * 0.1
        rollout_log_prob = torch.randn(1, 12) * 0.1
        mask = torch.zeros(1, 12)
        mask[0, :7] = 1.0  # only the first 7 tokens are response tokens

        rec = compute_is_info(_traj(), rollout_log_prob, old_log_prob, mask, THRESHOLD)
        expected = float(((old_log_prob - rollout_log_prob) * mask).sum())
        assert rec.rollout_seq_log_is == pytest.approx(expected, rel=1e-6)
        assert math.exp(rec.rollout_seq_log_is) == pytest.approx(rec.rollout_seq_is, rel=1e-5)

    def test_compute_is_info_log_sum_survives_where_the_exp_does_not(self):
        """A drift that censors rollout_seq_is must still leave an exact
        log-space value behind."""
        old_log_prob = torch.full((1, 200), -1.0)
        rollout_log_prob = torch.zeros(1, 200)  # log-ratio -1 per token -> sum -200
        mask = torch.ones(1, 200)

        rec = compute_is_info(_traj(), rollout_log_prob, old_log_prob, mask, THRESHOLD)
        assert rec.rollout_seq_is == 0.0  # fp32 exp underflowed
        assert rec.rollout_seq_log_is == pytest.approx(-200.0, rel=1e-6)


class TestInvariants:
    @pytest.mark.parametrize("seed", range(8))
    def test_ess_stays_in_range_and_the_brake_is_binary(self, single_rank_gather, seed):
        """Over randomized batches — including ones far outside fp32's range —
        ESS lives in [1, B], and the multiplier is either 1.0 or exactly
        lr_scale. Never 0, never NaN: the LR cannot collapse via the brake."""
        rng = random.Random(seed)
        centre = rng.choice([0.0, -150.0, 150.0])
        log_weights = [centre + rng.gauss(0.0, rng.choice([0.1, 5.0, 50.0])) for _ in range(B)]

        for ess, count in (
            (lambda i: (i["ess"], i["count"]))(compute_ess_info(_records(log_weights), THRESHOLD)),
            compute_global_ess_from_log_weights(log_weights, THRESHOLD)[0::4],
        ):
            assert math.isfinite(ess)
            assert 1.0 - 1e-9 <= ess <= B + 1e-9
            assert brake(ess, count) in (1.0, LR_SCALE)
