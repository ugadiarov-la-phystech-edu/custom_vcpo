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

"""CPU tests for the log-space (max-shifted) global ESS computation.

compute_global_ess_from_log_weights replaces the raw-space pipeline
(exp locally, all-reduce fp32 sums) whose range limits censored the brake
signal at both ends, observed in the 2026-08 fsdp2 replay run:

- a dominant sequence with log-weight ~+60 overflowed the squared sum in the
  fp32 all-reduce cast -> ESS read 0.0 -> the brake scaled the LR to 0 and
  the update was silently skipped (10 of the first 51 steps);
- batches with all log-weights <~ -87 flushed every weight to zero.

ESS = (sum w)^2 / sum w^2 is scale-invariant, so the max-shifted exponents
give the exact value at any point of the fp range; clipping commutes with exp
(min(w, c) = exp(min(s, log c))) and is applied in log space, shifted by the
max of the *clamped* exponents.

Run: pytest tests/workers/utils/test_ess_logspace_on_cpu.py
"""

import math

import pytest
import torch.distributed as dist

from verl.workers.utils.ess import compute_global_ess_from_log_weights


def naive_ess(weights: list[float]) -> tuple[float, float]:
    """Textbook ESS in float64 — valid reference for benign weight ranges."""
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    ess = sw * sw / sw2
    return ess, ess / len(weights)


class TestExactness:
    def test_matches_naive_on_benign_weights(self):
        weights = [3.0, 0.2, 1.5, 0.7]
        ref_ess, ref_ratio = naive_ess(weights)
        ess, ratio, ess_c, ratio_c, n = compute_global_ess_from_log_weights([math.log(w) for w in weights])
        assert n == 4
        assert ess == pytest.approx(ref_ess, rel=1e-12)
        assert ratio == pytest.approx(ref_ratio, rel=1e-12)
        # no threshold -> clipped variant is the same quantity
        assert ess_c == pytest.approx(ref_ess, rel=1e-12)
        assert ratio_c == pytest.approx(ref_ratio, rel=1e-12)

    def test_clipped_matches_naive_on_clipped_weights(self):
        weights = [8.0, 1.0, 0.5, 0.1]
        threshold = 2.0
        ref_ess, ref_ratio = naive_ess([min(w, threshold) for w in weights])
        _, _, ess_c, ratio_c, _ = compute_global_ess_from_log_weights(
            [math.log(w) for w in weights], rollout_is_threshold=threshold
        )
        assert ess_c == pytest.approx(ref_ess, rel=1e-12)
        assert ratio_c == pytest.approx(ref_ratio, rel=1e-12)

    def test_scale_invariance_under_extreme_common_shift(self):
        """Multiplying all weights by e^±500 (far outside even fp64 exp range
        in raw space) must not change either ratio."""
        base_logs = [1.1, -0.3, 0.0, -2.2]
        ref = compute_global_ess_from_log_weights(base_logs)
        for shift in (500.0, -500.0):
            shifted = compute_global_ess_from_log_weights([s + shift for s in base_logs])
            assert shifted[0] == pytest.approx(ref[0], rel=1e-9)  # ess
            assert shifted[1] == pytest.approx(ref[1], rel=1e-9)  # ratio
        # the clip threshold is absolute, so only the UNclipped variant is
        # shift-invariant; with no threshold both were compared above


class TestRangeRegressions:
    def test_overflow_dominant_weight_reads_one_over_n_not_zero(self):
        """Regression for the lr=0 steps: one log-weight at +60 made the old
        pipeline read ESS = 0.0 ((finite sum)^2 / (fp32-inf sq-sum)). The true
        value is ESS ~ 1 (total single-sequence domination) -> ratio ~ 1/B."""
        logs = [60.0, 0.0, 0.0, 0.0]
        ess, ratio, _, _, _ = compute_global_ess_from_log_weights(logs)
        assert ess == pytest.approx(1.0, rel=1e-9)
        assert ratio == pytest.approx(0.25, rel=1e-9)

    def test_overflow_dominant_weight_clipped_variant_still_exact(self):
        # clipped weights are [2, 1, 1, 1] -> ESS = 25/7
        logs = [60.0, 0.0, 0.0, 0.0]
        _, _, ess_c, ratio_c, _ = compute_global_ess_from_log_weights(logs, rollout_is_threshold=2.0)
        assert ess_c == pytest.approx(25.0 / 7.0, rel=1e-12)
        assert ratio_c == pytest.approx(25.0 / 28.0, rel=1e-12)

    def test_deep_underflow_equal_weights_read_full_ess(self):
        """All weights e^-200: raw-space fp32 flushed them all to zero (ESS
        read 0.0); identical weights really mean ESS ratio = 1."""
        out = compute_global_ess_from_log_weights([-200.0] * 4, rollout_is_threshold=2.0)
        ess, ratio, ess_c, ratio_c, n = out
        assert ess == pytest.approx(4.0, rel=1e-12)
        assert ratio == pytest.approx(1.0, rel=1e-12)
        assert ess_c == pytest.approx(4.0, rel=1e-12)
        assert ratio_c == pytest.approx(1.0, rel=1e-12)
        assert n == 4

    def test_deep_negative_spread_matches_shifted_reference(self):
        logs = [-200.0, -201.0, -202.5, -205.0]
        ref_ess, ref_ratio = naive_ess([math.exp(s + 200.0) for s in logs])
        ess, ratio, _, _, _ = compute_global_ess_from_log_weights(logs)
        assert ess == pytest.approx(ref_ess, rel=1e-12)
        assert ratio == pytest.approx(ref_ratio, rel=1e-12)

    def test_clipped_sums_shift_by_clamped_max_not_raw_max(self):
        """The trap: with a dominant raw log-weight of +60 and clip at log 2,
        shifting the clipped exponents by the RAW max would push them all to
        exp(-59)-and-below, recreating the underflow on the clipped path.
        Correct shift is the max of the clamped exponents (log 2 here)."""
        logs = [60.0, 0.0, -1.0]
        clipped_ref_ess, clipped_ref_ratio = naive_ess([2.0, 1.0, math.exp(-1.0)])
        _, _, ess_c, ratio_c, _ = compute_global_ess_from_log_weights(logs, rollout_is_threshold=2.0)
        assert ess_c == pytest.approx(clipped_ref_ess, rel=1e-12)
        assert ratio_c == pytest.approx(clipped_ref_ratio, rel=1e-12)

    def test_lower_bound_one_over_n_under_total_domination(self):
        """Weights beyond fp64 range relative to the max contribute 0 after
        the shift — the ratio bottoms out at exactly 1/N, never 0."""
        ess, ratio, _, _, _ = compute_global_ess_from_log_weights([0.0, -3000.0, -3000.0, -3000.0])
        assert ess == pytest.approx(1.0, abs=1e-15)
        assert ratio == pytest.approx(0.25, abs=1e-15)


class TestEdgeCases:
    def test_empty_returns_zeros(self):
        assert compute_global_ess_from_log_weights([]) == (0.0, 0.0, 0.0, 0.0, 0)

    def test_single_sequence(self):
        ess, ratio, ess_c, ratio_c, n = compute_global_ess_from_log_weights([-5.0], rollout_is_threshold=2.0)
        assert (ess, ratio, ess_c, ratio_c, n) == (1.0, 1.0, 1.0, 1.0, 1)

    @pytest.mark.parametrize("threshold", [None, 0.0, -1.0])
    def test_nonpositive_or_missing_threshold_disables_clipping(self, threshold):
        logs = [2.0, 0.0, -1.0]  # top weight e^2 > 2.0 would clip if enabled
        ess, ratio, ess_c, ratio_c, _ = compute_global_ess_from_log_weights(logs, rollout_is_threshold=threshold)
        assert ess_c == pytest.approx(ess, rel=1e-15)
        assert ratio_c == pytest.approx(ratio, rel=1e-15)

    def test_threshold_engages_only_above_clip(self):
        # all weights below the threshold -> clipping is a mathematical no-op
        logs = [math.log(0.5), math.log(1.5), math.log(0.9)]
        ess, ratio, ess_c, ratio_c, _ = compute_global_ess_from_log_weights(logs, rollout_is_threshold=2.0)
        assert ess_c == pytest.approx(ess, rel=1e-12)
        assert ratio_c == pytest.approx(ratio, rel=1e-12)


class TestNonFiniteInputs:
    """A non-finite log-weight means the measurement itself broke: a NaN
    log-prob after a bad step, or a -inf rollout log-prob turning the ratio
    into +inf. The ESS is then unknowable and must read NaN, so the brake can
    fail closed. Before this, the non-finite max made the shifted sums 0 and
    the ESS read a healthy-looking 0.0 -> full nominal lr on the worst batch
    of the run."""

    def test_nan_entry_gives_nan_ess_for_both_variants(self):
        ess, ratio, ess_c, ratio_c, n = compute_global_ess_from_log_weights([0.0, -1.0, float("nan"), -2.0], 2.0)
        assert math.isnan(ess) and math.isnan(ratio)
        assert math.isnan(ess_c) and math.isnan(ratio_c)
        assert n == 4  # count still reports the batch size: broken != empty

    def test_infinite_entry_corrupts_only_the_unclipped_variant(self):
        """+inf is unreadable as a raw weight, but clipping maps it to
        log(threshold) — a correct finite weight — so the CLIPPED ESS is
        knowable. Flagging both would brake every mini-batch containing one
        -inf rollout log-prob whenever use_clipped=True."""
        ess, ratio, ess_c, ratio_c, n = compute_global_ess_from_log_weights([0.0, -1.0, float("inf"), -2.0], 2.0)
        assert math.isnan(ess) and math.isnan(ratio)
        assert math.isfinite(ess_c) and 1.0 <= ess_c <= 4.0
        assert ratio_c == pytest.approx(ess_c / 4)
        assert n == 4

    def test_infinite_entry_without_a_threshold_corrupts_both(self):
        """No clipping -> the clipped variant IS the unclipped one."""
        ess, _, ess_c, _, _ = compute_global_ess_from_log_weights([0.0, float("inf")], None)
        assert math.isnan(ess) and math.isnan(ess_c)

    def test_corrupt_entry_does_not_read_as_zero(self):
        """The old behaviour, pinned so it cannot come back: a +inf weight
        used to make every shifted sum 0.0 and the ESS a plain 0.0."""
        ess, ratio, *_ = compute_global_ess_from_log_weights([float("inf"), 0.0, 0.0, 0.0])
        assert ess != 0.0 and ratio != 0.0

    def test_negative_infinity_is_a_legitimate_zero_weight(self):
        # exp(-inf) == 0 exactly; that is a weight of zero, not corruption.
        ess, ratio, _, _, n = compute_global_ess_from_log_weights([0.0, float("-inf"), float("-inf"), float("-inf")])
        assert (ess, ratio, n) == (1.0, 0.25, 4)

    def test_all_negative_infinity_reads_zero_over_a_non_empty_batch(self):
        ess, ratio, _, _, n = compute_global_ess_from_log_weights([float("-inf")] * 4)
        assert (ess, ratio, n) == (0.0, 0.0, 4)

    def test_finite_neighbours_of_a_corrupt_entry_do_not_rescue_it(self):
        """Even one bad sequence in an otherwise healthy mini-batch poisons
        the measurement — the brake must not be handed a plausible number."""
        logs = [0.1 * i for i in range(64)] + [float("nan")]
        ess, _, _, _, n = compute_global_ess_from_log_weights(logs)
        assert math.isnan(ess)
        assert n == 65


@pytest.fixture()
def single_rank_gloo(tmp_path_factory):
    created = False
    if not dist.is_initialized():
        init_file = tmp_path_factory.mktemp("dist") / "init"
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


class TestDistributedPath:
    @pytest.mark.usefixtures("single_rank_gloo")
    def test_allreduce_path_matches_local(self):
        """world_size=1 gloo: exercises both collectives (MAX on the maxima,
        SUM on the shifted sums) end to end."""
        logs = [60.0, 0.0, -1.0, -200.0]
        assert dist.is_initialized()
        with_dist = compute_global_ess_from_log_weights(logs, rollout_is_threshold=2.0)
        ref_ess, ref_ratio = naive_ess([1.0, math.exp(-60.0), math.exp(-61.0), 0.0])
        assert with_dist[0] == pytest.approx(ref_ess, rel=1e-9)
        assert with_dist[1] == pytest.approx(ref_ratio, rel=1e-9)
        clipped_ref_ess, clipped_ref_ratio = naive_ess([2.0, 1.0, math.exp(-1.0), math.exp(-200.0)])
        assert with_dist[2] == pytest.approx(clipped_ref_ess, rel=1e-9)
        assert with_dist[3] == pytest.approx(clipped_ref_ratio, rel=1e-9)
        assert with_dist[4] == 4

    @pytest.mark.usefixtures("single_rank_gloo")
    @pytest.mark.parametrize("bad", [float("nan")])
    def test_corruption_survives_the_collectives(self, bad):
        """The corruption flag rides the SUM all-reduce as a 6th element, so
        it cannot be lost the way a NaN in the MAX all-reduce would be
        (NCCL's MAX over NaN is implementation defined). The local max itself
        is taken over finite entries only, which is what keeps that MAX
        meaningful for the other ranks."""
        assert dist.is_initialized()
        ess, ratio, ess_c, ratio_c, n = compute_global_ess_from_log_weights([1.0, bad, -3.0], 2.0)
        assert math.isnan(ess) and math.isnan(ratio)
        assert math.isnan(ess_c) and math.isnan(ratio_c)
        assert n == 3

    @pytest.mark.usefixtures("single_rank_gloo")
    def test_healthy_batch_unaffected_by_the_corruption_flag(self):
        logs = [0.5, 0.0, -1.0]
        assert dist.is_initialized()
        ess, ratio, _, _, n = compute_global_ess_from_log_weights(logs)
        ref_ess, ref_ratio = naive_ess([math.exp(v) for v in logs])
        assert ess == pytest.approx(ref_ess, rel=1e-9)
        assert ratio == pytest.approx(ref_ratio, rel=1e-9)
        assert n == 3
