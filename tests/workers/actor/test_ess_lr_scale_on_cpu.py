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
"""Unit tests for the min-ESS LR brake (verl/workers/utils/ess.py):
brake (constant lr_scale multiplier) when the mini-batch's global ESS is
<= min_ess effective samples (inclusive boundary); full nominal lr above.
Replaces the removed auto-captured-base + trigger + sqrt/linear logic.

Run: pytest tests/workers/actor/test_ess_lr_scale_on_cpu.py
"""

import pytest

from verl.workers.config.actor import ESSScalingConfig
from verl.workers.utils.ess import compute_min_ess_lr_scale


class TestConfigValidation:
    def test_defaults_construct(self):
        cfg = ESSScalingConfig()
        assert cfg.min_ess == 1.1
        assert cfg.lr_scale == 0.5
        assert cfg.enable is False

    def test_min_ess_below_one_rejected(self):
        # ESS floors at 1: a threshold below it could never fire
        with pytest.raises(AssertionError, match="min_ess"):
            ESSScalingConfig(min_ess=0.9)

    def test_min_ess_of_exactly_one_accepted(self):
        assert ESSScalingConfig(min_ess=1.0).min_ess == 1.0

    def test_lr_scale_bounds(self):
        with pytest.raises(AssertionError, match="lr_scale"):
            ESSScalingConfig(lr_scale=0.0)
        with pytest.raises(AssertionError, match="lr_scale"):
            ESSScalingConfig(lr_scale=1.5)
        assert ESSScalingConfig(lr_scale=1.0).lr_scale == 1.0


class TestBrakeSides:
    def test_brakes_below_threshold(self):
        assert compute_min_ess_lr_scale(1.0, 1.1, 0.5) == 0.5
        assert compute_min_ess_lr_scale(1.05, 1.1, 0.5) == 0.5

    def test_full_lr_above_threshold(self):
        assert compute_min_ess_lr_scale(1.1000001, 1.1, 0.5) == 1.0
        assert compute_min_ess_lr_scale(2.0, 1.1, 0.5) == 1.0
        assert compute_min_ess_lr_scale(528.0, 1.1, 0.5) == 1.0

    def test_boundary_is_inclusive(self):
        # ess == min_ess brakes (the rule is ESS <= min_ess)
        assert compute_min_ess_lr_scale(1.1, 1.1, 0.5) == 0.5

    def test_constant_multiplier_no_shaping(self):
        # The multiplier does not depend on HOW far below the threshold the
        # ESS sits — no sqrt/linear shaping, just lr_scale.
        assert compute_min_ess_lr_scale(1.0, 1.1, 0.5) == compute_min_ess_lr_scale(1.0999, 1.1, 0.5)


class TestStructuralFloor:
    def test_exact_floor_ess_brakes_at_lr_scale_never_zero(self):
        """The max-shifted ESS computation floors ESS at exactly 1 for any
        non-empty batch (single dominant sequence). With min_ess >= 1 that
        always brakes — and the multiplier is exactly lr_scale, never 0."""
        scale = compute_min_ess_lr_scale(1.0, 1.1, 0.5)
        assert scale == 0.5
        assert scale > 0.0

    def test_empty_batch_is_a_noop(self):
        # count == 0: the ESS was not measured (empty global batch). Nothing
        # to brake.
        assert compute_min_ess_lr_scale(0.0, 1.1, 0.5, count=0) == 1.0

    def test_zero_ess_without_count_keeps_legacy_reading(self):
        # Callers that cannot supply a count keep the old "ess == 0 means an
        # empty batch" reading rather than braking on ambiguous input.
        assert compute_min_ess_lr_scale(0.0, 1.1, 0.5) == 1.0


class TestFailsClosed:
    """A measurement that broke must brake, not hand out full nominal lr.

    On finite log-weights the max-shifted computation cannot produce a
    non-positive or non-finite ESS, so those values mean the input itself was
    unusable — a NaN log-prob, or a -inf rollout log-prob making the sequence
    log-IS +inf. The rule used to return 1.0 for exactly those cases."""

    def test_zero_ess_over_a_non_empty_batch_brakes(self):
        assert compute_min_ess_lr_scale(0.0, 1.1, 0.5, count=528) == 0.5

    def test_nan_ess_brakes(self):
        assert compute_min_ess_lr_scale(float("nan"), 1.1, 0.5, count=528) == 0.5
        # ...and even without a count: NaN is never a healthy reading.
        assert compute_min_ess_lr_scale(float("nan"), 1.1, 0.5) == 0.5

    def test_infinite_ess_brakes(self):
        assert compute_min_ess_lr_scale(float("inf"), 1.1, 0.5, count=528) == 0.5
        assert compute_min_ess_lr_scale(float("-inf"), 1.1, 0.5, count=528) == 0.5

    def test_negative_ess_brakes(self):
        assert compute_min_ess_lr_scale(-1.0, 1.1, 0.5, count=528) == 0.5

    def test_no_data_wins_over_broken_data(self):
        # count == 0 is checked first: a NaN ESS over an empty batch is just
        # "nothing was measured", and braking a no-op step would be noise.
        assert compute_min_ess_lr_scale(float("nan"), 1.1, 0.5, count=0) == 1.0

    def test_healthy_batch_with_count_is_unbraked(self):
        assert compute_min_ess_lr_scale(400.0, 1.1, 0.5, count=528) == 1.0

    def test_degenerate_batch_with_count_still_brakes(self):
        assert compute_min_ess_lr_scale(1.0, 1.1, 0.5, count=528) == 0.5


class TestParameters:
    def test_lr_scale_value_passes_through(self):
        assert compute_min_ess_lr_scale(1.0, 1.1, 0.25) == 0.25
        assert compute_min_ess_lr_scale(1.0, 1.1, 1.0) == 1.0

    def test_min_ess_moves_the_threshold(self):
        # min_ess = 2: two-effective-sample batches now brake too
        assert compute_min_ess_lr_scale(1.9, 2.0, 0.5) == 0.5
        assert compute_min_ess_lr_scale(2.1, 2.0, 0.5) == 1.0

    def test_defaults_documented_values(self):
        # Production defaults: min_ess = 1.1 (10% above the structural floor,
        # i.e. ratio 1.1/528 ~= 0.002083 at B = 528), lr_scale = 0.5.
        assert compute_min_ess_lr_scale(1.0, 1.1, 0.5) == 0.5
        assert compute_min_ess_lr_scale(1.2, 1.1, 0.5) == 1.0
