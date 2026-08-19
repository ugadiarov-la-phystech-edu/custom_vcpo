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
Also covers the per-traj path's loss_multiplier resolution (megatron_actor).

Run: pytest tests/workers/actor/test_ess_lr_scale_on_cpu.py
"""

import pytest

from verl.workers.actor.megatron_actor import _resolve_loss_multiplier
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
        # ess == 0 only happens for an empty global batch (or the mbs=1
        # path's raw-space overflow censoring): no scaling.
        assert compute_min_ess_lr_scale(0.0, 1.1, 0.5) == 1.0


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


class TestResolveLossMultiplier:
    """The buffer-free per-traj path folds the advantage into loss_multiplier,
    so an exact 0.0 (zero-advantage trajectory) must be honored — a `x or 1.0`
    parse would silently promote it to a full-weight score gradient."""

    def test_missing_defaults_to_one(self):
        assert _resolve_loss_multiplier({}) == 1.0

    def test_none_defaults_to_one(self):
        assert _resolve_loss_multiplier({"loss_multiplier": None}) == 1.0

    def test_explicit_zero_is_honored(self):
        assert _resolve_loss_multiplier({"loss_multiplier": 0.0}) == 0.0

    def test_value_passthrough(self):
        assert _resolve_loss_multiplier({"loss_multiplier": 0.25}) == 0.25
