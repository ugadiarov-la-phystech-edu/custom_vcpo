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
"""Unit tests for the ESS-brake LR multiplier (compute_ess_lr_scale),
including the ess_scaling.trigger_ratio intervention threshold:
- legacy behavior (trigger_ratio=None): min(1, ess/base)
- with a threshold: full lr at or above it, legacy attenuation below it,
  with the documented discontinuity at the boundary

Run: pytest tests/workers/actor/test_ess_lr_scale_on_cpu.py
"""

import pytest

from verl.workers.actor.megatron_actor import compute_ess_lr_scale


class TestLegacyBehavior:
    def test_no_attenuation_at_or_above_base(self):
        assert compute_ess_lr_scale(0.5, 0.5) == 1.0
        assert compute_ess_lr_scale(0.9, 0.5) == 1.0

    def test_proportional_attenuation_below_base(self):
        assert compute_ess_lr_scale(0.25, 0.5) == pytest.approx(0.5)
        assert compute_ess_lr_scale(0.008, 0.016) == pytest.approx(0.5)
        assert compute_ess_lr_scale(0.0019, 0.016) == pytest.approx(0.11875)

    def test_tiny_base_guard(self):
        # base is floored at 1e-8, never a division by zero
        assert compute_ess_lr_scale(0.5, 0.0) == 1.0

    def test_explicit_none_trigger_is_legacy(self):
        assert compute_ess_lr_scale(0.4, 0.5, None) == pytest.approx(0.8)


class TestTriggerRatio:
    def test_full_lr_at_or_above_threshold(self):
        # ratio = 0.8 >= trigger 0.5 -> no intervention despite ess < base
        assert compute_ess_lr_scale(0.4, 0.5, 0.5) == 1.0
        # exactly at the threshold -> no intervention (strict "less than")
        assert compute_ess_lr_scale(0.25, 0.5, 0.5) == 1.0

    def test_legacy_attenuation_below_threshold(self):
        # ratio = 0.4 < trigger 0.5 -> legacy multiplier ess/base
        assert compute_ess_lr_scale(0.2, 0.5, 0.5) == pytest.approx(0.4)
        assert compute_ess_lr_scale(0.0019, 0.016, 0.5) == pytest.approx(0.11875)

    def test_discontinuity_at_threshold(self):
        eps = 1e-9
        at = compute_ess_lr_scale(0.25, 0.5, 0.5)
        below = compute_ess_lr_scale(0.25 - eps, 0.5, 0.5)
        assert at == 1.0
        assert below == pytest.approx(0.5, abs=1e-6)

    def test_trigger_one_matches_legacy(self):
        for ess in (0.1, 0.3, 0.5, 0.7):
            assert compute_ess_lr_scale(ess, 0.5, 1.0) == compute_ess_lr_scale(ess, 0.5)

    def test_trigger_above_one_is_inert(self):
        # ratios in [1, trigger) would cap at 1 anyway; below 1 the legacy
        # multiplier applies -> identical to legacy for any input
        for ess in (0.1, 0.5, 0.9, 1.5):
            assert compute_ess_lr_scale(ess, 0.5, 2.0) == compute_ess_lr_scale(ess, 0.5)
