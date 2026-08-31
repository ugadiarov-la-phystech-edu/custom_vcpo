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

"""Tests for the sync trainer's clean-training-time counters.

``compute_cumulative_timing_metrics`` gives the synchronous RayPPOTrainer the
same ``fully_async/timing/*`` metric family the fully-async recipe logs, so
sync runs drop into the existing val-vs-clean-time comparison tooling. Covered:
the subtraction semantics (step minus testing minus save_checkpoint, both timed
inside the step), accumulation across steps, missing-key and pathological-clock
robustness, per-loop counter isolation, exact tag parity with the async
trainer, and a source-level tripwire that fit() actually calls the helper
(plumbing not exercised by unit tests has bitten this project before).

Run: pytest tests/trainer/ppo/test_cumulative_timing_on_cpu.py
"""

import inspect
import os

import pytest

from verl.trainer.ppo.metric_utils import compute_cumulative_timing_metrics

WALL = "fully_async/timing/wall_time_since_first_sample"
VAL = "fully_async/timing/cumulative_validation_time"
SAVE = "fully_async/timing/cumulative_save_time"
TRAIN = "fully_async/timing/cumulative_training_time"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class TestSubtractionSemantics:
    def test_plain_step_is_all_training(self):
        out = compute_cumulative_timing_metrics({}, {"step": 100.0})
        assert out[TRAIN] == 100.0
        assert out[WALL] == 100.0
        assert out[VAL] == 0.0
        assert out[SAVE] == 0.0

    def test_testing_and_save_are_subtracted(self):
        out = compute_cumulative_timing_metrics({}, {"step": 120.0, "testing": 30.0, "save_checkpoint": 10.0})
        assert out[TRAIN] == 80.0
        assert out[WALL] == 120.0
        assert out[VAL] == 30.0
        assert out[SAVE] == 10.0

    def test_other_timing_keys_are_ignored(self):
        """gen/update_actor etc. are parts of the step, not subtractions."""
        out = compute_cumulative_timing_metrics({}, {"step": 100.0, "gen": 40.0, "update_actor": 50.0})
        assert out[TRAIN] == 100.0

    def test_missing_step_counts_as_zero(self):
        out = compute_cumulative_timing_metrics({}, {"testing": 5.0})
        assert out[WALL] == 0.0
        assert out[TRAIN] == 0.0
        assert out[VAL] == 5.0

    def test_pathological_clock_never_decreases_training_time(self):
        """testing+save exceeding the step that contains them clamps to 0, not negative."""
        state = {}
        compute_cumulative_timing_metrics(state, {"step": 100.0})
        out = compute_cumulative_timing_metrics(state, {"step": 10.0, "testing": 20.0})
        assert out[TRAIN] == 100.0  # unchanged, not 90


class TestAccumulation:
    def test_counters_accumulate_across_steps(self):
        state = {}
        compute_cumulative_timing_metrics(state, {"step": 100.0, "testing": 20.0})
        compute_cumulative_timing_metrics(state, {"step": 50.0})
        out = compute_cumulative_timing_metrics(state, {"step": 60.0, "save_checkpoint": 15.0})
        assert out[WALL] == pytest.approx(210.0)
        assert out[VAL] == pytest.approx(20.0)
        assert out[SAVE] == pytest.approx(15.0)
        assert out[TRAIN] == pytest.approx(80.0 + 50.0 + 45.0)

    def test_monotone_non_decreasing(self):
        state = {}
        prev = 0.0
        for raw in ({"step": 10.0}, {"step": 5.0, "testing": 5.0}, {"step": 0.0}, {"step": 7.0}):
            out = compute_cumulative_timing_metrics(state, raw)
            assert out[TRAIN] >= prev
            prev = out[TRAIN]

    def test_state_dicts_are_independent(self):
        a, b = {}, {}
        compute_cumulative_timing_metrics(a, {"step": 100.0})
        out_b = compute_cumulative_timing_metrics(b, {"step": 1.0})
        assert out_b[TRAIN] == 1.0
        assert compute_cumulative_timing_metrics(a, {"step": 0.0})[TRAIN] == 100.0


class TestCrossArmTagParity:
    def test_tags_match_the_fully_async_trainer(self):
        """The whole point of the tag choice: the async recipe logs the same four
        names, and the val-vs-time plot tooling keys on the cumulative_training_time
        one. Guard against either side renaming."""
        src = open(os.path.join(REPO_ROOT, "recipe", "fully_async_policy", "fully_async_trainer.py")).read()
        for tag in (WALL, VAL, SAVE, TRAIN):
            assert tag in src, f"async trainer no longer logs {tag}"

    def test_fit_invokes_the_helper(self):
        """Source-level tripwire: a helper nobody calls logs nothing. (A driver-side
        key that never reached its consumer has silently disabled a feature in this
        repo before.)"""
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        src = inspect.getsource(RayPPOTrainer.fit)
        assert "compute_cumulative_timing_metrics(self._cumulative_timing, timing_raw)" in src
