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
sync runs drop into the existing val-vs-clean-time comparison tooling.

On this fork RayPPOTrainer.fit times validation (``testing``) and checkpoint
saving (``save_checkpoint``) OUTSIDE the ``step`` timer, so the step IS the
clean training time and the wall clock of an iteration is step + testing +
save. The first version of the helper assumed the upstream layout (both inside
the step) and subtracted them from the step a second time; clamped at zero, it
reported +0 s of training on every validation step of the Qwen3-8B sync run
(2026-09-04). Covered here: the additive semantics, the exact identity
training == wall - validation - save, accumulation / monotonicity / per-loop
isolation, a replay of that run's real per-step timings, an AST tripwire that
pins the timer layout the helper relies on, tag parity with the async trainer,
and a source-level check that fit() actually calls the helper.

Run: pytest tests/trainer/ppo/test_cumulative_timing_on_cpu.py
"""

import ast
import inspect
import os
import random
import textwrap

import pytest

from verl.trainer.ppo.metric_utils import compute_cumulative_timing_metrics

WALL = "fully_async/timing/wall_time_since_first_sample"
VAL = "fully_async/timing/cumulative_validation_time"
SAVE = "fully_async/timing/cumulative_save_time"
TRAIN = "fully_async/timing/cumulative_training_time"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# (step, testing, save_checkpoint) per rollout step of the Qwen3-8B sync arm on
# remote_h100 (logs/qwen3-8b_sync_B128xn16_mini32.log, rollout steps 1-9,
# test_freq = save_freq = 3). With the old helper the logged
# cumulative_training_time after step 9 was 9,390.6 s.
REMOTE_H100_STEPS = [
    (1609.157, 0.0, 0.0),
    (1549.508, 0.0, 0.0),
    (1438.990, 1438.682, 46.613),
    (1533.789, 0.0, 0.0),
    (1525.032, 0.0, 0.0),
    (1575.760, 1391.846, 50.408),
    (1466.408, 0.0, 0.0),
    (1507.003, 0.0, 0.0),
    (1504.763, 1390.216, 48.321),
]
LOGGED_TRAINING_TIME_WITH_OLD_HELPER = 9390.6


def _replay(triples, state=None):
    state = {} if state is None else state
    out = None
    for step, testing, save in triples:
        raw = {"step": step}
        if testing:
            raw["testing"] = testing
        if save:
            raw["save_checkpoint"] = save
        out = compute_cumulative_timing_metrics(state, raw)
    return out


class TestAdditiveSemantics:
    def test_plain_step_is_all_training(self):
        out = compute_cumulative_timing_metrics({}, {"step": 100.0})
        assert out[TRAIN] == 100.0
        assert out[WALL] == 100.0
        assert out[VAL] == 0.0
        assert out[SAVE] == 0.0

    def test_testing_and_save_extend_wall_not_training(self):
        """Validation and saving happen after the step timer closes: they add to
        the iteration's wall time and their own counters, never to training."""
        out = compute_cumulative_timing_metrics({}, {"step": 120.0, "testing": 30.0, "save_checkpoint": 10.0})
        assert out[TRAIN] == 120.0
        assert out[WALL] == 160.0
        assert out[VAL] == 30.0
        assert out[SAVE] == 10.0

    def test_validation_step_trains_as_much_as_a_plain_step(self):
        """The bug's symptom: a validation step used to contribute ~0 training time."""
        plain = compute_cumulative_timing_metrics({}, {"step": 1439.0})
        with_val = compute_cumulative_timing_metrics({}, {"step": 1439.0, "testing": 1438.7, "save_checkpoint": 46.6})
        assert with_val[TRAIN] == plain[TRAIN]

    def test_other_timing_keys_are_ignored(self):
        """gen/update_actor/old_log_prob are parts of the step, not separate terms."""
        out = compute_cumulative_timing_metrics(
            {}, {"step": 100.0, "gen": 40.0, "update_actor": 50.0, "old_log_prob": 5.0, "reward": 2.0}
        )
        assert out[TRAIN] == 100.0
        assert out[WALL] == 100.0

    def test_missing_step_counts_as_zero(self):
        out = compute_cumulative_timing_metrics({}, {"testing": 5.0})
        assert out[WALL] == 5.0
        assert out[TRAIN] == 0.0
        assert out[VAL] == 5.0
        assert out[SAVE] == 0.0

    def test_empty_timing_raw_is_a_noop(self):
        state = {}
        compute_cumulative_timing_metrics(state, {"step": 10.0})
        out = compute_cumulative_timing_metrics(state, {})
        assert out[TRAIN] == 10.0 and out[WALL] == 10.0

    def test_values_are_floats_even_from_ints(self):
        out = compute_cumulative_timing_metrics({}, {"step": 3, "testing": 1, "save_checkpoint": 1})
        assert all(isinstance(out[k], float) for k in (WALL, VAL, SAVE, TRAIN))
        assert out[WALL] == 5.0


class TestIdentity:
    def test_training_equals_wall_minus_validation_minus_save(self):
        """The identity the val-vs-time overlays rely on (the async trainer holds it in
        stop-the-world mode; the sync trainer must hold it exactly, always)."""
        rng = random.Random(20260904)
        state = {}
        prev_train = 0.0
        for _ in range(500):
            raw = {"step": rng.uniform(0.0, 2000.0)}
            if rng.random() < 0.3:
                raw["testing"] = rng.uniform(0.0, 2000.0)
            if rng.random() < 0.3:
                raw["save_checkpoint"] = rng.uniform(0.0, 100.0)
            out = compute_cumulative_timing_metrics(state, raw)
            assert out[TRAIN] == pytest.approx(out[WALL] - out[VAL] - out[SAVE], abs=1e-6)
            assert out[TRAIN] >= prev_train
            prev_train = out[TRAIN]

    def test_training_is_the_sum_of_steps(self):
        rng = random.Random(1)
        triples = [(rng.uniform(1, 100), rng.uniform(0, 50), rng.uniform(0, 5)) for _ in range(50)]
        out = _replay(triples)
        assert out[TRAIN] == pytest.approx(sum(t[0] for t in triples))
        assert out[VAL] == pytest.approx(sum(t[1] for t in triples))
        assert out[SAVE] == pytest.approx(sum(t[2] for t in triples))
        assert out[WALL] == pytest.approx(sum(sum(t) for t in triples))


class TestAccumulation:
    def test_counters_accumulate_across_steps(self):
        state = {}
        compute_cumulative_timing_metrics(state, {"step": 100.0, "testing": 20.0})
        compute_cumulative_timing_metrics(state, {"step": 50.0})
        out = compute_cumulative_timing_metrics(state, {"step": 60.0, "save_checkpoint": 15.0})
        assert out[WALL] == pytest.approx(245.0)
        assert out[VAL] == pytest.approx(20.0)
        assert out[SAVE] == pytest.approx(15.0)
        assert out[TRAIN] == pytest.approx(210.0)

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

    def test_state_is_updated_in_place_and_mirrored_in_the_output(self):
        state = {}
        out = compute_cumulative_timing_metrics(state, {"step": 4.0, "testing": 2.0, "save_checkpoint": 1.0})
        assert state == {"wall": 7.0, "validation": 2.0, "save": 1.0, "training": 4.0}
        assert out == {WALL: 7.0, VAL: 2.0, SAVE: 1.0, TRAIN: 4.0}


class TestRemoteH100Replay:
    """Regression against the real run that exposed the bug."""

    def test_training_time_after_step_9_is_the_sum_of_steps(self):
        out = _replay(REMOTE_H100_STEPS)
        expected = sum(s for s, _, _ in REMOTE_H100_STEPS)  # 13,710.4 s
        assert out[TRAIN] == pytest.approx(expected, abs=1e-3)
        assert out[TRAIN] > LOGGED_TRAINING_TIME_WITH_OLD_HELPER + 4000.0

    def test_step_3_increment_is_step_3_itself(self):
        """With the old helper the increment at step 3 was exactly 0 (the clamp)."""
        state = {}
        _replay(REMOTE_H100_STEPS[:2], state)
        before = state["training"]
        _replay(REMOTE_H100_STEPS[2:3], state)
        assert state["training"] - before == pytest.approx(REMOTE_H100_STEPS[2][0], abs=1e-6)

    def test_validation_and_save_totals_and_identity(self):
        out = _replay(REMOTE_H100_STEPS)
        assert out[VAL] == pytest.approx(1438.682 + 1391.846 + 1390.216, abs=1e-3)
        assert out[SAVE] == pytest.approx(46.613 + 50.408 + 48.321, abs=1e-3)
        assert out[WALL] == pytest.approx(out[TRAIN] + out[VAL] + out[SAVE], abs=1e-6)


def _with_nodes(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.With)]


def _marked_timer_name(with_node):
    """The first positional string argument of a ``marked_timer(...)`` context, if any."""
    for item in with_node.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "marked_timer":
            if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                return call.args[0].value
    return None


@pytest.fixture(scope="module")
def fit_tree():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    return ast.parse(textwrap.dedent(inspect.getsource(RayPPOTrainer.fit)))


class TestTimerLayoutTripwire:
    """The helper adds step + testing + save because fit() times validation and saving
    outside the step timer. Pin that layout: if either block moves back inside the
    step (upstream verl's layout for the save), the helper must subtract it again."""

    def test_step_testing_and_save_timers_exist(self, fit_tree):
        names = {_marked_timer_name(w) for w in _with_nodes(fit_tree)}
        assert {"step", "testing", "save_checkpoint"} <= names, names

    def test_testing_and_save_are_not_inside_the_step_timer(self, fit_tree):
        step_withs = [w for w in _with_nodes(fit_tree) if _marked_timer_name(w) == "step"]
        assert len(step_withs) == 1
        inside = {_marked_timer_name(w) for w in _with_nodes(step_withs[0]) if w is not step_withs[0]}
        for name in ("testing", "save_checkpoint"):
            assert name not in inside, (
                f"marked_timer({name!r}) moved inside the step timer: "
                "compute_cumulative_timing_metrics must subtract it from the step again"
            )

    def test_gen_and_update_actor_are_inside_the_step_timer(self, fit_tree):
        """The other half of the layout: the step really is the training work."""
        step_with = next(w for w in _with_nodes(fit_tree) if _marked_timer_name(w) == "step")
        inside = {_marked_timer_name(w) for w in _with_nodes(step_with)}
        assert {"gen", "update_actor"} <= inside, inside


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

    def test_helper_is_called_after_validation_and_save_in_the_same_iteration(self):
        """It must see this iteration's testing/save_checkpoint durations: the call has
        to come after both blocks in fit()'s source order."""
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        src = inspect.getsource(RayPPOTrainer.fit)
        call = src.index("compute_cumulative_timing_metrics(self._cumulative_timing, timing_raw)")
        assert src.index('marked_timer("testing"') < call
        assert src.index('marked_timer("save_checkpoint"') < call
