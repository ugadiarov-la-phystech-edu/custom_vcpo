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
"""Rollouter concurrency ramp (async_training.concurrency_ramp).

The rollouter's in-flight cap is n_engines x bsz_per_dp_rank from the first second, so the first
mini-batch's 8k-token tails decode at ~528-sequence-per-engine speed (~7 min). The ramp holds a small
cap until the trainer's first mini-batch has been DELIVERED to the queue, then widens stage by stage
(one mini-batch of deliveries per stage) to the normal cap. Run:
    python -m pytest recipe/fully_async_policy/unittest/test_concurrency_ramp_on_cpu.py
"""

import inspect

import pytest
from omegaconf import OmegaConf

from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from recipe.fully_async_policy.replay_sizing import (
    concurrency_cap,
    first_minibatch_groups,
    parse_concurrency_ramp,
)


def _unwrap(actor_cls):
    return (
        getattr(actor_cls, "__ray_metadata__", None).modified_class
        if hasattr(actor_cls, "__ray_metadata__")
        else actor_cls
    )


FullyAsyncRollouter = _unwrap(_RollouterActor)

# The 5+3 arm: 5 engines, mini-batch 33 groups, first mini-batch 18 (requires_mini_batches=0.5),
# full cap 5 x 33 = 165, staleness quota 33 x (32 + 1) = 1089.
ENGINES, FIRST, MINI, FULL, HARD = 5, 18, 33, 165, 1089
RAMP = [4, 8, 16]


class TestParseConcurrencyRamp:
    @pytest.mark.parametrize("off", [None, "", "null", "None", "[]", []])
    def test_off_forms(self, off):
        assert parse_concurrency_ramp(off) == []

    @pytest.mark.parametrize(
        "value",
        [[4, 8, 16], (4, 8, 16), "[4,8,16]", "[4, 8, 16]", "4,8,16", OmegaConf.create([4, 8, 16]), [4.0, 8.0, 16.0]],
    )
    def test_list_forms(self, value):
        assert parse_concurrency_ramp(value) == [4, 8, 16]

    def test_single_stage_and_plateaus_are_fine(self):
        assert parse_concurrency_ramp("[4]") == [4]
        assert parse_concurrency_ramp([4, 4, 8]) == [4, 4, 8]

    @pytest.mark.parametrize("bad", [[0, 4], [-1], [4, 2], "[4,x]", [2.5], 7])
    def test_rejects_non_positive_decreasing_or_non_int(self, bad):
        with pytest.raises(ValueError, match="concurrency_ramp"):
            parse_concurrency_ramp(bad)


class TestConcurrencyCap:
    @pytest.mark.parametrize(
        "delivered, expected",
        [(0, 20), (17, 20), (18, 40), (50, 40), (51, 80), (83, 80), (84, 165), (10_000, 165)],
    )
    def test_stage_table_for_the_5plus3_arm(self, delivered, expected):
        """Stage 0 until the 18-group first mini-batch is delivered, then one full mini-batch (33) per stage."""
        assert concurrency_cap(delivered, RAMP, ENGINES, FIRST, MINI, FULL, HARD) == expected

    def test_thresholds_follow_the_full_first_minibatch_when_rmb_is_at_least_one(self):
        first = first_minibatch_groups(1.0, MINI, 16, 3) or MINI
        assert first == MINI
        caps = [concurrency_cap(d, RAMP, ENGINES, first, MINI, FULL, HARD) for d in (0, 32, 33, 65, 66, 98, 99)]
        assert caps == [20, 20, 40, 40, 80, 80, 165]

    @pytest.mark.parametrize("delivered", [0, 18, 84, 10_000])
    def test_ramp_off_is_the_full_cap(self, delivered):
        assert concurrency_cap(delivered, [], ENGINES, FIRST, MINI, FULL, HARD) == FULL

    def test_hard_cap_bounds_every_stage(self):
        assert concurrency_cap(0, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=15) == 15
        assert concurrency_cap(18, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=30) == 30
        assert concurrency_cap(84, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=30) == 30
        assert concurrency_cap(0, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=None) == 20

    def test_single_stage_ramp(self):
        assert concurrency_cap(0, [4], ENGINES, FIRST, MINI, FULL, HARD) == 20
        assert concurrency_cap(18, [4], ENGINES, FIRST, MINI, FULL, HARD) == FULL

    def test_cap_is_monotone_in_deliveries_and_never_below_one(self):
        prev = 0
        for d in range(0, 300):
            cap = concurrency_cap(d, RAMP, ENGINES, FIRST, MINI, FULL, HARD)
            assert cap >= max(1, prev)
            prev = cap
        assert concurrency_cap(0, [1], 1, 1, 1, 1, hard_cap=1) == 1

    def test_first_stage_covers_the_first_minibatch_in_one_wave(self):
        """Design rule for choosing ramp[0]: ramp[0] x n_engines >= first mini-batch, else a second wave."""
        assert RAMP[0] * ENGINES >= FIRST


class _Handles:
    def __init__(self, n):
        self.server_handles = list(range(n))


def _make_rollouter(ramp, delivered=0, n_engines=ENGINES, first=FIRST, mini=MINI, full=FULL, hard=HARD):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.concurrency_ramp = parse_concurrency_ramp(ramp)
    r.ramp_first_size = first
    r.required_samples = mini
    r.max_concurrent_samples = full
    r.full_concurrent_samples = full
    r.max_required_samples = hard
    r.total_generated_samples = delivered
    r._ramp_stage_logged = None
    r.async_rollout_manager = _Handles(n_engines)
    return r


class TestRollouterCap:
    def test_follows_deliveries(self):
        r = _make_rollouter("[4,8,16]")
        seen = []
        for delivered in (0, 17, 18, 51, 84, 200):
            r.total_generated_samples = delivered
            seen.append(r._concurrency_cap())
        assert seen == [20, 20, 40, 80, 165, 165]

    def test_restored_run_starts_at_the_full_cap(self):
        """total_generated_samples is restored from the queue checkpoint -> no warm-up after a resume."""
        assert _make_rollouter("[4,8,16]", delivered=500)._concurrency_cap() == FULL

    def test_ramp_off_returns_max_concurrent_samples(self):
        r = _make_rollouter(None)
        assert r._concurrency_cap() == FULL
        r.total_generated_samples = 5
        assert r._concurrency_cap() == FULL

    def test_hard_cap_applies(self):
        assert _make_rollouter("[4,8,16]", hard=30)._concurrency_cap() == 20
        assert _make_rollouter("[4,8,16]", delivered=18, hard=30)._concurrency_cap() == 30

    def test_stage_change_is_logged_once(self, capsys):
        r = _make_rollouter("[4,8,16]")
        r._concurrency_cap()
        r._concurrency_cap()
        r.total_generated_samples = 18
        r._concurrency_cap()
        out = capsys.readouterr().out
        assert out.count("concurrency cap -> 20 groups") == 1
        assert out.count("concurrency cap -> 40 groups") == 1


class TestWiring:
    def test_dispatch_loop_uses_the_dynamic_cap(self):
        import recipe.fully_async_policy.fully_async_rollouter as mod

        src = inspect.getsource(mod)
        assert "while len(self.active_tasks) >= self._concurrency_cap():" in src
        assert "while len(self.active_tasks) >= self.max_concurrent_samples:" not in src
        assert '"concurrency_cap": self._concurrency_cap(),' in src
        assert "self.full_concurrent_samples = self.max_concurrent_samples" in src

    def test_trainer_still_exports_the_sizing_helpers(self):
        from recipe.fully_async_policy import fully_async_trainer as t

        assert t.first_minibatch_groups is first_minibatch_groups
        assert callable(t.trainer_dp_size)

    def test_configs_document_the_knob_as_off(self):
        import os

        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("fully_async_ppo_megatron_trainer.yaml", "fully_async_ppo_trainer.yaml"):
            cfg = OmegaConf.load(os.path.join(here, "..", "config", name))
            assert cfg.async_training.concurrency_ramp is None
