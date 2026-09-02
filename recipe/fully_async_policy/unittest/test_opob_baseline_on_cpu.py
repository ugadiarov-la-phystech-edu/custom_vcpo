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
"""OPOB (VCPO's optimal off-policy baseline, arXiv:2602.17616 Eq. 7/13) on CPU.

    b* = sum_i W_i R_i / sum_i W_i,   W_i = ||g_i||^2 * w_i^2 (* 1/L_i^2)

Covers the pure helpers in ``staleness_utils`` (weights, baseline aggregation
modes, scope selection, the per-group diagnostics), the actor's
``_update_grad_buffers`` contract (buffer scales with/without ``norm_by_std``,
the ``-b*`` move on the last trajectory of a group, the diagnostics record)
and the metrics plumbing that turns ``actor/opob_records`` into the ``opob/*``
scalars.

Run: pytest recipe/fully_async_policy/unittest/test_opob_baseline_on_cpu.py
"""

from types import SimpleNamespace

import pytest

import verl.workers.actor.megatron_actor as megatron_actor
from recipe.fully_async_policy.detach_utils import process_structured_metrics
from recipe.fully_async_policy.staleness_utils import (
    TrajRecord,
    TrajRecordList,
    compute_opob_baseline,
    compute_opob_weights,
    summarize_opob_group,
)
from verl.utils.metric.utils import reduce_metrics
from verl.workers.actor.megatron_actor import MegatronPPOActor


def _record(uid, group_uid, reward, grad_norm, seq_is, seq_is_clipped=None, length=4, advantage=None):
    rec = TrajRecord(
        uid=uid,
        group_uid=group_uid,
        epoch_idx=0,
        minibatch_idx=0,
        trainer_global_step=0,
        trainer_local_step=0,
        param_version_start=0,
        param_version_end=0,
        trainer_param_version=0,
        response_length=length,
        prompt_length=2,
        advantage_scalar=reward if advantage is None else advantage,
        reward_scalar=reward,
    )
    rec.grad_norm_unscaled = grad_norm
    rec.rollout_seq_is = seq_is
    rec.rollout_seq_is_clipped = seq_is if seq_is_clipped is None else seq_is_clipped
    return rec


def _group(uid="g0", rewards=(1.0, -1.0, 1.0, -1.0), grad_norms=None, seq_is=None):
    grad_norms = (1.0,) * len(rewards) if grad_norms is None else grad_norms
    seq_is = (1.0,) * len(rewards) if seq_is is None else seq_is
    records = TrajRecordList()
    for i, (r, g, w) in enumerate(zip(rewards, grad_norms, seq_is, strict=True)):
        records.append(_record(f"{uid}-t{i}", uid, r, g, w))
    return records


# ------------------------------------------------------------------ compute_opob_weights


class TestOpobWeights:
    def test_weights_are_gradnorm_squared_times_is_squared(self):
        records = _group(rewards=(1.0, -1.0), grad_norms=(2.0, 3.0), seq_is=(0.5, 4.0))
        values, weights = compute_opob_weights(records, "g0")
        assert values == [1.0, -1.0]
        assert weights == pytest.approx([4.0 * 0.25, 9.0 * 16.0])

    def test_use_is_weights_false_keeps_gradnorm_only(self):
        records = _group(rewards=(1.0, -1.0), grad_norms=(2.0, 3.0), seq_is=(0.5, 4.0))
        _, weights = compute_opob_weights(records, "g0", use_is_weights=False)
        assert weights == pytest.approx([4.0, 9.0])

    def test_clipped_ratio_variant_uses_the_clipped_field(self):
        records = TrajRecordList([_record("t0", "g0", 1.0, 1.0, seq_is=100.0, seq_is_clipped=2.0)])
        _, unclipped = compute_opob_weights(records, "g0")
        _, clipped = compute_opob_weights(records, "g0", use_clipped_is_ratios=True)
        assert unclipped == pytest.approx([1e4])
        assert clipped == pytest.approx([4.0])

    def test_normalize_by_length_divides_by_length_squared(self):
        records = TrajRecordList([_record("t0", "g0", 1.0, 3.0, seq_is=1.0, length=3)])
        _, weights = compute_opob_weights(records, "g0", normalize_by_length=True)
        assert weights == pytest.approx([9.0 / 9.0])

    def test_group_scope_filters_other_groups(self):
        records = _group("g0", rewards=(1.0, -1.0)) + _group("g1", rewards=(0.0, 0.0))
        values, weights = compute_opob_weights(TrajRecordList(records), "g1")
        assert values == [0.0, 0.0]
        assert len(weights) == 2

    def test_minibatch_scope_takes_every_record_and_the_advantage(self):
        records = TrajRecordList(
            [
                _record("t0", "g0", reward=1.0, grad_norm=1.0, seq_is=1.0, advantage=0.7),
                _record("t1", "g1", reward=-1.0, grad_norm=1.0, seq_is=1.0, advantage=-0.3),
            ]
        )
        values, _ = compute_opob_weights(records, "g0", scope="minibatch")
        assert values == pytest.approx([0.7, -0.3])

    def test_missing_group_yields_empty_lists(self):
        values, weights = compute_opob_weights(_group(), "nope")
        assert values == [] and weights == []


# ------------------------------------------------------------------ compute_opob_baseline


class TestOpobBaseline:
    def test_uniform_weights_reduce_to_the_group_mean(self):
        records = _group(rewards=(1.0, 1.0, -1.0, -1.0))
        assert compute_opob_baseline(records, "g0") == pytest.approx(0.0, abs=1e-7)

    def test_is_weight_spread_makes_the_baseline_an_argmax(self):
        """w spanning 1e-6..1e6 (the replay regime): b* -> reward of the dominant trajectory."""
        records = _group(rewards=(1.0, -1.0, -1.0, -1.0), seq_is=(1e6, 1.0, 1e-3, 1e-6))
        assert compute_opob_baseline(records, "g0") == pytest.approx(1.0, abs=1e-9)

    def test_gradient_norm_dominance_without_is_weights(self):
        records = _group(rewards=(1.0, -1.0), grad_norms=(1e3, 1.0), seq_is=(1e-3, 1e3))
        assert compute_opob_baseline(records, "g0", use_is_weights=False) == pytest.approx(1.0, abs=1e-5)
        assert compute_opob_baseline(records, "g0", use_is_weights=True) == pytest.approx(-1.0, abs=1e-5)

    def test_median_and_winsorized_soften_the_argmax(self):
        records = _group(rewards=(1.0, -1.0, -1.0, -1.0), seq_is=(10.0, 1.0, 1.0, 1.0))
        mean = compute_opob_baseline(records, "g0", agg_mode="mean")
        median = compute_opob_baseline(records, "g0", agg_mode="median")
        winsor = compute_opob_baseline(records, "g0", agg_mode="winsorized_mean")
        assert mean == pytest.approx((100.0 - 3.0) / 103.0)
        assert median == pytest.approx(1.0)  # weight 100 vs 3: the median sits on the dominant trajectory
        assert -1.0 <= winsor <= 1.0

    def test_unknown_agg_mode_raises(self):
        with pytest.raises(NotImplementedError):
            compute_opob_baseline(_group(), "g0", agg_mode="mode")

    def test_minibatch_scope_baselines_the_advantages_over_all_groups(self):
        records = TrajRecordList(
            [
                _record("t0", "g0", reward=1.0, grad_norm=1.0, seq_is=1.0, advantage=2.0),
                _record("t1", "g1", reward=-1.0, grad_norm=1.0, seq_is=1.0, advantage=-1.0),
            ]
        )
        assert compute_opob_baseline(records, "g0", scope="minibatch") == pytest.approx(0.5, abs=1e-7)

    def test_all_zero_weights_fall_back_to_zero(self):
        records = _group(rewards=(1.0, 1.0), grad_norms=(0.0, 0.0))
        assert compute_opob_baseline(records, "g0") == 0.0

    def test_binary_rewards_with_norm_by_std_give_gr_po_scaled_advantages(self):
        """What the actor applies: (R - b*) / std. With b* at the dominant +1, the
        negatives get -2/std = -4 (std=0.5) and the positive gets 0."""
        records = _group(rewards=(1.0, -1.0, -1.0, -1.0), seq_is=(1e6, 1.0, 1.0, 1.0))
        b = compute_opob_baseline(records, "g0")
        std = 0.5 * (3.0**0.5)  # population std of (1,-1,-1,-1)
        effective = [(r - b) / std for r in (1.0, -1.0, -1.0, -1.0)]
        assert effective[0] == pytest.approx(0.0, abs=1e-6)
        assert effective[1] == pytest.approx(-2.0 / std, abs=1e-6)


# ------------------------------------------------------------------ summarize_opob_group


class TestSummarizeOpobGroup:
    def test_argmax_regime(self):
        values, weights = [1.0, -1.0, -1.0], [1e12, 1.0, 1.0]
        s = summarize_opob_group(values, weights, baseline=1.0)
        assert s["weight_conc"] == pytest.approx(1.0, abs=1e-9)
        assert s["dominant_reward"] == 1.0
        assert s["zeroed_frac"] == pytest.approx(1 / 3)
        assert s["n"] == 3 and s["baseline"] == 1.0

    def test_uniform_regime(self):
        s = summarize_opob_group([1.0, -1.0, 1.0, -1.0], [1.0] * 4, baseline=0.0)
        assert s["weight_conc"] == pytest.approx(0.25)
        assert s["zeroed_frac"] == 0.0

    def test_zero_tolerance_controls_zeroed_frac(self):
        s = summarize_opob_group([0.05, 0.5], [1.0, 1.0], baseline=0.0, zero_tol=0.1)
        assert s["zeroed_frac"] == 0.5
        s = summarize_opob_group([0.05, 0.5], [1.0, 1.0], baseline=0.0, zero_tol=1.0)
        assert s["zeroed_frac"] == 1.0

    def test_all_zero_weights_report_uniform_concentration(self):
        s = summarize_opob_group([1.0, -1.0], [0.0, 0.0], baseline=0.0)
        assert s["weight_conc"] == pytest.approx(0.5)

    def test_empty_scope(self):
        s = summarize_opob_group([], [], baseline=0.0)
        assert s == {"baseline": 0.0, "weight_conc": 0.0, "dominant_reward": 0.0, "zeroed_frac": 0.0, "n": 0}

    def test_accepts_tensor_like_baseline(self):
        import torch

        s = summarize_opob_group([1.0], [1.0], baseline=torch.tensor(1.0))
        assert isinstance(s["baseline"], float) and s["baseline"] == 1.0

    def test_grad_norm_stats_reported_when_given(self):
        s = summarize_opob_group([1.0, -1.0, -1.0], [1.0] * 3, baseline=0.0, grad_norms=[0.5, 4.0, None])
        assert s["traj_grad_norm_max"] == 4.0
        assert s["traj_grad_norm_mean"] == pytest.approx(2.25)  # None entries skipped
        s = summarize_opob_group([1.0], [1.0], baseline=0.0)
        assert "traj_grad_norm_max" not in s and "traj_grad_norm_mean" not in s
        s = summarize_opob_group([], [], baseline=0.0, grad_norms=[3.0])
        assert s["n"] == 0 and s["traj_grad_norm_max"] == 3.0


# ------------------------------------------------------------------ debug helpers


class TestOpobDebugHelpers:
    def test_grad_buffers_norm_combines_per_buffer_norms(self):
        import torch

        from verl.workers.utils.vcpo import grad_buffers_norm

        bufs = [torch.ones(4, dtype=torch.bfloat16), torch.full((4,), 2.0, dtype=torch.bfloat16)]
        assert grad_buffers_norm(bufs) == pytest.approx((4 + 16) ** 0.5)
        assert grad_buffers_norm([]) == 0.0

    def test_chunked_add_matches_foreach(self, monkeypatch):
        import torch

        from verl.workers.utils import vcpo

        monkeypatch.setattr(vcpo, "_ADD_CHUNK", 5)  # force several chunks per tensor
        dest = [torch.arange(12, dtype=torch.float32), torch.ones(3)]
        src = [torch.ones(12), torch.arange(3, dtype=torch.float32)]
        ref = [d.clone() for d in dest]
        torch._foreach_add_(ref, src, alpha=-1.5)
        vcpo._add_lists_(dest, src, alpha=-1.5)
        for d, r in zip(dest, ref, strict=True):
            assert torch.equal(d, r)

    def test_chunked_add_switch_is_used_by_accumulate_and_move(self, monkeypatch):
        import torch
        from types import SimpleNamespace

        from verl.workers.utils import vcpo

        monkeypatch.setenv("VCPO_OPOB_CHUNKED_ADD", "1")
        called = []
        monkeypatch.setattr(vcpo, "_add_lists_", lambda d, s, alpha: called.append(("add", alpha)))
        monkeypatch.setattr(vcpo.torch, "_foreach_add_", lambda *a, **k: called.append(("foreach",)))
        buf = SimpleNamespace(grad_data=torch.zeros(4))
        module = SimpleNamespace(buffers=[buf])
        gpu_dest = [SimpleNamespace(is_cuda=True)]  # device accumulators take the chunked/foreach paths
        vcpo.accumulate_grad_buffers([module], [torch.zeros(4)], scale=2.0)
        vcpo.move_grad_buffers([torch.zeros(4)], gpu_dest, scale=-0.5)
        assert called == [("add", 2.0), ("add", -0.5)]
        monkeypatch.setenv("VCPO_OPOB_CHUNKED_ADD", "0")
        vcpo.move_grad_buffers([torch.zeros(4)], gpu_dest, scale=1.0)
        assert called[-1] == ("foreach",)

    def test_chunked_add_is_the_default(self, monkeypatch):
        from verl.workers.utils import vcpo

        monkeypatch.delenv("VCPO_OPOB_CHUNKED_ADD", raising=False)
        assert vcpo._chunked_add_enabled()  # foreach is opt-in (diagnosis only)
        monkeypatch.setenv("VCPO_OPOB_CHUNKED_ADD", "0")
        assert not vcpo._chunked_add_enabled()

    def test_debug_flag_reads_env(self, monkeypatch):
        from verl.workers.utils.vcpo import _opob_debug_enabled

        monkeypatch.delenv("VCPO_OPOB_DEBUG", raising=False)
        assert not _opob_debug_enabled()
        monkeypatch.setenv("VCPO_OPOB_DEBUG", "0")
        assert not _opob_debug_enabled()
        monkeypatch.setenv("VCPO_OPOB_DEBUG", "1")
        assert _opob_debug_enabled()


# ------------------------------------------------------------------ actor._update_grad_buffers


def _make_buffer_actor(monkeypatch, scope="group", norm_by_std=False, **cfg):
    actor = MegatronPPOActor.__new__(MegatronPPOActor)
    actor.config = SimpleNamespace(
        grad_baselining=SimpleNamespace(
            scope=scope,
            agg_mode=cfg.get("agg_mode", "mean"),
            use_is_weights=cfg.get("use_is_weights", True),
            use_clipped_is_ratios=cfg.get("use_clipped_is_ratios", False),
            normalize_by_length=cfg.get("normalize_by_length", False),
            norm_by_std=norm_by_std,
        )
    )
    actor.actor_module = [object()]
    calls = {"accumulate": [], "move": [], "zeroed": []}
    accum, score = object(), object()

    def fake_accumulate(modules, bufs, scale):
        calls["accumulate"].append((bufs, scale))

    monkeypatch.setattr(megatron_actor, "accumulate_grad_buffers", fake_accumulate)
    monkeypatch.setattr(
        megatron_actor, "move_grad_buffers", lambda src, dest, scale: calls["move"].append((src, dest, scale))
    )
    monkeypatch.setattr(megatron_actor, "zero_grad_accum_buffers", lambda bufs: calls["zeroed"].append(bufs))
    return actor, calls, accum, score, norm_by_std


def _run_group(actor, calls, accum, score, norm_by_std, records, reward_std, opob_records):
    for i, rec in enumerate(records):
        actor._update_grad_buffers(
            accum_buffers=accum,
            score_gradient_buffers=score,
            local_traj_records=records,
            reward_scalar=rec.reward_scalar,
            reward_std=reward_std,
            adv_scalar=rec.advantage_scalar,
            group_uid=rec.group_uid,
            microbatch_loss_scale=1.0,
            norm_by_std=norm_by_std,
            is_last_traj_in_scope=(i == len(records) - 1),
            grad_baselining=True,
            opob_records=opob_records,
        )


class TestUpdateGradBuffersOpob:
    REWARDS = (1.0, -1.0, -1.0, -1.0)
    SEQ_IS = (1e6, 1.0, 1.0, 1.0)

    def test_raw_scales_and_baseline_move_without_norm_by_std(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch)
        records = _group(rewards=self.REWARDS, seq_is=self.SEQ_IS)
        opob_records = []
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.5, opob_records=opob_records)
        # Per trajectory: g_i * R_i into accum, g_i * 1 into score.
        assert [s for b, s in calls["accumulate"] if b is accum] == pytest.approx(list(self.REWARDS))
        assert [s for b, s in calls["accumulate"] if b is score] == [1] * 4
        # Only the last trajectory closes the scope: one move with -b*, one zero of the score buffer.
        expected_b = compute_opob_baseline(records, "g0")
        assert len(calls["move"]) == 1
        src, dest, scale = calls["move"][0]
        assert src is score and dest is accum and scale == pytest.approx(-expected_b)
        assert calls["zeroed"] == [score]
        assert expected_b == pytest.approx(1.0, abs=1e-9)

    def test_norm_by_std_scales_reward_and_baseline_by_one_over_std(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch, norm_by_std=True)
        records = _group(rewards=self.REWARDS, seq_is=self.SEQ_IS)
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.5, opob_records=[])
        assert [s for b, s in calls["accumulate"] if b is accum] == pytest.approx([r / 0.5 for r in self.REWARDS])
        assert [s for b, s in calls["accumulate"] if b is score] == pytest.approx([2.0] * 4)
        # The baseline itself is over raw rewards; the 1/std lives in the score buffer scale.
        assert calls["move"][0][2] == pytest.approx(-1.0, abs=1e-9)

    def test_norm_by_std_with_degenerate_std_falls_back_to_raw_scales(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch, norm_by_std=True)
        records = _group(rewards=(1.0, 1.0), seq_is=(1.0, 1.0))
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.0, opob_records=[])
        assert [s for b, s in calls["accumulate"] if b is accum] == [1.0, 1.0]
        assert [s for b, s in calls["accumulate"] if b is score] == [1, 1]

    def test_minibatch_scope_uses_the_advantage_scale(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch, scope="minibatch", norm_by_std=True)
        records = TrajRecordList(
            [
                _record("t0", "g0", reward=1.0, grad_norm=1.0, seq_is=1.0, advantage=2.0),
                _record("t1", "g1", reward=-1.0, grad_norm=1.0, seq_is=1.0, advantage=-1.0),
            ]
        )
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.5, opob_records=[])
        assert [s for b, s in calls["accumulate"] if b is accum] == [2.0, -1.0]
        assert [s for b, s in calls["accumulate"] if b is score] == [1, 1]
        assert calls["move"][0][2] == pytest.approx(-0.5, abs=1e-7)

    def test_diagnostics_record_emitted_once_per_closed_scope(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch)
        records = _group(rewards=self.REWARDS, seq_is=self.SEQ_IS)
        opob_records = []
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.5, opob_records=opob_records)
        assert len(opob_records) == 1
        rec = opob_records[0]
        assert set(rec) == {
            "baseline",
            "weight_conc",
            "dominant_reward",
            "zeroed_frac",
            "n",
            "traj_grad_norm_max",
            "traj_grad_norm_mean",
        }
        assert rec["traj_grad_norm_max"] == 1.0 and rec["traj_grad_norm_mean"] == 1.0  # all records: norm 1.0
        assert rec["baseline"] == pytest.approx(1.0, abs=1e-9)
        assert rec["weight_conc"] == pytest.approx(1.0, abs=1e-9)
        assert rec["dominant_reward"] == 1.0
        assert rec["zeroed_frac"] == pytest.approx(0.25)
        assert rec["n"] == 4

    def test_diagnostics_follow_the_configured_weighting(self, monkeypatch):
        """use_is_weights=False: the 1e6 IS ratio no longer dominates, the baseline is the plain mean."""
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch, use_is_weights=False)
        records = _group(rewards=self.REWARDS, seq_is=self.SEQ_IS)
        opob_records = []
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.5, opob_records=opob_records)
        assert opob_records[0]["baseline"] == pytest.approx(-0.5)
        assert opob_records[0]["weight_conc"] == pytest.approx(0.25)
        assert calls["move"][0][2] == pytest.approx(0.5)

    def test_no_records_list_means_no_diagnostics(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch)
        records = _group(rewards=self.REWARDS, seq_is=self.SEQ_IS)
        _run_group(actor, calls, accum, score, nbs, records, reward_std=0.5, opob_records=None)
        assert len(calls["move"]) == 1  # the baseline is still applied

    def test_buffer_free_path_untouched(self, monkeypatch):
        actor, calls, accum, score, nbs = _make_buffer_actor(monkeypatch)
        opob_records = []
        actor._update_grad_buffers(
            accum_buffers=accum,
            score_gradient_buffers=None,
            local_traj_records=_group(),
            reward_scalar=1.0,
            reward_std=0.5,
            adv_scalar=0.75,
            group_uid="g0",
            microbatch_loss_scale=1.0,
            is_last_traj_in_scope=True,
            grad_baselining=False,
            opob_records=opob_records,
        )
        assert calls["accumulate"] == [(accum, 0.75)]
        assert calls["move"] == [] and calls["zeroed"] == [] and opob_records == []


# ------------------------------------------------------------------ metrics plumbing


class TestOpobMetricsPlumbing:
    ENTRIES = [
        {"baseline": 1.0, "weight_conc": 0.99, "dominant_reward": 1.0, "zeroed_frac": 0.25, "n": 4},
        {"baseline": -1.0, "weight_conc": 0.9, "dominant_reward": -1.0, "zeroed_frac": 0.75, "n": 4},
        {"baseline": 0.0, "weight_conc": 0.25, "dominant_reward": 1.0, "zeroed_frac": 0.0, "n": 4},
    ]

    def test_reduce_metrics_keeps_opob_records_structured(self):
        """Per-worker lists of dicts are flattened, never mean-reduced like scalars."""
        metrics = {"actor/opob_records": [self.ENTRIES[:2], self.ENTRIES[2:]], "actor/grad_norm": [1.0, 3.0]}
        out = reduce_metrics(metrics)
        assert out["actor/opob_records"] == self.ENTRIES
        assert out["actor/grad_norm"] == 2.0

    def test_process_structured_metrics_emits_opob_scalars(self):
        payload = process_structured_metrics({"actor/opob_records": self.ENTRIES}, allow_media=False)
        assert payload["opob/baseline_mean"] == pytest.approx(0.0)
        assert payload["opob/baseline_abs_mean"] == pytest.approx(2 / 3)
        assert payload["opob/weight_conc_mean"] == pytest.approx((0.99 + 0.9 + 0.25) / 3)
        assert payload["opob/dominant_pos_frac"] == pytest.approx(2 / 3)
        assert payload["opob/zeroed_frac"] == pytest.approx(1 / 3)
        assert payload["opob/groups"] == 3

    def test_process_structured_metrics_ignores_malformed_entries(self):
        payload = process_structured_metrics(
            {"actor/opob_records": ["junk", {"baseline": None}, {"weight_conc": 0.5}]}, allow_media=False
        )
        assert "opob/baseline_mean" not in payload
        assert payload["opob/weight_conc_mean"] == 0.5

    def test_process_structured_metrics_reduces_traj_grad_norms(self):
        entries = [
            {"baseline": 0.0, "traj_grad_norm_max": 2.0, "traj_grad_norm_mean": 1.0},
            {"baseline": 0.0, "traj_grad_norm_max": 8.0, "traj_grad_norm_mean": 3.0},
            {"baseline": 0.0},  # a group without norms is skipped for the norm scalars
        ]
        payload = process_structured_metrics({"actor/opob_records": entries}, allow_media=False)
        assert payload["opob/traj_grad_norm_max"] == 8.0  # max over groups
        assert payload["opob/traj_grad_norm_mean"] == pytest.approx(2.0)  # mean of group means
        assert payload["opob/groups"] == 3

    def test_process_structured_metrics_without_opob_key_emits_nothing(self):
        payload = process_structured_metrics({"staleness/ess": []}, allow_media=False)
        assert not any(k.startswith("opob/") for k in payload)
