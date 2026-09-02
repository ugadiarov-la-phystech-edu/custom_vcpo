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
"""Unit tests for the rollout-level |A_i|-prioritized experience replay
(arXiv:2606.04560): the PER buffer (priorities, age eviction, capacity
backstop), the pure helpers (zero-variance filter, priorities, draw sizing
under the DP-divisibility constraint), the trainer's batch composition, the
mode's config validation, and the megatron patches (mini-batch-size meta
override + entropy logging at entropy_coeff=0).

Run: pytest recipe/fully_async_policy/unittest/test_rollout_replay_on_cpu.py
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from recipe.fully_async_policy.rollout_replay import (
    AdvantagePrioritizedReplayBuffer,
    resolve_replay_draw,
    rollout_priorities,
    zero_variance_group_mask,
)
from verl import DataProto

FullyAsyncTrainer = (
    _TrainerActor.__ray_metadata__.modified_class if hasattr(_TrainerActor, "__ray_metadata__") else _TrainerActor
)

RESP_LEN = 4
SEQ_LEN = 6


def make_rows(n_rows, uid_prefix="g", scores=None, advantages=None, base_token=0):
    """A minimal post-advantage batch: n_rows rollouts of one group."""
    tensors = {
        "input_ids": torch.arange(base_token, base_token + n_rows * SEQ_LEN).reshape(n_rows, SEQ_LEN),
        "attention_mask": torch.ones(n_rows, SEQ_LEN, dtype=torch.int64),
        "position_ids": torch.arange(SEQ_LEN).repeat(n_rows, 1),
        "responses": torch.arange(base_token, base_token + n_rows * RESP_LEN).reshape(n_rows, RESP_LEN),
        "response_mask": torch.ones(n_rows, RESP_LEN, dtype=torch.int64),
        "old_log_probs": torch.zeros(n_rows, RESP_LEN),
        "rollout_log_probs": torch.zeros(n_rows, RESP_LEN),
        "token_level_scores": torch.zeros(n_rows, RESP_LEN),
        "token_level_rewards": torch.zeros(n_rows, RESP_LEN),
        "advantages": torch.zeros(n_rows, RESP_LEN),
        "returns": torch.zeros(n_rows, RESP_LEN),
    }
    if scores is not None:
        tensors["token_level_scores"][:, -1] = torch.as_tensor(scores, dtype=torch.float32)
    if advantages is not None:
        tensors["advantages"] = torch.as_tensor(advantages, dtype=torch.float32).unsqueeze(-1).expand(-1, RESP_LEN)
    non_tensors = {"uid": np.array([uid_prefix] * n_rows, dtype=object)}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def make_batch(groups, meta=None):
    """groups: list of (uid, [scores...], [advantages...])."""
    parts = []
    for i, (uid, scores, advs) in enumerate(groups):
        parts.append(make_rows(len(scores), uid_prefix=uid, scores=scores, advantages=advs, base_token=1000 * i))
    batch = DataProto.concat(parts)
    batch.meta_info.update(meta or {"temperature": 1.0})
    return batch


_BASE_TOKEN_COUNTER = [0]


def buffer_rows(n, priority=1.0, tag="b"):
    """Rows with globally unique input_ids so identity survives across blocks."""
    _BASE_TOKEN_COUNTER[0] += 10000
    rows = make_rows(n, uid_prefix=tag, advantages=[priority] * n, base_token=_BASE_TOKEN_COUNTER[0])
    return rows, np.full(n, priority, dtype=np.float64)


# ---------------------------------------------------------------------------
# AdvantagePrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class _RecordingRng:
    """Stands in for numpy's Generator: records the probability vector and
    returns the first k indices, so the PER math is checked exactly."""

    def __init__(self):
        self.last_p = None

    def choice(self, n, size, replace, p):
        self.last_p = np.asarray(p)
        return np.arange(size)


class TestBufferSampling:
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_per_probability_vector_is_p_pow_alpha_normalized(self, alpha):
        buf = AdvantagePrioritizedReplayBuffer(priority_alpha=alpha, warmup_steps=0)
        rows, _ = buffer_rows(4)
        priorities = np.array([0.1, 0.2, 0.4, 0.8])
        buf.add(rows, priorities, birth_version=0)
        buf._rng = _RecordingRng()
        buf.sample(2, current_version=0)
        expected = priorities**alpha / (priorities**alpha).sum()
        np.testing.assert_allclose(buf._rng.last_p, expected, rtol=1e-12)

    def test_no_replacement_draw_has_no_duplicates(self):
        buf = AdvantagePrioritizedReplayBuffer(seed=7)
        for step in range(3):
            rows, prios = buffer_rows(5, priority=1.0 + step, tag=f"s{step}")
            buf.add(rows, prios, birth_version=step)
        drawn, _ = buf.sample(10, current_version=3)
        ids = drawn.batch["input_ids"][:, 0].tolist()
        assert len(ids) == 10
        assert len(set(ids)) == 10

    def test_draw_larger_than_buffer_returns_whole_buffer(self):
        buf = AdvantagePrioritizedReplayBuffer()
        rows, prios = buffer_rows(3)
        buf.add(rows, prios, birth_version=0)
        drawn, info = buf.sample(50, current_version=1)
        assert len(drawn) == 3
        assert (info["ages"] == 1).all()

    def test_sampling_does_not_consume_entries(self):
        buf = AdvantagePrioritizedReplayBuffer()
        rows, prios = buffer_rows(4)
        buf.add(rows, prios, birth_version=0)
        buf.sample(4, current_version=0)
        assert buf.size() == 4
        drawn, _ = buf.sample(4, current_version=0)
        assert len(drawn) == 4

    def test_empty_or_zero_draw_returns_none(self):
        buf = AdvantagePrioritizedReplayBuffer()
        assert buf.sample(3, current_version=0)[0] is None
        rows, prios = buffer_rows(2)
        buf.add(rows, prios, birth_version=0)
        assert buf.sample(0, current_version=0)[0] is None

    def test_with_replacement_can_repeat_rows(self):
        buf = AdvantagePrioritizedReplayBuffer(with_replacement=True, seed=0)
        rows, _ = buffer_rows(2)
        # One overwhelming priority: the 2-draw repeats it with near-certainty.
        buf.add(rows, np.array([1e9, 1e-9]), birth_version=0)
        drawn, _ = buf.sample(2, current_version=0)
        assert len(drawn) == 2
        ids = drawn.batch["input_ids"][:, 0].tolist()
        assert len(set(ids)) == 1

    def test_frozen_rows_are_unchanged_by_later_inserts(self):
        buf = AdvantagePrioritizedReplayBuffer(priority_alpha=1.0)
        rows = make_rows(2, uid_prefix="old", advantages=[0.5, -0.5])
        buf.add(rows, np.array([1e6, 1e6]), birth_version=0)
        newer = make_rows(2, uid_prefix="new", advantages=[3.0, -3.0], base_token=500)
        buf.add(newer, np.array([1e-6, 1e-6]), birth_version=1)
        drawn, _ = buf.sample(2, current_version=1)
        # The old rows dominate the draw and still carry their birth advantages.
        assert set(drawn.non_tensor_batch["uid"]) == {"old"}
        assert sorted(drawn.batch["advantages"][:, 0].tolist()) == [-0.5, 0.5]


class TestBufferEviction:
    def test_age_eviction_boundary(self):
        buf = AdvantagePrioritizedReplayBuffer(tau_max=2)
        for step in range(4):
            rows, prios = buffer_rows(2, tag=f"s{step}")
            buf.add(rows, prios, birth_version=step)
        evicted = buf.evict_older_than(3)
        # birth 0 has age 3 > tau_max=2 -> evicted; birth 1 (age 2) stays.
        assert evicted == 2
        assert buf.size() == 6
        stats = buf.stats(3)
        assert stats["buffer_age_max"] == 2

    def test_capacity_fifo_evicts_oldest_rows_including_partial_blocks(self):
        buf = AdvantagePrioritizedReplayBuffer(capacity=5)
        first = make_rows(4, uid_prefix="first", advantages=[1.0] * 4)
        buf.add(first, np.ones(4), birth_version=0)
        second = make_rows(4, uid_prefix="second", advantages=[1.0] * 4, base_token=100)
        buf.add(second, np.ones(4), birth_version=1)
        assert buf.size() == 5
        drawn, _ = buf.sample(5, current_version=1)
        uids = list(drawn.non_tensor_batch["uid"])
        assert uids.count("second") == 4
        assert uids.count("first") == 1  # 3 oldest rows of the first block dropped

    def test_add_rejects_nonpositive_priorities_and_ignores_empty(self):
        buf = AdvantagePrioritizedReplayBuffer()
        rows, _ = buffer_rows(2)
        with pytest.raises(AssertionError):
            buf.add(rows, np.array([1.0, 0.0]), birth_version=0)
        empty = rows.select_idxs([])
        buf.add(empty, np.array([]), birth_version=0)
        assert buf.size() == 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestZeroVarianceGroupMask:
    def test_mixed_groups_kept_degenerate_dropped(self):
        scores = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
        uids = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
        mask = zero_variance_group_mask(scores, uids)
        assert mask.tolist() == [True, True, False, False, False, False]

    def test_single_row_group_dropped(self):
        mask = zero_variance_group_mask(np.array([1.0]), np.array(["solo"], dtype=object))
        assert mask.tolist() == [False]

    def test_any_score_variation_keeps_the_group(self):
        scores = np.array([0.5, 0.5000001])
        uids = np.array(["a", "a"], dtype=object)
        assert zero_variance_group_mask(scores, uids).all()


class TestRolloutPriorities:
    def test_recovers_abs_advantage_plus_eps(self):
        advantages = torch.tensor([[2.0, 2.0, 2.0], [-0.5, -0.5, -0.5]])
        mask = torch.ones(2, 3)
        np.testing.assert_allclose(rollout_priorities(advantages, mask, eps=1e-6), [2.0 + 1e-6, 0.5 + 1e-6])

    def test_empty_response_degrades_to_eps(self):
        advantages = torch.tensor([[5.0, 5.0]])
        mask = torch.zeros(1, 2)
        np.testing.assert_allclose(rollout_priorities(advantages, mask, eps=1e-3), [1e-3])

    def test_respects_the_mask(self):
        advantages = torch.tensor([[3.0, 3.0, 999.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        np.testing.assert_allclose(rollout_priorities(advantages, mask, eps=0.001), [3.001])


class TestResolveReplayDraw:
    def test_exact_ratio_when_already_divisible(self):
        assert resolve_replay_draw(16, 0.5, 100, 3) == (8, 0)

    def test_adjusts_draw_up_to_hit_divisibility(self):
        # base = round(9.5) = 10, 19+10=29 % 3 = 2 -> 11 lands on 30.
        assert resolve_replay_draw(19, 0.5, 100, 3) == (11, 0)

    def test_adjusts_draw_down_when_buffer_limits(self):
        # base = min(10, 3) = 3 -> 22 % 3 = 1; up (4) exceeds the buffer -> down to 2.
        assert resolve_replay_draw(19, 0.5, 3, 3) == (2, 0)

    def test_warmup_forces_zero_draw_and_trims(self):
        assert resolve_replay_draw(19, 0.5, 100, 3, warmup_active=True) == (0, 1)
        assert resolve_replay_draw(18, 0.5, 100, 3, warmup_active=True) == (0, 0)

    def test_empty_buffer_falls_back_to_trim(self):
        assert resolve_replay_draw(19, 0.5, 0, 3) == (0, 1)

    def test_zero_fresh(self):
        assert resolve_replay_draw(0, 0.5, 100, 3) == (0, 0)

    def test_dp1_never_adjusts(self):
        assert resolve_replay_draw(19, 0.5, 100, 1) == (10, 0)

    @pytest.mark.parametrize("n_fresh", range(4, 40))
    @pytest.mark.parametrize("buffer_size", [0, 3, 7, 100])
    @pytest.mark.parametrize("dp", [1, 2, 3])
    def test_invariants_hold_across_the_grid(self, n_fresh, buffer_size, dp):
        for warmup in (False, True):
            draw, trim = resolve_replay_draw(n_fresh, 0.5, buffer_size, dp, warmup_active=warmup)
            assert 0 <= draw <= buffer_size
            assert 0 <= trim < dp
            assert trim < n_fresh
            if warmup:
                assert draw == 0
            assert (n_fresh - trim + draw) % dp == 0


# ---------------------------------------------------------------------------
# Trainer integration: _compose_training_batch
# ---------------------------------------------------------------------------


def make_trainer(dp_size=2, warmup_steps=0, replay_ratio=0.5, version=5, balance=False, seed=3):
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.replay_buffer = AdvantagePrioritizedReplayBuffer(
        replay_ratio=replay_ratio, warmup_steps=warmup_steps, seed=seed
    )
    trainer.trainer_dp_size = dp_size
    trainer.current_param_version = version
    trainer.replay_filtered_groups_cum = 0
    trainer._replay_trim_rng = np.random.default_rng(seed + 1)
    trainer.config = OmegaConf.create({"trainer": {"balance_batch": balance}})
    return trainer


def mixed_group(uid, sign=1.0):
    """A 4-rollout group with score variance (1 correct, 3 wrong)."""
    scores = [1.0 * sign, -1.0, -1.0, -1.0]
    advs = [1.5 * sign, -0.5, -0.5, -0.5]
    return (uid, scores, advs)


def degenerate_group(uid):
    return (uid, [-1.0] * 4, [0.0] * 4)


class TestComposeTrainingBatch:
    def test_filters_degenerate_groups_and_fills_the_buffer(self):
        trainer = make_trainer(dp_size=2)
        batch = make_batch([mixed_group("a"), degenerate_group("z"), mixed_group("b")])
        metrics = {}
        out = trainer._compose_training_batch(batch, metrics)
        assert len(out) == 8  # 2 surviving groups, empty buffer -> no draw
        assert set(out.non_tensor_batch["uid"]) == {"a", "b"}
        assert trainer.replay_buffer.size() == 8
        assert metrics["replay/filtered_zero_variance_groups"] == 1
        assert metrics["replay/fresh_groups_kept"] == 2
        assert metrics["replay/draw_size"] == 0
        assert metrics["replay/buffer_size"] == 8

    def test_second_step_draws_replay_and_keeps_fresh_anchor_first(self):
        trainer = make_trainer(dp_size=2)
        trainer._compose_training_batch(make_batch([mixed_group("a"), mixed_group("b")]), {})
        metrics = {}
        out = trainer._compose_training_batch(make_batch([mixed_group("c"), mixed_group("d")]), metrics)
        # 8 fresh survivors + draw 4 = 12 rows, fresh first.
        assert len(out) == 12
        assert set(out.non_tensor_batch["uid"][:8]) == {"c", "d"}
        assert set(out.non_tensor_batch["uid"][8:]) <= {"a", "b"}
        assert metrics["replay/draw_size"] == 4
        assert metrics["replay/replay_fraction"] == pytest.approx(4 / 12)
        assert metrics["replay/minibatch_age_mean"] == 0.0  # same version in this test
        assert out.meta_info["mini_batch_size"] == 6
        assert len(out.meta_info["global_token_num"]) == 12
        assert out.meta_info["temperature"] == 1.0

    def test_replayed_rows_keep_frozen_birth_values(self):
        trainer = make_trainer(dp_size=2)
        first = make_batch([mixed_group("a"), mixed_group("b")])
        trainer._compose_training_batch(first, {})
        out = trainer._compose_training_batch(make_batch([mixed_group("c"), mixed_group("d")]), {})
        replay_part = out.select_idxs(list(range(8, 12)))
        for i in range(4):
            uid = replay_part.non_tensor_batch["uid"][i]
            adv = replay_part.batch["advantages"][i, 0].item()
            assert uid in {"a", "b"}
            assert adv in {1.5, -0.5}

    def test_warmup_disables_the_draw_but_not_the_buffer(self):
        trainer = make_trainer(dp_size=2, warmup_steps=100)
        trainer._compose_training_batch(make_batch([mixed_group("a"), mixed_group("b")]), {})
        metrics = {}
        out = trainer._compose_training_batch(make_batch([mixed_group("c"), mixed_group("d")]), metrics)
        assert len(out) == 8
        assert metrics["replay/draw_size"] == 0
        assert metrics["replay/warmup_active"] == 1.0
        assert trainer.replay_buffer.size() == 16

    def test_trim_path_keeps_the_full_survivors_in_the_buffer(self):
        # 2 surviving groups of 4 = 8 rows, dp=3, empty buffer in warmup:
        # 8 % 3 = 2 -> trim 2 rows from the gradient batch only.
        trainer = make_trainer(dp_size=3, warmup_steps=100)
        metrics = {}
        out = trainer._compose_training_batch(make_batch([mixed_group("a"), mixed_group("b")]), metrics)
        assert len(out) == 6
        assert metrics["replay/trimmed_for_dp"] == 2
        assert trainer.replay_buffer.size() == 8

    def test_all_degenerate_returns_original_batch(self):
        trainer = make_trainer(dp_size=2)
        batch = make_batch([degenerate_group("z1"), degenerate_group("z2")])
        metrics = {}
        out = trainer._compose_training_batch(batch, metrics)
        assert out is batch
        assert metrics["replay/degenerate_step"] == 1.0
        assert trainer.replay_buffer.size() == 0

    def test_disabled_mode_is_identity(self):
        trainer = object.__new__(FullyAsyncTrainer)
        trainer.replay_buffer = None
        batch = make_batch([mixed_group("a")])
        assert trainer._compose_training_batch(batch, {}) is batch

    def test_resume_restarts_with_an_empty_buffer_and_refills(self):
        """The replay buffer is not checkpointed: a resumed run (fresh buffer,
        large restored param version) draws 0 on its first step, refills the
        buffer, and draws normally from the next step on — warmup does not
        re-trigger because it compares against the restored version."""
        trainer = make_trainer(dp_size=2, warmup_steps=20, version=500)
        metrics = {}
        out = trainer._compose_training_batch(make_batch([mixed_group("a"), mixed_group("b")]), metrics)
        assert len(out) == 8
        assert metrics["replay/draw_size"] == 0
        assert metrics["replay/warmup_active"] == 0.0
        assert trainer.replay_buffer.size() == 8
        metrics = {}
        out = trainer._compose_training_batch(make_batch([mixed_group("c"), mixed_group("d")]), metrics)
        assert metrics["replay/draw_size"] == 4
        assert len(out) == 12

    def test_age_eviction_runs_before_the_draw(self):
        trainer = make_trainer(dp_size=2, version=0)
        trainer.replay_buffer.tau_max = 1
        trainer._compose_training_batch(make_batch([mixed_group("a"), mixed_group("b")]), {})
        trainer.current_param_version = 3
        metrics = {}
        trainer._compose_training_batch(make_batch([mixed_group("c"), mixed_group("d")]), metrics)
        # birth-0 rows have age 3 > tau_max=1 -> evicted BEFORE the draw, so
        # despite 8 buffered rows nothing over-age reaches the gradient batch.
        assert metrics["replay/evicted_by_age"] == 8
        assert metrics["replay/draw_size"] == 0
        assert trainer.replay_buffer.size() == 8


# ---------------------------------------------------------------------------
# Config validation at trainer init
# ---------------------------------------------------------------------------


def replay_config(**overrides):
    cfg = OmegaConf.create(
        {
            "async_training": {
                "use_rollout_log_probs": True,
                "compute_prox_log_prob": False,
                "require_batches": 1,
                "rollout_replay": {
                    "enable": True,
                    "replay_ratio": 0.5,
                    "priority_alpha": 0.5,
                    "priority_eps": 1e-6,
                    "tau_max": 10,
                    "warmup_steps": 20,
                    "capacity": 30000,
                    "sampling_seed": 1234,
                    "with_replacement": False,
                },
            },
            "algorithm": {"adv_estimator": "grpo", "rollout_correction": {"rollout_is": None}},
            "actor_rollout_ref": {
                "rollout": {"calculate_log_probs": True},
                "actor": {
                    "megatron": {
                        "tensor_model_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                        "context_parallel_size": 1,
                    }
                },
            },
            "trainer": {"nnodes": 1, "n_gpus_per_node": 3},
        }
    )
    for path, value in overrides.items():
        OmegaConf.update(cfg, path, value)
    return cfg


def init_replay(cfg):
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.config = cfg
    trainer._init_rollout_replay(cfg)
    return trainer


class TestReplayConfigValidation:
    def test_valid_config_builds_the_buffer(self):
        trainer = init_replay(replay_config())
        assert trainer.replay_buffer is not None
        assert trainer.trainer_dp_size == 3
        assert trainer.replay_buffer.tau_max == 10
        assert trainer.replay_buffer.warmup_steps == 20

    def test_disabled_leaves_no_buffer(self):
        trainer = init_replay(replay_config(**{"async_training.rollout_replay.enable": False}))
        assert trainer.replay_buffer is None

    @pytest.mark.parametrize(
        "path,value",
        [
            ("async_training.use_rollout_log_probs", False),
            ("async_training.compute_prox_log_prob", True),
            ("async_training.require_batches", 4),
            ("algorithm.rollout_correction.rollout_is", "token"),
            ("algorithm.adv_estimator", "gae"),
            ("actor_rollout_ref.rollout.calculate_log_probs", False),
        ],
    )
    def test_incompatible_configs_are_rejected(self, path, value):
        with pytest.raises(AssertionError):
            init_replay(replay_config(**{path: value}))

    def test_dp_size_accounts_for_model_parallelism(self):
        cfg = replay_config(
            **{
                "trainer.n_gpus_per_node": 4,
                "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 2,
            }
        )
        assert init_replay(cfg).trainer_dp_size == 2


# ---------------------------------------------------------------------------
# Megatron patches: mini-batch-size meta override + entropy logging
# ---------------------------------------------------------------------------


class TestMegatronActorPatches:
    def _import_actor(self):
        return pytest.importorskip(
            "verl.workers.actor.megatron_actor", reason="megatron not importable on this machine"
        )

    def _actor_and_data(self, n_rows, config_mini):
        megatron_actor = self._import_actor()
        from types import SimpleNamespace

        fake = SimpleNamespace(
            config=SimpleNamespace(
                ppo_mini_batch_size=config_mini,
                ppo_epochs=1,
                data_loader_seed=None,
                shuffle=False,
                use_kl_loss=False,
            )
        )
        data = make_rows(n_rows, advantages=[0.1] * n_rows)
        return megatron_actor.MegatronPPOActor, fake, data

    def test_meta_key_overrides_the_config_value(self):
        actor_cls, fake, data = self._actor_and_data(n_rows=6, config_mini=4)
        data.meta_info["mini_batch_size"] = 6
        batches = list(actor_cls.make_minibatch_iterator(fake, data))
        assert len(batches) == 1
        assert len(batches[0]) == 6

    def test_without_the_meta_key_the_config_value_still_applies(self):
        actor_cls, fake, data = self._actor_and_data(n_rows=8, config_mini=4)
        batches = list(actor_cls.make_minibatch_iterator(fake, data))
        assert len(batches) == 2
        assert all(len(b) == 4 for b in batches)

    def _module_source(self):
        # update_policy is wrapped by @GPUMemoryLogger without functools.wraps,
        # so inspect.getsource on the method returns the decorator shim; read
        # the module file itself instead.
        megatron_actor = self._import_actor()
        with open(megatron_actor.__file__) as f:
            return f.read()

    def test_update_policy_honors_calculate_entropy_at_zero_coeff(self):
        """The pristine branch only computed entropy when entropy_coeff != 0;
        the should_calculate_entropy helper (ported from baselines) also honors
        actor.calculate_entropy so actor/entropy (the campaign's collapse
        indicator) is logged with entropy_coeff=0."""
        megatron_actor = self._import_actor()

        def cfg(coeff, calc):
            return OmegaConf.create({"entropy_coeff": coeff, "calculate_entropy": calc})

        assert megatron_actor.should_calculate_entropy(cfg(0, True)) is True
        assert megatron_actor.should_calculate_entropy(cfg(0, False)) is False
        assert megatron_actor.should_calculate_entropy(cfg(0.001, False)) is True
        source = self._module_source()
        assert "calculate_entropy = should_calculate_entropy(self.config)" in source

    def test_loss_func_emits_the_entropy_metric(self):
        source = self._module_source()
        assert 'stats["actor/entropy"] = entropy_loss.detach().item()' in source


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
