# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Tests for policy_loss.loss_mode="rollout_correction", the loss the fully_async
`is-pg` arm runs.

The arm claims to reproduce what branch `rollout-dapo` executes with
skip_recompute_old_log_prob=True: there the actor sets old_log_prob = log_prob.detach(),
so the PPO ratio is exactly 1, clipping never binds, and the whole correction is
trunc(pi_theta/pi_rollout, threshold) applied as an IS-weighted policy gradient.
test_gradient_matches_vanilla_at_unit_ratio turns that claim into an assertion.

Run: pytest tests/trainer/ppo/test_rollout_correction_loss_mode_on_cpu.py
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.utils.config import omega_conf_to_dataclass

ROLLOUT_IS_THRESHOLD = 2.0


def _actor_config(rollout_correction=None, loss_agg_mode="seq-mean-token-mean"):
    """A real ActorConfig, as the worker builds it from the launch script."""
    policy_loss = {
        "_target_": "verl.workers.config.PolicyLossConfig",
        "loss_mode": "rollout_correction",
    }
    if rollout_correction is not None:
        policy_loss["rollout_correction"] = rollout_correction
    return omega_conf_to_dataclass(
        {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "megatron",
            "ppo_micro_batch_size_per_gpu": 1,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "loss_agg_mode": loss_agg_mode,
            "policy_loss": policy_loss,
            "optim": {"_target_": "verl.workers.config.OptimizerConfig", "lr": 1e-6},
            "rollout_n": 1,
        }
    )


def _batch(seed=0, batch_size=3, seq_len=7):
    """log_prob (requires grad), rollout_log_prob, advantages, response_mask."""
    generator = torch.Generator().manual_seed(seed)
    log_prob = (-torch.rand(batch_size, seq_len, generator=generator) * 2).requires_grad_(True)
    # a spread of ratios around 1, including some beyond the truncation threshold
    rollout_log_prob = log_prob.detach() - (torch.rand(batch_size, seq_len, generator=generator) * 2 - 1.0)
    advantages = torch.randn(batch_size, seq_len, generator=generator)
    response_mask = torch.ones(batch_size, seq_len)
    response_mask[:, -2:] = 0  # padding must be ignored everywhere
    return log_prob, rollout_log_prob, advantages, response_mask


def _rollout_correction_cfg(**overrides):
    cfg = {
        "rollout_is": "token",
        "rollout_is_threshold": ROLLOUT_IS_THRESHOLD,
        "rollout_rs": None,
        "rollout_rs_threshold": None,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------- wrapper contract


def test_missing_sub_config_raises():
    """Without policy_loss.rollout_correction the arm must fail loudly, not silently
    train something else."""
    loss_fn = get_policy_loss_fn("rollout_correction")
    log_prob, rollout_log_prob, advantages, response_mask = _batch()

    with pytest.raises(ValueError, match="rollout_correction config not found"):
        loss_fn(
            old_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean",
            config=_actor_config(rollout_correction=None),
        )


def test_metrics_surface_has_no_clipfrac():
    """The loss never clips, so pg_clipfrac does not exist here (arm B logs it)."""
    loss_fn = get_policy_loss_fn("rollout_correction")
    log_prob, rollout_log_prob, advantages, response_mask = _batch()

    loss, metrics = loss_fn(
        old_log_prob=rollout_log_prob,  # bypass mode: old_log_prob IS the rollout log prob
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=_actor_config(_rollout_correction_cfg()),
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert "actor/ppo_kl" in metrics
    assert any(key.startswith("rollout_corr/") for key in metrics)
    assert "actor/pg_clipfrac" not in metrics
    assert "actor/pg_clipfrac_lower" not in metrics


def test_reduces_to_reinforce_without_is():
    """rollout_is=None must give plain -E[log pi * A] (no weights)."""
    loss_fn = get_policy_loss_fn("rollout_correction")
    log_prob, rollout_log_prob, advantages, response_mask = _batch()

    loss, _ = loss_fn(
        old_log_prob=rollout_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=_actor_config(_rollout_correction_cfg(rollout_is=None)),
    )

    expected = -(advantages * log_prob * response_mask).sum() / response_mask.sum()
    torch.testing.assert_close(loss, expected)


# ------------------------------------------------- equivalence with the source arm


def test_gradient_matches_vanilla_at_unit_ratio():
    """`rollout-dapo` runs the VANILLA loss with old_log_prob = log_prob.detach() and
    precomputed IS weights. This arm runs the rollout_correction loss instead. Both must
    produce the same gradient — that is what makes the port a reproduction rather than a
    different algorithm.
    """
    log_prob_a, rollout_log_prob, advantages, response_mask = _batch(seed=1)
    log_prob_b = log_prob_a.detach().clone().requires_grad_(True)

    # --- what rollout-dapo computes: vanilla loss, ratio == 1, IS weights applied
    weights_proto, _, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob_a.detach(),  # the fork passes the detached current policy
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=ROLLOUT_IS_THRESHOLD,
    )
    rollout_is_weights = weights_proto.batch["rollout_is_weights"]

    vanilla = get_policy_loss_fn("vanilla")
    loss_vanilla, metrics_vanilla = vanilla(
        old_log_prob=log_prob_a.detach(),
        log_prob=log_prob_a,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=_actor_config(_rollout_correction_cfg()),
        rollout_is_weights=rollout_is_weights,
    )
    loss_vanilla.backward()

    # --- what this arm computes: rollout_correction loss, weights computed internally
    loss_fn = get_policy_loss_fn("rollout_correction")
    loss_rc, _ = loss_fn(
        old_log_prob=rollout_log_prob,
        log_prob=log_prob_b,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=_actor_config(_rollout_correction_cfg()),
    )
    loss_rc.backward()

    torch.testing.assert_close(log_prob_a.grad, log_prob_b.grad, rtol=1e-5, atol=1e-6)
    # and the clip really was inert on the vanilla side
    assert metrics_vanilla["actor/pg_clipfrac"] == 0.0
    assert metrics_vanilla["actor/pg_clipfrac_lower"] == 0.0
    assert metrics_vanilla["actor/ppo_kl"] == pytest.approx(0.0, abs=1e-7)


def test_weights_are_truncated_at_the_threshold():
    """The trust region is the IS truncation: no weight may exceed the threshold."""
    log_prob, rollout_log_prob, _, response_mask = _batch(seed=2)
    weights_proto, _, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob.detach(),
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=ROLLOUT_IS_THRESHOLD,
    )
    weights = weights_proto.batch["rollout_is_weights"]
    assert weights[response_mask.bool()].max().item() <= ROLLOUT_IS_THRESHOLD + 1e-6
    # untruncated ratios do exceed it, i.e. the test data actually exercises the clamp
    raw = torch.exp(log_prob.detach() - rollout_log_prob)
    assert raw[response_mask.bool()].max().item() > ROLLOUT_IS_THRESHOLD
