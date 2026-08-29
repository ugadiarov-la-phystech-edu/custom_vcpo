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

"""Unit tests for the mu-anchored vanilla PPO loss (Arm A of
CLIP_IS_MIXING_ANCHORS_DISCUSSION.md §4, policy_loss.anchor_mode="mu").

Arm A feeds the cached behavior log-probs mu (rollout_log_probs, frozen at
generation) to the stock compute_policy_loss_vanilla as old_log_prob and drops
the token-IS weight (the pi/mu ratio IS the importance correction). These tests
pin down the properties the actor-side dispatch relies on:

- clip semantics per (advantage sign x drift direction) quadrant, hand-computed:
  A>0 up-drift clipped at 1+clip_ratio_high, A>0 down-drift unclipped (recovery
  gradient flows), A<0 down-drift clipped at 1-clip_ratio_low, A<0 r>>1 floored
  by dual-clip (clip_ratio_c) with pg_clipfrac_lower reporting it;
- the asymmetric DAPO band actually binds per side (0.2 low / 0.28 high);
- gradient contract: outside the band the away-gradient is exactly zero (the
  self-terminating property that bounds cumulative movement across replay
  reuse); inside it flows; padded positions are inert;
- regression: with the ratio-anchored old_log_prob (skip_recompute's detach)
  the clip is inert and the loss degenerates to REINFORCE — documenting why the
  anchor must be mu for the clip to exist at all;
- composition rule (doc §3): stacking a pi/mu rollout_is weight on top of the
  mu-anchored loss changes it — documenting why the actor must drop the weight;
- the A=+1 advantage fold of the mbs=1 per-traj path breaks the sign-dependent
  branch selection under the mu anchor — documenting why that path keeps real
  advantages when anchor_mode="mu";
- config plumbing: PolicyLossConfig.anchor_mode default, omegaconf round-trip,
  the yaml mirrors.

Run: pytest tests/trainer/ppo/test_mu_anchor_loss_on_cpu.py
"""

import os

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, get_policy_loss_fn
from verl.workers.config.actor import PolicyLossConfig


def _config(clip_low=0.2, clip_high=0.28, clip_c=3.0, loss_agg_mode="token-mean"):
    return OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": clip_low,
            "clip_ratio_high": clip_high,
            "clip_ratio_c": clip_c,
            "loss_agg_mode": loss_agg_mode,
            "global_batch_info": {},
            "policy_loss": {"loss_mode": "vanilla", "anchor_mode": "mu"},
        }
    )


def _call(log_prob, mu_log_prob, advantages, response_mask, config, rollout_is_weights=None):
    return compute_policy_loss_vanilla(
        old_log_prob=mu_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )


def _row(ratios, adv):
    """One-row batch with the given per-token pi/mu ratios and constant advantage."""
    ratios = torch.tensor([ratios], dtype=torch.float64)
    mu = torch.zeros_like(ratios)
    log_prob = torch.log(ratios)  # so exp(log_prob - mu) == ratios
    advantages = torch.full_like(ratios, float(adv))
    mask = torch.ones_like(ratios)
    return log_prob, mu, advantages, mask


class TestClipSemantics:
    def test_positive_adv_updrift_clipped_at_high(self):
        # A>0, ratios [1.0 (inside), 1.5 (above 1.28)]: token loss -A*min(r, 1.28)
        log_prob, mu, adv, mask = _row([1.0, 1.5], adv=2.0)
        loss, metrics = _call(log_prob, mu, adv, mask, _config())
        expected = -(2.0 * 1.0 + 2.0 * 1.28) / 2
        assert loss.item() == pytest.approx(expected, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.5)

    def test_positive_adv_downdrift_unclipped(self):
        # A>0, r=0.5 (below 1-0.2=0.8): max(-A*0.5, -A*0.8) = -A*0.5 — the
        # toward-mu recovery gradient keeps flowing, pg_clipfrac stays 0.
        log_prob, mu, adv, mask = _row([0.5], adv=1.0)
        loss, metrics = _call(log_prob, mu, adv, mask, _config())
        assert loss.item() == pytest.approx(-0.5, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == 0.0

    def test_negative_adv_downdrift_clipped_at_low(self):
        # A<0, r=0.5 (below 0.8): max(0.5, 0.8) = 0.8 — push-down stops at 1-eps_lo.
        log_prob, mu, adv, mask = _row([0.5], adv=-1.0)
        loss, metrics = _call(log_prob, mu, adv, mask, _config())
        assert loss.item() == pytest.approx(0.8, rel=1e-6)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)

    def test_negative_adv_huge_ratio_dual_clipped(self):
        # A<0, r=5.0 >> 1: unclipped branch gives 5.0, band branch 1.28, max=5.0;
        # dual-clip floors the loss at -A*clip_ratio_c = 3.0.
        log_prob, mu, adv, mask = _row([5.0], adv=-1.0)
        loss, metrics = _call(log_prob, mu, adv, mask, _config(clip_c=3.0))
        assert loss.item() == pytest.approx(3.0, rel=1e-6)
        assert metrics["actor/pg_clipfrac_lower"] == pytest.approx(1.0)

    def test_asymmetric_band_binds_per_side(self):
        # Same |drift| both sides of 1: up-side clips at 1.28, down-side at 0.8.
        cfg = _config(clip_low=0.2, clip_high=0.28)
        log_prob, mu, adv, mask = _row([1.5], adv=1.0)
        loss_up, _ = _call(log_prob, mu, adv, mask, cfg)
        assert loss_up.item() == pytest.approx(-1.28, rel=1e-6)
        log_prob, mu, adv, mask = _row([0.5], adv=-1.0)
        loss_down, _ = _call(log_prob, mu, adv, mask, cfg)
        assert loss_down.item() == pytest.approx(0.8, rel=1e-6)

    def test_ppo_kl_measures_pi_vs_mu(self):
        # ppo_kl = masked_mean(-(log_prob - mu)) — a real KL(pi||mu) proxy now.
        log_prob, mu, adv, mask = _row([2.0, 0.5], adv=1.0)
        _, metrics = _call(log_prob, mu, adv, mask, _config())
        import math

        expected = -(math.log(2.0) + math.log(0.5)) / 2
        assert metrics["actor/ppo_kl"] == pytest.approx(expected, rel=1e-6)


class TestGradientContract:
    def _grad(self, ratios, adv, clip_low=0.2, clip_high=0.28, clip_c=3.0):
        log_prob, mu, advantages, mask = _row(ratios, adv)
        log_prob.requires_grad_(True)
        loss, _ = _call(log_prob, mu, advantages, mask, _config(clip_low, clip_high, clip_c))
        loss.backward()
        return log_prob.grad[0]

    def test_positive_adv_above_band_gradient_is_zero(self):
        # The self-terminating property: once pi/mu >= 1+eps_hi the pushing
        # gradient is exactly zero and stays zero (mu never refreshes).
        grad = self._grad([2.0, 1.1], adv=1.0)
        assert grad[0].item() == 0.0
        assert grad[1].item() != 0.0

    def test_negative_adv_below_band_gradient_is_zero(self):
        grad = self._grad([0.5, 0.9], adv=-1.0)
        assert grad[0].item() == 0.0
        assert grad[1].item() != 0.0

    def test_inside_band_gradient_matches_reinforce_times_ratio(self):
        # Unclipped branch: d(-A*r)/dlogp = -A*r.
        grad = self._grad([1.1], adv=2.0)
        assert grad[0].item() == pytest.approx(-2.0 * 1.1, rel=1e-6)

    def test_padded_positions_inert(self):
        log_prob, mu, advantages, mask = _row([2.0, 1.5], adv=1.0)
        mask[0, 1] = 0.0
        log_prob.requires_grad_(True)
        loss, _ = _call(log_prob, mu, advantages, mask, _config())
        loss.backward()
        assert log_prob.grad[0, 1].item() == 0.0


class TestAnchorRegression:
    def test_detached_anchor_makes_clip_inert(self):
        # Today's behavior (anchor_mode=null): old_log_prob = log_prob.detach()
        # -> ratio == 1 everywhere, pg_clipfrac == ppo_kl == 0, and the loss
        # value equals the REINFORCE surrogate agg(-A). This is why the anchor
        # must be mu for the clip to exist at all.
        log_prob = torch.log(torch.tensor([[3.0, 0.1, 1.0]], dtype=torch.float64))
        advantages = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
        mask = torch.ones_like(log_prob)
        loss, metrics = _call(log_prob, log_prob.detach().clone(), advantages, mask, _config())
        assert metrics["actor/pg_clipfrac"] == 0.0
        assert metrics["actor/ppo_kl"] == 0.0
        assert loss.item() == pytest.approx((-advantages).mean().item(), rel=1e-6)

    def test_detached_anchor_gradient_is_pure_reinforce(self):
        log_prob = torch.log(torch.tensor([[3.0, 0.1]], dtype=torch.float64))
        log_prob.requires_grad_(True)
        advantages = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
        mask = torch.ones_like(advantages)
        loss, _ = _call(log_prob, log_prob.detach().clone(), advantages, mask, _config())
        loss.backward()
        # d/dlogp of -A*exp(logp - sg(logp)) at logp == anchor is -A (per-token mean).
        assert torch.allclose(log_prob.grad, -advantages / advantages.numel())


class TestCompositionRule:
    def test_stacked_is_weight_changes_the_loss(self):
        # Doc §3: the mu-anchored ratio already IS the importance correction.
        # Supplying the pi/mu token-IS weight on top multiplies it in a second
        # time and changes the loss — the actor must pass rollout_is_weights=None.
        log_prob, mu, adv, mask = _row([1.5, 0.7], adv=1.0)
        cfg = _config()
        loss_clean, _ = _call(log_prob, mu, adv, mask, cfg)
        weights = torch.clamp(torch.exp(log_prob - mu), max=2.0).detach()
        loss_stacked, _ = _call(log_prob, mu, adv, mask, cfg, rollout_is_weights=weights)
        assert loss_clean.item() != pytest.approx(loss_stacked.item(), rel=1e-9)


class TestAdvantageFoldBreaksSign:
    def test_folded_a1_loss_differs_for_negative_advantage(self):
        # The mbs=1 per-traj path folds the advantage into a scalar loss
        # multiplier over an A=+1 tensor. That is exact only for a loss linear
        # in the advantage; the clipped loss selects branches by sign, so for
        # A<0 the fold clips the wrong side. r=0.5, A=-1: real loss 0.8
        # (clipped at 1-eps_lo), folded loss (-1)*(-0.5) = 0.5 (the A=+1 branch
        # keeps the unclipped -A*r). This is why update_policy_per_traj keeps
        # real advantages when anchor_mode="mu".
        cfg = _config()
        log_prob, mu, adv, mask = _row([0.5], adv=-1.0)
        loss_real, _ = _call(log_prob, mu, adv, mask, cfg)
        log_prob, mu, adv_one, mask = _row([0.5], adv=1.0)
        loss_folded = -1.0 * _call(log_prob, mu, adv_one, mask, cfg)[0]
        assert loss_real.item() == pytest.approx(0.8, rel=1e-6)
        assert loss_folded.item() == pytest.approx(0.5, rel=1e-6)
        assert loss_real.item() != pytest.approx(loss_folded.item(), rel=1e-6)


class TestConfigPlumbing:
    def test_dataclass_default_is_none(self):
        assert PolicyLossConfig().anchor_mode is None

    def test_dataclass_round_trip(self):
        cfg = OmegaConf.structured(PolicyLossConfig(anchor_mode="mu"))
        assert cfg.anchor_mode == "mu"
        assert cfg.get("anchor_mode", None) == "mu"

    def test_registry_dispatch_vanilla(self):
        assert get_policy_loss_fn("vanilla") is compute_policy_loss_vanilla

    @pytest.mark.parametrize(
        "rel_path",
        [
            "verl/trainer/config/actor/actor.yaml",
            "verl/trainer/config/_generated_ppo_trainer.yaml",
            "verl/trainer/config/_generated_ppo_megatron_trainer.yaml",
        ],
    )
    def test_yaml_mirrors_carry_anchor_mode(self, rel_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        with open(os.path.join(repo_root, rel_path)) as f:
            content = f.read()
        assert "anchor_mode" in content, f"{rel_path} lacks policy_loss.anchor_mode"

    def test_yaml_default_is_null(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cfg = OmegaConf.load(os.path.join(repo_root, "verl/trainer/config/actor/actor.yaml"))
        assert cfg.policy_loss.anchor_mode is None
