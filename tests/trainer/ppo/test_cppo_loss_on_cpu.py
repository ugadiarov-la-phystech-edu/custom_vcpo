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

"""Unit tests for compute_policy_loss_cppo (CPPO, Binary-TV variant; paper/cppo.pdf).

The vectorized implementation is checked against an independent reference-mirror
written directly from the paper's per-token recurrences (Eq. 8-10, Eq. 22), plus
targeted scenario tests:

- mask semantics: toward-mu always kept (both advantage signs), token-level
  threshold, prefix-budget exhaustion and recovery, first-token full-delta slack,
  position-weight asymmetry (same divergence masked early / kept late);
- per-sequence delta_b calibration: quantile, floor and 5x-ceiling clamps,
  all-padding NaN fallback;
- loss/gradient contract: grad wrt log_prob == -A * sg(min(ratio, c)) * mask
  (agg-scaled), truncation cap applied, no gradient through mask or weight,
  rollout_is_weights composition, row independence, padded positions inert;
- degenerate settings: w_min=1 + huge delta_b reduces to the uniform token
  threshold; the ratio-anchored old_log_prob (skip_recompute's detach) makes the
  mask a no-op — documenting why loss_func must feed mu instead;
- config plumbing: CPPOConfig defaults, registry dispatch, generated yaml.

Run: pytest tests/trainer/ppo/test_cppo_loss_on_cpu.py
"""

import math
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss_cppo, get_policy_loss_fn
from verl.workers.config.actor import CPPOConfig, PolicyLossConfig

B, T = 3, 6


def _config(
    delta=0.15,
    clip_ratio_c=20.0,
    w_min=0.8,
    delta_b=0.02,
    delta_b_q=0.9,
    delta_b_k=1.0,
    loss_agg_mode="seq-mean-token-mean",
):
    return OmegaConf.create(
        {
            "clip_ratio": delta,
            "clip_ratio_c": clip_ratio_c,
            "loss_agg_mode": loss_agg_mode,
            "global_batch_info": {},
            "policy_loss": {
                "loss_mode": "cppo",
                "cppo": {
                    "cppo_w_min": w_min,
                    "cppo_delta_b": delta_b,
                    "cppo_delta_b_q": delta_b_q,
                    "cppo_delta_b_k": delta_b_k,
                },
            },
        }
    )


def _call(log_prob, old_log_prob, advantages, response_mask, config, loss_agg_mode=None, rollout_is_weights=None):
    return compute_policy_loss_cppo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode or config.loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )


def _rand_inputs(seed=0, b=B, t=T, mask=None):
    g = torch.Generator().manual_seed(seed)
    log_prob = -torch.nn.functional.softplus(torch.randn(b, t, generator=g))
    old_log_prob = -torch.nn.functional.softplus(torch.randn(b, t, generator=g))
    advantages = torch.randn(b, 1, generator=g).expand(b, t).contiguous()
    if mask is None:
        mask = torch.ones(b, t, dtype=torch.long)
    return log_prob.requires_grad_(True), old_log_prob, advantages * mask, mask


def _reference_mask(log_prob, old_log_prob, advantages, response_mask, cfg):
    """Independent per-token loop implementation of the CPPO mask, written
    directly from the paper's recurrences (not from the vectorized code)."""
    delta = float(cfg.clip_ratio)
    w_min = float(cfg.policy_loss.cppo.cppo_w_min)
    db_floor = float(cfg.policy_loss.cppo.cppo_delta_b)
    q = float(cfg.policy_loss.cppo.cppo_delta_b_q)
    k = float(cfg.policy_loss.cppo.cppo_delta_b_k)
    lp = log_prob.detach().numpy()
    olp = old_log_prob.detach().numpy()
    adv = advantages.detach().numpy()
    m = response_mask.numpy().astype(float)
    b, t_len = m.shape
    out = np.zeros_like(m)
    for i in range(b):
        d_row = np.abs(np.exp(lp[i]) - np.exp(olp[i])) * m[i]
        valid = m[i] > 0
        if valid.any():
            db_seq = float(np.clip(k * np.quantile(d_row[valid], q), db_floor, 5.0 * db_floor))
        else:
            db_seq = db_floor
        s_prev, w_prev = 0.0, 0.0
        for t in range(t_len):
            pos = t + 1  # 1-based over the PADDED length (reference convention)
            w_t = (w_min + (1.0 - w_min) * (t_len - pos) / max(t_len - 1.0, 1.0)) * m[i, t]
            z_t = w_t * d_row[t]
            c_t = min(delta, delta + db_seq * w_prev - s_prev)
            ratio = math.exp(min(max(lp[i, t] - olp[i, t], -20.0), 20.0))
            toward_mu = adv[i, t] * (ratio - 1.0) <= 0.0
            keep = (toward_mu or z_t <= c_t) and m[i, t] > 0
            out[i, t] = 1.0 if keep else 0.0
            s_prev += z_t
            w_prev += w_t
    return torch.from_numpy(out).float()


def _reference_loss(log_prob, old_log_prob, advantages, response_mask, cfg, rollout_is_weights=None):
    """Closed-form loss from the reference mask (seq-mean-token-mean over the
    full response_mask, matching agg_loss semantics)."""
    mask = _reference_mask(log_prob, old_log_prob, advantages, response_mask, cfg)
    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -20.0, 20.0)).detach()
    w = torch.clamp(ratio, max=float(cfg.clip_ratio_c))
    pg = -advantages * w * log_prob * mask
    if rollout_is_weights is not None:
        pg = pg * rollout_is_weights
    m = response_mask.float()
    per_seq = (pg * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)
    return per_seq.mean()


# ---------------------------------------------------------------- mirror property tests


class TestAgainstReferenceMirror:
    @pytest.mark.parametrize("seed", range(6))
    def test_loss_matches_reference_on_random_inputs(self, seed):
        cfg = _config()
        lp, olp, adv, mask = _rand_inputs(seed)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8), f"seed {seed}: {loss} vs {ref}"

    @pytest.mark.parametrize("seed", range(4))
    def test_loss_matches_reference_with_variable_lengths(self, seed):
        cfg = _config()
        mask = torch.zeros(B, T, dtype=torch.long)
        for i, k in enumerate([T, 3, 1]):
            mask[i, :k] = 1
        lp, olp, adv, _ = _rand_inputs(seed, mask=mask)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("delta,delta_b,w_min", [(0.05, 0.005, 0.5), (0.3, 0.1, 1.0), (0.15, 0.0, 0.8)])
    def test_loss_matches_reference_across_hyperparameters(self, delta, delta_b, w_min):
        cfg = _config(delta=delta, delta_b=delta_b, w_min=w_min)
        lp, olp, adv, mask = _rand_inputs(3)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- mask semantics


def _single_row(lp_vals, olp_vals, adv_scalar):
    lp = torch.tensor([lp_vals], dtype=torch.float32, requires_grad=True)
    olp = torch.tensor([olp_vals], dtype=torch.float32)
    adv = torch.full((1, len(lp_vals)), float(adv_scalar))
    mask = torch.ones(1, len(lp_vals), dtype=torch.long)
    return lp, olp, adv, mask


class TestMaskSemantics:
    def test_toward_mu_kept_positive_adv_huge_divergence(self):
        # A>0, pi far BELOW mu (ratio<1): correction direction -> kept even
        # though |pi - mu| dwarfs every threshold.
        cfg = _config(delta=1e-4, delta_b=1e-5)
        lp, olp, adv, mask = _single_row([-5.0] * 4, [-0.1] * 4, +2.0)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)
        assert metrics["actor/cppo_toward_mu_frac"] == pytest.approx(1.0)

    def test_toward_mu_kept_negative_adv_huge_divergence(self):
        # A<0, pi far ABOVE mu (ratio>1): the gradient pushes pi down toward mu.
        cfg = _config(delta=1e-4, delta_b=1e-5)
        lp, olp, adv, mask = _single_row([-0.1] * 4, [-5.0] * 4, -2.0)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)

    def test_away_over_token_threshold_masked(self):
        # A>0, pi ABOVE mu (away) with |pi-mu| ~ 0.63 >> delta: all masked.
        cfg = _config(delta=0.05, delta_b=0.001)
        lp, olp, adv, mask = _single_row([-0.05] * 4, [-1.5] * 4, +1.0)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)

    def test_first_token_gets_full_delta_slack(self):
        # t=1 has S_0=W_0=0 -> c_1 = delta exactly; an away token with
        # w_1*D_1 == delta - eps is kept.
        d_target = 0.10
        olp_v = math.log(0.5)
        lp_v = math.log(0.5 + d_target)  # D = 0.10, ratio > 1, A>0 -> away
        cfg = _config(delta=0.101, delta_b=1e-6, w_min=1.0)  # w_t = 1 everywhere
        lp, olp, adv, mask = _single_row([lp_v], [olp_v], +1.0)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)

    def test_prefix_budget_exhaustion_masks_late_small_divergences(self):
        # delta generous (token test alone would keep everything), delta_b tiny:
        # a high-divergence away prefix exhausts the budget, after which even a
        # SMALL away divergence is masked (c_t < Z_t despite Z_t << delta).
        cfg = _config(delta=0.5, delta_b=0.001, delta_b_k=0.0, w_min=1.0)
        # tokens 1-3: D = 0.3; token 4: D ~= 0.05 away. Divergence accrues into
        # S_t whether or not the token is kept, so t=1 spends 0.3 of the delta
        # slack and already at t=2 the remaining budget (0.5 - 0.3 + delta_b)
        # is below Z = 0.3; by t=4 even the small 0.05 divergence is masked.
        olp_row = [math.log(0.5)] * 4
        lp_row = [math.log(0.8)] * 3 + [math.log(0.55)]
        lp, olp, adv, mask = _single_row(lp_row, olp_row, +1.0)
        ref = _reference_mask(lp, olp, adv, mask, cfg)[0]
        assert ref.tolist() == [1.0, 0.0, 0.0, 0.0]
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.75)

    def test_low_divergence_prefix_leaves_budget_intact(self):
        # Same final token as above but with a near-on-policy prefix: kept.
        cfg = _config(delta=0.5, delta_b=0.001, delta_b_k=0.0, w_min=1.0)
        olp_row = [math.log(0.5)] * 4
        lp_row = [math.log(0.5)] * 3 + [math.log(0.55)]
        lp, olp, adv, mask = _single_row(lp_row, olp_row, +1.0)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)

    def test_position_weight_masks_early_keeps_late(self):
        # Same divergence D at the first and the last position; delta chosen
        # between w_min*D and 1.0*D -> early token masked, late token kept.
        d = 0.10
        olp_v, lp_v = math.log(0.5), math.log(0.5 + d)
        w_min = 0.5
        cfg = _config(delta=0.07, delta_b=100.0, w_min=w_min)  # budget never binds
        t_len = 6
        lp_row = [lp_v] + [olp_v] * (t_len - 2) + [lp_v]
        olp_row = [olp_v] * t_len
        lp, olp, adv, mask = _single_row(lp_row, olp_row, +1.0)
        ref = _reference_mask(lp, olp, adv, mask, cfg)[0]
        assert ref[0].item() == 0.0  # w_1 = 1.0 -> Z = 0.10 > 0.07
        assert ref[-1].item() == 1.0  # w_T = 0.5 -> Z = 0.05 <= 0.07
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0 / t_len)

    def test_zero_advantage_tokens_always_kept(self):
        # A == 0: A*(ratio-1) == 0 <= 0 -> toward-mu clause keeps them.
        cfg = _config(delta=1e-6, delta_b=1e-8)
        lp, olp, adv, mask = _single_row([-0.05] * 4, [-1.5] * 4, 0.0)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)

    def test_row_independence(self):
        # Row 0's decisions must not depend on row 1's content.
        cfg = _config()
        lp, olp, adv, mask = _rand_inputs(5)
        full_ref = _reference_mask(lp, olp, adv, mask, cfg)
        solo_ref = _reference_mask(lp[:1], olp[:1], adv[:1], mask[:1], cfg)
        assert torch.equal(full_ref[:1], solo_ref)
        loss_solo, _ = _call(lp[:1].detach().requires_grad_(True), olp[:1], adv[:1], mask[:1], cfg)
        ref_solo = _reference_loss(lp[:1], olp[:1], adv[:1], mask[:1], cfg)
        assert torch.allclose(loss_solo, ref_solo, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- delta_b calibration


class TestDeltaBCalibration:
    def _mask_onset(self, d_const, cfg, t_len=40):
        """First masked position for a constant-divergence away row (all tokens
        identical), from the reference mirror."""
        olp_v = math.log(0.4)
        lp_v = math.log(0.4 + d_const)
        lp, olp, adv, mask = _single_row([lp_v] * t_len, [olp_v] * t_len, +1.0)
        ref = _reference_mask(lp, olp, adv, mask, cfg)[0]
        masked = (ref == 0).nonzero()
        vec_loss, metrics = _call(lp, olp, adv, mask, cfg)
        ref_loss = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(vec_loss, ref_loss, rtol=1e-6, atol=1e-8)
        return int(masked[0].item()) if masked.numel() else None

    def test_quantile_calibration_extends_budget_for_high_divergence_seq(self):
        # Constant D=0.5 -> P90 quantile 0.5 -> delta_b clamped to 5*floor=0.1.
        # With delta_b FIXED at the floor (k=0) the budget dies much earlier.
        cfg_adaptive = _config(delta=5.0, delta_b=0.02, delta_b_k=1.0, w_min=1.0)
        cfg_floor = _config(delta=5.0, delta_b=0.02, delta_b_k=0.0, w_min=1.0)
        onset_adaptive = self._mask_onset(0.5, cfg_adaptive)
        onset_floor = self._mask_onset(0.5, cfg_floor)
        assert onset_floor is not None and onset_adaptive is not None
        assert onset_adaptive > onset_floor  # adaptive budget = 5x floor -> later onset

    def test_low_divergence_seq_never_masks_below_floor_budget(self):
        # Constant D=0.01 < delta_b floor 0.02: budget accrues faster than it
        # is spent -> never masks.
        cfg = _config(delta=5.0, delta_b=0.02, delta_b_k=1.0, w_min=1.0)
        assert self._mask_onset(0.01, cfg) is None

    def test_all_padding_row_is_safe(self):
        cfg = _config()
        lp, olp, adv, _ = _rand_inputs(7)
        mask = torch.ones(B, T, dtype=torch.long)
        mask[1] = 0  # fully padded row -> nanquantile falls back to the floor
        loss, metrics = _call(lp, olp, adv * mask, mask, cfg)
        assert torch.isfinite(loss)
        for v in metrics.values():
            assert math.isfinite(v)


# ---------------------------------------------------------------- gradient contract


class TestGradientContract:
    def test_grad_is_masked_truncated_weight_reinforce(self):
        cfg = _config(clip_ratio_c=1.5)
        lp, olp, adv, mask = _rand_inputs(9)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        loss.backward()
        ref_mask = _reference_mask(lp, olp, adv, mask, cfg)
        ratio = torch.exp(torch.clamp(lp.detach() - olp, -20.0, 20.0))
        w = torch.clamp(ratio, max=1.5)
        m = mask.float()
        # seq-mean-token-mean: each token's grad coefficient is scaled by
        # 1/(valid tokens in row) and 1/B (mean over rows)
        expected = -adv * w * ref_mask * m / m.sum(dim=-1, keepdim=True) / B
        assert torch.allclose(lp.grad, expected, rtol=1e-5, atol=1e-8)

    def test_truncation_cap_binds_in_gradient(self):
        # ratio e^2 ~ 7.39 with cap 2.0 -> grad coefficient uses 2.0
        cfg = _config(delta=10.0, delta_b=100.0, clip_ratio_c=2.0)
        lp, olp, adv, mask = _single_row([-0.5], [-2.5], -1.0)  # toward-mu (kept)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        loss.backward()
        assert lp.grad[0, 0].item() == pytest.approx(2.0, rel=1e-6)  # -A * cap = -(-1) * 2

    def test_no_gradient_through_mask_or_weight(self):
        # The loss must be LINEAR in log_prob (weight and mask detached):
        # grad is constant wrt log_prob value -> second backward through a
        # scaled input yields the same grad.
        cfg = _config()
        lp1, olp, adv, mask = _rand_inputs(11)
        loss1, _ = _call(lp1, olp, adv, mask, cfg)
        (g1,) = torch.autograd.grad(loss1, lp1)
        # same inputs, same detached surfaces -> identical coefficient matrix
        lp2 = lp1.detach().clone().requires_grad_(True)
        loss2, _ = _call(lp2, olp, adv, mask, cfg)
        (g2,) = torch.autograd.grad(loss2, lp2)
        assert torch.allclose(g1, g2, rtol=0, atol=0)

    def test_rollout_is_weights_compose_multiplicatively(self):
        cfg = _config()
        lp, olp, adv, mask = _rand_inputs(13)
        weights = torch.full((B, T), 0.5)
        loss_w, _ = _call(lp, olp, adv, mask, cfg, rollout_is_weights=weights)
        loss_1, _ = _call(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss_w, 0.5 * loss_1, rtol=1e-6, atol=1e-9)

    def test_padded_positions_are_inert(self):
        cfg = _config()
        mask = torch.ones(1, T, dtype=torch.long)
        mask[0, 3:] = 0
        lp, olp, adv, _ = _rand_inputs(15, b=1)
        adv = adv * mask
        loss_a, _ = _call(lp.detach().requires_grad_(True), olp, adv, mask, cfg)
        # scribble huge divergence into the padded region: nothing may change
        olp_b = olp.clone()
        olp_b[0, 3:] = -30.0
        loss_b, _ = _call(lp.detach().requires_grad_(True), olp_b, adv, mask, cfg)
        assert torch.allclose(loss_a, loss_b, rtol=0, atol=0)


# ---------------------------------------------------------------- degenerate settings


class TestDegenerateSettings:
    def test_reduces_to_uniform_token_threshold(self):
        # w_min=1 (flat weights) + enormous delta_b (budget never binds):
        # mask == toward_mu | (D_t <= delta) exactly (DPPO-style uniform test).
        cfg = _config(delta=0.12, delta_b=1000.0, w_min=1.0)
        lp, olp, adv, mask = _rand_inputs(17)
        ref = _reference_mask(lp, olp, adv, mask, cfg)
        d = (torch.exp(lp.detach()) - torch.exp(olp)).abs()
        ratio = torch.exp(torch.clamp(lp.detach() - olp, -20.0, 20.0))
        toward = (adv * (ratio - 1.0)) <= 0
        uniform = (toward | (d <= 0.12)).float() * mask.float()
        assert torch.equal(ref, uniform)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, _reference_loss(lp, olp, adv, mask, cfg), rtol=1e-6, atol=1e-8)

    def test_anchored_old_log_prob_disables_the_mask(self):
        """Documents the loss_func contract: with old_log_prob anchored at
        log_prob.detach() (skip_recompute default), ratio == 1 and D_t == 0 —
        every token is kept and CPPO degenerates. loss_func must feed mu."""
        cfg = _config(delta=1e-9, delta_b=1e-12)
        lp, _, adv, mask = _rand_inputs(19)
        anchored = lp.detach().clone()
        _, metrics = _call(lp, anchored, adv, mask, cfg)
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)
        assert metrics["actor/ppo_kl"] == pytest.approx(0.0)

    def test_agg_mode_token_mean_matches_manual(self):
        cfg = _config(loss_agg_mode="token-mean")
        lp, olp, adv, mask = _rand_inputs(21)
        loss, _ = _call(lp, olp, adv, mask, cfg, loss_agg_mode="token-mean")
        ref_mask = _reference_mask(lp, olp, adv, mask, cfg)
        ratio = torch.exp(torch.clamp(lp.detach() - olp, -20.0, 20.0))
        w = torch.clamp(ratio, max=float(cfg.clip_ratio_c))
        pg = -adv * w * lp * ref_mask
        m = mask.float()
        manual = (pg * m).sum() / m.sum()
        assert torch.allclose(loss, manual, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- metrics & config


class TestMetricsAndConfig:
    def test_metrics_keys_and_ranges(self):
        cfg = _config()
        lp, olp, adv, mask = _rand_inputs(23)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        for key in (
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/cppo_toward_mu_frac",
        ):
            assert key in metrics
        assert 0.0 <= metrics["actor/pg_clipfrac"] <= 1.0
        assert 0.0 <= metrics["actor/cppo_toward_mu_frac"] <= 1.0

    def test_registry_dispatch(self):
        assert get_policy_loss_fn("cppo") is compute_policy_loss_cppo

    def test_cppo_config_defaults(self):
        cfg = CPPOConfig()
        assert cfg.cppo_w_min == pytest.approx(0.8)
        assert cfg.cppo_delta_b == pytest.approx(0.02)
        assert cfg.cppo_delta_b_q == pytest.approx(0.9)
        assert cfg.cppo_delta_b_k == pytest.approx(1.0)
        assert cfg.get("cppo_w_min") == pytest.approx(0.8)  # BaseConfig dict-like access

    def test_policy_loss_config_carries_cppo(self):
        pl = PolicyLossConfig()
        assert isinstance(pl.cppo, CPPOConfig)

    def test_yaml_defaults_present(self):
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        for name in (
            "verl/trainer/config/actor/actor.yaml",
            "verl/trainer/config/_generated_ppo_megatron_trainer.yaml",
        ):
            cfg = OmegaConf.load(os.path.join(repo_root, name))
            node = cfg.actor_rollout_ref.actor.policy_loss.cppo if "generated" in name else cfg.policy_loss.cppo
            assert float(node.cppo_w_min) == pytest.approx(0.8), name
            assert float(node.cppo_delta_b) == pytest.approx(0.02), name

    def test_hyperparameter_validation(self):
        lp, olp, adv, mask = _rand_inputs(25)
        with pytest.raises(AssertionError, match="cppo_w_min"):
            _call(lp, olp, adv, mask, _config(w_min=0.0))
        with pytest.raises(AssertionError, match="cppo_delta_b_q"):
            _call(lp, olp, adv, mask, _config(delta_b_q=1.5))
