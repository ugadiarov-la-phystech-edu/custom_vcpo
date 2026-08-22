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
    delta_b_max_mult=5.0,
    w_len_mode="sequence",
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
                    "cppo_delta_b_max_mult": delta_b_max_mult,
                    "cppo_w_len_mode": w_len_mode,
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
    db_mult = float(cfg.policy_loss.cppo.get("cppo_delta_b_max_mult", 5.0))
    w_len_mode = str(cfg.policy_loss.cppo.get("cppo_w_len_mode", "sequence"))
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
            db_seq = float(np.clip(k * np.quantile(d_row[valid], q), db_floor, db_mult * db_floor))
        else:
            db_seq = db_floor
        # "sequence" spans this row's own valid length (paper Eq. 9); "padded" spans the
        # padded response width (reference implementation).
        t_span = float(m[i].sum()) if w_len_mode == "sequence" else float(t_len)
        s_prev, w_prev = 0.0, 0.0
        for t in range(t_len):
            pos = t + 1  # 1-based absolute token position
            # Algorithm 1 line 3 verbatim: w_t = 1 - (1 - w_min)(t - 1)/(T - 1)
            elapsed = min(max((pos - 1.0) / max(t_span - 1.0, 1.0), 0.0), 1.0)
            w_t = (1.0 - (1.0 - w_min) * elapsed) * m[i, t]
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
    @pytest.mark.parametrize("w_len_mode", ["sequence", "padded"])
    @pytest.mark.parametrize("seed", range(6))
    def test_loss_matches_reference_on_random_inputs(self, seed, w_len_mode):
        cfg = _config(w_len_mode=w_len_mode)
        lp, olp, adv, mask = _rand_inputs(seed)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8), f"seed {seed}: {loss} vs {ref}"

    @pytest.mark.parametrize("w_len_mode", ["sequence", "padded"])
    @pytest.mark.parametrize("seed", range(4))
    def test_loss_matches_reference_with_variable_lengths(self, seed, w_len_mode):
        cfg = _config(w_len_mode=w_len_mode)
        mask = torch.zeros(B, T, dtype=torch.long)
        for i, k in enumerate([T, 3, 1]):
            mask[i, :k] = 1
        lp, olp, adv, _ = _rand_inputs(seed, mask=mask)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("delta,delta_b,w_min", [(0.05, 0.005, 0.5), (0.3, 0.1, 1.0), (0.15, 0.0, 0.8)])
    @pytest.mark.parametrize("w_len_mode", ["sequence", "padded"])
    def test_loss_matches_reference_across_hyperparameters(self, delta, delta_b, w_min, w_len_mode):
        cfg = _config(delta=delta, delta_b=delta_b, w_min=w_min, w_len_mode=w_len_mode)
        lp, olp, adv, mask = _rand_inputs(3)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("delta_b_max_mult", [1.0, 2.0, 5.0, 10.0])
    def test_loss_matches_reference_across_budget_ceilings(self, delta_b_max_mult):
        """The ceiling is the one place the reference implementation departs from the
        paper's Eq. 22 (5x vs 2x), so both must be exercised."""
        cfg = _config(delta_b_max_mult=delta_b_max_mult, delta_b=0.01, delta_b_k=3.0)
        lp, olp, adv, mask = _rand_inputs(5)
        loss, _ = _call(lp, olp, adv, mask, cfg)
        ref = _reference_loss(lp, olp, adv, mask, cfg)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- w_t length convention


def _w_t(cfg, mask):
    """The position weights the loss builds, recovered through the reference mirror."""
    b, t = mask.shape
    w_min = float(cfg.policy_loss.cppo.cppo_w_min)
    mode = str(cfg.policy_loss.cppo.cppo_w_len_mode)
    out = torch.zeros(b, t)
    for i in range(b):
        span = float(mask[i].sum()) if mode == "sequence" else float(t)
        for j in range(t):
            pos = j + 1
            elapsed = min(max((pos - 1.0) / max(span - 1.0, 1.0), 0.0), 1.0)
            out[i, j] = (1.0 - (1.0 - w_min) * elapsed) * mask[i, j]
    return out


class TestWeightLengthModes:
    """The paper's Eq. 9 spans each response's own length; the reference implementation
    spans the padded width. They agree only on full-length rows, and the gap is what makes
    the position mechanism inert when responses are much shorter than the length cap
    (see CPPO_PORT_PARAMS_AND_CAVEATS_DISCUSSION.md)."""

    def test_sequence_mode_reaches_w_min_on_the_last_valid_token(self):
        cfg = _config(w_len_mode="sequence", w_min=0.8)
        mask = torch.zeros(1, 16, dtype=torch.long)
        mask[0, :4] = 1
        w = _w_t(cfg, mask)
        assert w[0, 3].item() == pytest.approx(0.8, abs=1e-6)  # last valid token hits the floor
        assert w[0, 0].item() == pytest.approx(1.0, abs=1e-6)  # first token gets full weight
        assert w[0, 4:].abs().sum().item() == 0.0  # padding is inert

    def test_padded_mode_leaves_the_schedule_nearly_flat_for_short_rows(self):
        """The concern that drove the default: a 4-token response inside a 16-wide pad
        only traverses a fifth of the schedule."""
        cfg = _config(w_len_mode="padded", w_min=0.8)
        mask = torch.zeros(1, 16, dtype=torch.long)
        mask[0, :4] = 1
        w = _w_t(cfg, mask)
        assert w[0, 3].item() > 0.93  # nowhere near w_min = 0.8
        assert w[0, 3].item() < 1.0

    def test_modes_agree_on_a_full_length_row(self):
        mask = torch.ones(1, 8, dtype=torch.long)
        w_seq = _w_t(_config(w_len_mode="sequence"), mask)
        w_pad = _w_t(_config(w_len_mode="padded"), mask)
        assert torch.allclose(w_seq, w_pad, atol=1e-7)

    def test_sequence_mode_is_invariant_to_trailing_padding(self):
        """Same response, different padding width -> identical loss under 'sequence'."""
        lp_vals = [-0.4, -0.9, -0.2, -1.1]
        olp_vals = [-1.9, -2.4, -0.25, -2.6]
        losses = []
        for width in (4, 9, 32):
            lp = torch.tensor([lp_vals + [0.0] * (width - 4)], requires_grad=True)
            olp = torch.tensor([olp_vals + [0.0] * (width - 4)])
            adv = torch.full((1, width), 1.0)
            mask = torch.zeros(1, width, dtype=torch.long)
            mask[0, :4] = 1
            loss, _ = _call(lp, olp, adv * mask, mask, _config(w_len_mode="sequence", loss_agg_mode="token-mean"))
            losses.append(loss.item())
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)
        assert losses[0] == pytest.approx(losses[2], rel=1e-6)

    @staticmethod
    def _padded_masks_at(delta):
        # moderate divergences (D ~ 0.07) so the token-level clause, not saturation,
        # decides: pi = e^-1.6 = 0.202 against mu = e^-2.0 = 0.135
        lp_vals = [-1.60, -1.62, -1.58, -1.60]
        olp_vals = [-2.0, -2.0, -2.0, -2.0]
        masks = []
        for width in (4, 64):
            lp = torch.tensor([lp_vals + [0.0] * (width - 4)], requires_grad=True)
            olp = torch.tensor([olp_vals + [0.0] * (width - 4)])
            adv = torch.full((1, width), 1.0)
            mask = torch.zeros(1, width, dtype=torch.long)
            mask[0, :4] = 1
            cfg = _config(w_len_mode="padded", delta=delta)
            masks.append(_reference_mask(lp, olp, adv * mask, mask, cfg)[0, :4].clone())
        return masks

    def test_padded_mode_is_not_invariant_to_trailing_padding(self):
        """The mirror image of the previous test: under 'padded' the very same response
        is masked differently depending only on how wide the micro-batch happens to be.
        Swept over the threshold so the claim does not hinge on one hand-picked delta."""
        flipped = [
            d / 100.0
            for d in range(1, 60)
            if not torch.equal(*self._padded_masks_at(d / 100.0))
        ]
        assert flipped, "expected some delta where padding width alone changes the mask"

    def test_first_token_always_carries_full_weight(self):
        """Algorithm 1 line 3 has (t - 1) in the numerator, so w_1 = 1 for every T —
        including T = 1, where the algebraically rearranged form degenerates to w_min
        (the LOOSEST weight) instead."""
        for mode in ("sequence", "padded"):
            for valid in (1, 2, 5):
                mask = torch.zeros(1, 5, dtype=torch.long)
                mask[0, :valid] = 1
                w = _w_t(_config(w_len_mode=mode, w_min=0.8), mask)
                assert w[0, 0].item() == pytest.approx(1.0, abs=1e-6), f"{mode}, valid={valid}"

    def test_single_token_response_is_gated_at_delta_not_delta_over_w_min(self):
        """The concrete consequence: a one-token response must be tested against delta
        itself. D = |e^-1.6 - e^-2.0| ~ 0.067 sits between delta = 0.06 and delta/0.8."""
        mask = torch.ones(1, 1, dtype=torch.long)
        lp = torch.tensor([[-1.6]], requires_grad=True)
        olp = torch.tensor([[-2.0]])
        adv = torch.tensor([[1.0]])  # positive advantage, ratio > 1 -> away from mu
        masked = _reference_mask(lp, olp, adv, mask, _config(delta=0.06, w_min=0.8))
        assert masked[0, 0].item() == 0.0  # w_1 = 1 -> 0.067 > 0.06 -> masked
        _, metrics = _call(lp, olp, adv, mask, _config(delta=0.06, w_min=0.8))
        assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)

    def test_single_valid_token_row_is_finite(self):
        """T - 1 == 0 must not divide by zero in either mode."""
        for mode in ("sequence", "padded"):
            mask = torch.zeros(1, 1, dtype=torch.long)
            mask[0, 0] = 1
            lp = torch.tensor([[-0.5]], requires_grad=True)
            olp = torch.tensor([[-2.0]])
            adv = torch.tensor([[1.0]])
            loss, _ = _call(lp, olp, adv, mask, _config(w_len_mode=mode))
            assert torch.isfinite(loss)

    def test_empty_row_is_safe_in_sequence_mode(self):
        mask = torch.zeros(2, 5, dtype=torch.long)
        mask[0, :3] = 1  # second row entirely padding
        lp, olp, adv, _ = _rand_inputs(1, b=2, t=5, mask=mask)
        loss, _ = _call(lp, olp, adv, mask, _config(w_len_mode="sequence"))
        assert torch.isfinite(loss)


class TestBudgetCeiling:
    """cppo_delta_b_max_mult: the paper's Eq. 22 clamps at 2x the floor, the reference
    implementation at 5x."""

    @staticmethod
    def _masked_fraction(cfg, seed=7):
        lp, olp, adv, mask = _rand_inputs(seed)
        _, metrics = _call(lp, olp, adv, mask, cfg)
        return metrics["actor/pg_clipfrac"]

    def test_higher_ceiling_masks_no_more_than_a_lower_one(self):
        """A larger ceiling can only raise delta_b^seq, which can only loosen the mask."""
        strict = self._masked_fraction(_config(delta_b=0.005, delta_b_k=5.0, delta_b_max_mult=1.0))
        loose = self._masked_fraction(_config(delta_b=0.005, delta_b_k=5.0, delta_b_max_mult=10.0))
        assert loose <= strict + 1e-9

    def test_ceiling_binds_only_above_the_floor_multiple(self):
        """With k*quantile below the floor the ceiling is irrelevant: 2x and 5x agree."""
        a = self._masked_fraction(_config(delta_b=1.0, delta_b_k=0.0, delta_b_max_mult=2.0))
        b = self._masked_fraction(_config(delta_b=1.0, delta_b_k=0.0, delta_b_max_mult=5.0))
        assert a == pytest.approx(b)

    def test_mult_of_one_pins_delta_b_to_the_floor(self):
        pinned = self._masked_fraction(_config(delta_b=0.01, delta_b_k=100.0, delta_b_max_mult=1.0))
        floor_only = self._masked_fraction(_config(delta_b=0.01, delta_b_k=0.0, delta_b_max_mult=5.0))
        assert pinned == pytest.approx(floor_only)


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
        loss_solo, _ = _call(
            lp[:1].detach().requires_grad_(True), olp[:1], adv[:1], mask[:1], cfg
        )
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
        uniform = ((toward | (d <= 0.12)).float() * mask.float())
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


class TestCalibrationDiagnostics:
    """delta_b has to be chosen from data (the paper calibrates it per model),
    and the quantity it is thresholded against — the WEIGHTED divergence
    Z_t = w_t*D_t — is not otherwise observable. These three metrics are the
    paper's Fig. 7 instrumentation."""

    def test_delta_b_seq_is_the_floor_when_the_calibration_is_off(self):
        """k = 0 pins delta_b^seq to the floor: the fixed-budget regime the
        paper uses for post-trained models."""
        lp, olp, adv, mask = _rand_inputs(0)
        _, m = _call(lp, olp, adv, mask, _config(delta_b=0.017, delta_b_k=0.0))
        assert m["actor/cppo_delta_b_seq"] == pytest.approx(0.017, rel=1e-6)

    def test_delta_b_seq_tracks_the_quantile_when_calibrated(self):
        """With k = 1 and a wide clamp it reports k * quantile(D), i.e. it
        follows the batch's own drift — the self-scaling behaviour."""
        lp, olp, adv, mask = _rand_inputs(1)
        cfg = _config(delta_b=1e-6, delta_b_k=1.0, delta_b_q=0.9, delta_b_max_mult=1e6)
        _, m = _call(lp, olp, adv, mask, cfg)

        import numpy as np

        d = (lp.detach().exp() - olp.exp()).abs().numpy()
        expected = float(np.mean([np.quantile(row, 0.9) for row in d]))
        assert m["actor/cppo_delta_b_seq"] == pytest.approx(expected, rel=1e-5)

    def test_delta_b_seq_honours_the_ceiling(self):
        cfg = _config(delta_b=0.01, delta_b_k=1000.0, delta_b_max_mult=3.0)
        lp, olp, adv, mask = _rand_inputs(2)
        _, m = _call(lp, olp, adv, mask, cfg)
        assert m["actor/cppo_delta_b_seq"] == pytest.approx(0.03, rel=1e-6)

    def test_weighted_div_mean_matches_a_hand_computed_masked_mean(self):
        w_min = 0.8
        mask = torch.zeros(1, 5, dtype=torch.long)
        mask[0, :4] = 1
        lp = torch.tensor([[-1.6, -1.6, -1.6, -1.6, 0.0]], requires_grad=True)
        olp = torch.tensor([[-2.0, -2.0, -2.0, -2.0, 0.0]])
        adv = torch.ones(1, 5) * mask
        _, m = _call(lp, olp, adv, mask, _config(w_min=w_min))

        d = abs(math.exp(-1.6) - math.exp(-2.0))
        # w_t over the row's own 4 valid tokens: 1, 1-0.2/3, 1-0.4/3, 0.8
        ws = [1.0 - (1.0 - w_min) * t / 3.0 for t in range(4)]
        expected = sum(w * d for w in ws) / 4
        assert m["actor/cppo_weighted_div_mean"] == pytest.approx(expected, rel=1e-5)

    def test_budget_mask_frac_is_zero_when_the_budget_cannot_bind(self):
        lp, olp, adv, mask = _rand_inputs(3)
        _, m = _call(lp, olp, adv, mask, _config(delta_b=1e6, delta_b_k=0.0))
        assert m["actor/cppo_budget_mask_frac"] == pytest.approx(0.0)

    def test_budget_mask_frac_accounts_for_the_masking_when_only_the_budget_binds(self):
        """delta is NOT just the token-level threshold: Eq. 8 also seeds the
        prefix threshold with it (c_t = min(delta, delta + delta_b*W - S)), so a
        huge delta makes BOTH clauses vacuous. To isolate the budget, keep delta
        comfortably above w_t*D_t and let the prefix sum outgrow that slack: with
        delta_b ~ 0 the running S exceeds delta after delta/(w*D) tokens, and
        every rejection from there on is the budget's."""
        T_long = 64
        mask = torch.ones(1, T_long, dtype=torch.long)
        lp = torch.full((1, T_long), -1.6, requires_grad=True)   # pi = 0.202
        olp = torch.full((1, T_long), -2.0)                      # mu = 0.135, D ~ 0.067
        adv = torch.ones(1, T_long)                              # positive adv, ratio > 1 -> away from mu
        _, m = _call(lp, olp, adv, mask, _config(delta=0.5, delta_b=1e-9, delta_b_k=0.0))

        # token-level clause can never reject here (w*D <= 0.067 << delta = 0.5)
        assert m["actor/pg_clipfrac"] > 0.5
        assert m["actor/cppo_budget_mask_frac"] == pytest.approx(m["actor/pg_clipfrac"], rel=1e-6)

    def test_budget_mask_frac_never_exceeds_the_total_masked_fraction(self):
        for seed in range(4):
            lp, olp, adv, mask = _rand_inputs(seed)
            _, m = _call(lp, olp, adv, mask, _config(delta=0.05, delta_b=0.01, delta_b_k=0.0))
            assert m["actor/cppo_budget_mask_frac"] <= m["actor/pg_clipfrac"] + 1e-9

    def test_diagnostics_are_finite_on_an_all_padding_row(self):
        mask = torch.zeros(2, 5, dtype=torch.long)
        mask[0, :3] = 1
        lp, olp, adv, _ = _rand_inputs(5, b=2, t=5, mask=mask)
        _, m = _call(lp, olp, adv, mask, _config())
        for k in ("actor/cppo_delta_b_seq", "actor/cppo_weighted_div_mean", "actor/cppo_budget_mask_frac"):
            assert math.isfinite(m[k]), k


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
