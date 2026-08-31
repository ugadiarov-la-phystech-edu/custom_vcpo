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

"""Unit tests for the adaptive TIS/clip soft blend (async_training.adaptive_anchor).

Per-token loss: L = (1-c2)*L_TIS + c2*L_clip, where L_TIS is the baseline
replay loss (ratio == 1, detached capped pi/mu weight) and L_clip is the
mu-anchored vanilla clip (Arm A). The driver-side AnchorBlendController maps a
smoothed drift signal (the clip piece's pg_clipfrac by default) to c2.

Covered here:
- controller threshold mapping (manual mode), dynamics (instant rise,
  rate-limited fall, EMA smoothing, NaN hold), and AUTO calibration (window
  semantics, freeze, floor, post-freeze equivalence to manual mode);
- the blend's loss/gradient identities on compute_policy_loss_vanilla:
  c2=0 == TIS bitwise, c2=1 == mu-anchored clip bitwise, in-band gradients
  c-independent, out-of-band gradients = (1-c2) * TIS gradient, dual-clip
  corner only in the clip piece, IS weights only on the TIS piece, padding
  inert, clip-piece metrics live even when its loss weight is zero;
- validate_adaptive_anchor_config accept/reject matrix;
- the recipe yaml carries the adaptive_anchor block with safe defaults.

Run: pytest tests/trainer/ppo/test_adaptive_anchor_on_cpu.py
"""

import math
import os

import pytest
import torch
from omegaconf import OmegaConf

from recipe.fully_async_policy.staleness_utils import (
    AnchorBlendController,
    validate_adaptive_anchor_config,
)
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

# ---------------------------------------------------------------------------
# shared fixtures (mirroring test_mu_anchor_loss_on_cpu.py)
# ---------------------------------------------------------------------------


def _config(clip_low=0.2, clip_high=0.28, clip_c=3.0, loss_agg_mode="token-mean"):
    return OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": clip_low,
            "clip_ratio_high": clip_high,
            "clip_ratio_c": clip_c,
            "loss_agg_mode": loss_agg_mode,
            "global_batch_info": {},
            "policy_loss": {"loss_mode": "vanilla", "anchor_mode": None},
        }
    )


def _row(ratios, adv, requires_grad=False):
    """One-row batch with the given per-token pi/mu ratios and constant advantage."""
    ratios = torch.tensor([ratios], dtype=torch.float64)
    mu = torch.zeros_like(ratios)
    log_prob = torch.log(ratios)
    if requires_grad:
        log_prob.requires_grad_(True)
    advantages = torch.full_like(ratios, float(adv))
    mask = torch.ones_like(ratios)
    return log_prob, mu, advantages, mask


def _tis_weights(log_prob, mu, threshold=2.0):
    """The production TIS weight sg(min(pi/mu, threshold))."""
    return torch.exp(log_prob - mu).clamp(max=threshold).detach()


def _piece_tis(log_prob, mu, adv, mask, config):
    return compute_policy_loss_vanilla(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
        rollout_is_weights=_tis_weights(log_prob, mu),
    )


def _piece_clip(log_prob, mu, adv, mask, config):
    return compute_policy_loss_vanilla(
        old_log_prob=mu,
        log_prob=log_prob,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )


def _blend(log_prob, mu, adv, mask, config, c2):
    """The actor's blend: (1-c2)*L_TIS + c2*L_clip; returns (loss, m_tis, m_clip)."""
    tis, m_tis = _piece_tis(log_prob, mu, adv, mask, config)
    clip, m_clip = _piece_clip(log_prob, mu, adv, mask, config)
    return (1.0 - c2) * tis + c2 * clip, m_tis, m_clip


def _grad(loss, log_prob):
    (g,) = torch.autograd.grad(loss, log_prob)
    return g


# ---------------------------------------------------------------------------
# controller: threshold mapping (manual mode)
# ---------------------------------------------------------------------------


def _manual(sig_low=0.02, sig_high=0.10, **kw):
    kw.setdefault("ema_beta", 0.5)
    return AnchorBlendController(sig_low=sig_low, sig_high=sig_high, **kw)


def _drive_to_steady(ctrl, sig, n=200):
    c2 = None
    for _ in range(n):
        c2 = ctrl.update(sig)
    return c2


class TestControllerMapping:
    def test_below_low_gives_c2_min(self):
        assert _drive_to_steady(_manual(), 0.01) == 0.0
        assert _drive_to_steady(_manual(c2_min=0.2), 0.01) == pytest.approx(0.2)

    def test_at_or_above_high_gives_one(self):
        assert _drive_to_steady(_manual(), 0.10) == pytest.approx(1.0)
        assert _drive_to_steady(_manual(), 0.50) == pytest.approx(1.0)

    def test_linear_interpolation(self):
        # sig_low=0.02, sig_high=0.10: sig=0.04 -> 0.25, 0.06 -> 0.5, 0.08 -> 0.75
        for sig, expected in [(0.04, 0.25), (0.06, 0.5), (0.08, 0.75)]:
            assert _drive_to_steady(_manual(), sig) == pytest.approx(expected, abs=1e-9)

    def test_exact_boundaries(self):
        assert _drive_to_steady(_manual(), 0.02) == 0.0
        assert _drive_to_steady(_manual(), 0.10) == pytest.approx(1.0)

    def test_scale_agnostic(self):
        # Same mapping at KL-scale thresholds.
        for sig, expected in [(0.001, 0.0), (0.003, 0.5), (0.004, 1.0)]:
            got = _drive_to_steady(_manual(sig_low=0.002, sig_high=0.004), sig)
            assert got == pytest.approx(expected, abs=1e-9)

    def test_one_sided_manual_rejected(self):
        with pytest.raises(AssertionError, match="BOTH"):
            AnchorBlendController(sig_low=0.02, sig_high=None)
        with pytest.raises(AssertionError, match="BOTH"):
            AnchorBlendController(sig_low=None, sig_high=0.10)

    def test_bad_threshold_order_rejected(self):
        with pytest.raises(AssertionError):
            AnchorBlendController(sig_low=0.10, sig_high=0.02)


class TestControllerDynamics:
    def test_instant_rise(self):
        # ema_beta ~ 0: ema follows the raw signal, so one crisis reading jumps
        # c2 straight to its target — increases are never slew-limited.
        ctrl = _manual(ema_beta=1e-9)
        assert ctrl.update(0.01) == 0.0
        assert ctrl.update(0.06) == pytest.approx(0.5, abs=1e-6)
        assert ctrl.update(0.10) == pytest.approx(1.0, abs=1e-6)

    def test_fall_rate_limited(self):
        ctrl = _manual(ema_beta=1e-9, c2_down_rate=0.05)
        ctrl.update(0.10)  # c2 -> 1.0
        trajectory = [ctrl.update(0.01) for _ in range(5)]
        assert trajectory == pytest.approx([0.95, 0.90, 0.85, 0.80, 0.75], abs=1e-6)

    def test_fall_stops_at_target(self):
        ctrl = _manual(ema_beta=1e-9, c2_down_rate=0.30)
        ctrl.update(0.06)  # c2 -> 0.5
        assert ctrl.update(0.05) == pytest.approx(0.375, abs=1e-6)  # target above c2-0.30

    def test_ema_smoothing(self):
        # beta=0.5, steady 0.01 then one 0.10 outlier: ema = 0.055 -> c2 0.4375,
        # not the raw-signal 1.0.
        ctrl = _manual(ema_beta=0.5)
        _drive_to_steady(ctrl, 0.01, n=50)
        assert ctrl.update(0.10) == pytest.approx((0.055 - 0.02) / 0.08, abs=1e-6)

    def test_nan_and_none_hold_state(self):
        ctrl = _manual(ema_beta=1e-9)
        ctrl.update(0.06)
        before = ctrl.state()
        assert ctrl.update(None) == pytest.approx(before["c2"])
        assert ctrl.update(float("nan")) == pytest.approx(before["c2"])
        assert ctrl.state()["sig_ema"] == pytest.approx(before["sig_ema"])

    def test_monotone_ramp_monotone_c2(self):
        ctrl = _manual(ema_beta=0.5)
        signals = [0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
        c2s = [ctrl.update(s) for s in signals]
        assert all(b >= a for a, b in zip(c2s, c2s[1:], strict=False))


# ---------------------------------------------------------------------------
# controller: AUTO calibration
# ---------------------------------------------------------------------------


def _auto(**kw):
    kw.setdefault("calib_skip", 3)
    kw.setdefault("calib_updates", 5)
    kw.setdefault("low_mult", 5.0)
    kw.setdefault("high_mult", 25.0)
    kw.setdefault("ema_beta", 1e-9)
    return AnchorBlendController(sig_low=None, sig_high=None, **kw)


class TestControllerAutoCalibration:
    def test_c2_pinned_during_calibration(self):
        ctrl = _auto(c2_min=0.1)
        # 3 skipped + 5 window updates, all with a huge signal: still c2_min.
        for _ in range(3 + 5 - 1):
            assert ctrl.update(10.0) == pytest.approx(0.1)
            assert not ctrl.calibrated
        ctrl.update(10.0)
        assert ctrl.calibrated

    def test_head_excluded_from_window(self):
        # Near-zero head (skipped) must not drag the median down.
        ctrl = _auto()
        for _ in range(3):
            ctrl.update(1e-9)
        for _ in range(5):
            ctrl.update(0.004)
        assert ctrl.sig_ref == pytest.approx(0.004)
        assert ctrl.sig_low == pytest.approx(0.02)
        assert ctrl.sig_high == pytest.approx(0.10)

    def test_median_of_window(self):
        ctrl = _auto()
        for _ in range(3):
            ctrl.update(0.001)
        for sig in [0.002, 0.010, 0.004, 0.100, 0.003]:
            ctrl.update(sig)
        assert ctrl.sig_ref == pytest.approx(0.004)  # median, not mean

    def test_nan_does_not_consume_window_slot(self):
        ctrl = _auto()
        for _ in range(3):
            ctrl.update(0.004)
        for _ in range(3):
            ctrl.update(float("nan"))
        for _ in range(4):
            ctrl.update(0.004)
        assert not ctrl.calibrated  # only 4 valid window samples so far
        ctrl.update(0.004)
        assert ctrl.calibrated

    def test_ref_floor(self):
        ctrl = _auto(sig_ref_floor=1e-4)
        for _ in range(3 + 5):
            ctrl.update(1e-7)
        assert ctrl.sig_ref == pytest.approx(1e-4)
        assert ctrl.sig_low == pytest.approx(5e-4)

    def test_thresholds_frozen_after_calibration(self):
        ctrl = _auto()
        for _ in range(3 + 5):
            ctrl.update(0.004)
        low, high = ctrl.sig_low, ctrl.sig_high
        for _ in range(50):
            ctrl.update(1.0)
        assert ctrl.sig_low == low and ctrl.sig_high == high

    def test_post_freeze_equals_manual(self):
        auto = _auto()
        for _ in range(3 + 5):
            auto.update(0.004)
        manual = AnchorBlendController(sig_low=auto.sig_low, sig_high=auto.sig_high, ema_beta=1e-9)
        signals = [0.004, 0.03, 0.08, 0.12, 0.05, 0.004, 0.004]
        # Align both controllers' state (ema/c2) before comparing trajectories.
        auto_c2 = [auto.update(s) for s in signals]
        manual.update(0.004)
        manual_c2 = [manual.update(s) for s in signals]
        # Same mapping; auto had one extra warm step, so compare from index 1.
        assert auto_c2[1:] == pytest.approx(manual_c2[1:], abs=1e-9)

    def test_manual_mode_skips_calibration(self):
        ctrl = _manual(ema_beta=1e-9)
        assert ctrl.calibrated
        assert ctrl.update(0.06) == pytest.approx(0.5, abs=1e-6)  # first update already maps


# ---------------------------------------------------------------------------
# blend loss math
# ---------------------------------------------------------------------------


class TestBlendLossMath:
    def test_c2_zero_is_tis_bitwise(self):
        log_prob, mu, adv, mask = _row([0.5, 1.0, 1.5, 5.0], adv=1.0)
        blended, _, _ = _blend(log_prob, mu, adv, mask, _config(), c2=0.0)
        tis, _ = _piece_tis(log_prob, mu, adv, mask, _config())
        assert blended.item() == tis.item()

    def test_c2_one_is_clip_bitwise(self):
        log_prob, mu, adv, mask = _row([0.5, 1.0, 1.5, 5.0], adv=-1.0)
        blended, _, _ = _blend(log_prob, mu, adv, mask, _config(), c2=1.0)
        clip, _ = _piece_clip(log_prob, mu, adv, mask, _config())
        assert blended.item() == clip.item()

    def test_in_band_gradient_c_independent(self):
        # r=1.1 (inside [0.8, 1.28]), A=1: both pieces have gradient -A*r per
        # token, so the blended gradient must not move with c2.
        grads = []
        for c2 in [0.0, 0.3, 0.7, 1.0]:
            log_prob, mu, adv, mask = _row([1.1], adv=1.0, requires_grad=True)
            loss, _, _ = _blend(log_prob, mu, adv, mask, _config(), c2=c2)
            grads.append(_grad(loss, log_prob))
        for g in grads[1:]:
            assert torch.allclose(g, grads[0], atol=1e-12)
        assert grads[0].item() == pytest.approx(-1.1, rel=1e-9)

    def test_out_of_band_gradient_is_leak_scaled_tis(self):
        # A>0, r=1.5 > 1.28: clip piece gradient 0; TIS gradient -A*min(1.5, 2).
        # Blend gradient must be exactly (1-c2) * TIS gradient.
        log_prob, mu, adv, mask = _row([1.5], adv=2.0, requires_grad=True)
        tis, _ = _piece_tis(log_prob, mu, adv, mask, _config())
        g_tis = _grad(tis, log_prob)
        assert g_tis.item() == pytest.approx(-2.0 * 1.5, rel=1e-9)
        for c2 in [0.25, 0.6, 0.9]:
            log_prob, mu, adv, mask = _row([1.5], adv=2.0, requires_grad=True)
            loss, _, _ = _blend(log_prob, mu, adv, mask, _config(), c2=c2)
            g = _grad(loss, log_prob)
            assert g.item() == pytest.approx((1.0 - c2) * g_tis.item(), rel=1e-9)

    def test_out_of_band_negative_adv_side(self):
        # A<0, r=0.5 < 0.8: clip piece gradient 0 (push-down stopped); TIS
        # gradient -A*w = +0.5. Blend = (1-c2) * that.
        for c2 in [0.0, 0.5, 1.0]:
            log_prob, mu, adv, mask = _row([0.5], adv=-1.0, requires_grad=True)
            loss, _, _ = _blend(log_prob, mu, adv, mask, _config(), c2=c2)
            g = _grad(loss, log_prob)
            assert g.item() == pytest.approx((1.0 - c2) * 0.5, rel=1e-9)

    def test_dual_clip_corner_only_in_clip_piece(self):
        # A<0, r=5 >> 1: dual-clip floors the clip piece (zero gradient,
        # pg_clipfrac_lower > 0); the TIS piece keeps its capped push
        # -A*min(5, 2) = +2.
        log_prob, mu, adv, mask = _row([5.0], adv=-1.0, requires_grad=True)
        loss, m_tis, m_clip = _blend(log_prob, mu, adv, mask, _config(), c2=0.5)
        assert m_clip["actor/pg_clipfrac_lower"] > 0.0
        assert m_tis["actor/pg_clipfrac_lower"] == 0.0
        g = _grad(loss, log_prob)
        assert g.item() == pytest.approx(0.5 * 2.0, rel=1e-9)

    def test_is_weights_touch_tis_piece_only(self):
        # The clip piece is called with rollout_is_weights=None; feeding the
        # weights into it too (double counting) must change the result.
        log_prob, mu, adv, mask = _row([1.5, 0.9], adv=1.0)
        clip_clean, _ = _piece_clip(log_prob, mu, adv, mask, _config())
        clip_stacked, _ = compute_policy_loss_vanilla(
            old_log_prob=mu,
            log_prob=log_prob,
            advantages=adv,
            response_mask=mask,
            loss_agg_mode="token-mean",
            config=_config(),
            rollout_is_weights=_tis_weights(log_prob, mu),
        )
        assert clip_clean.item() != pytest.approx(clip_stacked.item())

    def test_padded_positions_inert(self):
        log_prob, mu, adv, mask = _row([1.5, 3.0], adv=1.0, requires_grad=True)
        mask = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        loss, _, _ = _blend(log_prob, mu, adv, mask, _config(), c2=0.5)
        g = _grad(loss, log_prob)
        assert g[0, 1].item() == 0.0

    def test_clip_metrics_live_under_no_grad(self):
        # The actor evaluates the clip piece under torch.no_grad() when c2=0 —
        # its pg_clipfrac (the controller signal) must still be computed.
        log_prob, mu, adv, mask = _row([1.0, 1.5], adv=1.0)
        with torch.no_grad():
            _, m_clip = _piece_clip(log_prob, mu, adv, mask, _config())
        assert m_clip["actor/pg_clipfrac"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# startup validation matrix
# ---------------------------------------------------------------------------


def _good_cfgs():
    adaptive = {"enable": True, "signal": "clipfrac"}
    policy_loss = {"anchor_mode": None, "loss_mode": "vanilla"}
    rollout_corr = {"rollout_is": "token", "rollout_rs": None}
    return adaptive, policy_loss, rollout_corr


class TestValidateConfig:
    def test_good_config_passes(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)

    def test_kl_signal_passes(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        adaptive["signal"] = "kl"
        validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)

    def test_unknown_signal_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        adaptive["signal"] = "entropy"
        with pytest.raises(AssertionError, match="signal"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)

    def test_static_anchor_mode_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        policy_loss["anchor_mode"] = "mu"
        with pytest.raises(AssertionError, match="anchor_mode"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)

    def test_non_vanilla_loss_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        policy_loss["loss_mode"] = "cppo"
        with pytest.raises(AssertionError, match="vanilla"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)

    def test_no_skip_recompute_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        with pytest.raises(AssertionError, match="skip_recompute"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, False)

    def test_opob_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        with pytest.raises(AssertionError, match="grad_baselining"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, True, True)

    def test_seq_level_is_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        rollout_corr["rollout_is"] = "sequence"
        with pytest.raises(AssertionError, match="token"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)

    def test_missing_rollout_corr_rejected(self):
        adaptive, policy_loss, _ = _good_cfgs()
        with pytest.raises(AssertionError, match="rollout_is"):
            validate_adaptive_anchor_config(adaptive, policy_loss, None, False, True)

    def test_rollout_rs_rejected(self):
        adaptive, policy_loss, rollout_corr = _good_cfgs()
        rollout_corr["rollout_rs"] = "sequence"
        with pytest.raises(AssertionError, match="rollout_rs"):
            validate_adaptive_anchor_config(adaptive, policy_loss, rollout_corr, False, True)


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class TestLossFuncMetaInfoPlumbing:
    """forward_step hands loss_func a CONSTRUCTED meta_info dict, not the
    batch's — driver-stamped keys must be forwarded explicitly. Regression for
    the bug where anchor_blend_c2 was dropped and the blend ran inert."""

    @staticmethod
    def _cfg():
        return OmegaConf.create({"clip_ratio": 0.2, "entropy_coeff": 0.0, "clip_ratio_c": 3.0})

    def test_training_meta_forwards_anchor_blend_c2(self):
        from verl.workers.actor.megatron_actor import build_loss_func_meta_info

        meta = build_loss_func_meta_info(
            self._cfg(),
            {"anchor_blend_c2": 0.37, "global_seq_mean_count": 33, "collect_seq_log_is": True},
            forward_only=False,
            skip_recompute_old_log_prob=True,
            rollout_corr_cfg={"rollout_is": "token"},
        )
        assert meta["anchor_blend_c2"] == 0.37
        assert meta["skip_recompute_old_log_prob"] is True
        assert meta["rollout_corr_config"] == {"rollout_is": "token"}
        assert meta["global_seq_mean_count"] == 33
        assert meta["collect_seq_log_is"] is True
        assert meta["clip_ratio"] == 0.2 and meta["clip_ratio_c"] == 3.0

    def test_training_meta_defaults_blend_off(self):
        from verl.workers.actor.megatron_actor import build_loss_func_meta_info

        meta = build_loss_func_meta_info(
            self._cfg(), {}, forward_only=False, skip_recompute_old_log_prob=True, rollout_corr_cfg=None
        )
        # None (not a KeyError, not 0.0): loss_func's `blend_c2 is None` check
        # selects the static path when the driver did not stamp the key.
        assert meta["anchor_blend_c2"] is None

    def test_forward_only_meta_has_no_blend_key(self):
        from verl.workers.actor.megatron_actor import build_loss_func_meta_info

        meta = build_loss_func_meta_info(
            self._cfg(),
            {"anchor_blend_c2": 0.5},
            forward_only=True,
            skip_recompute_old_log_prob=False,
            rollout_corr_cfg=None,
        )
        # log-prob-only passes never reach the loss blend; .get() in loss_func
        # tolerates the absent key.
        assert "anchor_blend_c2" not in meta
        assert meta["loss_multiplier"] == 1.0

    def test_explicit_zero_loss_multiplier_honored(self):
        from verl.workers.actor.megatron_actor import build_loss_func_meta_info

        meta = build_loss_func_meta_info(
            self._cfg(),
            {"loss_multiplier": 0.0},
            forward_only=False,
            skip_recompute_old_log_prob=True,
            rollout_corr_cfg=None,
        )
        assert meta["loss_multiplier"] == 0.0


class TestGradMethodLogging:
    """Which gradient method is active (TIS / blend / PPO-clip) must be legible
    from metrics and log lines."""

    @pytest.mark.parametrize(
        "c2,code,name",
        [
            (0.0, 0.0, "TIS"),
            (-0.0, 0.0, "TIS"),
            (1e-6, 1.0, "blend"),
            (0.5, 1.0, "blend"),
            (1.0 - 1e-6, 1.0, "blend"),
            (1.0, 2.0, "PPO-clip"),
        ],
    )
    def test_grad_method_mapping(self, c2, code, name):
        assert AnchorBlendController.grad_method_code(c2) == code
        assert AnchorBlendController.grad_method_name(c2) == name

    def test_state_carries_grad_method(self):
        ctrl = AnchorBlendController(sig_low=0.01, sig_high=0.05)
        assert ctrl.state()["grad_method"] == 0.0  # starts at c2_min=0 -> pure TIS
        for _ in range(20):
            ctrl.update(0.03)  # mid-band signal -> partial c2
        state = ctrl.state()
        assert 0.0 < state["c2"] < 1.0
        assert state["grad_method"] == 1.0
        for _ in range(50):
            ctrl.update(1.0)  # saturating signal -> c2 pinned at 1
        state = ctrl.state()
        assert state["c2"] == 1.0
        assert state["grad_method"] == 2.0

    def test_state_grad_method_tracks_c2_descent(self):
        ctrl = AnchorBlendController(sig_low=0.01, sig_high=0.05, c2_down_rate=0.5)
        for _ in range(50):
            ctrl.update(1.0)
        assert ctrl.state()["grad_method"] == 2.0
        for _ in range(50):
            ctrl.update(0.0)  # healthy signal -> c2 decays back to c2_min
        assert ctrl.state()["c2"] == 0.0
        assert ctrl.state()["grad_method"] == 0.0


class TestConfigPlumbing:
    def test_recipe_yaml_carries_adaptive_anchor_defaults(self):
        cfg = OmegaConf.load(
            os.path.join(REPO_ROOT, "recipe", "fully_async_policy", "config", "fully_async_ppo_megatron_trainer.yaml")
        )
        block = cfg.async_training.adaptive_anchor
        assert block.enable is False  # off by default
        assert block.signal == "clipfrac"
        assert block.sig_low is None and block.sig_high is None  # AUTO by default
        assert block.low_mult < block.high_mult
        assert block.calib_updates >= 1
        assert 0.0 <= block.c2_min <= 1.0
        assert 0.0 < block.ema_beta < 1.0
        assert block.c2_down_rate > 0.0
        assert math.isclose(block.sig_ref_floor, 1e-4)
