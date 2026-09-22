# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""core_algos.kl_loss_to_reference: the KL-to-reference term shared by the actor backends, optionally
weighted by the truncated rollout IS weights of the policy-gradient term (actor.kl_loss_is_weighted).

Off (the default) it must be exactly the former block ``agg_loss(kl_penalty(...))``; on, every token's
KL is scaled by ``min(pi/mu, c)`` so stale replay tokens the current policy no longer visits are not
pushed back toward the reference.

Run: pytest tests/trainer/ppo/test_kl_loss_to_reference_on_cpu.py
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss, kl_loss_to_reference, kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

AGG_MODES = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
KL_TYPES = ["low_var_kl", "low_var_kl+", "k2", "kl"]


def _batch(seed=0, shape=(4, 6)):
    g = torch.Generator().manual_seed(seed)
    log_prob = -torch.rand(shape, generator=g) * 3
    ref_log_prob = -torch.rand(shape, generator=g) * 3
    mask = torch.ones(shape)
    mask[0, 4:] = 0  # padded tail
    mask[2, :] = 0  # a fully masked (vetoed) row
    weights = torch.rand(shape, generator=g) * 2  # in [0, 2), like truncated token weights
    weights = weights * mask
    return log_prob, ref_log_prob, mask, weights


@pytest.mark.parametrize("agg_mode", AGG_MODES)
@pytest.mark.parametrize("kl_type", KL_TYPES)
def test_off_is_the_former_block_bit_for_bit(agg_mode, kl_type):
    log_prob, ref, mask, weights = _batch()
    former = agg_loss(kl_penalty(log_prob, ref, kl_type), mask, agg_mode)
    for w in (None, weights):  # weights present in the batch but the knob off: ignored
        loss, monitor = kl_loss_to_reference(log_prob, ref, mask, kl_type, agg_mode, rollout_is_weights=w)
        torch.testing.assert_close(loss, former)
        torch.testing.assert_close(monitor, former)
        assert not monitor.requires_grad


@pytest.mark.parametrize("agg_mode", AGG_MODES)
def test_unit_weights_change_nothing(agg_mode):
    log_prob, ref, mask, _ = _batch(seed=1)
    former = agg_loss(kl_penalty(log_prob, ref, "low_var_kl"), mask, agg_mode)
    loss, _ = kl_loss_to_reference(
        log_prob, ref, mask, "low_var_kl", agg_mode, rollout_is_weights=torch.ones_like(mask), is_weighted=True
    )
    torch.testing.assert_close(loss, former)


def test_zero_weights_give_zero_loss_and_zero_gradient():
    log_prob, ref, mask, _ = _batch(seed=2)
    lp = log_prob.clone().requires_grad_(True)
    loss, monitor = kl_loss_to_reference(
        lp, ref, mask, "low_var_kl+", "token-mean", rollout_is_weights=torch.zeros_like(mask), is_weighted=True
    )
    assert loss.item() == 0.0
    assert monitor.item() > 0.0  # the monitor is still the unweighted drift
    loss.backward()
    assert torch.count_nonzero(lp.grad) == 0


@pytest.mark.parametrize("agg_mode", AGG_MODES)
@pytest.mark.parametrize("kl_type", KL_TYPES)
def test_weighted_value_is_the_aggregate_of_the_weighted_kl_matrix(agg_mode, kl_type):
    log_prob, ref, mask, weights = _batch(seed=3)
    kld = kl_penalty(log_prob, ref, kl_type)
    expected = agg_loss(kld * weights, mask, agg_mode)
    loss, monitor = kl_loss_to_reference(
        log_prob, ref, mask, kl_type, agg_mode, rollout_is_weights=weights, is_weighted=True
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(monitor, agg_loss(kld, mask, agg_mode))  # unweighted, for the drift plot


def test_masked_tokens_contribute_nothing_whatever_their_weight():
    log_prob, ref, mask, weights = _batch(seed=4)
    big = weights.clone()
    big[mask == 0] = 1e6  # garbage weights on padding / vetoed tokens
    ref_loss, _ = kl_loss_to_reference(
        log_prob, ref, mask, "low_var_kl", "token-mean", rollout_is_weights=weights, is_weighted=True
    )
    loss, _ = kl_loss_to_reference(
        log_prob, ref, mask, "low_var_kl", "token-mean", rollout_is_weights=big, is_weighted=True
    )
    torch.testing.assert_close(loss, ref_loss)


@pytest.mark.parametrize("level", ["token", "sequence"])
def test_weights_from_the_real_rollout_correction_helper(level):
    """Plumbing check with the weights the actor really computes (truncated at the threshold, masked, detached)."""
    log_prob, ref, mask, _ = _batch(seed=5)
    g = torch.Generator().manual_seed(6)
    rollout_log_prob = log_prob + torch.randn(log_prob.shape, generator=g) * 0.5
    threshold = 2.0
    weights, _ = compute_rollout_correction_weights(
        log_ratio=log_prob - rollout_log_prob,
        response_mask=mask,
        rollout_is=level,
        rollout_is_threshold=threshold,
    )
    assert weights.max() <= threshold
    assert not weights.requires_grad
    if level == "sequence":  # one weight per row, broadcast over the tokens
        for row in weights:
            assert torch.unique(row[row > 0]).numel() <= 1
    kld = kl_penalty(log_prob, ref, "low_var_kl+")
    loss, _ = kl_loss_to_reference(
        log_prob, ref, mask, "low_var_kl+", "seq-mean-token-mean", rollout_is_weights=weights, is_weighted=True
    )
    torch.testing.assert_close(loss, agg_loss(kld * weights, mask, "seq-mean-token-mean"))


def test_gradient_flows_only_through_log_prob_and_is_scaled_by_the_weight():
    """token-mean, "+" type: d loss / d log_prob = w * (log_prob - ref) / n_tokens; nothing reaches the weights."""
    log_prob, ref, mask, weights = _batch(seed=7)
    lp = log_prob.clone().requires_grad_(True)
    w = weights.clone().requires_grad_(True)
    loss, _ = kl_loss_to_reference(lp, ref, mask, "low_var_kl+", "token-mean", rollout_is_weights=w, is_weighted=True)
    loss.backward()
    expected = weights * (log_prob - ref) * mask / mask.sum()
    torch.testing.assert_close(lp.grad, expected)
    assert w.grad is None  # detached inside: IS weights change the measure, not the objective


def test_k3_keeps_its_own_gradient_under_the_weight():
    log_prob, ref, mask, weights = _batch(seed=8)
    lp = log_prob.clone().requires_grad_(True)
    loss, _ = kl_loss_to_reference(
        lp, ref, mask, "low_var_kl", "token-mean", rollout_is_weights=weights, is_weighted=True
    )
    loss.backward()
    expected = weights * (1 - torch.exp(ref - log_prob)) * mask / mask.sum()
    torch.testing.assert_close(lp.grad, expected)


def test_weighting_without_weights_is_refused():
    log_prob, ref, mask, _ = _batch()
    with pytest.raises(ValueError, match="kl_loss_is_weighted"):
        kl_loss_to_reference(log_prob, ref, mask, "low_var_kl", "token-mean", rollout_is_weights=None, is_weighted=True)


def test_mixed_dtypes_run_and_agree_in_dtype():
    """bf16 log-probs with fp32 weights (the actor's case): no promotion error, both aggregates share a dtype."""
    log_prob, ref, mask, weights = _batch(seed=9)
    loss, monitor = kl_loss_to_reference(
        log_prob.bfloat16(),
        ref.bfloat16(),
        mask,
        "low_var_kl+",
        "token-mean",
        rollout_is_weights=weights.float(),
        is_weighted=True,
    )
    assert loss.dtype == monitor.dtype
    assert torch.isfinite(loss) and torch.isfinite(monitor)


def test_agg_kwargs_are_forwarded_to_both_aggregates():
    """The FSDP loss path passes global_batch_info (dp_size, batch_num_tokens, ...) through."""
    log_prob, ref, mask, weights = _batch(seed=10)
    kld = kl_penalty(log_prob, ref, "low_var_kl")
    kw = dict(dp_size=4, batch_num_tokens=100)
    loss, monitor = kl_loss_to_reference(
        log_prob, ref, mask, "low_var_kl", "token-mean", rollout_is_weights=weights, is_weighted=True, **kw
    )
    torch.testing.assert_close(loss, agg_loss(kld * weights, mask, "token-mean", **kw))
    torch.testing.assert_close(monitor, agg_loss(kld, mask, "token-mean", **kw))
