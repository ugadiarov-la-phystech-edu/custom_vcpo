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
"""CPU tests for verl.workers.actor.entropy_utils (actor/entropy logging in the
Megatron update path, used e.g. by the fully-async recipe's bypass mode)."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.actor.entropy_utils import log_entropy_and_apply_to_loss, should_calculate_entropy
from verl.workers.config.actor import ActorConfig


class TestShouldCalculateEntropy:
    def test_default_config_is_off(self):
        cfg = SimpleNamespace(calculate_entropy=False, entropy_coeff=0)
        assert should_calculate_entropy(cfg) is False

    def test_calculate_entropy_flag_enables(self):
        cfg = SimpleNamespace(calculate_entropy=True, entropy_coeff=0)
        assert should_calculate_entropy(cfg) is True

    def test_nonzero_entropy_coeff_enables(self):
        cfg = SimpleNamespace(calculate_entropy=False, entropy_coeff=0.01)
        assert should_calculate_entropy(cfg) is True

    def test_config_without_field_falls_back_to_coeff(self):
        # configs predating the calculate_entropy field must keep working
        cfg = SimpleNamespace(entropy_coeff=0)
        assert should_calculate_entropy(cfg) is False
        cfg = SimpleNamespace(entropy_coeff=0.001)
        assert should_calculate_entropy(cfg) is True

    def test_actor_config_dataclass_has_field_default_false(self):
        cfg = ActorConfig(strategy="megatron", rollout_n=1, ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=1)
        assert cfg.calculate_entropy is False
        assert should_calculate_entropy(cfg) is False


def _make_inputs():
    """2 sequences x 4 response tokens; second half of seq 1 masked out."""
    entropy = torch.tensor(
        [
            [0.5, 1.0, 1.5, 2.0],
            [2.0, 4.0, 999.0, 999.0],  # masked positions get absurd values on purpose
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    pg_loss = torch.tensor(0.7)
    return pg_loss, entropy, response_mask


class TestLogEntropyAndApplyToLoss:
    def test_metric_logged_token_mean_respects_mask(self):
        pg_loss, entropy, mask = _make_inputs()
        metrics = {}
        log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="token-mean",
            entropy_coeff=0,
            metrics=metrics,
        )
        # token-mean over unmasked tokens: (0.5+1+1.5+2+2+4)/6
        assert metrics["actor/entropy"] == pytest.approx(11.0 / 6.0)

    def test_metric_logged_seq_mean_token_mean(self):
        pg_loss, entropy, mask = _make_inputs()
        metrics = {}
        log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            entropy_coeff=0,
            metrics=metrics,
        )
        # per-seq token means: (0.5+1+1.5+2)/4 = 1.25 and (2+4)/2 = 3.0 -> mean 2.125
        assert metrics["actor/entropy"] == pytest.approx(2.125)

    def test_zero_coeff_returns_loss_unchanged(self):
        pg_loss, entropy, mask = _make_inputs()
        metrics = {}
        out = log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="token-mean",
            entropy_coeff=0,
            metrics=metrics,
        )
        assert out is pg_loss  # objective must not be perturbed by pure monitoring

    def test_zero_coeff_accepts_no_grad_entropy_and_keeps_loss_graph(self):
        # mirrors production: entropy computed under torch.no_grad() when log-only,
        # while pg_loss carries the autograd graph
        pg_loss = torch.tensor(0.7, requires_grad=True) * 2.0
        _, entropy, mask = _make_inputs()
        with torch.no_grad():
            entropy = entropy.clone()
        metrics = {}
        out = log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="token-mean",
            entropy_coeff=0,
            metrics=metrics,
        )
        assert out.requires_grad
        assert "actor/entropy" in metrics

    def test_nonzero_coeff_subtracts_entropy_bonus(self):
        pg_loss, entropy, mask = _make_inputs()
        coeff = 0.01
        metrics = {}
        out = log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="token-mean",
            entropy_coeff=coeff,
            metrics=metrics,
        )
        expected = 0.7 - coeff * (11.0 / 6.0)
        assert out.item() == pytest.approx(expected)
        assert metrics["actor/entropy"] == pytest.approx(11.0 / 6.0)

    def test_nonzero_coeff_gradient_flows_through_entropy(self):
        _, entropy, mask = _make_inputs()
        entropy = entropy.clone().requires_grad_(True)
        pg_loss = torch.tensor(0.7, requires_grad=True)
        out = log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="token-mean",
            entropy_coeff=0.01,
            metrics={},
        )
        out.backward()
        assert entropy.grad is not None
        # masked positions must not receive gradient
        assert torch.all(entropy.grad[~mask] == 0)

    def test_metric_is_detached_python_float(self):
        pg_loss, entropy, mask = _make_inputs()
        entropy = entropy.clone().requires_grad_(True)
        metrics = {}
        log_entropy_and_apply_to_loss(
            pg_loss=pg_loss,
            entropy=entropy,
            response_mask=mask,
            loss_agg_mode="token-mean",
            entropy_coeff=0.01,
            metrics=metrics,
        )
        assert isinstance(metrics["actor/entropy"], float)
