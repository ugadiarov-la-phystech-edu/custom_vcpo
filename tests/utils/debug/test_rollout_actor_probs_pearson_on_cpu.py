# Copyright 2025 Individual Contributor: TomQunChaoA
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
"""Unit tests for rollout_actor_probs_pearson_corr, the metric logged as
training/rollout_actor_probs_pearson_corr from the deferred-correction path
when algorithm.rollout_correction.log_probs_pearson_corr is enabled.

Run: pytest tests/utils/debug/test_rollout_actor_probs_pearson_on_cpu.py
"""

import numpy as np
import pytest
import torch

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils.debug.metrics import rollout_actor_probs_pearson_corr


def _reference_pearson(log_p, log_q, mask):
    """Independent numpy reference on the masked prob values."""
    p = np.exp(log_p.numpy())[mask.numpy().astype(bool)]
    q = np.exp(log_q.numpy())[mask.numpy().astype(bool)]
    return float(np.corrcoef(p, q)[0, 1])


def test_identical_logprobs_give_corr_one():
    log_p = -torch.rand(4, 10)
    mask = torch.ones(4, 10)
    assert rollout_actor_probs_pearson_corr(log_p, log_p.clone(), mask) == pytest.approx(1.0, abs=1e-6)


def test_matches_numpy_reference_on_random_inputs():
    torch.manual_seed(0)
    log_p = -torch.rand(3, 8) * 2
    log_q = log_p + 0.3 * torch.randn(3, 8)
    mask = (torch.rand(3, 8) > 0.3).long()
    expected = _reference_pearson(log_p, log_q, mask)
    assert rollout_actor_probs_pearson_corr(log_p, log_q, mask) == pytest.approx(expected, abs=1e-6)


def test_anticorrelated_probs_are_negative():
    # probs p and (c - p) are perfectly anti-correlated
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    q = 0.9 - p
    assert rollout_actor_probs_pearson_corr(p.log(), q.log(), torch.ones_like(p)) == pytest.approx(-1.0, abs=1e-5)


def test_masked_tokens_are_excluded():
    log_p = -torch.rand(2, 6)
    log_q = log_p.clone()
    # corrupt the masked-out positions only; correlation must stay 1
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
    log_q[:, 3:] = -10.0 + torch.randn(2, 3)
    assert rollout_actor_probs_pearson_corr(log_p, log_q, mask) == pytest.approx(1.0, abs=1e-6)


def test_float_mask_is_accepted():
    log_p = -torch.rand(2, 5)
    mask = torch.ones(2, 5, dtype=torch.float32)  # actors pass float/long masks
    assert rollout_actor_probs_pearson_corr(log_p, log_p.clone(), mask) == pytest.approx(1.0, abs=1e-6)


def test_shape_mismatch_returns_zero():
    assert rollout_actor_probs_pearson_corr(torch.zeros(2, 5), torch.zeros(2, 6), torch.ones(2, 5)) == 0.0
    assert rollout_actor_probs_pearson_corr(torch.zeros(2, 5), torch.zeros(2, 5), torch.ones(2, 6)) == 0.0


def test_empty_or_single_token_mask_returns_zero():
    log_p = -torch.rand(2, 5)
    assert rollout_actor_probs_pearson_corr(log_p, log_p.clone(), torch.zeros(2, 5)) == 0.0
    single = torch.zeros(2, 5)
    single[0, 0] = 1
    assert rollout_actor_probs_pearson_corr(log_p, log_p.clone(), single) == 0.0


def test_constant_probs_return_zero_not_nan():
    # zero variance -> corrcoef is NaN; the wrapper must sanitize it
    log_p = torch.full((2, 5), -0.5)
    log_q = -torch.rand(2, 5)
    result = rollout_actor_probs_pearson_corr(log_p, log_q, torch.ones(2, 5))
    assert result == 0.0


def test_returns_python_float():
    log_p = -torch.rand(2, 5)
    result = rollout_actor_probs_pearson_corr(log_p, log_p.clone(), torch.ones(2, 5))
    assert isinstance(result, float)


def test_config_flag_default_off_and_gettable():
    # the actors gate on rollout_corr_config.get("log_probs_pearson_corr", False)
    cfg = RolloutCorrectionConfig()
    assert cfg.get("log_probs_pearson_corr", False) is False
    cfg_on = RolloutCorrectionConfig(log_probs_pearson_corr=True)
    assert cfg_on.get("log_probs_pearson_corr", False) is True
