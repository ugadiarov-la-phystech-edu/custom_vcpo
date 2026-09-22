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
"""core_algos.kl_penalty and its "+" (straight-through) variants.

A "+" suffix keeps the value of the named estimator and replaces its gradient by the k2
gradient ``logprob - ref_logprob``. kl_penalty used to pass the full name (e.g. "low_var_kl+")
to kl_penalty_forward, which knows only the base names, so every "+" type raised
NotImplementedError on the first update (the ORZ-7B KL run of 2026-09-22).

Run: pytest tests/trainer/ppo/test_kl_penalty_on_cpu.py
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import kl_penalty

BASE_TYPES = ["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"]


def _logprobs(seed=0, shape=(3, 5)):
    g = torch.Generator().manual_seed(seed)
    logprob = -torch.rand(shape, generator=g) * 3
    ref_logprob = -torch.rand(shape, generator=g) * 3
    return logprob, ref_logprob


def _grad(name, logprob, ref_logprob):
    lp = logprob.clone().requires_grad_(True)
    ref = ref_logprob.clone().requires_grad_(True)
    kl_penalty(lp, ref, name).sum().backward()
    return lp.grad, ref.grad


@pytest.mark.parametrize("base", BASE_TYPES)
def test_plus_variant_keeps_the_value_of_its_base_estimator(base):
    """Regression: "<base>+" used to raise NotImplementedError."""
    logprob, ref_logprob = _logprobs()
    torch.testing.assert_close(kl_penalty(logprob, ref_logprob, base + "+"), kl_penalty(logprob, ref_logprob, base))


@pytest.mark.parametrize("name", ["low_var_kl+", "k3+", "kl+", "k1+", "abs+", "k2+", "mse+"])
def test_plus_variant_has_the_k2_gradient(name):
    logprob, ref_logprob = _logprobs(seed=1)
    grad_lp, _ = _grad(name, logprob, ref_logprob)
    torch.testing.assert_close(grad_lp, logprob - ref_logprob)


def test_low_var_kl_keeps_its_own_biased_gradient():
    """d(k3)/d logprob = 1 - exp(ref - logprob): the reason the "+" variant exists."""
    logprob, ref_logprob = _logprobs(seed=2)
    grad_lp, _ = _grad("low_var_kl", logprob, ref_logprob)
    torch.testing.assert_close(grad_lp, 1 - torch.exp(ref_logprob - logprob))


def test_k2_plus_is_k2():
    logprob, ref_logprob = _logprobs(seed=3)
    torch.testing.assert_close(kl_penalty(logprob, ref_logprob, "k2+"), kl_penalty(logprob, ref_logprob, "k2"))
    torch.testing.assert_close(_grad("k2+", logprob, ref_logprob)[0], _grad("k2", logprob, ref_logprob)[0])


def test_straight_through_sends_no_gradient_to_the_reference_value_term():
    """The reference log-probs are constants in the loss; with "+" their gradient is the k2 one only."""
    logprob, ref_logprob = _logprobs(seed=4)
    _, grad_ref = _grad("low_var_kl+", logprob, ref_logprob)
    torch.testing.assert_close(grad_ref, ref_logprob - logprob)


@pytest.mark.parametrize("name", ["full", "unknown", "unknown+"])
def test_unknown_types_still_raise(name):
    logprob, ref_logprob = _logprobs()
    with pytest.raises(NotImplementedError):
        kl_penalty(logprob, ref_logprob, name)
