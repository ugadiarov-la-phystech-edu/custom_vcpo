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
"""Tests for should_calculate_entropy, the predicate update_policy uses on the megatron
path.

It decides whether the training forward computes token entropy — which on the non-fused
path clones the logits, so it must stay off unless asked for, and must be reachable at
entropy_coeff=0 (the fully_async `is-pg` arm has no old-log-prob forward, so this is its
only source of actor/entropy).

Run: pytest tests/workers/actor/test_entropy_flag_on_cpu.py
"""

from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.actor.megatron_actor import should_calculate_entropy


def _cfg(entropy_coeff, calculate_entropy):
    return OmegaConf.create({"entropy_coeff": entropy_coeff, "calculate_entropy": calculate_entropy})


def test_entropy_in_the_loss_always_wins():
    """Unchanged upstream behaviour: a non-zero coefficient needs entropy regardless."""
    assert should_calculate_entropy(_cfg(0.01, False)) is True
    assert should_calculate_entropy(_cfg(0.01, True)) is True


def test_diagnostic_request_is_honoured_at_zero_coefficient():
    assert should_calculate_entropy(_cfg(0, True)) is True


def test_off_by_default():
    """The expensive path must not turn on by accident: at entropy_coeff=0 with no
    request, no entropy (and no logits clone)."""
    assert should_calculate_entropy(_cfg(0, False)) is False
    # and when the key is absent entirely
    assert should_calculate_entropy(OmegaConf.create({"entropy_coeff": 0})) is False


def test_accepts_the_real_actor_config():
    config = omega_conf_to_dataclass(
        {
            "_target_": "verl.workers.config.McoreActorConfig",
            "strategy": "megatron",
            "ppo_micro_batch_size_per_gpu": 1,
            "entropy_coeff": 0,
            "calculate_entropy": True,
            "optim": {"_target_": "verl.workers.config.McoreOptimizerConfig", "lr": 1e-6},
            "rollout_n": 1,
        }
    )
    assert should_calculate_entropy(config) is True
