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
"""MegatronPPOActor built through its real __init__ (the other actor tests use __new__).

The reference policy of actor.use_kl_loss is a MegatronPPOActor around a model built with
wrap_with_ddp=False, so its chunks have no ``ddp_config``. __init__ used to read
``actor_module[0].ddp_config`` unconditionally and every Megatron run with a reference model died
at init with "'Float16Module' object has no attribute 'ddp_config'".

Run: pytest tests/workers/actor/test_megatron_actor_init_on_cpu.py
"""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.workers.actor.megatron_actor import MegatronPPOActor


class _Chunk(torch.nn.Module):
    """A model chunk as get_model_config sees it: a module with a ``config``; ``ddp_config`` only
    when the chunk stands for a DDP-wrapped (trainable) model."""

    def __init__(self, use_distributed_optimizer=None):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(sequence_parallel=False, finalize_model_grads_func=None)
        if use_distributed_optimizer is not None:
            self.ddp_config = SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer)


def _config():
    cfg = OmegaConf.create(
        {
            "strategy": "megatron",
            "profiler": {"tool": None},
            "megatron": {"tensor_model_parallel_size": 1, "sequence_parallel": False},
            "log_prob_micro_batch_size_per_gpu": 1,
        }
    )
    OmegaConf.set_struct(cfg, True)  # a missing key raises, as in the worker's Hydra config
    return cfg


def _actor(chunk, optimizer=None):
    return MegatronPPOActor(
        config=_config(),
        model_config=None,
        hf_config=None,
        tf_config=SimpleNamespace(sequence_parallel=False),
        actor_module=torch.nn.ModuleList([chunk]),
        actor_optimizer=optimizer,
    )


def test_a_reference_model_without_ddp_config_builds():
    """Regression: the reference chunk is not DDP-wrapped."""
    ref = _actor(_Chunk(use_distributed_optimizer=None))
    assert ref.use_distributed_opt is False
    assert ref.actor_optimizer is None


@pytest.mark.parametrize("flag", [True, False])
def test_a_ddp_wrapped_actor_keeps_its_distributed_optimizer_flag(flag):
    assert _actor(_Chunk(use_distributed_optimizer=flag)).use_distributed_opt is flag


def test_init_wires_megatron_grad_finalization_into_the_model_config():
    chunk = _Chunk(use_distributed_optimizer=True)
    _actor(chunk)
    assert chunk.config.finalize_model_grads_func is not None
