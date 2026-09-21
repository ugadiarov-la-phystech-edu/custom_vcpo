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
"""The reference model of actor.use_kl_loss on the ORZ-7B replay arm, with the arm's real config.

The ORZ arm is composed with use_kl_loss=True (`--cfg job --resolve`, no GPU, no Ray), and its
actor_rollout_ref section drives the real worker code on CPU:
  * DetachActorWorker.init_model for the "ref" role (recipe override -> megatron_workers.init_model),
    with the model build replaced by a small non-DDP module: the reference MegatronPPOActor is created
    (it used to die on the missing ddp_config), its parameters are offloaded when
    ref.megatron.param_offload is on, and no actor model is registered for the reset copy;
  * the reference forward (MegatronPPOActor.compute_log_prob) with the Megatron pipeline replaced by
    a stub: it reads only keys the reference config has, and returns the per-token log-probs;
  * the actor config of the same composition carries the KL settings the reference feeds.

Run: pytest recipe/fully_async_policy/unittest/test_kl_reference_model_on_cpu.py
"""

import functools
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import recipe.fully_async_policy.megatron_worker as megatron_worker_module
import verl.workers.actor.megatron_actor as megatron_actor_module
from recipe.fully_async_policy.megatron_worker import DetachActorWorker
from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.actor.megatron_actor import MegatronPPOActor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ORZ_ARM = os.path.join(
    REPO_ROOT,
    "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/"
    "grpo_novcpo_8gpu_orz72k_3+5_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_orz7b.sh",
)
KL_ENV = (("use_kl_loss", "True"), ("kl_loss_coef", "0.45"), ("kl_loss_type", "low_var_kl"))
RESPONSE_LEN = 4


@functools.cache
def _composed(extra_env=KL_ENV):
    if not os.path.exists(ORZ_ARM):
        raise unittest.SkipTest(f"{os.path.basename(ORZ_ARM)} not found")
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet", **dict(extra_env))
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
            ["bash", ORZ_ARM, "--cfg", "job", "--resolve"],
            cwd=REPO_ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.PIPE,
            timeout=900,
        )
        if proc.returncode != 0:
            raise unittest.SkipTest(f"could not compose the ORZ arm: {proc.stderr.decode()[-300:]}")
        out.flush()
        return OmegaConf.load(out.name)


def _worker_config(extra_env=KL_ENV):
    """The worker's config: the arm's actor_rollout_ref, strict like the Hydra config it receives."""
    cfg = OmegaConf.create(OmegaConf.to_container(_composed(extra_env).actor_rollout_ref))
    OmegaConf.set_struct(cfg, True)
    return cfg


class _RefChunk(torch.nn.Module):
    """What make_megatron_module returns for the reference: a Float16Module-like chunk with a
    ``config`` and no ``ddp_config`` (wrap_with_ddp=False)."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(sequence_parallel=False, finalize_model_grads_func=None)


def _ref_worker(cfg):
    """A DetachActorWorker with the state ActorRolloutRefWorker.__init__ leaves for role "ref"
    (torch.distributed, Megatron parallel state and the profiler are not set up on CPU)."""
    w = object.__new__(DetachActorWorker)
    w.config = cfg
    w._is_actor, w._is_rollout, w._is_ref = False, False, True
    w._ref_is_offload_param = bool(cfg.ref.megatron.get("param_offload", False))
    w.hf_config = SimpleNamespace()
    w.tf_config = SimpleNamespace(sequence_parallel=False)
    chunk = _RefChunk()
    w.built_with = []

    def fake_build(model_path, optim_config, override_model_config, override_transformer_config, **kwargs):
        w.built_with.append({"model_path": model_path, "optim_config": optim_config, **kwargs})
        return torch.nn.ModuleList([chunk]), w.hf_config

    w._build_model_optimizer = fake_build
    return w


@pytest.fixture
def no_registered_actor(monkeypatch):
    monkeypatch.setattr(megatron_worker_module, "_LOCAL_ACTOR_MODULES", {})
    return megatron_worker_module._LOCAL_ACTOR_MODULES


# ==================== the composed arm ====================


def test_the_arm_switches_the_reference_on_with_the_kl_knobs():
    cfg = _composed()
    actor = cfg.actor_rollout_ref.actor
    assert actor.use_kl_loss is True
    assert actor.kl_loss_coef == pytest.approx(0.45)
    assert actor.kl_loss_type == "low_var_kl"
    assert cfg.actor_rollout_ref.ref.megatron.param_offload is True  # the arm's ref_param_offload default
    assert cfg.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1


def test_the_actor_config_converts_with_the_kl_settings():
    actor_cfg = omega_conf_to_dataclass(_worker_config().actor)
    assert actor_cfg.use_kl_loss is True
    assert actor_cfg.kl_loss_coef == pytest.approx(0.45)
    assert actor_cfg.kl_loss_type == "low_var_kl"
    assert actor_cfg.update_policy_per_traj is True
    assert actor_cfg.grad_baselining.enable is False  # OPOB is incompatible with the KL loss


# ==================== reference worker init ====================


def test_reference_worker_init_builds_a_reference_actor(no_registered_actor):
    """Regression: init_model of the "ref" role died in MegatronPPOActor.__init__ on the
    missing ddp_config of the non-DDP reference model."""
    w = _ref_worker(_worker_config())
    DetachActorWorker.init_model(w)
    assert isinstance(w.ref_policy, MegatronPPOActor)
    assert w.ref_policy.use_distributed_opt is False
    assert w.ref_policy.actor_optimizer is None
    assert w.built_with[0]["optim_config"] is None  # the reference gets no optimizer
    assert w.built_with[0]["model_path"] == w.config.model.path
    assert "actor" not in no_registered_actor  # only the actor role registers its model for resets


def test_reference_parameters_are_offloaded_when_the_arm_asks_for_it(no_registered_actor, monkeypatch):
    offloaded = []
    monkeypatch.setattr(
        "verl.workers.megatron_workers.offload_megatron_model_to_cpu", lambda models: offloaded.append(models)
    )
    w = _ref_worker(_worker_config())
    DetachActorWorker.init_model(w)
    assert offloaded == [w.ref_module]


def test_reference_parameters_stay_when_offload_is_off(no_registered_actor, monkeypatch):
    offloaded = []
    monkeypatch.setattr(
        "verl.workers.megatron_workers.offload_megatron_model_to_cpu", lambda models: offloaded.append(models)
    )
    w = _ref_worker(_worker_config(KL_ENV + (("ref_param_offload", "False"),)))
    DetachActorWorker.init_model(w)
    assert offloaded == []


# ==================== reference forward ====================


@pytest.fixture
def single_process_group():
    """compute_log_prob broadcasts over the pipeline group: a 1-process gloo group stands in."""
    created = False
    if not dist.is_initialized():
        with tempfile.NamedTemporaryFile() as f:
            dist.init_process_group("gloo", init_method=f"file://{f.name}.pg", rank=0, world_size=1)
        created = True
    yield
    if created:
        dist.destroy_process_group()


@pytest.fixture
def megatron_pipeline_stub(monkeypatch, single_process_group):
    monkeypatch.setattr(megatron_actor_module.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(megatron_actor_module.mpu, "get_pipeline_model_parallel_last_rank", lambda: 0)
    monkeypatch.setattr(megatron_actor_module.mpu, "get_pipeline_model_parallel_group", lambda: None)
    monkeypatch.setattr(megatron_actor_module, "get_device_id", lambda: "cpu")


def _rollout_batch(n=3, prompt_len=2):
    seq_len = prompt_len + RESPONSE_LEN
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(n * seq_len).reshape(n, seq_len),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).repeat(n, 1),
            "responses": torch.zeros(n, RESPONSE_LEN, dtype=torch.long),
        },
        meta_info={"temperature": 1.0, "micro_batch_size": 1, "use_dynamic_bsz": False, "max_token_len": None},
    )


def test_reference_forward_returns_per_token_log_probs(no_registered_actor, megatron_pipeline_stub):
    w = _ref_worker(_worker_config())
    DetachActorWorker.init_model(w)
    data = _rollout_batch()
    seen = {}

    def fake_forward_backward_batch(batch, forward_only, calculate_entropy, micro_batch_size, **kwargs):
        seen.update(forward_only=forward_only, calculate_entropy=calculate_entropy, micro_batch_size=micro_batch_size)
        rows = batch.batch["input_ids"][:, :1].float()
        return {"output": [{"log_probs": -(rows[i : i + 1].expand(1, RESPONSE_LEN))} for i in range(len(rows))]}

    w.ref_policy.forward_backward_batch = fake_forward_backward_batch
    log_probs, _ = w.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
    assert seen == {"forward_only": True, "calculate_entropy": False, "micro_batch_size": 1}
    assert log_probs.shape == (3, RESPONSE_LEN)
    assert torch.equal(log_probs[:, 0], -data.batch["input_ids"][:, 0].float())
