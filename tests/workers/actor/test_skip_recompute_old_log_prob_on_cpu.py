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

"""CPU tests for the deferred old-log-prob path in DataParallelPPOActor.update_policy.

With ``async_training.skip_recompute_old_log_prob=True`` the trainer ships batches
without ``old_log_probs`` and without centrally computed ``rollout_is_weights``
(recipe/fully_async_policy/ray_trainer.py); the actor must anchor the PPO ratio on
its own forward pass and compute the rollout-correction weights per micro-batch.
That contract was implemented only in MegatronPPOActor; these tests pin the
dp_actor (FSDP backend) port of it.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import verl.workers.actor.dp_actor as dp_actor_module
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig


@pytest.fixture(scope="module", autouse=True)
def single_rank_gloo(tmp_path_factory):
    if not dist.is_initialized():
        init_file = tmp_path_factory.mktemp("dist") / "init"
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Keep the test on CPU even when a GPU is present."""
    monkeypatch.setattr(dp_actor_module, "get_device_id", lambda: "cpu")


def _make_config(**overrides) -> FSDPActorConfig:
    kwargs = dict(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        ppo_epochs=1,
        loss_agg_mode="token-mean",
        entropy_coeff=0,
        calculate_entropy=False,
        use_kl_loss=False,
        grad_clip=1.0,
        use_torch_compile=False,
        clip_ratio=0.2,
    )
    kwargs.update(overrides)
    return FSDPActorConfig(**kwargs)


def _make_actor(config: FSDPActorConfig) -> DataParallelPPOActor:
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()  # log_prob == 0 everywhere -> unclipped ratio anchor of exactly 1
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)

    def fake_forward(model_inputs, temperature, calculate_entropy=False):
        responses = model_inputs["responses"]
        x = torch.ones(*responses.shape, 1, dtype=torch.float32)
        log_prob = actor.actor_module(x).squeeze(-1)  # (bs, resp_len), depends on params
        return None, log_prob

    actor._forward_micro_batch = fake_forward
    return actor


def _make_batch(
    batch_size: int = 4,
    resp_len: int = 6,
    prompt_len: int = 2,
    with_old_log_probs: bool = False,
    skip_recompute: bool = True,
    rollout_corr_config: dict | None = None,
    include_corr_config: bool = True,
) -> DataProto:
    total_len = prompt_len + resp_len
    # Half the response tokens agree with the policy (weight 1); the other half have
    # rollout_log_prob = -log(4), so the token IS weight exp(0 - (-log 4)) = 4 is
    # truncated to the threshold 2.
    rollout_log_probs = torch.zeros(batch_size, resp_len)
    rollout_log_probs[:, resp_len // 2 :] = -torch.log(torch.tensor(4.0))

    tensors = {
        "responses": torch.ones(batch_size, resp_len, dtype=torch.long),
        "response_mask": torch.ones(batch_size, resp_len, dtype=torch.long),
        "input_ids": torch.ones(batch_size, total_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, total_len, dtype=torch.long),
        "position_ids": torch.arange(total_len).expand(batch_size, total_len),
        "advantages": torch.ones(batch_size, resp_len),
        "rollout_log_probs": rollout_log_probs,
    }
    if with_old_log_probs:
        tensors["old_log_probs"] = torch.zeros(batch_size, resp_len)

    data = DataProto.from_dict(tensors=tensors)
    data.meta_info["temperature"] = 1.0
    data.meta_info["skip_recompute_old_log_prob"] = skip_recompute
    if skip_recompute and include_corr_config:
        data.meta_info["rollout_corr_config"] = rollout_corr_config or {
            "rollout_is": "token",
            "rollout_is_threshold": 2.0,
        }
    return data


class TestSkipRecomputePath:
    def test_runs_without_old_log_probs_and_reports_corr_metrics(self):
        actor = _make_actor(_make_config())
        metrics = actor.update_policy(_make_batch())
        assert "actor/pg_loss" in metrics
        assert "rollout_corr/rollout_is_mean" in metrics, (
            "deferred path must compute rollout-correction weights and metrics in the actor"
        )
        # half the tokens exceed the threshold-2 truncation (raw weight 4)
        assert metrics["rollout_corr/rollout_is_ratio_fraction_high"][0] == pytest.approx(0.5)

    def test_is_weights_are_applied_to_the_loss(self):
        """Ratio anchors at exactly 1 (old = current), so with token-mean aggregation
        pg_loss = -mean(adv * w) = -(1 + 2) / 2 = -1.5 per micro-batch."""
        actor = _make_actor(_make_config())
        metrics = actor.update_policy(_make_batch())
        # metrics are recorded per micro-batch, already scaled by 1/grad_accumulation (2)
        for v in metrics["actor/pg_loss"]:
            assert v == pytest.approx(-1.5 / 2, rel=1e-5)

    def test_parameters_receive_gradients_and_update(self):
        actor = _make_actor(_make_config())
        before = actor.actor_module.weight.detach().clone()
        actor.update_policy(_make_batch())
        assert not torch.equal(before, actor.actor_module.weight.detach()), "optimizer step must move the parameters"

    def test_missing_rollout_log_probs_raises(self):
        actor = _make_actor(_make_config())
        data = _make_batch()
        data.batch.pop("rollout_log_probs")
        with pytest.raises(ValueError, match="requires rollout_log_probs"):
            actor.update_policy(data)

    def test_missing_corr_config_raises(self):
        actor = _make_actor(_make_config())
        data = _make_batch(include_corr_config=False)
        with pytest.raises(ValueError, match="requires rollout_corr_config"):
            actor.update_policy(data)

    def test_corr_config_falls_back_to_policy_loss_config(self):
        config = _make_config()
        config.policy_loss.rollout_correction = {"rollout_is": "token", "rollout_is_threshold": 2.0}
        actor = _make_actor(config)
        data = _make_batch(include_corr_config=False)
        metrics = actor.update_policy(data)
        assert "rollout_corr/rollout_is_mean" in metrics


class TestNonSkipPathUnchanged:
    def test_standard_path_still_requires_old_log_probs(self):
        actor = _make_actor(_make_config())
        data = _make_batch(with_old_log_probs=True, skip_recompute=False)
        metrics = actor.update_policy(data)
        assert "actor/pg_loss" in metrics
        # central-weights contract: no weights in batch -> none applied, ratio 1, adv 1
        for v in metrics["actor/pg_loss"]:
            assert v == pytest.approx(-1.0 / 2, rel=1e-5)
