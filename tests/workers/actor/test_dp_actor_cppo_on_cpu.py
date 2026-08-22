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
"""CPPO wiring on the FSDP path (verl/workers/actor/dp_actor.py).

The loss itself is covered by tests/trainer/ppo/test_cppo_loss_on_cpu.py; what this file
pins is everything the actor has to get right for the mask to mean anything in the replay
arms:

- the mu anchor. Every replay arm runs skip_recompute_old_log_prob=True, so
  `old_log_prob = log_prob.detach()` and the PPO ratio is anchored at 1. Handing that to
  CPPO would make D_t == 0 for every token and the trust region would never bind, so the
  actor must pass the cached rollout log-probs (mu) instead.
- the token-IS weights are not applied on top of CPPO's own truncated ratio.
- the three refusals (missing rollout_log_probs, rollout_rs, sequence-level rollout_is).
- seq_adv_post_scale, which all four min-ESS arms enable, is rejected: it calls the loss
  with unit advantages, collapsing CPPO's advantage-sign clause.
- the min-ESS brake keeps working underneath: it is orthogonal (step-level lr vs
  token-level mask).

Run: pytest tests/workers/actor/test_dp_actor_cppo_on_cpu.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import verl.trainer.ppo.core_algos as core_algos
import verl.workers.actor.dp_actor as dp_actor_module
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import CPPOConfig, ESSScalingConfig, FSDPActorConfig, PolicyLossConfig

RESP_LEN = 6
BATCH_SIZE = 4
NOMINAL_LR = 0.1
DELTA = 0.15


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
    monkeypatch.setattr(dp_actor_module, "get_device_id", lambda: "cpu")


def _config(loss_mode="cppo", seq_adv_post_scale=False, ess=False, **cppo_overrides) -> FSDPActorConfig:
    kwargs = dict(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=BATCH_SIZE,
        ppo_micro_batch_size_per_gpu=BATCH_SIZE,
        use_dynamic_bsz=False,
        ppo_epochs=1,
        loss_agg_mode="seq-mean-token-mean",
        entropy_coeff=0,
        calculate_entropy=False,
        use_kl_loss=False,
        grad_clip=1.0,
        use_torch_compile=False,
        clip_ratio=DELTA,
        clip_ratio_c=20.0,
        seq_adv_post_scale=seq_adv_post_scale,
        policy_loss=PolicyLossConfig(loss_mode=loss_mode, cppo=CPPOConfig(**cppo_overrides)),
    )
    if ess:
        kwargs["ess_scaling"] = ESSScalingConfig(enable=True, min_ess=1.1, lr_scale=0.5)
    return FSDPActorConfig(**kwargs)


def _make_actor(config: FSDPActorConfig, policy_logprob: float = -0.4) -> DataParallelPPOActor:
    """Actor whose forward returns a constant log-prob, so pi is a knob."""
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(policy_logprob)
    optimizer = torch.optim.SGD(module.parameters(), lr=NOMINAL_LR)
    actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)

    def fake_forward(model_inputs, temperature, calculate_entropy=False):
        responses = model_inputs["responses"]
        x = torch.ones(*responses.shape, 1, dtype=torch.float32)
        return None, actor.actor_module(x).squeeze(-1)

    actor._forward_micro_batch = fake_forward
    return actor


def _make_batch(
    mu_logprob: float = -2.0,
    advantages: list[float] | None = None,
    skip_recompute: bool = True,
    include_rollout_log_probs: bool = True,
    rollout_corr_config: dict | None = None,
) -> DataProto:
    adv = torch.ones(BATCH_SIZE, RESP_LEN)
    if advantages is not None:
        adv = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1).expand(BATCH_SIZE, RESP_LEN).clone()

    total_len = RESP_LEN + 2
    tensors = {
        "responses": torch.ones(BATCH_SIZE, RESP_LEN, dtype=torch.long),
        "response_mask": torch.ones(BATCH_SIZE, RESP_LEN, dtype=torch.long),
        "input_ids": torch.ones(BATCH_SIZE, total_len, dtype=torch.long),
        "attention_mask": torch.ones(BATCH_SIZE, total_len, dtype=torch.long),
        "position_ids": torch.arange(total_len).expand(BATCH_SIZE, total_len),
        "advantages": adv,
    }
    if include_rollout_log_probs:
        tensors["rollout_log_probs"] = torch.full((BATCH_SIZE, RESP_LEN), float(mu_logprob))
    if not skip_recompute:
        tensors["old_log_probs"] = torch.zeros(BATCH_SIZE, RESP_LEN)
    data = DataProto.from_dict(tensors=tensors)
    data.meta_info["temperature"] = 1.0
    data.meta_info["skip_recompute_old_log_prob"] = skip_recompute
    if skip_recompute:
        data.meta_info["rollout_corr_config"] = (
            {"rollout_is": "token", "rollout_is_threshold": 2.0} if rollout_corr_config is None
            else rollout_corr_config
        )
    return data


def _spy_on_loss(monkeypatch, loss_mode="cppo"):
    """Capture the arguments the actor hands the policy-loss function."""
    captured = {}
    original = core_algos.get_policy_loss_fn(loss_mode)

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(dp_actor_module, "get_policy_loss_fn", lambda mode: spy if mode == loss_mode else original)
    return captured


class TestMuAnchor:
    def test_loss_receives_rollout_log_probs_not_the_anchor(self, monkeypatch):
        """The whole point: with skip_recompute the anchor is log_prob.detach(), which
        would make every D_t zero."""
        captured = _spy_on_loss(monkeypatch)
        actor = _make_actor(_config(), policy_logprob=-0.4)
        actor.update_policy(_make_batch(mu_logprob=-2.0))

        assert torch.allclose(captured["old_log_prob"], torch.full_like(captured["old_log_prob"], -2.0))
        assert not torch.allclose(captured["old_log_prob"], captured["log_prob"].detach())

    def test_mask_actually_binds_against_mu(self, monkeypatch):
        """pi = e^-0.4 vs mu = e^-2.0 is D ~ 0.53, far above delta = 0.15, and the
        advantage is positive with ratio > 1 -> away from mu -> tokens are masked."""
        captured = _spy_on_loss(monkeypatch)
        actor = _make_actor(_config(), policy_logprob=-0.4)
        metrics = actor.update_policy(_make_batch(mu_logprob=-2.0, advantages=[1.0] * BATCH_SIZE))
        assert captured, "loss fn was never called"
        assert metrics["actor/pg_clipfrac"][0] > 0.9

    def test_close_policy_is_not_masked(self, monkeypatch):
        """Same wiring, tiny divergence -> nothing is masked, so the previous test is
        measuring the mask and not a constant."""
        _spy_on_loss(monkeypatch)
        actor = _make_actor(_config(), policy_logprob=-1.99)
        metrics = actor.update_policy(_make_batch(mu_logprob=-2.0, advantages=[1.0] * BATCH_SIZE))
        assert metrics["actor/pg_clipfrac"][0] == pytest.approx(0.0, abs=1e-6)

    def test_anchored_old_log_prob_would_disable_the_mask(self, monkeypatch):
        """Control: hand the loss the anchor instead of mu and the mask never binds —
        the failure mode the swap exists to prevent."""
        lp = torch.full((BATCH_SIZE, RESP_LEN), -0.4, requires_grad=True)
        _, metrics = core_algos.get_policy_loss_fn("cppo")(
            old_log_prob=lp.detach(),
            log_prob=lp,
            advantages=torch.ones(BATCH_SIZE, RESP_LEN),
            response_mask=torch.ones(BATCH_SIZE, RESP_LEN, dtype=torch.long),
            loss_agg_mode="seq-mean-token-mean",
            config=_config(),
            rollout_is_weights=None,
        )
        assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0, abs=1e-9)

    def test_token_is_weights_are_not_applied_on_top(self, monkeypatch):
        """CPPO's own truncated ratio replaces them; applying both would double-count."""
        captured = _spy_on_loss(monkeypatch)
        actor = _make_actor(_config())
        actor.update_policy(_make_batch())
        assert captured["rollout_is_weights"] is None

    def test_vanilla_still_gets_the_anchor_and_its_weights(self, monkeypatch):
        """The swap must not leak into other loss modes."""
        captured = _spy_on_loss(monkeypatch, loss_mode="vanilla")
        actor = _make_actor(_config(loss_mode="vanilla"))
        actor.update_policy(_make_batch())
        assert torch.allclose(captured["old_log_prob"], captured["log_prob"].detach())
        assert captured["rollout_is_weights"] is not None


class TestRefusals:
    def test_missing_rollout_log_probs_is_refused(self):
        """Refused either by the pipeline's own skip_recompute check (which fires first
        on this path) or by CPPO's assert — what matters is that mu-less batches never
        reach the mask."""
        actor = _make_actor(_config())
        with pytest.raises((AssertionError, ValueError), match="rollout_log_probs"):
            actor.update_policy(_make_batch(include_rollout_log_probs=False))

    def test_rollout_rs_is_refused(self):
        actor = _make_actor(_config())
        batch = _make_batch(
            rollout_corr_config={
                "rollout_is": "token",
                "rollout_is_threshold": 2.0,
                "rollout_rs": "token",
                "rollout_rs_threshold": 2.0,
            }
        )
        with pytest.raises(AssertionError, match="rollout_rs"):
            actor.update_policy(batch)

    def test_sequence_level_rollout_is_is_refused(self):
        actor = _make_actor(_config())
        batch = _make_batch(rollout_corr_config={"rollout_is": "sequence", "rollout_is_threshold": 2.0})
        with pytest.raises(AssertionError, match="rollout_is"):
            actor.update_policy(batch)

    def test_seq_adv_post_scale_is_refused(self):
        """All four min-ESS arms set this; with unit advantages CPPO's
        A_t*(rho_t - 1) <= 0 clause degenerates to rho_t <= 1."""
        actor = _make_actor(_config(seq_adv_post_scale=True))
        with pytest.raises(NotImplementedError, match="seq_adv_post_scale"):
            actor.update_policy(_make_batch())

    def test_seq_adv_post_scale_still_allowed_for_vanilla(self):
        actor = _make_actor(_config(loss_mode="vanilla", seq_adv_post_scale=True))
        metrics = actor.update_policy(_make_batch())
        assert "actor/pg_loss" in metrics or "actor/grad_norm" in metrics


class TestAdvantageSignIsPreserved:
    def test_negative_advantage_rows_use_the_toward_mu_clause(self, monkeypatch):
        """A row moving pi ABOVE mu with a negative advantage is moving toward mu in the
        loss's sense (A*(rho-1) <= 0), so it must be kept however large D is — the
        behaviour seq_adv_post_scale would have destroyed."""
        captured = _spy_on_loss(monkeypatch)
        actor = _make_actor(_config(), policy_logprob=-0.4)  # pi >> mu -> ratio > 1
        metrics = actor.update_policy(_make_batch(mu_logprob=-2.0, advantages=[-1.0] * BATCH_SIZE))
        assert torch.all(captured["advantages"] < 0), "the real advantages must reach the loss"
        assert metrics["actor/pg_clipfrac"][0] == pytest.approx(0.0, abs=1e-6)


class TestCoexistsWithTheEssBrake:
    def test_brake_engages_under_cppo(self):
        """The min-ESS brake is orthogonal (step-level lr vs token-level mask): a
        degenerate batch must still brake with loss_mode=cppo."""
        actor = _make_actor(_config(ess=True), policy_logprob=-0.4)
        stepped = []
        orig_step = actor.actor_optimizer.step

        def recording_step():
            stepped.append(float(actor.actor_optimizer.param_groups[0]["lr"]))
            return orig_step()

        actor.actor_optimizer.step = recording_step

        # one sequence dominates the IS weights -> ESS ~ 1 -> brake
        batch = _make_batch(mu_logprob=-2.0)
        batch.batch["rollout_log_probs"][1:] = -0.4  # rows 1..3 sit on top of pi
        metrics = actor.update_policy(batch)

        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_ess"] <= 1.1
        assert stepped == [pytest.approx(NOMINAL_LR * 0.5, rel=1e-6)]

    def test_ess_entry_is_emitted_with_cppo(self):
        actor = _make_actor(_config(ess=True), policy_logprob=-1.99)
        metrics = actor.update_policy(_make_batch(mu_logprob=-2.0))
        (entry,) = metrics["staleness/ess"]
        assert set(entry.keys()) >= {"minibatch_ess", "minibatch_ess_ratio", "ess_scaled_lr"}


class TestMegatronPathRefuses:
    """The mu substitution lives in dp_actor only. loss_mode=cppo is registered globally,
    so the Megatron actor has to refuse the anchored regime rather than train a loss that
    is CPPO in name only (ratio == 1 -> D_t == 0 -> nothing ever masked)."""

    def test_refuses_cppo_under_skip_recompute(self):
        from verl.workers.actor.megatron_actor import refuse_cppo_without_mu_anchor

        with pytest.raises(NotImplementedError, match="cppo"):
            refuse_cppo_without_mu_anchor("cppo", True)

    @pytest.mark.parametrize(
        "loss_mode,skip_recompute",
        [
            ("cppo", False),  # the paper's own regime: real old_log_probs, mask binds
            ("vanilla", True),
            ("vanilla", False),
            ("gspo", True),
        ],
    )
    def test_allows_everything_else(self, loss_mode, skip_recompute):
        from verl.workers.actor.megatron_actor import refuse_cppo_without_mu_anchor

        refuse_cppo_without_mu_anchor(loss_mode, skip_recompute)


class TestClipRatioCTrap:
    """CPPO repurposes clip_ratio_c as its truncated-IS cap (paper: 20.0), but the repo
    default is the dual-clip constant 3.0, so an un-overridden script trains a 6.7x more
    truncated objective. It is legal, so it warns rather than raising."""

    def test_warns_once_on_the_dual_clip_default(self, caplog):
        import verl.trainer.ppo.core_algos as ca

        ca._CPPO_CLIP_RATIO_C_WARNED = False
        cfg = _config()
        object.__setattr__(cfg, "clip_ratio_c", 3.0)  # the repo default a copied script inherits
        actor = _make_actor(cfg)
        with caplog.at_level("WARNING"):
            actor.update_policy(_make_batch())
        assert any("clip_ratio_c" in r.message for r in caplog.records)

    def test_warns_only_once_per_process(self, caplog):
        import verl.trainer.ppo.core_algos as ca

        ca._CPPO_CLIP_RATIO_C_WARNED = False
        cfg = _config()
        object.__setattr__(cfg, "clip_ratio_c", 3.0)
        actor = _make_actor(cfg)
        with caplog.at_level("WARNING"):
            actor.update_policy(_make_batch())
            actor.update_policy(_make_batch())
        assert sum("clip_ratio_c" in r.message for r in caplog.records) == 1

    def test_does_not_warn_for_the_cppo_cap(self, caplog):
        import verl.trainer.ppo.core_algos as ca

        ca._CPPO_CLIP_RATIO_C_WARNED = False
        actor = _make_actor(_config())  # the fixture already uses 20.0
        with caplog.at_level("WARNING"):
            actor.update_policy(_make_batch())
        assert not any("clip_ratio_c" in r.message for r in caplog.records)


class TestConfigSurface:
    def test_defaults_are_the_chosen_ones(self):
        cfg = CPPOConfig()
        assert (cfg.cppo_w_min, cfg.cppo_delta_b, cfg.cppo_delta_b_q, cfg.cppo_delta_b_k) == (0.8, 0.02, 0.9, 1.0)
        # the two knobs where the reference departs from the paper
        assert cfg.cppo_delta_b_max_mult == 5.0  # reference value; paper Eq. 22 uses 2
        assert cfg.cppo_w_len_mode == "sequence"  # paper Eq. 9; reference uses "padded"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cppo_w_min": 0.0},
            {"cppo_w_min": 1.5},
            {"cppo_delta_b": -0.1},
            {"cppo_delta_b_q": 1.5},
            {"cppo_delta_b_k": -1.0},
            {"cppo_delta_b_max_mult": 0.5},
            {"cppo_w_len_mode": "per-sequence"},
        ],
    )
    def test_invalid_values_are_rejected(self, kwargs):
        with pytest.raises(AssertionError):
            CPPOConfig(**kwargs)

    def test_policy_loss_config_carries_cppo(self):
        assert isinstance(PolicyLossConfig().cppo, CPPOConfig)

    def test_both_w_len_modes_run_through_the_actor(self, monkeypatch):
        for mode in ("sequence", "padded"):
            actor = _make_actor(_config(cppo_w_len_mode=mode), policy_logprob=-0.4)
            metrics = actor.update_policy(_make_batch(mu_logprob=-2.0))
            assert "actor/pg_clipfrac" in metrics
