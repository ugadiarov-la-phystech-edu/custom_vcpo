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

"""CPU tests for the FSDP port of ESS-guided LR scaling (VCPO) and the
sequence-level advantage post-scale loss parity mode in DataParallelPPOActor.

These drive the REAL dp_actor.update_policy on a stub linear model (same
harness style as test_skip_recompute_old_log_prob_on_cpu.py):

- staleness/ess structured metrics are emitted per mini-batch with the same
  keys the fully-async trainer consumes (_capture_ess_base, replay metrics);
- the brake scales the stepped LR by sqrt(ess_ratio/base) below the trigger,
  leaves it nominal at/above it, and always restores the nominal LR;
- base_ess_ratio=None resolves to meta_info["ess_base_override"] (auto-base),
  and stays a no-op while the override is unresolved;
- seq_adv_post_scale computes the clipped loss with UNIT advantages and
  post-scales per sequence — asserted to differ from in-loss advantages
  exactly where the clip branch would flip (negative advantages).

Run: pytest tests/workers/actor/test_dp_actor_ess_on_cpu.py
"""

import math

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import verl.workers.actor.dp_actor as dp_actor_module
from verl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import ESSScalingConfig, FSDPActorConfig

RESP_LEN = 6
BATCH_SIZE = 4
NOMINAL_LR = 0.1


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


def _make_config(**overrides) -> FSDPActorConfig:
    kwargs = dict(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=BATCH_SIZE,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        ppo_epochs=1,
        loss_agg_mode="seq-mean-token-mean",
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
        module.weight.zero_()  # log_prob == 0 everywhere -> policy anchor exp(0)=1
    optimizer = torch.optim.SGD(module.parameters(), lr=NOMINAL_LR)
    actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)

    def fake_forward(model_inputs, temperature, calculate_entropy=False):
        responses = model_inputs["responses"]
        x = torch.ones(*responses.shape, 1, dtype=torch.float32)
        log_prob = actor.actor_module(x).squeeze(-1)  # (bs, resp_len)
        return None, log_prob

    actor._forward_micro_batch = fake_forward
    return actor


def _record_stepped_lrs(actor) -> list[float]:
    """Wrap optimizer.step to record the LR actually used for each step."""
    stepped = []
    orig_step = actor.actor_optimizer.step

    def recording_step():
        stepped.append(float(actor.actor_optimizer.param_groups[0]["lr"]))
        return orig_step()

    actor.actor_optimizer.step = recording_step
    return stepped


def _make_batch(
    seq_is_targets: list[float],
    advantages: list[float] | None = None,
    resp_len: int = RESP_LEN,
    skip_recompute: bool = True,
    include_rollout_log_probs: bool = True,
) -> DataProto:
    """Build a batch whose per-sequence IS ratios (policy anchor = 1) equal
    seq_is_targets: rollout_log_probs per token = -ln(w)/resp_len."""
    assert len(seq_is_targets) == BATCH_SIZE
    rollout_log_probs = torch.zeros(BATCH_SIZE, resp_len)
    for i, w in enumerate(seq_is_targets):
        # float64 log so extreme targets (e^±hundreds) stay expressible; the
        # per-token value itself is small and fits fp32 comfortably
        rollout_log_probs[i, :] = -torch.log(torch.tensor(float(w), dtype=torch.float64)).float() / resp_len

    adv = torch.ones(BATCH_SIZE, resp_len)
    if advantages is not None:
        adv = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1).expand(BATCH_SIZE, resp_len).clone()

    total_len = resp_len + 2
    tensors = {
        "responses": torch.ones(BATCH_SIZE, resp_len, dtype=torch.long),
        "response_mask": torch.ones(BATCH_SIZE, resp_len, dtype=torch.long),
        "input_ids": torch.ones(BATCH_SIZE, total_len, dtype=torch.long),
        "attention_mask": torch.ones(BATCH_SIZE, total_len, dtype=torch.long),
        "position_ids": torch.arange(total_len).expand(BATCH_SIZE, total_len),
        "advantages": adv,
    }
    if include_rollout_log_probs:
        tensors["rollout_log_probs"] = rollout_log_probs
    if not skip_recompute:
        tensors["old_log_probs"] = torch.zeros(BATCH_SIZE, resp_len)
    data = DataProto.from_dict(tensors=tensors)
    data.meta_info["temperature"] = 1.0
    data.meta_info["skip_recompute_old_log_prob"] = skip_recompute
    if skip_recompute:
        data.meta_info["rollout_corr_config"] = {"rollout_is": "token", "rollout_is_threshold": 2.0}
    return data


def _ess_config(**ess_overrides) -> FSDPActorConfig:
    return _make_config(ess_scaling=ESSScalingConfig(enable=True, scaling_rule="sqrt", **ess_overrides))


class TestEssMetrics:
    def test_entry_emitted_with_trainer_contract_keys(self):
        actor = _make_actor(_ess_config(base_ess_ratio=1.0))
        metrics = actor.update_policy(_make_batch([1.0, 1.0, 1.0, 1.0]))
        (entry,) = metrics["staleness/ess"]
        assert set(entry.keys()) >= {
            "minibatch_idx",
            "minibatch_ess",
            "minibatch_ess_clipped",
            "minibatch_ess_ratio",
            "minibatch_ess_ratio_clipped",
            "ess_scaled_lr",
            "base_ess_ratio",
        }
        # equal weights -> ESS = B, ratio = 1 (clipped and unclipped)
        assert entry["minibatch_ess"] == pytest.approx(BATCH_SIZE, rel=1e-4)
        assert entry["minibatch_ess_ratio"] == pytest.approx(1.0, rel=1e-4)
        assert entry["minibatch_ess_ratio_clipped"] == pytest.approx(1.0, rel=1e-4)
        assert entry["base_ess_ratio"] == 1.0

    def test_degenerate_weights_give_low_ess_ratio(self):
        # one dominant weight -> ESS ~ 1, ratio ~ 1/B
        actor = _make_actor(_ess_config(base_ess_ratio=1.0))
        metrics = actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_ess"] == pytest.approx(1.0, rel=1e-4)
        assert entry["minibatch_ess_ratio"] == pytest.approx(0.25, rel=1e-4)

    def test_no_entry_when_ess_disabled(self):
        actor = _make_actor(_make_config())
        metrics = actor.update_policy(_make_batch([1.0, 1.0, 1.0, 1.0]))
        assert "staleness/ess" not in metrics


class TestEssBrake:
    def test_brake_scales_stepped_lr_below_trigger(self):
        actor = _make_actor(_ess_config(base_ess_ratio=1.0, trigger_ratio=0.5))
        stepped = _record_stepped_lrs(actor)
        actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))  # ratio 0.25 < 0.5
        assert stepped == [pytest.approx(NOMINAL_LR * 0.25**0.5, rel=1e-4)]
        assert actor.actor_optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)

    def test_full_lr_at_or_above_trigger(self):
        actor = _make_actor(_ess_config(base_ess_ratio=0.25, trigger_ratio=0.5))
        stepped = _record_stepped_lrs(actor)
        actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))  # ratio 0.25/0.25 = 1 >= 0.5
        assert stepped == [pytest.approx(NOMINAL_LR)]

    def test_legacy_rule_without_trigger(self):
        actor = _make_actor(_ess_config(base_ess_ratio=0.5))
        stepped = _record_stepped_lrs(actor)
        actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))  # ratio 0.25/0.5 = 0.5
        assert stepped == [pytest.approx(NOMINAL_LR * 0.5**0.5, rel=1e-4)]

    def test_auto_base_uses_meta_override(self):
        actor = _make_actor(_ess_config(base_ess_ratio=None, trigger_ratio=0.5))
        stepped = _record_stepped_lrs(actor)
        batch = _make_batch([8.0, 1e-8, 1e-8, 1e-8])
        batch.meta_info["ess_base_override"] = 1.0
        metrics = actor.update_policy(batch)
        (entry,) = metrics["staleness/ess"]
        assert entry["base_ess_ratio"] == 1.0
        assert stepped == [pytest.approx(NOMINAL_LR * 0.5, rel=1e-4)]

    def test_unresolved_auto_base_is_a_noop(self):
        actor = _make_actor(_ess_config(base_ess_ratio=None))
        stepped = _record_stepped_lrs(actor)
        metrics = actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))
        (entry,) = metrics["staleness/ess"]
        assert entry["base_ess_ratio"] is None
        assert stepped == [pytest.approx(NOMINAL_LR)]


class TestSeqAdvPostScale:
    def test_zero_advantages_give_zero_gradient(self):
        actor = _make_actor(_make_config(seq_adv_post_scale=True))
        actor.update_policy(_make_batch([1.0] * BATCH_SIZE, advantages=[0.0] * BATCH_SIZE))
        # loss was identically zero -> weight unchanged from 0
        assert float(actor.actor_module.weight.detach()) == 0.0

    def test_negative_advantage_flips_the_a1_loss_sign(self):
        """Parity semantics: loss(adv=-2) == -2 * loss(adv=+1), because the
        clip branch is selected with A=1 and the advantage applied after.
        In-loss advantages would pick the other clip branch for A<0."""
        loss_fn = get_policy_loss_fn("vanilla")
        actor = _make_actor(_make_config(seq_adv_post_scale=True))

        torch.manual_seed(0)
        old_lp = torch.randn(2, RESP_LEN) * 0.3
        log_prob = old_lp + torch.randn(2, RESP_LEN) * 0.5  # ratios away from 1 -> clipping active
        mask = torch.ones(2, RESP_LEN)

        def parity_loss(adv_scalars):
            adv = torch.tensor(adv_scalars).unsqueeze(-1).expand(2, RESP_LEN).clone()
            pg_loss, _ = actor._seq_adv_post_scale_loss(
                loss_fn,
                old_log_prob=old_lp,
                log_prob=log_prob,
                advantages=adv,
                response_mask=mask,
                orig_response_mask=mask,
                loss_agg_mode="seq-mean-token-mean",
                rollout_is_weights=None,
            )
            return pg_loss

        base = parity_loss([1.0, 1.0])
        scaled = parity_loss([-2.0, -2.0])
        torch.testing.assert_close(scaled, -2.0 * base, rtol=1e-5, atol=1e-7)

        # And it genuinely differs from in-loss advantages (clip branch flips):
        adv_in_loss = torch.full((2, RESP_LEN), -2.0)
        in_loss, _ = loss_fn(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=adv_in_loss,
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=actor.config,
            rollout_is_weights=None,
        )
        assert not torch.allclose(in_loss, scaled, rtol=1e-3)

    def test_mixed_advantages_weight_sequences_independently(self):
        loss_fn = get_policy_loss_fn("vanilla")
        actor = _make_actor(_make_config(seq_adv_post_scale=True))
        torch.manual_seed(1)
        old_lp = torch.randn(2, RESP_LEN) * 0.3
        log_prob = old_lp + torch.randn(2, RESP_LEN) * 0.5
        mask = torch.ones(2, RESP_LEN)

        # per-sequence unit-advantage losses, computed independently
        unit = torch.ones(1, RESP_LEN)
        per_seq = [
            loss_fn(
                old_log_prob=old_lp[i : i + 1],
                log_prob=log_prob[i : i + 1],
                advantages=unit,
                response_mask=mask[i : i + 1],
                loss_agg_mode="seq-mean-token-mean",
                config=actor.config,
                rollout_is_weights=None,
            )[0]
            for i in range(2)
        ]
        advs = [1.5, -0.5]
        expected = (advs[0] * per_seq[0] + advs[1] * per_seq[1]) / 2

        adv = torch.tensor(advs).unsqueeze(-1).expand(2, RESP_LEN).clone()
        pg_loss, _ = actor._seq_adv_post_scale_loss(
            loss_fn,
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=adv,
            response_mask=mask,
            orig_response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            rollout_is_weights=None,
        )
        torch.testing.assert_close(pg_loss, expected, rtol=1e-6, atol=1e-8)

    def test_guard_rejects_entropy_and_kl(self):
        actor = _make_actor(_make_config(seq_adv_post_scale=True, entropy_coeff=0.01))
        with pytest.raises(NotImplementedError):
            actor.update_policy(_make_batch([1.0] * BATCH_SIZE))

    def test_end_to_end_update_runs_with_parity_and_ess(self):
        # the full script configuration: parity loss + ESS brake together
        config = _make_config(
            seq_adv_post_scale=True,
            ess_scaling=ESSScalingConfig(enable=True, scaling_rule="sqrt", base_ess_ratio=1.0, trigger_ratio=0.33333),
        )
        actor = _make_actor(config)
        metrics = actor.update_policy(_make_batch([1.0] * BATCH_SIZE, advantages=[1.0, -1.0, 0.5, 0.0]))
        assert "staleness/ess" in metrics
        assert "actor/grad_norm" in metrics


class TestMicroBatchInvariance:
    """The ESS entry and (in parity mode) the resulting gradients must be
    invariant to how the mini-batch is cut into micro-batches — the property
    that separates the CPU harness from the real DP=3 run."""

    def test_ess_entry_invariant_to_micro_batching(self):
        weights = [3.0, 0.2, 1.5, 0.7]
        entries = []
        for micro in (1, 2, 4):
            actor = _make_actor(
                _make_config(
                    ppo_micro_batch_size_per_gpu=micro,
                    ess_scaling=ESSScalingConfig(enable=True, base_ess_ratio=1.0),
                )
            )
            (entry,) = actor.update_policy(_make_batch(weights))["staleness/ess"]
            entries.append(entry)
        for entry in entries[1:]:
            for key in ("minibatch_ess", "minibatch_ess_ratio", "minibatch_ess_ratio_clipped"):
                assert entry[key] == pytest.approx(entries[0][key], rel=1e-6)

    def test_parity_gradients_invariant_to_micro_batching(self):
        advs = [1.5, -0.5, 0.0, 2.0]
        final_weights = []
        for micro in (1, 2, 4):
            actor = _make_actor(_make_config(seq_adv_post_scale=True, ppo_micro_batch_size_per_gpu=micro))
            actor.update_policy(_make_batch([1.0] * BATCH_SIZE, advantages=advs))
            final_weights.append(float(actor.actor_module.weight.detach()))
        assert final_weights[0] != 0.0  # the update actually moved the weight
        for w in final_weights[1:]:
            assert w == pytest.approx(final_weights[0], rel=1e-5)


class TestTrainerContract:
    def test_entries_survive_reduce_metrics_and_auto_base_capture(self):
        """Pin the exact consumption path of the fully-async trainer:
        worker metrics -> reduce_metrics -> _capture_ess_base-style read."""
        from verl.utils.metric import reduce_metrics

        actor = _make_actor(_ess_config(base_ess_ratio=None))
        metrics = actor.update_policy(_make_batch([1.0] * BATCH_SIZE))
        reduced = reduce_metrics(metrics)

        entries = reduced["staleness/ess"]
        assert isinstance(entries, list) and all(isinstance(e, dict) for e in entries)
        # _capture_ess_base (fully_async_trainer): mean of minibatch_ess_ratio
        key = "minibatch_ess_ratio"
        values = [float(e[key]) for e in entries if isinstance(e, dict) and e.get(key) is not None]
        assert values
        captured_base = float(sum(values) / len(values))
        assert captured_base == pytest.approx(1.0, rel=1e-4)
        # scalar metrics still reduce to floats next to the structured key
        assert isinstance(reduced["actor/grad_norm"], float)


class TestMultiMiniBatch:
    def test_entry_per_minibatch_across_epochs_and_lists_reset(self):
        """2 mini-batches x 2 epochs -> 4 entries with a running minibatch_idx.
        Ratios: mb0 = [1,1] -> 1.0; mb1 = [8,1e-8] -> 0.5. A missing per-mini-
        batch reset would blend them (combined ratio ~0.379) — pin 0.5.
        Epoch 2 repeats the values: the policy anchor shift multiplies every
        sequence's weight by the same factor, and the ESS ratio is
        scale-invariant."""
        config = _make_config(
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=1,
            ppo_epochs=2,
            ess_scaling=ESSScalingConfig(enable=True, base_ess_ratio=1.0),
        )
        actor = _make_actor(config)
        metrics = actor.update_policy(_make_batch([1.0, 1.0, 8.0, 1e-8]))
        entries = metrics["staleness/ess"]
        assert [e["minibatch_idx"] for e in entries] == [0, 1, 2, 3]
        ratios = [e["minibatch_ess_ratio"] for e in entries]
        assert ratios[0] == pytest.approx(1.0, rel=1e-3)
        assert ratios[1] == pytest.approx(0.5, rel=1e-3)
        assert ratios[2] == pytest.approx(1.0, rel=1e-3)
        assert ratios[3] == pytest.approx(0.5, rel=1e-3)


class TestBf16Ess:
    def test_long_sequence_bf16_log_probs_stay_accurate(self):
        """With bf16 forward outputs and thousands of tokens, a bf16 running
        sum drops per-token increments once the partial sum grows (bf16 eps at
        2.0 is ~0.008 vs per-token terms ~5e-4). The fp32 IS computation must
        keep the ESS ratio at the analytic value."""
        resp_len = 4096
        actor = _make_actor(_ess_config(base_ess_ratio=1.0))
        orig_forward = actor._forward_micro_batch

        def bf16_forward(model_inputs, temperature, calculate_entropy=False):
            entropy, log_prob = orig_forward(model_inputs, temperature, calculate_entropy)
            return entropy, log_prob.to(torch.bfloat16)

        actor._forward_micro_batch = bf16_forward

        batch = _make_batch([8.0, 1e-8, 1e-8, 1e-8], resp_len=resp_len)
        batch.batch["rollout_log_probs"] = batch.batch["rollout_log_probs"].to(torch.bfloat16)
        (entry,) = actor.update_policy(batch)["staleness/ess"]
        # dominant-weight batch: ESS ~ 1, ratio ~ 1/B = 0.25; only input
        # quantization (~0.4% per token value) may perturb it
        assert entry["minibatch_ess_ratio"] == pytest.approx(0.25, rel=0.05)


class TestLogspaceEssRegressions:
    """End-to-end regressions for the log-space ESS fix, reproducing what the
    2026-08 fsdp2 replay run hit through the REAL update_policy path."""

    def test_overflow_dominant_weight_no_longer_zeroes_lr(self):
        """One sequence with log-weight +60 (w ~ e^60): the raw-space pipeline
        overflowed the fp32 squared sum, read ESS ratio = 0.0, and stepped at
        lr = 0 — a silently skipped update. The true ratio is ~1/B: maximum
        brake, but a real step."""
        actor = _make_actor(_ess_config(base_ess_ratio=1.0))
        stepped = _record_stepped_lrs(actor)
        metrics = actor.update_policy(_make_batch([math.exp(60.0), 1.0, 1.0, 1.0]))
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_ess_ratio"] == pytest.approx(0.25, rel=1e-6)
        # clipped weights are [2, 1, 1, 1] -> ESS = 25/7, exact on the clipped path too
        assert entry["minibatch_ess_ratio_clipped"] == pytest.approx(25.0 / 28.0, rel=1e-6)
        # legacy rule: min(1, 0.25/1.0) = 0.25, sqrt -> half the nominal LR, not zero
        assert stepped == [pytest.approx(NOMINAL_LR * 0.25**0.5, rel=1e-4)]

    def test_deep_underflow_equal_weights_run_unbraked(self):
        """All weights e^-138 (raw-space fp32: every weight flushed to 0.0,
        ESS ratio read 0 -> lr 0). Identical weights mean ESS ratio = 1:
        the brake must not engage at all."""
        actor = _make_actor(_ess_config(base_ess_ratio=1.0))
        stepped = _record_stepped_lrs(actor)
        metrics = actor.update_policy(_make_batch([1e-60] * BATCH_SIZE))
        (entry,) = metrics["staleness/ess"]
        assert entry["minibatch_ess_ratio"] == pytest.approx(1.0, rel=1e-6)
        assert entry["minibatch_ess_ratio_clipped"] == pytest.approx(1.0, rel=1e-6)
        assert stepped == [pytest.approx(NOMINAL_LR)]


class TestGuards:
    def test_missing_rollout_log_probs_raises_for_ess(self):
        actor = _make_actor(_ess_config(base_ess_ratio=1.0))
        batch = _make_batch([1.0] * BATCH_SIZE, skip_recompute=False, include_rollout_log_probs=False)
        with pytest.raises(ValueError, match="ess_scaling"):
            actor.update_policy(batch)

    def test_multiple_param_groups_scaled_and_restored(self):
        config = _ess_config(base_ess_ratio=1.0, trigger_ratio=0.5)
        module = nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            module.weight.zero_()
            module.bias.zero_()
        optimizer = torch.optim.SGD(
            [
                {"params": [module.weight], "lr": 0.1},
                {"params": [module.bias], "lr": 0.05},
            ]
        )
        actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)

        def fake_forward(model_inputs, temperature, calculate_entropy=False):
            responses = model_inputs["responses"]
            x = torch.ones(*responses.shape, 1, dtype=torch.float32)
            return None, actor.actor_module(x).squeeze(-1)

        actor._forward_micro_batch = fake_forward

        stepped = []
        orig_step = actor.actor_optimizer.step

        def recording_step():
            stepped.append(tuple(float(pg["lr"]) for pg in actor.actor_optimizer.param_groups))
            return orig_step()

        actor.actor_optimizer.step = recording_step

        actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))  # ratio 0.25 < trigger 0.5
        assert stepped == [(pytest.approx(0.1 * 0.5), pytest.approx(0.05 * 0.5))]
        assert [float(pg["lr"]) for pg in actor.actor_optimizer.param_groups] == [
            pytest.approx(0.1),
            pytest.approx(0.05),
        ]

    def test_tensor_lr_param_group_scaled_in_place(self):
        """torchao optimizers (_AdamW) store lr as a 0-dim Tensor and RAISE on
        plain-float reassignment ("lr was changed to a non-Tensor object") —
        the brake must mutate via fill_ and keep the object identity.
        Regression: smoke run 2026-08-17, update 2 crash on the bf16-sr arm."""
        config = _ess_config(base_ess_ratio=1.0, trigger_ratio=0.5)
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        optimizer = torch.optim.SGD([{"params": [module.weight], "lr": 0.1}])
        lr_tensor = torch.tensor(0.1)
        optimizer.param_groups[0]["lr"] = lr_tensor

        class _TensorLrGuard:
            """Mimic torchao's guard: reject non-Tensor lr at step time."""

            def __init__(self, opt):
                self._opt = opt
                self.param_groups = opt.param_groups
                self.stepped_lrs = []

            def step(self):
                for pg in self.param_groups:
                    if not torch.is_tensor(pg["lr"]):
                        raise RuntimeError("lr was changed to a non-Tensor object.")
                    self.stepped_lrs.append(float(pg["lr"]))
                return self._opt.step()

            def zero_grad(self, *a, **k):
                return self._opt.zero_grad(*a, **k)

        guard = _TensorLrGuard(optimizer)
        actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=guard)

        def fake_forward(model_inputs, temperature, calculate_entropy=False):
            responses = model_inputs["responses"]
            x = torch.ones(*responses.shape, 1, dtype=torch.float32)
            return None, actor.actor_module(x).squeeze(-1)

        actor._forward_micro_batch = fake_forward

        actor.update_policy(_make_batch([8.0, 1e-8, 1e-8, 1e-8]))  # ratio 0.25 < trigger 0.5
        assert guard.stepped_lrs == [pytest.approx(0.1 * 0.5)]
        pg_lr = actor.actor_optimizer.param_groups[0]["lr"]
        assert pg_lr is lr_tensor  # identity preserved: mutated in place, never replaced
        assert float(pg_lr) == pytest.approx(0.1)  # restored after the step

    def test_parity_slices_rollout_is_weights_per_sequence(self):
        actor = _make_actor(_make_config(seq_adv_post_scale=True))
        seen_weights, seen_advantages = [], []

        def stub_loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights):
            seen_weights.append(rollout_is_weights.clone())
            seen_advantages.append(advantages.clone())
            return torch.tensor(0.0, requires_grad=True), {}

        weights = torch.arange(2 * RESP_LEN, dtype=torch.float32).reshape(2, RESP_LEN)
        adv = torch.tensor([2.0, -1.0]).unsqueeze(-1).expand(2, RESP_LEN).clone()
        mask = torch.ones(2, RESP_LEN)
        actor._seq_adv_post_scale_loss(
            stub_loss_fn,
            old_log_prob=torch.zeros(2, RESP_LEN),
            log_prob=torch.zeros(2, RESP_LEN),
            advantages=adv,
            response_mask=mask,
            orig_response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            rollout_is_weights=weights,
        )
        assert len(seen_weights) == 2
        assert torch.equal(seen_weights[0], weights[0:1])
        assert torch.equal(seen_weights[1], weights[1:2])
        # the loss itself always sees unit advantages
        assert all(torch.all(a == 1.0) for a in seen_advantages)
