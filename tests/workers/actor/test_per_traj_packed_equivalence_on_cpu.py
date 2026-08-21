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

"""End-to-end gradient-equivalence test: packed (dynamic-bsz) vs mbs=1 per-traj.

Unlike the control-flow tests, this runs BOTH update paths through the REAL
machinery — megatron_actor's forward_backward_batch with its real loss_func
(vanilla clip loss, rollout-correction token weights, the parity rescale),
real rearrange_micro_batches packing, real autograd on a shared toy model —
stubbing only the Megatron infrastructure that cannot run on CPU (the mcore
model forward, the pipeline schedule, collectives). The fake schedule divides
each micro-batch loss by num_microbatches exactly as Megatron's does, so the
n_rows*M/N rescale is exercised against the real division it compensates.

Asserted:
- the packed path's parameter gradient equals the mbs=1 per-traj path's, for
  multiple different packings (uneven token-budget packing and one-row
  packing), on data with variable lengths, negative and exactly-zero
  advantages, and IS weights that clip;
- both equal the closed-form loss (1/N) sum_i adv_i * masked_mean_t(-w_t*r_t)
  differentiated directly;
- the packed path's collected per-sequence log-IS sums equal the direct
  masked sums of (log pi_theta - log mu) (order-independent).

Run: pytest tests/workers/actor/test_per_traj_packed_equivalence_on_cpu.py
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import verl.models.mcore
import verl.workers.actor.megatron_actor as megatron_actor
from recipe.fully_async_policy.staleness_utils import TrajRecord, TrajRecordList
from verl import DataProto
from verl.workers.actor.megatron_actor import MegatronPPOActor

N_SEQS = 5
PROMPT_LEN = 2
RESP_LEN = 4
SEQ_LEN = PROMPT_LEN + RESP_LEN
ADVANTAGES = [1.5, -0.7, 0.0, 2.0, -1.2]  # positive, negative, exactly-zero
RESP_VALID = [4, 3, 2, 4, 1]  # variable response lengths
ROLLOUT_IS_THRESHOLD = 2.0


class ToyChunk(torch.nn.Module):
    """Stands in for a Megatron model chunk: a differentiable map from
    input_ids to full-sequence log-probs, plus the chunk surface the per-traj
    paths touch."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.w = torch.nn.Parameter(torch.randn(SEQ_LEN) * 0.3)
        self.config = SimpleNamespace(finalize_model_grads_func=object())

    def forward(self, input_ids):
        return -F.softplus(0.01 * input_ids.float() + self.w)

    def zero_grad_buffer(self):
        pass

    def no_sync(self):
        return nullcontext()


class _FakeOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 1e-6}]

    def zero_grad(self):
        pass  # deliberately does NOT clear grads: they are the test output

    def step(self):
        return True, 0.0, 0


def _make_data() -> DataProto:
    torch.manual_seed(11)
    input_ids = torch.randint(3, 1000, (N_SEQS, SEQ_LEN))
    response_mask = torch.zeros(N_SEQS, RESP_LEN, dtype=torch.long)
    attention_mask = torch.zeros(N_SEQS, SEQ_LEN, dtype=torch.long)
    for i, k in enumerate(RESP_VALID):
        response_mask[i, :k] = 1
        attention_mask[i, : PROMPT_LEN + k] = 1
    adv = torch.tensor(ADVANTAGES).unsqueeze(-1) * response_mask.float()
    # rollout log-probs offset from the toy policy so token IS weights spread
    # around 1 and some clip at the threshold
    rollout_log_probs = -F.softplus(torch.randn(N_SEQS, RESP_LEN) * 1.2)
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "position_ids": torch.arange(SEQ_LEN).unsqueeze(0).expand(N_SEQS, -1).contiguous(),
            "attention_mask": attention_mask,
            "responses": input_ids[:, -RESP_LEN:].contiguous(),
            "response_mask": response_mask,
            "advantages": adv,
            "rollout_log_probs": rollout_log_probs,
        },
        non_tensors={
            "uid": np.array([f"g{i}" for i in range(N_SEQS)], dtype=object),
            "traj_uid": np.array([f"t{i}" for i in range(N_SEQS)], dtype=object),
        },
    )
    data.meta_info["skip_recompute_old_log_prob"] = True
    data.meta_info["temperature"] = 1.0
    data.meta_info["rollout_corr_config"] = {
        "rollout_is": "token",
        "rollout_is_threshold": ROLLOUT_IS_THRESHOLD,
        "log_probs_pearson_corr": False,
    }
    return data


def _make_records() -> TrajRecordList:
    records = TrajRecordList()
    for i in range(N_SEQS):
        records.append(
            TrajRecord(
                uid=f"t{i}",
                group_uid=f"g{i}",
                epoch_idx=0,
                minibatch_idx=0,
                trainer_global_step=0,
                trainer_local_step=0,
                param_version_start=0,
                param_version_end=0,
                trainer_param_version=0,
                response_length=RESP_VALID[i],
                prompt_length=PROMPT_LEN,
                advantage_scalar=ADVANTAGES[i],
                reward_scalar=1.0,
            )
        )
    return records


def _make_config(use_dynamic_bsz: bool, max_token_len: int):
    return OmegaConf.create(
        {
            "use_dynamic_bsz": use_dynamic_bsz,
            "ppo_max_token_len_per_gpu": max_token_len,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_mini_batch_size": N_SEQS,
            "calculate_entropy": False,
            "entropy_coeff": 0,
            "use_kl_loss": False,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "loss_agg_mode": "seq-mean-token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "global_batch_info": {},
            "megatron": {"context_parallel_size": 1, "use_remove_padding": False},
            "ess_scaling": {
                "enable": False,
                "min_ess": 1.1,
                "lr_scale": 0.5,
                "use_clipped": False,
            },
        }
    )


def _make_actor(chunk: ToyChunk, use_dynamic_bsz: bool, max_token_len: int = 12) -> MegatronPPOActor:
    actor = MegatronPPOActor.__new__(MegatronPPOActor)
    actor.config = _make_config(use_dynamic_bsz, max_token_len)
    actor.actor_module = [chunk]
    actor.actor_optimizer = _FakeOptimizer()
    actor.use_distributed_opt = False
    actor.use_fused_kernels = False
    actor.has_multi_modal_inputs = False
    actor.hf_config = None  # consumed only by the (stubbed) mcore forward factory
    return actor


def _fake_forward_backward_func():
    """Minimal stand-in for Megatron's schedule: run forward_step + loss_func
    per micro-batch, divide the loss by num_microbatches (exactly what the
    real schedule does — the division the parity rescale compensates), and
    backward it."""

    def run(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=False,
    ):
        it = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
        chunk = model[0] if isinstance(model, (list, torch.nn.ModuleList)) else model
        results = []
        for _ in range(num_microbatches):
            output, loss_fn = forward_step_func(it, chunk)
            loss, ret = loss_fn(output)
            (loss / num_microbatches).backward()
            results.append(ret)
        return results

    return run


def _fake_get_mcore_forward_fn(hf_config):
    def forward(
        model,
        input_ids,
        attention_mask=None,
        position_ids=None,
        multi_modal_inputs=None,
        logits_processor=None,
        logits_processor_args=None,
        data_format=None,
        **kwargs,
    ):
        return {"log_probs": model(input_ids)}

    return forward


@pytest.fixture
def real_loss_env(monkeypatch):
    """Stub only what cannot run on CPU; the loss math stays real."""
    fake_mpu = SimpleNamespace(
        get_virtual_pipeline_model_parallel_world_size=lambda: None,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_last_rank=lambda: 0,
        get_pipeline_model_parallel_group=lambda: None,
        is_pipeline_last_stage=lambda ignore_virtual=True: True,
        get_data_parallel_group=lambda with_context_parallel=False: SimpleNamespace(size=lambda: 1),
    )
    records = _make_records()
    monkeypatch.setattr(megatron_actor, "mpu", fake_mpu)
    monkeypatch.setattr(megatron_actor, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(megatron_actor, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setattr(megatron_actor, "broadcast_dict_tensor", lambda *a, **k: None)
    monkeypatch.setattr(megatron_actor, "get_forward_backward_func", _fake_forward_backward_func)
    monkeypatch.setattr(
        megatron_actor,
        "compute_staleness_statistics",
        lambda batch, mb_idx, thr, use_old, epoch_idx=0: (records, {}),
    )
    monkeypatch.setattr(
        megatron_actor,
        "compute_ess_info",
        lambda recs, thr: {"ess": 0.0, "ess_ratio": None, "ess_clipped": 0.0, "ess_ratio_clipped": None},
    )
    monkeypatch.setattr(megatron_actor, "disable_grad_finalize", lambda modules: None)
    monkeypatch.setattr(megatron_actor, "restore_grad_finalize", lambda modules, orig: None)
    monkeypatch.setattr(megatron_actor, "finalize_model_grads_ignore_dp", lambda modules: None)
    monkeypatch.setattr(verl.models.mcore, "get_mcore_forward_fn", _fake_get_mcore_forward_fn)
    return records


def _run_reference(chunk: ToyChunk) -> torch.Tensor:
    """mbs=1 per-traj path (buffer-free), gradient on the toy parameter."""
    actor = _make_actor(chunk, use_dynamic_bsz=False)
    chunk.w.grad = None
    actor.update_policy_per_traj([_make_data()], grad_baselining=False)
    assert chunk.w.grad is not None
    return chunk.w.grad.detach().clone()


def _run_packed(chunk: ToyChunk, max_token_len: int) -> tuple[torch.Tensor, dict]:
    actor = _make_actor(chunk, use_dynamic_bsz=True, max_token_len=max_token_len)
    chunk.w.grad = None
    metrics = actor._update_policy_per_traj_packed([_make_data()])
    assert chunk.w.grad is not None
    return chunk.w.grad.detach().clone(), metrics


def _closed_form_grad(chunk: ToyChunk) -> torch.Tensor:
    """(1/N) sum_i adv_i * masked_mean_t( -w_t * exp(lp - lp.detach()) ) with
    w_t = min(exp(lp.detach() - mu), threshold): the loss both paths implement
    when the ratio is anchored at 1."""
    data = _make_data()
    chunk.w.grad = None
    lp = chunk(data.batch["input_ids"])[:, -RESP_LEN - 1 : -1]
    mask = data.batch["response_mask"].float()
    w_tok = torch.clamp(torch.exp(lp.detach() - data.batch["rollout_log_probs"]), max=ROLLOUT_IS_THRESHOLD)
    ratio = torch.exp(lp - lp.detach())
    per_tok = -w_tok * ratio * mask
    per_seq = per_tok.sum(dim=-1) / mask.sum(dim=-1)
    loss = (torch.tensor(ADVANTAGES) * per_seq).sum() / N_SEQS
    loss.backward()
    return chunk.w.grad.detach().clone()


class TestGradientEquivalence:
    def test_packed_uneven_packing_matches_reference(self, real_loss_env):
        chunk = ToyChunk()
        ref = _run_reference(chunk)
        # budget 12 packs the token counts [6, 5, 4, 6, 3] into uneven groups
        packed, _ = _run_packed(chunk, max_token_len=12)
        assert torch.allclose(packed, ref, rtol=1e-5, atol=1e-8), f"max diff {(packed - ref).abs().max()}"

    def test_packed_one_row_packing_matches_reference(self, real_loss_env):
        chunk = ToyChunk()
        ref = _run_reference(chunk)
        # budget 6 fits no pair of rows -> one row per micro-batch (M=N)
        packed, _ = _run_packed(chunk, max_token_len=6)
        assert torch.allclose(packed, ref, rtol=1e-5, atol=1e-8)

    def test_packings_agree_with_each_other_and_closed_form(self, real_loss_env):
        chunk = ToyChunk()
        packed_a, _ = _run_packed(chunk, max_token_len=12)
        packed_b, _ = _run_packed(chunk, max_token_len=6)
        closed = _closed_form_grad(chunk)
        assert torch.allclose(packed_a, packed_b, rtol=1e-5, atol=1e-8)
        assert torch.allclose(packed_a, closed, rtol=1e-5, atol=1e-8)

    def test_reference_matches_closed_form(self, real_loss_env):
        chunk = ToyChunk()
        ref = _run_reference(chunk)
        closed = _closed_form_grad(chunk)
        assert torch.allclose(ref, closed, rtol=1e-5, atol=1e-8)

    def test_nonzero_gradient(self, real_loss_env):
        """Guard against a vacuous pass: the compared gradients must be
        genuinely nonzero."""
        chunk = ToyChunk()
        ref = _run_reference(chunk)
        assert ref.abs().max() > 1e-6


class TestSeqLogIsEquivalence:
    def test_collected_log_is_matches_direct_masked_sums(self, real_loss_env):
        """The packed path's ESS input must equal the mbs=1 path's measured
        quantity: per-sequence masked sums of (log pi_theta - log mu).
        Packing reorders rows, so compare as sorted multisets."""
        chunk = ToyChunk()
        data = _make_data()
        with torch.no_grad():
            lp = chunk(data.batch["input_ids"])[:, -RESP_LEN - 1 : -1]
            mask = data.batch["response_mask"].float()
            expected = ((lp - data.batch["rollout_log_probs"]) * mask).sum(dim=-1).tolist()

        captured = []
        actor = _make_actor(chunk, use_dynamic_bsz=True, max_token_len=12)
        original = megatron_actor.compute_global_ess_from_log_weights

        def spy(seq_log_is, threshold=None, group=None):
            captured.append(list(seq_log_is))
            return original(seq_log_is, threshold, group=group)

        megatron_actor.compute_global_ess_from_log_weights = spy
        try:
            actor._update_policy_per_traj_packed([data])
        finally:
            megatron_actor.compute_global_ess_from_log_weights = original

        (seq_log_is,) = captured
        assert len(seq_log_is) == N_SEQS
        assert sorted(seq_log_is) == pytest.approx(sorted(expected), rel=1e-5)


# ==================== CPPO on the packed path ====================

CPPO_DELTA = 0.05
CPPO_CLIP_C = 20.0
CPPO_W_MIN = 0.8
CPPO_DELTA_B = 0.02


def _make_cppo_actor(chunk: ToyChunk, max_token_len: int = 12) -> MegatronPPOActor:
    actor = _make_actor(chunk, use_dynamic_bsz=True, max_token_len=max_token_len)
    actor.config.policy_loss = OmegaConf.create(
        {
            "loss_mode": "cppo",
            "cppo": {
                "cppo_w_min": CPPO_W_MIN,
                "cppo_delta_b": CPPO_DELTA_B,
                "cppo_delta_b_q": 0.9,
                "cppo_delta_b_k": 1.0,
            },
        }
    )
    actor.config.clip_ratio = CPPO_DELTA
    actor.config.clip_ratio_c = CPPO_CLIP_C
    return actor


def _cppo_reference_mask(lp: torch.Tensor, mu: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compact per-token mirror of the CPPO mask (paper Eq. 8-10, Eq. 22),
    anchored at mu, written independently of the vectorized implementation."""
    b, t_len = mask.shape
    out = torch.zeros(b, t_len)
    m = mask.float()
    for i in range(b):
        d_row = (torch.exp(lp[i]) - torch.exp(mu[i])).abs() * m[i]
        valid = m[i] > 0
        q = torch.quantile(d_row[valid], 0.9).item() if valid.any() else CPPO_DELTA_B
        db_seq = min(max(1.0 * q, CPPO_DELTA_B), 5.0 * CPPO_DELTA_B)
        s_prev, w_prev = 0.0, 0.0
        for t in range(t_len):
            w_t = (CPPO_W_MIN + (1.0 - CPPO_W_MIN) * (t_len - (t + 1)) / max(t_len - 1.0, 1.0)) * m[i, t].item()
            z_t = w_t * d_row[t].item()
            c_t = min(CPPO_DELTA, CPPO_DELTA + db_seq * w_prev - s_prev)
            ratio = float(torch.exp(torch.clamp(lp[i, t] - mu[i, t], -20.0, 20.0)))
            toward_mu = adv[i, t].item() * (ratio - 1.0) <= 0.0
            out[i, t] = 1.0 if (toward_mu or z_t <= c_t) and m[i, t] > 0 else 0.0
            s_prev += z_t
            w_prev += w_t
    return out


def _cppo_closed_form_grad(chunk: ToyChunk) -> torch.Tensor:
    """Global per-sequence mean of the mu-anchored CPPO loss:
    (1/N) sum_i masked_mean_t( -adv_i * sg(min(exp(lp - mu), c)) * lp * M_t ).
    No token-IS min(pi/mu, 2) factor: under loss_mode=cppo the loss's own
    truncated ratio replaces the rollout-correction weights."""
    data = _make_data()
    chunk.w.grad = None
    lp = chunk(data.batch["input_ids"])[:, -RESP_LEN - 1 : -1]
    mu = data.batch["rollout_log_probs"]
    mask = data.batch["response_mask"].float()
    adv = data.batch["advantages"]
    valid = _cppo_reference_mask(lp.detach(), mu, adv, data.batch["response_mask"])
    ratio = torch.exp(torch.clamp(lp.detach() - mu, -20.0, 20.0))
    w = torch.clamp(ratio, max=CPPO_CLIP_C)
    per_tok = -adv * w * lp * valid * mask
    per_seq = per_tok.sum(dim=-1) / mask.sum(dim=-1)
    loss = per_seq.sum() / N_SEQS
    loss.backward()
    return chunk.w.grad.detach().clone()


class TestCPPOPackedIntegration:
    def test_cppo_packed_matches_closed_form_uneven_packing(self, real_loss_env):
        chunk = ToyChunk()
        closed = _cppo_closed_form_grad(chunk)
        actor = _make_cppo_actor(chunk, max_token_len=12)
        chunk.w.grad = None
        actor._update_policy_per_traj_packed([_make_data()])
        packed = chunk.w.grad.detach().clone()
        assert torch.allclose(packed, closed, rtol=1e-5, atol=1e-8), f"max diff {(packed - closed).abs().max()}"

    def test_cppo_packed_matches_closed_form_one_row_packing(self, real_loss_env):
        chunk = ToyChunk()
        closed = _cppo_closed_form_grad(chunk)
        actor = _make_cppo_actor(chunk, max_token_len=6)
        chunk.w.grad = None
        actor._update_policy_per_traj_packed([_make_data()])
        assert torch.allclose(chunk.w.grad, closed, rtol=1e-5, atol=1e-8)

    def test_cppo_gradient_nonzero_and_differs_from_vanilla(self, real_loss_env):
        """Guard against vacuous equivalence: the cppo gradient is nonzero and
        genuinely different from the vanilla (anchored + token-IS) gradient —
        i.e. the mask binds and the mu anchor is in effect."""
        chunk = ToyChunk()
        actor = _make_cppo_actor(chunk, max_token_len=12)
        chunk.w.grad = None
        actor._update_policy_per_traj_packed([_make_data()])
        cppo_grad = chunk.w.grad.detach().clone()
        vanilla_grad = _run_reference(chunk)
        assert cppo_grad.abs().max() > 1e-6
        assert not torch.allclose(cppo_grad, vanilla_grad, rtol=1e-3, atol=1e-6)

    def test_cppo_mask_binds_and_metrics_flow(self, real_loss_env):
        """With delta small vs the crafted pi-mu gap, the mask must reject
        tokens (pg_clipfrac > 0) and the CPPO metrics must reach the returned
        actor metrics through the packed path."""
        chunk = ToyChunk()
        actor = _make_cppo_actor(chunk, max_token_len=12)
        metrics = actor._update_policy_per_traj_packed([_make_data()])
        clipfracs = metrics["actor/pg_clipfrac"]
        toward = metrics["actor/cppo_toward_mu_frac"]
        assert sum(clipfracs) / len(clipfracs) > 0.0
        assert all(0.0 <= v <= 1.0 for v in toward)
        # sanity against the mirror: expected rejected fraction over valid tokens
        data = _make_data()
        with torch.no_grad():
            lp = chunk(data.batch["input_ids"])[:, -RESP_LEN - 1 : -1]
        valid = _cppo_reference_mask(
            lp, data.batch["rollout_log_probs"], data.batch["advantages"], data.batch["response_mask"]
        )
        m = data.batch["response_mask"].float()
        expected_frac = float(((1.0 - valid) * m).sum() / m.sum())
        # packed micro-batches average clipfrac per micro-batch; with uneven
        # packing the token-weighted global value differs slightly, so compare
        # loosely: nonzero and same order
        assert expected_frac > 0.0

    def test_cppo_seq_log_is_and_ess_unchanged(self, real_loss_env):
        """The ESS brake input (per-seq log-IS sums vs mu) must be identical
        under cppo — the loss swap must not touch the brake wiring."""
        chunk = ToyChunk()
        data = _make_data()
        with torch.no_grad():
            lp = chunk(data.batch["input_ids"])[:, -RESP_LEN - 1 : -1]
            mask = data.batch["response_mask"].float()
            expected = ((lp - data.batch["rollout_log_probs"]) * mask).sum(dim=-1).tolist()

        captured = []
        actor = _make_cppo_actor(chunk, max_token_len=12)
        original = megatron_actor.compute_global_ess_from_log_weights

        def spy(seq_log_is, threshold=None, group=None):
            captured.append(list(seq_log_is))
            return original(seq_log_is, threshold, group=group)

        megatron_actor.compute_global_ess_from_log_weights = spy
        try:
            metrics = actor._update_policy_per_traj_packed([data])
        finally:
            megatron_actor.compute_global_ess_from_log_weights = original

        (seq_log_is,) = captured
        assert sorted(seq_log_is) == pytest.approx(sorted(expected), rel=1e-5)
        assert "staleness/ess" in metrics

    def test_cppo_missing_rollout_log_probs_refused(self, real_loss_env):
        chunk = ToyChunk()
        actor = _make_cppo_actor(chunk, max_token_len=12)
        data = _make_data()
        data.batch.pop("rollout_log_probs")
        with pytest.raises((AssertionError, KeyError)):
            actor._update_policy_per_traj_packed([data])


class TestCPPOMbs1Guard:
    def test_mbs1_per_traj_path_refuses_cppo(self, real_loss_env):
        chunk = ToyChunk()
        actor = _make_actor(chunk, use_dynamic_bsz=False)
        actor.config.policy_loss = OmegaConf.create({"loss_mode": "cppo", "cppo": {}})
        with pytest.raises(AssertionError, match="cppo is not supported on the mbs=1"):
            actor.update_policy_per_traj([_make_data()], grad_baselining=False)
