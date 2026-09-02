# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""Host-resident OPOB accumulators (grad_baselining.accum_device=cpu) on CPU.

The 5+3 / fp32-master OPOB arm cannot hold the two grad-sized OPOB accumulators on the
GPU, so they live in pinned host memory: each trajectory's gradient is staged d2h once and
added into both accumulators on the CPU (optionally in fp32), the -b* move at the group
close happens on the host, and the result is copied back into Megatron's grad buffer at
the step with chunk-wise dtype conversion. These tests pin the buffer helpers, the actor's
allocation / update wiring, and the prepare_grads() skip in _compute_grad_norms.

Run: pytest recipe/fully_async_policy/unittest/test_opob_cpu_accum_on_cpu.py
"""

from types import SimpleNamespace

import pytest
import torch

import verl.workers.actor.megatron_actor as megatron_actor
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.utils import vcpo


def _module(*grad_tensors):
    """A Megatron-DDP-like module exposing ``buffers[i].grad_data``."""
    return SimpleNamespace(buffers=[SimpleNamespace(grad_data=g) for g in grad_tensors])


# ------------------------------------------------------------------ dtype resolution


class TestResolveAccumDtype:
    def test_auto_and_none_follow_the_grad_buffer(self):
        assert vcpo._resolve_accum_dtype(torch.bfloat16, "auto") is torch.bfloat16
        assert vcpo._resolve_accum_dtype(torch.bfloat16, None) is torch.bfloat16
        assert vcpo._resolve_accum_dtype(torch.float16, "") is torch.float16

    def test_named_dtypes_and_aliases(self):
        assert vcpo._resolve_accum_dtype(torch.bfloat16, "float32") is torch.float32
        assert vcpo._resolve_accum_dtype(torch.bfloat16, "fp32") is torch.float32
        assert vcpo._resolve_accum_dtype(torch.bfloat16, "torch.float32") is torch.float32
        assert vcpo._resolve_accum_dtype(torch.float32, "bf16") is torch.bfloat16

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            vcpo._resolve_accum_dtype(torch.bfloat16, "float99")


# ------------------------------------------------------------------ allocation


class TestAllocation:
    def test_default_allocation_matches_grad_buffers(self):
        m = _module(torch.ones(6, dtype=torch.bfloat16), torch.ones(3, dtype=torch.bfloat16))
        bufs = vcpo.allocate_grad_accum_buffers([m])
        assert [b.shape for b in bufs] == [(6,), (3,)]
        assert all(b.dtype is torch.bfloat16 and float(b.abs().sum()) == 0 for b in bufs)

    def test_cpu_fp32_allocation(self):
        m = _module(torch.ones(6, dtype=torch.bfloat16))
        bufs = vcpo.allocate_grad_accum_buffers([m], device="cpu", dtype="float32")
        assert bufs[0].device.type == "cpu" and bufs[0].dtype is torch.float32
        assert bufs[0].shape == (6,) and float(bufs[0].abs().sum()) == 0

    def test_other_dtype_on_grad_device(self):
        m = _module(torch.ones(4, dtype=torch.bfloat16))
        bufs = vcpo.allocate_grad_accum_buffers([m], device="cuda", dtype="float32")
        # grad buffers are CPU tensors in this test, so "cuda" resolves to the grad device
        assert bufs[0].dtype is torch.float32 and bufs[0].device == m.buffers[0].grad_data.device

    def test_staging_buffers_mirror_shape_and_dtype(self):
        m = _module(torch.ones(5, dtype=torch.bfloat16), torch.ones(2, dtype=torch.float32))
        st = vcpo.allocate_staging_buffers([m])
        assert [(s.shape, s.dtype, s.device.type) for s in st] == [
            ((5,), torch.bfloat16, "cpu"),
            ((2,), torch.float32, "cpu"),
        ]

    def test_stage_copies_grad_buffers(self):
        g = torch.arange(4, dtype=torch.float32)
        m = _module(g)
        st = vcpo.allocate_staging_buffers([m])
        vcpo.stage_grad_buffers([m], st)
        assert torch.equal(st[0], g)
        with pytest.raises(AssertionError):
            vcpo.stage_grad_buffers([m], [])


# ------------------------------------------------------------------ accumulation


class TestAccumulateMulti:
    def test_cpu_targets_share_one_source_and_convert_dtype(self):
        g = torch.tensor([1.0, -2.0, 0.5, 4.0], dtype=torch.bfloat16)
        m = _module(g)
        acc = torch.zeros(4, dtype=torch.float32)
        score = torch.zeros(4, dtype=torch.float32)
        vcpo.accumulate_grad_buffers_multi([m], [([acc], -1.5), ([score], 2.0)])
        assert torch.equal(acc, g.float() * -1.5)
        assert torch.equal(score, g.float() * 2.0)
        # second trajectory accumulates on top
        m.buffers[0].grad_data = torch.ones(4, dtype=torch.bfloat16)
        vcpo.accumulate_grad_buffers_multi([m], [([acc], 1.0), ([score], 1.0)])
        assert torch.equal(acc, g.float() * -1.5 + 1.0)
        assert torch.equal(score, g.float() * 2.0 + 1.0)

    def test_cpu_targets_use_staging_when_grads_live_on_cuda(self, monkeypatch):
        """The d2h path: grads flagged as cuda are staged once, then both targets read the staging copy."""
        g = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        m = _module(g)
        staged = []

        def fake_stage(modules, staging):
            staged.append(len(staging))
            staging[0].copy_(g)

        monkeypatch.setattr(vcpo, "stage_grad_buffers", fake_stage)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: self is g))  # only the grad "is cuda"
        acc = torch.zeros(2, dtype=torch.float32)
        score = torch.zeros(2, dtype=torch.float32)
        staging = [torch.empty(2, dtype=torch.bfloat16)]
        vcpo.accumulate_grad_buffers_multi([m], [([acc], 0.5), ([score], 1.0)], staging=staging)
        assert staged == [1]  # one d2h transfer for both targets
        assert torch.equal(acc, torch.tensor([0.5, 1.0]))
        assert torch.equal(score, torch.tensor([1.0, 2.0]))

    def test_cpu_targets_without_staging_on_cuda_grads_raise(self, monkeypatch):
        g = torch.ones(2, dtype=torch.bfloat16)
        m = _module(g)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: self is g))
        with pytest.raises(AssertionError):
            vcpo.accumulate_grad_buffers_multi([m], [([torch.zeros(2)], 1.0)], staging=None)

    def test_gpu_targets_delegate_to_accumulate_grad_buffers(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vcpo, "accumulate_grad_buffers", lambda modules, bufs, scale: calls.append((bufs, scale)))
        m = _module(torch.ones(2))
        gpu_bufs = [SimpleNamespace(is_cuda=True)]
        vcpo.accumulate_grad_buffers_multi([m], [(gpu_bufs, 3.0)])
        assert calls == [(gpu_bufs, 3.0)]

    def test_mismatched_lengths_raise(self):
        m = _module(torch.ones(2, dtype=torch.bfloat16))
        with pytest.raises(AssertionError):
            vcpo.accumulate_grad_buffers_multi([m], [([torch.zeros(2), torch.zeros(2)], 1.0)])


class TestAddIntoMulti:
    def test_shared_conversion_matches_reference(self, monkeypatch):
        monkeypatch.setattr(vcpo, "_COPY_CHUNK", 3)  # several chunks, last one partial
        monkeypatch.setattr(vcpo, "_SCRATCH", {})
        src = torch.tensor([1.0, -2.0, 0.5, 4.0, 3.0, -1.0, 2.0], dtype=torch.bfloat16)
        a = torch.arange(7, dtype=torch.float32)
        b = torch.ones(7, dtype=torch.float32)
        same = torch.zeros(7, dtype=torch.bfloat16)  # same dtype as src: plain add_
        vcpo._add_into_multi_([(a, 2.0), (b, -1.0), (same, 1.0)], src)
        assert torch.equal(a, torch.arange(7, dtype=torch.float32) + 2.0 * src.float())
        assert torch.equal(b, 1.0 - src.float())
        assert torch.equal(same, src)
        # one scratch of chunk size was allocated and is reused
        assert len(vcpo._SCRATCH) == 1 and next(iter(vcpo._SCRATCH.values())).numel() == 3
        vcpo._add_into_multi_([(a, 1.0)], src)
        assert len(vcpo._SCRATCH) == 1

    def test_scratch_grows_when_a_larger_chunk_is_needed(self, monkeypatch):
        monkeypatch.setattr(vcpo, "_SCRATCH", {})
        s = vcpo._conversion_scratch(torch.float32, torch.device("cpu"), 4)
        assert s.numel() == 4
        s2 = vcpo._conversion_scratch(torch.float32, torch.device("cpu"), 8)
        assert s2.numel() == 8
        s3 = vcpo._conversion_scratch(torch.float32, torch.device("cpu"), 2)
        assert s3.numel() == 2 and len(vcpo._SCRATCH) == 1

    def test_multi_targets_share_one_source_pass(self, monkeypatch):
        """accumulate_grad_buffers_multi converts each grad chunk once for all CPU targets."""
        monkeypatch.setattr(vcpo, "_COPY_CHUNK", 2)
        monkeypatch.setattr(vcpo, "_SCRATCH", {})
        copies = []
        real_copy = torch.Tensor.copy_

        def counting_copy(self, other, *a, **k):
            copies.append(self.numel())
            return real_copy(self, other, *a, **k)

        monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)
        g = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        m = _module(g)
        acc, score = torch.zeros(3), torch.zeros(3)
        vcpo.accumulate_grad_buffers_multi([m], [([acc], 1.0), ([score], 0.5)])
        assert torch.equal(acc, g.float()) and torch.equal(score, 0.5 * g.float())
        assert copies == [2, 1]  # two chunks converted once each, not once per target


class TestMoveAndCopyBack:
    def test_move_on_host_with_dtype_conversion(self):
        src = [torch.tensor([1.0, 2.0], dtype=torch.bfloat16)]
        dest = [torch.tensor([10.0, 20.0], dtype=torch.float32)]
        vcpo.move_grad_buffers(src, dest, scale=-0.5)
        assert torch.equal(dest[0], torch.tensor([9.5, 19.0]))

    def test_move_on_host_same_dtype(self):
        src = [torch.ones(3)]
        dest = [torch.zeros(3)]
        vcpo.move_grad_buffers(src, dest, scale=2.0)
        assert torch.equal(dest[0], torch.full((3,), 2.0))

    def test_copy_back_converts_chunkwise(self, monkeypatch):
        monkeypatch.setattr(vcpo, "_COPY_CHUNK", 3)  # force several chunks
        grad = torch.zeros(8, dtype=torch.bfloat16)
        m = _module(grad)
        acc = torch.arange(8, dtype=torch.float32) * 0.25
        vcpo.copy_accum_buffers_to_grad_buffers([m], [acc])
        assert torch.equal(grad, acc.to(torch.bfloat16))

    def test_copy_back_same_dtype_is_plain_copy(self):
        grad = torch.zeros(4)
        m = _module(grad)
        vcpo.copy_accum_buffers_to_grad_buffers([m], [torch.arange(4, dtype=torch.float32)])
        assert torch.equal(grad, torch.arange(4, dtype=torch.float32))

    def test_zero_works_on_host_accumulators(self):
        bufs = [torch.ones(3, dtype=torch.float32)]
        vcpo.zero_grad_accum_buffers(bufs)
        assert float(bufs[0].abs().sum()) == 0


# ------------------------------------------------------------------ grad-scaler probe


class TestOptimizerHasGradScaler:
    def test_plain_and_chained(self):
        assert not vcpo._optimizer_has_grad_scaler(None)
        assert not vcpo._optimizer_has_grad_scaler(SimpleNamespace())
        assert not vcpo._optimizer_has_grad_scaler(SimpleNamespace(grad_scaler=None))
        assert vcpo._optimizer_has_grad_scaler(SimpleNamespace(grad_scaler=object()))
        chained = SimpleNamespace(
            chained_optimizers=[SimpleNamespace(grad_scaler=None), SimpleNamespace(grad_scaler=1)]
        )
        assert vcpo._optimizer_has_grad_scaler(chained)
        assert not vcpo._optimizer_has_grad_scaler(SimpleNamespace(chained_optimizers=[SimpleNamespace()]))


# ------------------------------------------------------------------ actor wiring


def _actor(gb_cfg):
    actor = MegatronPPOActor.__new__(MegatronPPOActor)
    actor.config = SimpleNamespace(grad_baselining=gb_cfg, loss_agg_mode="seq-mean-token-mean")
    actor.actor_module = [_module(torch.zeros(2, dtype=torch.bfloat16))]
    return actor


class TestActorAllocation:
    def test_default_path_allocates_on_device_without_kwargs(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            megatron_actor, "allocate_grad_accum_buffers", lambda modules, **kw: calls.append(kw) or ["b"]
        )
        monkeypatch.setattr(megatron_actor, "allocate_staging_buffers", lambda modules: pytest.fail("no staging"))
        actor = _actor(SimpleNamespace(scope="group"))  # no accum_* attrs at all -> defaults
        accum, score, staging = actor._allocate_opob_buffers()
        assert calls == [{}, {}] and accum == ["b"] and score == ["b"] and staging is None

    def test_cpu_path_allocates_host_buffers_staging_and_threads(self, monkeypatch):
        calls, threads = [], []
        monkeypatch.setattr(
            megatron_actor, "allocate_grad_accum_buffers", lambda modules, **kw: calls.append(kw) or ["b"]
        )
        monkeypatch.setattr(megatron_actor, "allocate_staging_buffers", lambda modules: ["stage"])
        monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
        monkeypatch.setattr(torch, "set_num_threads", lambda n: threads.append(n))
        actor = _actor(SimpleNamespace(scope="group", accum_device="cpu", accum_dtype="float32", accum_cpu_threads=8))
        accum, score, staging = actor._allocate_opob_buffers()
        assert calls == [{"device": "cpu", "dtype": "float32"}] * 2
        assert staging == ["stage"] and threads == [8]

    def test_cpu_path_pins_the_thread_pool_exactly(self, monkeypatch):
        """torch defaults to one thread per core (112 on the H100 node): several ranks would
        oversubscribe the CPU quota, so the configured count is applied even when lower."""
        threads = []
        monkeypatch.setattr(megatron_actor, "allocate_grad_accum_buffers", lambda modules, **kw: ["b"])
        monkeypatch.setattr(megatron_actor, "allocate_staging_buffers", lambda modules: ["stage"])
        monkeypatch.setattr(torch, "get_num_threads", lambda: 112)
        monkeypatch.setattr(torch, "set_num_threads", lambda n: threads.append(n))
        actor = _actor(SimpleNamespace(scope="group", accum_device="cpu", accum_dtype="auto", accum_cpu_threads=8))
        actor._allocate_opob_buffers()
        assert threads == [8]
        # already at the configured count -> no call
        monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
        actor._allocate_opob_buffers()
        assert threads == [8]

    def test_cuda_with_explicit_dtype_passes_kwargs(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            megatron_actor, "allocate_grad_accum_buffers", lambda modules, **kw: calls.append(kw) or ["b"]
        )
        actor = _actor(SimpleNamespace(scope="group", accum_device="cuda", accum_dtype="float32"))
        _, _, staging = actor._allocate_opob_buffers()
        assert calls == [{"device": "cuda", "dtype": "float32"}] * 2 and staging is None


class TestActorUpdateGradBuffers:
    @staticmethod
    def _run(monkeypatch, staging):
        actor = _actor(SimpleNamespace(scope="group", norm_by_std=True))
        single, multi = [], []
        monkeypatch.setattr(
            megatron_actor, "accumulate_grad_buffers", lambda modules, bufs, scale: single.append((bufs, scale))
        )
        monkeypatch.setattr(
            megatron_actor,
            "accumulate_grad_buffers_multi",
            lambda modules, targets, staging=None: multi.append((targets, staging)),
        )
        accum, score = ["accum"], ["score"]
        actor._update_grad_buffers(
            accum_buffers=accum,
            score_gradient_buffers=score,
            local_traj_records=[],
            reward_scalar=1.0,
            reward_std=0.5,
            adv_scalar=2.0,
            group_uid="g0",
            microbatch_loss_scale=1.0,
            norm_by_std=True,
            is_last_traj_in_scope=False,
            grad_baselining=True,
            staging=staging,
        )
        return single, multi, accum, score

    def test_with_staging_uses_the_shared_d2h_path(self, monkeypatch):
        single, multi, accum, score = self._run(monkeypatch, staging=["stage"])
        assert single == []
        assert multi == [([(accum, 2.0), (score, 2.0)], ["stage"])]  # R/std and 1/std with R=1, std=0.5

    def test_without_staging_keeps_the_two_device_adds(self, monkeypatch):
        single, multi, accum, score = self._run(monkeypatch, staging=None)
        assert multi == []
        assert single == [(accum, 2.0), (score, 2.0)]


class TestComputeGradNormsSkipsPrepareGrads:
    @staticmethod
    def _actor_with_optimizer(monkeypatch, optimizer):
        actor = MegatronPPOActor.__new__(MegatronPPOActor)
        actor.actor_optimizer = optimizer
        actor.actor_module = [object()]
        monkeypatch.setattr(megatron_actor, "_get_local_model_grads_for_norm", lambda modules: ["g"])
        monkeypatch.setattr(megatron_actor, "get_grad_norm_fp32", lambda grads, grad_stats_parallel_group=None: 2.0)
        monkeypatch.setattr(megatron_actor.mpu, "get_tensor_model_parallel_group", lambda: None)
        return actor

    def test_no_grad_scaler_skips_prepare_grads(self, monkeypatch):
        calls = []
        opt = SimpleNamespace(prepare_grads=lambda: calls.append("prepare") or False)
        actor = self._actor_with_optimizer(monkeypatch, opt)
        unscaled, scaled = actor._compute_grad_norms(4, "seq-mean-token-mean", -0.5, microbatch_loss_scale=0.5)
        assert calls == []
        assert unscaled == pytest.approx(2.0 * 4 / 0.5) and scaled == pytest.approx(unscaled * 0.5)

    def test_grad_scaler_keeps_the_inf_check(self, monkeypatch):
        calls = []
        opt = SimpleNamespace(grad_scaler=object(), prepare_grads=lambda: calls.append("prepare") or True)
        actor = self._actor_with_optimizer(monkeypatch, opt)
        unscaled, scaled = actor._compute_grad_norms(4, "seq-mean-token-mean", 1.0)
        assert calls == ["prepare"]
        assert unscaled == float("inf") and scaled == float("inf")
