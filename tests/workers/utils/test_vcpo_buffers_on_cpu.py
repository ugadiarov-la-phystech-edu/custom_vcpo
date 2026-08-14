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

"""CPU tests for the VCPO gradient-buffer helpers (verl/workers/utils/vcpo.py).

These helpers are the numerical core of the OPOB per-trajectory path: they
accumulate advantage-scaled score gradients (``accum += adv_i * g_i``), apply
the closed-form baseline (``G_R - b* * G_S`` via ``move_grad_buffers`` with a
negative scale), and hand the result back to Megatron's main grad buffer.
The fakes below mimic Megatron DDP chunks: objects exposing ``.buffers`` with
``.grad_data`` tensors, which is all ``_iter_grad_buffers`` reads.

The equivalence test at the bottom pins the mathematical claim behind the
buffer-free (no-OPOB) path in ``update_policy_per_traj``: backward is linear
in the loss scale, so folding ``adv_i`` into the per-microbatch loss
multiplier yields exactly the gradient the buffer scheme accumulated.

Run: pytest tests/workers/utils/test_vcpo_buffers_on_cpu.py
"""

import torch
import torch.nn as nn

from verl.workers.utils.vcpo import (
    accumulate_grad_buffers,
    allocate_grad_accum_buffers,
    apply_scaled_grad_delta,
    copy_accum_buffers_to_grad_buffers,
    move_grad_buffers,
    snapshot_grad_buffers,
    zero_grad_accum_buffers,
)


class _FakeBuffer:
    def __init__(self, grad_data: torch.Tensor):
        self.grad_data = grad_data


class _FakeChunk:
    """Megatron-DDP-chunk stand-in: `.buffers` entries expose `.grad_data`."""

    def __init__(self, tensors: list[torch.Tensor]):
        self.buffers = [_FakeBuffer(t) for t in tensors]


def _make_chunk(*shapes_dtypes) -> _FakeChunk:
    torch.manual_seed(0)
    tensors = [torch.randn(*shape).to(dtype) for shape, dtype in shapes_dtypes]
    return _FakeChunk(tensors)


class TestBufferHelpers:
    def test_allocate_returns_independent_zeros_matching_shape_and_dtype(self):
        chunk = _make_chunk(((4, 3), torch.float32), ((5,), torch.bfloat16))
        accum = allocate_grad_accum_buffers([chunk])

        assert len(accum) == 2
        for buf, fake in zip(accum, chunk.buffers, strict=True):
            assert buf.shape == fake.grad_data.shape
            assert buf.dtype == fake.grad_data.dtype
            assert torch.all(buf == 0)

        # Independent storage: mutating the grad buffer must not leak into accum.
        chunk.buffers[0].grad_data.fill_(7.0)
        assert torch.all(accum[0] == 0)

    def test_accumulate_adds_scaled_gradient(self):
        chunk = _make_chunk(((3, 2), torch.float32), ((4,), torch.float32))
        grads = [b.grad_data.clone() for b in chunk.buffers]
        accum = allocate_grad_accum_buffers([chunk])

        accumulate_grad_buffers([chunk], accum, scale=1.5)
        accumulate_grad_buffers([chunk], accum, scale=-0.5)  # signed advantages

        for buf, g in zip(accum, grads, strict=True):
            torch.testing.assert_close(buf, (1.5 - 0.5) * g)

    def test_accumulate_with_zero_scale_is_a_no_op(self):
        chunk = _make_chunk(((3,), torch.float32))
        accum = allocate_grad_accum_buffers([chunk])
        accumulate_grad_buffers([chunk], accum, scale=0.0)
        assert torch.all(accum[0] == 0)

    def test_copy_accum_buffers_overwrites_grad_buffers(self):
        chunk = _make_chunk(((2, 2), torch.float32))
        accum = [torch.full((2, 2), 3.25)]
        copy_accum_buffers_to_grad_buffers([chunk], accum)
        torch.testing.assert_close(chunk.buffers[0].grad_data, accum[0])

    def test_move_grad_buffers_applies_opob_baseline_subtraction(self):
        # move_grad_buffers(src=G_S, dest=G_R, scale=-b) must produce G_R - b*G_S.
        g_r = [torch.randn(4), torch.randn(2, 3)]
        g_s = [torch.randn(4), torch.randn(2, 3)]
        expected = [r - 0.7 * s for r, s in zip(g_r, g_s, strict=True)]

        dest = [t.clone() for t in g_r]
        move_grad_buffers(src=g_s, dest=dest, scale=-0.7)
        for d, e in zip(dest, expected, strict=True):
            torch.testing.assert_close(d, e)

    def test_apply_scaled_grad_delta(self):
        # grad <- base + (grad - base) * scale
        chunk = _make_chunk(((5,), torch.float32))
        base = [torch.randn(5)]
        grad_before = chunk.buffers[0].grad_data.clone()

        apply_scaled_grad_delta([chunk], base, scale=0.25)
        torch.testing.assert_close(chunk.buffers[0].grad_data, base[0] + (grad_before - base[0]) * 0.25)

    def test_apply_scaled_grad_delta_scale_zero_restores_base(self):
        chunk = _make_chunk(((5,), torch.float32))
        base = [torch.randn(5)]
        apply_scaled_grad_delta([chunk], base, scale=0.0)
        torch.testing.assert_close(chunk.buffers[0].grad_data, base[0])

    def test_zero_grad_accum_buffers(self):
        accum = [torch.randn(3), torch.randn(2, 2)]
        zero_grad_accum_buffers(accum)
        for buf in accum:
            assert torch.all(buf == 0)

    def test_snapshot_clone_is_independent(self):
        chunk = _make_chunk(((3,), torch.float32))
        snap = snapshot_grad_buffers([chunk], to_cpu=False)
        torch.testing.assert_close(snap[0], chunk.buffers[0].grad_data)
        chunk.buffers[0].grad_data.add_(1.0)
        assert not torch.equal(snap[0], chunk.buffers[0].grad_data)


class TestAdvantageFoldingEquivalence:
    """The buffer-free per-traj path replaces `accum += adv_i * g_i` (per-sample
    backward with advantage 1, then scaled buffer accumulation) with a loss
    multiplied by `loss_scale * adv_i` and natural gradient accumulation.
    Backward is linear in the loss scale, so both must produce identical
    gradients — including for negative and exactly-zero advantages."""

    @staticmethod
    def _per_sample_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Arbitrary smooth nonlinear scalar loss.
        return (model(x).sum()) ** 2

    def test_folded_loss_scale_matches_buffer_accumulation(self):
        torch.manual_seed(1)
        model = nn.Linear(4, 3)
        xs = torch.randn(5, 4)
        advantages = [1.7, -0.9, 0.0, 3.2, -0.4]
        loss_scale = 1.0 / len(advantages)  # microbatch_loss_scale = 1/len(minibatch)
        params = list(model.parameters())

        # Path A — old buffer scheme: isolate each sample's gradient, then
        # accumulate it scaled by the advantage via the real vcpo helpers.
        accum = allocate_grad_accum_buffers([_FakeChunk([torch.zeros_like(p) for p in params])])
        for x, adv in zip(xs, advantages, strict=True):
            model.zero_grad()
            (loss_scale * self._per_sample_loss(model, x)).backward()
            accumulate_grad_buffers([_FakeChunk([p.grad for p in params])], accum, scale=adv)

        # Path B — buffer-free scheme: fold the advantage into the loss scale
        # and let autograd accumulate across samples.
        model.zero_grad()
        for x, adv in zip(xs, advantages, strict=True):
            (loss_scale * adv * self._per_sample_loss(model, x)).backward()

        for p, buf in zip(params, accum, strict=True):
            torch.testing.assert_close(p.grad, buf, rtol=1e-6, atol=1e-7)
