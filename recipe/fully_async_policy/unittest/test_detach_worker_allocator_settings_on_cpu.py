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

"""Regression pins for the trainer-only allocator setting in the detach workers.

The Megatron 6+2 dynamic-bsz config OOMed on a ~9.6 GiB contiguous fp32
logits-sized allocation while ~10 GiB sat reserved-but-unallocated (fragmentation;
smoke tests, 2026-07-30). The fix is ``set_expandable_segments(True)`` in
``DetachActorWorker.__init__`` — and it must stay scoped to the TRAINER worker
process: vLLM's sleep-mode allocator hard-asserts against expandable segments
(vllm/device_allocator/cumem.py), so the rollout worker must never enable it,
and the setting must never travel via the launch environment (env vars reach
the engine processes).

Removing the call does not error anywhere — the OOM just resurfaces hundreds of
steps into some future run. These tests turn that silent regression into a red
test.
"""

import pytest

import recipe.fully_async_policy.megatron_worker as megatron_worker_module


@pytest.fixture
def recorded_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(megatron_worker_module, "set_expandable_segments", lambda enable: calls.append(enable))
    return calls


def _neutralize_parent_init(monkeypatch):
    """Detach worker __init__s delegate to the heavy Megatron worker init; stub it out."""

    def stub_init(self, *args, **kwargs):
        pass

    monkeypatch.setattr(megatron_worker_module.AsyncActorRolloutRefWorker, "__init__", stub_init)
    monkeypatch.setattr(megatron_worker_module.ActorRolloutRefWorker, "__init__", stub_init)


class TestDetachActorWorkerAllocatorSettings:
    def test_trainer_worker_enables_expandable_segments(self, monkeypatch, recorded_calls):
        _neutralize_parent_init(monkeypatch)
        megatron_worker_module.DetachActorWorker()
        assert recorded_calls == [True], (
            "DetachActorWorker.__init__ must call set_expandable_segments(True) exactly once; "
            "without it the tp=1 dynamic-bsz trainer OOMs on fragmentation (see module docstring)"
        )

    def test_rollout_worker_does_not_touch_allocator_settings(self, monkeypatch, recorded_calls):
        _neutralize_parent_init(monkeypatch)
        megatron_worker_module.DetachAsyncRolloutWorker(config=None, role="rollout")
        assert recorded_calls == [], (
            "the rollout worker must NOT enable expandable segments: vLLM's sleep-mode "
            "allocator asserts against them (vllm/device_allocator/cumem.py)"
        )

    def test_trainer_worker_sets_allocator_after_parent_init(self, monkeypatch, recorded_calls):
        """The call must come after super().__init__ so nothing in parent init can undo it."""
        order = []

        def stub_init(self, *args, **kwargs):
            order.append("parent_init")

        monkeypatch.setattr(megatron_worker_module.AsyncActorRolloutRefWorker, "__init__", stub_init)
        monkeypatch.setattr(
            megatron_worker_module,
            "set_expandable_segments",
            lambda enable: order.append(("set_expandable_segments", enable)),
        )
        megatron_worker_module.DetachActorWorker()
        assert order == ["parent_init", ("set_expandable_segments", True)]
