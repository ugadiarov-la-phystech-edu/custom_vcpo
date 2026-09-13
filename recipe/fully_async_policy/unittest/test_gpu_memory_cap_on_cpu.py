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
"""VERL_GPU_MEM_CAP_GB: emulate a smaller card on the TRAINER processes only.

torch has no env-var form of the allocator cap (PYTORCH_CUDA_ALLOC_CONF rejects
`per_process_memory_fraction`), so it is an API call inside both DetachActorWorker classes
(FSDP2 and Megatron) — trainer-only, keeping it away from the vLLM engines that budget
against the device's total memory. Ported from the fsdp2 cppo branch (0360e6e); the function now
lives in recipe/fully_async_policy/gpu_memory_cap.py and is shared by both worker modules.

Run: pytest recipe/fully_async_policy/unittest/test_gpu_memory_cap_on_cpu.py
"""

import os
import re
from types import SimpleNamespace

import pytest

import recipe.fully_async_policy.gpu_memory_cap as gpu_memory_cap
from recipe.fully_async_policy.gpu_memory_cap import apply_gpu_memory_cap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TOTAL_GIB = 140.4  # an H200: 143771 MiB
TOTAL_BYTES = int(TOTAL_GIB * 1024**3)


class _FakeDevice:
    def __init__(self, available=True, total=TOTAL_BYTES):
        self.available = available
        self.total = total
        self.fractions = []

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=self.total)

    def set_per_process_memory_fraction(self, fraction):
        self.fractions.append(fraction)


@pytest.fixture()
def device(monkeypatch):
    dev = _FakeDevice()
    monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: dev)
    return dev


class TestUnset:
    def test_absent_env_is_a_noop(self, device, monkeypatch):
        monkeypatch.delenv("VERL_GPU_MEM_CAP_GB", raising=False)
        assert apply_gpu_memory_cap() is None
        assert device.fractions == []

    def test_empty_string_is_a_noop(self, device, monkeypatch):
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "")
        assert apply_gpu_memory_cap() is None
        assert device.fractions == []


class TestApplied:
    def test_h100_on_h200_fraction(self, device, monkeypatch):
        """The motivating case: 80 GiB of a 140.4 GiB card."""
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "80")
        fraction = apply_gpu_memory_cap()
        assert fraction == pytest.approx(80 / TOTAL_GIB, rel=1e-6)
        assert device.fractions == [fraction]
        assert 0.56 < fraction < 0.58

    def test_fraction_is_recomputed_per_device_size(self, monkeypatch):
        """The knob is in GB, not a fraction, so the same value means the same
        ceiling on cards of different capacity."""
        for total_gib, expected in ((140.4, 80 / 140.4), (94.0, 80 / 94.0)):
            dev = _FakeDevice(total=int(total_gib * 1024**3))
            monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda d=dev: d)
            monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "80")
            assert apply_gpu_memory_cap() == pytest.approx(expected, rel=1e-6)

    def test_fractional_values_are_accepted(self, device, monkeypatch):
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "79.5")
        assert apply_gpu_memory_cap() == pytest.approx(79.5 / TOTAL_GIB, rel=1e-6)


class TestGuards:
    def test_cap_at_or_above_the_device_is_refused(self, device, monkeypatch, caplog):
        """Capping at 100% would be a no-op that reads like protection."""
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "200")
        with caplog.at_level("WARNING"):
            assert apply_gpu_memory_cap() is None
        assert device.fractions == []
        assert any("uncapped" in r.message for r in caplog.records)

    @pytest.mark.parametrize("bad", ["0", "-5"])
    def test_non_positive_is_an_error(self, device, monkeypatch, bad):
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", bad)
        with pytest.raises(ValueError, match="VERL_GPU_MEM_CAP_GB"):
            apply_gpu_memory_cap()

    def test_garbage_value_raises(self, device, monkeypatch):
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "eighty")
        with pytest.raises(ValueError):
            apply_gpu_memory_cap()

    def test_no_cuda_is_a_noop(self, monkeypatch):
        dev = _FakeDevice(available=False)
        monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: dev)
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "80")
        assert apply_gpu_memory_cap() is None
        assert dev.fractions == []


WORKERS = {
    "fsdp2": "recipe/fully_async_policy/fsdp_workers.py",
    "megatron": "recipe/fully_async_policy/megatron_worker.py",
}


def _class_body(text, name):
    m = re.search(rf"^class {name}\b.*?(?=^class |\Z)", text, flags=re.S | re.M)
    assert m, name
    return m.group(0)


class TestWiring:
    """Both trainer workers must call it in DetachActorWorker.__init__, right after
    set_expandable_segments — never from the launch environment (it would reach the vLLM
    engines) and never from the rollout workers. Checked on the source text so the test
    needs neither Megatron nor a GPU."""

    @pytest.mark.parametrize("backend", sorted(WORKERS))
    def test_the_trainer_worker_calls_it_after_expandable_segments(self, backend):
        text = open(os.path.join(REPO_ROOT, WORKERS[backend])).read()
        assert "from recipe.fully_async_policy.gpu_memory_cap import apply_gpu_memory_cap" in text
        body = _class_body(text, "DetachActorWorker")
        assert "set_expandable_segments(True)" in body
        assert "apply_gpu_memory_cap()" in body
        assert body.index("set_expandable_segments(True)") < body.index("apply_gpu_memory_cap()")

    @pytest.mark.parametrize("backend", sorted(WORKERS))
    def test_the_rollout_worker_does_not(self, backend):
        text = open(os.path.join(REPO_ROOT, WORKERS[backend])).read()
        assert "apply_gpu_memory_cap" not in _class_body(text, "DetachAsyncRolloutWorker")

    def test_no_arm_exports_the_knob(self):
        """The arms only READ VERL_GPU_MEM_CAP_GB (emu_tag); setting it belongs to the launching shell."""
        import glob

        arms = glob.glob(os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo*.sh"))
        assert arms
        for arm in arms:
            assert "export VERL_GPU_MEM_CAP_GB" not in open(arm).read(), arm
