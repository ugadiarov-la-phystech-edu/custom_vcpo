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
`per_process_memory_fraction`), so it is an API call inside DetachActorWorker —
which is trainer-only, keeping it away from the vLLM engines that budget against
the device's total memory.

Run: pytest recipe/fully_async_policy/unittest/test_gpu_memory_cap_on_cpu.py
"""

from types import SimpleNamespace

import pytest

import recipe.fully_async_policy.fsdp_workers as fsdp_workers
from recipe.fully_async_policy.fsdp_workers import apply_gpu_memory_cap

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
    monkeypatch.setattr(fsdp_workers, "get_torch_device", lambda: dev)
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
            monkeypatch.setattr(fsdp_workers, "get_torch_device", lambda d=dev: d)
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
        monkeypatch.setattr(fsdp_workers, "get_torch_device", lambda: dev)
        monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "80")
        assert apply_gpu_memory_cap() is None
        assert dev.fractions == []


class TestWiring:
    def test_the_trainer_worker_calls_it(self):
        """It must run in DetachActorWorker.__init__ (trainer-only), next to
        set_expandable_segments — never from the launch environment, where it
        would also reach the vLLM engines."""
        import inspect

        src = inspect.getsource(fsdp_workers.DetachActorWorker.__init__)
        assert "apply_gpu_memory_cap()" in src
        assert "set_expandable_segments(True)" in src

    def test_the_rollout_worker_does_not(self):
        for cls_name in ("DetachAsyncRolloutWorker",):
            cls = getattr(fsdp_workers, cls_name, None)
            if cls is None or "__init__" not in cls.__dict__:
                continue
            import inspect

            assert "apply_gpu_memory_cap" not in inspect.getsource(cls.__init__), cls_name
