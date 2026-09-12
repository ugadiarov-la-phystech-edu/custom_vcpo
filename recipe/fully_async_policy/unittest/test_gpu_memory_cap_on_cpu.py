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
"""VERL_GPU_MEM_CAP_GB: emulate a smaller card on the ACTOR-role worker processes only.

torch has no env-var form of the allocator cap (PYTORCH_CUDA_ALLOC_CONF rejects
`per_process_memory_fraction`), so it is an API call in verl's ActorRolloutRefWorker.__init__
(both backends) guarded by `if self._is_actor`. That keeps it away from rollout-only workers
and from the vLLM server processes, which budget against the device's total memory and are
emulated by lowering rollout.gpu_memory_utilization instead.

Run: pytest recipe/fully_async_policy/unittest/test_gpu_memory_cap_on_cpu.py
"""

import glob
import os
from types import SimpleNamespace

import pytest

import verl.utils.gpu_memory_cap as gpu_memory_cap
from verl.utils.gpu_memory_cap import apply_gpu_memory_cap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
VERL_WORKERS = ("verl/workers/megatron_workers.py", "verl/workers/fsdp_workers.py")
RECIPE_WORKERS = ("recipe/fully_async_policy/megatron_worker.py", "recipe/fully_async_policy/fsdp_workers.py")
HOOK = "if self._is_actor:\n            apply_gpu_memory_cap()"

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


def _read(rel):
    with open(os.path.join(REPO_ROOT, rel)) as f:
        return f.read()


def _init_source(text):
    """Source of ActorRolloutRefWorker.__init__ (up to the next method def)."""
    start = text.index("class ActorRolloutRefWorker(")
    start = text.index("    def __init__(", start)
    end = text.index("\n    def ", start + 10)
    return text[start:end]


class TestWiring:
    @pytest.mark.parametrize("rel", VERL_WORKERS)
    def test_verl_actor_workers_call_it_for_the_actor_role_only(self, rel):
        """The hook lives in verl's ActorRolloutRefWorker.__init__ (both backends), after the role
        flags are known and BEFORE any model/optimizer allocation, guarded by `if self._is_actor` —
        never from the launch environment, where it would also reach the vLLM server processes."""
        text = _read(rel)
        assert "from verl.utils.gpu_memory_cap import apply_gpu_memory_cap" in text
        init = _init_source(text)
        assert init.count("apply_gpu_memory_cap()") == 1
        assert HOOK in init
        assert init.index("self._is_actor = ") < init.index(HOOK)
        # nothing heavy is built before the cap: the first model build happens in init_model, not here
        assert "build_model_optimizer" not in init and "_build_model_optimizer" not in init
        # the whole file calls it exactly once (no second, unguarded call elsewhere)
        assert text.count("apply_gpu_memory_cap()") == 1

    @pytest.mark.parametrize("rel", RECIPE_WORKERS)
    def test_recipe_workers_inherit_instead_of_calling_it(self, rel):
        """DetachActorWorker (role actor) gets the cap through ActorRolloutRefWorker.__init__;
        DetachAsyncRolloutWorker (role rollout) must not — neither should mention it."""
        assert "apply_gpu_memory_cap" not in _read(rel)

    def test_no_arm_exports_the_knob(self):
        """The shell arms only READ VERL_GPU_MEM_CAP_GB (emu_tag); exporting it there would hide an
        emulation in a script that is supposed to run un-capped on real H100s."""
        shell = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell")
        scripts = glob.glob(os.path.join(shell, "**", "*.sh"), recursive=True)
        assert scripts
        for path in scripts:
            with open(path) as f:
                assert "export VERL_GPU_MEM_CAP_GB" not in f.read(), path
