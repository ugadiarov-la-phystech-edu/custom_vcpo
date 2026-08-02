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
"""Unit tests for the stop-the-world accounting modes:
- serialize_validation: the trainer blocks on the validation it just launched
  (mirrored trigger condition), and the stall joins _step_wait_valid_time so
  the virtual clock excludes it
- pause_generation_during_save: the trainer brackets the whole checkpoint save
  with rollouter begin_save_pause/end_save_pause; the rollouter accounts the
  full frozen window once (save_checkpoint skips its internal pause), and the
  bracket resumes generation even when the save fails

Run: pytest recipe/fully_async_policy/unittest/test_stop_the_world_on_cpu.py
"""

import asyncio
import time
from collections import defaultdict
from types import SimpleNamespace

from omegaconf import OmegaConf

import recipe.fully_async_policy.fully_async_trainer as fat_module
from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor


def _unwrap_ray_actor_class(actor_cls):
    """Both classes are @ray.remote ActorClass wrappers; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap_ray_actor_class(_RollouterActor)
FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)


class _FakeRay:
    """The trainer calls ray.get(stub.method.remote(...)); the stubs below
    return plain callables from .remote, and this fake executes them."""

    @staticmethod
    def get(ref):
        return ref() if callable(ref) else ref


# ---------------------------------------------------------------------------
# serialize_validation (trainer side)
# ---------------------------------------------------------------------------


class _StubSynchronizer:
    def __init__(self, wait_duration=0.0):
        self.calls = []
        self._wait_duration = wait_duration
        self.wait_last_valid = SimpleNamespace(remote=self._wait_remote)
        self.sync_weights = SimpleNamespace(remote=self._sync_remote)

    def _wait_remote(self):
        def run():
            self.calls.append("wait")
            time.sleep(self._wait_duration)

        return run

    def _sync_remote(self, version, validate=False, global_steps=None):
        def run():
            self.calls.append("sync")

        return run


def _make_sync_trainer(serialize_validation, test_freq, param_version_before, wait_duration=0.0):
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.local_trigger_step = 1
    trainer.trigger_parameter_sync_step = 1
    trainer.serialize_validation = serialize_validation
    trainer.current_param_version = param_version_before
    trainer._step_wait_valid_time = 0.0
    trainer.rollouter_first_sample_time = None  # _add_cumulative_time_metrics no-ops
    trainer.structured_metrics = defaultdict(list)
    trainer.param_synchronizer = _StubSynchronizer(wait_duration=wait_duration)
    trainer.config = OmegaConf.create({"trainer": {"logger": ["console"]}, "rollout": {"test_freq": test_freq}})
    trainer.metrics_aggregator = SimpleNamespace(get_aggregated_metrics=lambda: {}, reset=lambda: None)
    trainer.logger = SimpleNamespace(log=lambda data, step: None)
    trainer.progress_bar = SimpleNamespace(update=lambda n: None)
    trainer._log_validation_data = lambda: None
    return trainer


def test_sync_triggers_validation_mirrors_rollouter_condition():
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.config = OmegaConf.create({"rollout": {"test_freq": 5}})
    for version, expected in [(0, False), (1, False), (4, False), (5, True), (10, True), (11, False)]:
        trainer.current_param_version = version
        assert trainer._sync_triggers_validation() is expected, f"version={version}"
    trainer.config = OmegaConf.create({"rollout": {"test_freq": 0}})
    trainer.current_param_version = 5
    assert trainer._sync_triggers_validation() is False


def test_serialized_validation_blocks_after_sync_and_excludes_stall(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    # version 4 -> 5, test_freq 5: this sync triggers validation
    trainer = _make_sync_trainer(serialize_validation=True, test_freq=5, param_version_before=4, wait_duration=0.03)
    trainer._trigger_parameter_sync_after_step(global_steps=1)
    # initial wait (previous validation) -> sync -> serialized wait
    assert trainer.param_synchronizer.calls == ["wait", "sync", "wait"]
    assert trainer._step_wait_valid_time >= 0.03


def test_serialized_validation_skips_non_validation_syncs(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    # version 5 -> 6, test_freq 5: no validation on this sync
    trainer = _make_sync_trainer(serialize_validation=True, test_freq=5, param_version_before=5, wait_duration=0.03)
    trainer._trigger_parameter_sync_after_step(global_steps=1)
    assert trainer.param_synchronizer.calls == ["wait", "sync"]
    # only the initial wait (for the previous validation) is accumulated
    assert trainer._step_wait_valid_time < 0.06


def test_serialize_validation_disabled_keeps_overlap(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    trainer = _make_sync_trainer(serialize_validation=False, test_freq=5, param_version_before=4)
    trainer._trigger_parameter_sync_after_step(global_steps=1)
    # no serialized wait even though this sync triggers validation
    assert trainer.param_synchronizer.calls == ["wait", "sync"]


# ---------------------------------------------------------------------------
# pause_generation_during_save (rollouter side)
# ---------------------------------------------------------------------------


def _make_rollouter(first_sample_time, save_queue_state=True):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.condition = asyncio.Condition()
    r.lock = r.condition._lock
    r.checkpointing = False
    r._external_save_pause_active = False
    r._external_save_pause_start = None
    r.dataloader_lock = asyncio.Lock()
    r.train_dataloader = type("DL", (), {"state_dict": lambda self: {}})()
    r.config = OmegaConf.create({"async_training": {"save_queue_state": save_queue_state}})
    r.first_sample_time = first_sample_time
    r.cumulative_checkpoint_pause = 0.0
    r.global_steps = 1
    r.staleness_samples = 0
    r.total_generated_samples = 0
    r.dropped_stale_samples = 0
    r.processed_sample_count = 0
    r.current_param_version = 1
    r.message_queue_client = None
    r.pause_calls = 0
    r.resume_calls = 0

    async def snapshot():
        return {}

    async def pause():
        r.pause_calls += 1

    async def resume():
        r.resume_calls += 1

    r._snapshot_internal_queues = snapshot
    r.pause = pause
    r.resume = resume
    return r


def test_save_pause_accounts_full_window_post_anchor():
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def run():
        await rollouter.begin_save_pause()
        assert rollouter.checkpointing is True
        assert rollouter.cumulative_checkpoint_pause == 0.0, "accounted only at end_save_pause"
        await asyncio.sleep(0.03)
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.pause_calls == 1 and rollouter.resume_calls == 1
    assert rollouter.checkpointing is False
    assert rollouter._external_save_pause_active is False
    assert rollouter.cumulative_checkpoint_pause >= 0.03


def test_save_pause_accounts_nothing_pre_anchor():
    rollouter = _make_rollouter(first_sample_time=None)

    async def run():
        await rollouter.begin_save_pause()
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.cumulative_checkpoint_pause == 0.0
    assert rollouter.pause_calls == 1 and rollouter.resume_calls == 1


def test_save_checkpoint_defers_to_external_pause(tmp_path):
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def run():
        await rollouter.begin_save_pause()
        await rollouter.save_checkpoint(str(tmp_path / "ckpt"))
        # save_checkpoint must not pause/resume/account on its own
        assert rollouter.pause_calls == 1 and rollouter.resume_calls == 0
        assert rollouter.checkpointing is True, "still frozen until end_save_pause"
        assert rollouter.cumulative_checkpoint_pause == 0.0
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.resume_calls == 1
    assert rollouter.cumulative_checkpoint_pause > 0.0


def test_save_checkpoint_without_external_pause_unchanged(tmp_path):
    rollouter = _make_rollouter(first_sample_time=100.0)
    asyncio.run(rollouter.save_checkpoint(str(tmp_path / "ckpt")))
    assert rollouter.pause_calls == 1 and rollouter.resume_calls == 1
    assert rollouter.checkpointing is False
    assert rollouter.cumulative_checkpoint_pause > 0.0


# ---------------------------------------------------------------------------
# _save_checkpoint bracket (trainer side)
# ---------------------------------------------------------------------------


class _StubSaveSynchronizer:
    def __init__(self, calls):
        self.pause_rollouter_for_save = SimpleNamespace(remote=lambda: lambda: calls.append("pause"))
        self.resume_rollouter_after_save = SimpleNamespace(remote=lambda: lambda: calls.append("resume"))


def _make_save_trainer(pause_generation_during_save, inner):
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.pause_generation_during_save = pause_generation_during_save
    trainer._save_checkpoint_inner = inner
    return trainer


def test_save_bracket_orders_pause_inner_resume(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []
    trainer = _make_save_trainer(True, inner=lambda: calls.append("inner"))
    trainer.param_synchronizer = _StubSaveSynchronizer(calls)
    trainer._save_checkpoint()
    assert calls == ["pause", "inner", "resume"]


def test_save_bracket_resumes_on_failure(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []

    def failing_inner():
        calls.append("inner")
        raise RuntimeError("disk full")

    trainer = _make_save_trainer(True, inner=failing_inner)
    trainer.param_synchronizer = _StubSaveSynchronizer(calls)
    try:
        trainer._save_checkpoint()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the save failure to propagate")
    assert calls == ["pause", "inner", "resume"]


def test_save_bracket_disabled_calls_inner_only(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []
    trainer = _make_save_trainer(False, inner=lambda: calls.append("inner"))
    trainer.param_synchronizer = _StubSaveSynchronizer(calls)
    trainer._save_checkpoint()
    assert calls == ["inner"]
