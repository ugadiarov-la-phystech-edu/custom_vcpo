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
  full frozen window (post-anchor only), and the bracket resumes generation
  even when the save fails

Run: pytest recipe/fully_async_policy/unittest/test_stop_the_world_on_cpu.py
"""

import asyncio
import time
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
    assert trainer._step_wait_valid_time >= 0.06, "both stalls must be excluded from the virtual clock"


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


def test_forced_validate_serializes_regardless_of_test_freq(monkeypatch):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    # version 5 -> 6 would not validate on cadence, but validate=True forces it
    trainer = _make_sync_trainer(serialize_validation=True, test_freq=5, param_version_before=5, wait_duration=0.03)
    trainer._trigger_parameter_sync_after_step(validate=True, global_steps=1)
    assert trainer.param_synchronizer.calls == ["wait", "sync", "wait"]


def test_non_sync_step_returns_early_and_touches_nothing(monkeypatch):
    """Between syncs (trigger_parameter_sync_step > 1) the method must not wait,
    not bump the version, and not log — the serialized wait included."""
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    trainer = _make_sync_trainer(serialize_validation=True, test_freq=1, param_version_before=4)
    trainer.trigger_parameter_sync_step = 4
    trainer.local_trigger_step = 1
    logged = []
    trainer.logger = SimpleNamespace(log=lambda data, step: logged.append(step))

    trainer._trigger_parameter_sync_after_step(global_steps=1)

    assert trainer.param_synchronizer.calls == []
    assert trainer.current_param_version == 4
    assert trainer.local_trigger_step == 2
    assert trainer._step_wait_valid_time == 0.0
    assert logged == []


def test_sync_logs_metrics_after_draining_validation_data(monkeypatch):
    """The timing metrics are attached to the aggregated step data, so the
    validation queue must be drained (anchor cached) before that log."""
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    trainer = _make_sync_trainer(serialize_validation=False, test_freq=5, param_version_before=4)
    order = []
    logged = []

    def drain():
        order.append("drain")
        trainer.rollouter_first_sample_time = 100.0
        trainer.rollouter_cumulative_validation_time = 5.0
        trainer.cumulative_save_time = 2.0
        trainer.timing_wall_offset = 0.0
        trainer.timing_validation_offset = 0.0
        trainer.timing_save_offset = 0.0
        trainer.virtual_free_time = None
        trainer.virtual_training_time_offset = 0.0
        trainer._step_virtual_start = None
        trainer._step_actual_start = None
        trainer._step_save_time = 0.0

    trainer._log_validation_data = drain
    trainer.logger = SimpleNamespace(log=lambda data, step: (order.append("log"), logged.append(data)))
    trainer._trigger_parameter_sync_after_step(global_steps=1)

    assert order[:2] == ["drain", "log"]
    step_data = logged[0]
    assert "fully_async/timing/wall_time_since_first_sample" in step_data
    assert step_data["fully_async/timing/cumulative_validation_time"] == 5.0
    assert step_data["fully_async/timing/cumulative_save_time"] == 2.0


# ---------------------------------------------------------------------------
# pause_generation_during_save (rollouter side)
# ---------------------------------------------------------------------------


def _make_rollouter(first_sample_time):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.condition = asyncio.Condition()
    r.lock = r.condition._lock
    r.checkpointing = False
    r._external_save_pause_active = False
    r._external_save_pause_start = None
    r.first_sample_time = first_sample_time
    r.cumulative_checkpoint_pause = 0.0
    r.pause_calls = 0
    r.resume_calls = 0

    async def pause():
        r.pause_calls += 1

    async def resume():
        r.resume_calls += 1

    r.pause = pause
    r.resume = resume
    return r


def test_save_pause_accounts_full_window_post_anchor():
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def run():
        await rollouter.begin_save_pause()
        assert rollouter.checkpointing is True
        assert rollouter._external_save_pause_active is True
        assert rollouter.cumulative_checkpoint_pause == 0.0, "accounted only at end_save_pause"
        await asyncio.sleep(0.03)
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.pause_calls == 1 and rollouter.resume_calls == 1
    assert rollouter.checkpointing is False
    assert rollouter._external_save_pause_active is False
    assert rollouter.cumulative_checkpoint_pause >= 0.03


def test_save_pause_accumulates_across_saves():
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def one_save():
        await rollouter.begin_save_pause()
        await asyncio.sleep(0.03)
        await rollouter.end_save_pause()

    asyncio.run(one_save())
    first = rollouter.cumulative_checkpoint_pause
    asyncio.run(one_save())
    assert rollouter.cumulative_checkpoint_pause >= first + 0.03


def test_save_pause_accounts_nothing_pre_anchor():
    rollouter = _make_rollouter(first_sample_time=None)

    async def run():
        await rollouter.begin_save_pause()
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.cumulative_checkpoint_pause == 0.0
    assert rollouter.pause_calls == 1 and rollouter.resume_calls == 1


def test_save_pause_excludes_time_spent_waiting_for_the_lock():
    """begin_save_pause takes its start stamp *inside* the lock: time spent
    queued behind a running validation belongs to that accumulator, not to the
    checkpoint one, and must never be counted twice."""
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def run():
        async def lock_holder():
            async with rollouter.lock:
                await asyncio.sleep(0.15)

        holder = asyncio.create_task(lock_holder())
        await asyncio.sleep(0.02)  # make sure the lock is taken

        started = time.time()
        await rollouter.begin_save_pause()
        await rollouter.end_save_pause()
        elapsed = time.time() - started
        await holder
        return elapsed

    elapsed = asyncio.run(run())
    assert elapsed >= 0.1, "begin_save_pause must really have queued behind the lock"
    assert rollouter.cumulative_checkpoint_pause < 0.05, "the lock wait must not be charged to the save"


def test_end_save_pause_without_begin_accounts_nothing():
    rollouter = _make_rollouter(first_sample_time=100.0)
    asyncio.run(rollouter.end_save_pause())
    assert rollouter.cumulative_checkpoint_pause == 0.0
    assert rollouter.resume_calls == 1
    assert rollouter.checkpointing is False


def test_save_pause_shifts_sample_stamps_for_the_trainer():
    """Cross-component check: the pause the rollouter accumulates is the same
    quantity the trainer subtracts from a sample's arrival time."""
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def run():
        await rollouter.begin_save_pause()
        await asyncio.sleep(0.05)
        await rollouter.end_save_pause()

    asyncio.run(run())
    pause = rollouter.cumulative_checkpoint_pause
    assert pause >= 0.05

    trainer = object.__new__(FullyAsyncTrainer)
    trainer.rollouter_first_sample_time = 100.0
    trainer.virtual_free_time = 100.0
    enqueue_time = 200.0
    trainer._open_virtual_step(
        250.0,
        [SimpleNamespace(enqueue_time=enqueue_time, validation_pause_before=0.0, checkpoint_pause_before=pause)],
    )
    assert abs(trainer._step_virtual_start - (enqueue_time - pause)) < 1e-9


def test_feed_loop_blocks_while_paused():
    """The same gate must also hold the feeder during an ordinary sync pause."""

    async def run():
        rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        rollouter.condition = asyncio.Condition()
        rollouter.lock = rollouter.condition._lock
        rollouter.paused = True
        rollouter.checkpointing = False
        rollouter.pending_queue = asyncio.Queue(maxsize=8)
        rollouter.global_steps = 1
        rollouter.total_rollout_steps = 1
        rollouter.first_sample_time = None
        rollouter.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}})
        rollouter._create_continuous_iterator = lambda: iter([(0, {"dummy": 1})])

        import recipe.fully_async_policy.fully_async_rollouter as rollouter_module

        saved = rollouter_module.prepare_single_generation_data
        rollouter_module.prepare_single_generation_data = lambda batch_dict, config: batch_dict
        try:
            feed = asyncio.create_task(rollouter._feed_samples())
            await asyncio.sleep(0.05)
            assert rollouter.pending_queue.qsize() == 0

            async with rollouter.lock:
                rollouter.paused = False
                rollouter.condition.notify_all()
            await asyncio.wait_for(feed, timeout=5)
            assert rollouter.pending_queue.qsize() == 2
        finally:
            rollouter_module.prepare_single_generation_data = saved

    asyncio.run(run())


def test_feed_loop_blocks_while_checkpointing():
    """The stop-the-world save must also stop the feeder, not only generation."""

    async def run():
        rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        rollouter.condition = asyncio.Condition()
        rollouter.lock = rollouter.condition._lock
        rollouter.paused = False
        rollouter.checkpointing = True
        rollouter.pending_queue = asyncio.Queue(maxsize=8)
        rollouter.global_steps = 1
        rollouter.total_rollout_steps = 1
        rollouter.first_sample_time = None
        rollouter.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}})
        rollouter._create_continuous_iterator = lambda: iter([(0, {"dummy": 1})])

        import recipe.fully_async_policy.fully_async_rollouter as rollouter_module

        saved = rollouter_module.prepare_single_generation_data
        rollouter_module.prepare_single_generation_data = lambda batch_dict, config: batch_dict
        try:
            feed = asyncio.create_task(rollouter._feed_samples())
            await asyncio.sleep(0.05)
            assert rollouter.pending_queue.qsize() == 0, "feeder must be frozen while checkpointing"
            assert rollouter.first_sample_time is not None, "the anchor is stamped at the draw itself"

            async with rollouter.lock:
                rollouter.checkpointing = False
                rollouter.condition.notify_all()
            await asyncio.wait_for(feed, timeout=5)
            # the sample plus the DONE sentinel
            assert rollouter.pending_queue.qsize() == 2
        finally:
            rollouter_module.prepare_single_generation_data = saved

    asyncio.run(run())


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


def test_check_save_checkpoint_drives_the_bracket_and_accounts_the_frozen_window(monkeypatch):
    """Through the real save path: _check_save_checkpoint -> _save_checkpoint
    bracket. The measured save time must cover the whole frozen window (pause
    + inner save + resume), since that is what the pipeline stood still for."""
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []

    trainer = object.__new__(FullyAsyncTrainer)
    trainer.pause_generation_during_save = True
    trainer.cumulative_save_time = 0.0
    trainer._step_save_time = 0.0
    trainer.current_param_version = 1
    trainer.last_ckpt_version = 0
    trainer.max_steps_duration = 0
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})

    def inner():
        calls.append("inner")
        time.sleep(0.05)

    trainer._save_checkpoint_inner = inner

    class _Sync:
        pause_rollouter_for_save = SimpleNamespace(remote=lambda: lambda: (calls.append("pause"), time.sleep(0.03)))
        resume_rollouter_after_save = SimpleNamespace(remote=lambda: lambda: calls.append("resume"))

    trainer.param_synchronizer = _Sync()

    timing_raw = {}
    trainer._check_save_checkpoint(timing_raw)

    assert calls == ["pause", "inner", "resume"]
    assert trainer.last_ckpt_version == 1
    assert timing_raw["save_checkpoint"] >= 0.08, "the pause is part of the frozen window"
    assert trainer.cumulative_save_time == trainer._step_save_time == timing_raw["save_checkpoint"]


def test_config_defaults_are_off():
    """Both modes must default to False so the unset config keeps the old behaviour."""
    for name in ("fully_async_ppo_trainer.yaml", "fully_async_ppo_megatron_trainer.yaml"):
        cfg = OmegaConf.load(f"recipe/fully_async_policy/config/{name}")
        assert cfg.async_training.serialize_validation is False, name
        assert cfg.async_training.pause_generation_during_save is False, name
