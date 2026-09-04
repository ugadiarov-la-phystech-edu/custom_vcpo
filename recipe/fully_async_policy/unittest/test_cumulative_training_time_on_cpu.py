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
"""Unit tests for the fully_async/timing/* metrics, in particular
cumulative_training_time: the wall clock an identical run with neither
validation nor checkpointing would have needed.

Covered:
- rollouter: the first-sample anchor, validation-time accumulation (only after
  the anchor), and the ValidateMetrics payload carried to the trainer
- trainer: checkpoint-save accounting, the metric math, and the virtual
  (no-validation-no-save) clock in rollout-bound / trainer-bound regimes
- resume: timing_state.json round-trip and chaining across restarts

Run: pytest recipe/fully_async_policy/unittest/test_cumulative_training_time_on_cpu.py
"""

import asyncio
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import ray.cloudpickle
from omegaconf import OmegaConf

from recipe.fully_async_policy.detach_utils import ValidateMetrics
from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from recipe.fully_async_policy.fully_async_trainer import _format_datetime


def _unwrap_ray_actor_class(actor_cls):
    """Both classes are @ray.remote ActorClass wrappers; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap_ray_actor_class(_RollouterActor)
FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)

TIMING_PREFIX = "fully_async/timing/"


class _StubMessageQueueClient:
    def __init__(self, val_payloads=None):
        self.put_payloads = []
        self._val_payloads = list(val_payloads or [])

    async def get_queue_size(self):
        return 0

    async def put_validate(self, data):
        self.put_payloads.append(data)

    def get_validate_sync(self):
        return self._val_payloads.pop(0) if self._val_payloads else None

    def last_validate_metrics(self) -> ValidateMetrics:
        return ray.cloudpickle.loads(self.put_payloads[-1])


class _StubLogger:
    def __init__(self):
        self.logged = []

    def log(self, data=None, step=None):
        self.logged.append((data, step))


def _make_rollouter(first_sample_time, test_freq=1, validate_duration=0.03):
    """Minimal FullyAsyncRollouter with only the attributes update_param_version touches."""
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.lock = asyncio.Lock()
    r.current_param_version = 0
    r.active_tasks = set()
    r.cancel_queue = asyncio.Queue()
    r.message_queue_client = _StubMessageQueueClient()
    r.idle_start_time = None
    r.version_start_time = None
    r.val_reward_fn = object()  # non-None enables validation
    r.config = OmegaConf.create({"rollout": {"test_freq": test_freq}})
    r.first_sample_time = first_sample_time
    r.cumulative_validation_time = 0.0
    r._validate = lambda: (time.sleep(validate_duration), {"val-core/acc": 1.0})[1]
    return r


def _make_trainer(
    first_sample_time=None, cumulative_validation_time=0.0, cumulative_save_time=0.0, run_start_datetime=None
):
    """Minimal FullyAsyncTrainer with only the attributes under test."""
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.run_start_datetime = run_start_datetime or _format_datetime(time.time())
    t.rollouter_first_sample_time = first_sample_time
    t.rollouter_cumulative_validation_time = cumulative_validation_time
    t.cumulative_save_time = cumulative_save_time
    t.timing_wall_offset = 0.0
    t.timing_validation_offset = 0.0
    t.timing_save_offset = 0.0
    t.virtual_free_time = None
    t.virtual_training_time_offset = 0.0
    t._step_virtual_start = None
    t._step_actual_start = None
    t._step_wait_valid_time = 0.0
    t._step_save_time = 0.0
    t.current_param_version = 0
    t.last_ckpt_version = 0
    t.max_steps_duration = 0
    t.logger = _StubLogger()
    return t


# ---------------------------------------------------------------- ValidateMetrics


def test_validate_metrics_defaults_and_pickle_roundtrip():
    default = ValidateMetrics(timing_raw={})
    assert default.first_sample_time is None
    assert default.cumulative_validation_time is None

    original = ValidateMetrics(timing_raw={"x": 1.0}, first_sample_time=123.5, cumulative_validation_time=7.25)
    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(original))
    assert restored.first_sample_time == 123.5
    assert restored.cumulative_validation_time == 7.25


# ---------------------------------------------------------------- rollouter side


def test_rollouter_accumulates_validation_time_after_anchor():
    anchor = time.time()
    rollouter = _make_rollouter(first_sample_time=anchor, test_freq=1)

    asyncio.run(rollouter.update_param_version(version=1))
    first_cumulative = rollouter.cumulative_validation_time
    assert first_cumulative >= 0.03, "validation duration must be accumulated"

    asyncio.run(rollouter.update_param_version(version=2))
    assert rollouter.cumulative_validation_time > first_cumulative, "must accumulate across validations"

    payload = rollouter.message_queue_client.last_validate_metrics()
    assert payload.first_sample_time == anchor
    assert payload.cumulative_validation_time == rollouter.cumulative_validation_time


def test_rollouter_ignores_validation_before_anchor():
    rollouter = _make_rollouter(first_sample_time=None, test_freq=1)

    # validation runs (val_before_train-style: forced, before any training sample)
    asyncio.run(rollouter.update_param_version(version=1, validate=True))

    payload = rollouter.message_queue_client.last_validate_metrics()
    assert payload.timing_raw.get("rollouter/validate_time", 0) >= 0.03, "validation itself must have run"
    assert rollouter.cumulative_validation_time == 0.0, "pre-anchor validation must not be accumulated"
    assert payload.first_sample_time is None


def test_rollouter_skips_accumulation_when_no_validation():
    rollouter = _make_rollouter(first_sample_time=time.time(), test_freq=10)

    asyncio.run(rollouter.update_param_version(version=1))  # 1 % 10 != 0 -> no validation

    payload = rollouter.message_queue_client.last_validate_metrics()
    assert payload.metrics is None
    assert rollouter.cumulative_validation_time == 0.0


def test_feed_samples_anchor_is_set_once():
    async def run(preset_anchor):
        rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        rollouter.condition = asyncio.Condition()
        rollouter.lock = rollouter.condition._lock
        rollouter.paused = False
        rollouter.checkpointing = False
        rollouter.pending_queue = asyncio.Queue(maxsize=8)
        rollouter.global_steps = 1
        rollouter.total_rollout_steps = 2
        rollouter.first_sample_time = preset_anchor
        rollouter.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}})
        rollouter._create_continuous_iterator = lambda: iter([(0, {"dummy": 1}), (0, {"dummy": 2})])

        import recipe.fully_async_policy.fully_async_rollouter as rollouter_module

        saved = rollouter_module.prepare_single_generation_data
        rollouter_module.prepare_single_generation_data = lambda batch_dict, config: batch_dict
        try:
            await rollouter._feed_samples()
        finally:
            rollouter_module.prepare_single_generation_data = saved
        return rollouter.first_sample_time

    before = time.time()
    anchor = asyncio.run(run(preset_anchor=None))
    assert anchor is not None and before <= anchor <= time.time(), "anchor must be set at the first draw"

    sentinel = 123.456
    assert asyncio.run(run(preset_anchor=sentinel)) == sentinel, "an existing anchor must never be overwritten"


class _StubBatch:
    def __init__(self):
        self.non_tensor_batch = {}

    def __len__(self):
        return 1


def _new_rollout_sample():
    from recipe.fully_async_policy.detach_utils import RolloutSample

    return RolloutSample(
        full_batch=_StubBatch(),
        agent_loop_output_list=[None],
        sample_id="s0",
        epoch=0,
        processing_times=[],
        tool_calls=[],
        param_version=0,
        param_version_start=[],
        param_version_end=[],
        rollout_status={},
    )


def _attach_streaming_stubs(rollouter):
    """Wire the minimum _process_single_sample_streaming needs; returns the list
    of RolloutSamples that reached the message queue."""
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0
    rollouter.processed_sample_count = 0
    put_samples = []

    class _MQ:
        async def put_sample(self, sample, param_version):
            put_samples.append(ray.cloudpickle.loads(sample))
            return True

    rollouter.message_queue_client = _MQ()

    async def _generate(full_batch, outputs):
        return full_batch, False

    rollouter.async_rollout_manager = SimpleNamespace(generate_single_sample_async=_generate)

    async def _stats():
        return {}

    rollouter.get_statistics = _stats
    return put_samples


def test_rollout_sample_stamp_defaults():
    sample = _new_rollout_sample()
    assert sample.enqueue_time is None
    assert sample.validation_pause_before == 0.0
    assert sample.checkpoint_pause_before == 0.0


def test_process_single_sample_stamps_virtual_timeline_fields():
    rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    rollouter.current_param_version = 3
    rollouter.cumulative_validation_time = 7.0
    rollouter.cumulative_checkpoint_pause = 2.0
    put_samples = _attach_streaming_stubs(rollouter)

    before = time.time()
    asyncio.run(rollouter._process_single_sample_streaming(_new_rollout_sample()))
    assert len(put_samples) == 1
    stamped = put_samples[0]
    assert before <= stamped.enqueue_time <= time.time()
    assert stamped.validation_pause_before == 7.0
    assert stamped.checkpoint_pause_before == 2.0


def test_stamps_track_the_live_pause_counters():
    """A sample generated after a validation must carry the *new* total, so the
    trainer shifts its virtual arrival back by the pause it actually suffered."""
    rollouter = _make_rollouter(first_sample_time=time.time(), test_freq=1)
    rollouter.cumulative_checkpoint_pause = 0.0
    validate_client = rollouter.message_queue_client

    put_samples = _attach_streaming_stubs(rollouter)
    asyncio.run(rollouter._process_single_sample_streaming(_new_rollout_sample()))
    assert put_samples[-1].validation_pause_before == 0.0

    # a validation runs (accumulating >= 0.03s), then another sample is enqueued
    rollouter.message_queue_client = validate_client
    asyncio.run(rollouter.update_param_version(version=1))
    accumulated = rollouter.cumulative_validation_time
    assert accumulated >= 0.03

    put_samples = _attach_streaming_stubs(rollouter)
    rollouter.cumulative_checkpoint_pause = 1.5
    asyncio.run(rollouter._process_single_sample_streaming(_new_rollout_sample()))
    assert put_samples[-1].validation_pause_before == accumulated
    assert put_samples[-1].checkpoint_pause_before == 1.5


def test_save_checkpoint_does_not_pause_generation_on_this_branch(tmp_path):
    """This branch saves only the dataloader (no queue snapshot), so an ordinary
    save must not pause generation nor add to the checkpoint-pause accumulator;
    the whole checkpoint pause comes from pause_generation_during_save."""
    rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    rollouter.dataloader_lock = asyncio.Lock()
    rollouter.train_dataloader = type("DL", (), {"state_dict": lambda self: {}})()
    rollouter.first_sample_time = 100.0
    rollouter.cumulative_checkpoint_pause = 0.0
    rollouter.checkpointing = False
    rollouter.pause_calls = 0
    rollouter.resume_calls = 0

    async def pause():
        rollouter.pause_calls += 1

    async def resume():
        rollouter.resume_calls += 1

    rollouter.pause = pause
    rollouter.resume = resume

    asyncio.run(rollouter.save_checkpoint(str(tmp_path / "ckpt")))
    assert rollouter.pause_calls == 0 and rollouter.resume_calls == 0
    assert rollouter.checkpointing is False
    assert rollouter.cumulative_checkpoint_pause == 0.0


# ---------------------------------------------------------------- trainer side


def test_trainer_accumulates_save_time():
    trainer = _make_trainer()
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})
    trainer._save_checkpoint = lambda **_: time.sleep(0.02)

    trainer.current_param_version = 1
    trainer._check_save_checkpoint(timing_raw={})
    first_total = trainer.cumulative_save_time
    assert first_total >= 0.02

    trainer.current_param_version = 2
    trainer._check_save_checkpoint(timing_raw={})
    assert trainer.cumulative_save_time >= first_total + 0.02


def test_trainer_save_time_counts_each_save_once_on_shared_timing_raw():
    # marked_timer ACCUMULATES into timing_raw, and fit() reuses the last step's
    # timing_raw for the post-loop save: only the delta may be counted.
    trainer = _make_trainer()
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})
    trainer._save_checkpoint = lambda **_: time.sleep(0.02)
    timing_raw = {}

    trainer.current_param_version = 1
    trainer._check_save_checkpoint(timing_raw)
    after_first = trainer.cumulative_save_time

    trainer.current_param_version = 2
    trainer._check_save_checkpoint(timing_raw)

    # the timer's running total is now both saves; the counter must be too, once each
    assert timing_raw["save_checkpoint"] >= 0.04
    assert abs(trainer.cumulative_save_time - timing_raw["save_checkpoint"]) < 1e-6
    assert trainer.cumulative_save_time < 2 * after_first + 0.02, "the first save must not be double-counted"


def test_check_save_checkpoint_also_feeds_the_step_save_exclusion():
    trainer = _make_trainer()
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})
    trainer._save_checkpoint = lambda **_: time.sleep(0.02)
    trainer.current_param_version = 1
    trainer._check_save_checkpoint(timing_raw={})
    assert trainer._step_save_time >= 0.02
    assert abs(trainer._step_save_time - trainer.cumulative_save_time) < 1e-6


def test_check_save_checkpoint_keeps_the_save_out_of_the_virtual_clock():
    """End-to-end over the production hooks: a save inside an open step must not
    advance the virtual clock, even though it advances the wall clock."""
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})
    trainer._save_checkpoint = lambda **_: time.sleep(0.1)
    trainer.current_param_version = 1

    start = time.time()
    trainer._open_virtual_step(start, [_sample(start)])
    trainer._check_save_checkpoint(timing_raw={})
    trainer._advance_virtual_clock()

    wall_elapsed = time.time() - start
    virtual_elapsed = trainer.virtual_free_time - start
    assert wall_elapsed >= 0.1
    assert virtual_elapsed < 0.05, "the save must be excluded from the virtual busy time"


def test_trainer_save_time_untouched_when_saving_disabled():
    trainer = _make_trainer()
    trainer.config = OmegaConf.create({"trainer": {"save_freq": -1, "esi_redundant_time": 0}})
    trainer.current_param_version = 1
    trainer._check_save_checkpoint(timing_raw={})
    assert trainer.cumulative_save_time == 0.0


def test_trainer_caches_rollouter_timing_from_validate_metrics():
    payload_with_values = ray.cloudpickle.dumps(
        ValidateMetrics(timing_raw={}, param_version=1, first_sample_time=100.0, cumulative_validation_time=5.0)
    )
    payload_without_values = ray.cloudpickle.dumps(ValidateMetrics(timing_raw={}, param_version=2))

    trainer = _make_trainer()
    trainer.message_queue_client = _StubMessageQueueClient(val_payloads=[payload_with_values, payload_without_values])
    trainer._log_validation_data()

    assert trainer.rollouter_first_sample_time == 100.0
    assert trainer.rollouter_cumulative_validation_time == 5.0, "None fields must not clobber cached values"
    # both payloads must have been drained in one call
    assert trainer.message_queue_client.get_validate_sync() is None


def test_add_cumulative_time_metrics_math():
    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)

    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 50.0
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 5.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 2.0
    # no batch has opened the virtual clock yet -> no exact metric
    assert TIMING_PREFIX + "cumulative_training_time" not in step_data

    # mid-step: started at virtual 120 / actual 130, 5s stalled on wait_last_valid
    trainer._step_virtual_start = 120.0
    trainer._step_actual_start = 130.0
    trainer._step_wait_valid_time = 5.0
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 120.0 + (150.0 - 130.0) - 5.0 - 100.0


def test_add_cumulative_time_metrics_noop_without_anchor():
    trainer = _make_trainer(first_sample_time=None)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data == {}


# ---------------------------------------------------------------- resume (timing_state.json)


def test_timing_state_checkpoint_roundtrip(tmp_path):
    saver = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    # mid-step save: virtual 120 / actual 130, 5s stalled on wait_last_valid
    saver._step_virtual_start = 120.0
    saver._step_actual_start = 130.0
    saver._step_wait_valid_time = 5.0
    saver._save_timing_state(str(tmp_path), save_start=150.0)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state == {
        "run_start_datetime": saver.run_start_datetime,
        "checkpoint_datetime": _format_datetime(150.0),
        "wall_time_since_first_sample": 50.0,
        "cumulative_validation_time": 5.0,
        "cumulative_save_time": 2.0,
        "cumulative_training_time": 35.0,  # virtual: 120 + (150-130) - 5 - 100
    }

    resumed = _make_trainer()
    resumed._restore_timing_state(str(tmp_path))
    assert resumed.timing_wall_offset == 50.0
    assert resumed.timing_validation_offset == 5.0
    assert resumed.timing_save_offset == 2.0
    assert resumed.virtual_training_time_offset == 35.0

    # resumed segment: 30s wall, 4s validation, 1s saving; a step opened at
    # virtual 1010 / actual 1012 -> every metric continues from the totals.
    resumed.rollouter_first_sample_time = 1000.0
    resumed.rollouter_cumulative_validation_time = 4.0
    resumed.cumulative_save_time = 1.0
    resumed._step_virtual_start = 1010.0
    resumed._step_actual_start = 1012.0
    step_data = {}
    resumed._add_cumulative_time_metrics(step_data, now=1030.0)
    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 80.0
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 9.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 3.0
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == (1010.0 + 18.0 - 1000.0) + 35.0


def test_save_checkpoint_inner_writes_timing_state_next_to_the_checkpoint(tmp_path, monkeypatch):
    """Through the real save path: timing_state.json must land in the step
    folder, and it must be written after the rollouter created that folder."""
    import recipe.fully_async_policy.fully_async_trainer as fat_module

    class _FakeRay:
        @staticmethod
        def get(ref):
            return ref() if callable(ref) else ref

    monkeypatch.setattr(fat_module, "ray", _FakeRay)

    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    trainer.current_param_version = 7
    trainer.use_critic = False
    trainer.config = OmegaConf.create(
        {"trainer": {"default_local_dir": str(tmp_path), "default_hdfs_dir": None, "max_actor_ckpt_to_keep": 1}}
    )
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=lambda *a, **kw: None)

    step_folder = tmp_path / "global_step_7"

    def rollouter_save(folder):
        # mirrors the production rollouter: it is what creates the folder
        def run():
            os.makedirs(folder, exist_ok=True)

        return run

    trainer.param_synchronizer = SimpleNamespace(rollouter_save_checkpoint=SimpleNamespace(remote=rollouter_save))

    # between steps, the last step ended at virtual 130
    trainer.virtual_free_time = 130.0
    before = time.time()
    trainer._save_checkpoint_inner()
    after = time.time()

    state = json.loads((step_folder / "timing_state.json").read_text())
    # the save stamps itself with the real clock, inside the window of the call
    assert state["run_start_datetime"] == trainer.run_start_datetime
    checkpoint_time = datetime.fromisoformat(state["checkpoint_datetime"]).timestamp()
    assert int(before) <= checkpoint_time <= int(after) + 1
    assert state["cumulative_validation_time"] == 5.0
    assert state["cumulative_save_time"] == 2.0
    assert state["cumulative_training_time"] == 30.0  # 130 - 100
    assert state["wall_time_since_first_sample"] > 0
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "7"


def test_timing_state_save_carries_offsets_forward_without_anchor(tmp_path):
    # A resumed run may checkpoint again before the rollouter reports its first
    # sample; the previous run's totals must pass through unchanged.
    trainer = _make_trainer(first_sample_time=None)
    trainer.timing_wall_offset = 50.0
    trainer.timing_validation_offset = 5.0
    trainer.timing_save_offset = 2.0
    trainer.virtual_training_time_offset = 35.0
    trainer._save_timing_state(str(tmp_path), save_start=999.0)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["wall_time_since_first_sample"] == 50.0
    assert state["cumulative_validation_time"] == 5.0
    assert state["cumulative_save_time"] == 2.0
    assert state["cumulative_training_time"] == 35.0


def test_timing_state_second_resume_chains_totals(tmp_path):
    # run 1 -> checkpoint -> run 2 (with offsets) -> checkpoint -> run 3:
    # totals must chain across multiple restarts, not just one.
    run2 = _make_trainer(first_sample_time=200.0, cumulative_validation_time=3.0, cumulative_save_time=1.0)
    run2.timing_wall_offset = 50.0
    run2.timing_validation_offset = 5.0
    run2.timing_save_offset = 2.0
    run2.virtual_training_time_offset = 35.0
    run2.virtual_free_time = 230.0  # between steps: last step ended at virtual 230
    run2._save_timing_state(str(tmp_path), save_start=240.0)

    run3 = _make_trainer()
    run3._restore_timing_state(str(tmp_path))
    assert run3.timing_wall_offset == 90.0
    assert run3.timing_validation_offset == 8.0
    assert run3.timing_save_offset == 3.0
    assert run3.virtual_training_time_offset == (230.0 - 200.0) + 35.0


def test_restore_timing_state_missing_file_keeps_zero_offsets(tmp_path):
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.timing_wall_offset == 0.0
    assert trainer.timing_validation_offset == 0.0
    assert trainer.timing_save_offset == 0.0
    assert trainer.virtual_training_time_offset == 0.0


def test_restore_timing_state_old_format_falls_back_to_naive(tmp_path):
    # Checkpoints written before the virtual-clock metric may lack the
    # cumulative_training_time key entirely: fall back to wall - val - save.
    (tmp_path / "timing_state.json").write_text(
        json.dumps(
            {"wall_time_since_first_sample": 50.0, "cumulative_validation_time": 5.0, "cumulative_save_time": 2.0}
        )
    )
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.virtual_training_time_offset == 43.0


def test_timing_state_records_run_start_and_checkpoint_datetimes(tmp_path):
    """The two absolute stamps: the run's start verbatim, and the save's own instant.

    checkpoint_datetime must come from save_start - the instant the durations above it are
    snapshotted at - not from "now", or it would drift by the duration of the save itself.
    """
    trainer = _make_trainer(first_sample_time=100.0, run_start_datetime="2026-08-23T00:20:00+03:00")
    trainer._save_timing_state(str(tmp_path), save_start=150.0)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["run_start_datetime"] == "2026-08-23T00:20:00+03:00"
    assert state["checkpoint_datetime"] == _format_datetime(150.0)


def test_timing_state_datetimes_are_timezone_aware_and_parse_back(tmp_path):
    """A file written on one machine must be unambiguous when read on another."""
    trainer = _make_trainer(first_sample_time=100.0)
    save_start = time.time()
    trainer._save_timing_state(str(tmp_path), save_start=save_start)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    for key in ("run_start_datetime", "checkpoint_datetime"):
        parsed = datetime.fromisoformat(state[key])
        assert parsed.tzinfo is not None, key
    # written with timespec="seconds", so the instant survives to the second
    assert datetime.fromisoformat(state["checkpoint_datetime"]).timestamp() == int(save_start)


def test_run_start_datetime_survives_resumes(tmp_path):
    """It answers "when did this run start", so a restart must not reset it."""
    run1 = _make_trainer(first_sample_time=100.0, run_start_datetime="2026-08-23T00:20:00+03:00")
    run1._save_timing_state(str(tmp_path), save_start=150.0)

    run2 = _make_trainer(run_start_datetime="2026-08-24T09:00:00+03:00")
    run2._restore_timing_state(str(tmp_path))
    assert run2.run_start_datetime == "2026-08-23T00:20:00+03:00"

    # ... and the next checkpoint of the resumed run reports the original start again
    run2.rollouter_first_sample_time = 1000.0
    run2._save_timing_state(str(tmp_path), save_start=1100.0)
    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["run_start_datetime"] == "2026-08-23T00:20:00+03:00"
    assert state["checkpoint_datetime"] == _format_datetime(1100.0)


def test_restore_timing_state_without_the_datetime_keys_keeps_this_process_start(tmp_path):
    # checkpoints written before these keys existed
    (tmp_path / "timing_state.json").write_text(
        json.dumps(
            {"wall_time_since_first_sample": 50.0, "cumulative_validation_time": 5.0, "cumulative_save_time": 2.0}
        )
    )
    trainer = _make_trainer(run_start_datetime="2026-08-24T09:00:00+03:00")
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.run_start_datetime == "2026-08-24T09:00:00+03:00"


def test_saved_clock_matches_where_the_clock_lands_after_the_save(tmp_path, monkeypatch):
    """The number written into timing_state.json must not sit ahead of the virtual clock.

    The snapshot is taken when the pipeline freezes and the close-out subtracts the whole
    frozen window; if the snapshot instead used a timestamp taken AFTER the rollouter pause
    RPC (as it did), it counted the pause as training time that the close-out then removed -
    so a later checkpoint could report a smaller cumulative_training_time than an earlier one.
    """
    import recipe.fully_async_policy.fully_async_trainer as fat_module

    class _FakeRay:
        @staticmethod
        def get(ref):
            return ref() if callable(ref) else ref

    monkeypatch.setattr(fat_module, "ray", _FakeRay)

    trainer = _make_trainer(first_sample_time=time.time() - 60.0)
    trainer.current_param_version = 1
    trainer.use_critic = False
    trainer.pause_generation_during_save = True
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "save_freq": 1,
                "esi_redundant_time": 0,
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "max_actor_ckpt_to_keep": None,
                "max_critic_ckpt_to_keep": None,
            }
        }
    )
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=lambda *a, **kw: None)
    trainer.param_synchronizer = SimpleNamespace(
        pause_rollouter_for_save=SimpleNamespace(remote=lambda: lambda: time.sleep(0.05)),
        resume_rollouter_after_save=SimpleNamespace(remote=lambda: lambda: None),
        rollouter_save_checkpoint=SimpleNamespace(remote=lambda folder: lambda: os.makedirs(folder, exist_ok=True)),
    )
    # a step is open, as it is when _check_save_checkpoint runs inside fit()
    trainer._step_virtual_start = trainer.virtual_free_time = 20.0
    trainer._step_actual_start = time.time() - 10.0

    trainer._check_save_checkpoint({})
    saved = json.loads((tmp_path / "global_step_1" / "timing_state.json").read_text())
    trainer._advance_virtual_clock()

    landed = trainer.virtual_free_time - trainer.rollouter_first_sample_time
    assert saved["cumulative_training_time"] <= landed + 1e-6, (
        f"saved {saved['cumulative_training_time']} is ahead of the clock at {landed}"
    )
    # the snapshot deliberately predates the save it is written during, so the pause shows up
    # in the trainer's running total rather than in the file it just wrote
    assert saved["cumulative_save_time"] == 0.0
    assert trainer.cumulative_save_time >= 0.05, "the pause belongs to the save, not to training"


def test_virtual_clock_never_runs_backwards(tmp_path):
    """Closing a step just after a save subtracts the save duration from a window that ends
    a little later; the total must not dip below where the previous step left it, or a
    cumulative_training_time series would tick backwards between checkpoints."""
    trainer = _make_trainer(first_sample_time=100.0)
    trainer.virtual_free_time = 130.0
    # a step that, measured naively, would close slightly EARLIER than the previous close
    trainer._step_virtual_start = 129.0
    trainer._step_actual_start = 200.0
    trainer._step_save_time = 5.0
    trainer._advance_virtual_clock(now=204.0)  # 129 + (204-200) - 5 = 128.0 < 130.0
    assert trainer.virtual_free_time == 130.0

    # a genuine advance is untouched
    trainer._step_virtual_start = 130.0
    trainer._step_actual_start = 300.0
    trainer._step_save_time = 0.0
    trainer._step_wait_valid_time = 0.0
    trainer._advance_virtual_clock(now=310.0)
    assert trainer.virtual_free_time == 140.0


# ------------------------------------------------- virtual (no-validation) clock


def _sample(enqueue_time, validation_pause_before=0.0, checkpoint_pause_before=0.0):
    return SimpleNamespace(
        enqueue_time=enqueue_time,
        validation_pause_before=validation_pause_before,
        checkpoint_pause_before=checkpoint_pause_before,
    )


def _run_step(trainer, consumer_end, samples, step_end, wait_valid=0.0, save_time=0.0):
    """Drive one trainer step through the production virtual-clock hooks."""
    trainer._step_wait_valid_time = wait_valid
    trainer._step_save_time = save_time
    trainer._open_virtual_step(consumer_end, samples)
    trainer._advance_virtual_clock(now=step_end)
    return trainer.virtual_free_time


def test_virtual_clock_rollout_bound_matches_no_validation_run():
    # Reference run (no validation): batches ready at t=10, 20 (R=10); trainer
    # busy U=6 per step -> steps end at 16 and 26.
    # Run with validation: an 8s validation pauses generation, so batch 2
    # arrives at t=28 stamped (28, pause=8) -> virtual ready 20. The trainer
    # idles 16..28 waiting; the metric must not count that induced wait.
    trainer = _make_trainer(first_sample_time=0.0)
    end1 = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=16.0)
    assert end1 == 16.0
    end2 = _run_step(trainer, consumer_end=28.0, samples=[_sample(28.0, 8.0)], step_end=34.0)
    assert end2 == 26.0, "virtual step 2 must end where the no-validation run ends"

    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=34.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 26.0


def test_virtual_clock_trainer_bound_matches_no_validation_run():
    # Reference run: batches ready at t=6, 12 (R=6); trainer busy U=10 -> steps
    # end at 16 and 26; the trainer is the bottleneck throughout.
    # Run with validation: batch 2 was enqueued at t=12 before an 8s validation
    # (16..24); the trainer trains straight through it from backlog, so wall
    # time is unchanged — and so must the metric be (a naive wall - validation
    # subtraction would wrongly report 26 - 8 = 18 here).
    trainer = _make_trainer(first_sample_time=0.0, cumulative_validation_time=8.0)
    _run_step(trainer, consumer_end=6.0, samples=[_sample(6.0, 0.0)], step_end=16.0)
    end2 = _run_step(trainer, consumer_end=16.0, samples=[_sample(12.0, 0.0)], step_end=26.0)
    assert end2 == 26.0

    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=26.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 26.0


def test_virtual_clock_excludes_wait_last_valid_stall():
    # A 3s wait_last_valid stall inside the step is validation-caused and must
    # not advance the virtual clock: 10s of measured step time -> 7s of busy.
    trainer = _make_trainer(first_sample_time=0.0)
    end = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=20.0, wait_valid=3.0)
    assert end == 17.0


def test_virtual_clock_excludes_checkpoint_save_time():
    # A 5s checkpoint save inside the step is not training and must not advance
    # the virtual clock: 15s of measured step time -> 10s of busy.
    trainer = _make_trainer(first_sample_time=0.0)
    end = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=25.0, save_time=5.0)
    assert end == 20.0


def test_virtual_clock_shifts_ready_times_by_checkpoint_pause():
    # A stop-the-world save froze generation for 4s: the batch that arrived at
    # t=24 would have been ready at t=20 without it.
    trainer = _make_trainer(first_sample_time=0.0)
    _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=14.0)
    end2 = _run_step(
        trainer,
        consumer_end=24.0,
        samples=[_sample(24.0, 0.0, checkpoint_pause_before=4.0)],
        step_end=30.0,
    )
    assert end2 == 26.0


def test_open_virtual_step_takes_last_sample_and_clamps_to_anchor():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 5.0
    # batch ready = max over samples of (enqueue - val pause - ckpt pause)
    #             = max(4, 9, 6) = 9
    samples = [_sample(10.0, 6.0), _sample(11.0, 2.0), _sample(12.0, 5.0, checkpoint_pause_before=1.0)]
    trainer._open_virtual_step(30.0, samples)
    assert trainer._step_virtual_start == 9.0
    assert trainer._step_actual_start == 30.0

    # samples stamped before the checkpoint_pause field existed: pause = 0
    trainer._open_virtual_step(35.0, [SimpleNamespace(enqueue_time=33.0, validation_pause_before=4.0)])
    assert trainer._step_virtual_start == 29.0

    # A stamp whose corrected time predates the run's t0 is clamped to it.
    # virtual_free_time must stay None here: on the very first step nothing else
    # bounds the ready time from below, so an unclamped stamp would make
    # cumulative_training_time negative.
    trainer_anchored = _make_trainer(first_sample_time=100.0)
    trainer_anchored._open_virtual_step(140.0, [_sample(120.0, validation_pause_before=90.0)])
    assert trainer_anchored._step_virtual_start == 100.0, "ready time must not precede the anchor"

    step_data = {}
    trainer_anchored._add_cumulative_time_metrics(step_data, now=140.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] >= 0.0


def test_open_virtual_step_without_stamps_does_not_snap_to_wall_time():
    # An unstamped batch carries no information about the virtual timeline;
    # falling back to the wall clock would jump cumulative_training_time
    # forward by the whole validation/save time accumulated so far.
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 12.0
    trainer._open_virtual_step(90.0, [SimpleNamespace()])
    assert trainer._step_virtual_start == 12.0
    assert trainer._step_actual_start == 90.0

    # before any step has closed there is nothing better than the actual time
    fresh = _make_trainer(first_sample_time=0.0)
    fresh._open_virtual_step(90.0, [SimpleNamespace()])
    assert fresh._step_virtual_start == 90.0


def test_advance_virtual_clock_noop_between_steps():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 26.0
    trainer._advance_virtual_clock(now=99.0)  # no open step: must not move
    assert trainer.virtual_free_time == 26.0


def test_virtual_now_between_steps_is_the_last_step_end():
    trainer = _make_trainer(first_sample_time=0.0)
    assert trainer._virtual_now(50.0) is None, "no step has run yet"
    trainer.virtual_free_time = 26.0
    assert trainer._virtual_now(99.0) == 26.0, "must not drift with wall time between steps"


def test_open_virtual_step_trainer_bound_uses_the_free_time():
    # batch was ready long before the trainer finished the previous step
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 40.0
    trainer._open_virtual_step(45.0, [_sample(12.0)])
    assert trainer._step_virtual_start == 40.0


def test_open_virtual_step_mixes_stamped_and_unstamped_samples():
    # the unstamped sample carries no information; the stamped one decides
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 5.0
    trainer._open_virtual_step(50.0, [SimpleNamespace(), _sample(30.0, validation_pause_before=8.0)])
    assert trainer._step_virtual_start == 22.0


def test_virtual_clock_full_schedule_matches_the_reference_run():
    """Four steps of a rollout-bound pipeline, disturbed by an 8s validation and
    a 5s stop-the-world save, must land exactly on the undisturbed schedule.

    Reference (no validation, no save): batches ready at 10/20/30/40, trainer
    busy 6s per step -> steps end at 16/26/36/46.
    """
    trainer = _make_trainer(first_sample_time=0.0)

    # step 1 & 2: undisturbed
    assert _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0)], step_end=16.0) == 16.0
    assert _run_step(trainer, consumer_end=20.0, samples=[_sample(20.0)], step_end=26.0) == 26.0

    # an 8s validation froze generation: batch 3 lands at 38 instead of 30, and
    # the trainer additionally stalls 3s on wait_last_valid inside the step
    end3 = _run_step(
        trainer,
        consumer_end=38.0,
        samples=[_sample(38.0, validation_pause_before=8.0)],
        step_end=47.0,
        wait_valid=3.0,
    )
    assert end3 == 36.0

    # a 5s stop-the-world save froze generation again: batch 4 lands at 53, and
    # the save itself sits inside the trainer's step
    end4 = _run_step(
        trainer,
        consumer_end=53.0,
        samples=[_sample(53.0, validation_pause_before=8.0, checkpoint_pause_before=5.0)],
        step_end=64.0,
        save_time=5.0,
    )
    assert end4 == 46.0

    trainer.rollouter_cumulative_validation_time = 8.0
    trainer.cumulative_save_time = 5.0
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=64.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 46.0
    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 64.0
    # the naive subtraction would report 64 - 8 - 5 = 51, i.e. 5s too high
    naive = 64.0 - 8.0 - 5.0
    assert naive != 46.0


def test_cumulative_training_time_is_monotonic_and_bounded_by_wall_time():
    trainer = _make_trainer(first_sample_time=0.0)
    schedule = [
        # (consumer_end, sample, step_end, wait_valid, save_time)
        (10.0, _sample(10.0), 16.0, 0.0, 0.0),
        (20.0, _sample(20.0), 26.0, 0.0, 0.0),
        (38.0, _sample(38.0, validation_pause_before=8.0), 47.0, 3.0, 0.0),
        (53.0, _sample(53.0, validation_pause_before=8.0, checkpoint_pause_before=5.0), 64.0, 0.0, 5.0),
    ]
    previous = None
    for consumer_end, sample, step_end, wait_valid, save_time in schedule:
        _run_step(trainer, consumer_end, [sample], step_end, wait_valid=wait_valid, save_time=save_time)
        step_data = {}
        trainer._add_cumulative_time_metrics(step_data, now=step_end)
        current = step_data[TIMING_PREFIX + "cumulative_training_time"]
        wall = step_data[TIMING_PREFIX + "wall_time_since_first_sample"]
        assert current <= wall, "the virtual clock can never run ahead of the wall clock"
        if previous is not None:
            assert current >= previous, "cumulative_training_time must never go backwards"
        previous = current


def test_get_samples_from_queue_opens_the_virtual_step(monkeypatch):
    """The metric is only correct if the production pull actually stamps the
    step; check the wiring, not just _open_virtual_step in isolation."""
    import recipe.fully_async_policy.fully_async_trainer as fat_module

    trainer = _make_trainer(first_sample_time=0.0)
    trainer.required_samples = 2
    trainer.config = OmegaConf.create({"trainer": {"balance_batch": False}})
    trainer.tokenizer = None
    trainer.virtual_free_time = 5.0

    samples = [
        ray.cloudpickle.dumps(_sample(30.0, validation_pause_before=4.0)),
        ray.cloudpickle.dumps(_sample(32.0, validation_pause_before=4.0, checkpoint_pause_before=2.0)),
    ]
    pending = list(samples)

    class _MQ:
        def get_sample_sync(self):
            return (pending.pop(0), len(pending)) if pending else (None, 0)

    trainer.message_queue_client = _MQ()
    monkeypatch.setattr(
        fat_module, "assemble_batch_from_rollout_samples", lambda *a, **kw: SimpleNamespace(meta_info={})
    )

    before = time.time()
    epoch, batch = trainer._get_samples_from_queue()
    assert epoch == 0 and batch is not None
    # batch ready = max(30 - 4, 32 - 4 - 2) = 26; free time 5 -> step opens at 26
    assert trainer._step_virtual_start == 26.0
    assert before <= trainer._step_actual_start <= time.time()


def test_get_samples_from_queue_does_not_open_a_step_on_shutdown(monkeypatch):
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.required_samples = 2
    trainer.config = OmegaConf.create({"trainer": {"balance_batch": False}})

    class _MQ:
        def get_sample_sync(self):
            return None, 0  # termination sentinel

    trainer.message_queue_client = _MQ()

    epoch, batch = trainer._get_samples_from_queue()
    assert (epoch, batch) == (None, None)
    assert trainer._step_virtual_start is None, "a shutdown pull must leave the clock closed"


def test_fit_loop_keeps_the_virtual_clock_wired():
    """fit() is too heavy to drive in a unit test, but its three virtual-clock
    hooks are exactly what makes the metric advance; guard their presence and
    order so they cannot be dropped silently."""
    import inspect

    source = inspect.getsource(FullyAsyncTrainer.fit)
    reset_wait = source.index("self._step_wait_valid_time = 0.0")
    reset_save = source.index("self._step_save_time = 0.0")
    sync = source.index("self._trigger_parameter_sync_after_step(global_steps=self.global_steps)")
    save = source.index("self._check_save_checkpoint(timing_raw)")
    advance = source.index("self._advance_virtual_clock()")

    # the per-step exclusion counters are reset before the step can add to them
    assert reset_wait < sync
    assert reset_save < save
    # the step is closed on the virtual timeline only after the save is accounted
    assert save < advance


# ------------------------------------------------------------------- aggregation


def test_timing_metrics_are_classified_as_last_not_min():
    """The four cumulative counters are added AFTER aggregation, but the rules must be
    right anyway: 'timing' contains the substring 'min', so name-based matching used to
    classify every fully_async/timing/* metric as a minimum."""
    from recipe.fully_async_policy.detach_utils import MetricsAggregator

    aggregator = MetricsAggregator(total_gpus=8)
    for key in (
        "fully_async/timing/cumulative_training_time",
        "fully_async/timing/wall_time_since_first_sample",
        "fully_async/timing/cumulative_validation_time",
        "fully_async/timing/cumulative_save_time",
    ):
        assert aggregator._get_aggregation_type(key) == "last", key

    # the same substring flaw hit every per-token timing metric
    assert aggregator._get_aggregation_type("timing_per_token_ms/gen") == "avg"
    # and the rules that were already right must stay right
    assert aggregator._get_aggregation_type("timing_s/step") == "time_sum"
    assert aggregator._get_aggregation_type("global_seqlen/min") == "min"
    assert aggregator._get_aggregation_type("global_seqlen/max") == "max"
    assert aggregator._get_aggregation_type("response_length/mean") == "avg"
    assert aggregator._get_aggregation_type("perf/total_num_tokens") == "sum"
    assert aggregator._get_aggregation_type("fully_async/count/total_generated_samples") == "last"


def test_timing_metrics_are_last_value_if_routed_through_the_aggregator():
    from recipe.fully_async_policy.detach_utils import MetricsAggregator

    aggregator = MetricsAggregator(total_gpus=8)
    for value in (100.0, 200.0, 300.0):
        aggregator.add_step_metrics(
            {"fully_async/timing/cumulative_training_time": value, "actor/pg_loss": value}, sample_count=33
        )
    aggregated = aggregator.get_aggregated_metrics()
    assert aggregated["fully_async/timing/cumulative_training_time"] == 300.0, "a counter must not be averaged"
    assert aggregated["actor/pg_loss"] == 200.0, "an ordinary metric is still averaged"


def test_sync_logs_instantaneous_timing_alongside_aggregated_step_metrics(monkeypatch):
    """End-to-end through the production sync path with a real MetricsAggregator: the
    per-step metrics are aggregated, the timing counters are not."""
    import recipe.fully_async_policy.fully_async_trainer as fat_module
    from recipe.fully_async_policy.detach_utils import MetricsAggregator

    class _FakeRay:
        @staticmethod
        def get(ref):
            return ref() if callable(ref) else ref

    monkeypatch.setattr(fat_module, "ray", _FakeRay)

    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    trainer.local_trigger_step = 1
    trainer.trigger_parameter_sync_step = 1
    trainer.serialize_validation = False
    trainer.config = OmegaConf.create({"trainer": {"logger": ["console"]}, "rollout": {"test_freq": 0}})
    trainer.metrics_aggregator = MetricsAggregator(total_gpus=8)
    trainer.progress_bar = SimpleNamespace(update=lambda n: None)
    trainer._log_validation_data = lambda: None
    trainer.param_synchronizer = SimpleNamespace(
        wait_last_valid=SimpleNamespace(remote=lambda: lambda: None),
        sync_weights=SimpleNamespace(remote=lambda version, validate=False, global_steps=None: lambda: None),
    )
    # two trainer steps feed the aggregator before the sync
    for value in (1.0, 3.0):
        trainer.metrics_aggregator.add_step_metrics({"actor/pg_loss": value}, sample_count=33)
    # a step is open on the virtual clock: started at virtual 120 / actual 130
    trainer._step_virtual_start = 120.0
    trainer._step_actual_start = 130.0
    trainer._step_wait_valid_time = 0.0

    logged = []
    trainer.logger = SimpleNamespace(log=lambda data, step: logged.append((data, step)))
    trainer._trigger_parameter_sync_after_step(global_steps=1)

    step_data = logged[0][0]
    assert step_data["actor/pg_loss"] == 2.0, "per-step metrics stay averaged"
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 5.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 2.0
    # instantaneous, not an average over the two steps
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] > 20.0
    assert (
        step_data[TIMING_PREFIX + "wall_time_since_first_sample"]
        > step_data[TIMING_PREFIX + "cumulative_training_time"]
    )


def test_log_validation_data_is_a_noop_on_an_empty_queue():
    trainer = _make_trainer()
    trainer.message_queue_client = _StubMessageQueueClient(val_payloads=[])
    trainer._log_validation_data()
    assert trainer.logger.logged == []
    assert trainer.rollouter_first_sample_time is None


def test_log_validation_data_logs_every_drained_payload():
    payloads = [
        ray.cloudpickle.dumps(
            ValidateMetrics(
                timing_raw={"rollouter/validate_time": 1.0},
                metrics={"val-core/acc": 0.5},
                param_version=1,
                first_sample_time=100.0,
                cumulative_validation_time=1.0,
            )
        ),
        ray.cloudpickle.dumps(
            ValidateMetrics(
                timing_raw={"rollouter/validate_time": 2.0},
                metrics={"val-core/acc": 0.6},
                param_version=2,
                first_sample_time=100.0,
                cumulative_validation_time=3.0,
            )
        ),
    ]
    trainer = _make_trainer()
    trainer.message_queue_client = _StubMessageQueueClient(val_payloads=payloads)
    trainer._log_validation_data()

    steps = [step for _, step in trainer.logger.logged]
    assert steps == [1, 1, 2, 2], "each payload logs its metrics and its timing at its own version"
    # the newest payload wins for the cached bookkeeping
    assert trainer.rollouter_cumulative_validation_time == 3.0


# ------------------------------------- stop-the-world identity / sync-arm parity
#
# The is-pg baseline scripts run with serialize_validation=True and
# pause_generation_during_save=True: validation and checkpoint saves freeze the
# whole pipeline, so they are pure time translations and the clean training time
# must equal wall - validation - save exactly -- the same identity the sync
# trainer's compute_cumulative_timing_metrics holds by construction. These tests
# drive the production virtual-clock hooks through one such schedule and check
# the four fully_async/timing/* totals against that identity and against the
# sync helper fed the same per-iteration durations.
#
# The schedule (rollout-bound: a batch every 10 s of generation, 6 s of training):
#   step 1: batch 1 ready t=10, trains 10..16; the sync at 16 triggers a
#           serialized validation 16..24 (generation paused, trainer blocked on
#           wait_last_valid inside the step) -> step 1 closes at 24
#   step 2: batch 2 would have been ready at 20, lands at 28 (stamped pause 8),
#           trains 28..34; a 5 s stop-the-world save 34..39 closes the step
#   step 3: batch 3 would have been ready at 30, lands at 43 (stamped 8 + 5),
#           trains 43..49
# No-pause reference run: steps end at 16, 26, 36 -> clean training time 36 s.
STW_WALL, STW_VALIDATION, STW_SAVE, STW_TRAINING = 49.0, 8.0, 5.0, 36.0


def _stop_the_world_schedule(trainer, probe=None):
    """Drive the schedule above; ``probe(trainer, now)`` is called after each closed step."""
    _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0)], step_end=24.0, wait_valid=8.0)
    trainer.rollouter_cumulative_validation_time = 8.0  # what ValidateMetrics carries after the pause
    if probe:
        probe(trainer, 24.0)
    _run_step(trainer, 28.0, [_sample(28.0, validation_pause_before=8.0)], step_end=39.0, save_time=5.0)
    trainer.cumulative_save_time = 5.0  # what _check_save_checkpoint accumulates
    if probe:
        probe(trainer, 39.0)
    _run_step(trainer, 43.0, [_sample(43.0, validation_pause_before=8.0, checkpoint_pause_before=5.0)], 49.0)
    if probe:
        probe(trainer, 49.0)
    return 49.0


def _timing(trainer, now):
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=now)
    return {k[len(TIMING_PREFIX) :]: v for k, v in step_data.items() if k.startswith(TIMING_PREFIX)}


def test_stop_the_world_identity_training_equals_wall_minus_validation_minus_save():
    trainer = _make_trainer(first_sample_time=0.0)
    now = _stop_the_world_schedule(trainer)
    t = _timing(trainer, now)
    assert t["wall_time_since_first_sample"] == STW_WALL
    assert t["cumulative_validation_time"] == STW_VALIDATION
    assert t["cumulative_save_time"] == STW_SAVE
    assert t["cumulative_training_time"] == STW_TRAINING
    assert abs(t["cumulative_training_time"] - (STW_WALL - STW_VALIDATION - STW_SAVE)) < 1e-6


def test_stop_the_world_identity_holds_at_every_step_boundary():
    """Not just at the end: the per-step logged series must be consistent with the
    three component series at every point where a step closes."""
    seen = []

    def probe(trainer, now):
        t = _timing(trainer, now)
        seen.append(t["cumulative_training_time"])
        assert (
            abs(
                t["cumulative_training_time"]
                - (t["wall_time_since_first_sample"] - t["cumulative_validation_time"] - t["cumulative_save_time"])
            )
            < 1e-6
        ), (now, t)

    _stop_the_world_schedule(_make_trainer(first_sample_time=0.0), probe=probe)
    assert seen == [16.0, 26.0, 36.0], "the no-validation-no-save run's step ends"


def test_stop_the_world_totals_match_the_sync_helper():
    """The sync trainer's compute_cumulative_timing_metrics, fed the same schedule as
    per-iteration durations, must report the same four totals -- the property that
    lets sync and async arms share one val-vs-clean-time axis. In the sync trainer
    generation is part of the step, so each sync 'step' is (generation not hidden
    by a pause) + training: 10+6, 4+6 and 4+6 seconds here."""
    from verl.trainer.ppo.metric_utils import compute_cumulative_timing_metrics

    trainer = _make_trainer(first_sample_time=0.0)
    async_totals = _timing(trainer, _stop_the_world_schedule(trainer))

    state = {}
    compute_cumulative_timing_metrics(state, {"step": 16.0, "testing": 8.0})
    compute_cumulative_timing_metrics(state, {"step": 10.0, "save_checkpoint": 5.0})
    sync_out = compute_cumulative_timing_metrics(state, {"step": 10.0})
    sync_totals = {k[len(TIMING_PREFIX) :]: v for k, v in sync_out.items()}

    assert (
        sync_totals
        == async_totals
        == {
            "wall_time_since_first_sample": STW_WALL,
            "cumulative_validation_time": STW_VALIDATION,
            "cumulative_save_time": STW_SAVE,
            "cumulative_training_time": STW_TRAINING,
        }
    )


def test_overlapped_mode_does_not_claim_the_identity():
    """Without serialize_validation the trainer keeps training from backlog during a
    validation (trainer-bound regime): wall time is unchanged by the validation, so
    wall - validation under-counts and the virtual clock is the only correct source.
    The metric must come from the virtual clock, not from the subtraction."""
    trainer = _make_trainer(first_sample_time=0.0, cumulative_validation_time=8.0)
    _run_step(trainer, consumer_end=6.0, samples=[_sample(6.0)], step_end=16.0)
    _run_step(trainer, consumer_end=16.0, samples=[_sample(12.0)], step_end=26.0)
    t = _timing(trainer, 26.0)
    assert t["cumulative_training_time"] == 26.0
    assert t["wall_time_since_first_sample"] - t["cumulative_validation_time"] == 18.0  # the naive, wrong answer
