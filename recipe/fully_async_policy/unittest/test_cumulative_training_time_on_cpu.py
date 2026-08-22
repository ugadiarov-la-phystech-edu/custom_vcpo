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
"""Unit tests for the fully_async/timing/cumulative_training_time metric:
- ValidateMetrics carries the rollouter timing fields through (cloud)pickle
- FullyAsyncRollouter anchors first_sample_time once and accumulates validation
  time only after the anchor exists
- FullyAsyncTrainer accumulates checkpoint-save time, caches the rollouter
  values from ValidateMetrics, and computes the final metric
- timing state survives a checkpoint round-trip (timing_state.json), so a
  resumed run continues cumulative_training_time instead of restarting at zero

Run: pytest recipe/fully_async_policy/unittest/test_cumulative_training_time_on_cpu.py
"""

import asyncio
import json
import time
from datetime import datetime

import ray.cloudpickle
from omegaconf import OmegaConf

from recipe.fully_async_policy.detach_utils import ValidateMetrics
from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor


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


_UNSET = object()


def _fmt(epoch: float) -> str:
    """The local-time rendering _save_timing_state writes for an epoch."""
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _make_trainer(
    first_sample_time=None,
    cumulative_validation_time=0.0,
    cumulative_save_time=0.0,
    run_first_sample_time=_UNSET,
):
    """Minimal FullyAsyncTrainer with only the attributes under test.

    ``run_first_sample_time`` is the chained origin; it defaults to the
    segment anchor, which is what a fresh run produces once its first sample
    arrives. Pass it explicitly to model a resume (origin restored from the
    checkpoint) or None to model a run that has seen no sample yet."""
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.rollouter_first_sample_time = first_sample_time
    t.run_first_sample_time = first_sample_time if run_first_sample_time is _UNSET else run_first_sample_time
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


# ---------------------------------------------------------------- trainer side


def test_trainer_accumulates_save_time():
    trainer = _make_trainer()
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})
    trainer._save_checkpoint = lambda: time.sleep(0.02)

    trainer.current_param_version = 1
    trainer._check_save_checkpoint(timing_raw={})
    first_total = trainer.cumulative_save_time
    assert first_total >= 0.02

    trainer.current_param_version = 2
    trainer._check_save_checkpoint(timing_raw={})
    assert trainer.cumulative_save_time >= first_total + 0.02


def test_save_time_is_not_double_counted_when_timing_raw_is_reused():
    """marked_timer ACCUMULATES into timing_raw, and the post-loop final save
    reuses the last iteration's dict. Reading the running total instead of this
    save's delta would add S_prev + S_final to both counters."""
    trainer = _make_trainer()
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 1, "esi_redundant_time": 0}})
    trainer._save_checkpoint = lambda: time.sleep(0.02)

    # a dict that already carries an earlier save from the same iteration
    timing_raw = {"save_checkpoint": 5.0}
    trainer.current_param_version = 1
    trainer._check_save_checkpoint(timing_raw=timing_raw)

    assert trainer.cumulative_save_time < 1.0, "the pre-existing 5.0 must not be re-counted"
    assert trainer.cumulative_save_time >= 0.02
    assert trainer._step_save_time < 1.0
    assert timing_raw["save_checkpoint"] >= 5.02, "the timer itself still accumulates"


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
        "wall_time_since_first_sample": 50.0,
        "cumulative_validation_time": 5.0,
        "cumulative_save_time": 2.0,
        "cumulative_training_time": 35.0,  # virtual: 120 + (150-130) - 5 - 100
        # wall-clock anchors: the raw origin epoch (what a resume reads back)
        # plus both moments rendered in local time for a human reader
        "first_sample_time": 100.0,
        "first_sample_datetime": _fmt(100.0),
        "checkpoint_saved_datetime": _fmt(150.0),
    }

    resumed = _make_trainer()
    resumed._restore_timing_state(str(tmp_path))
    assert resumed.run_first_sample_time == 100.0, "the origin travels with the checkpoint"
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
    # no sample in this segment and none inherited -> no origin to report,
    # but the save moment is always stamped
    assert state["first_sample_time"] is None
    assert state["first_sample_datetime"] is None
    assert state["checkpoint_saved_datetime"] == _fmt(999.0)


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


def test_origin_survives_a_resume_and_a_new_segment_anchor(tmp_path):
    """The point of chaining: after a resume, first_sample_datetime still names
    when the RUN began, not when this segment did. rollouter_first_sample_time
    re-anchors (every duration is measured from it) — the origin must not."""
    run1 = _make_trainer(first_sample_time=100.0)
    run1._save_timing_state(str(tmp_path), save_start=150.0)

    run2 = _make_trainer(first_sample_time=None, run_first_sample_time=None)
    run2._restore_timing_state(str(tmp_path))
    assert run2.run_first_sample_time == 100.0

    # this segment's rollouter reports its own, much later, first sample
    run2.message_queue_client = _StubMessageQueueClient(
        val_payloads=[ray.cloudpickle.dumps(ValidateMetrics(timing_raw={}, param_version=1, first_sample_time=9000.0))]
    )
    run2._log_validation_data()
    assert run2.rollouter_first_sample_time == 9000.0, "durations still measure from this segment"
    assert run2.run_first_sample_time == 100.0, "but the origin does not move"

    run2._save_timing_state(str(tmp_path), save_start=9100.0)
    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["first_sample_time"] == 100.0
    assert state["first_sample_datetime"] == _fmt(100.0)
    assert state["checkpoint_saved_datetime"] == _fmt(9100.0), "the save moment is this segment's"


def test_origin_is_set_once_within_a_fresh_run():
    """A fresh run adopts the first anchor it sees and keeps it, mirroring the
    rollouter's own set-once behaviour."""
    trainer = _make_trainer(first_sample_time=None, run_first_sample_time=None)
    trainer.message_queue_client = _StubMessageQueueClient(
        val_payloads=[
            ray.cloudpickle.dumps(ValidateMetrics(timing_raw={}, param_version=1, first_sample_time=100.0)),
            ray.cloudpickle.dumps(ValidateMetrics(timing_raw={}, param_version=2, first_sample_time=777.0)),
        ]
    )
    trainer._log_validation_data()
    assert trainer.run_first_sample_time == 100.0
    assert trainer.rollouter_first_sample_time == 777.0


def test_origin_ignores_none_first_sample_time():
    """A ValidateMetrics with no anchor must not blank the origin."""
    trainer = _make_trainer(first_sample_time=100.0)
    trainer.message_queue_client = _StubMessageQueueClient(
        val_payloads=[ray.cloudpickle.dumps(ValidateMetrics(timing_raw={}, param_version=1))]
    )
    trainer._log_validation_data()
    assert trainer.run_first_sample_time == 100.0


def test_origin_chains_across_two_resumes(tmp_path):
    """Origin survives resume-of-a-resume, alongside the totals that
    test_timing_state_second_resume_chains_totals pins."""
    run1 = _make_trainer(first_sample_time=100.0)
    run1._save_timing_state(str(tmp_path), save_start=150.0)

    run2 = _make_trainer(first_sample_time=200.0, run_first_sample_time=None)
    run2._restore_timing_state(str(tmp_path))
    run2._save_timing_state(str(tmp_path), save_start=240.0)

    run3 = _make_trainer(run_first_sample_time=None)
    run3._restore_timing_state(str(tmp_path))
    assert run3.run_first_sample_time == 100.0

    run3.rollouter_first_sample_time = 5000.0
    run3._save_timing_state(str(tmp_path), save_start=5050.0)
    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["first_sample_datetime"] == _fmt(100.0)


def test_legacy_checkpoint_without_origin_reanchors_to_this_segment(tmp_path):
    """timing_state.json written before first_sample_time existed: the run
    re-anchors to the current segment rather than raising."""
    (tmp_path / "timing_state.json").write_text(
        json.dumps(
            {
                "wall_time_since_first_sample": 50.0,
                "cumulative_validation_time": 5.0,
                "cumulative_save_time": 2.0,
                "cumulative_training_time": 35.0,
                "first_sample_datetime": "2026-01-01 00:00:00",  # string-only, unusable
                "checkpoint_saved_datetime": "2026-01-01 00:01:00",
            }
        )
    )
    trainer = _make_trainer(run_first_sample_time=None)
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.run_first_sample_time is None
    assert trainer.timing_wall_offset == 50.0  # the durations still restore

    trainer.message_queue_client = _StubMessageQueueClient(
        val_payloads=[ray.cloudpickle.dumps(ValidateMetrics(timing_raw={}, param_version=1, first_sample_time=300.0))]
    )
    trainer._log_validation_data()
    assert trainer.run_first_sample_time == 300.0


def test_legacy_restore_does_not_blank_a_known_origin(tmp_path):
    """Order-independence: the live flow restores before any anchor arrives
    (load_checkpoint is awaited before fit), but a restore that finds no
    first_sample_time must leave an already-known origin alone rather than
    clearing it."""
    (tmp_path / "timing_state.json").write_text(
        json.dumps({"wall_time_since_first_sample": 50.0, "cumulative_training_time": 35.0})
    )
    trainer = _make_trainer(first_sample_time=300.0)  # origin already adopted
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.run_first_sample_time == 300.0
    assert trainer.timing_wall_offset == 50.0


def test_anchors_do_not_disturb_the_duration_totals(tmp_path):
    """The origin is informational: an origin far from this segment's anchor
    must leave every duration exactly where it was."""
    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    trainer.run_first_sample_time = 1.0  # ancient origin from a much earlier run
    trainer._step_virtual_start = 120.0
    trainer._step_actual_start = 130.0
    trainer._step_wait_valid_time = 5.0
    trainer._save_timing_state(str(tmp_path), save_start=150.0)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["wall_time_since_first_sample"] == 50.0
    assert state["cumulative_validation_time"] == 5.0
    assert state["cumulative_save_time"] == 2.0
    assert state["cumulative_training_time"] == 35.0
    assert state["first_sample_time"] == 1.0


def test_checkpoint_saved_datetime_follows_save_start_not_now(tmp_path):
    """Every other value in the file is a snapshot at save_start; the stamp
    must be too, or a slow save would report a time nothing else agrees with."""
    trainer = _make_trainer(first_sample_time=100.0)
    trainer._save_timing_state(str(tmp_path), save_start=150.0)
    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["checkpoint_saved_datetime"] == _fmt(150.0)
    assert state["checkpoint_saved_datetime"] != _fmt(time.time())


def test_restore_timing_state_missing_file_keeps_zero_offsets(tmp_path):
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.timing_wall_offset == 0.0
    assert trainer.timing_validation_offset == 0.0
    assert trainer.timing_save_offset == 0.0
    assert trainer.virtual_training_time_offset == 0.0


# ------------------------------------------------- virtual (no-validation) clock


def _sample(enqueue_time, validation_pause_before=0.0, checkpoint_pause_before=0.0):
    from types import SimpleNamespace

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


def test_open_virtual_step_takes_last_sample_and_handles_missing_stamps():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 5.0
    # batch ready = max over samples of (enqueue - val pause - ckpt pause)
    #             = max(4, 9, 6) = 9
    samples = [_sample(10.0, 6.0), _sample(11.0, 2.0), _sample(12.0, 5.0, checkpoint_pause_before=1.0)]
    trainer._open_virtual_step(30.0, samples)
    assert trainer._step_virtual_start == 9.0
    assert trainer._step_actual_start == 30.0

    # samples stamped before the checkpoint_pause field existed: pause = 0
    from types import SimpleNamespace

    trainer._open_virtual_step(35.0, [SimpleNamespace(enqueue_time=33.0, validation_pause_before=4.0)])
    assert trainer._step_virtual_start == 29.0

    # no usable stamps: the batch imposes no constraint, so the step starts when
    # the TRAINER is free. Falling back to the actual ready time would pin the
    # virtual start to the real clock (see test_all_replay_step_does_not_snap...).
    trainer._open_virtual_step(40.0, [SimpleNamespace()])
    assert trainer._step_virtual_start == 5.0


def test_all_replay_step_does_not_snap_the_virtual_clock_to_wall_time():
    """A mini-batch composed entirely of REPLAYED groups passes no new samples,
    so _open_virtual_step sees an empty list. It must not restart the virtual
    timeline at the actual clock: since actual >= virtual always, that would
    discard every validation/save second excluded so far, and one such step
    after a validation sweep would collapse cumulative_training_time onto wall
    time for the rest of the run."""
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 20.0  # 20s of virtual work done...
    consumer_end = 100.0  # ...but 100s of wall clock, the gap being validation

    trainer._open_virtual_step(consumer_end, [])  # all-replay mini-batch
    assert trainer._step_virtual_start == 20.0, "virtual start must follow the virtual clock"
    assert trainer._step_actual_start == consumer_end

    # a 5s step advances the VIRTUAL clock by 5s, not to 105
    trainer._advance_virtual_clock(consumer_end + 5.0)
    assert abs(trainer.virtual_free_time - 25.0) < 1e-9
    assert abs(trainer._virtual_now(consumer_end + 5.0) - 25.0) < 1e-9


def test_restored_snapshot_stamps_cannot_drag_the_clock_before_this_segment():
    """save_queue_state=True hands the first post-resume batches samples pickled
    in the previous segment: absolute enqueue times minus that segment's
    accumulated pauses. Unclamped, that lands before this segment's anchor,
    makes virtual_now - anchor negative, and drops cumulative_training_time
    below the restored offset — which _save_timing_state then persists."""
    trainer = _make_trainer(first_sample_time=1000.0)
    trainer.timing_wall_offset = 500.0
    trainer.virtual_training_time_offset = 400.0
    trainer.virtual_free_time = None

    # sample enqueued at 990 in the OLD segment, carrying 300s of that segment's
    # validation pause -> naive ready time 690, i.e. 310s before this anchor
    stale = _sample(990.0, 300.0)
    trainer._open_virtual_step(1005.0, [stale])
    assert trainer._step_virtual_start >= 1000.0

    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=1010.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] >= 400.0, "must not fall below the restored offset"


def test_fresh_segment_stamps_are_not_clamped():
    """The clamp must not disturb ordinary in-segment stamps."""
    trainer = _make_trainer(first_sample_time=100.0)
    trainer.virtual_free_time = None
    trainer._open_virtual_step(200.0, [_sample(150.0, 10.0)])  # ready at 140 > anchor
    assert trainer._step_virtual_start == 140.0


def test_first_step_without_stamps_still_starts_at_the_actual_clock():
    """virtual_free_time is None before the first step, so there is nothing to
    fall back to and the actual arrival is the only available anchor."""
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = None
    trainer._open_virtual_step(42.0, [])
    assert trainer._step_virtual_start == 42.0


def test_all_replay_steps_preserve_an_earlier_validation_exclusion():
    """End to end: one stamped step, a validation gap, then an all-replay step.
    The excluded validation time must still be missing from the virtual clock."""
    trainer = _make_trainer(first_sample_time=0.0)
    trainer._open_virtual_step(10.0, [_sample(10.0, 0.0)])  # ready at 10, no pause
    trainer._advance_virtual_clock(12.0)  # 2s of work -> virtual 12
    assert abs(trainer.virtual_free_time - 12.0) < 1e-9

    # 60s of validation happens here: wall clock jumps to 72, virtual does not.
    trainer._open_virtual_step(72.0, [])  # all-replay mini-batch
    trainer._advance_virtual_clock(75.0)  # 3s of work
    assert abs(trainer.virtual_free_time - 15.0) < 1e-9  # 12 + 3, validation excluded
    assert trainer.virtual_free_time < 75.0


def test_rollouter_save_checkpoint_accumulates_pause(tmp_path):
    def make_rollouter(first_sample_time, save_queue_state=True):
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
        r.filtered_degenerate_groups = 0
        r.processed_sample_count = 0
        r.current_param_version = 1
        r.message_queue_client = None

        async def snapshot():
            return {}

        async def pause():
            await asyncio.sleep(0.03)

        async def resume():
            pass

        r._snapshot_internal_queues = snapshot
        r.pause = pause
        r.resume = resume
        return r

    # post-anchor: the pause window (>= the stubbed 0.03s pause) accumulates
    rollouter = make_rollouter(first_sample_time=100.0)
    asyncio.run(rollouter.save_checkpoint(str(tmp_path / "a")))
    first = rollouter.cumulative_checkpoint_pause
    assert first >= 0.03
    asyncio.run(rollouter.save_checkpoint(str(tmp_path / "b")))
    assert rollouter.cumulative_checkpoint_pause >= first + 0.03, "must accumulate across saves"

    # pre-anchor: nothing accumulates
    rollouter = make_rollouter(first_sample_time=None)
    asyncio.run(rollouter.save_checkpoint(str(tmp_path / "c")))
    assert rollouter.cumulative_checkpoint_pause == 0.0

    # save_queue_state=False: no pause happens, nothing accumulates
    rollouter = make_rollouter(first_sample_time=100.0, save_queue_state=False)
    asyncio.run(rollouter.save_checkpoint(str(tmp_path / "d")))
    assert rollouter.cumulative_checkpoint_pause == 0.0


def test_advance_virtual_clock_noop_between_steps():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 26.0
    trainer._advance_virtual_clock(now=99.0)  # no open step: must not move
    assert trainer.virtual_free_time == 26.0


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
