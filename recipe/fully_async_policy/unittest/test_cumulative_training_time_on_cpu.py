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

Run: pytest recipe/fully_async_policy/unittest/test_cumulative_training_time_on_cpu.py
"""

import asyncio
import time

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


def _make_trainer(first_sample_time=None, cumulative_validation_time=0.0, cumulative_save_time=0.0):
    """Minimal FullyAsyncTrainer with only the attributes under test."""
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.rollouter_first_sample_time = first_sample_time
    t.rollouter_cumulative_validation_time = cumulative_validation_time
    t.cumulative_save_time = cumulative_save_time
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
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 43.0


def test_add_cumulative_time_metrics_noop_without_anchor():
    trainer = _make_trainer(first_sample_time=None)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data == {}
