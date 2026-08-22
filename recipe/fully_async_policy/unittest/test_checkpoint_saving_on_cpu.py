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
"""Unit tests for what the trainer does around checkpoint contents:

- max_actor_ckpt_to_keep=null reaches the worker group as None (keep every checkpoint),
  and a set value is forwarded unchanged
- a save still writes the tracker file and snapshots the dataloader/timing state
- a resume whose load_contents cannot restore the model weights is refused instead of
  silently continuing from the pretrained weights (save_contents=['hf_model'])

Run: pytest recipe/fully_async_policy/unittest/test_checkpoint_saving_on_cpu.py
"""

import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import recipe.fully_async_policy.fully_async_trainer as fat_module
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor


def _unwrap_ray_actor_class(actor_cls):
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)


class _FakeRay:
    @staticmethod
    def get(ref):
        return ref() if callable(ref) else ref


class _StubWorkerGroup:
    def __init__(self, calls):
        self._calls = calls

    def save_checkpoint(self, local_path, remote_path, global_step, max_ckpt_to_keep=None):
        self._calls.append(("save_checkpoint", local_path, remote_path, global_step, max_ckpt_to_keep))

    def load_checkpoint(self, local_path, del_local_after_load=False):
        self._calls.append(("load_checkpoint", local_path))


class _StubSynchronizer:
    def __init__(self, calls):
        self.rollouter_save_checkpoint = SimpleNamespace(
            remote=lambda folder: lambda: calls.append(("rollouter_save", folder))
        )


def _make_save_trainer(tmp_path, calls, max_actor_ckpt_to_keep, param_version=10):
    trainer = object.__new__(FullyAsyncTrainer)
    trainer.current_param_version = param_version
    trainer.use_critic = False
    trainer.actor_rollout_wg = _StubWorkerGroup(calls)
    trainer.param_synchronizer = _StubSynchronizer(calls)
    trainer._save_timing_state = lambda folder, start: calls.append(("timing_state", folder))
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "max_actor_ckpt_to_keep": max_actor_ckpt_to_keep,
                "max_critic_ckpt_to_keep": max_actor_ckpt_to_keep,
            }
        }
    )
    return trainer


# ---------------------------------------------------------------------------
# max_actor_ckpt_to_keep plumbing
# ---------------------------------------------------------------------------


def test_null_max_ckpt_to_keep_reaches_the_worker_group_as_none(monkeypatch, tmp_path):
    """max_actor_ckpt_to_keep=null must disable rotation, not rotate to some default."""
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []
    trainer = _make_save_trainer(tmp_path, calls, max_actor_ckpt_to_keep=None)

    trainer._save_checkpoint_inner()

    save = next(call for call in calls if call[0] == "save_checkpoint")
    assert save[-1] is None


def test_set_max_ckpt_to_keep_is_forwarded_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []
    trainer = _make_save_trainer(tmp_path, calls, max_actor_ckpt_to_keep=1)

    trainer._save_checkpoint_inner()

    save = next(call for call in calls if call[0] == "save_checkpoint")
    assert save[-1] == 1


def test_save_writes_the_tracker_and_snapshots_dataloader_and_timing_state(monkeypatch, tmp_path):
    monkeypatch.setattr(fat_module, "ray", _FakeRay)
    calls = []
    trainer = _make_save_trainer(tmp_path, calls, max_actor_ckpt_to_keep=None, param_version=7)

    trainer._save_checkpoint_inner()

    folder = str(tmp_path / "global_step_7")
    assert [call[0] for call in calls] == ["save_checkpoint", "rollouter_save", "timing_state"]
    assert calls[0][1] == os.path.join(folder, "actor")
    assert calls[0][2] is None  # no hdfs path configured
    assert calls[0][3] == 7
    assert calls[1][1] == calls[2][1] == folder
    with open(tmp_path / "latest_checkpointed_iteration.txt") as f:
        assert f.read() == "7"


# ---------------------------------------------------------------------------
# resume guard
# ---------------------------------------------------------------------------


def _make_resume_trainer(tmp_path, calls, resume_mode, load_contents, param_version=10):
    os.makedirs(tmp_path / f"global_step_{param_version}", exist_ok=True)
    with open(tmp_path / "latest_checkpointed_iteration.txt", "w") as f:
        f.write(str(param_version))

    trainer = object.__new__(FullyAsyncTrainer)
    trainer.trigger_parameter_sync_step = 1
    trainer.use_critic = False
    trainer.actor_rollout_wg = _StubWorkerGroup(calls)
    trainer._restore_timing_state = lambda folder: calls.append(("restore_timing", folder))
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": resume_mode,
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "del_local_ckpt_after_load": False,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"load_contents": load_contents}}},
        }
    )
    return trainer


def test_resume_is_refused_when_the_contents_cannot_restore_the_weights(tmp_path):
    """'hf_model' is written but never read back: such a resume would load nothing."""
    calls = []
    trainer = _make_resume_trainer(tmp_path, calls, resume_mode="auto", load_contents=["hf_model"])

    with pytest.raises(ValueError) as excinfo:
        trainer.load_checkpoint()

    message = str(excinfo.value)
    assert "hf_model" in message
    assert "resume_mode=disable" in message
    assert calls == [], "nothing may be loaded before the check"


def test_resume_disabled_is_unaffected_by_the_contents(tmp_path):
    calls = []
    trainer = _make_resume_trainer(tmp_path, calls, resume_mode="disable", load_contents=["hf_model"])

    assert trainer.load_checkpoint() == 0
    assert calls == [("load_checkpoint", None)]


def test_resume_with_model_contents_proceeds(tmp_path):
    calls = []
    trainer = _make_resume_trainer(tmp_path, calls, resume_mode="auto", load_contents=["model", "extra"])

    assert trainer.load_checkpoint() == 10
    folder = str(tmp_path / "global_step_10")
    assert ("restore_timing", folder) in calls
    assert ("load_checkpoint", os.path.join(folder, "actor")) in calls


def test_resume_from_scratch_is_allowed_whatever_the_contents(tmp_path):
    """No checkpoint yet: there is nothing to fail to restore."""
    calls = []
    trainer = _make_resume_trainer(tmp_path, calls, resume_mode="auto", load_contents=["hf_model"])
    os.remove(tmp_path / "latest_checkpointed_iteration.txt")

    assert trainer.load_checkpoint() == 0
    assert calls == [("load_checkpoint", None)]
