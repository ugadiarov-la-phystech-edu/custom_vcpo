# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""The synchronous trainer's resume guard for weights-only checkpoints.

save_contents=['hf_model'] writes checkpoints the managers never read back;
RayPPOTrainer._load_checkpoint must refuse to "resume" from one instead of
silently continuing from the pretrained weights (the port of the guard the
fully-async trainer got in the "Save only hf_model" commit). Covered: refusal
with a message naming the fix, resume_mode=disable indifference, a proper
'model' resume proceeding to the worker loads, the from-scratch path, and the
resume_path variant.

Run: pytest tests/trainer/ppo/test_sync_checkpoint_saving_on_cpu.py
"""

import os

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role


class _RecordingWorkerGroup:
    def __init__(self, name, calls):
        self._name = name
        self._calls = calls

    def load_checkpoint(self, path, del_local_after_load=False):
        self._calls.append((self._name, path))


class _StatelessDataloader:
    def load_state_dict(self, state):  # pragma: no cover - only hit with a data.pt present
        raise AssertionError("no dataloader state exists in these tests")


def _make_trainer(tmp_path, calls, resume_mode, load_contents, step=10, with_checkpoint=True):
    """A RayPPOTrainer skeleton with just enough state for _load_checkpoint."""
    if with_checkpoint:
        ckpt = tmp_path / f"global_step_{step}"
        ckpt.mkdir()
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(step))

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": resume_mode,
                "resume_from_path": str(tmp_path / f"global_step_{step}"),
                "default_hdfs_dir": None,
                "default_local_dir": str(tmp_path),
                "del_local_ckpt_after_load": False,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"load_contents": load_contents}}},
        }
    )
    trainer.actor_rollout_wg = _RecordingWorkerGroup("actor", calls)
    trainer.critic_wg = _RecordingWorkerGroup("critic", calls)
    trainer.use_critic = False
    trainer.train_dataloader = _StatelessDataloader()
    return trainer


def test_resume_is_refused_when_the_contents_cannot_restore_the_weights(tmp_path):
    """'hf_model' is written but never read back: such a resume would load nothing."""
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="auto", load_contents=["hf_model"])

    with pytest.raises(ValueError) as excinfo:
        trainer._load_checkpoint()

    message = str(excinfo.value)
    assert "hf_model" in message
    assert "resume_mode=disable" in message
    assert calls == [], "nothing may be loaded before the check"


def test_resume_disabled_is_unaffected_by_the_contents(tmp_path):
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="disable", load_contents=["hf_model"])

    assert trainer._load_checkpoint() == 0
    assert calls == []


def test_resume_with_model_contents_proceeds(tmp_path):
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="auto", load_contents=["model", "optimizer"])

    trainer._load_checkpoint()

    assert trainer.global_steps == 10
    folder = str(tmp_path / "global_step_10")
    assert calls == [("actor", os.path.join(folder, "actor"))]


def test_resume_with_default_none_contents_proceeds(tmp_path):
    """load_contents=None means the manager default, which includes 'model'."""
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="auto", load_contents=None)

    trainer._load_checkpoint()
    assert trainer.global_steps == 10
    assert len(calls) == 1


def test_resume_from_scratch_is_allowed_whatever_the_contents(tmp_path):
    """No checkpoint yet: there is nothing to fail to restore."""
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="auto", load_contents=["hf_model"], with_checkpoint=False)

    assert trainer._load_checkpoint() == 0
    assert calls == []


def test_resume_path_with_hf_only_contents_is_refused_too(tmp_path):
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="resume_path", load_contents=["hf_model"])

    with pytest.raises(ValueError):
        trainer._load_checkpoint()
    assert calls == []


def test_critic_also_loads_when_enabled(tmp_path):
    calls = []
    trainer = _make_trainer(tmp_path, calls, resume_mode="auto", load_contents=["model"])
    trainer.use_critic = True

    trainer._load_checkpoint()
    folder = str(tmp_path / "global_step_10")
    assert ("critic", os.path.join(folder, str(Role.Critic))) in calls
