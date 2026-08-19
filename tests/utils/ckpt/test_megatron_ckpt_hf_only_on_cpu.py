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
"""CPU tests for hf_model-only saving on the Megatron checkpoint manager.

Regression for the production crash: checkpoint.save_contents=['hf_model']
made save_checkpoint run the dist-checkpointing pass with an empty state dict
and die on the unconditional debug log `state_dict['model'].keys()`
(KeyError: 'model') before writing anything. The fix skips the
dist-checkpointing pass entirely when there is no model/optimizer/extra state
to save, finalizes synchronously even under async_save, and writes the HF
config/tokenizer for hf_model-only checkpoints (bare weights are not loadable
as an HF export without them).

Stub style as in tests/workers/actor/test_per_traj_packed_on_cpu.py: a bare
__new__ instance with recorder stubs; module-level path/save helpers
monkeypatched.

Run: pytest tests/utils/ckpt/test_megatron_ckpt_hf_only_on_cpu.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

import verl.utils.checkpoint.megatron_checkpoint_manager as mcm
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def save_pretrained(self, path):
        self.calls.append(("save_pretrained", path))

    def save_hf_weights(self, model, path):
        self.calls.append(("save_hf_weights", path))


def _make_manager(tmp_path, save_contents, async_save=False, use_dist_checkpointing=True):
    m = MegatronCheckpointManager.__new__(MegatronCheckpointManager)
    m.rank = 0
    m.model = [object()]
    m.previous_saved_paths = []
    m.previous_global_step = 0
    m.checkpoint_config = SimpleNamespace(async_save=async_save)
    m.checkpoint_save_contents = list(save_contents)
    m.use_dist_checkpointing = use_dist_checkpointing
    m.use_hf_checkpoint = not use_dist_checkpointing
    m.peft_cls = None
    m.bridge = _Recorder()
    m.vanilla_bridge = False
    m.processing_class = _Recorder()
    # No name_or_path attribute -> the GenerationConfig fetch is skipped
    m.hf_config = _Recorder()
    m.generate_state_dict = _Recorder()
    return m


@pytest.fixture
def patched_env(monkeypatch, tmp_path):
    calls = {"dist_save": _Recorder(), "dist_path": _Recorder(), "barrier": _Recorder()}

    def fake_dist_path(path):
        calls["dist_path"]((path,), {})
        return os.path.join(path, "dist_ckpt")

    monkeypatch.setattr(mcm, "local_mkdir_safe", lambda p: p)
    monkeypatch.setattr(mcm, "get_dist_checkpoint_path", fake_dist_path)
    monkeypatch.setattr(mcm, "get_hf_model_checkpoint_path", lambda p: os.path.join(p, "huggingface"))
    monkeypatch.setattr(mcm, "save_dist_checkpointing", lambda **kw: calls["dist_save"]((), kw) or None)
    monkeypatch.setattr(mcm, "log_with_rank", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: calls["barrier"]((), {}), raising=False)
    return calls


def _ckpt_dir(tmp_path):
    # finalize_save_fn writes latest_checkpointed_iteration.txt two levels up
    d = tmp_path / "run" / "global_step_5"
    d.mkdir(parents=True)
    return str(d)


class TestHfModelOnly:
    def test_completes_without_dist_save(self, patched_env, tmp_path):
        """The production crash scenario: no KeyError, no dist-ckpt work."""
        m = _make_manager(tmp_path, ["hf_model"])
        path = _ckpt_dir(tmp_path)
        m.save_checkpoint(path, global_step=5)
        assert m.generate_state_dict.calls == []
        assert patched_env["dist_save"].calls == []
        assert patched_env["dist_path"].calls == []  # no empty dist_ckpt/ dir
        # HF weights written via the bridge into <ckpt>/huggingface
        assert m.bridge.calls == [("save_hf_weights", os.path.join(path, "huggingface"))]
        assert m.previous_saved_paths[-1].endswith("global_step_5")

    def test_config_and_tokenizer_written(self, patched_env, tmp_path):
        m = _make_manager(tmp_path, ["hf_model"])
        m.save_checkpoint(_ckpt_dir(tmp_path), global_step=5)
        assert any(c[0] == "save_pretrained" for c in m.hf_config.calls)
        assert any(c[0] == "save_pretrained" for c in m.processing_class.calls)

    def test_finalize_writes_latest_iteration(self, patched_env, tmp_path):
        m = _make_manager(tmp_path, ["hf_model"])
        m.save_checkpoint(_ckpt_dir(tmp_path), global_step=7)
        tracker = tmp_path / "latest_checkpointed_iteration.txt"
        assert tracker.read_text() == "7"

    def test_async_save_finalizes_synchronously(self, patched_env, tmp_path):
        """With the dist pass skipped there is no async request; the old code
        asserted one exists under async_save=True."""
        m = _make_manager(tmp_path, ["hf_model"], async_save=True)
        m.save_checkpoint(_ckpt_dir(tmp_path), global_step=5)
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "5"


class TestDistSavePreserved:
    def test_model_contents_still_dist_saves(self, patched_env, tmp_path):
        m = _make_manager(tmp_path, ["model"])
        m.generate_state_dict = lambda *a, **k: {"model": {"w": 1}}
        m.save_checkpoint(_ckpt_dir(tmp_path), global_step=5)
        assert len(patched_env["dist_save"].calls) == 1
        assert len(patched_env["dist_path"].calls) == 1
        assert len(patched_env["barrier"].calls) == 1  # sync save barriers
        # model-only: hf weights not written, but config/tokenizer are
        assert not any(c[0] == "save_hf_weights" for c in m.bridge.calls)
        assert any(c[0] == "save_pretrained" for c in m.hf_config.calls)

    def test_optimizer_only_skips_model_key_logging(self, patched_env, tmp_path):
        """generate_state_dict without model must not be indexed for 'model'."""
        m = _make_manager(tmp_path, ["optimizer"])
        m.generate_state_dict = lambda *a, **k: {"optimizer": {"state": 1}}
        m.save_checkpoint(_ckpt_dir(tmp_path), global_step=5)
        assert len(patched_env["dist_save"].calls) == 1
        # no model, no hf_model -> no HF artifacts at all
        assert m.bridge.calls == []
        assert m.hf_config.calls == []
