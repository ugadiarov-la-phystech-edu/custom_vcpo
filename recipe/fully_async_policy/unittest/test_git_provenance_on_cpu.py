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

"""Tests for the one-time git branch/commit provenance record.

Covers the three layers added for it:
- ``_TensorboardAdapter.log_text`` -> SummaryWriter.add_text;
- ``Tracking.log_text`` dispatch (only backends that support text get the call);
- ``FullyAsyncTrainer._log_git_provenance`` (real-repo happy path with the exact
  ``branch=... commit=... worktree=...`` shape, and the never-raises failure path).

Run: pytest recipe/fully_async_policy/unittest/test_git_provenance_on_cpu.py
"""

import os
import re
import subprocess

from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from verl.utils.tracking import Tracking, _TensorboardAdapter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _unwrap_ray_actor_class(actor_cls):
    """The class is a @ray.remote ActorClass wrapper; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)

GIT_STATE_RE = re.compile(r"^branch=(\S+) commit=([0-9a-f]{12}) worktree=(dirty|clean) root=(\S+)$")


class _FakeWriter:
    def __init__(self):
        self.texts = []

    def add_text(self, tag, text, step):
        self.texts.append((tag, text, step))


class _TextRecordingLogger:
    """Stands in for self.logger on the trainer; records log_text calls."""

    def __init__(self):
        self.texts = []

    def log_text(self, tag, text, step=0):
        self.texts.append((tag, text, step))


class _TextlessBackend:
    """A backend WITHOUT log_text (like the console logger) — must be skipped."""

    def log(self, data, step):
        pass


# ---------------------------------------------------------------------------
# adapter + dispatch
# ---------------------------------------------------------------------------


def test_tensorboard_adapter_log_text_writes_to_text_tab():
    adapter = object.__new__(_TensorboardAdapter)
    adapter.writer = _FakeWriter()
    adapter.log_text(tag="git/state", text="branch=b commit=c worktree=clean", step=0)
    assert adapter.writer.texts == [("git/state", "branch=b commit=c worktree=clean", 0)]


def test_tracking_log_text_dispatches_only_to_capable_backends():
    tracking = object.__new__(Tracking)
    tb = object.__new__(_TensorboardAdapter)
    tb.writer = _FakeWriter()
    tracking.logger = {"console": _TextlessBackend(), "tensorboard": tb}
    tracking.log_text(tag="git/state", text="hello", step=0)  # must not raise on console
    assert tb.writer.texts == [("git/state", "hello", 0)]


def test_tensorboard_text_round_trips_through_events_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    adapter = _TensorboardAdapter("proj", "exp")
    adapter.log_text(tag="git/state", text="branch=b commit=abc123 worktree=clean", step=0)
    adapter.finish()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = [f for f in os.listdir(tmp_path) if f.startswith("events.")]
    assert events
    acc = EventAccumulator(os.path.join(tmp_path, events[0]), size_guidance={"tensors": 0})
    acc.Reload()
    assert "git/state/text_summary" in acc.Tags()["tensors"]
    payload = acc.Tensors("git/state/text_summary")[0].tensor_proto.string_val[0]
    assert payload == b"branch=b commit=abc123 worktree=clean"


# ---------------------------------------------------------------------------
# trainer helper
# ---------------------------------------------------------------------------


def _trainer_with_recording_logger():
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.logger = _TextRecordingLogger()
    return t


def test_log_git_provenance_reports_real_branch_and_commit():
    trainer = _trainer_with_recording_logger()
    trainer._log_git_provenance()

    assert len(trainer.logger.texts) == 1
    tag, text, step = trainer.logger.texts[0]
    assert tag == "git/state" and step == 0
    m = GIT_STATE_RE.match(text)
    assert m, f"unexpected git/state text: {text!r}"

    branch, commit, _dirty, root = m.groups()
    expected_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    expected_commit = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert branch == expected_branch
    assert commit == expected_commit
    assert os.path.samefile(root, REPO_ROOT)


def test_log_git_provenance_never_raises_when_git_fails(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", _boom)
    trainer = _trainer_with_recording_logger()
    trainer._log_git_provenance()  # must not raise

    assert len(trainer.logger.texts) == 1
    tag, text, step = trainer.logger.texts[0]
    assert tag == "git/state" and step == 0
    assert text.startswith("unavailable")
    assert "FileNotFoundError" in text
