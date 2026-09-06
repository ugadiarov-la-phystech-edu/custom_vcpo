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
"""Unit tests for the two-tier checkpoint retention of the fully-async trainer
(async_training.resumable_ckpts_to_keep):

- prune_resumable_checkpoint_state: keeps hf_model / transformer_config /
  timing_state / data.pt everywhere, drops dist_ckpt + replay_buffer.pt +
  rollout_queue.pt + message_queue.pt from all but the N newest checkpoints,
  ignores foreign higher-numbered dirs, tolerates hf-only dirs, idempotent
- _save_checkpoint_inner wiring: pruning runs after the tracker write, never on
  the checkpoint just written, disabled when the knob is null
- _check_resumable_checkpoint: a clear error when resuming from a pruned or
  hf-only directory
- recipe yaml defaults and the trainer-source tripwire
- verify_checkpoints.py --resumable-last: expectation helper and per-dir checks

Run: pytest recipe/fully_async_policy/unittest/test_resumable_ckpt_pruning_on_cpu.py
"""

import importlib.util
import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from recipe.fully_async_policy import fully_async_trainer as fat_module
from recipe.fully_async_policy.fully_async_trainer import (
    RESUMABLE_CKPT_PIECES,
    list_checkpoint_steps,
    prune_resumable_checkpoint_state,
)
from recipe.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer as _TrainerActor,
)

FullyAsyncTrainer = (
    _TrainerActor.__ray_metadata__.modified_class if hasattr(_TrainerActor, "__ray_metadata__") else _TrainerActor
)

KEPT_FILES = ("actor/huggingface/model.safetensors", "actor/transformer_config.json", "timing_state.json", "data.pt")


def _write(path, size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)


def _make_step(root, step, resumable=True, hf=True):
    d = os.path.join(root, f"global_step_{step}")
    os.makedirs(d, exist_ok=True)
    if hf:
        for rel in KEPT_FILES:
            _write(os.path.join(d, rel))
    if resumable:
        _write(os.path.join(d, "actor", "dist_ckpt", "shard_0.pt"), size=100)
        _write(os.path.join(d, "actor", "dist_ckpt", "common.pt"), size=20)
        _write(os.path.join(d, "replay_buffer.pt"), size=50)
        _write(os.path.join(d, "rollout_queue.pt"), size=30)
        _write(os.path.join(d, "message_queue.pt"), size=10)
    return d


def _has_resumable(d):
    return [p for p in RESUMABLE_CKPT_PIECES if os.path.exists(os.path.join(d, p))]


def _has_kept(d):
    return all(os.path.exists(os.path.join(d, rel)) for rel in KEPT_FILES)


# ---------------------------------------------------------------- pruner


def test_list_checkpoint_steps_orders_and_filters(tmp_path):
    root = str(tmp_path)
    for s in (15, 5, 10):
        _make_step(root, s)
    os.makedirs(os.path.join(root, "global_step_notanumber"))
    os.makedirs(os.path.join(root, "tensorboard"))
    _write(os.path.join(root, "global_step_7"))  # a FILE with the prefix, not a dir
    assert [s for s, _ in list_checkpoint_steps(root)] == [5, 10, 15]
    assert list_checkpoint_steps(os.path.join(root, "missing")) == []
    assert list_checkpoint_steps("") == []


def test_prune_keeps_only_the_newest_resumable_state(tmp_path):
    root = str(tmp_path)
    dirs = {s: _make_step(root, s) for s in (5, 10, 15)}
    removed = prune_resumable_checkpoint_state(root, keep=1, current_step=15)
    assert _has_resumable(dirs[15]) == list(RESUMABLE_CKPT_PIECES)
    assert _has_resumable(dirs[10]) == [] and _has_resumable(dirs[5]) == []
    for d in dirs.values():
        assert _has_kept(d)
    removed_paths = {p for p, _ in removed}
    assert removed_paths == {os.path.join(dirs[s], piece) for s in (5, 10) for piece in RESUMABLE_CKPT_PIECES}
    sizes = dict(removed)
    assert sizes[os.path.join(dirs[5], "actor/dist_ckpt")] == 120  # directory size summed
    assert sizes[os.path.join(dirs[5], "replay_buffer.pt")] == 50


def test_prune_keep_two(tmp_path):
    root = str(tmp_path)
    dirs = {s: _make_step(root, s) for s in (5, 10, 15)}
    prune_resumable_checkpoint_state(root, keep=2, current_step=15)
    assert _has_resumable(dirs[5]) == []
    assert _has_resumable(dirs[10]) == list(RESUMABLE_CKPT_PIECES)
    assert _has_resumable(dirs[15]) == list(RESUMABLE_CKPT_PIECES)


@pytest.mark.parametrize("keep", [None, 0, -1])
def test_prune_disabled(tmp_path, keep):
    root = str(tmp_path)
    dirs = {s: _make_step(root, s) for s in (5, 10)}
    assert prune_resumable_checkpoint_state(root, keep=keep, current_step=10) == []
    for d in dirs.values():
        assert _has_resumable(d) == list(RESUMABLE_CKPT_PIECES)


def test_prune_tolerates_hf_only_dirs_and_is_idempotent(tmp_path):
    root = str(tmp_path)
    old_hf_only = _make_step(root, 5, resumable=False)
    partial = _make_step(root, 10, resumable=False)
    _write(os.path.join(partial, "replay_buffer.pt"))  # one piece only
    newest = _make_step(root, 15)
    removed = prune_resumable_checkpoint_state(root, keep=1, current_step=15)
    assert [p for p, _ in removed] == [os.path.join(partial, "replay_buffer.pt")]
    assert _has_kept(old_hf_only) and _has_kept(partial)
    assert _has_resumable(newest) == list(RESUMABLE_CKPT_PIECES)
    assert prune_resumable_checkpoint_state(root, keep=1, current_step=15) == []


def test_prune_never_touches_foreign_higher_steps_or_the_current_one(tmp_path):
    root = str(tmp_path)
    dirs = {s: _make_step(root, s) for s in (5, 10, 20)}
    # current step is 10: 20 is foreign (e.g. a resumed run's earlier future), left alone
    prune_resumable_checkpoint_state(root, keep=1, current_step=10)
    assert _has_resumable(dirs[5]) == []
    assert _has_resumable(dirs[10]) == list(RESUMABLE_CKPT_PIECES)
    assert _has_resumable(dirs[20]) == list(RESUMABLE_CKPT_PIECES)


def test_prune_with_fewer_dirs_than_keep_is_a_noop(tmp_path):
    root = str(tmp_path)
    d = _make_step(root, 5)
    assert prune_resumable_checkpoint_state(root, keep=3, current_step=5) == []
    assert _has_resumable(d) == list(RESUMABLE_CKPT_PIECES)
    assert prune_resumable_checkpoint_state(os.path.join(root, "nope"), keep=1, current_step=5) == []


# ---------------------------------------------------------------- trainer wiring


class _StubWorkerGroup:
    def __init__(self, calls):
        self.calls = calls

    def save_checkpoint(self, local_path, remote_path, step, max_ckpt_to_keep=None):
        self.calls.append(("actor", step, max_ckpt_to_keep))
        _write(os.path.join(local_path, "huggingface", "model.safetensors"))
        _write(os.path.join(local_path, "transformer_config.json"))
        _write(os.path.join(local_path, "dist_ckpt", "shard_0.pt"), size=100)


class _StubSynchronizer:
    def __init__(self, calls):
        self.calls = calls

    class _Remote:
        def __init__(self, fn):
            self.remote = fn

    @property
    def rollouter_save_checkpoint(self):
        def fn(folder):
            self.calls.append(("rollouter", folder))
            _write(os.path.join(folder, "data.pt"))
            _write(os.path.join(folder, "rollout_queue.pt"), size=30)
            _write(os.path.join(folder, "message_queue.pt"), size=10)
            return "ref"

        return self._Remote(fn)


def _make_trainer(root, keep, calls):
    t = object.__new__(FullyAsyncTrainer)
    t.config = OmegaConf.create(
        {"trainer": {"default_local_dir": root, "default_hdfs_dir": None, "max_actor_ckpt_to_keep": None}}
    )
    t.use_critic = False
    t.actor_rollout_wg = _StubWorkerGroup(calls)
    t.param_synchronizer = _StubSynchronizer(calls)
    t.resumable_ckpts_to_keep = keep
    t.replay_enable = True
    t.replay_save_state = True

    def _save_timing_state(folder, save_start):
        calls.append(("timing", folder))
        _write(os.path.join(folder, "timing_state.json"))

    def _save_replay_state(folder):
        calls.append(("replay", folder))
        _write(os.path.join(folder, "replay_buffer.pt"), size=50)

    t._save_timing_state = _save_timing_state
    t._save_replay_state = _save_replay_state
    return t


def test_save_inner_prunes_older_after_tracker_write(tmp_path, monkeypatch):
    monkeypatch.setattr(fat_module, "ray", SimpleNamespace(get=lambda ref: ref))
    root = str(tmp_path)
    calls = []
    t = _make_trainer(root, keep=1, calls=calls)
    real_prune = fat_module.prune_resumable_checkpoint_state

    def traced_prune(*a, **kw):
        calls.append(("prune", open(os.path.join(root, "latest_checkpointed_iteration.txt")).read()))
        return real_prune(*a, **kw)

    monkeypatch.setattr(fat_module, "prune_resumable_checkpoint_state", traced_prune)

    t.current_param_version = 5
    t._save_checkpoint_inner()
    d5 = os.path.join(root, "global_step_5")
    assert _has_resumable(d5) == list(RESUMABLE_CKPT_PIECES)  # the only checkpoint: kept whole
    assert calls[-1] == ("prune", "5")  # pruning ran last, after the tracker said 5

    t.current_param_version = 10
    t._save_checkpoint_inner()
    d10 = os.path.join(root, "global_step_10")
    assert _has_resumable(d10) == list(RESUMABLE_CKPT_PIECES)
    assert _has_resumable(d5) == []
    assert _has_kept(d5) and _has_kept(d10)
    assert open(os.path.join(root, "latest_checkpointed_iteration.txt")).read() == "10"
    assert calls[-1] == ("prune", "10")
    # the actor save still runs with max_ckpt_to_keep=None (whole-dir deletion stays off)
    assert ("actor", 10, None) in calls


def test_save_inner_without_the_knob_prunes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fat_module, "ray", SimpleNamespace(get=lambda ref: ref))
    root = str(tmp_path)
    t = _make_trainer(root, keep=None, calls=[])
    for v in (5, 10):
        t.current_param_version = v
        t._save_checkpoint_inner()
    for v in (5, 10):
        assert _has_resumable(os.path.join(root, f"global_step_{v}")) == list(RESUMABLE_CKPT_PIECES)


def test_trainer_reads_the_knob_from_config():
    src = open(fat_module.__file__).read()
    assert 'config.async_training.get("resumable_ckpts_to_keep", None)' in src
    body = src.split("def _save_checkpoint_inner")[1].split("def _prune_older_resumable_checkpoints")[0]
    assert body.index("latest_checkpointed_iteration.txt") < body.index("self._prune_older_resumable_checkpoints()")


def test_recipe_yamls_default_to_no_pruning():
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    for name in ("fully_async_ppo_trainer.yaml", "fully_async_ppo_megatron_trainer.yaml"):
        cfg = OmegaConf.load(os.path.join(cfg_dir, name))
        assert cfg.async_training.resumable_ckpts_to_keep is None, name


# ---------------------------------------------------------------- resume guard


def test_resume_guard_refuses_pruned_or_hf_only_dirs(tmp_path):
    root = str(tmp_path)
    pruned = _make_step(root, 5, resumable=False)
    with pytest.raises(RuntimeError, match="no actor/dist_ckpt"):
        FullyAsyncTrainer._check_resumable_checkpoint(pruned)
    full = _make_step(root, 10)
    FullyAsyncTrainer._check_resumable_checkpoint(full)  # no raise


def test_resume_guard_is_called_before_restoring_state():
    src = open(fat_module.__file__).read()
    body = src.split("def load_checkpoint(self)")[1]
    assert body.index("self._check_resumable_checkpoint(global_step_folder)") < body.index(
        "self._restore_timing_state(global_step_folder)"
    )


# ---------------------------------------------------------------- verify_checkpoints.py


def _load_verify_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "shell", "vcpo", "dapo", "replay_buffer", "verify_checkpoints.py"
    )
    spec = importlib.util.spec_from_file_location("verify_checkpoints", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Report:
    def __init__(self):
        self.results = []

    def check(self, ok, msg):
        self.results.append((bool(ok), msg))
        return bool(ok)

    def failures(self):
        return [m for ok, m in self.results if not ok]


def test_resumable_expectation_helper():
    vc = _load_verify_module()
    assert [vc.resumable_expectation(i, 3, 0) for i in range(3)] == [None, None, None]
    assert [vc.resumable_expectation(i, 3, 1) for i in range(3)] == [False, False, True]
    assert [vc.resumable_expectation(i, 3, 2) for i in range(3)] == [False, True, True]
    assert [vc.resumable_expectation(i, 2, 5) for i in range(2)] == [True, True]


def test_check_resumable_pieces_modes(tmp_path):
    vc = _load_verify_module()
    root = str(tmp_path)
    full = _make_step(root, 10)
    pruned = _make_step(root, 5, resumable=False)

    r = _Report()
    vc.check_resumable_pieces(full, r, expect_resumable=True)
    assert r.failures() == [] and len(r.results) == 3
    r = _Report()
    vc.check_resumable_pieces(pruned, r, expect_resumable=False)
    assert r.failures() == []
    r = _Report()
    vc.check_resumable_pieces(full, r, expect_resumable=False)  # should have been pruned
    assert len(r.failures()) == 1 and "left:" in r.failures()[0]
    r = _Report()
    vc.check_resumable_pieces(pruned, r, expect_resumable=True)  # should be full
    assert len(r.failures()) == 3
    # legacy hf-only rule
    r = _Report()
    vc.check_resumable_pieces(pruned, r, expect_resumable=None)
    assert r.failures() == []
    r = _Report()
    vc.check_resumable_pieces(full, r, expect_resumable=None)
    assert len(r.failures()) == 1 and "hf_model-only" in r.failures()[0]
