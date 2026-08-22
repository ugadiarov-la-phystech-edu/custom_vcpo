# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How checkpoint save/load contents map onto what the managers actually do.

Covers the two derived properties that let an 'hf_model'-only checkpoint work -- the huggingface
config/tokenizer must still be written, the distributed checkpoint must not be -- and the
resume-time predicate that refuses a checkpoint which would restore nothing.

torch.distributed.get_rank/get_world_size are patched instead of initializing a process group, so
nothing leaks into other suites running in the same pytest process.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, restores_model_weights


def make_manager(save_contents=None, load_contents=None, explicit_config=True):
    checkpoint_config = None
    if explicit_config:
        checkpoint_config = {"save_contents": save_contents, "load_contents": load_contents}
    with (
        mock.patch.object(torch.distributed, "get_rank", return_value=0),
        mock.patch.object(torch.distributed, "get_world_size", return_value=1),
    ):
        return BaseCheckpointManager(
            model=None,
            optimizer=None,
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=checkpoint_config,
        )


class TestCheckpointContents(unittest.TestCase):
    def assert_flags(self, manager, *, model, optimizer, extra, hf_model, hf_metadata, dist_checkpoint):
        self.assertEqual(manager.should_save_model, model)
        self.assertEqual(manager.should_save_optimizer, optimizer)
        self.assertEqual(manager.should_save_extra, extra)
        self.assertEqual(manager.should_save_hf_model, hf_model)
        self.assertEqual(manager.should_save_hf_metadata, hf_metadata)
        self.assertEqual(manager.should_save_dist_checkpoint, dist_checkpoint)

    def test_defaults_when_no_config_is_given(self):
        manager = make_manager(explicit_config=False)
        self.assert_flags(
            manager, model=True, optimizer=True, extra=True, hf_model=False, hf_metadata=True, dist_checkpoint=True
        )
        self.assertTrue(manager.should_load_model)
        self.assertTrue(manager.should_load_optimizer)
        self.assertTrue(manager.should_load_extra)

    def test_full_contents(self):
        manager = make_manager(["model", "optimizer", "extra"])
        self.assert_flags(
            manager, model=True, optimizer=True, extra=True, hf_model=False, hf_metadata=True, dist_checkpoint=True
        )

    def test_model_and_extra_only(self):
        """Optimizer-free but still a sharded checkpoint."""
        manager = make_manager(["model", "extra"])
        self.assert_flags(
            manager, model=True, optimizer=False, extra=True, hf_model=False, hf_metadata=True, dist_checkpoint=True
        )

    def test_hf_model_only(self):
        """The configuration the baseline arms run: weights in HF form and nothing else.

        hf_metadata must stay True (the tokenizer/config live next to the weights and the
        directory is unusable without them) while the distributed checkpoint is skipped
        entirely (there is nothing to put in it).
        """
        manager = make_manager(["hf_model"])
        self.assert_flags(
            manager, model=False, optimizer=False, extra=False, hf_model=True, hf_metadata=True, dist_checkpoint=False
        )

    def test_model_and_hf_model(self):
        manager = make_manager(["model", "hf_model"])
        self.assert_flags(
            manager, model=True, optimizer=False, extra=False, hf_model=True, hf_metadata=True, dist_checkpoint=True
        )

    def test_empty_contents(self):
        manager = make_manager([])
        self.assert_flags(
            manager, model=False, optimizer=False, extra=False, hf_model=False, hf_metadata=False, dist_checkpoint=False
        )

    def test_optimizer_only_still_needs_the_dist_checkpoint(self):
        manager = make_manager(["optimizer", "extra"])
        self.assert_flags(
            manager, model=False, optimizer=True, extra=True, hf_model=False, hf_metadata=False, dist_checkpoint=True
        )

    def test_load_contents_are_independent_of_save_contents(self):
        manager = make_manager(["hf_model"], ["model", "optimizer", "extra"])
        self.assertFalse(manager.should_save_model)
        self.assertTrue(manager.should_load_model)
        self.assertTrue(manager.should_load_optimizer)
        self.assertTrue(manager.should_load_extra)


class TestRestoresModelWeights(unittest.TestCase):
    def test_contents_with_model(self):
        self.assertTrue(restores_model_weights(["model", "optimizer", "extra"]))
        self.assertTrue(restores_model_weights(["model"]))

    def test_contents_without_model(self):
        # 'hf_model' is written by the managers but never read back
        self.assertFalse(restores_model_weights(["hf_model"]))
        self.assertFalse(restores_model_weights(["optimizer", "extra"]))
        self.assertFalse(restores_model_weights([]))

    def test_none_means_the_managers_default_which_includes_model(self):
        self.assertTrue(restores_model_weights(None))


class TestRemovePreviousSaveLocalPath(unittest.TestCase):
    def test_removes_a_list_of_directories(self):
        manager = make_manager(explicit_config=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, f"global_step_{step}") for step in (1, 2)]
            for path in paths:
                os.makedirs(os.path.join(path, "actor"))
            manager.remove_previous_save_local_path(paths)
            for path in paths:
                self.assertFalse(os.path.exists(path))

    def test_accepts_a_bare_string(self):
        manager = make_manager(explicit_config=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "global_step_1")
            os.makedirs(path)
            manager.remove_previous_save_local_path(path)
            self.assertFalse(os.path.exists(path))

    def test_tolerates_a_path_that_is_already_gone(self):
        manager = make_manager(explicit_config=False)
        with tempfile.TemporaryDirectory() as tmp:
            manager.remove_previous_save_local_path([os.path.join(tmp, "never_created")])


if __name__ == "__main__":
    unittest.main()
