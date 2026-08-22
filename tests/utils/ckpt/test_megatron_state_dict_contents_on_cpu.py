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
"""What MegatronCheckpointManager.generate_state_dict puts in a checkpoint.

These pin the behaviour that makes optimizer-free checkpoints (save_contents without 'model'
and/or without 'optimizer') possible:

* the model keys are dropped when the model is not being saved -- ALL of them, including the
  model0/model1/... produced by virtual pipeline parallelism;
* the model state dict is still built when the optimizer needs it as input, then dropped;
* callers must ask which model keys exist (model_state_dict_keys) instead of assuming "model",
  which used to raise KeyError for every contents list without 'model'.

The manager is built with __new__ so that __init__ (torch.distributed) is not needed, and
torch.distributed.barrier is patched out; no GPU and no process group are involved.
"""

import os
import unittest
from unittest import mock

import torch

from verl.utils.checkpoint.megatron_checkpoint_manager import (
    MegatronCheckpointManager,
    model_state_dict_keys,
)


class FakeModel:
    """Stands in for a megatron model chunk."""

    def __init__(self, name="w"):
        self.name = name

    def sharded_state_dict(self):
        return {self.name: object()}


class FakeOptimizer:
    def __init__(self):
        self.received_keys = None

    def sharded_state_dict(self, state_dict, is_loading=False):
        # megatron builds the optimizer's sharded state dict from the model's
        self.received_keys = sorted(state_dict.keys())
        return {"opt": object()}


class FakeScheduler:
    def state_dict(self):
        return {"lr": 1e-6}


def make_manager(num_model_chunks=1, lr_scheduler=None):
    manager = MegatronCheckpointManager.__new__(MegatronCheckpointManager)
    manager.model = [FakeModel(f"w{i}") for i in range(num_model_chunks)]
    manager.optimizer = FakeOptimizer()
    manager.lr_scheduler = lr_scheduler
    manager.get_rng_state = lambda: {"rng": 0}
    return manager


class TestModelStateDictKeys(unittest.TestCase):
    def test_single_model_chunk(self):
        self.assertEqual(model_state_dict_keys({"model": {}, "optimizer": {}, "rng_state": {}}), ["model"])

    def test_virtual_pipeline_chunks(self):
        keys = model_state_dict_keys({"model1": {}, "model0": {}, "rng_state": {}})
        self.assertEqual(keys, ["model0", "model1"])

    def test_no_model_saved(self):
        # save_contents=['hf_model'] leaves nothing behind; the caller must not assume "model"
        self.assertEqual(model_state_dict_keys({}), [])
        self.assertEqual(model_state_dict_keys({"optimizer": {}, "rng_state": {}}), [])

    def test_unrelated_keys_are_not_mistaken_for_model_keys(self):
        self.assertEqual(model_state_dict_keys({"model_config": {}, "modelling": {}}), [])


class TestGenerateStateDict(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(torch.distributed, "barrier", lambda *args, **kwargs: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_optimizer_extra(self):
        manager = make_manager()
        state_dict = manager.generate_state_dict(True, True, True)
        self.assertEqual(sorted(state_dict.keys()), ["model", "optimizer", "rng_state"])

    def test_lr_scheduler_rides_along_with_the_optimizer(self):
        manager = make_manager(lr_scheduler=FakeScheduler())
        state_dict = manager.generate_state_dict(True, True, True)
        self.assertEqual(sorted(state_dict.keys()), ["lr_scheduler", "model", "optimizer", "rng_state"])
        # ... and is absent from an optimizer-free checkpoint
        self.assertEqual(sorted(manager.generate_state_dict(True, False, True).keys()), ["model", "rng_state"])

    def test_model_and_extra_only(self):
        manager = make_manager()
        self.assertEqual(sorted(manager.generate_state_dict(True, False, True).keys()), ["model", "rng_state"])

    def test_hf_model_only_generates_nothing(self):
        """save_contents=['hf_model']: no model, no optimizer, no extra.

        The empty dict is the point: save_checkpoint used to read state_dict['model'] here
        unconditionally and died with KeyError before writing anything.
        """
        manager = make_manager()
        state_dict = manager.generate_state_dict(False, False, False)
        self.assertEqual(state_dict, {})
        self.assertEqual(model_state_dict_keys(state_dict), [])

    def test_optimizer_without_model_still_feeds_the_optimizer_the_model_keys(self):
        manager = make_manager()
        state_dict = manager.generate_state_dict(False, True, False)
        self.assertEqual(sorted(state_dict.keys()), ["optimizer"])
        # the model dict must be built first: it is the optimizer's input, then dropped
        self.assertEqual(manager.optimizer.received_keys, ["model"])

    def test_virtual_pipeline_model_keys_are_all_dropped(self):
        """With vpp the keys are model0/model1; popping only "model" leaked the full weights.

        The optimizer forces the model dict to be built, so this is the shape where the leak
        was reachable: save_contents without 'model' but with 'optimizer'.
        """
        manager = make_manager(num_model_chunks=2)
        with mock.patch(
            "verl.utils.checkpoint.megatron_checkpoint_manager.mpu.set_virtual_pipeline_model_parallel_rank"
        ):
            state_dict = manager.generate_state_dict(False, True, False)
            self.assertEqual(sorted(state_dict.keys()), ["optimizer"])
            self.assertEqual(manager.optimizer.received_keys, ["model0", "model1"])

            self.assertEqual(manager.generate_state_dict(False, False, False), {})
            kept = manager.generate_state_dict(True, False, False)
        self.assertEqual(sorted(kept.keys()), ["model0", "model1"])

    def test_model_chunks_are_not_built_when_nobody_needs_them(self):
        manager = make_manager()
        with mock.patch.object(FakeModel, "sharded_state_dict", side_effect=AssertionError("should not be called")):
            self.assertEqual(manager.generate_state_dict(False, False, True), {"rng_state": {"rng": 0}})


if __name__ == "__main__":
    unittest.main()


class TestSaveCheckpointHfModelOnly(unittest.TestCase):
    """Drive the real save_checkpoint for save_contents=['hf_model'] on CPU.

    This is the path the baseline arms take. It used to die with KeyError: 'model' before
    writing anything, and even once past that it left the huggingface directory without the
    tokenizer (the rank-0 metadata save was gated on 'model' being saved) and wrote an empty
    dist_ckpt/ that looks like a real checkpoint.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        import warnings

        from transformers import AutoModelForCausalLM, Qwen3Config

        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls._tmp.name, "base_model")
        config = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            tie_word_embeddings=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_config(config)
        model.save_pretrained(cls.model_path)
        cls.hf_config = model.config
        cls.state_dict = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _make_manager(self, tmpdir, save_contents):
        from types import SimpleNamespace

        manager = MegatronCheckpointManager.__new__(MegatronCheckpointManager)
        manager.checkpoint_save_contents = save_contents
        manager.checkpoint_load_contents = save_contents
        manager.checkpoint_config = SimpleNamespace(async_save=False)
        manager.previous_global_step = None
        manager.previous_saved_paths = []
        manager.rank = 0
        manager.model = [FakeModel()]
        manager.optimizer = FakeOptimizer()
        manager.lr_scheduler = None
        manager.get_rng_state = lambda: {"rng": 0}
        manager.peft_cls = None
        manager.bridge = None
        manager.vanilla_bridge = True
        manager.use_dist_checkpointing = True
        manager.use_hf_checkpoint = False
        manager.is_value_model = False
        manager.share_embeddings_and_output_weights = False
        manager.param_dtype = torch.bfloat16
        manager.hf_config = self.hf_config
        manager.transformer_config = None
        manager.config = SimpleNamespace(model=SimpleNamespace(path=self.model_path))
        manager.weight_saver = lambda *args, **kwargs: self.state_dict
        self.tokenizer_saved_to = []
        manager.processing_class = SimpleNamespace(save_pretrained=self.tokenizer_saved_to.append)
        return manager

    def test_hf_model_only_save(self):
        import tempfile

        saved_dist_checkpoints = []
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir, ["hf_model"])
            local_path = os.path.join(tmpdir, "global_step_10", "actor")
            with (
                mock.patch.object(torch.distributed, "barrier", lambda *a, **k: None),
                mock.patch(
                    "verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing",
                    side_effect=lambda **kwargs: saved_dist_checkpoints.append(kwargs),
                ),
            ):
                manager.save_checkpoint(local_path=local_path, global_step=10)

            hf_dir = os.path.join(local_path, "huggingface")
            written = sorted(os.listdir(hf_dir))
            # weights and the config that makes them loadable
            self.assertIn("config.json", written)
            self.assertIn("model.safetensors", written)
            # the tokenizer save is the fix: without it the directory needs the base repo
            self.assertEqual(self.tokenizer_saved_to, [hf_dir])
            # ... and nothing sharded was written at all
            self.assertEqual(saved_dist_checkpoints, [])
            self.assertFalse(os.path.exists(os.path.join(local_path, "dist_ckpt")))
            self.assertEqual(manager.previous_saved_paths, [local_path])
            with open(os.path.join(tmpdir, "latest_checkpointed_iteration.txt")) as f:
                self.assertEqual(f.read(), "10")

    def test_optimizer_only_save_does_not_die_logging_absent_model_keys(self):
        """save_contents without 'model' but with 'optimizer': the dist checkpoint is written,
        and the progress log must ask which model keys exist instead of assuming "model"."""
        import tempfile

        saved_dist_checkpoints = []
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir, ["optimizer"])
            local_path = os.path.join(tmpdir, "global_step_10", "actor")
            with (
                mock.patch.object(torch.distributed, "barrier", lambda *a, **k: None),
                mock.patch(
                    "verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing",
                    side_effect=lambda **kwargs: saved_dist_checkpoints.append(kwargs),
                ),
            ):
                manager.save_checkpoint(local_path=local_path, global_step=10)

            self.assertEqual(len(saved_dist_checkpoints), 1)
            self.assertEqual(sorted(saved_dist_checkpoints[0]["sharded_state_dict"].keys()), ["optimizer"])
            # no weights in any form, so no huggingface metadata either
            self.assertEqual(self.tokenizer_saved_to, [])

    def test_model_contents_still_write_the_dist_checkpoint(self):
        """The other direction: a normal checkpoint is unaffected by the skip."""
        import tempfile

        saved_dist_checkpoints = []
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir, ["model", "optimizer"])
            local_path = os.path.join(tmpdir, "global_step_10", "actor")
            with (
                mock.patch.object(torch.distributed, "barrier", lambda *a, **k: None),
                mock.patch(
                    "verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing",
                    side_effect=lambda **kwargs: saved_dist_checkpoints.append(kwargs),
                ),
            ):
                manager.save_checkpoint(local_path=local_path, global_step=10)

            self.assertEqual(len(saved_dist_checkpoints), 1)
            self.assertEqual(sorted(saved_dist_checkpoints[0]["sharded_state_dict"].keys()), ["model", "optimizer"])
            self.assertEqual(self.tokenizer_saved_to, [os.path.join(local_path, "huggingface")])
