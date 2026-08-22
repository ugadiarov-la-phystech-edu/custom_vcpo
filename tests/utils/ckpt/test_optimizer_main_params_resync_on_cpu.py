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
"""Resyncing an optimizer's master params after a model-only checkpoint load.

Megatron's mixed-precision/distributed optimizers hold fp32 master copies made when the optimizer
is built, i.e. before any checkpoint is loaded. Restoring the model without the optimizer leaves
them stale, and the first step() copies them back over the restored weights.
"""

import unittest

from verl.utils.checkpoint.checkpoint_manager import resync_optimizer_main_params


class FakeMegatronOptimizer:
    """Exposes reload_model_params, like MegatronOptimizer/ChainedOptimizer."""

    def __init__(self):
        self.reload_calls = 0

    def reload_model_params(self):
        self.reload_calls += 1


class FakeTorchOptimizer:
    """Steps on the model parameters themselves (torch/FSDP): no master copies, no hook."""


class TestResyncOptimizerMainParams(unittest.TestCase):
    def test_model_only_load_reloads_the_master_params(self):
        optimizer = FakeMegatronOptimizer()
        self.assertTrue(resync_optimizer_main_params(optimizer, loaded_model=True, loaded_optimizer=False))
        self.assertEqual(optimizer.reload_calls, 1)

    def test_optimizer_was_loaded_so_its_master_params_are_already_correct(self):
        optimizer = FakeMegatronOptimizer()
        self.assertFalse(resync_optimizer_main_params(optimizer, loaded_model=True, loaded_optimizer=True))
        self.assertEqual(optimizer.reload_calls, 0)

    def test_no_model_was_loaded(self):
        """save/load_contents=['hf_model']: the weights in memory are the ones the optimizer knows."""
        optimizer = FakeMegatronOptimizer()
        self.assertFalse(resync_optimizer_main_params(optimizer, loaded_model=False, loaded_optimizer=False))
        self.assertEqual(optimizer.reload_calls, 0)

    def test_no_optimizer_at_all(self):
        self.assertFalse(resync_optimizer_main_params(None, loaded_model=True, loaded_optimizer=False))

    def test_optimizer_without_master_copies_is_left_alone(self):
        self.assertFalse(resync_optimizer_main_params(FakeTorchOptimizer(), loaded_model=True, loaded_optimizer=False))

    def test_repeated_loads_each_resync(self):
        optimizer = FakeMegatronOptimizer()
        for _ in range(3):
            resync_optimizer_main_params(optimizer, loaded_model=True, loaded_optimizer=False)
        self.assertEqual(optimizer.reload_calls, 3)


if __name__ == "__main__":
    unittest.main()
