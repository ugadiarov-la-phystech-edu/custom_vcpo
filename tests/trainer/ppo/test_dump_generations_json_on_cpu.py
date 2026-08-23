# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Dumping rollout generations must survive whatever the reward functions return.

verl/utils/reward_score/math_dapo.py returns `acc` as a numpy bool and the reward managers
forward the scorer's dict verbatim, so a run with trainer.rollout_data_dir set used to die at
its first dump with "Object of type bool_ is not JSON serializable" - after training, before
the first checkpoint.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, json_default


class TestJsonDefault(unittest.TestCase):
    def test_numpy_scalars_unwrap_to_python(self):
        self.assertIs(json_default(np.bool_(True)), True)
        self.assertEqual(json_default(np.float32(0.5)), 0.5)
        self.assertEqual(json_default(np.int64(7)), 7)

    def test_torch_scalars_unwrap(self):
        self.assertEqual(json_default(torch.tensor(3.0)), 3.0)

    def test_arrays_become_lists(self):
        self.assertEqual(json_default(np.array([1, 2])), [1, 2])
        self.assertEqual(json_default(torch.tensor([1.0, 2.0])), [1.0, 2.0])

    def test_sets_become_sorted_lists(self):
        self.assertEqual(json_default({"b", "a"}), ["a", "b"])

    def test_anything_else_falls_back_to_str_instead_of_raising(self):
        class Weird:
            def __repr__(self):
                return "weird"

        self.assertEqual(json_default(Weird()), "weird")

    def test_it_is_usable_as_the_json_default_hook(self):
        payload = {"acc": np.bool_(False), "score": np.float32(1.5)}
        self.assertEqual(json.loads(json.dumps(payload, default=json_default)), {"acc": False, "score": 1.5})


class TestDumpGenerations(unittest.TestCase):
    """The real _dump_generations, with the reward extra infos a dapo run produces."""

    def _dump(self, reward_extra_infos_dict, tmpdir):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.global_steps = 3
        trainer._dump_generations(
            inputs=["prompt a", "prompt b"],
            outputs=["answer a", "answer b"],
            gts=["1", "2"],
            scores=[1.0, 0.0],
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=tmpdir,
        )
        with open(os.path.join(tmpdir, "3.jsonl")) as f:
            return [json.loads(line) for line in f]

    def test_numpy_bool_accuracy_is_dumped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = self._dump({"acc": [np.bool_(True), np.bool_(False)], "pred": ["1", "3"]}, tmpdir)
        self.assertEqual([e["acc"] for e in entries], [True, False])
        self.assertEqual([e["pred"] for e in entries], ["1", "3"])
        self.assertEqual([e["step"] for e in entries], [3, 3])
        self.assertEqual([e["input"] for e in entries], ["prompt a", "prompt b"])

    def test_mixed_numpy_types_are_dumped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = self._dump(
                {"acc": [np.bool_(True), np.bool_(True)], "reward": [np.float32(0.25), np.float64(-1.0)]},
                tmpdir,
            )
        self.assertEqual([e["reward"] for e in entries], [0.25, -1.0])

    def test_extra_infos_of_the_wrong_length_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = self._dump({"acc": [np.bool_(True)]}, tmpdir)  # 1 value for 2 samples
        self.assertNotIn("acc", entries[0])

    def test_plain_python_values_are_unaffected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = self._dump({"acc": [True, False]}, tmpdir)
        self.assertEqual([e["acc"] for e in entries], [True, False])


if __name__ == "__main__":
    unittest.main()
