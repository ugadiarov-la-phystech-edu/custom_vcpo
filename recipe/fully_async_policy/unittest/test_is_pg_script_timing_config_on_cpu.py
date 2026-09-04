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
"""The fully-async is-pg baseline arm's timing-related config, composed as a launch
would.

The fully_async/timing/cumulative_training_time series of an async arm is only
directly comparable with a synchronous main_ppo arm (whose helper reports exactly
wall - validation - save) when validation and checkpoint saves freeze the whole
pipeline: async_training.serialize_validation and pause_generation_during_save
both True. This pins those two switches, and that validation / saving are actually
scheduled (rollout.test_freq / trainer.save_freq > 0), for the megatron is-pg script
the sync Qwen3-8B arm is compared against.

Composing runs the real script with ``--cfg job --resolve``; skips if that cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_is_pg_script_timing_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")
SCRIPT = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh"


def compose(script_name):
    path = os.path.join(BASELINE, script_name)
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{script_name} not found")
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet")
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
            ["bash", path, "--cfg", "job", "--resolve"],
            cwd=REPO_ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.PIPE,
            timeout=900,
        )
        if proc.returncode != 0:
            raise unittest.SkipTest(f"could not compose {script_name}: {proc.stderr.decode()[-300:]}")
        out.flush()
        out.seek(0)
        return OmegaConf.load(out.name)


class TestIsPgArmTimingConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SCRIPT)

    def test_stop_the_world_switches_are_on(self):
        a = self.cfg.async_training
        self.assertTrue(a.serialize_validation)
        self.assertTrue(a.pause_generation_during_save)

    def test_validation_and_saving_are_scheduled(self):
        c = self.cfg
        self.assertGreater(int(c.rollout.test_freq), 0)
        self.assertGreater(int(c.trainer.save_freq), 0)

    def test_layout_is_the_5_plus_3_megatron_arm(self):
        c = self.cfg
        self.assertEqual(c.actor_rollout_ref.actor.strategy, "megatron")
        self.assertEqual(c.trainer.n_gpus_per_node + c.rollout.n_gpus_per_node, 8)
        self.assertEqual(c.rollout.n_gpus_per_node, 5)


if __name__ == "__main__":
    unittest.main()
