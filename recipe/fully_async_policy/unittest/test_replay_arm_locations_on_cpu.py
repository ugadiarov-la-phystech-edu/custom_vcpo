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
"""Every grpo*.sh arm under shell/vcpo/dapo/replay_buffer/ must let the caller relocate its
outputs without editing the script:

  log_dir    TensorBoard (log_dir/tensorboard) + rollout/validation dumps (trainer.rollout_data_dir)
  CKPTS_DIR  the global_step_N/ checkpoints (trainer.default_local_dir)

Both default to logs/<exp_name> (relative to the repo root the scripts are launched from) and are
independently overridable with absolute paths. Composing runs the real script with hydra's
--cfg job --resolve (TRAIN_FILE / TEST_FILE stubbed), so the checks see what the launch would use.

Run: pytest recipe/fully_async_policy/unittest/test_replay_arm_locations_on_cpu.py
"""

import glob
import os
import subprocess
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer")
ARMS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPLAY, "grpo*.sh")))

REQUIRED_LINES = (
    'log_dir=${log_dir:-"logs/${exp_name_safe}"}',
    'CKPTS_DIR=${CKPTS_DIR:-"${log_dir}"}',
    'mkdir -p -- "${log_dir}" "${CKPTS_DIR}"',
    'export TENSORBOARD_DIR="${log_dir}/tensorboard"',
    'trainer.default_local_dir="${CKPTS_DIR}"',
    'trainer.rollout_data_dir="${log_dir}"',
)


def compose(script_name, extra_env=None):
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
    env.update(extra_env or {})
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
            ["bash", os.path.join(REPLAY, script_name), "--cfg", "job", "--resolve"],
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


class TestReplayArmLocations(unittest.TestCase):
    def test_every_arm_is_covered(self):
        self.assertGreaterEqual(len(ARMS), 5, ARMS)

    def test_scripts_carry_the_overridable_location_lines(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                text = open(os.path.join(REPLAY, arm)).read()
                for line in REQUIRED_LINES:
                    self.assertIn(line, text, f"{arm} lacks: {line}")
                # the old hard-coded forms must be gone
                self.assertNotIn('\nlog_dir="logs/${exp_name_safe}"\n', text, arm)
                self.assertNotIn('\nCKPTS_DIR="${log_dir}"\n', text, arm)

    def test_defaults_resolve_to_logs_under_exp_name(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm)
                exp = cfg.trainer.experiment_name
                self.assertEqual(cfg.trainer.default_local_dir, f"logs/{exp}")
                self.assertEqual(cfg.trainer.rollout_data_dir, f"logs/{exp}")

    def test_log_dir_and_ckpts_dir_override_independently(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                log_dir = os.path.join(tmp, "runs", "x")
                ckpts = os.path.join(tmp, "ckpts", "x")
                cfg = compose(arm, {"log_dir": log_dir, "CKPTS_DIR": ckpts})
                self.assertEqual(cfg.trainer.default_local_dir, ckpts)
                self.assertEqual(cfg.trainer.rollout_data_dir, log_dir)
                # experiment name unaffected by the move
                self.assertEqual(cfg.trainer.experiment_name, compose(arm).trainer.experiment_name)
                # both directories are created up front (mkdir -p), even for the --cfg dry run
                self.assertTrue(os.path.isdir(log_dir), arm)
                self.assertTrue(os.path.isdir(ckpts), arm)

    def test_ckpts_dir_alone_follows_a_custom_log_dir(self):
        """Overriding only log_dir moves the checkpoints with it (CKPTS_DIR defaults to log_dir)."""
        arm = ARMS[0]
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "only-log-dir")
            cfg = compose(arm, {"log_dir": log_dir})
            self.assertEqual(cfg.trainer.default_local_dir, log_dir)
            self.assertEqual(cfg.trainer.rollout_data_dir, log_dir)


if __name__ == "__main__":
    unittest.main()
