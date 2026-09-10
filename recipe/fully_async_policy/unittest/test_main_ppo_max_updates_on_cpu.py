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
"""Every synchronous main_ppo arm under shell/vcpo/dapo/baseline/ (main_ppo_sync_*.sh) accepts an
env knob ``max_updates`` — a cap on OPTIMIZER UPDATES. One rollout step runs
train_batch_size / ppo_mini_batch_size * ppo_epochs updates (4 on the B128/mini32 arms, 1 on the
deepmath B32 arm), so the script rounds the cap UP to whole rollout steps and passes it as verl's
trainer.total_training_steps, whose last step validates, saves and ends the run
(verl/trainer/ppo/ray_trainer.py is_last_step). null (the default) leaves total_epochs in charge.

Composing runs the real script with hydra's --cfg job --resolve (TRAIN_FILE / TEST_FILE stubbed).

Run: pytest recipe/fully_async_policy/unittest/test_main_ppo_max_updates_on_cpu.py
"""

import glob
import math
import os
import subprocess
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")
ARMS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASELINE, "main_ppo_sync_*.sh")))

KNOB_LINE = "max_updates=${max_updates:-null}"
HYDRA_LINE = "trainer.total_training_steps=${total_training_steps} \\"
PASSTHROUGH_LINE = 'trainer.total_epochs=${total_epochs} "$@"'


def run_script(script_name, extra_env=None, extra_args=()):
    """Compose (or fail) the script; returns (returncode, cfg_or_None, stderr)."""
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
    env.update(extra_env or {})
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
            # extra overrides go BEFORE the hydra flags: hydra's argparse takes positionals in one
            # group, and the script appends "$@" after its own overrides (the OOM smoke does the same)
            ["bash", os.path.join(BASELINE, script_name), *extra_args, "--cfg", "job", "--resolve"],
            cwd=REPO_ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.PIPE,
            timeout=900,
        )
        if proc.returncode != 0:
            return proc.returncode, None, proc.stderr.decode()
        out.flush()
        out.seek(0)
        return 0, OmegaConf.load(out.name), ""


def compose(script_name, extra_env=None, extra_args=()):
    code, cfg, err = run_script(script_name, extra_env, extra_args)
    if code != 0:
        raise unittest.SkipTest(f"could not compose {script_name}: {err[-300:]}")
    return cfg


def updates_per_step(cfg):
    actor = cfg.actor_rollout_ref.actor
    return cfg.data.train_batch_size // actor.ppo_mini_batch_size * actor.ppo_epochs


class TestMaxUpdatesKnob(unittest.TestCase):
    def test_every_sync_arm_is_covered(self):
        self.assertGreaterEqual(len(ARMS), 3, ARMS)  # openPangu, Qwen3 and deepmath sync arms here

    def test_scripts_carry_the_knob_and_the_hydra_line_before_the_passthrough(self):
        for arm in ARMS:
            with open(os.path.join(BASELINE, arm)) as f:
                text = f.read()
            for line in (KNOB_LINE, HYDRA_LINE, PASSTHROUGH_LINE):
                self.assertIn(line, text, f"{arm} lacks {line!r}")
            self.assertIn("updates_per_step=$(( train_prompt_bsz /", text, arm)
            # a trailing trainer.total_training_steps=N in "$@" must come AFTER ours to win
            self.assertLess(text.index(HYDRA_LINE), text.index(PASSTHROUGH_LINE), arm)
            self.assertEqual(text.count(KNOB_LINE), 1, arm)

    def test_default_leaves_total_epochs_in_charge(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm)
                self.assertIsNone(cfg.trainer.total_training_steps)
                self.assertEqual(cfg.trainer.total_epochs, 3)
                self.assertGreaterEqual(updates_per_step(cfg), 1)

    def test_cap_is_converted_to_rollout_steps_rounding_up(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                ups = updates_per_step(compose(arm))
                for cap in (200, 7, 1, ups, ups + 1):
                    cfg = compose(arm, {"max_updates": str(cap)})
                    self.assertEqual(cfg.trainer.total_training_steps, math.ceil(cap / ups), (cap, ups))
                    self.assertEqual(cfg.trainer.total_epochs, 3)  # untouched

    def test_geometry_of_the_arms_is_what_the_docs_say(self):
        """4 updates per rollout step on the B128/mini32 arms, 1 on the deepmath B32 arm."""
        for arm in ARMS:
            with self.subTest(arm=arm):
                ups = updates_per_step(compose(arm))
                self.assertEqual(ups, 1 if "B32xn16" in arm else 4, arm)
                cfg = compose(arm, {"max_updates": "200"})
                self.assertEqual(cfg.trainer.total_training_steps, 200 // ups)

    def test_cap_follows_batch_geometry_overrides(self):
        """A mini-batch or ppo_epochs override changes updates_per_step and hence the conversion."""
        arm = next(a for a in ARMS if "B128" in a)
        cfg = compose(arm, {"max_updates": "40", "train_prompt_mini_bsz": "64"})
        self.assertEqual(updates_per_step(cfg), 2)
        self.assertEqual(cfg.trainer.total_training_steps, 20)
        cfg = compose(arm, {"max_updates": "40", "ppo_epochs": "2"})
        self.assertEqual(updates_per_step(cfg), 8)
        self.assertEqual(cfg.trainer.total_training_steps, 5)

    def test_explicit_cli_override_beats_the_knob(self):
        """The OOM smoke appends trainer.total_training_steps=2 after the script's own line."""
        arm = ARMS[0]
        for env in ({"max_updates": "200"}, {}):
            code, cfg, err = run_script(arm, env, extra_args=("trainer.total_training_steps=2",))
            self.assertEqual(code, 0, err[-300:])
            self.assertEqual(cfg.trainer.total_training_steps, 2)

    def test_invalid_caps_fail_fast(self):
        arm = ARMS[0]
        for bad in ("0", "-3", "2.5", "many"):
            with self.subTest(bad=bad):
                code, cfg, err = run_script(arm, {"max_updates": bad})
                self.assertNotEqual(code, 0, bad)
                self.assertIsNone(cfg)
                self.assertIn("max_updates must be a positive integer or null", err, bad)

    def test_degenerate_geometry_fails_fast(self):
        """mini-batch larger than the batch would give 0 updates per step: refuse to launch."""
        arm = next(a for a in ARMS if "B128" in a)
        code, cfg, err = run_script(arm, {"train_prompt_mini_bsz": "256"})
        self.assertNotEqual(code, 0)
        self.assertIn("updates_per_step must be >= 1", err)


if __name__ == "__main__":
    unittest.main()
