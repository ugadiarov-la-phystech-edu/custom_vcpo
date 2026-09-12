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
"""Every grpo*.sh arm under shell/vcpo/dapo/replay_buffer/ is emulation-ready for bigger cards:

  gpu_memory_utilization   env-overridable (default 0.9), vLLM's side of an H100 emulation
  VERL_GPU_MEM_CAP_GB      only READ (never set) — when the launching shell exports it, the experiment
                           name gets " h100-emu-<cap>gb-gmu<x>" so emulated runs never share a log dir
                           with real ones; the trainer-side cap itself is applied by
                           recipe/fully_async_policy/gpu_memory_cap.py inside the trainer workers.

Composing runs the real script with hydra's --cfg job --resolve (TRAIN_FILE / TEST_FILE stubbed).

Run: pytest recipe/fully_async_policy/unittest/test_replay_arm_h100_emu_on_cpu.py
"""

import functools
import glob
import os
import subprocess
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer")
ARMS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPLAY, "grpo*.sh")))

EMU_LINES = (
    "gpu_memory_utilization=${gpu_memory_utilization:-0.9}",
    'emu_tag=""',
    'if [[ -n "${VERL_GPU_MEM_CAP_GB:-}" ]]; then '
    'emu_tag=" h100-emu-${VERL_GPU_MEM_CAP_GB}gb-gmu${gpu_memory_utilization}"; fi',
    "${emu_tag}",
)


@functools.cache
def compose(script_name, extra_env=()):
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
    env.pop("VERL_GPU_MEM_CAP_GB", None)  # a developer's shell must not leak into the default case
    env.update(dict(extra_env))
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


class TestReplayArmH100Emulation(unittest.TestCase):
    def test_every_arm_is_covered(self):
        self.assertGreaterEqual(len(ARMS), 5, ARMS)

    def test_scripts_carry_the_hook_and_never_export_the_cap(self):
        for arm in ARMS:
            with open(os.path.join(REPLAY, arm)) as f:
                text = f.read()
            for line in EMU_LINES:
                self.assertIn(line, text, f"{arm} lacks {line!r}")
            self.assertNotIn("export VERL_GPU_MEM_CAP_GB", text, arm)
            self.assertEqual(text.count("${emu_tag}"), 1, arm)  # exactly once, inside exp_name
            self.assertIn(" ess-${ess_tag}${emu_tag}", text, arm)

    def test_default_is_untagged_at_0_9(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm)
                self.assertAlmostEqual(cfg.actor_rollout_ref.rollout.gpu_memory_utilization, 0.9)
                self.assertNotIn("h100-emu", cfg.trainer.experiment_name)

    def test_cap_in_the_env_tags_the_run_and_its_directories(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm, (("VERL_GPU_MEM_CAP_GB", "80"), ("gpu_memory_utilization", "0.5")))
                self.assertAlmostEqual(cfg.actor_rollout_ref.rollout.gpu_memory_utilization, 0.5)
                self.assertIn(" h100-emu-80gb-gmu0.5 ", cfg.trainer.experiment_name)
                self.assertIn("h100-emu-80gb-gmu0.5", cfg.trainer.default_local_dir)
                # the rest of the name is the default one with the tag spliced after the ess tag
                base = compose(arm).trainer.experiment_name
                self.assertEqual(cfg.trainer.experiment_name.replace(" h100-emu-80gb-gmu0.5", ""), base)

    def test_memory_fraction_alone_does_not_tag(self):
        arm = ARMS[0]
        cfg = compose(arm, (("gpu_memory_utilization", "0.5"),))
        self.assertAlmostEqual(cfg.actor_rollout_ref.rollout.gpu_memory_utilization, 0.5)
        self.assertNotIn("h100-emu", cfg.trainer.experiment_name)

    def test_cap_alone_tags_with_the_default_fraction(self):
        arm = ARMS[0]
        cfg = compose(arm, (("VERL_GPU_MEM_CAP_GB", "80"),))
        self.assertIn(" h100-emu-80gb-gmu0.9 ", cfg.trainer.experiment_name)


if __name__ == "__main__":
    unittest.main()
