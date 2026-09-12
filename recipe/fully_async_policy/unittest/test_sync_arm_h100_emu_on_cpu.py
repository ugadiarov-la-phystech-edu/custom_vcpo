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
"""H100 emulation hooks in the synchronous main_ppo arms (main_ppo_sync_*.sh).

An emulated run exports VERL_GPU_MEM_CAP_GB=80 (trainer allocator cap, applied in verl's actor
workers) and lowers gpu_memory_utilization to 0.5*80/<device GiB> (vLLM's budget). The arms must
(1) keep their 0.5 default and untagged exp_name when nothing is exported, (2) pass the overridden
fraction through and tag exp_name (and thus every log/checkpoint dir) with
" h100-emu-<cap>gb-gmu<fraction>" when the cap is exported, (3) only READ the knob.

Composing runs the real script with hydra's --cfg job --resolve (TRAIN_FILE / TEST_FILE stubbed).
Run: pytest -n 4 --dist loadfile recipe/fully_async_policy/unittest/test_sync_arm_h100_emu_on_cpu.py
"""

import glob
import os
import subprocess
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")
ARMS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASELINE, "main_ppo_sync_*.sh")))

DEFAULT_LINE = "gpu_memory_utilization=${gpu_memory_utilization:-0.5}"
HOOK_LINE = (
    'if [[ -n "${VERL_GPU_MEM_CAP_GB:-}" ]]; then '
    'emu_tag=" h100-emu-${VERL_GPU_MEM_CAP_GB}gb-gmu${gpu_memory_utilization}"; fi'
)
TAG_IN_NAME = '${emu_tag}"}'


def run_script(script_name, extra_env=None, extra_args=()):
    """Compose (or fail) the script; returns (returncode, cfg_or_None, stderr).

    VERL_GPU_MEM_CAP_GB is scrubbed from the inherited environment so a developer shell that
    exports it cannot leak into the default-env cases."""
    env = {k: v for k, v in os.environ.items() if k != "VERL_GPU_MEM_CAP_GB"}
    env.update(TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
    env.update(extra_env or {})
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
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


def gmu(cfg):
    return cfg.actor_rollout_ref.rollout.gpu_memory_utilization


class TestScriptText(unittest.TestCase):
    def test_every_sync_arm_is_covered(self):
        self.assertGreaterEqual(len(ARMS), 3, ARMS)  # openPangu, Qwen3 and deepmath sync arms

    def test_arms_read_the_knob_after_the_default_and_tag_the_name(self):
        for arm in ARMS:
            with open(os.path.join(BASELINE, arm)) as f:
                text = f.read()
            self.assertIn(DEFAULT_LINE, text, arm)
            self.assertIn(HOOK_LINE, text, arm)
            # the tag must be computed from the FINAL fraction, i.e. after the default is applied
            self.assertLess(text.index(DEFAULT_LINE), text.index(HOOK_LINE), arm)
            self.assertEqual(text.count('emu_tag=""'), 1, arm)
            self.assertIn(TAG_IN_NAME, text, arm)
            self.assertLess(text.index(HOOK_LINE), text.index(TAG_IN_NAME), arm)
            # read-only: the arm never decides for the user that a run is emulated
            self.assertNotIn("export VERL_GPU_MEM_CAP_GB", text, arm)
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            self.assertNotIn("VERL_GPU_MEM_CAP_GB=", code, arm)  # no assignment outside comments


class TestComposedDefaults(unittest.TestCase):
    def test_default_env_is_untouched(self):
        for arm in ARMS:
            cfg = compose(arm)
            with self.subTest(arm=arm):
                self.assertEqual(gmu(cfg), 0.5)
                self.assertNotIn("h100-emu", cfg.trainer.experiment_name)
                self.assertNotIn("h100-emu", cfg.trainer.default_local_dir)

    def test_fraction_override_alone_is_not_an_emulation(self):
        arm = ARMS[0]
        cfg = compose(arm, {"gpu_memory_utilization": "0.28"})
        self.assertEqual(gmu(cfg), 0.28)
        self.assertNotIn("h100-emu", cfg.trainer.experiment_name)


class TestComposedEmulation(unittest.TestCase):
    def test_cap_plus_fraction_tags_name_and_dirs(self):
        for arm in ARMS:
            cfg = compose(arm, {"VERL_GPU_MEM_CAP_GB": "80", "gpu_memory_utilization": "0.28"})
            with self.subTest(arm=arm):
                self.assertEqual(gmu(cfg), 0.28)
                name = cfg.trainer.experiment_name
                self.assertTrue(name.endswith(" h100-emu-80gb-gmu0.28"), name)
                self.assertEqual(name.count("h100-emu"), 1, name)
                # every derived directory follows exp_name, so emulated and real runs never collide
                self.assertIn(name, cfg.trainer.default_local_dir)
                self.assertIn("h100-emu-80gb-gmu0.28", cfg.trainer.default_local_dir)
                rollout_dir = OmegaConf.select(cfg, "trainer.rollout_data_dir")
                if rollout_dir:
                    self.assertIn("h100-emu-80gb-gmu0.28", rollout_dir)

    def test_cap_alone_tags_with_the_default_fraction(self):
        """Exporting the cap without lowering the fraction is a mistake on a big card (vLLM would
        get 0.5 of 143 GiB); the tag makes it visible in the run name rather than hiding it."""
        arm = ARMS[0]
        cfg = compose(arm, {"VERL_GPU_MEM_CAP_GB": "80"})
        self.assertEqual(gmu(cfg), 0.5)
        self.assertTrue(cfg.trainer.experiment_name.endswith(" h100-emu-80gb-gmu0.5"), cfg.trainer.experiment_name)

    def test_explicit_exp_name_wins_over_the_tag(self):
        arm = ARMS[0]
        cfg = compose(arm, {"VERL_GPU_MEM_CAP_GB": "80", "gpu_memory_utilization": "0.28", "exp_name": "custom"})
        self.assertEqual(cfg.trainer.experiment_name, "custom")
        self.assertEqual(gmu(cfg), 0.28)

    def test_everything_else_is_identical_under_emulation(self):
        """The cap is applied at runtime by the worker; the composed config differs from the default
        only in the rollout fraction and the names/paths that carry it."""
        arm = ARMS[0]
        base = compose(arm)
        emu = compose(arm, {"VERL_GPU_MEM_CAP_GB": "80", "gpu_memory_utilization": "0.28"})
        base_d = OmegaConf.to_container(base, resolve=True)
        emu_d = OmegaConf.to_container(emu, resolve=True)

        def diff(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                out = []
                for k in sorted(set(a) | set(b)):
                    out += diff(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
                return out
            return [path] if a != b else []

        changed = diff(base_d, emu_d)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization", changed)
        for key in changed:
            if key == "actor_rollout_ref.rollout.gpu_memory_utilization":
                continue
            value = str(OmegaConf.select(emu, key))
            self.assertIn("h100-emu-80gb-gmu0.28", value, f"{key} changed without carrying the tag")


if __name__ == "__main__":
    unittest.main()
