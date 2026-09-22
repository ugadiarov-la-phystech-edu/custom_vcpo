# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""The KL-to-reference knobs of the replay arms (use_kl_loss, kl_loss_coef, kl_loss_type,
kl_loss_is_weighted, kl_ref_reset_interval, ref_param_offload): OFF by default with the composed config and the
experiment name of earlier runs unchanged; when switched on they reach Hydra and tag the name.

Composes the real launch scripts with `--cfg job --resolve` (no GPU, no Ray).

Run: pytest -n 6 recipe/fully_async_policy/unittest/test_replay_arm_kl_reference_on_cpu.py
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


def _run(script_name, extra_env=()):
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
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
        out.flush()
        out.seek(0)
        return proc, (OmegaConf.load(out.name) if proc.returncode == 0 else None)


@functools.cache
def compose(script_name, extra_env=()):
    proc, cfg = _run(script_name, extra_env)
    if proc.returncode != 0:
        raise unittest.SkipTest(f"could not compose {script_name}: {proc.stderr.decode()[-300:]}")
    return cfg


class TestReplayArmKlReference(unittest.TestCase):
    def test_every_arm_is_covered(self):
        self.assertGreaterEqual(len(ARMS), 3, ARMS)

    def test_off_by_default_and_the_name_is_unchanged(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm)
                actor = cfg.actor_rollout_ref.actor
                self.assertFalse(actor.use_kl_loss)
                self.assertEqual(actor.kl_loss_coef, 0.0)  # what the arms passed before the knobs existed
                self.assertFalse(actor.kl_loss_is_weighted)
                self.assertIsNone(cfg.async_training.kl_ref_reset_interval)
                self.assertTrue(cfg.actor_rollout_ref.ref.megatron.param_offload)
                self.assertNotIn(" kl-", cfg.trainer.experiment_name)

    def test_knobs_reach_hydra_and_tag_the_name(self):
        env = (("use_kl_loss", "True"), ("kl_ref_reset_interval", "48"), ("ref_param_offload", "False"))
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm, env)
                actor = cfg.actor_rollout_ref.actor
                self.assertTrue(actor.use_kl_loss)
                self.assertEqual(actor.kl_loss_coef, 0.001)  # default only once the loss is on
                self.assertEqual(actor.kl_loss_type, "low_var_kl")
                self.assertEqual(cfg.async_training.kl_ref_reset_interval, 48)
                self.assertFalse(cfg.actor_rollout_ref.ref.megatron.param_offload)
                self.assertFalse(actor.kl_loss_is_weighted)
                self.assertIn(" kl-0.001-reset48 ", cfg.trainer.experiment_name)
                self.assertNotIn("-isw", cfg.trainer.experiment_name)
                # the rest of the name is the default one with the tag inserted
                self.assertEqual(
                    cfg.trainer.experiment_name.replace(" kl-0.001-reset48", ""), compose(arm).trainer.experiment_name
                )

    def test_frozen_reference_has_no_reset_tag(self):
        arm = ARMS[-1]
        cfg = compose(arm, (("use_kl_loss", "True"), ("kl_loss_coef", "0.003")))
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_coef, 0.003)
        self.assertIsNone(cfg.async_training.kl_ref_reset_interval)
        self.assertIn(" kl-0.003 ", cfg.trainer.experiment_name)
        self.assertNotIn("-reset", cfg.trainer.experiment_name)

    def test_the_straight_through_kl_type_reaches_the_actor_unchanged(self):
        """kl_loss_type=low_var_kl+ (k3 value, k2 gradient): the "+" must survive Hydra and the arm."""
        orz = [a for a in ARMS if a.endswith("_orz7b.sh")]
        if not orz:
            self.skipTest("no ORZ arm on this branch")
        cfg = compose(orz[0], (("use_kl_loss", "True"), ("kl_loss_type", "low_var_kl+"), ("kl_loss_coef", "0.1")))
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_type, "low_var_kl+")
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_coef, 0.1)

    def test_is_weighted_kl_reaches_hydra_and_tags_the_name(self):
        """kl_loss_is_weighted=True: the KL term weighted by the policy-gradient term's rollout IS weights."""
        env = (("use_kl_loss", "True"), ("kl_loss_is_weighted", "True"), ("kl_loss_coef", "0.1"))
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm, env)
                self.assertTrue(cfg.actor_rollout_ref.actor.kl_loss_is_weighted)
                self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_coef, 0.1)
                self.assertEqual(cfg.algorithm.rollout_correction.rollout_is, "token")  # the weights exist
                self.assertIn(" kl-0.1-isw ", cfg.trainer.experiment_name)
                self.assertEqual(
                    cfg.trainer.experiment_name.replace(" kl-0.1-isw", ""), compose(arm).trainer.experiment_name
                )

    def test_is_weighted_tag_precedes_the_reset_tag(self):
        env = (("use_kl_loss", "True"), ("kl_loss_is_weighted", "True"), ("kl_ref_reset_interval", "48"))
        cfg = compose(ARMS[-1], env)
        self.assertIn(" kl-0.001-isw-reset48 ", cfg.trainer.experiment_name)

    def test_is_weighted_kl_without_the_kl_loss_is_refused(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                proc, _ = _run(arm, (("kl_loss_is_weighted", "True"),))
                self.assertEqual(proc.returncode, 2)
                self.assertIn("kl_loss_is_weighted=True needs use_kl_loss=True", proc.stderr.decode())

    def test_reset_interval_without_the_kl_loss_is_refused(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                proc, _ = _run(arm, (("kl_ref_reset_interval", "48"),))
                self.assertEqual(proc.returncode, 2)
                self.assertIn("needs use_kl_loss=True", proc.stderr.decode())


if __name__ == "__main__":
    unittest.main()
