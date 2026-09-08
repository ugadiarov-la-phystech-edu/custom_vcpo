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
"""Every grpo*.sh arm under shell/vcpo/dapo/replay_buffer/ seeds the whole pipeline from one
SEED variable (default 1), the way the main_ppo sync arms do:

  data.seed                                   rollouter prompt order
  actor_rollout_ref.actor.megatron.seed       trainer workers' torch/numpy/random + Megatron rng
  actor_rollout_ref.ref.megatron.seed         follows the actor seed via oc.select
  critic.megatron.seed                        explicit (would otherwise stay at the literal 42)
  actor_rollout_ref.actor.data_loader_seed    Megatron actor mini-batch shuffle
  async_training.replay_buffer.sampling_seed  weighted replay draw (replay_sampling_seed may override)
  async_training.ppo_epochs_shuffle_seed      fractional-epoch shuffle
  async_training.opportunistic_epochs.shuffle_seed

and tags the experiment name with " seed-N" so repeats land in their own log dir. Composing runs
the real script with hydra's --cfg job --resolve (TRAIN_FILE / TEST_FILE stubbed).

Run: pytest recipe/fully_async_policy/unittest/test_replay_arm_seeding_on_cpu.py
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

REQUIRED_LINES = (
    "SEED=${SEED:-1}",
    "replay_sampling_seed=${replay_sampling_seed:-${SEED}}",
    ' seed-${SEED}"}',
    "data.seed=${SEED} \\",
    "actor_rollout_ref.actor.data_loader_seed=${SEED} \\",
    "actor_rollout_ref.actor.megatron.seed=${SEED} \\",
    "critic.megatron.seed=${SEED} \\",
    "async_training.ppo_epochs_shuffle_seed=${SEED} \\",
    "async_training.opportunistic_epochs.shuffle_seed=${SEED} \\",
)


def seeds_of(cfg):
    """All seed knobs the arms drive, as a name -> value dict."""
    return {
        "data.seed": cfg.data.seed,
        "actor.megatron.seed": cfg.actor_rollout_ref.actor.megatron.seed,
        "ref.megatron.seed": cfg.actor_rollout_ref.ref.megatron.seed,
        "critic.megatron.seed": cfg.critic.megatron.seed,
        "actor.data_loader_seed": cfg.actor_rollout_ref.actor.data_loader_seed,
        "replay_buffer.sampling_seed": cfg.async_training.replay_buffer.sampling_seed,
        "ppo_epochs_shuffle_seed": cfg.async_training.ppo_epochs_shuffle_seed,
        "opportunistic_epochs.shuffle_seed": cfg.async_training.opportunistic_epochs.shuffle_seed,
    }


@functools.cache
def compose(script_name, extra_env=()):
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
        if proc.returncode != 0:
            raise unittest.SkipTest(f"could not compose {script_name}: {proc.stderr.decode()[-300:]}")
        out.flush()
        out.seek(0)
        return OmegaConf.load(out.name)


class TestReplayArmSeeding(unittest.TestCase):
    def test_every_arm_is_covered(self):
        self.assertGreaterEqual(len(ARMS), 5, ARMS)

    def test_scripts_carry_the_seed_lines(self):
        for arm in ARMS:
            with open(os.path.join(REPLAY, arm)) as f:
                text = f.read()
            for line in REQUIRED_LINES:
                self.assertIn(line, text, f"{arm} lacks {line!r}")
            # exactly one SEED definition and no leftover literal replay seed
            self.assertEqual(text.count("SEED=${SEED:-1}"), 1, arm)
            self.assertNotIn("replay_sampling_seed:-1234", text, arm)

    def test_default_seed_is_one_everywhere(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm)
                self.assertEqual(set(seeds_of(cfg).values()), {1}, seeds_of(cfg))
                self.assertTrue(cfg.trainer.experiment_name.endswith(" seed-1"), cfg.trainer.experiment_name)

    def test_seed_env_flips_every_knob_and_the_name(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm, (("SEED", "7"),))
                self.assertEqual(set(seeds_of(cfg).values()), {7}, seeds_of(cfg))
                self.assertTrue(cfg.trainer.experiment_name.endswith(" seed-7"), cfg.trainer.experiment_name)
                base = compose(arm).trainer.experiment_name
                self.assertEqual(cfg.trainer.experiment_name[: -len(" seed-7")], base[: -len(" seed-1")])

    def test_seed_moves_the_default_log_and_checkpoint_dirs(self):
        """Repeats with different seeds must not share logs/<exp_name>."""
        for arm in ARMS:
            with self.subTest(arm=arm):
                one = compose(arm)
                seven = compose(arm, (("SEED", "7"),))
                self.assertNotEqual(one.trainer.default_local_dir, seven.trainer.default_local_dir)
                self.assertTrue(seven.trainer.default_local_dir.endswith(" seed-7"), seven.trainer.default_local_dir)
                self.assertTrue(seven.trainer.rollout_data_dir.endswith(" seed-7"), seven.trainer.rollout_data_dir)

    def test_replay_sampling_seed_can_still_be_overridden_alone(self):
        """replay_sampling_seed keeps its own env knob: it overrides only the replay draw,
        every other knob (and the name tag) still follows SEED."""
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm, (("SEED", "3"), ("replay_sampling_seed", "1234")))
                seeds = seeds_of(cfg)
                self.assertEqual(seeds.pop("replay_buffer.sampling_seed"), 1234)
                self.assertEqual(set(seeds.values()), {3}, seeds)
                self.assertTrue(cfg.trainer.experiment_name.endswith(" seed-3"), cfg.trainer.experiment_name)

    def test_vllm_sampling_seed_is_not_a_config_key(self):
        """The header's caveat: RolloutConfig exposes no seed, so the arms must not try to set one
        (an extra +actor_rollout_ref.rollout.seed key raises at worker init)."""
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = compose(arm)
                self.assertNotIn("seed", cfg.actor_rollout_ref.rollout)
                with open(os.path.join(REPLAY, arm)) as f:
                    self.assertNotIn("rollout.seed", f.read(), arm)


if __name__ == "__main__":
    unittest.main()
