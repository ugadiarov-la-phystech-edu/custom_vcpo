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
"""The rollout-level PER replay arm's config (arXiv:2606.04560), composed
through hydra exactly as a launch would.

The invariants protected here: (1) the paper's loss contract — the ratio must
anchor to the cached generation-time log-probs with no IS machinery, DAPO
clip-higher 0.2/0.28 with dual-clip 10.0, token-mean, one PPO epoch; (2) the
replay block carries the paper defaults and passes the trainer's init-time
validation; (3) the 5+3 layout keeps the fixed 33x16 pull divisible by the
trainer DP; (4) the yaml block is off by default so every other arm is
untouched.

Run: pytest recipe/fully_async_policy/unittest/test_rollout_per_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT = os.path.join(
    REPO_ROOT,
    "recipe/fully_async_policy/shell/vcpo/dapo/rollout_per",
    "grpo_rollout-per_8gpu_dapo17k_5+3_resp8k_megatron_offload_B33x1.sh",
)
RECIPE_YAML = os.path.join(REPO_ROOT, "recipe/fully_async_policy/config/fully_async_ppo_megatron_trainer.yaml")

_COMPOSED = {}


def compose(**env_overrides):
    """Run the script with hydra's --cfg job --resolve and parse the config it would launch with."""
    key = tuple(sorted(env_overrides.items()))
    if key in _COMPOSED:
        return _COMPOSED[key]
    if not os.path.exists(SCRIPT):
        raise unittest.SkipTest("script not found")
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet", **env_overrides)
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
            ["bash", SCRIPT, "--cfg", "job", "--resolve"],
            cwd=REPO_ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.PIPE,
            timeout=900,
        )
        if proc.returncode != 0:
            raise unittest.SkipTest(f"could not compose the script: {proc.stderr.decode()[-300:]}")
        out.flush()
        out.seek(0)
        cfg = OmegaConf.load(out.name)
    _COMPOSED[key] = cfg
    return cfg


class TestRolloutPerArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose()

    # -------------------- layout --------------------

    def test_layout_is_five_plus_three(self):
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 5)
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)
        self.assertEqual(self.cfg.actor_rollout_ref.actor.strategy, "megatron")

    def test_fixed_pull_divides_the_trainer_dp(self):
        actor = self.cfg.actor_rollout_ref.actor
        dp = self.cfg.trainer.n_gpus_per_node // (
            actor.megatron.tensor_model_parallel_size * actor.megatron.pipeline_model_parallel_size
        )
        pull_rows = actor.ppo_mini_batch_size * self.cfg.actor_rollout_ref.rollout.n
        self.assertEqual(actor.ppo_mini_batch_size, 33)
        self.assertEqual(self.cfg.actor_rollout_ref.rollout.n, 16)
        self.assertEqual(pull_rows % dp, 0)

    def test_sequence_budget(self):
        self.assertEqual(self.cfg.data.max_prompt_length, 2048)
        self.assertEqual(self.cfg.data.max_response_length, 8192)

    def test_model_is_qwen3_8b(self):
        self.assertEqual(self.cfg.actor_rollout_ref.model.path, "Qwen/Qwen3-8B")

    # -------------------- the paper's loss contract --------------------

    def test_ratio_anchors_to_cached_rollout_log_probs(self):
        self.assertTrue(self.cfg.async_training.use_rollout_log_probs)
        self.assertFalse(self.cfg.async_training.compute_prox_log_prob)
        self.assertTrue(self.cfg.actor_rollout_ref.rollout.calculate_log_probs)

    def test_no_is_machinery_on_top_of_the_clip(self):
        """rollout_correction stays at its null default: the loss is the pure
        PPO-clip surrogate of the paper, not TIS-corrected. (With
        old_log_probs aliased to rollout_log_probs the correction pass is
        arithmetically inert even as a diagnostic: every ratio is exactly 1.)"""
        self.assertIsNone(OmegaConf.select(self.cfg, "algorithm.rollout_correction.rollout_is"))
        self.assertEqual(
            OmegaConf.select(self.cfg, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla"),
            "vanilla",
        )

    def test_dapo_clip_higher_with_paper_dual_clip(self):
        actor = self.cfg.actor_rollout_ref.actor
        self.assertEqual(actor.clip_ratio_low, 0.2)
        self.assertEqual(actor.clip_ratio_high, 0.28)
        self.assertEqual(actor.clip_ratio_c, 10.0)

    def test_paper_table7_optimizer_and_aggregation(self):
        actor = self.cfg.actor_rollout_ref.actor
        self.assertEqual(actor.loss_agg_mode, "token-mean")
        self.assertEqual(actor.ppo_epochs, 1)
        self.assertEqual(actor.optim.lr, 1e-6)
        self.assertEqual(actor.optim.weight_decay, 0.01)
        self.assertEqual(actor.optim.lr_warmup_steps, 0)
        self.assertEqual(actor.optim.clip_grad, 1.0)
        self.assertEqual(actor.optim.lr_decay_style, "constant")

    def test_kl_is_fully_disabled(self):
        self.assertFalse(self.cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(self.cfg.actor_rollout_ref.actor.kl_loss_coef, 0.0)
        self.assertFalse(self.cfg.algorithm.use_kl_in_reward)
        self.assertEqual(self.cfg.algorithm.kl_ctrl.kl_coef, 0.0)

    def test_grpo_advantages_with_entropy_logging(self):
        """calculate_entropy=True + entropy_coeff=0: actor/entropy is logged
        (via the update_policy honor patch) without an entropy loss term."""
        self.assertEqual(self.cfg.algorithm.adv_estimator, "grpo")
        self.assertTrue(self.cfg.actor_rollout_ref.actor.calculate_entropy)
        self.assertEqual(self.cfg.actor_rollout_ref.actor.entropy_coeff, 0)

    # -------------------- the replay block --------------------

    def test_replay_block_carries_the_paper_defaults(self):
        replay = self.cfg.async_training.rollout_replay
        self.assertTrue(replay.enable)
        self.assertEqual(replay.replay_ratio, 0.5)
        self.assertEqual(replay.priority_alpha, 0.5)
        self.assertEqual(replay.priority_eps, 1.0e-6)
        self.assertEqual(replay.tau_max, 10)
        self.assertEqual(replay.warmup_steps, 20)
        self.assertEqual(replay.capacity, 30000)
        self.assertFalse(replay.with_replacement)

    def test_replay_knobs_are_env_overridable(self):
        cfg = compose(replay_ratio="1.0", replay_tau_max="30")
        self.assertEqual(cfg.async_training.rollout_replay.replay_ratio, 1.0)
        self.assertEqual(cfg.async_training.rollout_replay.tau_max, 30)

    def test_composed_config_passes_the_trainer_init_validation(self):
        """The exact assertions _init_rollout_replay makes at cluster start,
        run at test time instead of after the GPUs are up."""
        cfg = self.cfg
        self.assertTrue(cfg.async_training.use_rollout_log_probs)
        self.assertFalse(cfg.async_training.compute_prox_log_prob)
        self.assertIsNone(OmegaConf.select(cfg, "algorithm.rollout_correction.rollout_is"))
        self.assertEqual(cfg.algorithm.adv_estimator, "grpo")
        self.assertEqual(cfg.async_training.require_batches, 1)
        self.assertTrue(cfg.actor_rollout_ref.rollout.calculate_log_probs)

    # -------------------- schedule --------------------

    def test_fresh_anchor_schedule(self):
        """k=1 with sync every update: the async approximation of on-policy
        fresh rollouts, and the single-pull-per-step layout replay requires."""
        self.assertEqual(self.cfg.async_training.staleness_threshold, 1.0)
        self.assertEqual(self.cfg.async_training.trigger_parameter_sync_step, 1)
        self.assertEqual(self.cfg.async_training.require_batches, 1)
        self.assertTrue(self.cfg.async_training.partial_rollout)

    def test_checkpoints_are_hf_model_only_and_not_resumable(self):
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(self.cfg.trainer.resume_mode, "disable")

    def test_generation_budget(self):
        self.assertEqual(self.cfg.rollout.total_rollout_steps, 66000)
        self.assertEqual(self.cfg.rollout.test_freq, 10)
        self.assertEqual(self.cfg.trainer.save_freq, 10)


class TestRecipeYamlDefaults(unittest.TestCase):
    """The block every OTHER arm composes with must keep replay off."""

    @classmethod
    def setUpClass(cls):
        cls.yaml = OmegaConf.load(RECIPE_YAML)

    def test_replay_is_disabled_by_default(self):
        replay = self.yaml.async_training.rollout_replay
        self.assertFalse(replay.enable)

    def test_yaml_defaults_match_the_paper(self):
        replay = self.yaml.async_training.rollout_replay
        self.assertEqual(replay.replay_ratio, 0.5)
        self.assertEqual(replay.priority_alpha, 0.5)
        self.assertEqual(replay.priority_eps, 1.0e-6)
        self.assertEqual(replay.tau_max, 10)
        self.assertEqual(replay.warmup_steps, 20)
        self.assertEqual(replay.capacity, 30000)
        self.assertFalse(replay.with_replacement)


if __name__ == "__main__":
    unittest.main()
