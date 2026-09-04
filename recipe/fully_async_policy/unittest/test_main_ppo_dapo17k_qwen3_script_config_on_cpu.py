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
"""The synchronous main_ppo Qwen3-8B / DAPO-17k reference arm's config, composed as a
launch would.

Invariants protected here:
  * the 4-updates-per-rollout-step geometry (train_batch_size 128, ppo_mini_batch_size 32,
    ppo_epochs 1, n 16) that makes the PPO clip a working trust region on updates 2-4;
  * the STOCK loss: vanilla mode, symmetric 0.2 band, dual-clip 3.0, token-mean, no KL
    anywhere, no entropy bonus, old_log_probs recomputed (bypass_mode false);
  * the optimizer block (lr 1e-6 constant, no warmup, wd 0.01, clip_grad 1.0);
  * the data / reward wiring (DAPO-17k, both aime parquets, NO custom reward function —
    the built-in math_dapo scorer handles math_dapo and aime* data sources);
  * the colocated-memory knobs (no offload, gpu_memory_utilization 0.5,
    max_num_batched_tokens 10240) and the async-arm-compatible validation sampling.

Composing runs the real script with ``--cfg job --resolve``; skips if that cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_main_ppo_dapo17k_qwen3_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")
SCRIPT = "main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh"

_COMPOSED = {}


def compose(script_name, **env_overrides):
    if script_name in _COMPOSED:
        return _COMPOSED[script_name]
    path = os.path.join(BASELINE, script_name)
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{script_name} not found")
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", **env_overrides)
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
        cfg = OmegaConf.load(out.name)
    _COMPOSED[script_name] = cfg
    return cfg


class TestSyncDapo17kQwen3ArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SCRIPT)

    def test_four_updates_per_rollout_step_geometry(self):
        """128 prompts generated per step, 32-prompt mini-batches, one epoch -> 4 updates."""
        c = self.cfg
        self.assertEqual(c.data.train_batch_size, 128)
        self.assertEqual(c.actor_rollout_ref.actor.ppo_mini_batch_size, 32)
        self.assertEqual(c.data.train_batch_size // c.actor_rollout_ref.actor.ppo_mini_batch_size, 4)
        self.assertEqual(c.data.train_batch_size % c.actor_rollout_ref.actor.ppo_mini_batch_size, 0)
        self.assertEqual(c.actor_rollout_ref.actor.ppo_epochs, 1)
        self.assertEqual(c.actor_rollout_ref.rollout.n, 16)
        self.assertEqual(c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1)
        # 512 sequences per optimizer step split over 8 DP ranks (tp=pp=cp=1).
        dp = c.trainer.n_gpus_per_node * c.trainer.nnodes
        self.assertEqual((c.actor_rollout_ref.actor.ppo_mini_batch_size * c.actor_rollout_ref.rollout.n) % dp, 0)

    def test_stock_ppo_loss_defaults(self):
        """verl's default vanilla loss: symmetric 0.2 band, dual-clip 3.0, token-mean."""
        c = self.cfg
        self.assertEqual(c.actor_rollout_ref.actor.policy_loss.loss_mode, "vanilla")
        self.assertEqual(float(c.actor_rollout_ref.actor.clip_ratio), 0.2)
        self.assertEqual(float(c.actor_rollout_ref.actor.clip_ratio_low), 0.2)
        self.assertEqual(float(c.actor_rollout_ref.actor.clip_ratio_high), 0.2)
        self.assertEqual(float(c.actor_rollout_ref.actor.clip_ratio_c), 3.0)
        self.assertEqual(c.actor_rollout_ref.actor.loss_agg_mode, "token-mean")

    def test_no_kl_no_entropy_anywhere(self):
        """No KL in reward, no KL loss, no entropy bonus — but entropy IS logged."""
        c = self.cfg
        self.assertFalse(c.algorithm.use_kl_in_reward)
        self.assertFalse(c.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(float(c.actor_rollout_ref.actor.entropy_coeff), 0.0)
        self.assertTrue(c.actor_rollout_ref.actor.calculate_entropy)

    def test_grpo_without_critic(self):
        self.assertEqual(self.cfg.algorithm.adv_estimator, "grpo")

    def test_optimizer_block(self):
        c = self.cfg
        self.assertEqual(float(c.actor_rollout_ref.actor.optim.lr), 1e-6)
        self.assertEqual(float(c.actor_rollout_ref.actor.optim.weight_decay), 0.01)
        self.assertEqual(int(c.actor_rollout_ref.actor.optim.lr_warmup_steps), 0)
        self.assertEqual(c.actor_rollout_ref.actor.optim.lr_decay_style, "constant")
        self.assertEqual(float(c.actor_rollout_ref.actor.optim.clip_grad), 1.0)

    def test_sync_megatron_hybrid_engine(self):
        c = self.cfg
        self.assertTrue(c.actor_rollout_ref.hybrid_engine)
        self.assertEqual(c.actor_rollout_ref.actor.strategy, "megatron")
        self.assertEqual(c.actor_rollout_ref.actor.megatron.tensor_model_parallel_size, 1)
        self.assertEqual(c.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size, 1)
        self.assertEqual(c.actor_rollout_ref.actor.megatron.context_parallel_size, 1)
        self.assertEqual(c.trainer.n_gpus_per_node, 8)
        self.assertEqual(c.trainer.nnodes, 1)

    def test_model_and_sampling(self):
        c = self.cfg
        self.assertEqual(c.actor_rollout_ref.model.path, "Qwen/Qwen3-8B")
        self.assertEqual(float(c.actor_rollout_ref.rollout.temperature), 1.0)
        self.assertEqual(float(c.actor_rollout_ref.rollout.top_p), 1.0)
        self.assertEqual(c.data.max_prompt_length, 2048)
        self.assertEqual(c.data.max_response_length, 8192)
        self.assertEqual(c.data.prompt_key, "prompt")

    def test_validation_sampling_matches_async_arms(self):
        """Same val_kwargs as every Qwen3-8B async arm -> comparable val-core curves."""
        v = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertEqual(float(v.temperature), 0.8)
        self.assertEqual(float(v.top_p), 0.7)
        self.assertEqual(v.top_k, -1)
        self.assertTrue(v.do_sample)
        self.assertEqual(v.n, 1)

    def test_data_and_builtin_reward(self):
        """DAPO-17k train, both aime parquets for val, and NO custom scorer: the built-in
        router sends math_dapo / aime* data sources to the math_dapo scorer (+1 / -1)."""
        c = self.cfg
        self.assertEqual(c.data.train_files, "/tmp/train.parquet")  # env override in compose()
        val_files = list(c.data.val_files)
        self.assertEqual(len(val_files), 2)
        self.assertTrue(val_files[0].endswith("/dapo/aime-2024.parquet"), val_files)
        self.assertTrue(val_files[1].endswith("/dapo/aime-2025.parquet"), val_files)
        self.assertIsNone(c.custom_reward_function.path)

    def test_diagnostics_on_correction_off(self):
        """rollout_log_probs cached for rollout_corr/* metrics only; old_log_probs are
        recomputed by the trainer (decoupled mode), no IS weights in the loss."""
        c = self.cfg
        self.assertTrue(c.actor_rollout_ref.rollout.calculate_log_probs)
        self.assertIsNone(c.algorithm.rollout_correction.rollout_is)
        self.assertFalse(c.algorithm.rollout_correction.bypass_mode)

    def test_colocated_memory_knobs(self):
        """Resident trainer (no offload) caps vLLM at 0.5 of the GPU; prefill budget 10240."""
        c = self.cfg
        m = c.actor_rollout_ref.actor.megatron
        self.assertFalse(m.param_offload)
        self.assertFalse(m.optimizer_offload)
        self.assertFalse(m.grad_offload)
        self.assertEqual(float(c.actor_rollout_ref.rollout.gpu_memory_utilization), 0.5)
        self.assertEqual(c.actor_rollout_ref.rollout.max_num_batched_tokens, 10240)
        self.assertTrue(c.actor_rollout_ref.rollout.enable_chunked_prefill)
        self.assertEqual(c.actor_rollout_ref.rollout.tensor_model_parallel_size, 1)

    def test_single_seed_feeds_every_seed_knob(self):
        """SEED (default 1) reaches every seed main_ppo exposes, and the derived
        ref/critic megatron seeds follow via oc.select. vLLM's engine seed is not a
        config field (RolloutConfig has none), so it is deliberately not asserted."""
        c = self.cfg
        self.assertEqual(c.data.seed, 1)
        self.assertEqual(c.actor_rollout_ref.actor.megatron.seed, 1)
        self.assertEqual(c.actor_rollout_ref.actor.data_loader_seed, 1)
        self.assertEqual(c.actor_rollout_ref.ref.megatron.seed, 1)
        self.assertEqual(c.critic.megatron.seed, 1)
        self.assertNotIn("seed", c.actor_rollout_ref.rollout)
        self.assertTrue(c.trainer.experiment_name.endswith(" seed-1"), c.trainer.experiment_name)

    def test_seed_override_propagates(self):
        """SEED=7 in the environment moves all three knobs and the exp_name tag together."""
        env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", SEED="7")
        path = os.path.join(BASELINE, SCRIPT)
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
                raise unittest.SkipTest(f"could not compose with SEED=7: {proc.stderr.decode()[-300:]}")
            out.flush()
            out.seek(0)
            c = OmegaConf.load(out.name)
        self.assertEqual(c.data.seed, 7)
        self.assertEqual(c.actor_rollout_ref.actor.megatron.seed, 7)
        self.assertEqual(c.actor_rollout_ref.actor.data_loader_seed, 7)
        self.assertEqual(c.actor_rollout_ref.ref.megatron.seed, 7)
        self.assertTrue(c.trainer.experiment_name.endswith(" seed-7"), c.trainer.experiment_name)

    def test_checkpoint_policy(self):
        c = self.cfg
        self.assertEqual(list(c.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(c.trainer.resume_mode, "disable")
        self.assertIsNone(c.trainer.max_actor_ckpt_to_keep)
        self.assertEqual(c.trainer.save_freq, 2)
        self.assertEqual(c.trainer.test_freq, 2)
        self.assertTrue(c.trainer.val_before_train)


if __name__ == "__main__":
    unittest.main()
