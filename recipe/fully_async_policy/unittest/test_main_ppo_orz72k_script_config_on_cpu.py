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
"""The synchronous main_ppo ORZ-72k continuation arm's config, composed as a launch would.

The invariant protected here is the strictly-on-policy geometry that makes this the
ORZ-parity arm: train_batch_size == ppo_mini_batch_size with ppo_epochs=1 means exactly
one optimizer step per generation batch — the regime ORZ-7B was originally trained in.
Any drift in those three knobs silently turns the arm into multi-step off-policy PPO,
which is a different experiment. The second invariant is the schedule parity block
(wd=0, no warmup, constant lr 1e-6, no KL anywhere, no entropy bonus) and the wiring
of the tiered ORZ scorer + ORZ-prompt validation parquets.

Composing runs the real script with ``--cfg job --resolve``; skips if that cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_main_ppo_orz72k_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")
SCRIPT = "main_ppo_sync_8gpu_orz72k_grpo_B32xn32_orz7b.sh"

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


class TestSyncOrz72kArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SCRIPT)

    def test_strictly_on_policy_geometry(self):
        """One optimizer step per generation batch: batch == minibatch, one epoch."""
        c = self.cfg
        self.assertEqual(c.data.train_batch_size, 32)
        self.assertEqual(c.actor_rollout_ref.actor.ppo_mini_batch_size, 32)
        self.assertEqual(c.data.train_batch_size, c.actor_rollout_ref.actor.ppo_mini_batch_size)
        self.assertEqual(c.actor_rollout_ref.actor.ppo_epochs, 1)
        self.assertEqual(c.actor_rollout_ref.rollout.n, 32)

    def test_orz_schedule_parity(self):
        c = self.cfg
        self.assertEqual(float(c.actor_rollout_ref.actor.optim.lr), 1e-6)
        self.assertEqual(float(c.actor_rollout_ref.actor.optim.weight_decay), 0.0)
        self.assertEqual(int(c.actor_rollout_ref.actor.optim.lr_warmup_steps), 0)
        self.assertEqual(c.actor_rollout_ref.actor.optim.lr_decay_style, "constant")
        self.assertEqual(float(c.actor_rollout_ref.actor.optim.clip_grad), 1.0)

    def test_no_kl_no_entropy_anywhere(self):
        """ORZ parity: no KL in reward, no KL loss, no entropy bonus — but entropy IS logged."""
        c = self.cfg
        self.assertFalse(c.algorithm.use_kl_in_reward)
        self.assertFalse(c.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(float(c.actor_rollout_ref.actor.entropy_coeff), 0.0)
        self.assertTrue(c.actor_rollout_ref.actor.calculate_entropy)

    def test_grpo_without_critic(self):
        self.assertEqual(self.cfg.algorithm.adv_estimator, "grpo")

    def test_token_mean_aggregation(self):
        """ORZ parity: their packed PolicyLoss is effectively a global token mean
        (action_mask is None under packing), i.e. verl's token-mean — not the
        house seq-mean-token-mean the async arms use."""
        self.assertEqual(self.cfg.actor_rollout_ref.actor.loss_agg_mode, "token-mean")

    def test_clip_higher_band(self):
        """Inert at ratio==1, but the protective default if geometry ever changes."""
        c = self.cfg
        self.assertEqual(float(c.actor_rollout_ref.actor.clip_ratio_low), 0.2)
        self.assertEqual(float(c.actor_rollout_ref.actor.clip_ratio_high), 0.28)

    def test_sync_megatron_hybrid_engine(self):
        c = self.cfg
        self.assertTrue(c.actor_rollout_ref.hybrid_engine)
        self.assertEqual(c.actor_rollout_ref.actor.strategy, "megatron")
        self.assertEqual(c.actor_rollout_ref.actor.megatron.tensor_model_parallel_size, 1)
        self.assertEqual(c.trainer.n_gpus_per_node, 8)
        self.assertEqual(c.trainer.nnodes, 1)

    def test_model_and_sampling(self):
        c = self.cfg
        self.assertEqual(c.actor_rollout_ref.model.path, "Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
        self.assertEqual(float(c.actor_rollout_ref.rollout.temperature), 1.0)
        self.assertEqual(float(c.actor_rollout_ref.rollout.top_p), 1.0)
        self.assertEqual(c.data.max_prompt_length, 2048)
        self.assertEqual(c.data.max_response_length, 8192)
        self.assertEqual(c.data.prompt_key, "prompt")

    def test_diagnostics_on_correction_off(self):
        """rollout_log_probs are cached for the rollout_corr/* diagnostic metrics, but NO
        importance correction is applied — the loss stays the uncorrected ORZ-parity
        objective. The helper adds weights only when rollout_is is set (rollout_corr_helper:
        'Metrics can be monitored before enabling IS weight correction')."""
        c = self.cfg
        self.assertTrue(c.actor_rollout_ref.rollout.calculate_log_probs)
        self.assertIsNone(c.algorithm.rollout_correction.rollout_is)
        self.assertFalse(c.algorithm.rollout_correction.bypass_mode)  # decoupled path computes the metrics

    def test_scorer_wired_and_loadable(self):
        """The hydra path must resolve to the TIERED scorer, and actually load."""
        c = self.cfg
        path = os.path.join(REPO_ROOT, c.custom_reward_function.path)
        self.assertTrue(os.path.exists(path), path)
        self.assertEqual(c.custom_reward_function.name, "compute_score")
        source = open(path).read()
        # tier markers: vendored ORZ equality + hard-deadline sympy tier
        self.assertIn("_is_equiv_orz", source)
        self.assertIn("ORZ_MATH_SYMPY_TIMEOUT", source)
        import importlib.util

        spec = importlib.util.spec_from_file_location("orz_scorer_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.compute_score(
            data_source="aime2024_orz",
            solution_str="<answer>\\boxed{42}</answer>",
            ground_truth="42",
        )
        self.assertEqual(result["acc"], True)

    def test_validation_on_orz_prompt_parquets(self):
        """aime-2024/2025 exactly as the _final branch's ORZ-72k arms (x32 ORZ-prompt parquets)."""
        vals = list(self.cfg.data.val_files)
        self.assertEqual(len(vals), 2)
        self.assertTrue(vals[0].endswith("aime-2024-orz.parquet"), vals)
        self.assertTrue(vals[1].endswith("aime-2025-orz.parquet"), vals)
        self.assertEqual(self.cfg.actor_rollout_ref.rollout.val_kwargs.n, 1)
        self.assertEqual(float(self.cfg.actor_rollout_ref.rollout.val_kwargs.temperature), 1.0)
        self.assertTrue(self.cfg.actor_rollout_ref.rollout.val_kwargs.do_sample)


if __name__ == "__main__":
    unittest.main()
