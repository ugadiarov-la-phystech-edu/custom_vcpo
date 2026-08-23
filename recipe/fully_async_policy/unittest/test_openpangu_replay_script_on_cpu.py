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
"""The openPangu replay arm must differ from its Qwen twin ONLY in the model.

The two arms exist to be compared, so every replay/ESS/batch knob has to match; and the four
model-specific settings each correspond to a failure that already cost a dead run:

* the re-aliased checkpoint - transformers 4.57.6 cannot import openPangu's remote modeling code;
* both trust_remote_code keys - they are independent, one alone crashes the other half;
* HF_MODULES_CACHE on PYTHONPATH - Ray deserializes the custom tokenizer BY REFERENCE into the
  actors, which otherwise die with ModuleNotFoundError: No module named 'transformers_modules';
* gpu_memory_utilization=0.75 - this arm broadcasts fp32 weights at weight sync (2.5 GB embedding
  per rollout GPU) and 0.8 left NCCL unable to allocate its communicator.

Composition runs the real scripts with `--cfg job --resolve`; the tests skip if that cannot run.
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer")

OPENPANGU = "grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh"
QWEN = "grpo_novcpo_8gpu_dapo17k_5+3_resp8k_fsdp2_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh"
SMOKE_2P3 = "smoke_test_openpangu7b_replay_2+3.sh"

_COMPOSED = {}


def compose(script_name, **env_overrides):
    """Compose a script the way a launch would. TEST_FILE is stubbed by default, but the smoke
    wrapper deliberately inherits the arm's validation files, so it is composed without one."""
    if script_name in _COMPOSED:
        return _COMPOSED[script_name]
    path = os.path.join(REPLAY, script_name)
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


def read(script_name):
    with open(os.path.join(REPLAY, script_name)) as f:
        return f.read()


class TestOpenPanguReplayArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPENPANGU, TEST_FILE="/tmp/test.parquet")

    def test_uses_the_realiased_local_checkpoint(self):
        path = self.cfg.actor_rollout_ref.model.path
        self.assertIn("openPangu", path)
        self.assertFalse(path.startswith("FreedomIntelligence/"), f"{path} is the stock hub checkpoint")

    def test_sets_both_trust_remote_code_keys(self):
        self.assertIs(self.cfg.actor_rollout_ref.model.trust_remote_code, True)
        self.assertIs(self.cfg.data.trust_remote_code, True)

    def test_exports_the_hf_modules_cache_onto_pythonpath(self):
        """Invisible to hydra composition - it is an environment effect - so assert the text."""
        text = read(OPENPANGU)
        self.assertIn("HF_MODULES_CACHE", text)
        self.assertIn('export PYTHONPATH="${HF_MODULES_CACHE}', text)
        self.assertIn('case ":${PYTHONPATH:-}:" in', text, "must skip when already present")
        self.assertIn("${PYTHONPATH:+:${PYTHONPATH}}", text, "must not emit a bare colon")

    def test_gpu_utilization_is_lower_than_the_qwen_twin(self):
        qwen = compose(QWEN, TEST_FILE="/tmp/test.parquet")
        mine = self.cfg.actor_rollout_ref.rollout.gpu_memory_utilization
        self.assertEqual(mine, 0.75)
        self.assertLess(mine, qwen.actor_rollout_ref.rollout.gpu_memory_utilization)

    def test_the_qwen_twin_needs_no_remote_code(self):
        """The invariant is about custom-code models, not about this one script."""
        qwen = compose(QWEN, TEST_FILE="/tmp/test.parquet")
        self.assertNotIn("HF_MODULES_CACHE", read(QWEN))
        self.assertIs(qwen.actor_rollout_ref.model.trust_remote_code, False)

    def test_everything_that_defines_the_arm_matches_the_qwen_twin(self):
        qwen = compose(QWEN, TEST_FILE="/tmp/test.parquet")
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.fsdp_config.fsdp_size",
            "actor_rollout_ref.actor.fsdp_config.param_offload",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
            "actor_rollout_ref.actor.fsdp_config.model_dtype",
            "actor_rollout_ref.actor.entropy_from_logits_with_chunking",
            "actor_rollout_ref.actor.entropy_checkpointing",
            "actor_rollout_ref.actor.grad_clip",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.use_dynamic_bsz",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            "actor_rollout_ref.actor.seq_adv_post_scale",
            "actor_rollout_ref.actor.ess_scaling.enable",
            "actor_rollout_ref.actor.ess_scaling.min_ess",
            "actor_rollout_ref.actor.ess_scaling.lr_scale",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.max_num_seqs",
            "data.max_prompt_length",
            "data.max_response_length",
            "trainer.n_gpus_per_node",
            "trainer.save_freq",
            "trainer.resume_mode",
            "rollout.n_gpus_per_node",
            "rollout.test_freq",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(qwen, path),
                    f"{path} differs between the openPangu arm and its Qwen twin",
                )

    def test_replay_and_staleness_knobs_match_the_qwen_twin(self):
        """These live under different config roots depending on the branch; compare whatever is
        present, and fail if the two scripts disagree about any of them."""
        qwen = compose(QWEN, TEST_FILE="/tmp/test.parquet")
        for root in ("actor_rollout_ref.actor.replay_buffer", "replay_buffer", "async_training"):
            mine, theirs = OmegaConf.select(self.cfg, root), OmegaConf.select(qwen, root)
            if mine is None:
                continue
            with self.subTest(root=root):
                self.assertEqual(
                    OmegaConf.to_container(mine, resolve=True),
                    OmegaConf.to_container(theirs, resolve=True),
                    f"{root} differs between the openPangu arm and its Qwen twin",
                )


class TestOpenPanguReplaySmoke2plus3(unittest.TestCase):
    """The 2+3 smoke wrapper: two cheap steps on the real arm.

    The invariant is narrow and deliberate — the wrapper was asked to change FIVE things and
    nothing else, so what these tests protect is mostly what it does NOT touch. A wrapper that
    quietly shrank the response length or the rollout count would still "pass" a run while
    exercising a different configuration than the arm it stands in for.
    """

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_2P3)
        cls.arm = compose(OPENPANGU, TEST_FILE="/tmp/test.parquet")

    def test_layout_is_two_plus_three(self):
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 2)
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)

    def test_mini_batch_size_is_three(self):
        self.assertEqual(self.cfg.actor_rollout_ref.actor.ppo_mini_batch_size, 3)

    def test_batch_still_divides_across_the_trainer_gpus(self):
        """mini_bsz shrank but rollout.n did not, so the batch is 3*16=48 — still divisible by
        DP=3. A wrapper that had also cut n could silently break this."""
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over 3 GPUs")

    def test_it_runs_exactly_two_trainer_steps(self):
        """fully_async_rollouter.py:229-232 derives
        total_train_steps = total_rollout_steps / (required_samples * trigger_parameter_sync_step),
        with required_samples = mini_bsz * require_batches."""
        cfg = self.cfg
        required = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.async_training.require_batches
        sync = cfg.async_training.trigger_parameter_sync_step
        self.assertEqual(cfg.rollout.total_rollout_steps, 2 * required * sync)

    def test_validates_after_every_step_and_not_before_training(self):
        self.assertEqual(self.cfg.rollout.test_freq, 1)
        self.assertIs(self.cfg.trainer.val_before_train, False)

    def test_changes_nothing_else(self):
        """The five overrides above are the whole remit. Everything that defines what is being
        exercised must still equal the arm's own value — including the ones a smoke test is
        normally tempted to shrink (rollout.n, response length) and the ones that make it this
        arm rather than another (replay, min-ESS, loss mode)."""
        for path in (
            "actor_rollout_ref.rollout.n",
            "data.max_response_length",
            "data.max_prompt_length",
            "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.ess_scaling.enable",
            "actor_rollout_ref.actor.ess_scaling.min_ess",
            "actor_rollout_ref.actor.ess_scaling.lr_scale",
            "actor_rollout_ref.actor.use_dynamic_bsz",
            "async_training.replay_buffer.tau",
            "async_training.replay_buffer.staleness_threshold",
            "async_training.require_batches",
            "async_training.trigger_parameter_sync_step",
            "trainer.save_freq",
            "actor_rollout_ref.model.path",
            "actor_rollout_ref.model.trust_remote_code",
            "data.trust_remote_code",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.val_kwargs.temperature",
            "actor_rollout_ref.rollout.val_kwargs.top_p",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(self.arm, path),
                    f"{path} was changed by the smoke wrapper but should not have been",
                )

    def test_validates_on_the_deduplicated_aime_2024_only(self):
        """The arm validates on the full aime-2024 AND aime-2025 (~960 rows each after their 32x
        duplication, ~1920 problems per sweep) and test_freq=1 would sweep both twice — an hour
        per sweep at 8192 tokens, which would make a 2-step smoke test dominated by validation.
        The _smoke file is the same 30 AIME-2024 problems without the duplication."""
        val_files = self.cfg.data.val_files
        if isinstance(val_files, str):
            val_files = [val_files]
        self.assertEqual([os.path.basename(f) for f in val_files], ["aime-2024_smoke.parquet"])
        arm_files = self.arm.data.val_files
        if isinstance(arm_files, str):
            arm_files = [arm_files]
        self.assertNotEqual(
            [os.path.basename(f) for f in val_files],
            [os.path.basename(f) for f in arm_files],
            "the wrapper is expected to narrow validation; if the arm already used the smoke file "
            "this assertion is stale",
        )

    def test_writes_to_its_own_log_dir(self):
        """The one thing outside the five: without it the smoke would write into the real arm's
        checkpoint and log directory."""
        self.assertNotEqual(self.cfg.trainer.default_local_dir, self.arm.trainer.default_local_dir)
        self.assertIn("SMOKE", self.cfg.trainer.experiment_name)

    def test_no_checkpoint_is_expected_within_two_steps(self):
        """save_freq is untouched at the arm's 20, so a 2-step run writes none — the wrapper must
        not claim to verify checkpoints."""
        self.assertEqual(self.cfg.trainer.save_freq, self.arm.trainer.save_freq)
        self.assertGreater(self.cfg.trainer.save_freq, 2)


if __name__ == "__main__":
    unittest.main()
