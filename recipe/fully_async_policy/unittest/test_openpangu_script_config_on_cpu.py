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
"""The openPangu arm's config, composed through hydra exactly as a launch would.

The invariant worth protecting is not "this script sets a flag" but "a custom-code model sets BOTH
trust_remote_code keys". They are independent - data.trust_remote_code feeds the dataset-side
tokenizer, actor_rollout_ref.model.trust_remote_code feeds HFModelConfig (agent-loop tokenizer,
weight load, vLLM engine) - and nothing links them, so setting one and not the other builds half a
run and then dies in the other half.

Composing runs the real script with `--cfg job --resolve`, which needs the repo's environment; the
tests skip if the composition cannot run (e.g. no hydra/verl importable).

Run: pytest recipe/fully_async_policy/unittest/test_openpangu_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")

OPENPANGU = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh"
QWEN_FSDP2 = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_fsdp2_ppo-epochs=2_B33x1_is-pg.sh"

# Architectures shipped inside transformers itself; anything else may carry custom code. Local
# directories count as custom on purpose: the openPangu checkpoint is re-aliased to Llama but its
# TOKENIZER still resolves through tokenization_openpangu.py.
NATIVE_MODEL_PREFIXES = ("Qwen/", "meta-llama/", "mistralai/", "deepseek-ai/")

_COMPOSED = {}


def compose(script_name):
    """Run the script with hydra's --cfg job --resolve and parse the config it would launch with."""
    if script_name in _COMPOSED:
        return _COMPOSED[script_name]
    path = os.path.join(BASELINE, script_name)
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{script_name} not found")
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
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


def needs_remote_code(model_path):
    return not model_path.startswith(NATIVE_MODEL_PREFIXES)


class TestOpenPanguArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPENPANGU)

    def test_points_at_a_realiased_local_checkpoint(self):
        """Not the hub id: transformers 4.57.6 cannot import openPangu's remote modeling code."""
        path = self.cfg.actor_rollout_ref.model.path
        self.assertFalse(path.startswith("FreedomIntelligence/"), f"{path} is the stock hub checkpoint")
        self.assertIn("openPangu", path)

    def test_sets_both_trust_remote_code_keys(self):
        self.assertIs(self.cfg.actor_rollout_ref.model.trust_remote_code, True)
        self.assertIs(self.cfg.data.trust_remote_code, True)

    def test_keeps_the_arm_identical_to_the_qwen_fsdp2_twin(self):
        """Only the model may differ; everything defining the experiment must match."""
        qwen = compose(QWEN_FSDP2)
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_epochs",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.calculate_entropy",
            "actor_rollout_ref.rollout.n",
            "data.max_prompt_length",
            "data.max_response_length",
            "trainer.save_freq",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            "async_training.staleness_threshold",
            "async_training.use_rollout_log_probs",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(qwen, path),
                    f"{path} differs between the openPangu arm and its Qwen twin",
                )

    def test_batch_shape_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over {cfg.trainer.n_gpus_per_node} GPUs")

    def test_checkpoints_are_hf_model_only_and_not_resumable(self):
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(self.cfg.trainer.resume_mode, "disable")

    def test_prompt_budget_matches_what_was_measured(self):
        """The measured maximum over all three parquets under the Pangu tokenizer is 795 tokens."""
        self.assertGreaterEqual(self.cfg.data.max_prompt_length, 1024)


class TestQwenArmNeedsNoRemoteCode(unittest.TestCase):
    """The converse, so the invariant is about custom-code models rather than about one script."""

    def test_native_model_does_not_enable_remote_code(self):
        cfg = compose(QWEN_FSDP2)
        path = cfg.actor_rollout_ref.model.path
        if needs_remote_code(path):
            self.skipTest(f"{path} is not a natively-supported architecture")
        self.assertIs(cfg.actor_rollout_ref.model.trust_remote_code, False)
        self.assertIs(cfg.data.trust_remote_code, False)


if __name__ == "__main__":
    unittest.main()
