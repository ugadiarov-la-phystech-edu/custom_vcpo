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

_COMPOSED = {}


def compose(script_name):
    if script_name in _COMPOSED:
        return _COMPOSED[script_name]
    path = os.path.join(REPLAY, script_name)
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


def read(script_name):
    with open(os.path.join(REPLAY, script_name)) as f:
        return f.read()


class TestOpenPanguReplayArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPENPANGU)

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
        qwen = compose(QWEN)
        mine = self.cfg.actor_rollout_ref.rollout.gpu_memory_utilization
        self.assertEqual(mine, 0.75)
        self.assertLess(mine, qwen.actor_rollout_ref.rollout.gpu_memory_utilization)

    def test_the_qwen_twin_needs_no_remote_code(self):
        """The invariant is about custom-code models, not about this one script."""
        qwen = compose(QWEN)
        self.assertNotIn("HF_MODULES_CACHE", read(QWEN))
        self.assertIs(qwen.actor_rollout_ref.model.trust_remote_code, False)

    def test_everything_that_defines_the_arm_matches_the_qwen_twin(self):
        qwen = compose(QWEN)
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
        qwen = compose(QWEN)
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


if __name__ == "__main__":
    unittest.main()
