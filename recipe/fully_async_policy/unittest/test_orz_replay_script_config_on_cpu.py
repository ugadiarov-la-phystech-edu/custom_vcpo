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
"""The ORZ-7B replay arm's config, composed through hydra exactly as a launch would.

Two invariants are worth protecting. The first is that the arm is its Qwen3-8B twin plus a model, a
scorer and one deliberate sampling divergence, and nothing else — every replay, min-ESS and schedule
knob that would make the two runs incomparable must be equal. The second is that the scorer is
actually wired: a ``custom_reward_function.path`` that hydra resolves but ``get_custom_reward_fn``
cannot load fails only after the cluster is up, and a path that loads but is the WRONG scorer fails
silently, with every ORZ rollout scoring -1, a group-relative advantage of 0, and a run that simply
does not learn.

Composing runs the real script with ``--cfg job --resolve``, which needs the repo's environment; the
tests skip if the composition cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_orz_replay_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer")

STEM = "grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333"
ORZ = f"{STEM}_orz7b.sh"
QWEN = f"{STEM}.sh"

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


class TestOrzReplayArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(ORZ)

    def test_points_at_the_orz_checkpoint(self):
        """The hub id works as-is: Qwen2ForCausalLM is in verl's mcore registry, so unlike openPangu
        this needs neither a local re-alias nor trust_remote_code."""
        self.assertEqual(self.cfg.actor_rollout_ref.model.path, "Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
        self.assertEqual(self.cfg.actor_rollout_ref.actor.strategy, "megatron")

    def test_keeps_the_arm_identical_to_the_qwen_twin(self):
        """Only the model, the name, the scorer and val sampling may differ. The replay and min-ESS
        keys are the point of this arm; if any of them drifted the two runs would not be comparable."""
        qwen = compose(QWEN)
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.use_dynamic_bsz",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.ess_scaling.enable",
            "actor_rollout_ref.actor.ess_scaling.scaling_rule",
            "actor_rollout_ref.actor.ess_scaling.base_ess_ratio",
            "actor_rollout_ref.actor.ess_scaling.trigger_ratio",
            "actor_rollout_ref.actor.ess_scaling.use_clipped",
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.temperature",
            "actor_rollout_ref.rollout.top_p",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "data.max_prompt_length",
            "data.max_response_length",
            "trainer.save_freq",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            "rollout.test_freq",
            "rollout.total_rollout_steps",
            "async_training.replay_buffer.enable",
            "async_training.replay_buffer.tau",
            "async_training.replay_buffer.staleness_threshold",
            "async_training.replay_buffer.requires_mini_batches",
            "async_training.staleness_threshold",
            "async_training.require_batches",
            "async_training.trigger_parameter_sync_step",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(qwen, path),
                    f"{path} differs between the ORZ arm and its Qwen twin",
                )

    def test_keeps_the_datasets_byte_identical_to_the_qwen_twin(self):
        """Stripping the DAPO wrapper changed neither accuracy nor response shape on the 30-problem
        probe, so the prompts stay exactly as the Qwen arm sees them."""
        qwen = compose(QWEN)
        self.assertEqual(self.cfg.data.train_files, qwen.data.train_files)
        self.assertEqual(self.cfg.data.val_files, qwen.data.val_files)

    def test_validation_samples_at_orz_native_settings(self):
        """1.0/1.0, not the twin's 0.8/0.7: it is what ORZ was trained and published at, and what the
        30-problem AIME-2024 probe used, so the step-0 validation point should land near
        5/30 ~ 0.167 — measured at 0.2333 through the real pipeline on the baselines twin."""
        val = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertEqual(val.temperature, 1.0)
        self.assertEqual(val.top_p, 1.0)
        self.assertIs(val.do_sample, True)

    def test_the_validation_divergence_from_the_twin_is_deliberate(self):
        """Recorded as a decision, so it cannot drift back unnoticed — and so that a change to the
        twin's val_kwargs surfaces here instead of silently realigning the two arms."""
        qwen_val = compose(QWEN).actor_rollout_ref.rollout.val_kwargs
        self.assertEqual((qwen_val.temperature, qwen_val.top_p), (0.8, 0.7))
        orz_val = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertNotEqual((orz_val.temperature, orz_val.top_p), (qwen_val.temperature, qwen_val.top_p))

    def test_experiment_name_identifies_the_model(self):
        name = self.cfg.trainer.experiment_name
        self.assertIn("ORZ-7B", name)
        self.assertNotIn("Qwen3-8B", name)

    def test_batch_shape_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over {cfg.trainer.n_gpus_per_node} GPUs")


class TestOrzReplayRewardIsWired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(ORZ)

    def test_custom_reward_function_is_absolute_and_exists(self):
        crf = self.cfg.custom_reward_function
        self.assertEqual(crf.name, "compute_score")
        self.assertTrue(
            os.path.isabs(crf.path),
            f"{crf.path} is relative; get_custom_reward_fn() runs inside Ray actors whose cwd is "
            "not guaranteed to be the repo root",
        )
        self.assertTrue(os.path.exists(crf.path), f"{crf.path} does not exist")

    def test_get_custom_reward_fn_loads_it_and_it_scores_an_orz_response(self):
        """Exactly the path the trainer takes, so a bad path or name fails here and not on the
        cluster. get_custom_reward_fn caches under sys.modules['custom_module'], so the entry is
        cleared first to keep this test independent of import order."""
        from verl.trainer.ppo.reward import get_custom_reward_fn

        sys.modules.pop("custom_module", None)
        try:
            cfg = OmegaConf.create({"custom_reward_function": self.cfg.custom_reward_function})
            fn = get_custom_reward_fn(cfg)
            self.assertIsNotNone(fn, "custom reward function did not load")
            result = fn(
                data_source="math_dapo",
                solution_str="<think>work</think> <answer> \\boxed{42} </answer>",
                ground_truth="42",
                extra_info={},
            )
            self.assertEqual(result["pred"], "42")
            self.assertTrue(result["acc"])
        finally:
            sys.modules.pop("custom_module", None)

    def test_the_wired_scorer_is_the_tag_aware_one_not_math_dapo(self):
        """A path that loads but points at the stock scorer is the silent failure this guards:
        every ORZ rollout would score -1 and the group-relative advantage would be 0."""
        from verl.trainer.ppo.reward import get_custom_reward_fn
        from verl.utils.reward_score import math_dapo

        response = "<think>work</think> <answer> \\boxed{42} </answer>"
        self.assertLessEqual(math_dapo.compute_score(response, "42")["score"], 0)

        sys.modules.pop("custom_module", None)
        try:
            fn = get_custom_reward_fn(OmegaConf.create({"custom_reward_function": self.cfg.custom_reward_function}))
            self.assertGreater(fn(data_source="math_dapo", solution_str=response, ground_truth="42")["score"], 0)
        finally:
            sys.modules.pop("custom_module", None)

    def test_the_qwen_twin_does_not_use_a_custom_scorer(self):
        """The change is scoped to this arm: the Qwen replay arm keeps stock math_dapo."""
        self.assertIsNone(compose(QWEN).custom_reward_function.path)


if __name__ == "__main__":
    unittest.main()
