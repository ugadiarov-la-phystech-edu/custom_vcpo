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
"""The Open-Reasoner-Zero-7B arm's config, composed through hydra exactly as a launch would.

Two invariants are worth protecting here. The first is that the arm is its Qwen3-8B twin plus a
model, a scorer and ONE deliberate sampling divergence, and nothing else - every other schedule knob
that would make the two runs incomparable must be equal. The second is that the scorer is actually
wired: a
``custom_reward_function.path`` that hydra resolves but ``get_custom_reward_fn`` cannot load fails
only after the cluster is up, and a path that loads but is the WRONG scorer fails silently, with
every rollout scoring -1, a group-relative advantage of 0, and a run that simply does not learn.

Composing runs the real script with ``--cfg job --resolve``, which needs the repo's environment; the
tests skip if the composition cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_orz_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASELINE = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/baseline")

ORZ = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg_orz7b.sh"
QWEN_MEGATRON = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh"
SMOKE_2P3 = "smoke_test_orz7b_2+3.sh"

_COMPOSED = {}


def compose(script_name, **env_overrides):
    """Run the script with hydra's --cfg job --resolve and parse the config it would launch with.

    Data paths are stubbed so the composition does not need the parquets - but only where the test
    is not about them: the smoke wrapper picks its own validation file, so TEST_FILE must not be
    injected there or the test would assert on its own stub.
    """
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


class TestOrzArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(ORZ, TEST_FILE="/tmp/test.parquet")

    def test_points_at_the_orz_checkpoint(self):
        """The hub id works as-is: Qwen2ForCausalLM is in verl's mcore registry, so unlike
        openPangu this needs neither a local re-alias nor trust_remote_code."""
        self.assertEqual(self.cfg.actor_rollout_ref.model.path, "Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
        self.assertEqual(self.cfg.actor_rollout_ref.actor.strategy, "megatron")

    def test_keeps_the_arm_identical_to_the_qwen_megatron_twin(self):
        """Only the model, the name and the scorer may differ."""
        qwen = compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet")
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_epochs",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.calculate_entropy",
            "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.actor.optim.lr",
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
            "rollout.total_rollout_steps",
            "rollout.test_freq",
            "async_training.staleness_threshold",
            "async_training.require_batches",
            "async_training.use_rollout_log_probs",
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
        qwen = compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet")
        self.assertEqual(self.cfg.data.train_files, qwen.data.train_files)
        self.assertEqual(self.cfg.data.val_files, qwen.data.val_files)

    def test_experiment_name_identifies_the_model(self):
        name = self.cfg.trainer.experiment_name
        self.assertIn("ORZ-7B", name)
        self.assertNotIn("Qwen3-8B", name)

    def test_validation_samples_at_orz_native_settings(self):
        """1.0/1.0, not the twin's 0.8/0.7: it is what ORZ was trained and published at, and what
        the 30-problem AIME-2024 probe used, so the step-0 validation point should land near
        5/30 ~ 0.167. A near-zero first reading then means the scorer is not wired."""
        val = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertEqual(val.temperature, 1.0)
        self.assertEqual(val.top_p, 1.0)
        self.assertIs(val.do_sample, True)

    def test_the_validation_divergence_from_the_twin_is_deliberate(self):
        """Recorded as a decision, so it cannot drift back to the twin's values unnoticed - and so
        that a future change to the twin's val_kwargs surfaces here instead of silently realigning
        the two arms."""
        qwen_val = compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet").actor_rollout_ref.rollout.val_kwargs
        self.assertEqual((qwen_val.temperature, qwen_val.top_p), (0.8, 0.7))
        orz_val = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertNotEqual((orz_val.temperature, orz_val.top_p), (qwen_val.temperature, qwen_val.top_p))

    def test_training_rollouts_sample_identically_in_both_arms(self):
        """The divergence is validation-only: the off-policy correction must see the same sampling
        distribution in both arms, or rollout_corr/* stops being comparable."""
        qwen = compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet")
        for key in ("temperature", "top_p", "top_k"):
            with self.subTest(key=key):
                self.assertEqual(
                    self.cfg.actor_rollout_ref.rollout[key],
                    qwen.actor_rollout_ref.rollout[key],
                )

    def test_batch_shape_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over {cfg.trainer.n_gpus_per_node} GPUs")

    def test_checkpoints_are_hf_model_only_and_not_resumable(self):
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(self.cfg.trainer.resume_mode, "disable")


class TestOrzRewardIsWired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(ORZ, TEST_FILE="/tmp/test.parquet")

    def test_custom_reward_function_is_set_and_the_file_exists(self):
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
        from verl.utils.reward_score import math_dapo

        response = "<think>work</think> <answer> \\boxed{42} </answer>"
        self.assertLessEqual(math_dapo.compute_score(response, "42")["score"], 0)

        sys.modules.pop("custom_module", None)
        try:
            fn = get_custom_reward_fn_for(self.cfg)
            self.assertGreater(fn(data_source="math_dapo", solution_str=response, ground_truth="42")["score"], 0)
        finally:
            sys.modules.pop("custom_module", None)

    def test_the_qwen_twin_does_not_use_a_custom_scorer(self):
        """The change is scoped to this arm: the Qwen baseline keeps stock math_dapo."""
        self.assertIsNone(compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet").custom_reward_function.path)


class TestOrzSmoke2plus3(unittest.TestCase):
    """The 2+3 smoke test: two cheap steps, validated and checkpointed at every one.

    It runs the real arm with env overrides, so what matters is that the overrides survive
    composition - a typo in one of them turns a 20-minute check into a multi-hour run, or into one
    that never validates.
    """

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_2P3)

    def test_layout_is_two_plus_three(self):
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 2)
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)

    def test_it_runs_exactly_two_trainer_steps(self):
        cfg = self.cfg
        per_step = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.async_training.require_batches
        self.assertEqual(cfg.rollout.total_rollout_steps, 2 * per_step)

    def test_batch_divides_across_the_trainer_gpus(self):
        """tp=pp=1, so Megatron's DP is exactly the three trainer GPUs."""
        cfg = self.cfg
        self.assertEqual(cfg.actor_rollout_ref.actor.megatron.tensor_model_parallel_size, 1)
        self.assertEqual(cfg.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size, 1)
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over 3 GPUs")

    def test_rollouts_are_cheap_but_the_length_is_real(self):
        """n is the speed lever. The response length is deliberately the arm's own 8192 - ORZ's
        measured responses reach 6892 tokens, so a shortened cap would truncate the <answer> block
        and make the reward check meaningless - but it must never EXCEED the arm's."""
        arm = compose(ORZ, TEST_FILE="/tmp/test.parquet")
        self.assertLess(self.cfg.actor_rollout_ref.rollout.n, 16)
        self.assertEqual(self.cfg.data.max_response_length, arm.data.max_response_length)

    def test_validation_is_aime_2024_only_after_every_step_and_not_before_training(self):
        """Both validations are then of a trained model. The untrained reference point (which
        should land near the offline probe's 5/30 = 0.167) is available on demand with
        val_before_train=True, and the script's report annotates it when present."""
        cfg = self.cfg
        self.assertIs(cfg.trainer.val_before_train, False)
        self.assertEqual(cfg.rollout.test_freq, 1)
        val_files = cfg.data.val_files
        if isinstance(val_files, str):
            val_files = [val_files]
        # the deduplicated copy: same 30 AIME-2024 problems, without the 32x duplication that
        # made a validation sweep the dominant cost of this test
        self.assertEqual([os.path.basename(f) for f in val_files], ["aime-2024_smoke.parquet"])

    def test_checkpoint_after_every_step(self):
        """save_freq and test_freq are both in param-version units, and one param version ticks per
        trainer step here (trigger_parameter_sync_step=1), so 1 really is 'every step'."""
        self.assertEqual(self.cfg.async_training.trigger_parameter_sync_step, 1)
        self.assertEqual(self.cfg.trainer.save_freq, 1)
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])

    def test_gradient_cannot_be_identically_zero(self):
        """With 3 prompts x n=2 a group can easily tie, and a tied group has GRPO advantage 0 - the
        weights could not move and "the checkpoints differ" would be unverifiable."""
        self.assertGreater(self.cfg.actor_rollout_ref.actor.entropy_coeff, 0)
        self.assertGreater(self.cfg.actor_rollout_ref.actor.optim.lr, 1e-6)

    def test_it_still_uses_the_arms_model_and_scorer(self):
        """The whole point is to exercise the real ORZ path; a smoke test that silently fell back
        to stock math_dapo would pass while testing nothing this arm cares about."""
        self.assertEqual(self.cfg.actor_rollout_ref.model.path, "Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
        self.assertEqual(self.cfg.actor_rollout_ref.actor.strategy, "megatron")
        self.assertTrue(self.cfg.custom_reward_function.path.endswith("orz_tag_aware_math.py"))
        self.assertEqual(self.cfg.custom_reward_function.name, "compute_score")

    def test_validation_sampling_matches_the_probe_the_reference_came_from(self):
        """The 0.167 reference was measured at T=1.0/top_p=1.0; comparing a val point taken at
        different sampling to it would be meaningless."""
        val = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertEqual((val.temperature, val.top_p), (1.0, 1.0))
        self.assertEqual(val.n, 1)

    def test_rollout_dumps_land_where_the_reward_check_looks(self):
        """The script's reward check globs CKPTS_DIR/*.jsonl; if rollout_data_dir pointed anywhere
        else the check would report 'no rollout dumps' on a perfectly good run."""
        self.assertEqual(self.cfg.trainer.rollout_data_dir, self.cfg.trainer.default_local_dir)


def get_custom_reward_fn_for(cfg):
    from verl.trainer.ppo.reward import get_custom_reward_fn

    return get_custom_reward_fn(OmegaConf.create({"custom_reward_function": cfg.custom_reward_function}))


if __name__ == "__main__":
    unittest.main()
