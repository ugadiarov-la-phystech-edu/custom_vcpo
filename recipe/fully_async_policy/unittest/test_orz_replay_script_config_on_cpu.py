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

Three invariants are worth protecting. The first is that the arm is its Qwen3-8B twin plus a model,
the ORZ-72k datasets, a scorer and one deliberate sampling divergence, and nothing else — every
replay, min-ESS and schedule knob that would make the two runs incomparable must be equal. The
second is that the scorer is actually wired: a ``custom_reward_function.path`` that hydra resolves
but ``get_custom_reward_fn`` cannot load fails only after the cluster is up, and a path that loads
but is the WRONG scorer fails silently, with every ORZ rollout scoring -1, a group-relative
advantage of 0, and a run that simply does not learn. The third is that the wired scorer carries
the LaTeX equality tiers of the baselines_main-ppo sync ORZ-72k arm: ~24% of orz-math-72k ground
truths are LaTeX expressions, and the pre-tier scorer scores a correct ``280/83`` against
``\\frac{280}{83}`` as wrong.

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

STEM = "grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5"
ORZ = (
    STEM.replace("dapo17k_5+3", "orz72k_3+5")
    .replace("tau=16_k=64", "tau=8_k=32")
    .replace("min-ess=1.1", "min-ess=1.07")
    + "_orz7b.sh"
)
QWEN = f"{STEM}.sh"
SMOKE_3P3 = "smoke_test_orz7b_replay_3+3.sh"

_COMPOSED = {}


def compose(script_name, stub_test_file=True):
    """Run the script with hydra's --cfg job --resolve and parse the config it would launch with.

    TRAIN_FILE is always stubbed so composition needs no parquet. TEST_FILE is stubbed too, except
    where the test is about the validation file the script itself picks (the smoke wrapper).
    """
    if script_name in _COMPOSED:
        return _COMPOSED[script_name]
    path = os.path.join(REPLAY, script_name)
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{script_name} not found")
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet")
    if stub_test_file:
        env["TEST_FILE"] = "/tmp/test.parquet"
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
        """Only the model, the name, the scorer, the datasets, val sampling, the GPU layout (with
        the mini-batch it forces), the replay depth (tau/k, with the generation quota that follows
        k) and the min-ESS floor may differ. The remaining replay and min-ESS keys are the point of
        this arm; if any of them drifted the two runs would not be comparable."""
        qwen = compose(QWEN)
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.use_dynamic_bsz",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.ess_scaling.enable",
            "actor_rollout_ref.actor.ess_scaling.lr_scale",
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
            "rollout.test_freq",
            "rollout.total_rollout_steps",
            "async_training.replay_buffer.enable",
            "async_training.replay_buffer.requires_mini_batches",
            "async_training.require_batches",
            "async_training.trigger_parameter_sync_step",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(qwen, path),
                    f"{path} differs between the ORZ arm and its Qwen twin",
                )

    def test_trains_on_orz72k_and_validates_on_orz_prompt_aime(self):
        """The datasets are the baselines_main-ppo sync ORZ-72k arm's: ORZ's own RL set, whose
        prompt column carries ORZ's <answer>-tag instruction, and the AIME sets rewritten with the
        same instruction. The Qwen twin keeps the DAPO files, so the divergence is deliberate."""
        env = dict(os.environ)
        env.pop("TRAIN_FILE", None)
        env.pop("TEST_FILE", None)
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
            proc = subprocess.run(
                ["bash", os.path.join(REPLAY, ORZ), "--cfg", "job", "--resolve"],
                cwd=REPO_ROOT,
                env=env,
                stdout=out,
                stderr=subprocess.PIPE,
                timeout=900,
            )
            if proc.returncode != 0:
                raise unittest.SkipTest(f"could not compose {ORZ}: {proc.stderr.decode()[-300:]}")
            out.flush()
            out.seek(0)
            cfg = OmegaConf.load(out.name)
        self.assertEqual(os.path.basename(cfg.data.train_files), "orz-math-72k.parquet")
        self.assertEqual(
            [os.path.basename(f) for f in cfg.data.val_files],
            ["aime-2024-orz.parquet", "aime-2025-orz.parquet"],
        )
        self.assertIn("/orz/", cfg.data.train_files)
        qwen = compose(QWEN)
        self.assertEqual(os.path.basename(qwen.data.train_files), "train.parquet")  # the stub: twin untouched

    def test_prompt_handling_knobs_match_the_qwen_twin(self):
        """ "Same prompt processing" is a dataset swap on this recipe: the ORZ instruction lives in
        the parquet's prompt column, and every data.* knob that shapes the prompt is the twin's."""
        qwen = compose(QWEN)
        for path in (
            "data.prompt_key",
            "data.truncation",
            "data.max_prompt_length",
            "data.max_response_length",
            "data.filter_overlong_prompts",
            "data.return_raw_chat",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(qwen, path))

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

    def test_experiment_name_identifies_the_model_and_the_dataset(self):
        name = self.cfg.trainer.experiment_name
        self.assertIn("ORZ-7B", name)
        self.assertIn("ORZ72K", name)
        self.assertNotIn("Qwen3-8B", name)
        self.assertNotIn("DAPO17K", name)

    def test_min_ess_floor_is_tighter_than_the_twins(self):
        """min_ess=1.07 against the twin's 1.1: the brake fires only within 7% of the structural
        ESS=1 floor, so fewer marginal mini-batches are braked. lr_scale stays the twin's."""
        ess = self.cfg.actor_rollout_ref.actor.ess_scaling
        self.assertIs(ess.enable, True)
        self.assertAlmostEqual(ess.min_ess, 1.07)
        self.assertAlmostEqual(compose(QWEN).actor_rollout_ref.actor.ess_scaling.min_ess, 1.1)
        self.assertIn("min-ess-1.07", self.cfg.trainer.experiment_name)

    def test_reuse_halflife_is_one_and_the_twin_has_none(self):
        """The reuse decay (REPLAY_REUSE_PENALTY_DISCUSSION.md) is on for this arm at nu=1 and off
        (null) for the Qwen twin, whose draws must stay bit-for-bit what they were."""
        self.assertEqual(self.cfg.async_training.replay_buffer.reuse_halflife, 1)
        self.assertIn(" nu-1 ", self.cfg.trainer.experiment_name)
        qwen = compose(QWEN)
        self.assertIsNone(qwen.async_training.replay_buffer.reuse_halflife)
        self.assertNotIn("nu-", qwen.trainer.experiment_name)

    def test_replay_depth_is_half_the_twins(self):
        """tau=8 / k=32 against the twin's 16 / 64: the ORZ-7B post-mortems tie its divergences to
        deep staleness, so this arm halves the reuse depth while keeping the terminal sampling
        weight (2^-4 at the eviction horizon). The rollouter's generation quota follows k so no
        group is generated only to be evicted unseen."""
        rb = self.cfg.async_training.replay_buffer
        self.assertEqual((rb.tau, rb.staleness_threshold), (8, 32))
        self.assertEqual(self.cfg.async_training.staleness_threshold, 32)
        qwen = compose(QWEN)
        self.assertEqual(
            (qwen.async_training.replay_buffer.tau, qwen.async_training.replay_buffer.staleness_threshold),
            (16, 64),
        )
        self.assertEqual(qwen.async_training.staleness_threshold, qwen.async_training.replay_buffer.staleness_threshold)

    def test_layout_is_three_rollout_plus_five_trainer_gpus(self):
        """ORZ-7B's short responses make the per-traj update the bottleneck, so this arm hands the
        trainer 5 GPUs (the Qwen twin runs 5+3). tp=pp=1, so DP is exactly the trainer GPU count."""
        cfg = self.cfg
        self.assertEqual(cfg.rollout.n_gpus_per_node, 3)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 5)
        self.assertEqual(cfg.actor_rollout_ref.actor.megatron.tensor_model_parallel_size, 1)
        self.assertEqual(cfg.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size, 1)
        qwen = compose(QWEN)
        self.assertEqual((qwen.rollout.n_gpus_per_node, qwen.trainer.n_gpus_per_node), (5, 3))

    def test_batch_shape_divides_across_the_trainer_gpus(self):
        """35 groups x 16 = 560 sequences over DP=5; the twin's 33 x 16 = 528 does not divide by 5,
        which is the only reason the mini-batch differs from the twin's."""
        cfg = self.cfg
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_mini_batch_size, 35)
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over {cfg.trainer.n_gpus_per_node} GPUs")
        qwen = compose(QWEN)
        self.assertNotEqual(qwen.actor_rollout_ref.actor.ppo_mini_batch_size * qwen.actor_rollout_ref.rollout.n % 5, 0)

    def test_rollout_concurrency_follows_the_mini_batch(self):
        """bsz_per_dp_rank defaults to the mini-batch, so the 3 engines hold 3 x 35 in flight."""
        self.assertEqual(self.cfg.async_training.bsz_per_dp_rank, self.cfg.actor_rollout_ref.actor.ppo_mini_batch_size)


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

    def test_the_wired_scorer_has_the_latex_tiers(self):
        """The scorer the cluster loads must be the tiered one from baselines_main-ppo: a correct
        slash fraction against a LaTeX ground truth (ORZ's is_equiv tier) and an unreduced
        fraction (the sympy tier, when a parser backend is available) both score correct. The
        pre-tier file fails the first assertion."""
        from verl.trainer.ppo.reward import get_custom_reward_fn

        sys.modules.pop("custom_module", None)
        try:
            fn = get_custom_reward_fn(OmegaConf.create({"custom_reward_function": self.cfg.custom_reward_function}))
            self.assertTrue(
                fn(
                    data_source="math_dapo",
                    solution_str="<think>work</think> <answer> \\boxed{280/83} </answer>",
                    ground_truth="\\frac{280}{83}",
                )["acc"]
            )
            self.assertFalse(
                fn(
                    data_source="math_dapo",
                    solution_str="<think>work</think> <answer> \\boxed{281/83} </answer>",
                    ground_truth="\\frac{280}{83}",
                )["acc"]
            )
            from recipe.fully_async_policy.reward import orz_tag_aware_math as orz

            if orz._sympy_tier_enabled():
                self.assertTrue(
                    fn(
                        data_source="math_dapo",
                        solution_str="<answer> \\boxed{\\frac{2}{4}} </answer>",
                        ground_truth="\\frac{1}{2}",
                    )["acc"]
                )
        finally:
            sys.modules.pop("custom_module", None)

    def test_the_qwen_twin_does_not_use_a_custom_scorer(self):
        """The change is scoped to this arm: the Qwen replay arm keeps stock math_dapo."""
        self.assertIsNone(compose(QWEN).custom_reward_function.path)


class TestOrzReplaySmoke3plus3(unittest.TestCase):
    """The 3+3 smoke test: two cheap updates (one fresh, one pure replay), validated and
    checkpointed at every one.

    It runs the real arm with env overrides, so what matters is that the overrides survive
    composition - a typo in one of them turns a 20-minute check into a multi-hour run (replay mode
    keeps composing pure-replay mini-batches until eviction drains the buffer), or into one that
    never validates.
    """

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_3P3, stub_test_file=False)

    def test_layout_is_three_plus_three(self):
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 3)
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)

    def test_generation_budget_is_exactly_one_fresh_mini_batch(self):
        cfg = self.cfg
        per_update = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.async_training.require_batches
        self.assertEqual(cfg.rollout.total_rollout_steps, per_update)
        self.assertEqual(cfg.async_training.replay_buffer.requires_mini_batches, 1)

    def test_replay_eviction_ends_the_run_after_the_second_update(self):
        """staleness_threshold=1: the fresh groups survive exactly one more (pure-replay) update
        and are then evicted, so the fit loop exits deterministically after 2 updates. tau is the
        arm's own, so the replay weighting is unchanged."""
        rb = self.cfg.async_training.replay_buffer
        self.assertIs(rb.enable, True)
        self.assertEqual(rb.staleness_threshold, 1)
        self.assertEqual(rb.tau, compose(ORZ).async_training.replay_buffer.tau)

    def test_inherits_the_arms_reuse_decay(self):
        """nu=1 is inert in a 2-update smoke (no group is drawn with times_trained > 1) but the
        wrapper must not silently override it either."""
        self.assertEqual(self.cfg.async_training.replay_buffer.reuse_halflife, 1)

    def test_batch_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        self.assertEqual(cfg.actor_rollout_ref.actor.megatron.tensor_model_parallel_size, 1)
        self.assertEqual(cfg.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size, 1)
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over 3 GPUs")

    def test_rollouts_are_cheap_but_the_length_is_real(self):
        arm = compose(ORZ)
        self.assertLess(self.cfg.actor_rollout_ref.rollout.n, 16)
        self.assertEqual(self.cfg.data.max_response_length, arm.data.max_response_length)

    def test_validation_is_the_30_row_orz_aime_after_every_update_and_not_before_training(self):
        cfg = self.cfg
        self.assertIs(cfg.trainer.val_before_train, False)
        self.assertEqual(cfg.rollout.test_freq, 1)
        val_files = cfg.data.val_files
        if isinstance(val_files, str):
            val_files = [val_files]
        self.assertEqual([os.path.basename(f) for f in val_files], ["aime-2024-orz_smoke.parquet"])
        self.assertTrue(os.path.isabs(val_files[0]))
        self.assertEqual(os.path.dirname(val_files[0]), os.path.join(REPO_ROOT, cfg.trainer.default_local_dir))

    def test_checkpoint_after_every_update(self):
        self.assertEqual(self.cfg.async_training.trigger_parameter_sync_step, 1)
        self.assertEqual(self.cfg.trainer.save_freq, 1)
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])

    def test_gradient_cannot_be_identically_zero(self):
        self.assertGreater(self.cfg.actor_rollout_ref.actor.entropy_coeff, 0)
        self.assertGreater(self.cfg.actor_rollout_ref.actor.optim.lr, 1e-6)

    def test_it_still_uses_the_arms_model_scorer_and_training_set(self):
        cfg = self.cfg
        self.assertEqual(cfg.actor_rollout_ref.model.path, "Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
        self.assertEqual(cfg.actor_rollout_ref.actor.strategy, "megatron")
        self.assertTrue(cfg.custom_reward_function.path.endswith("orz_tag_aware_math.py"))
        self.assertEqual(cfg.custom_reward_function.name, "compute_score")
        self.assertEqual(os.path.basename(cfg.data.train_files), "train.parquet")  # stubbed by compose()
        self.assertIs(cfg.actor_rollout_ref.actor.ess_scaling.enable, True)

    def test_validation_sampling_matches_the_probe_the_reference_came_from(self):
        val = self.cfg.actor_rollout_ref.rollout.val_kwargs
        self.assertEqual((val.temperature, val.top_p), (1.0, 1.0))
        self.assertEqual(val.n, 1)

    def test_rollout_dumps_land_where_the_reward_check_looks(self):
        self.assertEqual(self.cfg.trainer.rollout_data_dir, self.cfg.trainer.default_local_dir)


if __name__ == "__main__":
    unittest.main()
