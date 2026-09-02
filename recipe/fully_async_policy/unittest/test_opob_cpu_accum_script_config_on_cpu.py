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
"""The 5+3 / fp32-master / host-accumulator OPOB arm's config, composed through hydra.

The arm is the 5+3 sqrt-brake replay twin plus OPOB, with OPOB's accumulators moved to
pinned host memory and the optimizer's bf16-master deviation removed. Pinned here: the
layout and its divisibility (33 prompts over DP=3, whole groups per rank), the OPOB knobs
(paper defaults + norm_by_std, accum_device=cpu, accum_dtype=float32), the optimizer block
(HDO on, NO precision-aware / main_params_dtype overrides), and twin-equality with the 5+3
twin for everything else.

Composing runs the real scripts with ``--cfg job --resolve``; the tests skip if it cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_opob_cpu_accum_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer")

TAIL = "resp8k_megatron_offload_replay_tau=16_k=64_ess-sqrt_base=0.006113_trig=0.33333"
ARM = f"grpo_novcpo_8gpu_dapo17k_5+3_{TAIL}_opob_fp32-masters_cpu-accum.sh"
TWIN = f"grpo_novcpo_8gpu_dapo17k_5+3_{TAIL}.sh"
OPOB44 = f"grpo_novcpo_8gpu_dapo17k_4+4_{TAIL}_opob.sh"

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


class TestCpuAccumArmScript(unittest.TestCase):
    """Static checks that need no composition."""

    def setUp(self):
        self.path = os.path.join(REPLAY, ARM)
        if not os.path.exists(self.path):
            raise unittest.SkipTest(f"{ARM} not found")
        with open(self.path) as f:
            self.text = f.read()

    def test_parses(self):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        proc = subprocess.run(["bash", "-n", self.path], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())

    def test_passes_the_accumulator_knobs_and_drops_the_bf16_master_overrides(self):
        for knob in ("accum_device", "accum_dtype", "accum_cpu_threads"):
            self.assertIn(f"actor_rollout_ref.actor.grad_baselining.{knob}=", self.text, knob)
        self.assertIn("grad_baselining_accum_device=${grad_baselining_accum_device:-cpu}", self.text)
        self.assertIn("grad_baselining_accum_dtype=${grad_baselining_accum_dtype:-float32}", self.text)
        self.assertNotIn("use_precision_aware_optimizer", self.text.split("python -m")[1])
        self.assertNotIn("main_params_dtype", self.text.split("python -m")[1])
        self.assertIn("override_optimizer_config.optimizer_cpu_offload=True", self.text)
        self.assertIn("n_gpus_rollout=${n_gpus_rollout:-5}", self.text)
        self.assertIn("train_tp=1", self.text)
        self.assertIn("train_prompt_mini_bsz=${train_prompt_mini_bsz:-33}", self.text)

    def test_smoke_wrapper_targets_this_script(self):
        smoke = os.path.join(REPLAY, ARM.replace(".sh", "_smoke_test_memory.sh"))
        self.assertTrue(os.path.exists(smoke), smoke)
        with open(smoke) as f:
            text = f.read()
        self.assertIn(f'SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/{ARM}"', text)
        self.assertIn("train_prompt_mini_bsz % 3", text)
        self.assertIn("devices={'cpu'}", text)
        proc = subprocess.run(["bash", "-n", smoke], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())


class TestCpuAccumArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(ARM)

    # ---------------------------------------------------------------- OPOB

    def test_opob_knobs(self):
        gb = self.cfg.actor_rollout_ref.actor.grad_baselining
        self.assertTrue(gb.enable)
        self.assertEqual(gb.scope, "group")
        self.assertEqual(gb.agg_mode, "mean")
        self.assertTrue(gb.use_is_weights)
        self.assertFalse(gb.use_clipped_is_ratios)
        self.assertFalse(gb.normalize_by_length)
        self.assertTrue(gb.norm_by_std)
        self.assertEqual(gb.accum_device, "cpu")
        self.assertEqual(str(gb.accum_dtype), "float32")
        self.assertEqual(gb.accum_cpu_threads, 16)

    def test_opob_knobs_match_the_4plus4_arm_except_accumulator_placement(self):
        opob44 = compose(OPOB44)
        for key in (
            "enable",
            "scope",
            "agg_mode",
            "use_is_weights",
            "use_clipped_is_ratios",
            "normalize_by_length",
            "norm_by_std",
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    OmegaConf.select(self.cfg, f"actor_rollout_ref.actor.grad_baselining.{key}"),
                    OmegaConf.select(opob44, f"actor_rollout_ref.actor.grad_baselining.{key}"),
                )
        self.assertEqual(opob44.actor_rollout_ref.actor.grad_baselining.accum_device, "cuda")

    def test_per_traj_premises_hold(self):
        actor = self.cfg.actor_rollout_ref.actor
        self.assertTrue(actor.update_policy_per_traj)
        self.assertTrue(self.cfg.async_training.skip_recompute_old_log_prob)
        self.assertFalse(actor.use_dynamic_bsz)
        self.assertEqual(actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertIn(actor.loss_agg_mode, ("seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"))

    # ---------------------------------------------------------------- optimizer

    def test_fp32_masters_with_offloaded_moments(self):
        opt = self.cfg.actor_rollout_ref.actor.optim.override_optimizer_config
        self.assertTrue(opt.optimizer_cpu_offload)
        self.assertEqual(opt.optimizer_offload_fraction, 1.0)
        self.assertTrue(opt.use_torch_optimizer_for_cpu_offload)
        self.assertFalse(opt.overlap_cpu_optimizer_d2h_h2d)
        self.assertIsNone(OmegaConf.select(opt, "use_precision_aware_optimizer", default=None))
        self.assertIsNone(OmegaConf.select(opt, "main_params_dtype", default=None))
        mega = self.cfg.actor_rollout_ref.actor.megatron
        self.assertFalse(mega.override_ddp_config.grad_reduce_in_fp32)  # bf16 grad buffers stay
        self.assertEqual(mega.override_transformer_config.recompute_granularity, "full")

    # ---------------------------------------------------------------- layout

    def test_five_plus_three_pure_dp(self):
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 5)
        mega = self.cfg.actor_rollout_ref.actor.megatron
        self.assertEqual(mega.tensor_model_parallel_size, 1)
        self.assertEqual(mega.pipeline_model_parallel_size, 1)
        self.assertEqual(mega.context_parallel_size, 1)
        self.assertFalse(mega.sequence_parallel)

    def test_prompt_groups_divide_across_dp_ranks(self):
        actor = self.cfg.actor_rollout_ref.actor
        mega = actor.megatron
        dp = self.cfg.trainer.n_gpus_per_node // (
            mega.tensor_model_parallel_size * mega.pipeline_model_parallel_size * mega.context_parallel_size
        )
        self.assertEqual(dp, 3)
        mini = actor.ppo_mini_batch_size
        n = self.cfg.actor_rollout_ref.rollout.n
        self.assertEqual(mini, 33)
        self.assertEqual(mini % dp, 0, "group-scope OPOB needs whole prompt-groups per DP rank")
        self.assertEqual((mini * n) % dp, 0)
        self.assertEqual(self.cfg.async_training.bsz_per_dp_rank, mini)

    # ---------------------------------------------------------------- twin identity

    def test_keeps_everything_else_identical_to_the_5plus3_twin(self):
        twin = compose(TWIN)
        for path in (
            "actor_rollout_ref.model.path",
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.use_dynamic_bsz",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            "actor_rollout_ref.actor.update_policy_per_traj",
            "actor_rollout_ref.actor.loss_agg_mode",
            "actor_rollout_ref.actor.clip_ratio",
            "actor_rollout_ref.actor.clip_ratio_low",
            "actor_rollout_ref.actor.clip_ratio_high",
            "actor_rollout_ref.actor.clip_ratio_c",
            "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.actor.calculate_entropy",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.optim.clip_grad",
            "actor_rollout_ref.actor.optim.lr_decay_steps",
            "actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload",
            "actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction",
            "actor_rollout_ref.actor.megatron.override_ddp_config",
            "actor_rollout_ref.actor.megatron.override_transformer_config",
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
            "actor_rollout_ref.actor.megatron.sequence_parallel",
            "actor_rollout_ref.actor.megatron.seed",
            "actor_rollout_ref.actor.ess_scaling",
            "actor_rollout_ref.actor.use_rollout_log_probs",
            "actor_rollout_ref.actor.checkpoint.save_contents",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.temperature",
            "actor_rollout_ref.rollout.top_p",
            "actor_rollout_ref.rollout.val_kwargs",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            "algorithm.adv_estimator",
            "algorithm.rollout_correction",
            "data.train_files",
            "data.val_files",
            "data.max_prompt_length",
            "data.max_response_length",
            "data.seed",
            "trainer.save_freq",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "trainer.val_before_train",
            "rollout.n_gpus_per_node",
            "rollout.test_freq",
            "rollout.total_rollout_steps",
            "async_training.replay_buffer",
            "async_training.staleness_threshold",
            "async_training.require_batches",
            "async_training.trigger_parameter_sync_step",
            "async_training.skip_recompute_old_log_prob",
            "async_training.use_rollout_log_probs",
            "async_training.serialize_validation",
            "async_training.pause_generation_during_save",
            "async_training.bsz_per_dp_rank",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(twin, path),
                    f"{path} differs between the cpu-accum OPOB arm and its 5+3 twin",
                )

    def test_differs_from_the_twin_only_where_intended(self):
        twin = compose(TWIN)
        self.assertFalse(twin.actor_rollout_ref.actor.grad_baselining.enable)
        topt = twin.actor_rollout_ref.actor.optim.override_optimizer_config
        self.assertTrue(topt.use_precision_aware_optimizer)  # the twin's bf16-master deviation
        self.assertEqual(str(topt.main_params_dtype), "bfloat16")

    # ---------------------------------------------------------------- naming

    def test_experiment_name_tags(self):
        name = self.cfg.trainer.experiment_name
        for tag in (
            "opob-group-w2-normstd-cpuaccum-float32",
            "fp32-masters",
            " 5-3 ",
            "tp1dp3",
            "B-33",
            "replay tau-16 k-64 rmb-1",
        ):
            with self.subTest(tag=tag):
                self.assertIn(tag, name)


if __name__ == "__main__":
    unittest.main()
