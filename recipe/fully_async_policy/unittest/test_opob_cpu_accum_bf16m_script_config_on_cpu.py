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
"""The 5+3 OPOB arm that keeps the twin's HDO + bf16-master optimizer and moves only OPOB's
accumulators to the host (``..._opob_cpu-accum.sh``), composed through hydra.

Pinned: the optimizer block is byte-identical to the 5+3 twin's (HDO, precision-aware,
main_params_dtype=bfloat16), the OPOB accumulators are host-resident fp32, the layout is
5+3 / TP=1 with 33 prompts (whole groups per DP rank), and everything else equals the twin.
The fp32-master sibling (``..._opob_fp32-masters_cpu-accum.sh``) differs from this arm only
in the two precision-aware overrides.

Run: pytest recipe/fully_async_policy/unittest/test_opob_cpu_accum_bf16m_script_config_on_cpu.py
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
ARM = f"grpo_novcpo_8gpu_dapo17k_5+3_{TAIL}_opob_cpu-accum.sh"
TWIN = f"grpo_novcpo_8gpu_dapo17k_5+3_{TAIL}.sh"
FP32_SIBLING = f"grpo_novcpo_8gpu_dapo17k_5+3_{TAIL}_opob_fp32-masters_cpu-accum.sh"

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


class TestBf16MasterCpuAccumArmScript(unittest.TestCase):
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

    def test_keeps_the_bf16_master_overrides_and_passes_the_accumulator_knobs(self):
        cmd = self.text.split("python -m")[1]
        self.assertIn("override_optimizer_config.use_precision_aware_optimizer=True", cmd)
        self.assertIn("override_optimizer_config.main_params_dtype=bfloat16", cmd)
        self.assertIn("override_optimizer_config.optimizer_cpu_offload=True", cmd)
        for knob in ("accum_device", "accum_dtype", "accum_cpu_threads"):
            self.assertIn(f"actor_rollout_ref.actor.grad_baselining.{knob}=", cmd, knob)
        self.assertIn("grad_baselining_accum_device=${grad_baselining_accum_device:-cpu}", self.text)
        self.assertIn("n_gpus_rollout=${n_gpus_rollout:-5}", self.text)
        self.assertIn("train_tp=1", self.text)

    def test_smoke_wrapper_targets_this_script_and_expects_bf16_masters(self):
        smoke = os.path.join(REPLAY, ARM.replace(".sh", "_smoke_test_memory.sh"))
        self.assertTrue(os.path.exists(smoke), smoke)
        with open(smoke) as f:
            text = f.read()
        self.assertIn(f'SCRIPT="recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/{ARM}"', text)
        self.assertIn("train_prompt_mini_bsz % 3", text)
        self.assertIn("devices={'cpu'}", text)
        # the optimizer sanity check is inverted vs the fp32 sibling: bf16 masters are expected
        self.assertIn('bad "optimizer config lacks main_params_dtype=bfloat16', text)
        proc = subprocess.run(["bash", "-n", smoke], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())


class TestBf16MasterCpuAccumArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(ARM)

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

    def test_optimizer_block_equals_the_twin(self):
        twin = compose(TWIN)
        mine = self.cfg.actor_rollout_ref.actor.optim
        self.assertEqual(
            OmegaConf.to_container(mine.override_optimizer_config, resolve=True),
            OmegaConf.to_container(twin.actor_rollout_ref.actor.optim.override_optimizer_config, resolve=True),
        )
        self.assertTrue(mine.override_optimizer_config.use_precision_aware_optimizer)
        self.assertEqual(str(mine.override_optimizer_config.main_params_dtype), "bfloat16")
        self.assertTrue(mine.override_optimizer_config.optimizer_cpu_offload)

    def test_layout_and_divisibility(self):
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 5)
        mega = self.cfg.actor_rollout_ref.actor.megatron
        self.assertEqual(mega.tensor_model_parallel_size, 1)
        self.assertFalse(mega.sequence_parallel)
        mini = self.cfg.actor_rollout_ref.actor.ppo_mini_batch_size
        self.assertEqual(mini, 33)
        self.assertEqual(mini % 3, 0, "group-scope OPOB needs whole prompt-groups per DP rank")

    def test_differs_from_the_fp32_sibling_only_in_master_precision(self):
        sib = compose(FP32_SIBLING)
        for path in (
            "actor_rollout_ref.actor.grad_baselining",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.megatron",
            "actor_rollout_ref.actor.ess_scaling",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload",
            "actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction",
            "async_training.replay_buffer",
            "trainer.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            "data.max_response_length",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(sib, path))
        sopt = sib.actor_rollout_ref.actor.optim.override_optimizer_config
        self.assertIsNone(OmegaConf.select(sopt, "use_precision_aware_optimizer", default=None))
        self.assertIsNone(OmegaConf.select(sopt, "main_params_dtype", default=None))

    def test_keeps_everything_else_identical_to_the_twin(self):
        twin = compose(TWIN)
        for path in (
            "actor_rollout_ref.model.path",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.update_policy_per_traj",
            "actor_rollout_ref.actor.loss_agg_mode",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.optim.override_optimizer_config",
            "actor_rollout_ref.actor.megatron",
            "actor_rollout_ref.actor.ess_scaling",
            "actor_rollout_ref.actor.checkpoint.save_contents",
            "actor_rollout_ref.rollout.n",
            "algorithm.rollout_correction",
            "data.train_files",
            "data.max_response_length",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            "rollout.test_freq",
            "async_training.replay_buffer",
            "async_training.staleness_threshold",
            "async_training.skip_recompute_old_log_prob",
            "async_training.bsz_per_dp_rank",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(twin, path))
        self.assertFalse(twin.actor_rollout_ref.actor.grad_baselining.enable)

    def test_experiment_name_tags(self):
        name = self.cfg.trainer.experiment_name
        for tag in ("opob-group-w2-normstd-cpuaccum-float32", "bf16-masters", " 5-3 ", "tp1dp3", "B-33"):
            with self.subTest(tag=tag):
                self.assertIn(tag, name)
        self.assertNotIn("fp32-masters", name)


if __name__ == "__main__":
    unittest.main()
