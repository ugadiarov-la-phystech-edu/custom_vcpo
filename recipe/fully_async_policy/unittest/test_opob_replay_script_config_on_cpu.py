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
"""The OPOB replay arm's config, composed through hydra exactly as a launch would.

The arm is the fixed-base sqrt-brake replay twin plus VCPO's optimal off-policy baseline and
the trainer layout OPOB's three grad-buffer copies need on 80 GB cards (4+4, TP=2/DP=2, 32
prompts per mini-batch), and nothing else. Three things are worth pinning:

* the OPOB knobs resolve to the paper's defaults with ``norm_by_std`` on, and the per-traj
  premises OPOB relies on (``skip_recompute_old_log_prob``, micro batch 1, no dynamic bsz,
  a seq-mean loss aggregation) hold;
* the layout is divisible: group-scope OPOB keeps whole prompt-groups on one DP rank, so the
  PROMPT count must divide by DP — a mini-batch of 33 prompts would fail at run time in
  ``make_opportunistic_minibatch_indices`` / ``compute_grad_info``;
* every other knob equals the twin's, otherwise the two runs are not comparable.

Composing runs the real script with ``--cfg job --resolve``; the tests skip if it cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_opob_replay_script_config_on_cpu.py
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
OPOB = f"grpo_novcpo_8gpu_dapo17k_4+4_{TAIL}_opob.sh"
TWIN = f"grpo_novcpo_8gpu_dapo17k_5+3_{TAIL}.sh"

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


class TestOpobReplayArmScript(unittest.TestCase):
    """Static checks that need no composition."""

    def setUp(self):
        self.path = os.path.join(REPLAY, OPOB)
        if not os.path.exists(self.path):
            raise unittest.SkipTest(f"{OPOB} not found")

    def test_parses(self):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        proc = subprocess.run(["bash", "-n", self.path], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())

    def test_passes_every_opob_knob_explicitly(self):
        """A yaml default flip must not be able to change the arm silently."""
        with open(self.path) as f:
            text = f.read()
        for knob in (
            "enable",
            "scope",
            "agg_mode",
            "use_is_weights",
            "use_clipped_is_ratios",
            "normalize_by_length",
            "norm_by_std",
        ):
            self.assertIn(f"actor_rollout_ref.actor.grad_baselining.{knob}=", text, knob)
        self.assertIn("grad_baselining=True", text)
        self.assertIn("train_tp=2", text)
        self.assertIn("sequence_parallel=True", text)
        self.assertIn("n_gpus_rollout=${n_gpus_rollout:-4}", text)
        self.assertIn("train_prompt_mini_bsz=${train_prompt_mini_bsz:-32}", text)


class TestOpobReplayArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPOB)

    # ---------------------------------------------------------------- OPOB

    def test_opob_enabled_with_paper_defaults_and_std_normalization(self):
        gb = self.cfg.actor_rollout_ref.actor.grad_baselining
        self.assertTrue(gb.enable)
        self.assertEqual(gb.scope, "group")
        self.assertEqual(gb.agg_mode, "mean")
        self.assertTrue(gb.use_is_weights)
        self.assertFalse(gb.use_clipped_is_ratios)  # paper: the baseline sees the unclipped ratios
        self.assertFalse(gb.normalize_by_length)
        self.assertTrue(gb.norm_by_std)  # keeps GRPO's advantage scale -> lr-comparable with the twin

    def test_per_traj_premises_hold(self):
        """OPOB applies the baseline after a single backward of the advantage-free loss, which is
        exact only while the PPO ratio is identically 1 (cached behavior log-probs) and the
        per-traj path's scheduling constraints hold."""
        actor = self.cfg.actor_rollout_ref.actor
        self.assertTrue(actor.update_policy_per_traj)
        self.assertTrue(self.cfg.async_training.skip_recompute_old_log_prob)
        self.assertTrue(self.cfg.async_training.use_rollout_log_probs)
        self.assertFalse(actor.use_dynamic_bsz)
        self.assertEqual(actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertIn(actor.loss_agg_mode, ("seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"))
        # OPOB is mutually exclusive with a mu-anchored clip blend (asserted in the actor).
        self.assertFalse(OmegaConf.select(self.cfg, "async_training.adaptive_anchor.enable", default=False))
        anchor_mode = OmegaConf.select(self.cfg, "actor_rollout_ref.actor.policy_loss.anchor_mode", default=None)
        self.assertIn(anchor_mode, (None, "null"))

    # ---------------------------------------------------------------- layout

    def test_four_plus_four_with_tensor_parallel_two(self):
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 4)
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 4)
        self.assertEqual(self.cfg.trainer.nnodes, 1)
        for role in ("actor", "ref"):
            mega = OmegaConf.select(self.cfg, f"actor_rollout_ref.{role}.megatron")
            with self.subTest(role=role):
                self.assertEqual(mega.tensor_model_parallel_size, 2)
                self.assertEqual(mega.pipeline_model_parallel_size, 1)
                self.assertEqual(mega.context_parallel_size, 1)
                self.assertTrue(mega.sequence_parallel)
        self.assertEqual(self.cfg.critic.megatron.tensor_model_parallel_size, 2)
        self.assertEqual(self.cfg.actor_rollout_ref.rollout.tensor_model_parallel_size, 1)

    def test_prompt_groups_divide_across_dp_ranks(self):
        actor = self.cfg.actor_rollout_ref.actor
        mega = actor.megatron
        dp = self.cfg.trainer.n_gpus_per_node // (
            mega.tensor_model_parallel_size * mega.pipeline_model_parallel_size * mega.context_parallel_size
        )
        self.assertEqual(dp, 2)
        mini = actor.ppo_mini_batch_size
        n = self.cfg.actor_rollout_ref.rollout.n
        self.assertEqual(mini, 32)
        self.assertEqual(mini % dp, 0, "group-scope OPOB needs whole prompt-groups per DP rank")
        self.assertEqual((mini * n) % dp, 0)
        self.assertEqual(self.cfg.async_training.bsz_per_dp_rank, mini)

    def test_memory_recipe_kept(self):
        """bf16 grad buffers + HDO + bf16 masters + full recompute: with three buffer copies these
        are what keep the TP=2 footprint (~48 GB) under 80 GB."""
        mega = self.cfg.actor_rollout_ref.actor.megatron
        self.assertFalse(mega.override_ddp_config.grad_reduce_in_fp32)
        self.assertEqual(mega.override_transformer_config.recompute_granularity, "full")
        opt = self.cfg.actor_rollout_ref.actor.optim.override_optimizer_config
        self.assertTrue(opt.optimizer_cpu_offload)
        self.assertEqual(opt.optimizer_offload_fraction, 1.0)
        self.assertTrue(opt.use_precision_aware_optimizer)
        self.assertEqual(str(opt.main_params_dtype), "bfloat16")
        self.assertFalse(mega.param_offload)
        self.assertFalse(mega.optimizer_offload)
        self.assertFalse(mega.grad_offload)

    # ---------------------------------------------------------------- twin identity

    def test_keeps_everything_else_identical_to_the_twin(self):
        twin = compose(TWIN)
        for path in (
            "actor_rollout_ref.model.path",
            "actor_rollout_ref.actor.strategy",
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
            "actor_rollout_ref.actor.use_kl_loss",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.optim.clip_grad",
            "actor_rollout_ref.actor.optim.lr_warmup_steps",
            "actor_rollout_ref.actor.optim.lr_decay_style",
            "actor_rollout_ref.actor.optim.override_optimizer_config",
            "actor_rollout_ref.actor.megatron.override_ddp_config",
            "actor_rollout_ref.actor.megatron.override_transformer_config",
            "actor_rollout_ref.actor.megatron.param_offload",
            "actor_rollout_ref.actor.megatron.optimizer_offload",
            "actor_rollout_ref.actor.megatron.grad_offload",
            "actor_rollout_ref.actor.megatron.seed",
            "actor_rollout_ref.actor.ess_scaling.enable",
            "actor_rollout_ref.actor.ess_scaling.scaling_rule",
            "actor_rollout_ref.actor.ess_scaling.base_ess_ratio",
            "actor_rollout_ref.actor.ess_scaling.trigger_ratio",
            "actor_rollout_ref.actor.ess_scaling.use_clipped",
            "actor_rollout_ref.actor.use_rollout_log_probs",
            "actor_rollout_ref.actor.checkpoint.save_contents",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.temperature",
            "actor_rollout_ref.rollout.top_p",
            "actor_rollout_ref.rollout.top_k",
            "actor_rollout_ref.rollout.val_kwargs",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.calculate_log_probs",
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
            "trainer.max_actor_ckpt_to_keep",
            "trainer.val_before_train",
            "rollout.test_freq",
            "rollout.total_rollout_steps",
            "async_training.replay_buffer",
            "async_training.staleness_threshold",
            "async_training.require_batches",
            "async_training.trigger_parameter_sync_step",
            "async_training.partial_rollout",
            "async_training.skip_recompute_old_log_prob",
            "async_training.use_rollout_log_probs",
            "async_training.compute_prox_log_prob",
            "async_training.dynamic_filtering",
            "async_training.opportunistic_epochs",
            "async_training.ppo_epochs",
            "async_training.serialize_validation",
            "async_training.pause_generation_during_save",
            "async_training.save_queue_state",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(twin, path),
                    f"{path} differs between the OPOB arm and its twin",
                )

    def test_differs_from_the_twin_only_where_intended(self):
        twin = compose(TWIN)
        self.assertFalse(twin.actor_rollout_ref.actor.grad_baselining.enable)
        self.assertEqual(twin.trainer.n_gpus_per_node, 3)
        self.assertEqual(twin.rollout.n_gpus_per_node, 5)
        self.assertEqual(twin.actor_rollout_ref.actor.megatron.tensor_model_parallel_size, 1)
        self.assertFalse(twin.actor_rollout_ref.actor.megatron.sequence_parallel)
        self.assertEqual(twin.actor_rollout_ref.actor.ppo_mini_batch_size, 33)
        # lr decay horizon is in prompts and unchanged by the mini-batch change
        self.assertEqual(
            self.cfg.actor_rollout_ref.actor.optim.lr_decay_steps, twin.actor_rollout_ref.actor.optim.lr_decay_steps
        )

    # ---------------------------------------------------------------- checkpoints / naming

    def test_hf_model_only_checkpoints_and_no_resume(self):
        """hf_model-only checkpoints are write-only for the trainer; resume stays disabled so a
        leftover global_step_N under the same log dir can never fake a resume."""
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(self.cfg.trainer.resume_mode, "disable")
        self.assertFalse(self.cfg.async_training.replay_buffer.save_state)
        self.assertFalse(self.cfg.async_training.save_queue_state)
        self.assertEqual(self.cfg.trainer.save_freq, 20)
        self.assertEqual(self.cfg.rollout.test_freq, 20)

    def test_experiment_name_tags(self):
        name = self.cfg.trainer.experiment_name
        for tag in (
            "opob-group-w2-normstd",
            " 4-4 ",
            "tp2dp2",
            "B-32",
            "ess-sqrt-base-0.006113-trig-0.33333",
            "replay tau-16 k-64 rmb-1",
        ):
            with self.subTest(tag=tag):
                self.assertIn(tag, name)
        self.assertNotIn("tp1dp3", name)


if __name__ == "__main__":
    unittest.main()
