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
"""The openPangu-7B replay arm's config, composed through hydra exactly as a launch would.

Invariants worth protecting:

- the arm is its Qwen3-8B replay twin plus the model, the two trust_remote_code flags the re-aliased
  checkpoint needs, the BOS switch, the shorter replay depth (tau 8 / k 32, generation quota following
  k, min_ess 1.07, reuse decay nu=1) and the name — nothing else may drift, or the two runs stop being
  comparable up to model + depth;
- prompt processing and reward calculation are the openPangu SYNC arm's: BOS prepended
  (data.add_bos_token_to_prompt), both trust_remote_code keys (dataset tokenizer AND agent-loop /
  Megatron / vLLM side — they are independent and setting one without the other crashes the other half),
  validation sampling 0.8/0.7, stock math_dapo scorer (no custom reward function);
- the 3+3 smoke wrapper's overrides survive composition (a typo there turns a 20-minute plumbing check
  into a multi-hour run, since replay mode only stops once eviction drains the buffer).

Composing runs the real script with ``--cfg job --resolve``; the tests skip if that cannot run.

Run: pytest recipe/fully_async_policy/unittest/test_openpangu_replay_script_config_on_cpu.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer")

QWEN = "grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=16_k=64_min-ess=1.1_ess-lr-scale=0.5.sh"
PANGU = QWEN.replace("tau=16_k=64_min-ess=1.1", "tau=8_k=32_min-ess=1.07").replace(".sh", "_openpangu7b.sh")
SMOKE_3P3 = "smoke_test_openpangu7b_replay_3+3.sh"
PANGU_FRESH = PANGU.replace("_openpangu7b.sh", "_fresh=0.5_openpangu7b.sh")
REALIASED_MODEL = "/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama"

_COMPOSED = {}


def compose(script_name, stub_test_file=True):
    """Run the script with hydra's --cfg job --resolve and parse the config it would launch with."""
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


def compose_env(script_name, extra_env):
    """Uncached variant of compose() with extra environment overrides."""
    env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
    env.update(extra_env)
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
        proc = subprocess.run(
            ["bash", os.path.join(REPLAY, script_name), "--cfg", "job", "--resolve"],
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
        return OmegaConf.load(out.name)


def script_text(script_name):
    with open(os.path.join(REPLAY, script_name)) as f:
        return f.read()


class TestOpenPanguReplayArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(PANGU)
        cls.qwen = compose(QWEN)

    # ---- the model and what it needs --------------------------------------------------------

    def test_points_at_the_re_aliased_local_checkpoint(self):
        """Not the hub id: the remote modeling code does not import under this transformers and vLLM
        has no PanguEmbeddedForCausalLM, so the checkpoint is re-aliased to Llama locally."""
        self.assertEqual(self.cfg.actor_rollout_ref.model.path, REALIASED_MODEL)
        self.assertEqual(self.cfg.actor_rollout_ref.actor.strategy, "megatron")

    def test_both_trust_remote_code_keys_are_on(self):
        """data.trust_remote_code feeds the dataset tokenizer; actor_rollout_ref.model.trust_remote_code
        feeds the agent-loop tokenizer, the Megatron weight load and the vLLM engine. Independent keys."""
        self.assertIs(self.cfg.data.trust_remote_code, True)
        self.assertIs(self.cfg.actor_rollout_ref.model.trust_remote_code, True)
        self.assertFalse(self.qwen.data.trust_remote_code)
        self.assertFalse(self.qwen.actor_rollout_ref.model.trust_remote_code)

    def test_bos_is_prepended_like_the_openpangu_sync_arm(self):
        """The official recipe tokenizes the rendered template with a plain tokenizer(text) call, which
        prepends <s>; verl's default drops it. The flag is the one prompt-level difference from the twin
        and makes this arm comparable with the openPangu sync arm, not with the FSDP2 openPangu arm."""
        self.assertIs(self.cfg.data.add_bos_token_to_prompt, True)
        self.assertFalse(self.qwen.data.add_bos_token_to_prompt)

    def test_reward_is_the_stock_math_dapo_scorer(self):
        """No custom reward function, same as the twin and the openPangu sync arm: math_dapo is
        format-agnostic (last 'Answer:' match in the final 300 characters)."""
        self.assertIsNone(self.cfg.custom_reward_function.path)
        self.assertEqual(self.cfg.reward_model.reward_manager, self.qwen.reward_model.reward_manager)

    def test_validation_sampling_matches_the_twin_and_the_sync_arm(self):
        val, qval = self.cfg.actor_rollout_ref.rollout.val_kwargs, self.qwen.actor_rollout_ref.rollout.val_kwargs
        self.assertEqual((val.temperature, val.top_p, val.n), (0.8, 0.7, 1))
        self.assertEqual((val.temperature, val.top_p, val.n), (qval.temperature, qval.top_p, qval.n))

    # ---- the deliberate divergences --------------------------------------------------------

    def test_replay_depth_is_the_orz_style_short_one(self):
        """tau 8 / k 32 with the generation quota following k, min_ess 1.07 and reuse decay nu=1: the
        FSDP2 openPangu replay arm collapsed at update 57 under the twin's 16/64."""
        rb = self.cfg.async_training.replay_buffer
        self.assertIs(rb.enable, True)
        self.assertEqual((rb.tau, rb.staleness_threshold), (8, 32))
        self.assertEqual(self.cfg.async_training.staleness_threshold, 32)
        self.assertAlmostEqual(self.cfg.actor_rollout_ref.actor.ess_scaling.min_ess, 1.07)
        self.assertEqual(rb.reuse_halflife, 1)
        qrb = self.qwen.async_training.replay_buffer
        self.assertEqual((qrb.tau, qrb.staleness_threshold), (16, 64))
        self.assertEqual(self.qwen.async_training.staleness_threshold, 64)
        self.assertAlmostEqual(self.qwen.actor_rollout_ref.actor.ess_scaling.min_ess, 1.1)
        self.assertIsNone(qrb.reuse_halflife)

    def test_fresh_share_gate_is_off_by_default_like_the_twin(self):
        self.assertEqual(self.cfg.async_training.replay_buffer.min_fresh_ratio, 0)
        self.assertEqual(self.qwen.async_training.replay_buffer.min_fresh_ratio, 0)
        self.assertNotIn("fresh-", self.cfg.trainer.experiment_name)

    def test_experiment_name_identifies_model_bos_and_depth(self):
        name = self.cfg.trainer.experiment_name
        for tag in ("openPangu-7B", " bos", " nu-1 ", "tau-8 k-32", "min-ess-1.07"):
            self.assertIn(tag, name)
        self.assertNotIn("Qwen3-8B", name)
        self.assertNotEqual(name, self.qwen.trainer.experiment_name)

    def test_validates_and_saves_every_5_updates_unlike_the_twin(self):
        """test_freq/save_freq 5 (the twin: 20): the openPangu curves are read at finer
        granularity, in parameter-version units (one version per replay update)."""
        self.assertEqual(self.cfg.rollout.test_freq, 5)
        self.assertEqual(self.cfg.trainer.save_freq, 5)
        self.assertEqual(self.qwen.rollout.test_freq, 20)
        self.assertEqual(self.qwen.trainer.save_freq, 20)

    # ---- everything else is the twin's -----------------------------------------------------

    def test_keeps_everything_else_identical_to_the_qwen_twin(self):
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            "actor_rollout_ref.actor.use_dynamic_bsz",
            "actor_rollout_ref.actor.update_policy_per_traj",
            "actor_rollout_ref.actor.grad_baselining.enable",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
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
            "actor_rollout_ref.actor.ess_scaling.enable",
            "actor_rollout_ref.actor.ess_scaling.lr_scale",
            "actor_rollout_ref.actor.ess_scaling.use_clipped",
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size",
            "actor_rollout_ref.actor.megatron.param_offload",
            "actor_rollout_ref.actor.use_rollout_log_probs",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.temperature",
            "actor_rollout_ref.rollout.top_p",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            "actor_rollout_ref.rollout.calculate_log_probs",
            "data.train_files",
            "data.val_files",
            "data.prompt_key",
            "data.truncation",
            "data.max_prompt_length",
            "data.max_response_length",
            "data.filter_overlong_prompts",
            "data.return_raw_chat",
            "algorithm.adv_estimator",
            "algorithm.rollout_correction.rollout_is",
            "algorithm.rollout_correction.rollout_is_threshold",
            "algorithm.rollout_correction.bypass_mode",
            "algorithm.rollout_correction.use_policy_gradient",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            "rollout.total_rollout_steps",
            "async_training.replay_buffer.enable",
            "async_training.replay_buffer.requires_mini_batches",
            "async_training.replay_buffer.sampling_seed",
            "async_training.replay_buffer.save_state",
            "async_training.require_batches",
            "async_training.trigger_parameter_sync_step",
            "async_training.skip_recompute_old_log_prob",
            "async_training.use_rollout_log_probs",
            "async_training.partial_rollout",
            "async_training.serialize_validation",
            "async_training.pause_generation_during_save",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(self.qwen, path),
                    f"{path} differs between the openPangu arm and its Qwen twin",
                )

    def test_batch_shape_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        self.assertEqual((cfg.rollout.n_gpus_per_node, cfg.trainer.n_gpus_per_node), (5, 3))
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over {cfg.trainer.n_gpus_per_node} GPUs")

    def test_checkpoints_are_hf_model_only_and_not_resumable(self):
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(self.cfg.trainer.resume_mode, "disable")


class TestOpenPanguReplayFreshGateVariant(unittest.TestCase):
    """The fresh=0.5 variant is the base arm plus exactly one knob: the trainer waits for
    ceil(0.5 x mini) groups arrived from the rollouter since the last composition."""

    @classmethod
    def setUpClass(cls):
        cls.base = compose(PANGU)
        cls.cfg = compose(PANGU_FRESH)

    def test_gate_is_on_at_0_5_and_tagged(self):
        self.assertAlmostEqual(self.cfg.async_training.replay_buffer.min_fresh_ratio, 0.5)
        self.assertEqual(self.base.async_training.replay_buffer.min_fresh_ratio, 0)
        self.assertIn(" nu-1 fresh-0.5 ", self.cfg.trainer.experiment_name)
        self.assertNotIn("fresh-", self.base.trainer.experiment_name)

    def test_everything_else_equals_the_base(self):
        """Identical to the base once the gate knob, the name tag (which the experiment name
        propagates into the checkpoint / rollout-dump / validation-dump paths) and the
        validation / save cadence (15 vs 5, tested separately) are removed. In particular the
        checkpoint policy is the base's hf-only one."""
        import json

        a = OmegaConf.to_container(self.cfg, resolve=True)
        b = OmegaConf.to_container(self.base, resolve=True)
        for cfg in (a, b):
            cfg["async_training"]["replay_buffer"].pop("min_fresh_ratio")
            cfg["rollout"].pop("test_freq")
            cfg["trainer"].pop("save_freq")
        a_text = json.dumps(a, sort_keys=True)
        self.assertIn(" fresh-0.5", a_text)
        self.assertEqual(json.loads(a_text.replace(" fresh-0.5", "")), b)

    def test_knob_stays_env_overridable(self):
        text = script_text(PANGU_FRESH)
        self.assertIn("replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.5}", text)
        self.assertIn("FRESH-SHARE GATE", text)

    def test_locations_default_to_logs_under_exp_name(self):
        """log_dir (TensorBoard + rollout dumps) and CKPTS_DIR (checkpoints) both default to
        logs/<exp_name>, like the base arm."""
        exp = self.cfg.trainer.experiment_name
        self.assertEqual(self.cfg.trainer.default_local_dir, f"logs/{exp}")
        self.assertEqual(self.cfg.trainer.rollout_data_dir, f"logs/{exp}")
        self.assertEqual(self.base.trainer.default_local_dir, f"logs/{self.base.trainer.experiment_name}")

    def test_locations_are_env_overridable_separately(self):
        """log_dir and CKPTS_DIR can be pointed elsewhere independently (absolute paths); the
        checkpoint policy keys are unaffected by the move."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "runs", "x")
            ckpts = os.path.join(tmp, "ckpts", "x")
            env = dict(os.environ, TRAIN_FILE="/tmp/train.parquet", TEST_FILE="/tmp/test.parquet")
            env["log_dir"] = log_dir
            env["CKPTS_DIR"] = ckpts
            with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as out:
                proc = subprocess.run(
                    ["bash", os.path.join(REPLAY, PANGU_FRESH), "--cfg", "job", "--resolve"],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=out,
                    stderr=subprocess.PIPE,
                    timeout=900,
                )
                if proc.returncode != 0:
                    raise unittest.SkipTest(f"could not compose {PANGU_FRESH}: {proc.stderr.decode()[-300:]}")
                out.flush()
                out.seek(0)
                cfg = OmegaConf.load(out.name)
            self.assertEqual(cfg.trainer.default_local_dir, ckpts)
            self.assertEqual(cfg.trainer.rollout_data_dir, log_dir)
            self.assertEqual(cfg.trainer.experiment_name, self.cfg.trainer.experiment_name)
            self.assertIsNone(cfg.async_training.get("resumable_ckpts_to_keep", None))
            # the script creates both directories up front (mkdir -p), even for a --cfg dry run
            self.assertTrue(os.path.isdir(log_dir) and os.path.isdir(ckpts))
        text = script_text(PANGU_FRESH)
        self.assertIn('log_dir=${log_dir:-"logs/${exp_name_safe}"}', text)
        self.assertIn('CKPTS_DIR=${CKPTS_DIR:-"${log_dir}"}', text)
        self.assertIn('export TENSORBOARD_DIR="${log_dir}/tensorboard"', text)

    def test_validates_and_saves_every_15_updates(self):
        """test_freq/save_freq 15 on this variant (the base openPangu arm: 5, the twin: 20) in
        parameter-version units; both stay env-overridable."""
        self.assertEqual(self.cfg.rollout.test_freq, 15)
        self.assertEqual(self.cfg.trainer.save_freq, 15)
        self.assertEqual(self.base.rollout.test_freq, 5)
        self.assertEqual(self.base.trainer.save_freq, 5)
        cfg = compose_env(PANGU_FRESH, {"test_freq": "3", "save_freq": "6"})
        self.assertEqual(cfg.rollout.test_freq, 3)
        self.assertEqual(cfg.trainer.save_freq, 6)

    def test_checkpoints_are_hf_model_only_like_the_base(self):
        """hf_model at every save and nothing resumable: no dist-ckpt, replay buffer or queue
        snapshots, resume_mode=disable, nothing to prune. Same as the base arm."""
        for cfg in (self.cfg, self.base):
            self.assertEqual(list(cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
            self.assertEqual(cfg.trainer.resume_mode, "disable")
            self.assertIsNone(cfg.trainer.max_actor_ckpt_to_keep)  # would rmtree hf_model too
            self.assertIs(cfg.async_training.replay_buffer.save_state, False)
            self.assertIs(cfg.async_training.save_queue_state, False)
            self.assertIsNone(cfg.async_training.get("resumable_ckpts_to_keep", None))
        text = script_text(PANGU_FRESH)
        self.assertIn("ckpt_save_contents=${ckpt_save_contents:-\"['hf_model']\"}", text)
        self.assertIn("resumable_ckpts_to_keep=${resumable_ckpts_to_keep:-null}", text)
        self.assertIn("resume_mode=${resume_mode:-disable}", text)

    def test_two_tier_resumable_policy_stays_reachable_through_env(self):
        """The knobs documented in CHECKPOINTS switch the full resumable policy back on without
        editing the script: full save contents, one resumable checkpoint kept, resume_mode=auto,
        replay buffer + queue snapshots."""
        cfg = compose_env(
            PANGU_FRESH,
            {
                "ckpt_save_contents": "['model','optimizer','extra','hf_model']",
                "resumable_ckpts_to_keep": "1",
                "resume_mode": "auto",
                "replay_save_state": "True",
                "save_queue_state": "True",
            },
        )
        ckpt = cfg.actor_rollout_ref.actor.checkpoint
        self.assertEqual(list(ckpt.save_contents), ["model", "optimizer", "extra", "hf_model"])
        self.assertEqual(cfg.async_training.resumable_ckpts_to_keep, 1)
        self.assertEqual(cfg.trainer.resume_mode, "auto")
        self.assertIs(cfg.async_training.replay_buffer.save_state, True)
        self.assertIs(cfg.async_training.save_queue_state, True)
        self.assertIsNone(cfg.trainer.max_actor_ckpt_to_keep)
        self.assertEqual(cfg.trainer.save_freq, 15)


class TestOpenPanguReplayArmScriptText(unittest.TestCase):
    """Source tripwires for things hydra composition cannot see."""

    def setUp(self):
        if not os.path.exists(os.path.join(REPLAY, PANGU)):
            raise unittest.SkipTest(f"{PANGU} not found")
        self.text = script_text(PANGU)

    def test_hf_modules_cache_is_put_on_pythonpath(self):
        """Ray workers unpickle the trust_remote_code PanguTokenizer by reference
        (transformers_modules.<hash>...); that dynamic package must be importable in every worker."""
        self.assertIn("HF_MODULES_CACHE=${HF_MODULES_CACHE:-", self.text)
        self.assertIn('export PYTHONPATH="${HF_MODULES_CACHE}', self.text)

    def test_passes_both_trust_flags_and_the_bos_flag_to_hydra(self):
        self.assertIn("data.trust_remote_code=${trust_remote_code}", self.text)
        self.assertIn("actor_rollout_ref.model.trust_remote_code=${trust_remote_code}", self.text)
        self.assertIn("data.add_bos_token_to_prompt=${add_bos_token_to_prompt}", self.text)
        self.assertIn('async_training.replay_buffer.reuse_halflife="${replay_reuse_halflife}"', self.text)
        self.assertIn('async_training.replay_buffer.min_fresh_ratio="${replay_min_fresh_ratio}"', self.text)
        self.assertIn("replay_min_fresh_ratio=${replay_min_fresh_ratio:-0}", self.text)

    def test_no_asyncrl_style_divergences(self):
        """The AsyncRL port forced Transformer Engine fused attention and capped GPU memory through an env
        hook; this arm keeps verl's defaults (flash attention under Megatron auto) like every other arm."""
        self.assertNotIn("attention_backend=fused", self.text)
        self.assertNotIn("VERL_GPU_MEM_CAP_GB", self.text)
        self.assertIn("export VLLM_USE_FLASHINFER_SAMPLER=0", self.text)

    def test_header_documents_the_openpangu_specifics(self):
        for section in ("THE MODEL", "MEGATRON AND THE ATTENTION BIAS", "BOS", "REWARD", "REPLAY DEPTH", "MEMORY"):
            self.assertIn(f"# {section}", self.text)


class TestOpenPanguReplaySmoke3plus3(unittest.TestCase):
    """The 3+3 smoke: two updates (one fresh, one pure replay), checkpoint + bias-sync verification."""

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_3P3, stub_test_file=False)
        cls.arm = compose(PANGU)

    def test_layout_is_three_plus_three(self):
        self.assertEqual((self.cfg.rollout.n_gpus_per_node, self.cfg.trainer.n_gpus_per_node), (3, 3))

    def test_generation_budget_is_exactly_one_fresh_mini_batch(self):
        cfg = self.cfg
        per_update = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.async_training.require_batches
        self.assertEqual(cfg.rollout.total_rollout_steps, per_update)
        self.assertEqual(cfg.async_training.replay_buffer.requires_mini_batches, 1)

    def test_replay_eviction_ends_the_run_after_the_second_update(self):
        rb = self.cfg.async_training.replay_buffer
        self.assertIs(rb.enable, True)
        self.assertEqual(rb.staleness_threshold, 1)
        self.assertEqual(rb.tau, self.arm.async_training.replay_buffer.tau)
        self.assertEqual(rb.reuse_halflife, self.arm.async_training.replay_buffer.reuse_halflife)
        self.assertEqual(rb.min_fresh_ratio, 0)  # the gate would only add idle time to a 2-update smoke

    def test_batch_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0)

    def test_rollouts_are_cheap_but_the_length_is_real(self):
        self.assertLess(self.cfg.actor_rollout_ref.rollout.n, 16)
        self.assertEqual(self.cfg.data.max_response_length, self.arm.data.max_response_length)

    def test_validation_is_the_30_row_dapo_aime_after_every_update_and_not_before_training(self):
        cfg = self.cfg
        self.assertIs(cfg.trainer.val_before_train, False)
        self.assertEqual(cfg.rollout.test_freq, 1)
        val_files = cfg.data.val_files
        if isinstance(val_files, str):
            val_files = [val_files]
        self.assertEqual([os.path.basename(f) for f in val_files], ["aime-2024_smoke.parquet"])

    def test_checkpoint_after_every_update(self):
        self.assertEqual(self.cfg.async_training.trigger_parameter_sync_step, 1)
        self.assertEqual(self.cfg.trainer.save_freq, 1)
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])

    def test_gradient_cannot_be_identically_zero(self):
        self.assertGreater(self.cfg.actor_rollout_ref.actor.entropy_coeff, 0)
        self.assertGreater(self.cfg.actor_rollout_ref.actor.optim.lr, 1e-6)

    def test_it_still_uses_the_arms_model_flags_and_scorer(self):
        cfg = self.cfg
        self.assertEqual(cfg.actor_rollout_ref.model.path, REALIASED_MODEL)
        self.assertIs(cfg.data.trust_remote_code, True)
        self.assertIs(cfg.actor_rollout_ref.model.trust_remote_code, True)
        self.assertIs(cfg.data.add_bos_token_to_prompt, True)
        self.assertIsNone(cfg.custom_reward_function.path)
        self.assertIs(cfg.actor_rollout_ref.actor.ess_scaling.enable, True)

    def test_rollout_dumps_land_where_the_checks_look(self):
        self.assertEqual(self.cfg.trainer.rollout_data_dir, self.cfg.trainer.default_local_dir)

    def test_wrapper_verifies_against_the_base_model_in_bf16_and_checks_the_bias_sync(self):
        """verify_checkpoints --base-model catches a saver that forgot o_proj.bias (34 missing tensors);
        Megatron saves bf16; the KL check catches a wrong bias in the vLLM sync from update 1."""
        text = script_text(SMOKE_3P3)
        self.assertIn("--base-model", text)
        self.assertIn("--dtype BF16", text)
        self.assertIn("--expect 2", text)
        self.assertIn("rollout_corr/kl", text)
        self.assertIn("HF_MODULES_CACHE=${HF_MODULES_CACHE:-", text)


if __name__ == "__main__":
    unittest.main()
