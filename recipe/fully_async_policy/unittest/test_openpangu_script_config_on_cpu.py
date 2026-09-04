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
SMOKE_3P3 = "smoke_test_openpangu_3+3.sh"

# Architectures shipped inside transformers itself; anything else may carry custom code. Local
# directories count as custom on purpose: the openPangu checkpoint is re-aliased to Llama but its
# TOKENIZER still resolves through tokenization_openpangu.py.
NATIVE_MODEL_PREFIXES = ("Qwen/", "meta-llama/", "mistralai/", "deepseek-ai/")

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


def needs_remote_code(model_path):
    return not model_path.startswith(NATIVE_MODEL_PREFIXES)


class TestOpenPanguArmConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPENPANGU, TEST_FILE="/tmp/test.parquet")

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
        qwen = compose(QWEN_FSDP2, TEST_FILE="/tmp/test.parquet")
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


class TestOpenPanguSmoke3plus3(unittest.TestCase):
    """The 3+3 smoke test: two cheap steps, validated and checkpointed at every one.

    It runs the real arm with env overrides, so what matters is that the overrides survive
    composition - a typo in one of them turns a 20-minute check into a multi-hour run, or into one
    that never validates.
    """

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_3P3)

    def test_layout_is_three_plus_three(self):
        self.assertEqual(self.cfg.rollout.n_gpus_per_node, 3)
        self.assertEqual(self.cfg.trainer.n_gpus_per_node, 3)

    def test_it_runs_exactly_two_trainer_steps(self):
        cfg = self.cfg
        per_step = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.async_training.require_batches
        self.assertEqual(cfg.rollout.total_rollout_steps, 2 * per_step)

    def test_batch_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over 3 GPUs")

    def test_rollouts_are_cheap(self):
        """n is the speed lever. The response length is deliberately the arm's own 8192 - generation
        is exercised at the real length - but it must never exceed it, or the smoke test would be
        heavier than the run it stands in for."""
        self.assertLess(self.cfg.actor_rollout_ref.rollout.n, 16)
        arm = compose(OPENPANGU, TEST_FILE="/tmp/test.parquet")
        self.assertLessEqual(self.cfg.data.max_response_length, arm.data.max_response_length)

    def test_validation_is_aime_2024_only_every_step_and_not_before_training(self):
        cfg = self.cfg
        self.assertIs(cfg.trainer.val_before_train, False)
        self.assertEqual(cfg.rollout.test_freq, 1)
        val_files = cfg.data.val_files
        if isinstance(val_files, str):
            val_files = [val_files]
        # the deduplicated copy: same 30 AIME-2024 problems, without the 32x duplication that
        # made a validation sweep the dominant cost of this test
        self.assertEqual([os.path.basename(f) for f in val_files], ["aime-2024_smoke.parquet"])

    def test_checkpoint_every_step(self):
        self.assertEqual(self.cfg.trainer.save_freq, 1)
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])

    def test_gradient_cannot_be_identically_zero(self):
        """At a 512-token cap every answer is truncated, so all rewards in a group tie and the GRPO
        advantage is 0. Without an entropy term the weights cannot move and "the checkpoints differ"
        would be unverifiable."""
        self.assertGreater(self.cfg.actor_rollout_ref.actor.entropy_coeff, 0)
        self.assertGreater(self.cfg.actor_rollout_ref.actor.optim.lr, 1e-6)

    def test_it_still_exercises_the_openpangu_path(self):
        """The point of the smoke test: the custom tokenizer and the re-aliased checkpoint."""
        self.assertIs(self.cfg.actor_rollout_ref.model.trust_remote_code, True)
        self.assertIs(self.cfg.data.trust_remote_code, True)
        self.assertIn("openPangu", self.cfg.actor_rollout_ref.model.path)


class TestOpenPanguScriptsExportTheHfModulesCache(unittest.TestCase):
    """Both openPangu scripts must put HF_MODULES_CACHE on PYTHONPATH before launching.

    Ray serializes the trust_remote_code tokenizer BY REFERENCE
    (transformers_modules.<hash>.tokenization_openpangu.PanguTokenizer) into the rollouter actor's
    constructor arguments. That dynamic package is only on sys.path in a process that has itself
    loaded remote code - the driver, not the actors - so without this the run dies at actor creation
    with "ModuleNotFoundError: No module named 'transformers_modules'", before touching a GPU and
    with nothing in the message pointing at the tokenizer. Cost one dead run on remote_smoke.

    Asserted on the script text because the export is an environment effect: it is invisible to
    hydra composition. The mechanism itself is covered by
    tests/models/test_openpangu_tokenizer_contract_on_cpu.py.
    """

    def _read(self, name):
        path = os.path.join(BASELINE, name)
        if not os.path.exists(path):
            self.skipTest(f"{name} not found")
        with open(path) as f:
            return f.read()

    def test_both_scripts_prepend_the_hf_modules_cache(self):
        for name in (OPENPANGU, SMOKE_3P3):
            with self.subTest(script=name):
                text = self._read(name)
                self.assertIn("HF_MODULES_CACHE", text)
                self.assertIn('export PYTHONPATH="${HF_MODULES_CACHE}', text)

    def test_the_export_is_idempotent_and_survives_an_empty_pythonpath(self):
        """Two guards worth having: the wrapper runs the arm, so the block runs twice, and an
        unset PYTHONPATH must not become a bare ':' (which would put the CWD on sys.path)."""
        text = self._read(OPENPANGU)
        self.assertIn('case ":${PYTHONPATH:-}:" in', text, "the block must skip when already present")
        self.assertIn("${PYTHONPATH:+:${PYTHONPATH}}", text, "must not emit a trailing colon")

    def test_qwen_arms_do_not_need_it(self):
        """Native architectures load no remote code, so nothing references transformers_modules."""
        self.assertNotIn("HF_MODULES_CACHE", self._read(QWEN_FSDP2))


class TestQwenArmNeedsNoRemoteCode(unittest.TestCase):
    """The converse, so the invariant is about custom-code models rather than about one script."""

    def test_native_model_does_not_enable_remote_code(self):
        cfg = compose(QWEN_FSDP2, TEST_FILE="/tmp/test.parquet")
        path = cfg.actor_rollout_ref.model.path
        if needs_remote_code(path):
            self.skipTest(f"{path} is not a natively-supported architecture")
        self.assertIs(cfg.actor_rollout_ref.model.trust_remote_code, False)
        self.assertIs(cfg.data.trust_remote_code, False)


# ============================================================================ Megatron arms

OPENPANGU_MEGATRON = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh"
QWEN_MEGATRON = "grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_is-pg.sh"
SMOKE_MEGATRON_3P3 = "smoke_test_openpangu_megatron_3+3.sh"
OPENPANGU_SYNC = "main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh"
QWEN_SYNC = "main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh"


class _OpenPanguMegatronArmMixin:
    """What every openPangu arm on strategy=megatron must set, whichever trainer it uses."""

    def test_is_megatron(self):
        self.assertEqual(self.cfg.actor_rollout_ref.actor.strategy, "megatron")

    def test_points_at_a_realiased_local_checkpoint(self):
        path = self.cfg.actor_rollout_ref.model.path
        self.assertFalse(path.startswith("FreedomIntelligence/"), f"{path} is the stock hub checkpoint")
        self.assertIn("openPangu", path)

    def test_sets_both_trust_remote_code_keys(self):
        self.assertIs(self.cfg.actor_rollout_ref.model.trust_remote_code, True)
        self.assertIs(self.cfg.data.trust_remote_code, True)

    def test_prepends_bos_like_the_official_recipe(self):
        """The decision for the Megatron arms: train on prompts that start with <s>."""
        self.assertIs(self.cfg.data.add_bos_token_to_prompt, True)

    def test_no_attention_backend_override_and_no_memory_cap(self):
        """Not ported from the AsyncRL scripts: forced TE fused attention (auto already picks it on
        Hopper, and forcing removes the flash fallback) and the H200 per-process memory cap, which
        would throttle an 80 GB H100 trainer to 72 GB."""
        otc = OmegaConf.select(self.cfg, "actor_rollout_ref.actor.megatron.override_transformer_config") or {}
        # the stock megatron config carries attention_backend=flash (TE picks fused on Hopper anyway);
        # what must not appear is the forced "fused" of the AsyncRL scripts
        self.assertNotEqual(otc.get("attention_backend"), "fused")
        qwen = compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet")
        qwen_otc = OmegaConf.select(qwen, "actor_rollout_ref.actor.megatron.override_transformer_config") or {}
        self.assertEqual(otc.get("attention_backend"), qwen_otc.get("attention_backend"))
        with open(os.path.join(BASELINE, self.SCRIPT)) as f:
            text = f.read()
        self.assertNotIn("VERL_GPU_MEM_CAP_GB", text)
        self.assertNotIn("attention_backend=fused", text)

    def test_uses_the_bf16_hdo_recipe(self):
        opt = OmegaConf.select(self.cfg, "actor_rollout_ref.actor.optim.override_optimizer_config")
        self.assertTrue(opt.optimizer_cpu_offload)
        self.assertEqual(opt.main_params_dtype, "bfloat16")
        self.assertEqual(self.cfg.actor_rollout_ref.actor.megatron.dtype, "bfloat16")

    def test_checkpoints_are_hf_model_only_and_not_resumable(self):
        self.assertEqual(list(self.cfg.actor_rollout_ref.actor.checkpoint.save_contents), ["hf_model"])
        self.assertEqual(self.cfg.trainer.resume_mode, "disable")


class TestOpenPanguMegatronIsPgArm(_OpenPanguMegatronArmMixin, unittest.TestCase):
    SCRIPT = OPENPANGU_MEGATRON

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPENPANGU_MEGATRON, TEST_FILE="/tmp/test.parquet")

    def test_keeps_the_arm_identical_to_the_qwen_megatron_twin(self):
        """Only the model, the remote-code flags, BOS and the name may differ."""
        qwen = compose(QWEN_MEGATRON, TEST_FILE="/tmp/test.parquet")
        for path in (
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_epochs",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.policy_loss.rollout_correction",
            "actor_rollout_ref.actor.calculate_entropy",
            "actor_rollout_ref.actor.loss_agg_mode",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.optim.override_optimizer_config",
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
            "actor_rollout_ref.actor.megatron.override_transformer_config",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.val_kwargs.temperature",
            "actor_rollout_ref.rollout.val_kwargs.top_p",
            "data.max_prompt_length",
            "data.max_response_length",
            "trainer.save_freq",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            "rollout.test_freq",
            "rollout.total_rollout_steps",
            "async_training.staleness_threshold",
            "async_training.use_rollout_log_probs",
            "async_training.serialize_validation",
            "async_training.pause_generation_during_save",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(qwen, path),
                    f"{path} differs between the openPangu Megatron arm and its Qwen twin",
                )

    def test_differs_from_the_fsdp2_openpangu_arm_only_where_documented(self):
        """Backend, BOS and validation sampling: three documented differences, nothing else in the
        experiment definition."""
        fsdp2 = compose(OPENPANGU, TEST_FILE="/tmp/test.parquet")
        self.assertEqual(fsdp2.actor_rollout_ref.actor.strategy, "fsdp2")
        self.assertIs(fsdp2.data.add_bos_token_to_prompt, False)
        fsdp2_val_t = fsdp2.actor_rollout_ref.rollout.val_kwargs.temperature
        self.assertNotEqual(fsdp2_val_t, self.cfg.actor_rollout_ref.rollout.val_kwargs.temperature)
        for path in (
            "actor_rollout_ref.model.path",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_epochs",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.rollout.n",
            "data.max_prompt_length",
            "data.max_response_length",
            "async_training.staleness_threshold",
            "rollout.total_rollout_steps",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(fsdp2, path))

    def test_batch_shape_divides_across_the_trainer_gpus(self):
        cfg = self.cfg
        seqs = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
        self.assertEqual(seqs % cfg.trainer.n_gpus_per_node, 0, f"{seqs} seqs over {cfg.trainer.n_gpus_per_node} GPUs")


class TestOpenPanguMegatronSmoke3plus3(_OpenPanguMegatronArmMixin, unittest.TestCase):
    """The wrapper must run the MEGATRON arm through the FSDP2 smoke's shortening, unchanged."""

    SCRIPT = OPENPANGU_MEGATRON  # the wrapper's env overrides do not touch the arm's text

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_MEGATRON_3P3)

    def test_layout_is_three_plus_three_and_two_steps(self):
        cfg = self.cfg
        self.assertEqual(cfg.rollout.n_gpus_per_node, 3)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 3)
        per_step = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.async_training.require_batches
        self.assertEqual(cfg.rollout.total_rollout_steps, 2 * per_step)
        self.assertEqual(cfg.trainer.save_freq, 1)
        self.assertEqual(cfg.rollout.test_freq, 1)

    def test_same_shortening_as_the_fsdp2_smoke(self):
        fsdp2 = compose(SMOKE_3P3)
        for path in (
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.actor.optim.lr",
            "data.max_response_length",
            "trainer.val_before_train",
            "rollout.total_rollout_steps",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(fsdp2, path))

    def test_wrapper_verifies_bf16_checkpoints(self):
        with open(os.path.join(BASELINE, SMOKE_MEGATRON_3P3)) as f:
            text = f.read()
        self.assertIn("VERIFY_DTYPE=${VERIFY_DTYPE:-BF16}", text)
        with open(os.path.join(BASELINE, SMOKE_3P3)) as f:
            base = f.read()
        self.assertIn('--dtype "${VERIFY_DTYPE:-F32}"', base, "the FSDP2 smoke must keep verifying fp32 by default")
        self.assertIn("--base-model", base, "parameter-name diff against the base model is the o_proj.bias check")


class TestOpenPanguSyncArm(_OpenPanguMegatronArmMixin, unittest.TestCase):
    SCRIPT = OPENPANGU_SYNC

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(OPENPANGU_SYNC, TEST_FILE="/tmp/test.parquet")

    def test_keeps_the_arm_identical_to_the_qwen_sync_twin(self):
        qwen = compose(QWEN_SYNC, TEST_FILE="/tmp/test.parquet")
        for path in (
            "actor_rollout_ref.hybrid_engine",
            "actor_rollout_ref.actor.strategy",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_epochs",
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.clip_ratio",
            "actor_rollout_ref.actor.clip_ratio_c",
            "actor_rollout_ref.actor.loss_agg_mode",
            "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.optim.override_optimizer_config",
            "actor_rollout_ref.actor.megatron.seed",
            "actor_rollout_ref.actor.data_loader_seed",
            "actor_rollout_ref.rollout.n",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            "actor_rollout_ref.rollout.val_kwargs.temperature",
            "actor_rollout_ref.rollout.val_kwargs.top_p",
            "data.train_batch_size",
            "data.seed",
            "data.max_prompt_length",
            "data.max_response_length",
            "trainer.test_freq",
            "trainer.save_freq",
            "trainer.resume_mode",
            "trainer.n_gpus_per_node",
            "trainer.total_epochs",
            "critic.megatron.seed",
        ):
            with self.subTest(key=path):
                self.assertEqual(
                    OmegaConf.select(self.cfg, path),
                    OmegaConf.select(qwen, path),
                    f"{path} differs between the openPangu sync arm and its Qwen twin",
                )

    def test_seed_is_in_the_experiment_name(self):
        self.assertIn("seed-", self.cfg.trainer.experiment_name)
        self.assertIn("bos", self.cfg.trainer.experiment_name)


class TestBosFlagDefaultsOffEverywhereElse(unittest.TestCase):
    """The flag exists for openPangu; every other script composes with it off, exactly as before."""

    def test_existing_scripts_do_not_prepend_bos(self):
        for name, env in (
            (OPENPANGU, {"TEST_FILE": "/tmp/test.parquet"}),
            (QWEN_FSDP2, {"TEST_FILE": "/tmp/test.parquet"}),
            (QWEN_MEGATRON, {"TEST_FILE": "/tmp/test.parquet"}),
            (QWEN_SYNC, {"TEST_FILE": "/tmp/test.parquet"}),
            (SMOKE_3P3, {}),
        ):
            with self.subTest(script=name):
                cfg = compose(name, **env)
                self.assertIs(cfg.data.add_bos_token_to_prompt, False)


class TestMegatronScriptsExportTheHfModulesCache(unittest.TestCase):
    def test_all_openpangu_scripts_prepend_the_hf_modules_cache(self):
        for name in (OPENPANGU_MEGATRON, OPENPANGU_SYNC, SMOKE_OOM_SYNC):
            with self.subTest(script=name):
                with open(os.path.join(BASELINE, name)) as f:
                    text = f.read()
                self.assertIn('case ":${PYTHONPATH:-}:" in', text)
                self.assertIn('export PYTHONPATH="${HF_MODULES_CACHE}', text)


SMOKE_OOM_SYNC = "smoke_test_oom_openpangu7b_sync.sh"
SMOKE_OOM_QWEN_SYNC = "smoke_test_oom_qwen3-8b_sync.sh"


class TestOpenPanguSyncOomSmoke(_OpenPanguMegatronArmMixin, unittest.TestCase):
    """The fast check of the sync openPangu arm: the Qwen OOM-smoke protocol on the openPangu arm,
    plus a checkpoint verification against the base model. Composed as a launch would."""

    SCRIPT = OPENPANGU_SYNC  # the wrapper's env overrides do not touch the arm's text

    @classmethod
    def setUpClass(cls):
        if sys.platform.startswith("win"):
            raise unittest.SkipTest("bash-only")
        cls.cfg = compose(SMOKE_OOM_SYNC)

    def test_runs_two_steps_on_the_real_8_gpu_layout(self):
        cfg = self.cfg
        self.assertEqual(cfg.trainer.total_training_steps, 2)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 8)
        self.assertIs(cfg.actor_rollout_ref.hybrid_engine, True)
        self.assertIs(cfg.trainer.val_before_train, False)

    def test_same_shortening_as_the_qwen_oom_smoke(self):
        """Batch 16 / mini 8 (two updates per step), the arm's own 8192 tokens and memory knobs."""
        qwen = compose(SMOKE_OOM_QWEN_SYNC)
        for path in (
            "data.train_batch_size",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.rollout.n",
            "data.max_response_length",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            "actor_rollout_ref.actor.megatron.param_offload",
            "trainer.total_training_steps",
            "trainer.val_before_train",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(qwen, path))
        self.assertEqual(self.cfg.data.train_batch_size, 16)
        self.assertEqual(self.cfg.actor_rollout_ref.actor.ppo_mini_batch_size, 8)
        self.assertEqual(self.cfg.data.max_response_length, 8192)

    def test_memory_settings_are_the_arm_s_own(self):
        arm = compose(OPENPANGU_SYNC, TEST_FILE="/tmp/test.parquet")
        for path in (
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            "data.max_response_length",
            "data.max_prompt_length",
            "actor_rollout_ref.actor.megatron.param_offload",
            "actor_rollout_ref.actor.optim.override_optimizer_config",
        ):
            with self.subTest(key=path):
                self.assertEqual(OmegaConf.select(self.cfg, path), OmegaConf.select(arm, path))

    def test_wrapper_verifies_the_checkpoint_against_the_base_model(self):
        with open(os.path.join(BASELINE, SMOKE_OOM_SYNC)) as f:
            text = f.read()
        self.assertIn("verify_checkpoints.py", text)
        self.assertIn("--base-model", text, "the parameter-name diff is the o_proj.bias check")
        self.assertIn("--dtype BF16", text)
        self.assertIn("--no-timing-state", text, "main_ppo writes no timing_state.json")
        self.assertIn("froze .* MLP bias tensors at zero", text)

    def test_verifier_accepts_the_no_timing_state_switch(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, os.path.join(BASELINE, "verify_checkpoints.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-300:])
        self.assertIn("--no-timing-state", proc.stdout)


if __name__ == "__main__":
    unittest.main()
