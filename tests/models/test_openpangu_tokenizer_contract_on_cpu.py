# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""What the openPangu arm assumes about the re-aliased checkpoint, checked against the real one.

CPU only, but it needs the ~16 GB checkpoint on disk, so it is skipped unless
OPENPANGU_MODEL_PATH is set:

    OPENPANGU_MODEL_PATH=/home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama \\
        pytest tests/models/test_openpangu_tokenizer_contract_on_cpu.py -q

Each failure here corresponds to a silent, expensive failure mode in a real run: an empty dataset
(broken chat template), rollouts that never stop (unsamplable EOS), a run trained in the wrong
thinking mode, or checkpoints nothing can load back.
"""

import os
import tempfile
import unittest

MODEL_PATH = os.environ.get("OPENPANGU_MODEL_PATH")
DATA_DIR = os.environ.get("OPENPANGU_DATA_DIR", "/home/jovyan/datasets/math_datasets/dapo")
MAX_PROMPT_LENGTH = 2048

THINK_OPEN, THINK_CLOSE = "[unused16]", "[unused17]"


@unittest.skipUnless(MODEL_PATH, "set OPENPANGU_MODEL_PATH to the re-aliased checkpoint")
class TestOpenPanguCheckpointContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        cls.cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # ---------------------------------------------------------------- the re-alias itself

    def test_config_presents_as_llama(self):
        """Why: transformers 4.57.6 cannot import the stock remote modeling code, and vLLM 0.11.0
        has no PanguEmbeddedForCausalLM."""
        self.assertEqual(self.cfg.model_type, "llama")
        self.assertEqual(self.cfg.architectures, ["LlamaForCausalLM"])

    def test_attention_bias_survived_the_realias(self):
        """Pangu has bias on q/k/v/o; plain Llama does not. Losing this loads a different model."""
        self.assertIs(getattr(self.cfg, "attention_bias", False), True)
        self.assertIs(getattr(self.cfg, "mlp_bias", True), False)

    # ---------------------------------------------------------------- tokenizer

    def test_tokenizer_loads_through_verl_s_loader(self):
        """verl calls hf_tokenizer(path, trust_remote_code=...) with its default use_fast; openPangu
        has no fast variant, so the slow custom class must resolve anyway."""
        from verl.utils import hf_tokenizer

        tok = hf_tokenizer(MODEL_PATH, trust_remote_code=True)
        self.assertEqual(type(tok).__name__, "PanguTokenizer")
        self.assertFalse(tok.is_fast)

    def test_eos_is_samplable_under_verl_s_vllm_logit_mask(self):
        """verl passes no stop/eos to vLLM and masks logits[..., len(tokenizer):] = -inf. An eos id
        at or above len(tokenizer) would make every rollout run to the length cap."""
        self.assertLess(self.cfg.eos_token_id, len(self.tok))
        self.assertLess(self.tok.eos_token_id, len(self.tok))

    def test_tokenizer_and_config_agree_on_eos(self):
        self.assertEqual(self.tok.eos_token_id, self.cfg.eos_token_id)

    # ---------------------------------------------------------------- chat template

    def test_chat_template_is_set(self):
        """A missing template raises inside the overlong-prompt filter, which swallows the
        exception and returns max_prompt_length+1 - i.e. it silently empties the dataset."""
        self.assertIsNotNone(self.tok.chat_template)

    def test_template_renders_the_pangu_role_markers(self):
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": "2+2?"}], add_generation_prompt=True, tokenize=False
        )
        for marker in ("[unused9]", "[unused10]"):
            self.assertIn(marker, rendered)
        self.assertTrue(rendered.endswith("助手："), rendered[-20:])

    def test_slow_think_is_the_default_mode(self):
        """The arm trains slow think. A template injecting /no_think or /auto_think would mean the
        run is not the experiment we think it is."""
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": "2+2?"}], add_generation_prompt=True, tokenize=False
        )
        self.assertNotIn("/no_think", rendered)
        self.assertNotIn("/auto_think", rendered)

    def test_generation_prompt_only_appends(self):
        """What verl's agent loop assumes when it concatenates prompt and response."""
        messages = [{"role": "user", "content": "2+2?"}]
        with_gen = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without = self.tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        self.assertTrue(with_gen.startswith(without))

    # ---------------------------------------------------------------- reward decode

    def test_thinking_delimiters_survive_the_reward_decode(self):
        """Reward managers decode with skip_special_tokens=True. [unused16]/[unused17] are ordinary
        tokens and survive (so a delimiter-aware scorer would be possible); the eos does not."""
        for marker in (THINK_OPEN, THINK_CLOSE):
            ids = self.tok.encode(marker, add_special_tokens=False)
            self.assertEqual(len(ids), 1, f"{marker} should be a single token")
            self.assertIn(marker, self.tok.decode(ids, skip_special_tokens=True))
        eos_ids = self.tok.encode(self.tok.eos_token, add_special_tokens=False)
        self.assertEqual(self.tok.decode(eos_ids, skip_special_tokens=True), "")

    def test_a_slow_think_response_still_scores_after_a_real_round_trip(self):
        """End to end through the tokenizer: encode -> decode(skip_special_tokens=True) -> scorer,
        exactly what the reward manager does."""
        from verl.utils.reward_score import math_dapo

        response = (
            f"{THINK_OPEN}\nSuppose 7.\nAnswer: 7\nNo, redo.\n" + "Filler. " * 40 + f"\n{THINK_CLOSE}\nAnswer: 42"
        )
        decoded = self.tok.decode(self.tok.encode(response, add_special_tokens=False), skip_special_tokens=True)
        result = math_dapo.compute_score(decoded, "42")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["pred"], "42")

    # ---------------------------------------------------------------- checkpointing

    def test_tokenizer_round_trips_through_save_pretrained(self):
        """Every checkpoint this arm writes carries the tokenizer; if it cannot be loaded back, the
        checkpoint is useless for eval. save_pretrained must copy the custom module and auto_map."""
        import json

        from transformers import AutoTokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            self.tok.save_pretrained(tmpdir)
            self.assertIn("tokenization_openpangu.py", os.listdir(tmpdir))
            with open(os.path.join(tmpdir, "tokenizer_config.json")) as f:
                self.assertIn("auto_map", json.load(f))
            reloaded = AutoTokenizer.from_pretrained(tmpdir, trust_remote_code=True)
            self.assertEqual(reloaded.eos_token_id, self.tok.eos_token_id)
            self.assertEqual(len(reloaded), len(self.tok))
            messages = [{"role": "user", "content": "2+2?"}]
            self.assertEqual(
                reloaded.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
                self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
            )

    # ---------------------------------------------------------------- ray transport

    def test_ray_actors_can_receive_the_tokenizer(self):
        """The failure that killed the first smoke run, both directions.

        Ray pickles this tokenizer BY REFERENCE (transformers_modules.<hash>....PanguTokenizer);
        that dynamic package is only importable in a process that has loaded remote code, so an
        actor dies while unpickling its own constructor arguments unless HF_MODULES_CACHE is on
        PYTHONPATH - which is exactly what the openPangu scripts export.
        """
        import subprocess
        import sys

        from transformers.utils import HF_MODULES_CACHE

        program = (
            "import ray\n"
            "from transformers import AutoTokenizer\n"
            "@ray.remote(num_cpus=1)\n"
            "class C:\n"
            "    def __init__(self, tok): self.n = type(tok).__name__\n"
            "    def n_(self): return self.n\n"
            "ray.init(address='local', num_cpus=2, include_dashboard=False, log_to_driver=False)\n"
            f"tok = AutoTokenizer.from_pretrained({MODEL_PATH!r}, trust_remote_code=True)\n"
            "print('GOT', ray.get(C.remote(tok).n_.remote()))\n"
            "ray.shutdown()\n"
        )

        def run(pythonpath):
            env = dict(os.environ, PYTHONPATH=pythonpath)
            return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env, timeout=900)

        without = run("")
        self.assertNotIn("GOT PanguTokenizer", without.stdout, "expected the actor to die without the fix")
        self.assertIn("transformers_modules", without.stdout + without.stderr)

        with_fix = run(HF_MODULES_CACHE)
        self.assertIn("GOT PanguTokenizer", with_fix.stdout, with_fix.stderr[-400:])

    # ---------------------------------------------------------------- the datasets

    def test_every_prompt_fits_under_max_prompt_length(self):
        """filter_overlong_prompts drops rows over the limit - silently, and a broken template looks
        identical to an oversized prompt."""
        import pandas as pd

        files = [
            f
            for f in (
                os.path.join(DATA_DIR, name)
                for name in ("dapo-math-17k.parquet", "aime-2024.parquet", "aime-2025.parquet")
            )
            if os.path.exists(f)
        ]
        if not files:
            self.skipTest(f"no parquets under {DATA_DIR} (set OPENPANGU_DATA_DIR)")
        for path in files:
            with self.subTest(dataset=os.path.basename(path)):
                df = pd.read_parquet(path).head(1000)
                lengths = [
                    len(self.tok.apply_chat_template(list(p), add_generation_prompt=True, tokenize=True))
                    for p in df["prompt"]
                ]
                self.assertLess(max(lengths), MAX_PROMPT_LENGTH, f"{path}: longest prompt {max(lengths)} tokens")


if __name__ == "__main__":
    unittest.main()
