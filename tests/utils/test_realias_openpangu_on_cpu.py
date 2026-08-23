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
"""scripts/realias_openpangu_to_llama.py: the config rewrite the openPangu arm depends on.

The rewrite is what lets transformers 4.57.6 and vLLM 0.11.0 load the checkpoint at all (the stock
remote code imports LossKwargs, removed in transformers >= 4.54). Getting it subtly wrong - dropping
rms_norm_eps, forgetting the attention bias - produces a model that loads and computes the wrong
thing, so every field that diverges from LlamaConfig's defaults is asserted here.
"""

import importlib.util
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "realias_openpangu_to_llama",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "realias_openpangu_to_llama.py"),
)
realias_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(realias_module)
realias = realias_module.realias


def pangu_config(**overrides):
    """The fields of the real openPangu-Embedded-7B config.json that the rewrite reads."""
    cfg = {
        "architectures": ["PanguEmbeddedForCausalLM"],
        "model_type": "PanguEmbedded",
        "auto_map": {
            "AutoConfig": "configuration_openpangu_dense.PanguEmbeddedConfig",
            "AutoModelForCausalLM": "modeling_openpangu_dense.PanguEmbeddedForCausalLM",
        },
        "bias": True,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "eos_token_id": 45892,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "max_position_embeddings": 32768,
        "num_attention_heads": 32,
        "num_hidden_layers": 34,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 16000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": 153376,
    }
    cfg.update(overrides)
    return cfg


class TestRealias(unittest.TestCase):
    def test_architecture_is_rewritten_to_llama(self):
        out = realias(pangu_config())
        self.assertEqual(out["architectures"], ["LlamaForCausalLM"])
        self.assertEqual(out["model_type"], "llama")

    def test_auto_map_is_dropped(self):
        """This is what stops transformers importing modeling_openpangu_dense.py."""
        self.assertNotIn("auto_map", realias(pangu_config()))

    def test_attention_bias_carries_pangu_s_single_bias_flag(self):
        """Pangu's `bias` drives q/k/v/o; Llama's attention_bias covers exactly those four."""
        self.assertIs(realias(pangu_config(bias=True))["attention_bias"], True)
        self.assertIs(realias(pangu_config(bias=False))["attention_bias"], False)

    def test_mlp_has_no_bias(self):
        """Pangu's MLP hard-codes bias=False, and Llama would otherwise default it separately."""
        self.assertIs(realias(pangu_config())["mlp_bias"], False)

    def test_bias_is_left_in_place_for_vllm(self):
        """vLLM reads `bias` directly as a fallback for attention_bias."""
        self.assertIs(realias(pangu_config())["bias"], True)

    def test_fields_diverging_from_llama_defaults_survive(self):
        out = realias(pangu_config())
        for key in ("rms_norm_eps", "pad_token_id", "eos_token_id", "vocab_size", "rope_theta"):
            self.assertEqual(out[key], pangu_config()[key], f"{key} must survive the rewrite")

    def test_missing_llama_divergent_key_is_refused_not_guessed(self):
        for key in ("rms_norm_eps", "pad_token_id", "attention_dropout"):
            with self.subTest(missing=key):
                cfg = pangu_config()
                del cfg[key]
                with self.assertRaises(SystemExit):
                    realias(cfg)

    def test_input_is_not_mutated(self):
        cfg = pangu_config()
        realias(cfg)
        self.assertEqual(cfg["model_type"], "PanguEmbedded")
        self.assertIn("auto_map", cfg)

    def test_idempotent(self):
        once = realias(pangu_config())
        self.assertEqual(realias(once), once)

    def test_head_dim_is_left_implicit(self):
        """hidden_size // num_attention_heads is 128 under both configs; setting it would be noise."""
        self.assertNotIn("head_dim", realias(pangu_config()))

    def test_main_rewrites_a_checkpoint_directory_and_backs_it_up(self):
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(pangu_config(), f)
            with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
                json.dump({"auto_map": {"AutoTokenizer": ["tokenization_openpangu.PanguTokenizer", None]}}, f)

            argv = sys.argv
            sys.argv = ["realias", "--src", tmpdir, "--out", tmpdir, "--no-download"]
            try:
                self.assertEqual(realias_module.main(), 0)
            finally:
                sys.argv = argv

            with open(os.path.join(tmpdir, "config.json")) as f:
                rewritten = json.load(f)
            self.assertEqual(rewritten["model_type"], "llama")
            # the original is recoverable
            with open(os.path.join(tmpdir, "config.json.pangu.bak")) as f:
                self.assertEqual(json.load(f)["model_type"], "PanguEmbedded")
            # the TOKENIZER's auto_map must be untouched: it is still custom code
            with open(os.path.join(tmpdir, "tokenizer_config.json")) as f:
                self.assertIn("auto_map", json.load(f))


if __name__ == "__main__":
    unittest.main()
