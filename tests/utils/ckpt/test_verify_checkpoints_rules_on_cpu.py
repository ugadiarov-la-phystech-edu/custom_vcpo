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
"""The two checkpoint-verification rules that must not be model-specific.

Both were written against Qwen3-8B and produced FALSE FAILURES on the first real openPangu
checkpoints (2026-08-23): a correct save was reported as missing its tokenizer and as missing 34
parameters. A verifier that cries wolf on good checkpoints is worse than none, so the rules are
pinned here for both tokenizer flavours and for the buffers transformers declines to save.
"""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "verify_checkpoints",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "recipe/fully_async_policy/shell/vcpo/dapo/baseline/verify_checkpoints.py",
    ),
)
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


class TestTokenizerFilesPresent(unittest.TestCase):
    """Which files count as "the tokenizer is there" depends on the tokenizer, not on the model."""

    def test_fast_tokenizer_layout(self):
        ok, _ = verify.tokenizer_files_present({"tokenizer_config.json", "tokenizer.json", "vocab.json"})
        self.assertTrue(ok)

    def test_slow_sentencepiece_layout(self):
        """openPangu's PanguTokenizer: tokenizer.model, and NO tokenizer.json anywhere - the base
        checkpoint does not have one either, so requiring it fails a faithful copy."""
        ok, _ = verify.tokenizer_files_present(
            {"tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "tokenization_openpangu.py"}
        )
        self.assertTrue(ok)

    def test_config_without_any_vocabulary_is_incomplete(self):
        ok, missing = verify.tokenizer_files_present({"tokenizer_config.json"})
        self.assertFalse(ok)
        self.assertIn("tokenizer.model", missing)

    def test_vocabulary_without_the_config_is_incomplete(self):
        ok, missing = verify.tokenizer_files_present({"tokenizer.json"})
        self.assertFalse(ok)
        self.assertEqual(missing, "tokenizer_config.json")

    def test_no_tokenizer_at_all(self):
        ok, _ = verify.tokenizer_files_present({"config.json", "model.safetensors"})
        self.assertFalse(ok)


class TestNonPersistentBuffers(unittest.TestCase):
    """transformers recomputes these from the config and does not save them; a base checkpoint
    exported by an older version may still carry them (openPangu ships 34 rotary inv_freq)."""

    def test_rotary_inv_freq_is_ignorable(self):
        for layer in (0, 7, 33):
            with self.subTest(layer=layer):
                self.assertTrue(verify.is_non_persistent_buffer(f"model.layers.{layer}.self_attn.rotary_emb.inv_freq"))

    def test_other_known_buffers(self):
        self.assertTrue(verify.is_non_persistent_buffer("h.0.attn.masked_bias"))
        self.assertTrue(verify.is_non_persistent_buffer("h.0.attn.attention.bias"))

    def test_real_weights_are_never_ignorable(self):
        for key in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ):
            with self.subTest(key=key):
                self.assertFalse(verify.is_non_persistent_buffer(key))

    def test_a_genuinely_missing_weight_is_not_masked_by_the_rule(self):
        """The rule must not become a blanket excuse: only the buffer suffixes are skipped."""
        base = {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
            "lm_head.weight",
        }
        saved = {"model.layers.0.self_attn.q_proj.weight"}
        missing = sorted(k for k in base - saved if not verify.is_non_persistent_buffer(k))
        self.assertEqual(missing, ["lm_head.weight"])


if __name__ == "__main__":
    unittest.main()
