# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""``data.add_bos_token_to_prompt``: the BOS the openPangu tokenizer expects, and nothing else.

verl tokenizes rendered chat templates with ``add_special_tokens=False``, which drops the BOS of
tokenizers that rely on ``add_bos_token=True`` instead of emitting it from the template (openPangu).
The flag prepends it through one helper used by RLHFDataset and both single-turn agent loops, so
the dataset's lengths and the generated prompts agree. Pinned here with stub tokenizers (no network,
no checkpoint):

* the helper's decision table (flag, add_bos_token, bos_token_id, template-already-has-BOS);
* RLHFDataset end to end on a tiny parquet: input_ids / raw_prompt_ids / the overlong filter;
* the default (flag off) is byte-identical to the previous behaviour;
* both agent loops share the helper (source tripwire; the loops need a live server to run).

Run: pytest tests/utils/dataset/test_add_bos_token_to_prompt_on_cpu.py -q
"""

import os
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from verl.utils.dataset.prompt_utils import maybe_prepend_bos, tokenizer_wants_bos

BOS_ID = 1
EOS_ID = 2


class _StubTokenizer:
    """A whitespace tokenizer with a Pangu-shaped chat template and configurable BOS behaviour.

    ``apply_chat_template(tokenize=True)`` deliberately never adds BOS (like HF, which calls the
    tokenizer with add_special_tokens=False), while ``__call__`` / ``encode(add_special_tokens=True)``
    add it when ``add_bos_token`` is set - the real tokenizer's two paths.
    """

    def __init__(self, add_bos_token=True, bos_token="<s>", bos_in_template=False):
        self.add_bos_token = add_bos_token
        self.bos_token = bos_token
        self.bos_token_id = BOS_ID if bos_token is not None else None
        self.eos_token = "[eos]"
        self.eos_token_id = EOS_ID
        self.pad_token_id = 0
        self.chat_template = "stub"
        self._bos_in_template = bos_in_template
        self._vocab = {"<s>": BOS_ID, "[eos]": EOS_ID}

    # -- template ---------------------------------------------------------------------------
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        text = "".join(f"[{m['role']}] {m['content']} [eos] " for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        if self._bos_in_template:
            text = self.bos_token + " " + text
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    # -- tokenization -----------------------------------------------------------------------
    def _ids(self, text):
        ids = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = 10 + len(self._vocab)
            ids.append(self._vocab[tok])
        return ids

    def encode(self, text, add_special_tokens=True):
        ids = self._ids(text)
        if add_special_tokens and self.add_bos_token and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def __len__(self):
        return 1000


MESSAGES = [{"role": "user", "content": "What is 1+1?"}]


# ---------------------------------------------------------------------------- the helper


class TestMaybePrependBos(unittest.TestCase):
    def test_pangu_shape_gets_exactly_one_bos(self):
        tok = _StubTokenizer(add_bos_token=True)
        raw = tok.apply_chat_template(MESSAGES, tokenize=False)
        ids = tok.encode(raw, add_special_tokens=False)
        out = maybe_prepend_bos(tok, raw, ids, enabled=True)
        self.assertEqual(out, [BOS_ID] + ids)
        # and that is exactly the official recipe: apply_chat_template(tokenize=False) + tokenizer(text)
        self.assertEqual(out, tok(raw)["input_ids"][0].tolist())

    def test_flag_off_is_a_noop(self):
        tok = _StubTokenizer(add_bos_token=True)
        raw = tok.apply_chat_template(MESSAGES, tokenize=False)
        ids = tok.encode(raw, add_special_tokens=False)
        self.assertEqual(maybe_prepend_bos(tok, raw, ids, enabled=False), ids)

    def test_tokenizer_without_add_bos_token_is_a_noop(self):
        tok = _StubTokenizer(add_bos_token=False)
        raw = tok.apply_chat_template(MESSAGES, tokenize=False)
        ids = tok.encode(raw, add_special_tokens=False)
        self.assertEqual(maybe_prepend_bos(tok, raw, ids, enabled=True), ids)

    def test_tokenizer_without_bos_token_is_a_noop(self):
        """Qwen2/Qwen3: no BOS token at all, the flag must be inert."""
        tok = _StubTokenizer(add_bos_token=True, bos_token=None)
        raw = tok.apply_chat_template(MESSAGES, tokenize=False)
        ids = tok.encode(raw, add_special_tokens=False)
        self.assertFalse(tokenizer_wants_bos(tok))
        self.assertEqual(maybe_prepend_bos(tok, raw, ids, enabled=True), ids)

    def test_template_that_already_emits_bos_is_not_doubled(self):
        """Llama-2/3, Mistral: the template text starts with the BOS string."""
        tok = _StubTokenizer(add_bos_token=True, bos_in_template=True)
        raw = tok.apply_chat_template(MESSAGES, tokenize=False)
        ids = tok.encode(raw, add_special_tokens=False)
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(maybe_prepend_bos(tok, raw, ids, enabled=True), ids)

    def test_ids_that_already_start_with_bos_are_not_doubled(self):
        tok = _StubTokenizer(add_bos_token=True)
        raw = tok.apply_chat_template(MESSAGES, tokenize=False)
        ids = [BOS_ID] + tok.encode(raw, add_special_tokens=False)
        self.assertEqual(maybe_prepend_bos(tok, raw, ids, enabled=True), ids)

    def test_returns_a_fresh_list(self):
        tok = _StubTokenizer()
        ids = (5, 6, 7)
        out = maybe_prepend_bos(tok, "x", ids, enabled=False)
        self.assertIsInstance(out, list)
        self.assertEqual(out, [5, 6, 7])

    def test_empty_prompt_still_gets_bos(self):
        tok = _StubTokenizer()
        self.assertEqual(maybe_prepend_bos(tok, "", [], enabled=True), [BOS_ID])

    def test_tokenizer_wants_bos_table(self):
        self.assertTrue(tokenizer_wants_bos(_StubTokenizer(add_bos_token=True)))
        self.assertFalse(tokenizer_wants_bos(_StubTokenizer(add_bos_token=False)))
        self.assertFalse(tokenizer_wants_bos(_StubTokenizer(bos_token=None)))
        self.assertFalse(tokenizer_wants_bos(object()))


# ---------------------------------------------------------------------------- RLHFDataset


def _write_parquet(path, n_rows=4, long_row=False):
    import pandas as pd

    rows = []
    for i in range(n_rows):
        content = f"Problem number {i}: what is {i}+{i}?"
        rows.append(
            {
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": content}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(2 * i)},
                "extra_info": {"index": i},
            }
        )
    if long_row:
        rows.append(
            {
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": " ".join(f"w{j}" for j in range(40))}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "0"},
                "extra_info": {"index": n_rows},
            }
        )
    pd.DataFrame(rows).to_parquet(path)


def _dataset(parquet, tokenizer, **cfg):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 64,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "truncation": "left",
            "return_raw_chat": True,
            **cfg,
        }
    )
    return RLHFDataset(data_files=parquet, tokenizer=tokenizer, config=config)


class TestRLHFDatasetBos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.parquet = os.path.join(cls.tmp.name, "train.parquet")
        _write_parquet(cls.parquet)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _row(self, flag, tokenizer=None):
        ds = _dataset(self.parquet, tokenizer or _StubTokenizer(), add_bos_token_to_prompt=flag)
        return ds, ds[0]

    def _unpadded(self, row):
        mask = row["attention_mask"].bool()
        return row["input_ids"][mask].tolist()

    def test_flag_on_prepends_bos_to_both_id_views(self):
        ds, row = self._row(True)
        tok = ds.tokenizer
        expected = [BOS_ID] + tok.apply_chat_template(ds.dataframe[0]["prompt"], tokenize=True)
        self.assertEqual(row["raw_prompt_ids"], expected)
        self.assertEqual(self._unpadded(row), expected)

    def test_flag_off_is_the_previous_behaviour(self):
        ds, row = self._row(False)
        tok = ds.tokenizer
        expected = tok.apply_chat_template(ds.dataframe[0]["prompt"], tokenize=True)
        self.assertEqual(row["raw_prompt_ids"], expected)
        self.assertEqual(self._unpadded(row), expected)
        self.assertNotEqual(expected[0], BOS_ID)

    def test_default_is_off(self):
        ds = _dataset(self.parquet, _StubTokenizer())
        self.assertFalse(ds.add_bos_token_to_prompt)
        self.assertNotEqual(ds[0]["raw_prompt_ids"][0], BOS_ID)

    def test_position_ids_and_mask_cover_the_bos(self):
        _, row = self._row(True)
        n = int(row["attention_mask"].sum())
        self.assertEqual(len(row["raw_prompt_ids"]), n)
        self.assertEqual(row["position_ids"][row["attention_mask"].bool()].tolist(), list(range(n)))

    def test_flag_on_with_a_qwen_like_tokenizer_changes_nothing(self):
        ds, row = self._row(True, tokenizer=_StubTokenizer(bos_token=None))
        expected = ds.tokenizer.apply_chat_template(ds.dataframe[0]["prompt"], tokenize=True)
        self.assertEqual(row["raw_prompt_ids"], expected)

    def test_flag_on_with_bos_in_template_does_not_double(self):
        ds, row = self._row(True, tokenizer=_StubTokenizer(bos_in_template=True))
        ids = row["raw_prompt_ids"]
        self.assertEqual(ids[0], BOS_ID)
        self.assertNotEqual(ids[1], BOS_ID)

    def test_overlong_filter_counts_the_bos(self):
        """A prompt of exactly max_prompt_length tokens without BOS is one too long with it."""
        tok = _StubTokenizer()
        n_no_bos = len(tok.apply_chat_template(MESSAGES, tokenize=True))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "edge.parquet")
            import pandas as pd

            pd.DataFrame(
                [
                    {
                        "data_source": "math_dapo",
                        "prompt": MESSAGES,
                        "ability": "math",
                        "reward_model": {"style": "rule", "ground_truth": "2"},
                        "extra_info": {"index": 0},
                    }
                ]
            ).to_parquet(p)
            kept_off = len(_dataset(p, tok, add_bos_token_to_prompt=False, max_prompt_length=n_no_bos))
            kept_on = len(_dataset(p, tok, add_bos_token_to_prompt=True, max_prompt_length=n_no_bos))
            kept_on_roomy = len(_dataset(p, tok, add_bos_token_to_prompt=True, max_prompt_length=n_no_bos + 1))
        self.assertEqual((kept_off, kept_on, kept_on_roomy), (1, 0, 1))

    def test_left_truncation_keeps_the_limit_with_bos(self):
        tok = _StubTokenizer()
        ds = _dataset(
            self.parquet, tok, add_bos_token_to_prompt=True, filter_overlong_prompts=False, max_prompt_length=5
        )
        row = ds[0]
        self.assertEqual(len(row["raw_prompt_ids"]), 5)
        self.assertEqual(int(row["attention_mask"].sum()), 5)


# ---------------------------------------------------------------------------- agent loops


class TestAgentLoopsShareTheHelper(unittest.TestCase):
    """The loops need a running rollout server; pin the wiring on their source instead."""

    def _source(self, module_name):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(module_name))

    def test_both_single_turn_loops_read_the_flag_and_call_the_helper(self):
        for mod in (
            "verl.experimental.agent_loop.single_turn_agent_loop",
            "recipe.fully_async_policy.agent_loop.partial_single_turn_agent_loop",
        ):
            with self.subTest(module=mod):
                src = self._source(mod)
                self.assertIn('data.get("add_bos_token_to_prompt", False)', src)
                self.assertIn("maybe_prepend_bos(", src)
                self.assertIn("add_special_tokens=False", src)
                # the old apply_chat_template(tokenize=True) call skipped BOS and bypassed the helper
                self.assertNotIn("tokenize=True, **self.apply_chat_template_kwargs", src)

    def test_dataset_and_loops_default_to_off(self):
        src = self._source("verl.utils.dataset.rl_dataset")
        self.assertIn('config.get("add_bos_token_to_prompt", False)', src)


if __name__ == "__main__":
    unittest.main()
