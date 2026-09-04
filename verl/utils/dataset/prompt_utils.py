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
"""Prompt-token helpers shared by the dataset and the agent loops.

verl renders chat templates to text and tokenizes with ``add_special_tokens=False`` (dataset), or
calls ``apply_chat_template(tokenize=True)`` (agent loops), which does the same. Both drop the BOS
token of tokenizers that rely on ``add_bos_token=True`` rather than emitting BOS from the template
itself. openPangu-Embedded-7B is such a model: its README tokenizes the rendered template with a
plain ``tokenizer(text)`` call, so every prompt the model was built for starts with ``<s>``.

``data.add_bos_token_to_prompt`` (default False) turns that on; :func:`maybe_prepend_bos` is the
single place that decides, so the dataset's ``input_ids`` / ``raw_prompt_ids`` and the agent loops'
``prompt_ids`` always agree.
"""

from __future__ import annotations

from collections.abc import Sequence


def tokenizer_wants_bos(tokenizer) -> bool:
    """True when this tokenizer would itself prepend BOS on ``tokenizer(text)``."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    return bos_id is not None and bool(getattr(tokenizer, "add_bos_token", False))


def maybe_prepend_bos(tokenizer, raw_prompt: str, prompt_ids: Sequence[int], enabled: bool) -> list[int]:
    """Return ``prompt_ids`` with the tokenizer's BOS id in front when that is what the model expects.

    The id is prepended only if ALL of the following hold, otherwise ``prompt_ids`` comes back
    unchanged (as a list):

    * ``enabled`` (``data.add_bos_token_to_prompt``);
    * the tokenizer has a ``bos_token_id`` and ``add_bos_token`` is true - the same condition under
      which a plain ``tokenizer(text)`` call adds BOS, so this reproduces the HF reference path;
    * the rendered template does not already start with ``bos_token`` (Llama-2/3, Mistral put BOS in
      the template text; adding another would double it);
    * ``prompt_ids`` does not already start with the BOS id.

    Tokenizers without a BOS token (Qwen2/Qwen3) are always a no-op.
    """
    ids = list(prompt_ids)
    if not enabled or not tokenizer_wants_bos(tokenizer):
        return ids
    bos_id = tokenizer.bos_token_id
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and raw_prompt.startswith(bos_token):
        return ids
    if ids and ids[0] == bos_id:
        return ids
    return [bos_id] + ids
