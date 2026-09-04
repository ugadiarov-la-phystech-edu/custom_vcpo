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

from __future__ import annotations

from collections.abc import Sequence


def tokenizer_wants_bos(tokenizer) -> bool:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    return bos_id is not None and bool(getattr(tokenizer, "add_bos_token", False))


def maybe_prepend_bos(tokenizer, raw_prompt: str, prompt_ids: Sequence[int], enabled: bool) -> list[int]:
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
