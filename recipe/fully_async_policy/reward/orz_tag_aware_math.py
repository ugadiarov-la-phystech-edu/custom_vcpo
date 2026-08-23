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

import re
from typing import Optional

from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

__all__ = ["extract_answer", "compute_score"]

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ANSWER_OPEN = "<answer>"
_ANSWER_LINE = re.compile(r"(?i)answer\s*:\s*([^\n<]+)")

_TAIL_CHARS = 300
_MAX_BARE_ANSWER_CHARS = 160

INVALID = "[INVALID]"


def _answer_region(solution_str: str) -> tuple[str, bool]:
    blocks = _ANSWER_BLOCK.findall(solution_str)
    if blocks:
        return blocks[-1], True

    idx = solution_str.rfind(_ANSWER_OPEN)
    if idx >= 0:
        return solution_str[idx + len(_ANSWER_OPEN) :], False

    return solution_str[-_TAIL_CHARS:], False


def _unbox(region: str) -> Optional[str]:
    boxed = last_boxed_only_string(region)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except AssertionError:
        return None


def extract_answer(solution_str: str) -> str:
    region, is_complete_block = _answer_region(solution_str)

    unboxed = _unbox(region)
    if unboxed is not None:
        return normalize_final_answer(unboxed)

    lines = _ANSWER_LINE.findall(region)
    if lines:
        return normalize_final_answer(lines[-1])

    stripped = region.strip()
    if is_complete_block and 0 < len(stripped) <= _MAX_BARE_ANSWER_CHARS:
        return normalize_final_answer(stripped)

    return INVALID


def compute_score(
    data_source: Optional[str] = None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    del data_source, extra_info, kwargs

    pred = extract_answer(solution_str)

    gt = ground_truth
    unboxed_gt = _unbox(gt) if isinstance(gt, str) else None
    gt = normalize_final_answer(unboxed_gt if unboxed_gt is not None else gt)

    acc = bool(pred == gt)
    return {"score": 1.0 if acc else -1.0, "acc": acc, "pred": pred}
