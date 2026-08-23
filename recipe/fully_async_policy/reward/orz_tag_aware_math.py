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
"""Tag-aware math scorer for Open-Reasoner-Zero models.

Why this exists
---------------
ORZ's own chat template mandates ``<think> ... </think> <answer> ... </answer>``, and the model
obeys the template rather than the DAPO prompt's "put your answer on its own line after
'Answer:'" instruction. Measured on the 30 deduplicated AIME-2024 problems with
``Open-Reasoner-Zero/Open-Reasoner-Zero-7B`` (T=1.0, top_p=1.0, 8192 max tokens), on one GPU:

    prompt form            stock math_dapo   this scorer   'Answer:' line present
    DAPO wrapper (as-is)        0/30             5/30            5/30
    wrapper stripped            0/30             5/30            1/30

30/30 responses carried both ``<answer>`` and ``\\boxed{}``; 28-29/30 closed the tag on the same
line as the content. ``math_dapo`` captures ``(?i)Answer\\s*:\\s*([^\\n]+)`` to end of *line*, so a
same-line ``</answer>`` leaks into the prediction, and a bare ``\\boxed{}`` answer block yields
``[INVALID]``. Either way stock scoring reports 0 accuracy for a model whose published AIME-2024
pass@1 is ~15-18% -- the parser, not the maths, is the failure.

Extraction contract
-------------------
1. the **last** ``<answer> ... </answer>`` block wins (an unterminated ``<answer>`` from a truncated
   rollout falls back to everything after it);
2. inside that block: last ``\\boxed{}`` > last ``Answer:`` line > the block's own text;
3. with no ``<answer>`` tag at all, degrade to ``math_dapo``'s window (last 300 chars), so a
   plain-format response still scores.

Restricting to the answer block is also the anti-reward-hacking property: a ``\\boxed{}`` that
appears mid-reasoning cannot win once the model has emitted a real answer block.

Wired into an arm via::

    custom_reward_function.path=recipe/fully_async_policy/reward/orz_tag_aware_math.py
    custom_reward_function.name=compute_score
"""

import re
from typing import Optional

from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

__all__ = ["extract_answer", "compute_score"]

# Non-greedy so the *last* complete block is the one that wins.
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ANSWER_OPEN = "<answer>"
# Unlike math_dapo's pattern this stops at '<' as well as at a newline, so a same-line closing tag
# never leaks into the prediction.
_ANSWER_LINE = re.compile(r"(?i)answer\s*:\s*([^\n<]+)")

# math_dapo's window; kept identical so the no-tag fallback behaves exactly like the stock scorer.
_TAIL_CHARS = 300
# The longest answer in MATH-500 has 159 characters; anything longer is prose, not an answer, so a
# bare answer block that exceeds this is reported as invalid rather than normalized into noise.
_MAX_BARE_ANSWER_CHARS = 160

INVALID = "[INVALID]"


def _answer_region(solution_str: str) -> tuple[str, bool]:
    """Return ``(region, is_complete_block)``.

    ``is_complete_block`` says whether the region came from a properly closed ``<answer>`` block.
    Only such a region may be taken verbatim as the answer -- a truncated block or the plain-text
    tail must still yield a ``\\boxed{}`` or an ``Answer:`` line to count for anything.
    """
    blocks = _ANSWER_BLOCK.findall(solution_str)
    if blocks:
        return blocks[-1], True

    idx = solution_str.rfind(_ANSWER_OPEN)
    if idx >= 0:
        # Truncated rollout: the tag opened but the response was cut before it closed.
        return solution_str[idx + len(_ANSWER_OPEN) :], False

    return solution_str[-_TAIL_CHARS:], False


def _unbox(region: str) -> Optional[str]:
    boxed = last_boxed_only_string(region)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except AssertionError:
        # Malformed \boxed{ (unbalanced braces from a truncated rollout).
        return None


def extract_answer(solution_str: str) -> str:
    """Extract the normalized final answer, or ``'[INVALID]'`` when there is nothing to extract."""
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
    """Score one rollout. Same return contract as ``verl.utils.reward_score.math_dapo.compute_score``.

    Args:
        data_source: dataset tag, unused (kept for the reward-manager call signature).
        solution_str: the decoded response.
        ground_truth: the reference answer, plain or ``\\boxed{}``-wrapped.
        extra_info: per-sample extras, unused.

    Returns:
        ``{"score": 1.0 | -1.0, "acc": bool, "pred": str}``.
    """
    del data_source, extra_info, kwargs

    pred = extract_answer(solution_str)

    gt = ground_truth
    unboxed_gt = _unbox(gt) if isinstance(gt, str) else None
    gt = normalize_final_answer(unboxed_gt if unboxed_gt is not None else gt)

    acc = bool(pred == gt)
    return {"score": 1.0 if acc else -1.0, "acc": acc, "pred": pred}
