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

import multiprocessing
import os
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


def _extract_answer_raw(solution_str: str) -> Optional[str]:
    """The un-normalized answer candidate, or None when there is nothing to extract.

    Same precedence as ``extract_answer`` (boxed > 'Answer:' line > bare complete block); the raw
    form is what the ORZ equality tiers below compare (they do their own normalization).
    """
    region, is_complete_block = _answer_region(solution_str)

    unboxed = _unbox(region)
    if unboxed is not None:
        return unboxed

    lines = _ANSWER_LINE.findall(region)
    if lines:
        return lines[-1]

    stripped = region.strip()
    if is_complete_block and 0 < len(stripped) <= _MAX_BARE_ANSWER_CHARS:
        return stripped

    return None


def extract_answer(solution_str: str) -> str:
    """Extract the normalized final answer, or ``'[INVALID]'`` when there is nothing to extract."""
    raw = _extract_answer_raw(solution_str)
    return INVALID if raw is None else normalize_final_answer(raw)


# =====================================================================================
# ORZ equality tiers (vendored from Open-Reasoner-Zero, orz/ppo/tools/math_utils.py, MIT
# license). dapo-math-17k ground truths are 100% plain integers, so normalized-string
# equality was sufficient; the ORZ 72k collection has ~24% LaTeX-expression ground
# truths (\frac{280}{83}, 8\sqrt{3}, ...) where string equality produces false
# negatives ORZ's own training never had (280/83 vs \frac{280}{83}, 0.5 vs
# \frac{1}{2}). ORZ's is_equal is: _strip_string normalization + float compare
# ("is_equiv"), then sympy parse_latex symbolic/numeric equality. Ported verbatim
# except: the sympy tier runs in a forked child under a HARD wall-clock deadline
# (ORZ_MATH_SYMPY_TIMEOUT, default 1.0 s) and is killed on overrun — ORZ's
# executor+timeout equivalent, but leak-free (a killed child cannot keep burning
# CPU; the 2026-08-30 remote_mary stall came from an unbounded in-process tier) —
# plus a 128-char pre-filter, and parse_latex falls back to the lark backend where
# the antlr4 runtime is absent. ORZ_MATH_SYMPY_TIER=0 disables the tier.
# =====================================================================================


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        ia, ib = int(a), int(b)
        assert string == f"{ia}/{ib}"
        return "\\frac{" + str(ia) + "}{" + str(ib) + "}"
    except Exception:  # noqa: BLE001
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace(",", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _is_equiv_orz(str1: str, str2: str) -> bool:
    """ORZ's is_equiv: _strip_string both sides, float-compare, else string-compare."""
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        try:
            return float(ss1) == float(ss2)
        except Exception:  # noqa: BLE001
            return ss1 == ss2
    except Exception:  # noqa: BLE001
        return str1 == str2


_SYMPY_TIER_MAX_CHARS = 128
_sympy_backend: Optional[str] = None  # "antlr" | "lark", set by the probe
_sympy_tier_state: Optional[bool] = None  # None = not probed yet


def _sympy_tier_enabled() -> bool:
    global _sympy_backend, _sympy_tier_state
    if _sympy_tier_state is None:
        if os.environ.get("ORZ_MATH_SYMPY_TIER", "1") == "0":
            _sympy_tier_state = False
        elif not hasattr(multiprocessing, "get_context"):
            # The tier is time-bounded by a killable fork child; without
            # multiprocessing it would be silently unbounded (the exact failure
            # that stalled the 2026-08-30 remote_mary run) — refuse instead.
            print("[orz_tag_aware_math] sympy latex tier disabled (no multiprocessing)")
            _sympy_tier_state = False
        else:
            try:
                multiprocessing.get_context("fork")
                from sympy.parsing.latex import parse_latex

                try:
                    parse_latex("1")
                    _sympy_backend = "antlr"
                except ImportError:
                    # antlr4 runtime missing; try the lark backend.
                    parse_latex("1", backend="lark")
                    _sympy_backend = "lark"
                _sympy_tier_state = True
            except Exception as exc:  # noqa: BLE001
                print(f"[orz_tag_aware_math] sympy latex tier disabled ({type(exc).__name__}: {exc})")
                _sympy_tier_state = False
    return _sympy_tier_state


def _latex_equal_worker(str1: str, str2: str, backend: str, send_conn) -> None:
    """Child-process body for the sympy tier: parse both strings (raw pair, then
    stripped pair) and send the boolean verdict. Runs under a hard deadline
    enforced by the parent — anything slow here gets killed, not waited on."""
    try:
        from sympy.parsing.latex import parse_latex

        def _parse(s):
            return parse_latex(s) if backend == "antlr" else parse_latex(s, backend="lark")

        result = False
        for a, b in ((str1, str2), (_strip_string(str1), _strip_string(str2))):
            try:
                sym1, sym2 = _parse(a), _parse(b)
                if sym1 == sym2 or sym1.evalf() == sym2.evalf():
                    result = True
                    break
            except Exception:  # noqa: BLE001
                continue
        send_conn.send(result)
    except Exception:  # noqa: BLE001
        try:
            send_conn.send(False)
        except Exception:  # noqa: BLE001
            pass


def _is_latex_equal(str1: str, str2: str) -> bool:
    """ORZ's sympy tier: symbolic-or-numeric equality of the raw pair, retried on
    the stripped pair. Each comparison runs in a forked child with a HARD wall-clock
    deadline (ORZ_MATH_SYMPY_TIMEOUT, default 1.0 s — ORZ's own bound) and is
    killed on overrun: parse_latex/evalf can burn CPU for minutes on short inputs
    (power towers evalf, pathological grammars), and a thread-based timeout would
    leak that CPU; a killed fork child cannot. Length guard stays as a cheap
    pre-filter. Timeout/kill/crash/exception in the child all count as not-equal,
    matching ORZ's timeout semantics."""
    if not _sympy_tier_enabled():
        return False
    if len(str1) > _SYMPY_TIER_MAX_CHARS or len(str2) > _SYMPY_TIER_MAX_CHARS:
        return False
    timeout = float(os.environ.get("ORZ_MATH_SYMPY_TIMEOUT", "1.0"))
    try:
        ctx = multiprocessing.get_context("fork")
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_latex_equal_worker, args=(str1, str2, _sympy_backend, send_conn), daemon=True)
        proc.start()
        send_conn.close()
        result = False
        if recv_conn.poll(timeout):
            try:
                result = bool(recv_conn.recv())
            except (EOFError, OSError):
                result = False
        recv_conn.close()
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=1.0)
        return result
    except Exception:  # noqa: BLE001
        return False


def compute_score(
    data_source: Optional[str] = None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Score one rollout. Same return contract as ``verl.utils.reward_score.math_dapo.compute_score``.

    Equality is tiered (first hit wins):

    1. math_dapo normalized-string equality (the original behavior -- integer ground
       truths from dapo-math-17k/AceReason short-circuit here);
    2. ORZ ``is_equiv``: ``_strip_string`` normalization + float compare (handles
       280/83 vs \\frac{280}{83}, 0.5 vs \\frac{1}{2}, units, percents);
    3. ORZ sympy tier: ``parse_latex`` symbolic/numeric equality, run in a forked
       child under a hard wall-clock deadline (ORZ_MATH_SYMPY_TIMEOUT, default
       1.0 s) and killed on overrun; length-guarded; ORZ_MATH_SYMPY_TIER=0
       disables the tier entirely.

    Args:
        data_source: dataset tag, unused (kept for the reward-manager call signature).
        solution_str: the decoded response.
        ground_truth: the reference answer, plain or ``\\boxed{}``-wrapped.
        extra_info: per-sample extras, unused.

    Returns:
        ``{"score": 1.0 | -1.0, "acc": bool, "pred": str}``.
    """
    del data_source, extra_info, kwargs

    raw_pred = _extract_answer_raw(solution_str)
    pred = INVALID if raw_pred is None else normalize_final_answer(raw_pred)

    raw_gt = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    unboxed_gt = _unbox(raw_gt)
    if unboxed_gt is not None:
        raw_gt = unboxed_gt
    gt = normalize_final_answer(raw_gt)

    if raw_pred is None or not raw_gt.strip():
        # Nothing extracted, or an empty ground truth: never correct.
        return {"score": -1.0, "acc": False, "pred": pred}

    acc = bool(pred == gt) or _is_equiv_orz(raw_gt, raw_pred) or _is_latex_equal(raw_gt, raw_pred)
    return {"score": 1.0 if acc else -1.0, "acc": acc, "pred": pred}
