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

import multiprocessing
import os
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


def _extract_answer_raw(solution_str: str) -> Optional[str]:
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
    raw = _extract_answer_raw(solution_str)
    return INVALID if raw is None else normalize_final_answer(raw)


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
_sympy_backend: Optional[str] = None
_sympy_tier_state: Optional[bool] = None


def _sympy_tier_enabled() -> bool:
    global _sympy_backend, _sympy_tier_state
    if _sympy_tier_state is None:
        if os.environ.get("ORZ_MATH_SYMPY_TIER", "1") == "0":
            _sympy_tier_state = False
        elif not hasattr(multiprocessing, "get_context"):
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
                    parse_latex("1", backend="lark")
                    _sympy_backend = "lark"
                _sympy_tier_state = True
            except Exception as exc:  # noqa: BLE001
                print(f"[orz_tag_aware_math] sympy latex tier disabled ({type(exc).__name__}: {exc})")
                _sympy_tier_state = False
    return _sympy_tier_state


def _latex_equal_worker(str1: str, str2: str, backend: str, send_conn) -> None:
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
    del data_source, extra_info, kwargs

    raw_pred = _extract_answer_raw(solution_str)
    pred = INVALID if raw_pred is None else normalize_final_answer(raw_pred)

    raw_gt = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    unboxed_gt = _unbox(raw_gt)
    if unboxed_gt is not None:
        raw_gt = unboxed_gt
    gt = normalize_final_answer(raw_gt)

    if raw_pred is None or not raw_gt.strip():
        return {"score": -1.0, "acc": False, "pred": pred}

    acc = bool(pred == gt) or _is_equiv_orz(raw_gt, raw_pred) or _is_latex_equal(raw_gt, raw_pred)
    return {"score": 1.0 if acc else -1.0, "acc": acc, "pred": pred}
