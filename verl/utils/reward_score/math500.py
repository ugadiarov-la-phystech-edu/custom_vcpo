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

import re

from . import math_dapo

try:
    from . import math_verify as _math_verify

    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MATH_VERIFY = False

_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def compute_score(solution_str: str, ground_truth: str) -> dict:
    result = math_dapo.compute_score(solution_str, ground_truth)
    if result["acc"] or not _HAS_MATH_VERIFY:
        return result
    matches = _ANSWER_PATTERN.findall(solution_str[-300:])
    if not matches:
        return result
    pred = matches[-1].strip()
    if _math_verify.compute_score("\\boxed{" + pred + "}", ground_truth):
        return {"score": 1.0, "acc": True, "pred": pred}
    return result
