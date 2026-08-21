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

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

__all__ = ["ess_from_log_weights", "compute_global_ess_from_log_weights", "compute_min_ess_lr_scale"]


def _clamped_exponents(seq_log_is: Sequence[float], rollout_is_threshold: float | None):
    s = torch.as_tensor([float(v) for v in seq_log_is], dtype=torch.float64)
    if rollout_is_threshold is not None and float(rollout_is_threshold) > 0:
        return s, torch.clamp(s, max=math.log(float(rollout_is_threshold)))
    return s, s


def _shifted_sums(log_w: torch.Tensor, shift: float) -> tuple[float, float]:
    if log_w.numel() == 0 or not math.isfinite(shift):
        return 0.0, 0.0
    e = torch.exp(log_w - shift)
    return float(e.sum()), float((e * e).sum())


def _is_corrupt(log_w: torch.Tensor) -> bool:
    return bool(log_w.numel()) and bool((torch.isnan(log_w) | (log_w == math.inf)).any())


def _ess_pair(sum_w: float, sum_w_sq: float, count: int, corrupt: bool) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    if corrupt:
        return math.nan, math.nan
    if sum_w_sq <= 0.0:
        return 0.0, 0.0
    ess = (sum_w * sum_w) / sum_w_sq
    return ess, ess / count


def ess_from_log_weights(
    seq_log_is: Sequence[float],
    rollout_is_threshold: float | None = None,
):
    s, s_clip = _clamped_exponents(seq_log_is, rollout_is_threshold)
    count = int(s.numel())
    corrupt = _is_corrupt(s)
    shift = s.max().item() if count else float("-inf")
    shift_clip = s_clip.max().item() if count else float("-inf")
    ess, ratio = _ess_pair(*_shifted_sums(s, shift), count, corrupt)
    ess_c, ratio_c = _ess_pair(*_shifted_sums(s_clip, shift_clip), count, corrupt)
    return ess, ratio, ess_c, ratio_c, count


def compute_global_ess_from_log_weights(
    seq_log_is: Sequence[float],
    rollout_is_threshold: float | None = None,
    group=None,
):
    s, s_clip = _clamped_exponents(seq_log_is, rollout_is_threshold)
    finite = torch.isfinite(s)
    finite_clip = torch.isfinite(s_clip)
    corrupt_local = _is_corrupt(s)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    neg_inf = float("-inf")

    def _finite_max(log_w: torch.Tensor, mask: torch.Tensor) -> float:
        kept = log_w[mask]
        return kept.max().item() if kept.numel() else neg_inf

    maxes = torch.tensor(
        [_finite_max(s, finite), _finite_max(s_clip, finite_clip)],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX, group=group)
    shift, shift_clip = maxes.tolist()

    w_sum, w_sq_sum = _shifted_sums(s[finite], shift)
    wc_sum, wc_sq_sum = _shifted_sums(s_clip[finite_clip], shift_clip)
    sums = torch.tensor(
        [w_sum, w_sq_sum, wc_sum, wc_sq_sum, float(s.numel()), 1.0 if corrupt_local else 0.0],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    g_sum, g_sq_sum, gc_sum, gc_sq_sum, count, corrupt = sums.tolist()
    count = int(count)
    corrupt = corrupt > 0.0

    ess, ess_ratio = _ess_pair(g_sum, g_sq_sum, count, corrupt)
    ess_clipped, ess_ratio_clipped = _ess_pair(gc_sum, gc_sq_sum, count, corrupt)
    return ess, ess_ratio, ess_clipped, ess_ratio_clipped, count


def compute_min_ess_lr_scale(ess: float, min_ess: float, lr_scale: float, count: int | None = None) -> float:
    ess = float(ess)
    if count is not None and int(count) <= 0:
        return 1.0
    if not math.isfinite(ess):
        return float(lr_scale)
    if ess <= 0:
        return float(lr_scale) if count is not None else 1.0
    if ess <= float(min_ess):
        return float(lr_scale)
    return 1.0
