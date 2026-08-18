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

__all__ = ["compute_ess_lr_scale", "compute_global_ess_from_log_weights", "resolve_ess_base"]


def compute_global_ess_from_log_weights(
    seq_log_is: Sequence[float],
    rollout_is_threshold: float | None = None,
    group=None,
):
    s = torch.as_tensor([float(v) for v in seq_log_is], dtype=torch.float64)
    if rollout_is_threshold is not None and float(rollout_is_threshold) > 0:
        s_clip = torch.clamp(s, max=math.log(float(rollout_is_threshold)))
    else:
        s_clip = s

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    neg_inf = float("-inf")
    maxes = torch.tensor(
        [
            s.max().item() if s.numel() else neg_inf,
            s_clip.max().item() if s_clip.numel() else neg_inf,
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX, group=group)
    shift, shift_clip = maxes.tolist()

    def _shifted_sums(log_w: torch.Tensor, m: float) -> tuple[float, float]:
        if log_w.numel() == 0 or not math.isfinite(m):
            return 0.0, 0.0
        e = torch.exp(log_w - m)
        return float(e.sum()), float((e * e).sum())

    w_sum, w_sq_sum = _shifted_sums(s, shift)
    wc_sum, wc_sq_sum = _shifted_sums(s_clip, shift_clip)
    sums = torch.tensor(
        [w_sum, w_sq_sum, wc_sum, wc_sq_sum, float(s.numel())],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    g_sum, g_sq_sum, gc_sum, gc_sq_sum, count = sums.tolist()
    count = int(count)

    def _ess(sum_w: float, sum_w_sq: float) -> tuple[float, float]:
        if count <= 0 or sum_w_sq <= 0.0:
            return 0.0, 0.0
        ess = (sum_w * sum_w) / sum_w_sq
        return ess, ess / count

    ess, ess_ratio = _ess(g_sum, g_sq_sum)
    ess_clipped, ess_ratio_clipped = _ess(gc_sum, gc_sq_sum)
    return ess, ess_ratio, ess_clipped, ess_ratio_clipped, count


def resolve_ess_base(config_base, override):
    return config_base if config_base is not None else override


def compute_ess_lr_scale(ess_ratio: float, base_ess_ratio: float, trigger_ratio: float | None = None) -> float:
    ratio = float(ess_ratio) / max(float(base_ess_ratio), 1e-8)
    if trigger_ratio is not None and ratio >= float(trigger_ratio):
        return 1.0
    return min(1.0, ratio)
