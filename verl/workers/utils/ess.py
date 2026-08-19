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

"""Backend-neutral helpers for ESS-guided LR scaling (VCPO).

Pure functions with no Megatron/FSDP imports; the caller supplies the process
group to reduce over (e.g. Megatron's DP-with-CP group).
"""

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

__all__ = ["compute_global_ess_from_log_weights", "compute_min_ess_lr_scale"]


def compute_global_ess_from_log_weights(
    seq_log_is: Sequence[float],
    rollout_is_threshold: float | None = None,
    group=None,
):
    """Global ESS of sequence-level IS weights, computed from their LOG values.

    ESS = (Σw)²/Σw² is invariant to a common scale on the weights, so both the
    unclipped and clipped variants are evaluated on max-shifted exponents
    w̃ = exp(s − max s): the largest weight contributes exactly 1 and every sum
    lives in [1, N], independent of where the raw weights sit on the fp range.
    The raw-space pipelines this replaces censored the measurement at both
    ends: storing w = exp(s) in fp32 turns s ≳ 88.7 into inf (ESS then reads
    NaN/0 and the brake scales the LR to zero) and flushes every weight of an
    all-below-−87 batch to 0.

    Clipping commutes with exp — min(w, c) = exp(min(s, log c)) — so the
    clipped variant clamps in log space FIRST and is shifted by the max of the
    *clamped* exponents. Shifting it by the raw max instead would underflow
    the clipped sums whenever the dominant raw weight sits far above the clip.

    A threshold that is None or <= 0 disables clipping (clipped == unclipped),
    matching the collection-time gate of the raw-space path.

    When torch.distributed is initialized the maxima and shifted sums are
    all-reduced over ``group`` (None = the world group; Megatron callers pass
    the DP-with-CP group). Two tiny collectives: MAX on 2 doubles, SUM on 5.

    Returns (ess, ess_ratio, ess_clipped, ess_ratio_clipped, global_count);
    all zeros when the global batch is empty.
    """
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
        # The rank holding the global max contributes exactly 1 to sum_w_sq,
        # so a positive count implies a positive denominator — no eps needed.
        if count <= 0 or sum_w_sq <= 0.0:
            return 0.0, 0.0
        ess = (sum_w * sum_w) / sum_w_sq
        return ess, ess / count

    ess, ess_ratio = _ess(g_sum, g_sq_sum)
    ess_clipped, ess_ratio_clipped = _ess(gc_sum, gc_sq_sum)
    return ess, ess_ratio, ess_clipped, ess_ratio_clipped, count


def compute_min_ess_lr_scale(ess: float, min_ess: float, lr_scale: float) -> float:
    """LR multiplier of the min-ESS brake.

    Brake when the mini-batch's global ESS carries at most ``min_ess``
    effective samples (inclusive: ess == min_ess brakes): return ``lr_scale``;
    above the threshold return 1.0 (full nominal lr). In ratio units this is
    ess_ratio <= min_ess / B — the raw-ESS form needs no batch size.

    No measured reference is involved: the threshold sits just above the
    structural floor ESS = 1 that the max-shifted computation guarantees for
    any non-empty batch (a single dominant sequence reads exactly 1), so
    degenerate mini-batches always brake at exactly lr_scale — never 0.
    ess == 0 means an empty global batch: no scaling."""
    if ess > 0 and float(ess) <= float(min_ess):
        return float(lr_scale)
    return 1.0
