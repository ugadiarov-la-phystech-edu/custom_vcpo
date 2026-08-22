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

Pure functions shared by the Megatron per-traj path (megatron_actor) and the
FSDP path (dp_actor). No Megatron/FSDP imports here.
"""

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

__all__ = ["compute_global_ess_from_log_weights", "compute_min_ess_lr_scale"]


def _is_corrupt(log_w: torch.Tensor, clipped: bool = False) -> bool:
    """True when a log-weight cannot be interpreted as a weight at all.

    NaN means broken upstream log-probs; +inf cannot arise from a finite sum of
    log-probs, so unclipped it means the same. -inf is NOT corrupt — it is
    exactly a zero weight (a token the policy assigns probability 0), and the
    max-shift handles it without any special case.

    The CLIPPED variant is stricter about what counts as unknowable: a +inf
    exponent is clamped to log(threshold), a finite and perfectly correct
    clipped weight, so only NaN makes the clipped ESS unmeasurable. Flagging
    the two variants together would brake every mini-batch containing one
    -inf rollout log-prob whenever use_clipped=True, even though the clipped
    quantity the brake reads is well defined."""
    if not log_w.numel():
        return False
    if clipped:
        return bool(torch.isnan(log_w).any())
    return bool((torch.isnan(log_w) | (log_w == math.inf)).any())


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
    The raw-space pipeline this replaces (exp locally, all-reduce fp32 sums)
    censored the measurement at both ends: a dominant sequence with s ≳ 44
    overflowed Σw² in the fp32 all-reduce cast, reading ESS = 0.0 and scaling
    the LR to zero, while a batch with all s ≲ −87 flushed every weight to 0.

    Clipping commutes with exp — min(w, c) = exp(min(s, log c)) — so the
    clipped variant clamps in log space FIRST and is shifted by the max of the
    *clamped* exponents. Shifting it by the raw max instead would underflow
    the clipped sums whenever the dominant raw weight sits far above the clip.

    A threshold that is None or <= 0 disables clipping (clipped == unclipped),
    matching the collection-time gate of the raw-space path.

    When torch.distributed is initialized the maxima and shifted sums are
    all-reduced over ``group`` (None = the world group, the DP group of the
    FSDP trainer). Two tiny collectives: MAX on 2 doubles, SUM on 7 — the last
    two being PER-VARIANT corruption flags, so a log-weight that cannot be read
    as a weight on any rank makes that variant's ESS NaN globally instead of
    reading as a healthy batch, and the brake fails closed on it. The variants
    differ in what counts as unreadable: NaN for both, and +inf for the
    unclipped one only (clipping maps it to log(threshold), a correct finite
    weight). See _is_corrupt.

    Returns (ess, ess_ratio, ess_clipped, ess_ratio_clipped, global_count);
    all zeros when the global batch is empty, NaN ESS when it is corrupt.
    """
    s = torch.as_tensor([float(v) for v in seq_log_is], dtype=torch.float64)
    clipping_active = rollout_is_threshold is not None and float(rollout_is_threshold) > 0
    if clipping_active:
        s_clip = torch.clamp(s, max=math.log(float(rollout_is_threshold)))
    else:
        s_clip = s

    finite = torch.isfinite(s)
    finite_clip = torch.isfinite(s_clip)
    corrupt_local = _is_corrupt(s)
    # Only an ACTIVE clamp makes +inf readable; with clipping disabled the
    # "clipped" variant is the unclipped one and inherits its stricter rule.
    corrupt_clip_local = _is_corrupt(s_clip, clipped=True) if clipping_active else corrupt_local

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    neg_inf = float("-inf")

    def _finite_max(log_w: torch.Tensor, mask: torch.Tensor) -> float:
        # Non-finite entries are excluded from the shift: one rank's NaN must
        # not poison the MAX all-reduce (whose result is then implementation
        # defined). The corruption travels as a flag in the SUM below instead,
        # which turns the ESS into NaN for every rank at once.
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

    def _shifted_sums(log_w: torch.Tensor, m: float) -> tuple[float, float]:
        if log_w.numel() == 0 or not math.isfinite(m):
            return 0.0, 0.0
        e = torch.exp(log_w - m)
        return float(e.sum()), float((e * e).sum())

    w_sum, w_sq_sum = _shifted_sums(s[finite], shift)
    wc_sum, wc_sq_sum = _shifted_sums(s_clip[finite_clip], shift_clip)
    sums = torch.tensor(
        [
            w_sum,
            w_sq_sum,
            wc_sum,
            wc_sq_sum,
            float(s.numel()),
            1.0 if corrupt_local else 0.0,
            1.0 if corrupt_clip_local else 0.0,
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    g_sum, g_sq_sum, gc_sum, gc_sq_sum, count, corrupt, corrupt_clip = sums.tolist()
    count = int(count)
    corrupt = corrupt > 0.0
    corrupt_clip = corrupt_clip > 0.0

    def _ess(sum_w: float, sum_w_sq: float, corrupt: bool = corrupt) -> tuple[float, float]:
        # The rank holding the global max contributes exactly 1 to sum_w_sq,
        # so a positive count implies a positive denominator — no eps needed.
        if count <= 0:
            return 0.0, 0.0
        if corrupt:
            # The ESS of a batch carrying a non-finite log-weight is unknowable.
            # Reporting NaN (rather than a sum that silently reads as healthy)
            # is what lets the brake fail closed on it.
            return math.nan, math.nan
        if sum_w_sq <= 0.0:
            return 0.0, 0.0
        ess = (sum_w * sum_w) / sum_w_sq
        return ess, ess / count

    ess, ess_ratio = _ess(g_sum, g_sq_sum)
    ess_clipped, ess_ratio_clipped = _ess(gc_sum, gc_sq_sum, corrupt=corrupt_clip)
    return ess, ess_ratio, ess_clipped, ess_ratio_clipped, count


def compute_min_ess_lr_scale(ess: float, min_ess: float, lr_scale: float, count: int | None = None) -> float:
    """LR multiplier of the min-ESS brake.

    Brake when the mini-batch's global ESS carries at most ``min_ess``
    effective samples (inclusive: ess == min_ess brakes): return ``lr_scale``;
    above the threshold return 1.0 (full nominal lr). In ratio units this is
    ess_ratio <= min_ess / B — the raw-ESS form needs no batch size.

    No measured reference is involved: the threshold sits just above the
    structural floor ESS = 1 that the max-shifted computation guarantees for
    any non-empty batch (a single dominant sequence reads exactly 1), so
    degenerate mini-batches always brake at exactly lr_scale — never 0.

    ``count`` is the number of sequences the ESS was measured over, and it is
    what separates "no data" from "broken measurement":

    * count == 0 — the ESS was not measured (an empty global batch): no
      scaling, 1.0.
    * a non-finite ESS, or ESS <= 0 over a non-empty batch, cannot happen with
      the max-shifted computation on finite log-weights, so it means the
      measurement itself broke (a NaN log-prob, or a -inf rollout log-prob
      making the ratio +inf). The brake then FAILS CLOSED (lr_scale) rather
      than handing out full nominal lr on what is almost certainly the most
      degenerate mini-batch of the run.

    Callers that cannot supply ``count`` keep the legacy reading of ess == 0
    as "empty batch" (1.0); non-finite still fails closed."""
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
