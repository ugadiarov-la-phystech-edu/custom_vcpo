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

__all__ = ["ess_from_log_weights", "compute_global_ess_from_log_weights", "compute_min_ess_lr_scale"]


def _clamped_exponents(seq_log_is: Sequence[float], rollout_is_threshold: float | None):
    """(unclipped, clipped) exponent tensors in float64.

    Clipping commutes with exp — min(w, c) = exp(min(s, log c)) — so the clip
    is applied in log space here and the clipped variant is later shifted by
    the max of the *clamped* exponents. A threshold that is None or <= 0
    disables clipping (clipped is the same tensor as unclipped)."""
    s = torch.as_tensor([float(v) for v in seq_log_is], dtype=torch.float64)
    if rollout_is_threshold is not None and float(rollout_is_threshold) > 0:
        return s, torch.clamp(s, max=math.log(float(rollout_is_threshold)))
    return s, s


def _shifted_sums(log_w: torch.Tensor, shift: float) -> tuple[float, float]:
    """(Σ w̃, Σ w̃²) for w̃ = exp(log_w − shift); zeros for an empty batch or a
    non-finite shift (an all -inf batch: every weight is exactly 0)."""
    if log_w.numel() == 0 or not math.isfinite(shift):
        return 0.0, 0.0
    e = torch.exp(log_w - shift)
    return float(e.sum()), float((e * e).sum())


def _is_corrupt(log_w: torch.Tensor) -> bool:
    """True when a log-weight cannot be interpreted as a weight at all.

    NaN means broken upstream log-probs; +inf cannot arise from a finite sum
    of log-probs, so it means the same. -inf is NOT corrupt — it is exactly a
    zero weight (a token the policy assigns probability 0), and the max-shift
    handles it without any special case."""
    return bool(log_w.numel()) and bool((torch.isnan(log_w) | (log_w == math.inf)).any())


def _ess_pair(sum_w: float, sum_w_sq: float, count: int, corrupt: bool) -> tuple[float, float]:
    """(ess, ess_ratio) from shifted sums.

    ``corrupt`` marks a batch that carried a non-finite log-weight: the ESS is
    unknowable, so it is reported as NaN and the brake decides what to do with
    it (fail closed) rather than a poisoned sum silently reading as healthy.
    The rank holding the global max contributes exactly 1 to sum_w_sq, so a
    positive count implies a positive denominator — no eps needed."""
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
    """Collective-free global ESS from per-sequence LOG IS weights.

    Same max-shifted arithmetic as :func:`compute_global_ess_from_log_weights`
    for callers that already hold the whole batch (the mbs=1 per-traj path
    gathers its records first). See that function for why log space is used.

    Returns (ess, ess_ratio, ess_clipped, ess_ratio_clipped, count); all zeros
    for an empty batch, NaN ESS when any log-weight is non-finite."""
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
    the DP-with-CP group). Two tiny collectives: MAX on 2 doubles, SUM on 6 —
    the 6th being a corruption flag, so a non-finite log-weight on any rank
    (broken upstream log-probs) makes the ESS NaN globally instead of reading
    as a healthy batch. The brake fails closed on NaN.

    Returns (ess, ess_ratio, ess_clipped, ess_ratio_clipped, global_count);
    all zeros when the global batch is empty, NaN ESS when it is corrupt.
    """
    s, s_clip = _clamped_exponents(seq_log_is, rollout_is_threshold)
    finite = torch.isfinite(s)
    finite_clip = torch.isfinite(s_clip)
    corrupt_local = _is_corrupt(s)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    neg_inf = float("-inf")

    def _finite_max(log_w: torch.Tensor, mask: torch.Tensor) -> float:
        # Non-finite entries are excluded from the shift: one rank's NaN must
        # not poison the MAX all-reduce. The corruption travels instead as a
        # flag in the SUM below, which turns the result into NaN for everyone.
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

    * count == 0 — the ESS was not measured (an empty global batch, or a path
      that never fills the IS fields): no scaling, 1.0.
    * a non-finite ESS, or ESS <= 0 over a non-empty batch, cannot happen with
      the max-shifted computation, so it means the measurement itself broke.
      The brake then FAILS CLOSED (lr_scale) rather than handing out full lr
      on what is almost certainly a degenerate mini-batch — the raw-space
      pipeline this replaced did the opposite and ran the collapse steps of
      the 2026-08 replay runs at full lr.

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
