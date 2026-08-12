"""Driver-side dynamic estimator of the ESS-scaling reference (base_ess_ratio).

Replaces the one-shot first-update auto-calibration with a running estimate
built from the *new cohort* of every replay minibatch: trajectories the
trainer consumes for the first time, i.e. the freshest data the async
pipeline can deliver.  The brake then measures "replay excess"
(minibatch ESS relative to fresh-data ESS) instead of comparing against a
frozen constant that drifts out of date as response length and policy
sharpness evolve.

Measurement happens actor-side in ``staleness_utils.compute_ess_info`` (raw
and top-1-winsorized cohort moments plus per-staleness-bucket robust
log-weight variances, piggybacked on the existing DP all-gather) and travels
to the driver inside the structured ``staleness/ess`` metric entries.  This
class pools the moments (accumulate-then-fold, so small cohorts never touch
the EMA directly), applies an optional length correction
``base = exp(-sigma^2 * L)``, clamps the result to a validated range, and
runs a weighted least-squares regression of per-token log-weight variance
against staleness whose s->0 intercept estimates the numerics-only
on-policy reference rho_on (logged as a diagnostic, never used for braking).

Pure python + stdlib on purpose: no torch/megatron imports, so the module is
trivially unit-testable on CPU and the estimator state pickles into
``replay_buffer.pt`` alongside the buffer.
"""

import math
from collections import deque


def _ess_ratio(w_sum: float, w_sq_sum: float, n: int, eps: float = 1e-8) -> float:
    """ESS/n = (sum w)^2 / (n * sum w^2) for pooled weight moments."""
    if n <= 0:
        return 0.0
    return float(w_sum) ** 2 / (n * float(w_sq_sum) + eps)


class EssBaseEstimator:
    """New-cohort EMA estimator of the ESS brake reference.

    Lifecycle per update (driven by ``FullyAsyncTrainer``):
      1. ``seed(base)`` — set-once warm start from the first-update capture
         (or the persisted base after resume); also the default clamp ceiling.
      2. ``observe_entries(entries)`` — ingest this update's measured payloads.
      3. ``current_base()`` — clamped operating reference for the *next*
         update's ``ess_base_override`` (one-update lag by design).
      4. ``diagnostics()`` — scalar metrics for logging.
    """

    def __init__(
        self,
        beta: float = 0.05,
        min_seqs: int = 64,
        clamp_min: float = 0.002,
        clamp_max: float | None = None,
        length_correction: bool = False,
        winsor_top1: bool = True,
        window: int = 100,
    ):
        self.beta = float(beta)
        self.min_seqs = int(min_seqs)
        self.clamp_min = float(clamp_min)
        self.clamp_max = None if clamp_max is None else float(clamp_max)
        self.length_correction = bool(length_correction)
        self.winsor_top1 = bool(winsor_top1)
        self.window = int(window)

        # Set-once warm start; doubles as the clamp ceiling when clamp_max is
        # null ("never brake harder than the first-update calibration would").
        self.seed_base: float | None = None
        # Pending pooled moments of fresh-cohort weights, folded into the EMA
        # only once >= min_seqs sequences accumulated (small-sample ESS is
        # biased optimistic, so per-cohort ratios never touch the EMA).
        self.pend_n = 0
        self.pend_w = 0.0
        self.pend_w_sq = 0.0
        self.pend_len = 0.0
        # EMA'd pooled moments; base = m1^2 / m2.
        self.m1: float | None = None
        self.m2: float | None = None
        self.len_ema: float | None = None
        # Sliding window of per-update bucket points for the rho_on regression.
        self.bucket_window: deque = deque(maxlen=self.window)
        # Mean response length of the last observed minibatch (drives the
        # length correction: the reference should be quoted at the length of
        # the data actually being braked).
        self.last_mb_len: float | None = None

        self._diag: dict = {}
        self._clamp_info: tuple | None = None
        self._len_corr = 1.0

    @classmethod
    def from_config(cls, cfg) -> "EssBaseEstimator":
        """Build from the async_training.ess_base_estimator mapping; null
        values fall back to defaults (clamp_max=null means seed ceiling)."""

        def _get(key, default):
            value = cfg.get(key, default) if cfg is not None else default
            return default if value is None else value

        return cls(
            beta=float(_get("beta", 0.05)),
            min_seqs=int(_get("min_seqs", 64)),
            clamp_min=float(_get("clamp_min", 0.002)),
            clamp_max=None if cfg is None or cfg.get("clamp_max", None) is None else float(cfg.get("clamp_max")),
            length_correction=bool(_get("length_correction", False)),
            winsor_top1=bool(_get("winsor_top1", True)),
            window=int(_get("window", 100)),
        )

    # ------------------------------------------------------------------ input

    def seed(self, base: float | None) -> None:
        """Set-once warm start (first-update capture / persisted base)."""
        if self.seed_base is None and base is not None and float(base) > 0.0:
            self.seed_base = float(base)

    def observe_entries(self, entries: list) -> None:
        """Extract the estimator payloads from the structured staleness/ess
        entries (identical across ranks after the metric flatten — take the
        first complete one) and ingest them."""
        cohort = None
        buckets = None
        mb_len = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if cohort is None and isinstance(entry.get("new_cohort"), dict):
                cohort = entry["new_cohort"]
            if buckets is None and isinstance(entry.get("staleness_buckets"), list):
                buckets = entry["staleness_buckets"]
            if mb_len is None and entry.get("minibatch_mean_len"):
                mb_len = float(entry["minibatch_mean_len"])
        self.observe(cohort, buckets, mb_len)

    def observe(self, cohort: dict | None, buckets: list | None, minibatch_mean_len: float | None) -> None:
        if minibatch_mean_len is not None and minibatch_mean_len > 0:
            self.last_mb_len = float(minibatch_mean_len)

        diag: dict = {}
        if cohort:
            n = int(cohort.get("n", 0))
            if n > 0:
                w_sum = float(cohort["w_sum"] if self.winsor_top1 else cohort["w_sum_raw"])
                w_sq_sum = float(cohort["w_sq_sum"] if self.winsor_top1 else cohort["w_sq_sum_raw"])
                diag["replay/ess_base_cohort_n"] = float(n)
                diag["replay/ess_base_cohort_ess"] = _ess_ratio(cohort["w_sum"], cohort["w_sq_sum"], n)
                diag["replay/ess_base_cohort_ess_raw"] = _ess_ratio(
                    cohort["w_sum_raw"], cohort["w_sq_sum_raw"], n
                )
                self.pend_n += n
                self.pend_w += w_sum
                self.pend_w_sq += w_sq_sum
                self.pend_len += float(cohort.get("len_sum", 0.0))
                if self.pend_n >= self.min_seqs:
                    self._fold()
        diag["replay/ess_base_pending_n"] = float(self.pend_n)

        if buckets:
            points = [
                (float(b["s"]), float(b["var_log_w"]) / max(float(b["mean_len"]), 1.0), float(b["n"]))
                for b in buckets
                if isinstance(b, dict) and b.get("mean_len")
            ]
            if points:
                self.bucket_window.append(points)
        diag.update(self._regress())
        self._diag = diag

    # ------------------------------------------------------------- estimation

    def _fold(self) -> None:
        mean1 = self.pend_w / self.pend_n
        mean2 = self.pend_w_sq / self.pend_n
        mean_len = self.pend_len / self.pend_n if self.pend_len > 0 else None
        if self.m1 is None:
            self.m1, self.m2 = mean1, mean2
            self.len_ema = mean_len
        else:
            b = self.beta
            self.m1 = (1 - b) * self.m1 + b * mean1
            self.m2 = (1 - b) * self.m2 + b * mean2
            if mean_len is not None:
                self.len_ema = mean_len if self.len_ema is None else (1 - b) * self.len_ema + b * mean_len
        self.pend_n = 0
        self.pend_w = 0.0
        self.pend_w_sq = 0.0
        self.pend_len = 0.0

    def _estimate(self) -> float | None:
        """Unclamped operating estimate; None until the first fold."""
        if self.m1 is None or self.m2 is None or self.m2 <= 0:
            return None
        ess = min(max((self.m1**2) / self.m2, 1e-8), 1.0)
        if self.length_correction and self.len_ema and self.last_mb_len:
            # base = exp(-sigma^2 * L_mb) with sigma^2 implied by the fresh
            # cohort's ESS at its own mean length (lognormal-weight model).
            sigma2 = -math.log(ess) / max(self.len_ema, 1.0)
            corrected = math.exp(-sigma2 * self.last_mb_len)
            self._len_corr = corrected / ess
            return corrected
        self._len_corr = 1.0
        return ess

    def current_base(self) -> float | None:
        raw = self._estimate()
        if raw is None:
            self._clamp_info = None
            return self.seed_base
        ceiling = self.clamp_max if self.clamp_max is not None else self.seed_base
        clamped_low = raw < self.clamp_min
        base = max(raw, self.clamp_min)
        clamped_high = False
        if ceiling is not None and base > ceiling:
            base = ceiling
            clamped_high = True
        self._clamp_info = (raw, clamped_low, clamped_high)
        return base

    def _regress(self) -> dict:
        """Weighted least squares of per-token log-weight variance vs
        staleness over the sliding window; the s->0 intercept estimates the
        numerics-only per-token variance sigma^2_num, hence rho_on."""
        points = [p for update_points in self.bucket_window for p in update_points]
        out: dict = {}
        distinct_s = {p[0] for p in points}
        if len(points) < 8 or len(distinct_s) < 3:
            return out
        s_w = s_x = s_y = s_xx = s_xy = 0.0
        for x, y, w in points:
            s_w += w
            s_x += w * x
            s_y += w * y
            s_xx += w * x * x
            s_xy += w * x * y
        denom = s_w * s_xx - s_x * s_x
        if denom <= 1e-12:
            return out
        slope = (s_w * s_xy - s_x * s_y) / denom
        intercept = (s_y - slope * s_x) / s_w
        y_bar = s_y / s_w
        ss_tot = sum(w * (y - y_bar) ** 2 for _, y, w in points)
        ss_res = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points)
        sigma2_num = max(intercept, 0.0)
        out["replay/ess_base_sigma2_num"] = sigma2_num
        out["replay/ess_base_delta2"] = slope
        out["replay/ess_base_reg_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        if self.last_mb_len:
            out["replay/ess_base_rho_on_est"] = math.exp(-sigma2_num * self.last_mb_len)
        return out

    # ------------------------------------------------------------ diagnostics

    def diagnostics(self) -> dict:
        d = dict(self._diag)
        if self._clamp_info is not None:
            raw, clamped_low, clamped_high = self._clamp_info
            d["replay/ess_base_unclamped"] = float(raw)
            d["replay/ess_base_clamped_low"] = float(clamped_low)
            d["replay/ess_base_clamped_high"] = float(clamped_high)
        d["replay/ess_base_from_estimator"] = float(self.m1 is not None)
        d["replay/ess_base_len_correction"] = float(self._len_corr)
        return d

    # ------------------------------------------------------------- checkpoint

    def state_dict(self) -> dict:
        return {
            "seed_base": self.seed_base,
            "pend_n": self.pend_n,
            "pend_w": self.pend_w,
            "pend_w_sq": self.pend_w_sq,
            "pend_len": self.pend_len,
            "m1": self.m1,
            "m2": self.m2,
            "len_ema": self.len_ema,
            "last_mb_len": self.last_mb_len,
            "bucket_window": [list(update_points) for update_points in self.bucket_window],
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed_base = state.get("seed_base", None)
        self.pend_n = int(state.get("pend_n", 0))
        self.pend_w = float(state.get("pend_w", 0.0))
        self.pend_w_sq = float(state.get("pend_w_sq", 0.0))
        self.pend_len = float(state.get("pend_len", 0.0))
        self.m1 = state.get("m1", None)
        self.m2 = state.get("m2", None)
        self.len_ema = state.get("len_ema", None)
        self.last_mb_len = state.get("last_mb_len", None)
        self.bucket_window = deque(
            [[tuple(p) for p in update_points] for update_points in state.get("bucket_window", [])],
            maxlen=self.window,
        )
