import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import torch
from megatron.core import parallel_state as mpu

from verl import DataProto
from verl.utils.torch_functional import allgather_dict_into_list
from verl.workers.utils.ess import ess_from_log_weights


@dataclass
class TrajRecord:
    """Record of a single trajectory, populated incrementally during training.

    Core fields are set in ``compute_staleness_statistics``.  Optional fields
    (IS weights, log-prob lists, grad norms, loss) are filled in later by the
    actor or trainer.
    """

    # --- identity ---
    uid: Any
    group_uid: Any

    # --- training position ---
    epoch_idx: int
    minibatch_idx: int
    trainer_global_step: int
    trainer_local_step: int

    # --- staleness / versioning ---
    param_version_start: Any
    param_version_end: Any
    trainer_param_version: Any

    # --- trajectory statistics ---
    response_length: int
    prompt_length: Optional[int]
    advantage_scalar: float
    reward_scalar: float

    # --- filled in after loss / grad computation ---
    traj_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    grad_norm_unscaled: Optional[float] = None

    # --- IS / staleness fields (set by compute_is_info) ---
    old_log_probs: Optional[list[float]] = None
    rollout_log_probs: Optional[list[float]] = None
    kl_rollout_old: Optional[float] = None  # KL(rollout||old), K3 f-divergence form
    rollout_is_geom_mean: Optional[float] = None
    rollout_seq_is: Optional[float] = None
    rollout_seq_is_clipped: Optional[float] = None
    # Masked sum of the token log-ratios, i.e. log(rollout_seq_is), kept
    # unexp'd. ESS is computed from this: exp() in fp32 flushes sums below
    # ~-87 to 0 and turns sums above ~88.7 into inf, which censored the brake
    # signal at both ends (see verl/workers/utils/ess.py).
    rollout_seq_log_is: Optional[float] = None


class TrajRecordList(list):
    """List of :class:`TrajRecord` objects with optional lookup by ``traj_uid``."""

    def __getitem__(self, key):
        if isinstance(key, int | slice):
            return super().__getitem__(key)
        key_str = str(key)
        for record in self:
            if str(record.uid) == key_str:
                return record
        raise KeyError(key)


def mean_nonzero(x: torch.Tensor, dim=None, keepdim: bool = False):
    """
    Mean over nonzero entries of `x`.
    - If dim is None: mean over all nonzero elements.
    - If dim is int/tuple: mean over nonzero elements along that dim.
    - If there are no nonzero elements in the reduction, returns 0 there.
    """
    mask = x != 0
    x_ = x if torch.is_floating_point(x) else x.float()

    sum_ = (x_ * mask).sum(dim=dim, keepdim=keepdim)
    cnt = mask.sum(dim=dim, keepdim=keepdim)

    return torch.where(cnt > 0, sum_ / cnt.clamp_min(1), torch.zeros_like(sum_))


def compute_global_ess_ratio(
    local_is_sum: float,
    local_is_sq_sum: float,
    local_minibatch_size: int,
    eps: float = 1e-8,
):
    """All-reduce IS sums and counts over the data-parallel group to compute global ESS."""
    if local_minibatch_size is None:
        local_minibatch_size = 0

    global_is_sum = float(local_is_sum or 0.0)
    global_is_sq_sum = float(local_is_sq_sum or 0.0)
    global_minibatch_size = int(local_minibatch_size or 0)

    if torch.distributed.is_initialized():
        try:
            dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        except Exception:
            dp_group = None

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor(
            [global_is_sum, global_is_sq_sum, float(global_minibatch_size)],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        global_is_sum, global_is_sq_sum, global_minibatch_size = tensor.tolist()
        global_minibatch_size = int(global_minibatch_size)

    if global_minibatch_size > 0:
        global_ess = (global_is_sum**2) / (global_is_sq_sum + eps)
        global_ess_ratio = global_ess / global_minibatch_size
    else:
        global_ess = 0.0
        global_ess_ratio = 0.0

    return global_ess, global_ess_ratio, global_is_sum, global_is_sq_sum, global_minibatch_size


def compute_staleness_statistics(
    batch: DataProto,
    minibatch_idx: int,
    rollout_is_threshold: float | None,
    use_old_log_probs: bool = False,
    epoch_idx: int = 0,
) -> tuple[list[TrajRecord], dict]:
    """
    Computes the local per-traj :class:`TrajRecord` list as well as ESS info.

    Core fields populated here (see :class:`TrajRecord` for the full schema):
        uid, group_uid, epoch_idx, minibatch_idx,
        trainer_global_step, trainer_local_step,
        param_version_start, param_version_end, trainer_param_version,
        response_length, prompt_length, advantage_scalar, reward_scalar

    The following fields are filled in later by the actor/trainer:
        grad_norm, grad_norm_unscaled, traj_loss

    ``epoch_idx`` identifies the ppo_epochs pass this minibatch belongs to (stamped onto
    ``meta_info`` by ``make_minibatch_iterator``); ``minibatch_idx`` counts across epochs,
    while ``meta_info['minibatch_idx_in_epoch']`` is the within-epoch index.

    When ``use_old_log_probs=True``, IS-related fields are also populated via
    :func:`compute_is_info`.
    """
    traj_uids = batch.non_tensor_batch["traj_uid"]
    meta_info = getattr(batch, "meta_info", {}) or {}
    non_tensor_batch = batch.non_tensor_batch

    trainer_global_step = int(meta_info.get("trainer_global_step", -1) or -1)
    trainer_local_step = int(meta_info.get("trainer_local_step", -1) or -1)
    trainer_param_version = meta_info.get("trainer_param_version", None)

    # Scalars are required in non_tensor_batch for per-trajectory baselining.
    reward_scalars = non_tensor_batch.get("reward_scalar")
    if reward_scalars is None:
        raise KeyError("Missing non_tensor key 'reward_scalar' required by compute_staleness_statistics.")
    advantage_scalars = non_tensor_batch.get("advantage_scalar")
    if advantage_scalars is None:
        raise KeyError("Missing non_tensor key 'advantage_scalar' required by compute_staleness_statistics.")

    param_version_start_all = non_tensor_batch.get("param_version_start")
    param_version_end_all = non_tensor_batch.get("param_version_end")
    if param_version_start_all is None or param_version_end_all is None:
        raise KeyError(
            "Missing param version metadata. Expected non_tensor keys 'param_version_start'/'param_version_end'."
        )

    local_records = TrajRecordList()

    for idx, traj_uid in enumerate(traj_uids):
        group_uid = batch.non_tensor_batch["uid"][idx]
        response_mask = batch.batch["response_mask"][idx]
        reward_scalar = float(reward_scalars[idx])
        adv_scalar = float(advantage_scalars[idx])

        param_version_start = param_version_start_all[idx]
        param_version_end = param_version_end_all[idx]

        attention_mask = batch.batch["attention_mask"][idx]
        response_len = int(response_mask.sum().item())
        prompt_len = None
        if attention_mask is not None:
            prompt_len = max(int(attention_mask.sum().item()) - response_len, 0)

        record = TrajRecord(
            uid=traj_uid,
            group_uid=group_uid,
            epoch_idx=int(epoch_idx),
            minibatch_idx=int(minibatch_idx),
            trainer_global_step=trainer_global_step,
            trainer_local_step=trainer_local_step,
            param_version_start=param_version_start,
            param_version_end=param_version_end,
            trainer_param_version=trainer_param_version,
            response_length=response_len,
            prompt_length=prompt_len,
            advantage_scalar=adv_scalar,
            reward_scalar=reward_scalar,
        )

        if use_old_log_probs:
            old_log_prob = batch.batch["old_log_probs"][idx]
            rollout_log_prob = batch.batch["rollout_log_probs"][idx]

            record = compute_is_info(
                record,
                rollout_log_prob,
                old_log_prob,
                response_mask,
                rollout_is_threshold,
            )

        local_records.append(record)

    if not use_old_log_probs:
        return local_records, {}

    staleness_info = compute_ess_info(list(local_records), rollout_is_threshold)

    return local_records, staleness_info


def compute_is_info(
    record: TrajRecord,
    rollout_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_threshold: float | None,
) -> TrajRecord:
    """
    Update record with IS / KL statistics:
        old_log_probs
        rollout_log_probs
        kl_rollout_old  (K3 f-divergence form)
        rollout_is_geom_mean
        rollout_seq_log_is
        rollout_seq_is
        rollout_seq_is_clipped
    """
    mask_device = old_log_prob.device
    response_mask = response_mask.to(device=mask_device)
    mask_float = response_mask.to(dtype=old_log_prob.dtype)
    token_count = mask_float.sum()

    # Add per-token logprobs to the record so the trainer can plot token-level scatters.
    # Store *all* tokens where response_mask == 1 (can be large).
    with torch.no_grad():
        mask = response_mask.reshape(-1).bool()
        old_tokens = old_log_prob.reshape(-1)[mask]
        record.old_log_probs = old_tokens.detach().float().cpu().tolist()
        if rollout_log_prob is not None:
            rollout_tokens = rollout_log_prob.reshape(-1)[mask]
            record.rollout_log_probs = rollout_tokens.detach().float().cpu().tolist()

    # NOTE: The expressions below use the K3 f-divergence form:
    #   K3(P||Q) = E_P[exp(log(P/Q)) - log(P/Q) - 1]
    if rollout_log_prob is not None:
        # KL(rollout_log_prob || old_log_prob)
        with torch.no_grad():
            log_ratio = old_log_prob - rollout_log_prob
            k3_matrix = torch.exp(log_ratio) - log_ratio - 1
            k3_value = (k3_matrix * mask_float).sum() / (token_count + 1e-8)

            record.kl_rollout_old = float(k3_value.detach().item())

            # Geometric Mean of IS ratios
            geom_mean = torch.exp(((log_ratio * mask_float).sum()) / (token_count + 1e-8))
            record.rollout_is_geom_mean = float(geom_mean.detach().item())

            # Log-space first: this is what the ESS reduction consumes, and
            # unlike its exp() it is exact over the whole range of drifts.
            seq_log_is = (log_ratio.float() * mask_float.float()).sum()
            record.rollout_seq_log_is = float(seq_log_is.detach().item())

            seq_is = torch.exp((log_ratio * mask_float).sum())
            seq_is_value = float(seq_is.detach().item())
            record.rollout_seq_is = seq_is_value
            if rollout_is_threshold is not None and rollout_is_threshold > 0:
                record.rollout_seq_is_clipped = min(seq_is_value, float(rollout_is_threshold))
    return record


def compute_ess_info(local_records_list: list[TrajRecord], rollout_is_threshold: float | None, eps: float = 1e-8):
    """
    ESS Calculations:
        ess = (sum w_i)^2 / (sum w_i^2)
        ess_ratio = ess / minibatch_size

        ess_clipped = ess with clipped w_i
        ess_ratio_clipped = ess_clipped / minibatch_size

    Evaluated on the per-sequence LOG weights (``rollout_seq_log_is``) via the
    max-shifted computation in :func:`~verl.workers.utils.ess.ess_from_log_weights`,
    the same arithmetic the packed (dynamic-bsz) path uses — so both per-traj
    paths brake on the same number. Computing it from ``rollout_seq_is``
    instead censored the result at both ends of the fp32 range (a batch of
    weights below exp(-87) read ESS = 0, one above exp(88.7) read NaN) and the
    brake then ran those steps unbraked.

    ``eps`` is accepted for backwards compatibility and no longer used: the
    max-shift makes the denominator positive whenever the batch is non-empty.
    """
    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    is_leader = mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0

    # allgather_dict_into_list requires plain dicts; serialize TrajRecord objects first.
    local_dicts: list[dict] = [asdict(r) for r in local_records_list] if is_leader else []
    global_records: list[dict] = allgather_dict_into_list(local_dicts, group=dp_group)

    seq_log_is: list[float] = []
    for rec in global_records:
        log_is = rec.get("rollout_seq_log_is")
        if log_is is None:
            # Records written before rollout_seq_log_is existed: recover the
            # exponent where the stored weight still carries information.
            # A censored 0/inf/NaN cannot be recovered and is skipped, exactly
            # as a record with no IS fields at all is.
            seq_is = rec.get("rollout_seq_is")
            if seq_is is None or not math.isfinite(float(seq_is)) or float(seq_is) <= 0.0:
                continue
            log_is = math.log(float(seq_is))
        seq_log_is.append(float(log_is))

    ESS_unclipped, ess_ratio_unclipped, ESS, ess_ratio, count = ess_from_log_weights(seq_log_is, rollout_is_threshold)

    staleness_info = {
        "ess": ESS_unclipped,
        "ess_ratio": ess_ratio_unclipped,
        "ess_clipped": ESS,
        "ess_ratio_clipped": ess_ratio,
        # Number of sequences the ESS was measured over: the brake needs it to
        # tell "not measured" (0 -> no scaling) from "measurement broke".
        "count": count,
    }

    return staleness_info


def rearrange_minibatch(batch: DataProto) -> DataProto:
    """
    Rearrange minibatch to make trajectories of same group contiguous
    """
    group_uids = batch.non_tensor_batch.get("uid")
    if group_uids is None:
        return batch

    group_to_indices: dict[Any, list[int]] = {}
    group_order: list[Any] = []
    for idx, group_uid in enumerate(list(group_uids)):
        if group_uid not in group_to_indices:
            group_to_indices[group_uid] = []
            group_order.append(group_uid)
        group_to_indices[group_uid].append(idx)

    new_indices = [idx for group_uid in group_order for idx in group_to_indices[group_uid]]
    if new_indices == list(range(len(group_uids))):
        return batch

    return batch[new_indices]


def compute_grad_info(batch: DataProto, scope: Literal["group", "minibatch"] = "group", eps: float = 1e-8):
    n_resp_per_rollout = batch.meta_info["n_resp_per_rollout"]
    batch = rearrange_minibatch(batch)
    traj_uids = batch.non_tensor_batch["traj_uid"]
    raw_rewards = batch.non_tensor_batch["reward_scalar"]

    group_traj_counts = defaultdict(int)
    group_to_trajs = defaultdict(list)
    group_rewards = defaultdict(list)
    is_last_traj_in_scope = {}
    reward_by_traj_uid = {}
    reward_std_by_traj_uid = {}
    grpo_adv_by_traj_uid = {}

    for idx, traj_uid in enumerate(traj_uids):
        # Is last traj in group
        group_uid = batch.non_tensor_batch["uid"][idx]
        group_traj_counts[group_uid] += 1

        if scope == "group":
            is_last_traj_in_scope[traj_uid] = group_traj_counts[group_uid] == n_resp_per_rollout
        else:
            is_last_traj_in_scope[traj_uid] = idx == len(traj_uids) - 1

        group_to_trajs[group_uid].append(traj_uid)

        raw_reward = batch.non_tensor_batch["reward_scalar"][idx]
        reward = float(raw_reward)
        group_rewards[group_uid].append(reward)
        reward_by_traj_uid[traj_uid] = reward

    for traj_count in group_traj_counts.values():
        assert traj_count == n_resp_per_rollout

    for group_uid, traj_list in group_to_trajs.items():
        rewards = group_rewards.get(group_uid, [])
        if rewards:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            reward_std = float(torch.std(rewards_tensor, unbiased=False).item())
            reward_mean = float(torch.mean(rewards_tensor).item())
        else:
            reward_std = 0.0
            reward_mean = 0.0

        for traj_uid in traj_list:
            reward_std_by_traj_uid[traj_uid] = reward_std
            reward = reward_by_traj_uid[traj_uid]
            grpo_adv_by_traj_uid[traj_uid] = (reward - reward_mean) / (reward_std + eps)

    batch.meta_info["is_last_traj_in_scope"] = is_last_traj_in_scope
    if raw_rewards is not None:
        batch.meta_info["reward_std_by_traj_uid"] = reward_std_by_traj_uid
        batch.meta_info["grpo_adv_by_traj_uid"] = grpo_adv_by_traj_uid

    return batch


def compute_opob_baseline(
    local_traj_records: list[TrajRecord],
    group_uid: int,
    eps: float = 1e-8,
    use_is_weights: bool = True,
    use_clipped_is_ratios: bool = False,
    normalize_by_length: bool = False,
    agg_mode: str = "mean",
    scope: str = "group",
):
    """
    Optimal Off Policy Baseline:
        b = (sum_i W_i * R_i) / (sum_i W_i)
    where
        W_i = ||g_i||^2 * (ratio_i^2) * (1/L_i^2 if enabled)
    """
    weights = []
    values = []

    def _in_scope(rec: TrajRecord):
        if scope == "minibatch":
            return True
        return rec.group_uid == group_uid

    with torch.no_grad():
        for rec in local_traj_records:
            if _in_scope(rec):
                if scope == "minibatch":
                    rwd = rec.advantage_scalar
                else:
                    rwd = rec.reward_scalar

                if use_clipped_is_ratios:
                    seq_is_ratio = rec.rollout_seq_is_clipped
                else:
                    seq_is_ratio = rec.rollout_seq_is

                grad_norm = rec.grad_norm_unscaled
                weight = grad_norm**2
                if use_is_weights:
                    weight *= seq_is_ratio**2

                if normalize_by_length:
                    length = rec.response_length
                    weight = weight / (length**2)

                weights.append(weight)
                values.append(rwd)

    if agg_mode == "mean":
        baseline = get_weighted_mean(values, weights, eps=eps)
    elif agg_mode == "median":
        baseline = get_weighted_median(values, weights)
    elif agg_mode == "winsorized_mean":
        baseline = get_weighted_winsorized_mean(values, weights, eps=eps)
    else:
        raise NotImplementedError(f"Unsupported agg_mode: {agg_mode}")

    return baseline


def get_weighted_mean(values, weights, eps: float = 1e-8):
    """Compute a weighted mean from value/weight sequences."""
    if torch.is_tensor(values) or torch.is_tensor(weights):
        values_t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
        weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=torch.float32)
        if values_t.numel() == 0:
            return values_t.new_tensor(0.0)
        return (values_t * weights_t).sum() / (weights_t.sum() + eps)
    if not values:
        return 0.0
    numer = 0.0
    denom = 0.0
    for value, weight in zip(values, weights, strict=False):
        numer += float(value) * float(weight)
        denom += float(weight)
    return numer / (denom + eps)


def get_weighted_median(values, weights):
    """Compute a weighted median from value/weight sequences."""
    if torch.is_tensor(values) or torch.is_tensor(weights):
        values_t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
        weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=torch.float32)
        if values_t.numel() == 0:
            return values_t.new_tensor(0.0)
        total_w = weights_t.sum()
        if total_w == 0:
            return values_t.new_tensor(0.0)
        sort_idx = torch.argsort(values_t)
        sorted_vals = values_t[sort_idx]
        sorted_w = weights_t[sort_idx]
        cum_w = torch.cumsum(sorted_w, dim=0)
        cutoff = 0.5 * total_w
        median_idx = torch.searchsorted(cum_w, cutoff, right=False)
        median_idx = torch.clamp(median_idx, max=sorted_vals.numel() - 1)
        return sorted_vals[median_idx]

    if not values:
        return 0.0
    total_w = 0.0
    for weight in weights:
        total_w += float(weight)
    if total_w == 0.0:
        return 0.0
    sorted_pairs = sorted(zip(values, weights, strict=False), key=lambda pair: pair[0])
    cum_w = 0.0
    cutoff = 0.5 * total_w
    for value, weight in sorted_pairs:
        cum_w += float(weight)
        if cum_w >= cutoff:
            return float(value)
    return float(sorted_pairs[-1][0])


def get_weighted_winsorized_mean(values, weights, lower_q: float = 0.05, upper_q: float = 0.95, eps: float = 1e-8):
    """Compute a weighted winsorized mean by clipping values to weighted quantiles."""
    lower_q = min(max(lower_q, 0.0), 1.0)
    upper_q = min(max(upper_q, 0.0), 1.0)
    if upper_q < lower_q:
        lower_q, upper_q = upper_q, lower_q

    if torch.is_tensor(values) or torch.is_tensor(weights):
        values_t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
        weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=torch.float32)
        if values_t.numel() == 0:
            return values_t.new_tensor(0.0)
        total_w = weights_t.sum()
        if total_w == 0:
            return values_t.new_tensor(0.0)

        sort_idx = torch.argsort(values_t)
        sorted_vals = values_t[sort_idx]
        sorted_w = weights_t[sort_idx]
        cum_w = torch.cumsum(sorted_w, dim=0)

        lower_cut = lower_q * total_w
        upper_cut = upper_q * total_w
        lower_idx = torch.searchsorted(cum_w, lower_cut, right=False)
        upper_idx = torch.searchsorted(cum_w, upper_cut, right=False)
        lower_idx = torch.clamp(lower_idx, max=sorted_vals.numel() - 1)
        upper_idx = torch.clamp(upper_idx, max=sorted_vals.numel() - 1)

        lower_val = sorted_vals[lower_idx]
        upper_val = sorted_vals[upper_idx]
        clipped = torch.clamp(values_t, min=lower_val, max=upper_val)
        return (clipped * weights_t).sum() / (weights_t.sum() + eps)

    if not values:
        return 0.0
    total_w = 0.0
    for weight in weights:
        total_w += float(weight)
    if total_w == 0.0:
        return 0.0

    sorted_pairs = sorted(zip(values, weights, strict=False), key=lambda pair: pair[0])
    cum_w = 0.0
    lower_cut = lower_q * total_w
    upper_cut = upper_q * total_w
    lower_val = sorted_pairs[0][0]
    upper_val = sorted_pairs[-1][0]
    for value, weight in sorted_pairs:
        cum_w += float(weight)
        if cum_w >= lower_cut:
            lower_val = value
            break
    cum_w = 0.0
    for value, weight in sorted_pairs:
        cum_w += float(weight)
        if cum_w >= upper_cut:
            upper_val = value
            break

    numer = 0.0
    denom = 0.0
    for value, weight in zip(values, weights, strict=False):
        clipped = min(max(float(value), float(lower_val)), float(upper_val))
        numer += clipped * float(weight)
        denom += float(weight)
    return numer / (denom + eps)


class AnchorBlendController:
    """Driver-side controller for the soft TIS/clip loss blend
    (CLIP_IS_MIXING_ANCHORS_DISCUSSION.md; per-token loss
    ``L = (1-c2)*L_TIS + c2*L_clip``).

    Maps a smoothed drift signal (default: the mu-anchored clip piece's
    ``actor/pg_clipfrac`` -- dimensionless and backend-transferable; KL needs
    per-stack recalibration) to the clip-piece weight ``c2 in [c2_min, 1]``:

    * ``sig_ema <- beta*sig_ema + (1-beta)*sig``; a missing/NaN signal holds
      the previous EMA and c2 (a failed measurement must not move the blend).
      With ``ema_beta_up`` set, the EMA is attack/release-asymmetric: a signal
      above the current EMA is smoothed with the (smaller) ``ema_beta_up``, one
      below it with ``ema_beta`` — a divergence ramp reaches the thresholds
      within a couple of updates while the descent stays smoothed. (The
      symmetric beta=0.75 EMA was the binding lag in the update-437 blow-up of
      the 2026-08 anchor-blend run: raw clipfrac crossed sig_high two updates
      before the EMA-driven target did.)
    * target ``c2 = clamp((sig_ema - sig_low) / (sig_high - sig_low),
      c2_min, 1)`` -- proportional control on the tail fraction.
    * Asymmetric slew: increases apply instantly (safety); decreases are
      rate-limited by ``c2_down_rate`` per update (the hysteresis/dwell analog
      of a hard switch, without discrete state).

    AUTO threshold mode (default, the house ``base=auto`` pattern): with
    ``sig_low``/``sig_high`` both None the thresholds self-calibrate from the
    run's own healthy signal -- the median of ``calib_updates`` valid samples
    collected after skipping the first ``calib_skip`` updates (clipfrac starts
    near 0 while the buffer/staleness populate), floored at ``sig_ref_floor``,
    then frozen as ``low_mult*ref`` / ``high_mult*ref``. During calibration
    ``c2 = c2_min``. Setting both thresholds explicitly disables calibration;
    setting only one is a config error.
    """

    def __init__(
        self,
        sig_low: Optional[float] = None,
        sig_high: Optional[float] = None,
        low_mult: float = 5.0,
        high_mult: float = 25.0,
        calib_skip: int = 10,
        calib_updates: int = 20,
        sig_ref_floor: float = 1e-4,
        c2_min: float = 0.0,
        ema_beta: float = 0.75,
        ema_beta_up: Optional[float] = None,
        c2_down_rate: float = 0.05,
    ):
        manual = (sig_low is not None) or (sig_high is not None)
        if manual:
            assert sig_low is not None and sig_high is not None, (
                "adaptive_anchor: set BOTH sig_low and sig_high (manual thresholds) or NEITHER (auto calibration); "
                f"got sig_low={sig_low}, sig_high={sig_high}"
            )
            assert 0.0 <= float(sig_low) < float(sig_high), (
                f"adaptive_anchor: need 0 <= sig_low < sig_high, got {sig_low}, {sig_high}"
            )
        assert 0.0 <= float(c2_min) <= 1.0, f"adaptive_anchor: c2_min must be in [0, 1], got {c2_min}"
        assert 0.0 < float(ema_beta) < 1.0, f"adaptive_anchor: ema_beta must be in (0, 1), got {ema_beta}"
        if ema_beta_up is not None:
            assert 0.0 <= float(ema_beta_up) < 1.0, (
                f"adaptive_anchor: ema_beta_up must be in [0, 1) (0 = attack jumps to the raw signal), "
                f"got {ema_beta_up}"
            )
        assert float(c2_down_rate) > 0.0, f"adaptive_anchor: c2_down_rate must be > 0, got {c2_down_rate}"
        assert float(low_mult) < float(high_mult), (
            f"adaptive_anchor: need low_mult < high_mult, got {low_mult}, {high_mult}"
        )
        assert int(calib_updates) >= 1, f"adaptive_anchor: calib_updates must be >= 1, got {calib_updates}"

        self.auto = not manual
        self.sig_low = None if self.auto else float(sig_low)
        self.sig_high = None if self.auto else float(sig_high)
        self.low_mult = float(low_mult)
        self.high_mult = float(high_mult)
        self.calib_skip = int(calib_skip)
        self.calib_updates = int(calib_updates)
        self.sig_ref_floor = float(sig_ref_floor)
        self.c2_min = float(c2_min)
        self.ema_beta = float(ema_beta)
        self.ema_beta_up = None if ema_beta_up is None else float(ema_beta_up)
        self.c2_down_rate = float(c2_down_rate)

        self.sig_ref: Optional[float] = None
        self._calib_window: list[float] = []
        self._updates_seen = 0
        self._sig_ema: Optional[float] = None
        self._c2 = self.c2_min

    @property
    def calibrated(self) -> bool:
        """True once thresholds are available (immediately in manual mode)."""
        return self.sig_low is not None and self.sig_high is not None

    @staticmethod
    def grad_method_code(c2: float) -> float:
        """Numeric gradient-method indicator for dashboards: 0 = pure TIS, 1 = soft blend, 2 = pure PPO clip."""
        return 0.0 if c2 <= 0.0 else 2.0 if c2 >= 1.0 else 1.0

    @staticmethod
    def grad_method_name(c2: float) -> str:
        """Human-readable gradient method for a given c2 (log lines)."""
        return "TIS" if c2 <= 0.0 else "PPO-clip" if c2 >= 1.0 else "blend"

    def update(self, sig: Optional[float]) -> float:
        """Consume the just-finished update's signal; return c2 for the NEXT update."""
        self._updates_seen += 1
        valid = sig is not None and math.isfinite(float(sig))
        if valid:
            sig = float(sig)
            if self._sig_ema is None:
                self._sig_ema = sig
            else:
                beta = self.ema_beta_up if (self.ema_beta_up is not None and sig > self._sig_ema) else self.ema_beta
                self._sig_ema = beta * self._sig_ema + (1.0 - beta) * sig
            if self.auto and not self.calibrated and self._updates_seen > self.calib_skip:
                self._calib_window.append(sig)
                if len(self._calib_window) >= self.calib_updates:
                    ordered = sorted(self._calib_window)
                    mid = len(ordered) // 2
                    median = ordered[mid] if len(ordered) % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
                    self.sig_ref = max(median, self.sig_ref_floor)
                    self.sig_low = self.low_mult * self.sig_ref
                    self.sig_high = self.high_mult * self.sig_ref

        if not self.calibrated or self._sig_ema is None:
            self._c2 = self.c2_min
            return self._c2
        if not valid:
            return self._c2

        target = (self._sig_ema - self.sig_low) / (self.sig_high - self.sig_low)
        target = min(max(target, self.c2_min), 1.0)
        if target >= self._c2:
            self._c2 = target  # rise instantly (safety)
        else:
            self._c2 = max(target, self._c2 - self.c2_down_rate)  # rate-limited descent
        return self._c2

    def state(self) -> dict[str, float]:
        """Metrics snapshot (prefixed hybrid/ by the trainer)."""
        out = {"c2": self._c2, "calibrated": float(self.calibrated), "grad_method": self.grad_method_code(self._c2)}
        if self._sig_ema is not None:
            out["sig_ema"] = self._sig_ema
        if self.calibrated:
            out["sig_low"] = float(self.sig_low)
            out["sig_high"] = float(self.sig_high)
        if self.sig_ref is not None:
            out["sig_ref"] = self.sig_ref
        return out


def validate_adaptive_anchor_config(
    adaptive_cfg,
    policy_loss_cfg,
    rollout_corr_cfg,
    grad_baselining_enable: bool,
    skip_recompute_old_log_prob: bool,
) -> None:
    """Fail-fast startup validation for async_training.adaptive_anchor.

    Mirrors the actor's static anchor_mode='mu' assert battery on the driver so
    a mid-run c2 > 0 mini-batch cannot die hundreds of steps in. Pure function
    over plain dict-likes (``.get``) for CPU unit tests.
    """
    signal = adaptive_cfg.get("signal", "clipfrac")
    assert signal in ("clipfrac", "kl"), f"adaptive_anchor.signal must be 'clipfrac' or 'kl', got {signal!r}"
    assert policy_loss_cfg.get("anchor_mode", None) is None, (
        "adaptive_anchor owns the TIS/clip blend: set policy_loss.anchor_mode=null "
        f"(got {policy_loss_cfg.get('anchor_mode')!r}); static anchor_mode='mu' and the "
        "adaptive blend are mutually exclusive"
    )
    loss_mode = policy_loss_cfg.get("loss_mode", "vanilla")
    assert loss_mode == "vanilla", (
        f"adaptive_anchor requires policy_loss.loss_mode='vanilla' (got {loss_mode!r}): the clip piece "
        "reuses the stock vanilla surrogate"
    )
    assert skip_recompute_old_log_prob, (
        "adaptive_anchor requires async_training.skip_recompute_old_log_prob=True (the TIS piece anchors "
        "at log_prob.detach() and the clip piece at the cached rollout log-probs)"
    )
    assert not grad_baselining_enable, (
        "adaptive_anchor is incompatible with grad_baselining (OPOB): the clipped piece selects branches "
        "by the advantage sign and cannot be advantage-folded"
    )
    assert rollout_corr_cfg is not None and rollout_corr_cfg.get("rollout_is", None) == "token", (
        "adaptive_anchor requires algorithm.rollout_correction.rollout_is='token' (the TIS piece's weights "
        "and the drift metrics both come from the token-level deferred correction)"
    )
    assert rollout_corr_cfg.get("rollout_rs", None) is None, (
        "adaptive_anchor does not support rollout_rs rejection: rejected tokens punch holes in "
        "response_mask, silently dropping exactly the highest-divergence tokens the clip piece must see"
    )
