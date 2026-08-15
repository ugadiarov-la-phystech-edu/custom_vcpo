import torch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from verl import DataProto
from megatron.core import parallel_state as mpu
from verl.utils.torch_functional import allgather_dict_into_list


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


    # --- IS / staleness fields (set by compute_is_info) ---
    old_log_probs: Optional[List[float]] = None
    rollout_log_probs: Optional[List[float]] = None
    kl_rollout_old: Optional[float] = None          # KL(rollout||old), K3 f-divergence form
    rollout_is_geom_mean: Optional[float] = None
    rollout_seq_is: Optional[float] = None
    rollout_seq_is_clipped: Optional[float] = None


class TrajRecordList(list):
    """List of :class:`TrajRecord` objects with optional lookup by ``traj_uid``."""

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        key_str = str(key)
        for record in self:
            if str(record.uid) == key_str:
                return record
        raise KeyError(key)


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
        global_ess = (global_is_sum ** 2) / (global_is_sq_sum + eps)
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
) -> Tuple[list[TrajRecord], Dict]:
    """
    Computes the local per-traj :class:`TrajRecord` list as well as ESS info.

    Core fields populated here (see :class:`TrajRecord` for the full schema):
        uid, group_uid, epoch_idx, minibatch_idx,
        trainer_global_step, trainer_local_step,
        param_version_start, param_version_end, trainer_param_version,
        response_length, prompt_length, advantage_scalar, reward_scalar

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
            "Missing param version metadata. Expected non_tensor keys "
            "'param_version_start'/'param_version_end'."
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

            seq_is = torch.exp(((log_ratio * mask_float).sum()))
            seq_is_value = float(seq_is.detach().item())
            record.rollout_seq_is = seq_is_value
            if rollout_is_threshold is not None and rollout_is_threshold > 0:
                record.rollout_seq_is_clipped = min(seq_is_value, float(rollout_is_threshold))
    return record

def compute_ess_info(local_records_list: List[TrajRecord], rollout_is_threshold: float | None, eps: float = 1e-8):
    """
    ESS Calculations:
        ess = (sum w_i)^2 / (sum w_i^2) 
        ess_ratio = ess / minibatch_size
        
        ess_clipped = ess with clipped w_i
        ess_ratio_clipped = ess_clipped / minibatch_size

    """
    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    is_leader = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
    )

    # allgather_dict_into_list requires plain dicts; serialize TrajRecord objects first.
    local_dicts: list[dict] = [asdict(r) for r in local_records_list] if is_leader else []
    global_records: list[dict] = allgather_dict_into_list(local_dicts, group=dp_group)

    IS_sum = 0.0
    IS_sq_sum = 0.0
    IS_sum_unclipped = 0.0
    IS_sq_sum_unclipped = 0.0
    ESS = 0.0
    ESS_unclipped = 0.0
    minibatch_size = 0

    for rec in global_records:
        seq_is_unclipped = rec.get("rollout_seq_is")
        if seq_is_unclipped is None:
            continue
        seq_is_clipped = rec.get("rollout_seq_is_clipped")
        if seq_is_clipped is None:
            seq_is_clipped = float(seq_is_unclipped)
            if rollout_is_threshold is not None and rollout_is_threshold > 0:
                seq_is_clipped = min(seq_is_clipped, float(rollout_is_threshold))

        IS_sum_unclipped += float(seq_is_unclipped)
        IS_sq_sum_unclipped += float(seq_is_unclipped) ** 2
        IS_sum += float(seq_is_clipped)
        IS_sq_sum += float(seq_is_clipped) ** 2
        minibatch_size += 1

    if minibatch_size > 0:
        ESS = (IS_sum) ** 2 / (IS_sq_sum + eps)
        ess_ratio = ESS / minibatch_size
        ESS_unclipped = (IS_sum_unclipped) ** 2 / (IS_sq_sum_unclipped + eps)
        ess_ratio_unclipped = ESS_unclipped / minibatch_size
    else:
        ESS = 0.0
        ess_ratio = 0.0
        ESS_unclipped = 0.0
        ess_ratio_unclipped = 0.0

    staleness_info = {
        "ess": ESS_unclipped,
        "ess_ratio": ess_ratio_unclipped,
        "ess_clipped": ESS,
        "ess_ratio_clipped": ess_ratio,
    }

    return staleness_info

