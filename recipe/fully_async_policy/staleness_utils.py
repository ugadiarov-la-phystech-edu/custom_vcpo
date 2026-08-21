from dataclasses import dataclass
from typing import Any, Optional

from verl import DataProto


@dataclass
class TrajRecord:
    """Record of a single trajectory: identity, training position, staleness
    versioning and the scalar trajectory statistics, all set in
    ``compute_staleness_statistics``.

    The IS/ESS fields this record used to carry are gone: the min-ESS brake
    measures the mini-batch ESS directly from per-sequence LOG IS ratios in
    ``dp_actor`` (see verl/workers/utils/ess.py), which needs no per-trajectory
    bookkeeping and no raw-space exp.
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


def compute_staleness_statistics(
    batch: DataProto,
    minibatch_idx: int,
    epoch_idx: int = 0,
) -> list[TrajRecord]:
    """
    Computes the local per-traj :class:`TrajRecord` list.

    Core fields populated here (see :class:`TrajRecord` for the full schema):
        uid, group_uid, epoch_idx, minibatch_idx,
        trainer_global_step, trainer_local_step,
        param_version_start, param_version_end, trainer_param_version,
        response_length, prompt_length, advantage_scalar, reward_scalar

    ``epoch_idx`` identifies the ppo_epochs pass this minibatch belongs to (stamped onto
    ``meta_info`` by ``make_minibatch_iterator``); ``minibatch_idx`` counts across epochs,
    while ``meta_info['minibatch_idx_in_epoch']`` is the within-epoch index.
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

        local_records.append(record)

    return local_records
