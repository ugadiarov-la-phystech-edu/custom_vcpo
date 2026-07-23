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

"""CPU tests for per-epoch bookkeeping of ppo_epochs replays.

``make_minibatch_iterator`` stamps ``epoch_idx`` / ``minibatch_idx_in_epoch`` onto each
minibatch's ``meta_info`` so that ``TrajRecord.epoch_idx`` is accurate when
``ppo_epochs > 1`` (it used to be hardcoded to 0).  The stamping relies on
``DataProto.make_iterator`` emitting epochs contiguously; that assumption is pinned here.
"""

import numpy as np
import torch

from recipe.fully_async_policy.staleness_utils import compute_staleness_statistics
from verl import DataProto


def _make_dataproto(batch_size: int, response_len: int = 4, prompt_len: int = 3) -> DataProto:
    """Minimal batch carrying every field compute_staleness_statistics requires."""
    total_len = prompt_len + response_len
    attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
    response_mask = torch.zeros(batch_size, total_len, dtype=torch.long)
    response_mask[:, prompt_len:] = 1

    return DataProto.from_dict(
        tensors={
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "input_ids": torch.arange(batch_size * total_len).reshape(batch_size, total_len),
        },
        non_tensors={
            "uid": np.array([f"group{i // 2}" for i in range(batch_size)], dtype=object),
            "traj_uid": np.array([f"traj{i}" for i in range(batch_size)], dtype=object),
            "reward_scalar": np.array([float(i) for i in range(batch_size)], dtype=object),
            "advantage_scalar": np.array([float(i) * 0.5 for i in range(batch_size)], dtype=object),
            "param_version_start": np.array([0] * batch_size, dtype=object),
            "param_version_end": np.array([0] * batch_size, dtype=object),
        },
    )


def _stamp_epochs(data: DataProto, mini_batch_size: int, epochs: int):
    """Mirror of the generator in MegatronPPOActor.make_minibatch_iterator."""
    n_minibatches_per_epoch = data.batch.batch_size[0] // mini_batch_size
    base = data.make_iterator(mini_batch_size=mini_batch_size, epochs=epochs, seed=0)
    out = []
    for i, mb in enumerate(base):
        out.append((i // n_minibatches_per_epoch, i % n_minibatches_per_epoch, mb))
    return out


class TestEpochStamping:
    def test_epochs_are_contiguous_single_minibatch(self):
        """ppo_epochs=2 over one minibatch per epoch -> epochs [0, 1] (the k=2 DAPO shape)."""
        data = _make_dataproto(batch_size=8)
        stamped = _stamp_epochs(data, mini_batch_size=8, epochs=2)
        assert [e for e, _, _ in stamped] == [0, 1]
        assert [m for _, m, _ in stamped] == [0, 0]

    def test_epochs_are_contiguous_multi_minibatch(self):
        """2 minibatches per epoch x 3 epochs -> [0,0,1,1,2,2] with within-epoch [0,1,0,1,0,1]."""
        data = _make_dataproto(batch_size=8)
        stamped = _stamp_epochs(data, mini_batch_size=4, epochs=3)
        assert [e for e, _, _ in stamped] == [0, 0, 1, 1, 2, 2]
        assert [m for _, m, _ in stamped] == [0, 1, 0, 1, 0, 1]

    def test_single_epoch_is_all_zero(self):
        """ppo_epochs=1 keeps every epoch_idx at 0, matching the previous hardcoded value."""
        data = _make_dataproto(batch_size=8)
        stamped = _stamp_epochs(data, mini_batch_size=4, epochs=1)
        assert [e for e, _, _ in stamped] == [0, 0]

    def test_every_minibatch_is_full_size(self):
        """make_iterator asserts exact divisibility, so the epoch formula never sees a short tail."""
        data = _make_dataproto(batch_size=8)
        stamped = _stamp_epochs(data, mini_batch_size=4, epochs=2)
        assert all(len(mb) == 4 for _, _, mb in stamped)


class TestComputeStalenessStatisticsEpochIdx:
    def test_epoch_idx_is_propagated_into_traj_records(self):
        data = _make_dataproto(batch_size=4)
        records, _ = compute_staleness_statistics(data, minibatch_idx=7, rollout_is_threshold=None, epoch_idx=1)
        assert len(records) == 4
        assert all(r.epoch_idx == 1 for r in records)
        assert all(r.minibatch_idx == 7 for r in records)

    def test_epoch_idx_defaults_to_zero(self):
        """Callers that omit epoch_idx keep the old behaviour."""
        data = _make_dataproto(batch_size=4)
        records, _ = compute_staleness_statistics(data, minibatch_idx=0, rollout_is_threshold=None)
        assert all(r.epoch_idx == 0 for r in records)

    def test_epoch_idx_is_coerced_to_int(self):
        data = _make_dataproto(batch_size=2)
        records, _ = compute_staleness_statistics(
            data, minibatch_idx=0, rollout_is_threshold=None, epoch_idx=np.int64(2)
        )
        assert all(isinstance(r.epoch_idx, int) and r.epoch_idx == 2 for r in records)

    def test_records_are_serializable_with_epoch_idx(self):
        """metrics['actor/local_traj_records'] goes through asdict(); epoch_idx must survive."""
        from dataclasses import asdict

        data = _make_dataproto(batch_size=2)
        records, _ = compute_staleness_statistics(data, minibatch_idx=3, rollout_is_threshold=None, epoch_idx=1)
        dumped = [asdict(r) for r in records]
        assert all(d["epoch_idx"] == 1 and d["minibatch_idx"] == 3 for d in dumped)
