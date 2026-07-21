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
"""Helpers for computing/logging policy entropy during actor updates.

Entropy is calculated in the update forward pass either because it is part of
the loss (``entropy_coeff != 0``) or purely for monitoring
(``actor.calculate_entropy=True``, e.g. in the fully-async recipe's bypass mode
where the ``compute_old_log_prob`` forward that normally logs ``actor/entropy``
is skipped).
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss

__all__ = ["should_calculate_entropy", "log_entropy_and_apply_to_loss"]


def should_calculate_entropy(config) -> bool:
    """Whether the update forward pass needs to compute entropy.

    True when entropy participates in the loss (``entropy_coeff != 0``) or when
    entropy logging is explicitly requested via ``actor.calculate_entropy``.
    ``getattr`` keeps configs that predate the ``calculate_entropy`` field working.
    """
    return bool(getattr(config, "calculate_entropy", False)) or config.entropy_coeff != 0


def log_entropy_and_apply_to_loss(
    pg_loss: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    entropy_coeff: float,
    metrics: dict,
) -> torch.Tensor:
    """Record ``actor/entropy`` and fold the entropy bonus into the loss if enabled.

    The metric is always recorded (detached). The returned loss is
    ``pg_loss - entropy_coeff * entropy_loss`` when ``entropy_coeff != 0`` and
    ``pg_loss`` unchanged otherwise, so pure monitoring never perturbs the
    objective (the caller may then compute ``entropy`` under ``torch.no_grad()``).
    """
    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    metrics["actor/entropy"] = entropy_loss.detach().item()
    if entropy_coeff != 0:
        return pg_loss - entropy_coeff * entropy_loss
    return pg_loss
