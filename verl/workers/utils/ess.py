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

__all__ = ["compute_ess_lr_scale", "resolve_ess_base"]


def resolve_ess_base(config_base, override):
    """Resolve the ESS-scaling reference ratio.

    An explicit config value wins; base_ess_ratio=None (auto-calibration)
    resolves to the driver-provided override — the first update's measured
    on-policy ESS ratio, passed back via meta_info["ess_base_override"].
    Returns None while neither is available (scaling is then a no-op)."""
    return config_base if config_base is not None else override


def compute_ess_lr_scale(ess_ratio: float, base_ess_ratio: float, trigger_ratio: float | None = None) -> float:
    """LR multiplier of the ESS brake (before the sqrt/linear rule).

    Legacy (trigger_ratio=None): min(1, ess_ratio / base) — attenuate whenever
    the measured ESS falls below the reference. With ess_scaling.trigger_ratio
    set, scaling engages only when ess_ratio / base < trigger_ratio; at or
    above the threshold the mini-batch runs at full nominal lr (multiplier 1),
    so the multiplier jumps discontinuously at the threshold."""
    ratio = float(ess_ratio) / max(float(base_ess_ratio), 1e-8)
    if trigger_ratio is not None and ratio >= float(trigger_ratio):
        return 1.0
    return min(1.0, ratio)
