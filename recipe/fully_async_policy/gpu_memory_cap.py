# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import os

from verl.utils.device import get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_gpu_memory_cap() -> float | None:
    cap_gb = os.environ.get("VERL_GPU_MEM_CAP_GB")
    if not cap_gb:
        return None
    cap_bytes = float(cap_gb) * (1024**3)
    if cap_bytes <= 0:
        raise ValueError(f"VERL_GPU_MEM_CAP_GB must be positive, got {cap_gb!r}")
    device = get_torch_device()
    if not device.is_available():
        return None
    total_bytes = device.get_device_properties(device.current_device()).total_memory
    if cap_bytes >= total_bytes:
        logger.warning(
            "VERL_GPU_MEM_CAP_GB=%s is at or above the device's %.1f GiB; leaving the allocator uncapped",
            cap_gb,
            total_bytes / 1024**3,
        )
        return None
    fraction = cap_bytes / total_bytes
    device.set_per_process_memory_fraction(fraction)
    logger.warning(
        "Capped the trainer allocator at %.1f GiB of %.1f GiB (fraction %.4f) via VERL_GPU_MEM_CAP_GB",
        cap_bytes / 1024**3,
        total_bytes / 1024**3,
        fraction,
    )
    return fraction
