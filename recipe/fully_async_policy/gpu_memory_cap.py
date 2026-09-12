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
"""VERL_GPU_MEM_CAP_GB: cap the TRAINER processes' PyTorch allocator to emulate a smaller card.

Shared by the FSDP2 and Megatron DetachActorWorker classes (recipe/fully_async_policy/
fsdp_workers.py, megatron_worker.py), which call apply_gpu_memory_cap() right after
set_expandable_segments(True) in __init__. Never called from the rollout workers: vLLM budgets
weights and KV cache against the device's TOTAL memory, so its side of an emulation is
rollout.gpu_memory_utilization (0.9 * 80 / 143.8 = 0.50 on an H200 for an 80 GiB H100 recipe).
Ported from replay_buffer_vcpo_ess_threshold_fsdp2_min-ess_cppo (commit 0360e6e).
"""

import logging
import os

from verl.utils.device import get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_gpu_memory_cap() -> float | None:
    """Cap this process's PyTorch allocator at VERL_GPU_MEM_CAP_GB gigabytes.

    Emulating a smaller card (e.g. running an 80 GB H100 recipe on a 143 GB
    H200) so the memory envelope of a run is representative. Returns the applied
    fraction, or None when the knob is unset / not applicable.

    Two reasons this lives here rather than in the launch environment:

    * torch has no env-var form of it — PYTORCH_CUDA_ALLOC_CONF rejects
      `per_process_memory_fraction` ("Unrecognized CachingAllocator option"), so
      it has to be an API call inside the process;
    * it must NOT reach the rollout engines. vLLM budgets weights, activations
      and KV cache as a fraction of the device's TOTAL memory, so a hidden
      allocator ceiling would leave it planning for memory it cannot get. The
      rollout side is capped by lowering rollout.gpu_memory_utilization instead
      (0.9 on an 80 GB card == 0.51 on a 143 GB one). DetachActorWorker is
      trainer-only, which is what makes this safe — the same argument as
      set_expandable_segments below.

    Caveat worth knowing when reading the resulting numbers: this bounds the
    caching allocator, not the hardware. The CUDA context, NCCL buffers and
    cuBLAS workspaces sit outside it (order 1-2 GB), and fragmentation against a
    soft cap differs from a real wall, so fitting under the cap is evidence the
    recipe fits the smaller card, not proof.
    """
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
