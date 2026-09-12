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
"""VERL_GPU_MEM_CAP_GB: cap a worker process's PyTorch allocator to emulate a smaller card.

Ported from replay_buffer_vcpo_ess_threshold_fsdp2_min-ess_cppo (commit 0360e6e), where it lived
in the recipe's DetachActorWorker. Here it is called from verl's ActorRolloutRefWorker.__init__
(megatron_workers.py, fsdp_workers.py) for actor-role processes only, which covers the fully-async
recipe's DetachActorWorker by inheritance and skips its DetachAsyncRolloutWorker.
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
      (the same absolute budget: fraction_H100 * 80 / device_GiB, e.g. 0.5 on an
      80 GB card == 0.28 on a 143 GB one). It is therefore called only from
      ACTOR-role worker processes (ActorRolloutRefWorker.__init__ under
      `if self._is_actor`): the fully-async trainer workers, and the synchronous
      hybrid workers, whose vLLM engines are separate server processes in
      rollout.mode=async. With rollout.mode=sync (in-process vLLM) the cap would
      also bound vLLM's torch-side activations — no arm here uses that mode.

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
