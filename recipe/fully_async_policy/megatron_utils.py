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
# limitations under the License.

import torch
from megatron.core.distributed import DistributedDataParallel as DDP


@torch.no_grad()
def copy_megatron_model_to_cpu(models):
    """
    Copy Megatron model parameters to CPU memory (non-destructive copy).
    Unlike offload_megatron_model_to_cpu which moves data, this function creates
    independent copies on CPU while keeping GPU data intact.

    Args:
        models: List of model chunks (DDP-wrapped or unwrapped)

    Returns:
        dict: CPU state containing copied parameters and buffers
    """
    cpu_state = {}

    for model_idx, model_chunk in enumerate(models):
        if isinstance(model_chunk, DDP):
            # Handle DDP-wrapped models
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            buffer_states = []

            for buffers in model_chunk_all_buffers:
                buffer_list = []
                for buffer in buffers:
                    buffer_state = {}

                    # Copy parameter data to CPU
                    if buffer.param_data.storage().size() > 0:
                        buffer_state["param_data"] = buffer.param_data.data.cpu().clone().pin_memory()

                    buffer_list.append(buffer_state)
                buffer_states.append(buffer_list)

            cpu_state[f"model_chunk_{model_idx}"] = {"buffer_states": buffer_states, "is_ddp": True}
        else:
            # Handle non-DDP models (ref module)
            model_state = {}
            for name, param in model_chunk.named_parameters():
                param_state = {"data": param.data.cpu().clone().pin_memory()}
                model_state[name] = param_state

            cpu_state[f"model_chunk_{model_idx}"] = {"model_state": model_state, "is_ddp": False}

    return cpu_state


@torch.no_grad()
def restore_megatron_model_from_cpu(models, cpu_state):
    """
    Restore Megatron model parameters from CPU memory back to GPU.

    Args:
        models: List of model chunks to restore to
        cpu_state: CPU state dict returned from copy_megatron_model_to_cpu
    """
    for model_idx, model_chunk in enumerate(models):
        chunk_key = f"model_chunk_{model_idx}"
        if chunk_key not in cpu_state:
            continue

        chunk_state = cpu_state[chunk_key]

        if chunk_state["is_ddp"] and isinstance(model_chunk, DDP):
            # Restore DDP buffers
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            buffer_states = chunk_state["buffer_states"]

            for buffers, buffer_list in zip(model_chunk_all_buffers, buffer_states, strict=False):
                for buffer, buffer_state in zip(buffers, buffer_list, strict=False):
                    # Restore parameter data
                    if "param_data" in buffer_state:
                        buffer.param_data.data.copy_(buffer_state["param_data"].to(buffer.param_data.device))

        elif not chunk_state["is_ddp"] and not isinstance(model_chunk, DDP):
            # Restore non-DDP models
            model_state = chunk_state["model_state"]
            for name, param in model_chunk.named_parameters():
                if name in model_state:
                    param_state = model_state[name]
                    param.data.copy_(param_state["data"].to(param.device))


def _bare_named_parameters(chunk, wrapper_types) -> dict:
    while isinstance(chunk, wrapper_types):
        chunk = chunk.module
    return dict(chunk.named_parameters())


@torch.no_grad()
def copy_actor_params_to_ref(actor_chunks, ref_chunks, wrapper_types=None) -> int:
    """Overwrite the reference model's parameters with the actor's current ones, in place
    (the KL-reference reset of async_training.kl_ref_reset_interval).

    Both arguments are lists of model chunks of the SAME parallel layout living in the same
    process; the actor's are DDP/Float16Module-wrapped, the reference's are not DDP-wrapped,
    so both are unwrapped to the bare model before the names are matched. The target may be
    offloaded (its ``param.data`` on the CPU): ``copy_`` moves the values across devices. The
    source must hold real storage, i.e. the actor's parameters must not be offloaded.
    Returns the number of parameter tensors copied."""
    if wrapper_types is None:
        from verl.utils.megatron_utils import ALL_MODULE_WRAPPER_CLASSNAMES as wrapper_types
    actor_chunks, ref_chunks = list(actor_chunks), list(ref_chunks)
    if len(actor_chunks) != len(ref_chunks):
        raise ValueError(f"actor has {len(actor_chunks)} model chunks, the reference {len(ref_chunks)}")
    copied = 0
    for idx, (actor_chunk, ref_chunk) in enumerate(zip(actor_chunks, ref_chunks, strict=True)):
        src = _bare_named_parameters(actor_chunk, wrapper_types)
        dst = _bare_named_parameters(ref_chunk, wrapper_types)
        if src.keys() != dst.keys():
            diff = sorted(src.keys() ^ dst.keys())
            raise ValueError(f"chunk {idx}: actor and reference parameter names differ, e.g. {diff[:4]}")
        for name, target in dst.items():
            source = src[name]
            if source.shape != target.shape:
                raise ValueError(
                    f"chunk {idx}: {name} is {tuple(source.shape)} in the actor, {tuple(target.shape)} in the reference"
                )
            if source.numel() > 0 and source.data.untyped_storage().size() == 0:
                raise RuntimeError(
                    f"chunk {idx}: actor parameter {name} has no storage (offloaded?); cannot reset the reference"
                )
            target.data.copy_(source.data)
            copied += 1
    return copied
