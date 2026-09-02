from __future__ import annotations
import torch
import torch.distributed as dist
from typing import Iterable, Optional, Sequence, List

from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron_utils import get_model_config

from megatron.core import parallel_state as mpu, tensor_parallel
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_conditional_embedding_grads,
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_embedding_grads,
    _update_router_expert_bias,
    _get_main_grad_attr,
    get_attr_wrapped_model,
    _unshard_if_dtensor,
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
    _reshard_if_dtensor
)
from verl.utils.metric import reduce_metrics
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.transformer.module import param_is_not_shared
from contextlib import nullcontext

def _iter_grad_buffers(modules: Iterable[torch.nn.Module]) -> Iterable[torch.Tensor]:
    """
    Iterator over megatron grad buffers
    """
    for module in modules:
        buffers = []
        if hasattr(module, "buffers"):
            buffers.extend(module.buffers)
        if hasattr(module, "expert_parallel_buffers"):
            buffers.extend(module.expert_parallel_buffers)
        if not buffers and hasattr(module, "param_and_grad_buffer"):
            buffers.append(module.param_and_grad_buffer)
        for buffer in buffers:
            yield buffer.grad_data


def _opob_debug_enabled() -> bool:
    """VCPO_OPOB_DEBUG=1 turns on per-scope-close / per-step buffer-norm tracing in the OPOB path."""
    import os

    return os.environ.get("VCPO_OPOB_DEBUG", "0") not in ("", "0", "false", "False")


def grad_buffers_norm(buffers: Sequence[torch.Tensor]) -> float:
    """L2 norm over a list of grad-sized buffers without materializing fp32 copies
    (per-tensor foreach norms combined in fp32). Diagnostics only."""
    buffers = list(buffers)
    if not buffers:
        return 0.0
    with torch.no_grad():
        # dtype=float32 forces fp32 accumulation: a bf16 reduction over ~1e10
        # elements plateaus (ulp > increment) and returns a content-independent
        # constant (observed ~1.16e3 for the Qwen3-8B TP=2 grad buffer).
        norms = torch.stack([torch.linalg.vector_norm(b, dtype=torch.float32) for b in buffers])
        return float(torch.linalg.vector_norm(norms).item())


def top_param_slices(modules: Iterable[torch.nn.Module], buffers: Sequence[torch.Tensor], k: int = 5) -> list:
    """Diagnostics: the k parameter slices of ``buffers`` (laid out like the modules' grad
    buffers) with the largest L2 norm, as (name, norm, numel). Uses Megatron's
    ``_ParamAndGradBuffer.param_index_map`` for the offsets; returns [] when unavailable."""
    out = []
    buffers = list(buffers)
    idx = 0
    with torch.no_grad():
        for module in modules:
            names = {p: n for n, p in module.named_parameters()}
            mod_buffers = []
            if hasattr(module, "buffers"):
                mod_buffers.extend(module.buffers)
            if hasattr(module, "expert_parallel_buffers"):
                mod_buffers.extend(module.expert_parallel_buffers)
            if not mod_buffers and hasattr(module, "param_and_grad_buffer"):
                mod_buffers.append(module.param_and_grad_buffer)
            for buffer in mod_buffers:
                if idx >= len(buffers):
                    return out
                data = buffers[idx]
                idx += 1
                index_map = getattr(buffer, "param_index_map", None)
                if not index_map:
                    continue
                for param, entry in index_map.items():
                    start, end = int(entry[0]), int(entry[1])
                    sl_norm = float(torch.linalg.vector_norm(data[start:end], dtype=torch.float32))
                    out.append((names.get(param, "?"), sl_norm, end - start))
    out.sort(key=lambda t: -t[1])
    return out[:k]


def allocate_grad_accum_buffers(
    modules: Iterable[torch.nn.Module],
) -> list[torch.Tensor]:
    """
    Allocated memory for grad buffer on GPU, initialized to zeros
    """
    with torch.no_grad():
        accum_buffers: list[torch.Tensor] = []
        for grad_data in _iter_grad_buffers(modules):
            accum_buffers.append(torch.zeros_like(grad_data))
    total_gb = sum(b.numel() * b.element_size() for b in accum_buffers) / 1024**3
    dtypes = {str(b.dtype) for b in accum_buffers}
    print(f"[vcpo] allocated grad accum buffers: {total_gb:.2f} GiB, dtypes={dtypes}")
    return accum_buffers

def snapshot_grad_buffers(
    modules: Iterable[torch.nn.Module],
    to_cpu: bool = True,
    pin_memory: bool = False,
) -> list[torch.Tensor]:
    """
    Move grad buffers to CPU for inspection
    """
    snapshots: list[torch.Tensor] = []
    for grad_data in _iter_grad_buffers(modules):
        if to_cpu:
            cpu_copy = torch.empty_like(grad_data, device="cpu", pin_memory=pin_memory)
            cpu_copy.copy_(grad_data.detach(), non_blocking=pin_memory)
            snapshots.append(cpu_copy)
        else:
            snapshots.append(grad_data.detach().clone())
    return snapshots

def apply_scaled_grad_delta(
    modules: Iterable[torch.nn.Module],
    snapshots: Sequence[torch.Tensor],
    scale: float,
) -> None:
    """
    In place scale addition
        grad <- base + (grad - base) * scale
    """
    with torch.no_grad():
        for grad_data, base in zip(_iter_grad_buffers(modules), snapshots):
            if base.device != grad_data.device:
                base = base.to(device=grad_data.device, non_blocking=base.is_pinned())
            grad_data.sub_(base).mul_(scale).add_(base)

def zero_grad_accum_buffers(accum_buffers: Sequence[torch.Tensor]) -> None:
    """
    Zero grad buffer in place
    """
    with torch.no_grad():
        for buffer in accum_buffers:
            buffer.zero_()

def accumulate_grad_buffers(
    modules: Iterable[torch.nn.Module],
    accum_buffers: Sequence[torch.Tensor],
    scale: float,
) -> None:
    """
    Move accum_buffers into grad buffers of modules
    """
    grad_buffers = list(_iter_grad_buffers(modules))
    with torch.no_grad():
        try:
            torch._foreach_add_(accum_buffers, grad_buffers, alpha=scale)
            return
        except Exception as e:
            print(f"[MegatronUtils] _foreach_add failed: {e}")
            pass
        for grad_data, buffer in zip(grad_buffers, accum_buffers):
            buffer.add_(grad_data, alpha=scale)


def copy_accum_buffers_to_grad_buffers(
    modules: Iterable[torch.nn.Module],
    accum_buffers: Sequence[torch.Tensor],
) -> None:
    """
    Copy accum_buffers into grad buffers of modules
    """
    with torch.no_grad():
        for grad_data, buffer in zip(_iter_grad_buffers(modules), accum_buffers):
            grad_data.copy_(buffer)

def move_grad_buffers(
    src: Sequence[torch.Tensor],
    dest: Sequence[torch.Tensor],
    scale: float = 1,
) -> None:
    """
    Move into src into dest grad buffer
    """
    with torch.no_grad():
        try:
            torch._foreach_add_(dest, src, alpha=scale)
            return
        except Exception as e:
            print(f"[MegatronUtils] _foreach_add failed: {e}")
            pass
        for src_grad, dest_grad in zip(src, dest):
            dest_grad.add_(src_grad, alpha = scale)

def _get_local_model_grads_for_norm(actor_modules: List[torch.nn.Module]) -> List[torch.Tensor]:
    grads_for_norm: List[torch.Tensor] = []
    for model_chunk in actor_modules:
        ddp_config = getattr(model_chunk, "ddp_config", None)
        use_custom_fsdp = bool(getattr(ddp_config, "use_custom_fsdp", False))
        for param in model_chunk.parameters():
            if not param.requires_grad:
                continue
            if not param_is_not_shared(param):
                continue
            if not tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
                continue
            grad_attr = _get_main_grad_attr(param, use_custom_fsdp=use_custom_fsdp)
            grad = getattr(param, grad_attr, None)
            if grad is not None:
                grads_for_norm.append(grad)
    return grads_for_norm

def _allreduce_grads_cp(model):
    """
    Based on Megatron Core's _allreduce_non_tensor_model_parallel_grads
    """
    if mpu.get_context_parallel_world_size() <= 1:
        return

    params_avg = []
    grads_avg = []

    for model_chunk in model:
        ddp_config = model_chunk.ddp_config
        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
            if param.requires_grad:
                grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                grad = _unshard_if_dtensor(grad)
                
                grads_avg.append(grad.data)
                params_avg.append(param)

    # Loop grads and perform correct all-reduce
    if grads_avg:
        coalesced = _flatten_dense_tensors(grads_avg)
        torch.distributed.all_reduce(
            coalesced, op=torch.distributed.ReduceOp.AVG, group=mpu.get_context_parallel_group()
        )
        for param, buf, synced in zip(
            params_avg, grads_avg, _unflatten_dense_tensors(coalesced, grads_avg)
        ):
            buf.copy_(synced)
            grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
            orig_grad = getattr(param, grad_attr)
            setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def finalize_model_grads_ignore_dp(model: Sequence[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads within DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    # if config.timers is not None:
    #     config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    # for model_chunk in model:
    #     model_chunk.finish_grad_sync()
    # if config.timers is not None:
    #     config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT).
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # Do CP all reduce grads
    _allreduce_grads_cp(model)

    # All-reduce layer-norm grads (for sequence parallelism) and non-tensor parallel modules.
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_non_tensor_model_parallel_grads(model, config)
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    assert not config.moe_router_enable_expert_bias, f"Expert Bias not supported"

    # Disable all reduce across TP x CP x DP
    # if config.moe_router_enable_expert_bias:
    #     _update_router_expert_bias(model, config)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        last_rank = mpu.get_pipeline_model_parallel_last_rank()
        pp_group = mpu.get_pipeline_model_parallel_group()

        if not isinstance(last_rank, list):
            assert not isinstance(last_rank, list)
            last_rank = [last_rank]
            assert not isinstance(pp_group, list)
            pp_group = [pp_group]

        # need to do a broadcast for every pp group, even though num_tokens should be the same.
        num_tokens_list = []
        for lr, group in zip(last_rank, pp_group):
            torch.distributed.broadcast(num_tokens, src=lr, group=group)
            num_tokens_list.append(torch.clone(num_tokens))
        assert all(x.item() == num_tokens_list[0] for x in num_tokens_list)

        # all-reduce across DP ranks.
        # torch.distributed.all_reduce(
        #     num_tokens, group=mpu.get_data_parallel_group(with_context_parallel=True)
        # )
        # for model_chunk in model:
        #     if num_tokens > 0:
        #         scaling = 1.0 / num_tokens
        #         model_chunk.scale_gradients(scaling)


def _noop_finalize_model_grads(model: Sequence[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """Schedule-level finalize replacement for the buffer-free per-traj path.

    When gradients accumulate directly in Megatron's main grad buffer across
    forward_backward_batch calls, the TP/CP/PP partial-grad all-reduces in the
    finalize step must not run per call — re-reducing an already-reduced
    accumulated buffer double-counts earlier trajectories' contributions.
    The real finalize runs once, on the fully accumulated gradient, via
    `finalize_model_grads_ignore_dp` at optimizer-step time."""
    pass


def disable_grad_finalize(actor_modules: Sequence[torch.nn.Module]):
    config = get_model_config(actor_modules[0])
    orig_finalize = config.finalize_model_grads_func
    config.finalize_model_grads_func = _noop_finalize_model_grads
    return orig_finalize


def restore_grad_finalize(actor_modules: Sequence[torch.nn.Module], orig_finalize) -> None:
    config = get_model_config(actor_modules[0])
    config.finalize_model_grads_func = orig_finalize


def disable_dp_sync(actor_modules: Iterable[torch.nn.Module]) -> Tuple:
    config = get_model_config(actor_modules[0])
    orig_no_sync = config.no_sync_func
    orig_grad_sync = config.grad_sync_func
    orig_finalize = config.finalize_model_grads_func

    config.no_sync_func = nullcontext
    config.grad_sync_func = None
    config.finalize_model_grads_func = finalize_model_grads_ignore_dp

    return orig_no_sync, orig_grad_sync, orig_finalize

def restore_dp_sync(actor_modules: Iterable[torch.nn.Module], orig_no_sync, orig_grad_sync, orig_finalize):
    config = get_model_config(actor_modules[0])
    config.no_sync_func = orig_no_sync
    config.grad_sync_func = orig_grad_sync
    config.finalize_model_grads_func = orig_finalize