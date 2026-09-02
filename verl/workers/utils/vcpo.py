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

_COPY_CHUNK = 2**28  # elements per chunk for host<->device copies with dtype conversion


def _resolve_accum_dtype(grad_dtype: torch.dtype, accum_dtype: str | None) -> torch.dtype:
    if accum_dtype in (None, "", "auto"):
        return grad_dtype
    name = str(accum_dtype).replace("torch.", "")
    aliases = {"fp32": "float32", "float": "float32", "bf16": "bfloat16", "fp16": "float16", "half": "float16"}
    name = aliases.get(name, name)
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"grad_baselining.accum_dtype={accum_dtype!r} is not a torch dtype")
    return dtype


def allocate_grad_accum_buffers(
    modules: Iterable[torch.nn.Module],
    device: str = "cuda",
    dtype: str | None = None,
) -> list[torch.Tensor]:
    """
    Allocate accumulators shaped like the modules' grad buffers, initialized to zeros.

    device="cuda": on the grad buffers' device (the paper's single-backward layout).
    device="cpu": pinned host memory; see accumulate_grad_buffers_multi for the d2h path.
    dtype: None/"auto" = the grad buffer's dtype, else e.g. "float32".
    """
    with torch.no_grad():
        accum_buffers: list[torch.Tensor] = []
        for grad_data in _iter_grad_buffers(modules):
            acc_dtype = _resolve_accum_dtype(grad_data.dtype, dtype)
            if device == "cpu":
                accum_buffers.append(
                    torch.zeros(grad_data.shape, dtype=acc_dtype, device="cpu", pin_memory=torch.cuda.is_available())
                )
            elif acc_dtype == grad_data.dtype:
                accum_buffers.append(torch.zeros_like(grad_data))
            else:
                accum_buffers.append(torch.zeros(grad_data.shape, dtype=acc_dtype, device=grad_data.device))
    total_gb = sum(b.numel() * b.element_size() for b in accum_buffers) / 1024**3
    dtypes = {str(b.dtype) for b in accum_buffers}
    devices = {str(b.device) for b in accum_buffers}
    print(f"[vcpo] allocated grad accum buffers: {total_gb:.2f} GiB, dtypes={dtypes}, devices={devices}")
    return accum_buffers


def allocate_staging_buffers(modules: Iterable[torch.nn.Module]) -> list[torch.Tensor]:
    """Pinned host copies of the grad buffers (same shape/dtype), used to move each
    trajectory's gradient to CPU-resident accumulators with a single d2h transfer."""
    staging: list[torch.Tensor] = []
    for grad_data in _iter_grad_buffers(modules):
        pin = torch.cuda.is_available()
        staging.append(torch.empty(grad_data.shape, dtype=grad_data.dtype, device="cpu", pin_memory=pin))
    total_gb = sum(b.numel() * b.element_size() for b in staging) / 1024**3
    print(f"[vcpo] allocated pinned staging buffers: {total_gb:.2f} GiB")
    return staging


def stage_grad_buffers(modules: Iterable[torch.nn.Module], staging: Sequence[torch.Tensor]) -> None:
    """Copy the modules' grad buffers into the pinned ``staging`` tensors (blocking)."""
    grad_buffers = list(_iter_grad_buffers(modules))
    assert len(grad_buffers) == len(staging), (len(grad_buffers), len(staging))
    with torch.no_grad():
        for grad_data, stage in zip(grad_buffers, staging, strict=True):
            stage.copy_(grad_data, non_blocking=True)
        if grad_buffers and grad_buffers[0].is_cuda:
            torch.cuda.current_stream(grad_buffers[0].device).synchronize()


def _add_into_(dest: torch.Tensor, src: torch.Tensor, alpha: float) -> None:
    """dest += alpha * src for same-device tensors, converting src's dtype chunk-wise when needed."""
    if dest.dtype == src.dtype:
        dest.add_(src, alpha=alpha)
        return
    d_flat, s_flat = dest.view(-1), src.view(-1)
    for start in range(0, d_flat.numel(), _COPY_CHUNK):
        d_flat[start : start + _COPY_CHUNK].add_(s_flat[start : start + _COPY_CHUNK].to(dest.dtype), alpha=alpha)


def accumulate_grad_buffers_multi(
    modules: Iterable[torch.nn.Module],
    targets: Sequence[tuple[Sequence[torch.Tensor], float]],
    staging: Sequence[torch.Tensor] | None = None,
) -> None:
    """buffers += scale * grad_buffers for every (buffers, scale) in ``targets``.

    GPU targets use the foreach path of accumulate_grad_buffers. CPU-resident targets
    share ONE d2h transfer of the grad buffers (into ``staging``) and are then updated on
    the host; mixing device kinds across targets is allowed.
    """
    grad_buffers = list(_iter_grad_buffers(modules))
    cpu_targets = [(bufs, scale) for bufs, scale in targets if bufs and not bufs[0].is_cuda]
    gpu_targets = [(bufs, scale) for bufs, scale in targets if bufs and bufs[0].is_cuda]
    for bufs, scale in gpu_targets:
        accumulate_grad_buffers(modules, bufs, scale)
    if not cpu_targets:
        return
    if grad_buffers and grad_buffers[0].is_cuda:
        assert staging is not None, "CPU-resident accumulators need pinned staging buffers (allocate_staging_buffers)"
        stage_grad_buffers(modules, staging)
        source = list(staging)
    else:
        source = grad_buffers  # already on the host (tests)
    with torch.no_grad():
        for bufs, scale in cpu_targets:
            assert len(bufs) == len(source), (len(bufs), len(source))
            for dest, src in zip(bufs, source, strict=True):
                _add_into_(dest, src, float(scale))

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
    Copy accum_buffers into grad buffers of modules (host-resident / other-dtype accumulators
    are converted chunk-wise on their own device first, so no grad-sized temporary is created).
    """
    with torch.no_grad():
        for grad_data, buffer in zip(_iter_grad_buffers(modules), accum_buffers, strict=True):
            if buffer.device == grad_data.device and buffer.dtype == grad_data.dtype:
                grad_data.copy_(buffer)
                continue
            g_flat, b_flat = grad_data.view(-1), buffer.view(-1)
            for start in range(0, g_flat.numel(), _COPY_CHUNK):
                chunk = b_flat[start : start + _COPY_CHUNK]
                if chunk.dtype != grad_data.dtype:
                    chunk = chunk.to(grad_data.dtype)
                g_flat[start : start + _COPY_CHUNK].copy_(chunk, non_blocking=chunk.is_pinned())
        if accum_buffers and not accum_buffers[0].is_cuda:
            grads = list(_iter_grad_buffers(modules))
            if grads and grads[0].is_cuda:
                torch.cuda.current_stream(grads[0].device).synchronize()

def move_grad_buffers(
    src: Sequence[torch.Tensor],
    dest: Sequence[torch.Tensor],
    scale: float = 1,
) -> None:
    """
    Move into src into dest grad buffer
    """
    with torch.no_grad():
        if dest and not dest[0].is_cuda:
            # Host-resident accumulators: plain (multithreaded) CPU adds.
            for src_grad, dest_grad in zip(src, dest, strict=True):
                _add_into_(dest_grad, src_grad, float(scale))
            return
        try:
            torch._foreach_add_(dest, src, alpha=scale)
            return
        except Exception as e:
            print(f"[MegatronUtils] _foreach_add failed: {e}")
            pass
        for src_grad, dest_grad in zip(src, dest):
            dest_grad.add_(src_grad, alpha = scale)

def _optimizer_has_grad_scaler(optimizer) -> bool:
    """True when the (possibly chained / hybrid-offload) Megatron optimizer carries a grad
    scaler, i.e. prepare_grads() is needed for its inf/nan check (fp16 loss scaling)."""
    if optimizer is None:
        return False
    if getattr(optimizer, "grad_scaler", None) is not None:
        return True
    for sub in getattr(optimizer, "chained_optimizers", []) or []:
        if _optimizer_has_grad_scaler(sub):
            return True
    return False


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