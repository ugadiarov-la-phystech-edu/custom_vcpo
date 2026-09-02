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


def find_param(modules: Iterable[torch.nn.Module], name_suffix: str):
    """Diagnostics: (name, param) of the first parameter whose name ends with ``name_suffix``."""
    for module in modules:
        for name, param in module.named_parameters():
            if name.endswith(name_suffix):
                return name, param
    return None, None


def param_slice(modules: Iterable[torch.nn.Module], buffers: Sequence[torch.Tensor], param):
    """Diagnostics: the view of ``param``'s slice inside ``buffers`` (laid out like the grad
    buffers), or None when the offsets are unavailable."""
    buffers = list(buffers)
    idx = 0
    for module in modules:
        mod_buffers = []
        if hasattr(module, "buffers"):
            mod_buffers.extend(module.buffers)
        if hasattr(module, "expert_parallel_buffers"):
            mod_buffers.extend(module.expert_parallel_buffers)
        if not mod_buffers and hasattr(module, "param_and_grad_buffer"):
            mod_buffers.append(module.param_and_grad_buffer)
        for buffer in mod_buffers:
            if idx >= len(buffers):
                return None
            data = buffers[idx]
            idx += 1
            index_map = getattr(buffer, "param_index_map", None)
            if index_map and param in index_map:
                start, end = int(index_map[param][0]), int(index_map[param][1])
                return data[start:end]
    return None


def param_slice_norm(modules: Iterable[torch.nn.Module], buffers: Sequence[torch.Tensor], param) -> float:
    """Diagnostics: L2 norm of ``param``'s slice inside ``buffers`` (laid out like the grad buffers)."""
    sl = param_slice(modules, buffers, param)
    if sl is None:
        return float("nan")
    with torch.no_grad():
        return float(torch.linalg.vector_norm(sl, dtype=torch.float32))


def slice_update_report(before, after, expected_delta, suspect) -> str:
    """Diagnostics: decompose ``after - before`` of a buffer slice against the intended
    update ``expected_delta`` and a ``suspect`` tensor (e.g. the other accumulator's slice).
    Returns norms of the residual and cosines, all in fp32."""
    with torch.no_grad():
        delta = after.float() - before.float()
        resid = delta - expected_delta.float()
        s = suspect.float()

        def cos(a, b):
            na, nb = a.norm(), b.norm()
            return float((a * b).sum() / (na * nb + 1e-12))

        return (
            f"|delta|={float(delta.norm()):.4e} |expected|={float(expected_delta.float().norm()):.4e} "
            f"|delta-expected|={float(resid.norm()):.4e} cos(resid,suspect)={cos(resid, s):.4f} "
            f"|resid|/|suspect|={float(resid.norm() / (s.norm() + 1e-12)):.4f}"
        )


def param_grad_state(param) -> str:
    """Diagnostics: norms/pointers of ``param.main_grad`` and ``param.grad`` plus Megatron's fusion flag."""
    with torch.no_grad():
        mg = getattr(param, "main_grad", None)
        g = getattr(param, "grad", None)
        mg_s = (
            f"main_grad|={float(torch.linalg.vector_norm(mg, dtype=torch.float32)):.4e}@{hex(mg.data_ptr())}"
            if mg is not None
            else "main_grad=None"
        )
        g_s = (
            f"grad|={float(torch.linalg.vector_norm(g, dtype=torch.float32)):.4e}@{hex(g.data_ptr())}"
            if g is not None
            else "grad=None"
        )
    return f"|{mg_s} |{g_s} added_to_main={getattr(param, 'grad_added_to_main_grad', None)}"


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

    GPU targets go through accumulate_grad_buffers (chunked add_). CPU-resident targets
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

_ADD_CHUNK = 2**28  # elements per chunked add_ (512 MiB of bf16)


def _chunked_add_enabled() -> bool:
    """Default ON: the grad-sized OPOB accumulators are updated with per-tensor, chunked
    ``add_`` instead of ``torch._foreach_add_``.

    Why: with torch 2.8.0+cu128 on H100, ``_foreach_add_([score], [main_grad], alpha)`` on
    the 4.1e9-element bf16 Megatron grad buffer (Qwen3-8B, TP=2) was measured in situ to add
    +-alpha * (the *other* accumulator's output_layer slice) instead of alpha * main_grad
    whenever alpha != 1 (2026-09-02/03 OPOB smokes: score-add residual 0.3-0.7 with cosine
    +-0.98 to the accum buffer; per-tensor chunked add_: residual 4e-5, actor/grad_norm 0.14
    instead of 1e7-1e9). A standalone probe of the same op/size/layout did not reproduce it,
    so the trigger is process-specific; chunked add_ avoids the foreach kernel entirely.
    Set VCPO_OPOB_CHUNKED_ADD=0 to get the old foreach path back (diagnosis only)."""
    import os

    return os.environ.get("VCPO_OPOB_CHUNKED_ADD", "1") not in ("", "0", "false", "False")


def _add_lists_(dest: Sequence[torch.Tensor], src: Sequence[torch.Tensor], alpha: float) -> None:
    """dest[i] += alpha * src[i], chunked so no single kernel spans > 2^28 elements."""
    with torch.no_grad():
        for d, s in zip(dest, src, strict=True):
            d_flat, s_flat = d.view(-1), s.view(-1)
            for start in range(0, d_flat.numel(), _ADD_CHUNK):
                d_flat[start : start + _ADD_CHUNK].add_(s_flat[start : start + _ADD_CHUNK], alpha=alpha)


def accumulate_grad_buffers(
    modules: Iterable[torch.nn.Module],
    accum_buffers: Sequence[torch.Tensor],
    scale: float,
) -> None:
    """
    Move accum_buffers into grad buffers of modules
    """
    grad_buffers = list(_iter_grad_buffers(modules))
    assert len(grad_buffers) == len(accum_buffers), (len(grad_buffers), len(accum_buffers))
    if _chunked_add_enabled():
        _add_lists_(accum_buffers, grad_buffers, float(scale))
        return
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
    if dest and not dest[0].is_cuda:
        # Host-resident accumulators: plain (multithreaded) CPU adds.
        with torch.no_grad():
            for src_grad, dest_grad in zip(src, dest, strict=True):
                _add_into_(dest_grad, src_grad, float(scale))
        return
    if _chunked_add_enabled():
        _add_lists_(dest, src, float(scale))
        return
    with torch.no_grad():
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