# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [veRL](https://github.com/volcengine/verl) (base commit `15a9b0f5`) implementing **VCPO** (Variance-Controlled Off-Policy Optimization, arXiv:2602.17616) — stable *asynchronous* RL training for LLMs. Async RL pipelines rollout generation concurrently with learning, but stale rollouts make importance-sampling ratios heavy-tailed and the policy gradient high-variance (predicted by collapsing effective sample size, ESS). VCPO adds two variance controls that keep training stable up to k=128 steps off-policy:

1. **ESS-guided step scaling** — scale the effective learning rate by `min(1, ess_ratio / base_ess_ratio)` (sqrt or linear rule), where `ess_ratio = ESS/B = (Σw)²/(B·Σw²)` over sequence-level importance weights `w`. `base_ess_ratio` is the *empirical on-policy reference* ρ_on (paper uses 1.0 for single-turn math, 0.55 for multiturn tool use); the paper computes ESS from *unclipped* ratios even though the loss uses TIS-clipped ones.
2. **OPOB** — closed-form off-policy optimal baseline (no critic): `b* = Σ(W_i·R_i)/ΣW_i` with `W_i = ‖∇log π(τ_i)‖²·ratio_i²` (optionally length-normalized), computed from per-trajectory gradient norms in a *single* backward pass: per-traj score gradients accumulate into a reward-weighted buffer and a score buffer (final grad `= G_R − b*·G_S`), with the DP all-reduce deferred until after per-trajectory stats are computed (~19% step-time overhead vs ~100% for a naive second backward).

The loss combines both with sequence-level truncated IS (threshold c=8.0 in the paper): `L = −E[min(π/μ, c)·(R − b*)·log π]`, optimized with AdamW.

VCPO is implemented for the **Megatron backend**. Environment per README: Megatron-Core 0.13.1 + vLLM 0.11.0.

Local work on top of the VCPO base (see `git log` since `ff7c8aa`): message-queue and cancel-queue checkpointing, async tool-use loop with validation + "simple-tir" tool parser, code sandbox server, dataset-source-aware reward calculation, configurable batch size per rollout DP replica.

## Commands

```bash
# Local uv environment (Python 3.12, torch 2.8.0+cu128, vLLM 0.11.0, verl 0.8.0 deps,
# flash-attn 2.8.1, Megatron-Core 0.13.1, TransformerEngine 2.6.0.post1).
# Recreate with: bash scripts/setup_uv_env.sh
source /home/elfray/programs/uv.sh && source /samsung/uv-envs/vcpo-env/bin/activate
# Run from the repo root so the local (fork) verl package shadows the installed verl 0.8.0.

# Install (Python-only dev iteration, alternative to the uv env)
pip install -e .[test,vllm]     # or .[test,sglang]

# Lint / format / typecheck (ruff + mypy + license/docstring checks + config regen)
pre-commit run                  # staged changes
pre-commit run --all-files
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg

# Tests (pytest; layout mirrors verl/ sub-namespaces, e.g. tests/trainer ↔ verl/trainer)
pytest tests/trainer/test_foo.py::test_bar          # single test
pytest tests/**/test_*_on_cpu.py                    # *_on_cpu.py suffix = CPU-only; everything else assumes GPU
# tests/special_* dirs: special_distributed (multi-GPU), special_e2e (end-to-end scripts),
# special_sanity (quick checks), special_npu, special_standalone

# Training (edit model/data paths in the script first; data: hf download lukhuang/vcpo --repo-type dataset --local-dir data)
bash recipe/fully_async_policy/shell/vcpo/gsm8k/synchronous.sh      # sync baseline (k=0)
bash recipe/fully_async_policy/shell/vcpo/math/vcpo_k=10.sh          # async VCPO, k = staleness_threshold
bash recipe/fully_async_policy/shell/vcpo/multiturn/vcpo_k=2.sh      # long-horizon tool use (SimpleTIR setting)
```

Editing config dataclasses under `verl/trainer/config/` or `verl/workers/config/` requires regenerating `verl/trainer/config/_generated_*.yaml` via `scripts/generate_trainer_config.sh` (the `autogen-trainer-cfg` pre-commit hook does and verifies this).

## Architecture

### Fully-async training pipeline (`recipe/fully_async_policy/`)

Two long-lived Ray actors run concurrently, connected by a `MessageQueue` Ray actor:

- `fully_async_main.py` — Hydra entrypoint (`--config-name=fully_async_ppo_megatron_trainer`). `FullyAsyncTaskRunner` builds the rollouter, trainer, `MessageQueue`/`MessageQueueClient`, and `ParameterSynchronizer`, then launches `rollouter.fit()` and `trainer.fit()` in parallel. Strategy selects the worker module: `megatron_worker.py` or `fsdp_workers.py` (both define `DetachActorWorker` / `DetachAsyncRolloutWorker`).
- `fully_async_rollouter.py` — `FullyAsyncRollouter` streams generation and pushes finished `RolloutSample`s into the queue. Maintains an in-memory `cancel_queue` for **partial rollouts**: when a parameter sync pauses generation, in-flight samples are parked and resumed after the sync. `async_training.bsz_per_dp_rank` drives `max_concurrent_samples` per rollout DP replica.
- `fully_async_trainer.py` — `FullyAsyncTrainer.fit()` loops: pull samples from queue → compute rewards/advantages → actor update → trigger parameter sync. With OPOB enabled it pins each rollout group to one DP rank (`dp_group_key="uid"`).
- `param_sync.py` — `ParameterSynchronizer` broadcasts trainer weights to rollout workers over an NCCL group and bumps the parameter version in the queue and rollouter.
- `message_queue.py` — queue of samples + separate validation queue. Checkpointing: `save_state`/`load_state` persist queue contents and the parameter version to `message_queue.pt` when `async_training.save_queue_state=True`; the rollouter's `cancel_queue` is snapshotted/restored separately.
- `staleness_utils.py` — VCPO bookkeeping: per-trajectory `TrajRecord`s, sequence-level IS ratios, ESS computation all-reduced over the DP group (`compute_ess_info`, `compute_global_ess_ratio`), and `compute_opob_baseline`.

### VCPO core (Megatron backend)

- `verl/workers/utils/vcpo.py` — grad-buffer plumbing: allocate/accumulate/move per-trajectory gradient buffers, `apply_scaled_grad_delta`, and `disable_dp_sync`/`finalize_model_grads_ignore_dp` to hand-control DP gradient sync during per-trajectory accumulation.
- `verl/workers/actor/megatron_actor.py` — the per-trajectory update path (enabled by `actor.update_policy_per_traj`). OPOB: accumulate per-traj score gradients, then subtract `compute_opob_baseline(...)` via `move_grad_buffers(scale=-b)`. ESS scaling: `_optimizer_step_with_buffer` scales LR by the ess ratio before stepping, then restores it.
- Config dataclasses in `verl/workers/config/actor.py`: `ESSScalingConfig` (`actor.ess_scaling.enable/scaling_rule/base_ess_ratio/use_clipped`) and `GradBaselineConfig` (`actor.grad_baselining.enable/scope/agg_mode/use_is_weights/...`).
- Async-specific keys live under `async_training.*` in `recipe/fully_async_policy/config/*.yaml`: `staleness_threshold` (the "k"), `trigger_parameter_sync_step`, `partial_rollout`, `use_rollout_log_probs`, `save_queue_state`, `bsz_per_dp_rank`. Rollout IS-correction keys under `algorithm.rollout_correction.*`.

### Async tool-use loop and code sandbox

- `recipe/fully_async_policy/agent_loop/partial_tool_agent_loop.py` — `AsyncPartialToolAgentLoop` (registered as `async_partial_tool_agent`), a state machine extending `verl/experimental/agent_loop/tool_agent_loop.py` that supports partial rollout across tool calls; `extra_fields["validate"]` disables the partial path during validation.
- `verl/experimental/agent_loop/tool_parser.py` — tool parsers registered by name (`hermes`, `gpt-oss`, `simple-tir`); `SimpleTIRToolParser` implements the SimpleTIR interleaved reasoning/tool-call format used in multiturn experiments.
- `recipe/fully_async_policy/code_sandbox/` — FastAPI Python-execution sandbox (`code_sandbox.py`) used as the tool backend; `test_tool_agent_loop.py` and `debug_tool_loop.sh` exercise the loop standalone.

### Launch script shape (`recipe/fully_async_policy/shell/vcpo/`)

Each experiment dir (`gsm8k/`, `math/`, `multiturn/`) has `synchronous.sh` and `vcpo_k=N.sh`. Scripts split GPUs between rollout and training (`n_gpus_rollout` / `n_gpus_training`), then invoke `python -m recipe.fully_async_policy.fully_async_main` with Hydra overrides for data paths, Megatron parallelism (`train_tp/pp/cp`), `actor.update_policy_per_traj`, `actor.ess_scaling.*`, `actor.grad_baselining.*`, and `async_training.*`.
