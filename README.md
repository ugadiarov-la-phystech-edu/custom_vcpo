# Staleness-Aware Experience Replay (SER) for fully asynchronous GRPO

- Branch 'qwen3-8b_orz-7b' contains the implementation of SER for the Qwen3-8B and ORZ-7B models.
- Branch 'baselines_qwen3-8b_orz-7b' contains the code for the Hybrid Sync baseline for the Qwen3-8B and ORZ-7B models.
- Branch 'openpangu-7b' contains the implementation of SER for the OpenPangu-7B model.
- Branch 'baselines_openpangu-7b' contains the code for the Hybrid Sync baseline for the OpenPangu-7B model.

## Installation

Requirements: Linux x86-64, NVIDIA GPUs with a CUDA 12.8-capable driver, and internet access to PyPI, GitHub and
huggingface.co. A system CUDA toolkit is optional. Without `nvcc`, the script builds TransformerEngine against the CUDA
headers from pip wheels.

`scripts/setup_uv_env.sh` creates a [uv](https://docs.astral.sh/uv/) virtual environment with Python 3.12,
torch 2.8.0 (cu128), vLLM 0.11.0, flash-attn 2.8.1, FlashInfer 0.3.1, Megatron-Core 0.13.1 and TransformerEngine
2.6.0.post1, then prints the installed versions as a sanity check.

```bash
bash scripts/setup_uv_env.sh /path/to/envs/vcpo-env     # default location: $HOME/uv-envs/vcpo-env
source /path/to/envs/vcpo-env/activate_vcpo.sh           # uv on PATH + venv activated + cd to the repo root
```

The script installs uv into `~/.local/bin` if uv is not found. Options (environment variables; see
`bash scripts/setup_uv_env.sh --help`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENV_DIR` | `$HOME/uv-envs/vcpo-env` | where to create the venv (the first argument takes precedence) |
| `UV_ACTIVATE` | unset | script to source that puts `uv` on `PATH` (e.g. one that also sets `UV_CACHE_DIR`) |
| `INSTALL_UV` | `1` | install uv when it is missing; `0` fails instead |
| `USE_MEGATRON` | `1` | install Megatron-Core and TransformerEngine (needed by the training scripts) |
| `FORCE` | `0` | `1` deletes an existing `ENV_DIR` first; without it an existing directory is never touched |
| `PYTHON_VERSION` / `FLASH_ATTN_WHEEL` | `3.12` / cp312 wheel | change both together |
| `MAX_JOBS` | `8` | parallel jobs for the TransformerEngine build |

uv's own `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` are respected. Put them on the same large filesystem as the venv
when the home directory is small. The generated `activate_vcpo.sh` keeps these settings.

**Always run training from the repository root: the local (forked) `verl` package must shadow the `verl` wheel installed
as a dependency.**

## Data

The scripts read the datasets straight from the Hugging Face Hub through `hf://` URLs. Nothing needs to be downloaded
by hand; `datasets` caches the files under `HF_HOME` on first use.

| Role | Dataset | Rows | `data_source` |
| --- | --- | ---: | --- |
| train | [elfray/dapo-math-17k](https://huggingface.co/datasets/elfray/dapo-math-17k) | 17,398 | `math_dapo` |
| validation | [elfray/aime-2024](https://huggingface.co/datasets/elfray/aime-2024) (30 problems × 32) | 960 | `math_dapo` |
| validation | [elfray/aime-2025](https://huggingface.co/datasets/elfray/aime-2025) (30 problems × 32) | 960 | `aime2025_dapo` |
| validation | [elfray/math500_x3](https://huggingface.co/datasets/elfray/math500_x3) (500 problems × 3) | 1,500 | `math500_dapo` |

All four use the DAPO prompt ("… The last line of your response should be of the form Answer: $Answer …") in verl's
parquet schema (`prompt`, `data_source`, `reward_model.ground_truth`, …). Validation metrics are reported per
`data_source`, so AIME-2024 appears as `math_dapo`. To use local copies, override `TRAIN_FILE` / `TEST_FILE` with
paths.

## SER Training for Qwen3-8B and ORZ-7b

Two launchers in `recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/`, each for one node with 8 H100 GPUs (80 GB
class):

| Script | Model | GPUs rollout + train | Mini-batch | `tau` / `k` | `min_ess` | Concurrency ramp | Validation sampling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grpo_novcpo_8gpu_dapo17k_5+3_…_nu=1_fresh=0.5.sh` | `Qwen/Qwen3-8B` | 5 + 3 | 33 groups × 16 | 8 / 32 | 1.1 | `[5, 12, 20]` | T = 0.8, top-p = 0.7 |
| `grpo_novcpo_8gpu_orz72k_3+5_…_orz7b.sh` | `Open-Reasoner-Zero/Open-Reasoner-Zero-7B` | 3 + 5 | 35 groups × 16 | 4 / 16 | 1.07 | `[7, 14, 24]` | T = 1.0, top-p = 1.0 |

Reuse half-life $\nu \leftarrow$ replay_reuse_halflife

Staleness half-life $h \leftarrow$ replay_tau

Replay staleness threshold $k \leftarrow$ replay_staleness_threshold

ESS threshold $\kappa \leftarrow$ min_ess

Learining rate scale $\lambda \leftarrow$ ess_lr_scale

Fresh-share ratio $f \leftarrow$ replay_min_fresh_ratio


```bash
source /path/to/envs/vcpo-env/activate_vcpo.sh

# Qwen3-8B
bash "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh"

# Open-Reasoner-Zero-7B
bash "recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_orz72k_3+5_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.07_ess-lr-scale=0.5_orz7b.sh"
```

Override a default with an environment variable, or append any Hydra override after the script name:

```bash
SEED=2 max_updates=300 test_freq=6 \
bash recipe/fully_async_policy/shell/vcpo/dapo/replay_buffer/grpo_novcpo_8gpu_dapo17k_5+3_*.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

## Outputs and monitoring

Log metrics go to `logs/<exp_name>/` under the repository root:

- `tensorboard/`: all metrics (`tensorboard --logdir logs`);
- `global_step_<N>/actor/huggingface/`: Hugging Face exports every `save_freq` updates (`resume_mode=disable`; these
  checkpoints do not carry optimizer or replay-buffer state).

