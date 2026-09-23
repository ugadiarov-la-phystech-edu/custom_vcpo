# Staleness-Aware Experience Replay (SER) for fully asynchronous GRPO

 - The 'baselines_qwen3-8b_orz-7b' branch is used to run Hybrid Sync baselines for the Qwen3-8B and ORZ-7B models. It contains code close to vanilla VERL, with changes related to metrics accounting, checkpointing, and reward scoring for ORZ-7B and the MATH-500 dataset.
 - The 'baselines_openpangu-7b' branch is used to run the Hybrid Sync baseline for OpenPangu-7B; it additionally contains a Megatron port of the OpenPangu-7B architecture and functionality for OpenPangu-specific prompt handling.
 - The 'qwen3-8b_orz-7b' branch significantly reworks the VERL codebase to implement the SER method, with changes that also affect the Hybrid Sync baseline. It contains the script for launching SER with the Qwen3-8B and Open-Reasoner-Zero-7B models.
 - Compared to the 'qwen3-8b_orz-7b' branch, the 'openpangu-7b' branch additionally contains a Megatron port of the OpenPangu-7B architecture and functionality for OpenPangu-specific prompt handling. It contains the script for launching SER with the OpenPangu-7B model.

All branches use the same Python environment.

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

## Hybrid Sync Training for Qwen3-8B  and ORZ-7b

The Hybrid Sync baseline is verl's synchronous PPO trainer (`verl.trainer.main_ppo`) with a hybrid engine: vLLM
generation and Megatron training share all 8 GPUs of one node and alternate. Each step generates responses for a batch
of prompts with the current policy, then trains on that batch.

Two launchers in `recipe/fully_async_policy/shell/vcpo/dapo/baseline/`, each for one node with 8 H100 GPUs (80 GB
class):

| Script | Model | Validation sampling | Reward |
| --- | --- | --- | --- |
| `main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh` | `Qwen/Qwen3-8B` | T = 0.8, top-p = 0.7 | stock `math_dapo` |
| `main_ppo_sync_8gpu_grpo_B128xn16_mini32_orz7b.sh` | `Open-Reasoner-Zero/Open-Reasoner-Zero-7B` | T = 1.0, top-p = 1.0 | `recipe/fully_async_policy/reward/orz_tag_aware_math.py` |

```bash
source /path/to/envs/vcpo-env/activate_vcpo.sh

# Qwen3-8B
bash recipe/fully_async_policy/shell/vcpo/dapo/baseline/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh

# Open-Reasoner-Zero-7B
bash recipe/fully_async_policy/shell/vcpo/dapo/baseline/main_ppo_sync_8gpu_grpo_B128xn16_mini32_orz7b.sh
```

Override a default with an environment variable, or append any Hydra override after the script name:

```bash
SEED=2 max_updates=300 test_freq=6 \
bash recipe/fully_async_policy/shell/vcpo/dapo/baseline/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_qwen3-8b.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

## Outputs and monitoring

Log metrics go to `logs/<exp_name>/` under the repository root:

- `tensorboard/`: all metrics (`tensorboard --logdir logs`);
- `global_step_<N>/actor/huggingface/`: Hugging Face exports every `save_freq` updates (`resume_mode=disable`; these
  checkpoints do not carry optimizer or replay-buffer state).

