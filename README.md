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

## Hybrid Sync Training for OpenPangu-7B

The Hybrid Sync baseline is verl's synchronous PPO trainer (`verl.trainer.main_ppo`) with a hybrid engine: vLLM
generation and Megatron training share all 8 GPUs of one node and alternate. Each step generates responses for a batch
of prompts with the current policy, then trains on that batch.

### Prepare the model

The launcher expects openPangu-Embedded-7B re-aliased to the Llama architecture, so that Megatron and vLLM load it
with their Llama code paths. The weights are unchanged; only `config.json` is rewritten.
The tokenizer keeps its remote code, so `trust_remote_code=True` is still required.

```bash
python scripts/realias_openpangu_to_llama.py --out $HOME/models/openPangu-Embedded-7B-llama
# downloads FreedomIntelligence/openPangu-Embedded-7B (~16 GB); use --src <dir> for a local copy
```

Point `MODEL_PATH` at the re-aliased directory when launching. The launcher also puts `$HF_HOME/modules` on
`PYTHONPATH`, so the Ray workers can import the tokenizer's remote code, and prepends the BOS token to every prompt
(`add_bos_token_to_prompt=True`), as openPangu expects.

### Launch

One launcher in `recipe/fully_async_policy/shell/vcpo/dapo/baseline/`, for one node with 8 H100 GPUs (80 GB class):

| Script | Model | Batch | Mini-batch | PPO epochs | Validation sampling | Reward |
| --- | --- | --- | --- | --- | --- | --- |
| `main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh` | openPangu-Embedded-7B (Llama re-aliased) | 128 prompts × 16 | 32 prompts (4 updates per step) | 1 | T = 0.8, top-p = 0.7 | stock `math_dapo` |

Validation and checkpoints run every 3 steps (`test_freq`, `save_freq`), and the run lasts 3 epochs over the training
set (`total_epochs`). These count trainer steps of 128 prompts, not optimizer updates; `max_updates` is rounded up to
whole steps.

```bash
source /path/to/envs/vcpo-env/activate_vcpo.sh
MODEL_PATH=$HOME/models/openPangu-Embedded-7B-llama \
bash recipe/fully_async_policy/shell/vcpo/dapo/baseline/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh
```

Override a default with an environment variable, or append any Hydra override after the script name:

```bash
MODEL_PATH=$HOME/models/openPangu-Embedded-7B-llama SEED=2 max_updates=300 test_freq=6 \
bash recipe/fully_async_policy/shell/vcpo/dapo/baseline/main_ppo_sync_8gpu_dapo17k_grpo_B128xn16_mini32_openpangu7b.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

## Outputs and monitoring

Log metrics go to `logs/<exp_name>/` under the repository root:

- `tensorboard/`: all metrics (`tensorboard --logdir logs`);
- `global_step_<N>/actor/huggingface/`: Hugging Face exports every `save_freq` trainer steps (`resume_mode=disable`;
  these checkpoints do not carry optimizer state).

