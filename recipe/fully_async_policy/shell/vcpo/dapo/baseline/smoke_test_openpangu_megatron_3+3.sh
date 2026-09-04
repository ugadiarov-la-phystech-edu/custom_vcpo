#!/usr/bin/env bash
# =============================================================================
# smoke_test_openpangu_megatron_3+3.sh
#
# The MEGATRON openPangu arm
#   grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh
# through the same 2-step, 3+3 plumbing test as the FSDP2 arm: this is a thin wrapper over
# smoke_test_openpangu_3+3.sh (read its header for what is shortened and why), which runs
# the real arm script with ARM_SCRIPT swapped and then verify_checkpoints.py.
#
# What this one is FOR, beyond "it runs": the Megatron path carries production code the
# FSDP2 arm never needed (config_converter add_bias_linear, loader/weight_converter/saver
# handling of o_proj.bias, frozen MLP biases, model_initializer.py) and only a GPU run
# proves it end to end:
#   * verify_checkpoints.py --base-model diffs parameter NAMES of every hf_model save
#     against the re-aliased checkpoint: a saver that forgot o_proj.bias fails with
#     "parameter missing vs the base model" (34 tensors). --dtype BF16 because Megatron
#     saves bf16, not the FSDP2 arm's fp32.
#   * the weights must CHANGE between the two checkpoints (entropy_coeff/lr from the
#     base smoke), which the frozen MLP biases must not prevent.
#   * watch rollout_corr/* in the log: a wrong or missing o_proj bias in the vLLM sync
#     shows up as a trainer-vs-vLLM KL far above the Qwen arms' level from step 1.
#
# Usage:  bash smoke_test_openpangu_megatron_3+3.sh
# Env:    everything smoke_test_openpangu_3+3.sh accepts (MODEL_PATH, TRAIN_FILE, ...).
# =============================================================================

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh"}
export exp_name=${exp_name:-"SMOKE-openpangu7b-megatron-3+3"}
# Megatron hf_model saves are bf16 (the FSDP2 arm's are fp32).
export VERIFY_DTYPE=${VERIFY_DTYPE:-BF16}

exec bash "${HERE}/smoke_test_openpangu_3+3.sh" "$@"
