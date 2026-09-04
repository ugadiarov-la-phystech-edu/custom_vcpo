#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ARM_SCRIPT=${ARM_SCRIPT:-"${HERE}/grpo_novcpo_k=1_8gpu_dapo17k_5+3_resp8k_megatron_offload_openpangu7b_ppo-epochs=2_B33x1_is-pg.sh"}
export exp_name=${exp_name:-"SMOKE-openpangu7b-megatron-3+3"}
export VERIFY_DTYPE=${VERIFY_DTYPE:-BF16}

exec bash "${HERE}/smoke_test_openpangu_3+3.sh" "$@"
