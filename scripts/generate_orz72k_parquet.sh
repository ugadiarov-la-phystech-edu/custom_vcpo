#!/usr/bin/env bash
# Generate orz-math-72k.parquet from Open-Reasoner-Zero/orz_math_72k_collection_extended
# in the dapo-math-17k schema, with the configuration decided 2026-08-30:
#   * prompts wrapped in ORZ's own inner instruction (boxed-inside-<answer>), NOT the
#     DAPO wrapper — the chat template supplies the outer conversation layer;
#   * filters: degenerate problems (<20 chars), empty/normalizing-empty ground truths,
#     normalized ground truths >40 chars, exact-duplicate problems (~24k in the raw set);
#   * decontamination against the ACTUAL validation sets only (aime-2024 + aime-2025);
#     aime-2026 deliberately excluded — its one overlapping problem stays in training.
#
# Run from the repo root (verl must be importable). Defaults target the cloud.ru
# checkouts (remote_mary/remote_h100 shared FS); override via env:
#   DATASETS_ROOT=/path/to/math_datasets OUT=/path/out.parquet bash scripts/generate_orz72k_parquet.sh
#
# Expected report (raw 72,444): kept 47,981; duplicates ~23,888; contaminated 0;
# prompt tokens p99 < 500, none over 2048. Review the printed filter report after
# every regeneration.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATASETS_ROOT=${DATASETS_ROOT:-/home/jovyan/datasets/math_datasets}
OUT=${OUT:-${DATASETS_ROOT}/orz/orz-math-72k.parquet}
TOKENIZER=${TOKENIZER:-Open-Reasoner-Zero/Open-Reasoner-Zero-7B}
# Optional: point at a local copy of the JSON instead of downloading from HF.
ORZ_JSON=${ORZ_JSON:-}

json_arg=()
if [ -n "${ORZ_JSON}" ]; then
    json_arg=(--json "${ORZ_JSON}")
fi

PYTHONPATH="${REPO_ROOT}" python "${REPO_ROOT}/scripts/convert_orz72k_to_verl_parquet.py" \
    --out "${OUT}" \
    --val-parquets "${DATASETS_ROOT}/dapo/aime-2024.parquet" "${DATASETS_ROOT}/dapo/aime-2025.parquet" \
    --tokenizer "${TOKENIZER}" \
    "${json_arg[@]}" "$@"
