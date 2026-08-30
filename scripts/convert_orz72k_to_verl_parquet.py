# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Open-Reasoner-Zero/orz_math_72k_collection_extended to the dapo-math-17k
parquet schema used by the fully-async training scripts.

Schema (matched byte-for-byte against dapo-math-17k.parquet on remote_mary):
    source_prompt: str      -- the bare problem statement
    solution: str           -- the ground-truth answer
    data_source: "math_dapo"
    prompt: [{role: user, content: <ORZ inner instruction + problem>}]
    ability: "MATH"
    reward_model: {ground_truth: str, style: "rule-lighteval/MATH_v2"}
    extra_info: {index: uuid5 of the problem text}

Prompt wrapper: ORZ's OWN inner instruction (verbatim from the reference repo,
playground/zero_setting_base.py) rather than the DAPO wrapper -- ORZ-7B was
trained being told to put \\boxed{} inside <answer> tags; the outer conversation
template is supplied by the tokenizer's chat template at collation time and must
NOT be baked in here.

Filters (each counted in the report):
    * empty ground truth, or one that normalizes to empty;
    * normalized ground truth longer than --max-gt-chars (default 40): long
      verbal answers can never exact-match -- pure generation waste;
    * exact-duplicate problems (whitespace-collapsed);
    * decontamination against the AIME validation parquets (normalized
      problem-text containment in either direction); hits are printed for
      review and dropped.

Budget note for launches: two full epochs of the ~70k kept prompts is ~140k --
override total_rollout_steps accordingly (the acereason script defaults to 82182).

Usage (remote_mary; decontaminate against the ACTUAL validation sets only —
aime-2026 is deliberately excluded per 2026-08-30 decision, its one overlapping
problem stays in the training set):
    python convert_orz72k_to_verl_parquet.py \
        --out /home/jovyan/datasets/math_datasets/orz/orz-math-72k.parquet \
        --val-parquets /home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet \
                       /home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet \
        --tokenizer Open-Reasoner-Zero/Open-Reasoner-Zero-7B
"""

import argparse
import json
import os
import re
import uuid

import pandas as pd

# Verbatim from Open-Reasoner-Zero playground/zero_setting_base.py
# (prompt_instruction_template_jinja), with the {{prompt}} slot filled per row.
ORZ_INNER_INSTRUCTION = (
    "You must put your answer inside <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>. And your final answer will be extracted "
    "automatically by the \\boxed{{}} tag.\nThis is the problem:\n{problem}"
)

DATA_SOURCE = "math_dapo"
ABILITY = "MATH"
REWARD_STYLE = "rule-lighteval/MATH_v2"
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "orz_math_72k_collection_extended")


def _norm_text(s: str) -> str:
    """Whitespace/case-collapsed text for dedup and contamination checks."""
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", default=None, help="path to orz_math_72k_collection_extended.json (default: download from HF)"
    )
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--val-parquets", nargs="*", default=[], help="validation parquets to decontaminate against")
    parser.add_argument(
        "--max-gt-chars", type=int, default=40, help="drop rows whose normalized gt is longer than this"
    )
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer for the prompt token-length report (optional)")
    args = parser.parse_args()

    json_path = args.json
    if json_path is None:
        from huggingface_hub import hf_hub_download

        json_path = hf_hub_download(
            "Open-Reasoner-Zero/orz_math_72k_collection_extended",
            "orz_math_72k_collection_extended.json",
            repo_type="dataset",
        )
    data = json.load(open(json_path))
    print(f"loaded {len(data)} items from {json_path}")

    from verl.utils.reward_score.math_dapo import normalize_final_answer

    # Validation problems for decontamination.
    val_norms = []  # (norm_text, tag, raw)
    for vp in args.val_parquets:
        vdf = pd.read_parquet(vp)
        col = "source_prompt" if "source_prompt" in vdf.columns else "prompt"
        for raw in vdf[col]:
            text = raw if isinstance(raw, str) else " ".join(str(m.get("content", "")) for m in raw)
            val_norms.append((_norm_text(text), os.path.basename(vp), text))
    print(f"decontamination reference: {len(val_norms)} validation rows from {len(args.val_parquets)} files")

    # Below this many normalized characters a "problem" is junk (empty or a bare
    # contest number like "12.") and, separately, too short for the containment
    # check to be meaningful (an empty key is contained in EVERYTHING).
    min_problem_chars = 20

    rows = []
    seen = set()
    dropped = {
        "schema": 0,
        "degenerate_problem": 0,
        "empty_gt": 0,
        "normalizes_empty": 0,
        "long_gt": 0,
        "duplicate": 0,
        "contaminated": 0,
    }
    contamination_hits = []
    for item in data:
        try:
            problem = str(item[0]["value"])
            gt = str(item[1]["ground_truth"]["value"])
        except (KeyError, IndexError, TypeError):
            dropped["schema"] += 1
            continue
        if len(_norm_text(problem)) < min_problem_chars:
            dropped["degenerate_problem"] += 1
            continue
        if not gt.strip():
            dropped["empty_gt"] += 1
            continue
        gt_norm = normalize_final_answer(gt)
        if not gt_norm:
            dropped["normalizes_empty"] += 1
            continue
        if len(gt_norm) > args.max_gt_chars:
            dropped["long_gt"] += 1
            continue
        key = _norm_text(problem)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        hit = next((v for v in val_norms if key in v[0] or v[0] in key), None)
        if hit is not None:
            dropped["contaminated"] += 1
            contamination_hits.append((problem[:120], hit[1], hit[2][:120]))
            continue
        seen.add(key)
        rows.append(
            {
                "source_prompt": problem,
                "solution": gt,
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": ORZ_INNER_INSTRUCTION.format(problem=problem)}],
                "ability": ABILITY,
                "reward_model": {"ground_truth": gt, "style": REWARD_STYLE},
                "extra_info": {"index": str(uuid.uuid5(UUID_NAMESPACE, problem))},
            }
        )

    print("\n=== filter report ===")
    for k, v in dropped.items():
        print(f"dropped[{k}]: {v}")
    print(f"kept: {len(rows)} of {len(data)}")
    if contamination_hits:
        print("\n=== contamination hits (dropped; review) ===")
        for prob, vfile, vtext in contamination_hits:
            print(f"[{vfile}]\n  train: {prob}\n  val:   {vtext}")

    gts = [r["solution"] for r in rows]
    num = sum(1 for g in gts if re.fullmatch(r"-?\d+(\.\d+)?", g.strip()))
    latex = sum(1 for g in gts if "\\" in g)
    print(f"\nkept gt types: plain numeric {num} ({num / len(gts):.1%}), latex {latex} ({latex / len(gts):.1%})")

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        import random

        random.seed(0)
        sample = random.sample(rows, min(2000, len(rows)))
        tlens = sorted(
            len(tok.apply_chat_template([dict(m) for m in r["prompt"]], add_generation_prompt=True)) for r in sample
        )
        n = len(tlens)
        print(
            f"prompt tokens (chat template applied, n={n}): p50={tlens[n // 2]} "
            f"p95={tlens[int(n * 0.95)]} p99={tlens[int(n * 0.99)]} max={tlens[-1]} "
            f">2048: {sum(1 for t in tlens if t > 2048)}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
