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

"""Rebuild validation parquets with ORZ-original prompts (2026-08-30 decision).

AIME-2024/2025: row-for-row transformation of the existing DAPO-wrapper
validation parquets -- the bare problem (extra_info.raw_problem, cross-checked
against the wrapper-stripped prompt) is re-wrapped in ORZ's own inner
instruction (verbatim from Open-Reasoner-Zero playground/zero_setting_base.py;
the tokenizer chat template supplies the outer conversation layer,
byte-verified against their jinja). Row count/x32 duplication, ground truths,
ability and extra_info are preserved.

math500: built from ORZ's OWN eval file (data/eval_data/math500.json in the
reference repo: [{"prompt": [{from: user, value: <bare problem>}],
"final_answer": <gt>}]) -- the exact 500 problems and ground truths ORZ
evaluated, x1 like their protocol, same ORZ instruction wrapper.

data_source gets a fresh *_orz stamp everywhere so the val-core metric curves
can never be conflated with the DAPO-wrapper protocol of earlier runs.

Usage (remote_mary):
    PYTHONPATH=<repo> python convert_val_parquets_to_orz_prompts.py \
        --root /home/jovyan/datasets/math_datasets \
        --out-dir /home/jovyan/datasets/math_datasets/orz \
        --math500-json /path/to/math500.json
"""

import argparse
import json
import os

import pandas as pd

ORZ_INNER_INSTRUCTION = (
    "You must put your answer inside <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>. And your final answer will be extracted "
    "automatically by the \\boxed{{}} tag.\nThis is the problem:\n{problem}"
)

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

# (relative source path, output basename, new data_source stamp)
PARQUET_TARGETS = [
    ("dapo/aime-2024.parquet", "aime-2024-orz.parquet", "aime2024_orz"),
    ("dapo/aime-2025.parquet", "aime-2025-orz.parquet", "aime2025_orz"),
]
MATH500_STAMP = "math500_orz"
MATH500_OUT = "math500-orz.parquet"


def bare_problem(row) -> str:
    extra = row["extra_info"]
    raw = None
    if extra is not None:
        try:
            raw = dict(extra).get("raw_problem")
        except (TypeError, ValueError):
            raw = None
    if raw:
        return str(raw)
    content = row["prompt"][0]["content"]
    assert content.startswith(DAPO_PREFIX) and content.endswith(DAPO_SUFFIX), (
        f"cannot recover bare problem: unexpected wrapper in {content[:120]!r}..."
    )
    return content[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="math_datasets root holding the source parquets")
    parser.add_argument("--out-dir", required=True, help="output directory for the *-orz parquets")
    parser.add_argument("--math500-json", required=True, help="ORZ repo data/eval_data/math500.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for rel_src, out_name, stamp in PARQUET_TARGETS:
        src = os.path.join(args.root, rel_src)
        df = pd.read_parquet(src)
        problems = df.apply(bare_problem, axis=1)
        # Cross-check: where raw_problem exists, the wrapper-strip must agree.
        checked = 0
        for i in range(len(df)):
            content = df.iloc[i]["prompt"][0]["content"]
            if content.startswith(DAPO_PREFIX) and content.endswith(DAPO_SUFFIX):
                stripped = content[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)]
                assert stripped == problems.iloc[i], f"row {i}: raw_problem != wrapper-stripped prompt"
                checked += 1
        out = df.copy()
        out["data_source"] = stamp
        out["prompt"] = [
            [{"role": "user", "content": ORZ_INNER_INSTRUCTION.format(problem=p)}] for p in problems
        ]
        dst = os.path.join(args.out_dir, out_name)
        out.to_parquet(dst, index=False)
        print(
            f"{rel_src}: {len(out)} rows ({problems.nunique()} unique problems), "
            f"wrapper cross-checked on {checked} rows, stamp={stamp} -> {dst}"
        )

    # math500 from ORZ's own eval JSON (x1, exactly their 500 problems + gts).
    data = json.load(open(args.math500_json))
    rows = []
    for i, item in enumerate(data):
        problem = str(item["prompt"][0]["value"])
        gt = str(item["final_answer"])
        assert problem.strip() and gt.strip(), f"math500 row {i}: empty problem or gt"
        rows.append(
            {
                "data_source": MATH500_STAMP,
                "prompt": [{"role": "user", "content": ORZ_INNER_INSTRUCTION.format(problem=problem)}],
                "ability": "MATH",
                "reward_model": {"ground_truth": gt, "style": "rule-lighteval/MATH_v2"},
                "extra_info": {"index": i, "split": "test"},
            }
        )
    dst = os.path.join(args.out_dir, MATH500_OUT)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    latex = sum(1 for r in rows if "\\" in r["reward_model"]["ground_truth"])
    print(
        f"{os.path.basename(args.math500_json)}: {len(rows)} rows (x1, ORZ's own problems; "
        f"{latex} latex gts), stamp={MATH500_STAMP} -> {dst}"
    )


if __name__ == "__main__":
    main()
