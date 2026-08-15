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
"""
Preprocess the HuggingFaceH4/MATH-500 dataset to parquet format (validation).

Prompts use the SAME template as the DAPO-Math-17k training set and the
AIME-24 validation set ("Answer:" on its own line), so validation measures
math ability rather than format transfer, and answers are scored by the same
math_dapo (Minerva-style "Answer:" extraction) scorer. The data_source stamp
"math500_dapo" (MATH-500 in the DAPO answer format) keeps its metrics separate:
val-core/math500_dapo/acc/mean@1.
"""

import argparse
import os

import datasets

DAPO_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math500", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    print("Loading the HuggingFaceH4/MATH-500 dataset...", flush=True)
    dataset = datasets.load_dataset(args.local_dataset_path or "HuggingFaceH4/MATH-500")

    test_dataset = dataset["test"]  # MATH-500 ships a single 500-problem test split

    def process_fn(example, idx):
        question = DAPO_PROMPT_TEMPLATE.format(problem=example.pop("problem"))
        # MATH-500 carries the extracted final answer in the "answer" column
        answer = example.pop("answer")
        return {
            "data_source": "math500_dapo",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": idx},
        }

    test_dataset = test_dataset.map(function=process_fn, with_indices=True, remove_columns=test_dataset.column_names)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "math500.parquet")
    test_dataset.to_parquet(out_path)
    print(f"Wrote {len(test_dataset)} rows to {out_path}")
