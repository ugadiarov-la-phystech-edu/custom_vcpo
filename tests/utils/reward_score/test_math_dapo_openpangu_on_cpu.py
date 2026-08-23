# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""The answer-extraction contract the openPangu-7B arm relies on.

openPangu-Embedded-7B trains in slow-think mode: it emits a reasoning block delimited by
[unused16] ... [unused17] and then the answer. verl has no thinking-delimiter handling anywhere -
math_dapo takes solution_str[-300:] and the LAST (?i)Answer\\s*:\\s*(...) match - so the arm depends
on the DAPO prompt keeping the final answer on the last line. These tests pin that behaviour (and
its two failure modes) so a change to either the prompt or the scorer is a visible decision rather
than a silent reward shift.

Nothing here needs the model: the strings are what the reward manager sees after
decode(skip_special_tokens=True). The tokenizer half is
tests/models/test_openpangu_tokenizer_contract_on_cpu.py.
"""

import unittest

from verl.utils.reward_score import default_compute_score, math_dapo

THINK_OPEN, THINK_CLOSE = "[unused16]", "[unused17]"


def slow_think(reasoning: str, epilogue: str) -> str:
    """A response shaped like openPangu's slow-think output."""
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{epilogue}"


class TestOpenPanguAnswerExtraction(unittest.TestCase):
    def test_final_answer_wins_over_a_decoy_inside_the_thinking_block(self):
        """The decisive case: the CoT contains an abandoned "Answer: 7"."""
        response = slow_think(
            "Suppose the total is 7.\nAnswer: 7\nThat double-counts; redo it.\n" + "Filler step. " * 40,
            "Combining the cases gives 42.\nAnswer: 42",
        )
        result = math_dapo.compute_score(response, "42")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["pred"], "42")
        self.assertTrue(result["acc"])

    def test_the_delimiters_themselves_do_not_matter(self):
        """math_dapo is format-agnostic: the same content without delimiters scores identically."""
        with_markers = slow_think("work work", "Answer: 42")
        without = "work work\nAnswer: 42"
        self.assertEqual(math_dapo.compute_score(with_markers, "42"), math_dapo.compute_score(without, "42"))

    def test_wrong_final_answer_is_distinguishable_from_unparseable(self):
        wrong = math_dapo.compute_score(slow_think("work", "Answer: 41"), "42")
        self.assertEqual(wrong["score"], -1.0)
        self.assertFalse(wrong["acc"])
        self.assertEqual(wrong["pred"], "41", "a wrong answer must report what was extracted")

    def test_boxed_without_an_answer_line_is_invalid(self):
        """Expected, not a bug: this dispatch path requires the Answer: line, which is why the
        DAPO prompt asks for it."""
        result = math_dapo.compute_score(slow_think("work", "So \\boxed{42}"), "42")
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["pred"], "[INVALID]")

    def test_verbose_epilogue_after_the_answer_scores_invalid(self):
        """THE live exposure of this arm: >300 chars after the answer push it out of the window.

        Same failure on the Qwen arms; pinned here so the [INVALID] rate has a known cause.
        """
        response = slow_think("work", "Answer: 42\n" + "Let me re-check the arithmetic once more. " * 12)
        result = math_dapo.compute_score(response, "42")
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["pred"], "[INVALID]")

    def test_a_short_epilogue_still_scores(self):
        """The boundary: the window is 300 chars, so a brief sign-off is harmless."""
        response = slow_think("work", "Answer: 42\nThat completes the proof.")
        self.assertEqual(math_dapo.compute_score(response, "42")["score"], 1.0)

    def test_answer_line_spelling_variants(self):
        for epilogue in ("Answer: 42", "answer: 42", "ANSWER: 42", "Answer:42", "Answer:   42  "):
            with self.subTest(epilogue=epilogue):
                result = math_dapo.compute_score(slow_think("work", epilogue), "42")
                self.assertEqual(result["score"], 1.0, f"{epilogue!r} should parse")

    def test_truncated_mid_thinking_is_invalid_not_a_cot_search(self):
        """A response cut off by the length cap has no answer; it must not score from the CoT."""
        response = f"{THINK_OPEN}\nA promising route gives Answer: 42\n" + "still thinking. " * 30
        result = math_dapo.compute_score(response, "42")
        self.assertEqual(result["pred"], "[INVALID]")
        self.assertEqual(result["score"], -1.0)

    def test_both_validation_data_sources_dispatch(self):
        """dapo-math-17k and aime-2024 are data_source=math_dapo, aime-2025 is aime2025_dapo."""
        for data_source in ("math_dapo", "aime2025_dapo"):
            with self.subTest(data_source=data_source):
                result = default_compute_score(data_source, slow_think("work", "Answer: 42"), "42")
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["pred"], "42")


if __name__ == "__main__":
    unittest.main()
