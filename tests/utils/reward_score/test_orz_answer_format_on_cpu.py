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
"""The answer-extraction contract the Open-Reasoner-Zero-7B arm relies on.

ORZ's chat template mandates ``<think> ... </think> <answer> ... </answer>`` and the model obeys the
template rather than the DAPO prompt. Measured on the 30 deduplicated AIME-2024 problems (ORZ-7B,
T=1.0, top_p=1.0, 8192 tokens): 30/30 responses carried both ``<answer>`` and ``\\boxed{}``, 28-29/30
closed the tag on the same line as the content, and stock ``math_dapo`` scored 0/30 both with and
without the DAPO wrapper, while the tag-aware scorer scored 5/30 (~ORZ's published AIME-2024 pass@1).

These tests pin the shapes that measurement produced, so a change to either scorer becomes a visible
decision rather than a silent reward shift. Nothing here needs the model: the strings are what the
reward manager sees after ``decode(skip_special_tokens=True)``.
"""

import unittest

from recipe.fully_async_policy.reward import orz_tag_aware_math as orz
from verl.utils.reward_score import math_dapo

# The six response tails ORZ can emit, as enumerated before the probe was run.
SHAPES = {
    "boxed inside the answer block": "<think>work</think> <answer> \\boxed{42} </answer>",
    "Answer: on the same line as the tags": "<answer> Answer: 42 </answer>",
    "Answer: on its own line inside the block": "<answer>\nAnswer: 42\n</answer>",
    "Answer: flush against the tags": "<answer>Answer: 42</answer>",
    "bare value inside the answer block": "<answer> 42 </answer>",
    "plain DAPO format, no tags at all": "Reasoning.\nAnswer: 42",
}


def score(solution_str, ground_truth="42"):
    return orz.compute_score(data_source="math_dapo", solution_str=solution_str, ground_truth=ground_truth)


class TestOrzScorerAcceptsEveryShape(unittest.TestCase):
    def test_all_six_shapes_score_correct(self):
        for name, response in SHAPES.items():
            with self.subTest(shape=name):
                result = score(response)
                self.assertEqual(result["pred"], "42")
                self.assertTrue(result["acc"])
                self.assertEqual(result["score"], 1.0)

    def test_stock_math_dapo_fails_four_of_the_six(self):
        """The regression this arm exists to avoid. Kept as a test so that if math_dapo is ever
        fixed upstream, this fails loudly and the custom scorer can be reconsidered."""
        failing = {n for n, r in SHAPES.items() if math_dapo.compute_score(r, "42")["score"] <= 0}
        self.assertEqual(
            failing,
            {
                "boxed inside the answer block",
                "Answer: on the same line as the tags",
                "Answer: flush against the tags",
                "bare value inside the answer block",
            },
        )

    def test_same_line_closing_tag_never_leaks_into_pred(self):
        """math_dapo yields '42</answer>' here: a wrong-looking answer, worse than [INVALID],
        because it makes a parser bug read as bad maths."""
        self.assertEqual(math_dapo.compute_score("<answer> Answer: 42 </answer>", "42")["pred"], "42</answer>")
        for response in ("<answer> Answer: 42 </answer>", "<answer>Answer: 42</answer>", "<answer> 42 </answer>"):
            with self.subTest(response=response):
                self.assertNotIn("</answer>", score(response)["pred"])


class TestOrzScorerRejectsWhatItShould(unittest.TestCase):
    def test_a_wrong_answer_is_distinguishable_from_an_unparseable_one(self):
        wrong = score("<answer> \\boxed{41} </answer>")
        self.assertEqual(wrong["pred"], "41")
        self.assertFalse(wrong["acc"])
        unparseable = score("<think>I give up")
        self.assertEqual(unparseable["pred"], orz.INVALID)
        self.assertFalse(unparseable["acc"])

    def test_mid_reasoning_boxed_cannot_win(self):
        """The reward-hacking hole: once a real answer block exists, a \\boxed{} from the reasoning
        must not be picked up, in either direction."""
        self.assertFalse(score("<think>\\boxed{42} guess</think> <answer> \\boxed{41} </answer>")["acc"])
        self.assertTrue(score("<think>\\boxed{41} guess</think> <answer> \\boxed{42} </answer>")["acc"])

    def test_the_last_answer_block_wins(self):
        self.assertTrue(score("<answer> \\boxed{41} </answer>\nwait\n<answer> \\boxed{42} </answer>")["acc"])

    def test_a_truncated_answer_block_needs_a_real_answer_marker(self):
        """A rollout cut mid-block may still carry a complete \\boxed{}; if it does not, the block's
        raw text must NOT be accepted as the answer."""
        self.assertTrue(score("<think>w</think> <answer> \\boxed{42}")["acc"])
        self.assertEqual(score("<think>w</think> <answer> the answer is fourty")["pred"], orz.INVALID)
        self.assertEqual(score("<answer> \\boxed{42")["pred"], orz.INVALID)

    def test_prose_is_never_promoted_to_an_answer(self):
        """A closed but wordy block is reported invalid rather than normalized into noise, so
        [INVALID] rates stay a usable diagnostic."""
        self.assertEqual(score("<answer> " + "x" * 200 + " </answer>")["pred"], orz.INVALID)

    def test_no_tag_and_no_marker_degrades_exactly_like_math_dapo(self):
        for response in ("nothing useful here", "The answer might be 42 or maybe not"):
            with self.subTest(response=response):
                self.assertEqual(score(response)["pred"], math_dapo.compute_score(response, "42")["pred"])


class TestOrzScorerContract(unittest.TestCase):
    """The reward managers and the rollout dumps depend on the exact return shape."""

    def test_returns_the_same_dict_shape_as_math_dapo(self):
        mine = score("<answer> \\boxed{42} </answer>")
        stock = math_dapo.compute_score("Answer: 42", "42")
        self.assertEqual(set(mine), set(stock))
        self.assertIsInstance(mine["score"], float)
        self.assertIsInstance(mine["acc"], bool)  # not numpy.bool_: _dump_generations json-encodes it
        self.assertIsInstance(mine["pred"], str)

    def test_scores_are_the_plus_minus_one_math_dapo_uses(self):
        self.assertEqual(score("<answer> \\boxed{42} </answer>")["score"], 1.0)
        self.assertEqual(score("<answer> \\boxed{41} </answer>")["score"], -1.0)

    def test_accepts_the_reward_manager_call_signature(self):
        """Both managers call with these four keywords and nothing else."""
        self.assertTrue(
            orz.compute_score(
                data_source="math_dapo",
                solution_str="<answer> \\boxed{42} </answer>",
                ground_truth="42",
                extra_info={"rollout_reward_scores": {}},
            )["acc"]
        )

    def test_a_boxed_ground_truth_is_unwrapped_like_math_dapo_does(self):
        self.assertTrue(score("<answer> \\boxed{42} </answer>", ground_truth="\\boxed{42}")["acc"])

    def test_latex_ground_truths_normalize_on_both_sides(self):
        self.assertTrue(score("<answer> \\boxed{\\frac{1}{2}} </answer>", ground_truth="\\frac{1}{2}")["acc"])


class TestOrzEqualityTierTwo(unittest.TestCase):
    """ORZ's is_equiv tier (_strip_string + float compare), vendored for the 72k set's
    LaTeX ground truths -- these pairs FAIL normalized-string equality (tier 1)."""

    def test_slash_fraction_matches_latex_fraction(self):
        self.assertTrue(score("<answer> \\boxed{280/83} </answer>", ground_truth="\\frac{280}{83}")["acc"])

    def test_decimal_half_matches_latex_half(self):
        self.assertTrue(score("<answer> \\boxed{0.5} </answer>", ground_truth="\\frac{1}{2}")["acc"])

    def test_float_equal_decimals_match(self):
        self.assertTrue(score("<answer> \\boxed{2.0} </answer>", ground_truth="2")["acc"])

    def test_percent_sign_is_stripped(self):
        self.assertTrue(score("<answer> \\boxed{45\\%} </answer>", ground_truth="45")["acc"])

    def test_right_units_are_stripped(self):
        self.assertTrue(score("<answer> \\boxed{5\\text{ cm}} </answer>", ground_truth="5")["acc"])

    def test_left_right_and_spacing_normalize(self):
        self.assertTrue(
            score(
                "<answer> \\boxed{(3, \\frac{\\pi}{2})} </answer>",
                ground_truth="\\left( 3, \\frac{\\pi}{2} \\right)",
            )["acc"]
        )

    def test_wrong_fraction_stays_wrong(self):
        self.assertFalse(score("<answer> \\boxed{281/83} </answer>", ground_truth="\\frac{280}{83}")["acc"])

    def test_is_equiv_directly(self):
        self.assertTrue(orz._is_equiv_orz("\\frac{280}{83}", "280/83"))
        self.assertFalse(orz._is_equiv_orz("\\frac{280}{83}", "280/84"))


class TestOrzEqualityTierThree(unittest.TestCase):
    """The sympy parse_latex tier (symbolic/numeric equality). Skipped when no latex
    parser backend (antlr4/lark) is importable."""

    def setUp(self):
        if not orz._sympy_tier_enabled():
            self.skipTest("no sympy latex parser backend available")

    def test_unreduced_fraction_matches(self):
        # tier 1: strings differ; tier 2: no float parse, strings differ; tier 3: 2/4 == 1/2
        self.assertTrue(score("<answer> \\boxed{\\frac{2}{4}} </answer>", ground_truth="\\frac{1}{2}")["acc"])

    def test_radical_forms_match(self):
        self.assertTrue(score("<answer> \\boxed{\\sqrt{8}} </answer>", ground_truth="2\\sqrt{2}")["acc"])

    def test_non_equal_pairs_stay_non_equal(self):
        self.assertFalse(score("<answer> \\boxed{\\frac{1}{3}} </answer>", ground_truth="\\frac{1}{4}")["acc"])

    def test_length_guard_rejects_long_inputs(self):
        long_expr = "1+" * 100 + "1"
        self.assertFalse(orz._is_latex_equal(long_expr, long_expr))

    def test_flag_off_disables_the_tier(self):
        import os

        saved_state, saved_env = orz._sympy_tier_state, os.environ.get("ORZ_MATH_SYMPY_TIER")
        try:
            orz._sympy_tier_state = None
            os.environ["ORZ_MATH_SYMPY_TIER"] = "0"
            self.assertFalse(orz._is_latex_equal("\\frac{2}{4}", "\\frac{1}{2}"))
        finally:
            orz._sympy_tier_state = saved_state
            if saved_env is None:
                os.environ.pop("ORZ_MATH_SYMPY_TIER", None)
            else:
                os.environ["ORZ_MATH_SYMPY_TIER"] = saved_env


class TestOrzEqualityEdgeCases(unittest.TestCase):
    def test_empty_ground_truth_never_matches(self):
        self.assertFalse(score("<answer> \\boxed{} </answer>", ground_truth="")["acc"])
        self.assertFalse(score("<answer> 42 </answer>", ground_truth=" ")["acc"])

    def test_unparseable_response_never_reaches_the_tiers(self):
        self.assertFalse(score("no tags, no markers, just prose", ground_truth="\\frac{1}{2}")["acc"])

    def test_integer_fast_path_unchanged(self):
        # dapo17k/AceReason regression: integers short-circuit at tier 1.
        self.assertTrue(score("<answer> \\boxed{42} </answer>", ground_truth="42")["acc"])
        self.assertFalse(score("<answer> \\boxed{43} </answer>", ground_truth="42")["acc"])

    def test_verbal_ground_truth_exact_match_still_possible(self):
        gt = "Jenna will win"
        self.assertTrue(score(f"<answer> {gt} </answer>", ground_truth=gt)["acc"])


if __name__ == "__main__":
    unittest.main()
