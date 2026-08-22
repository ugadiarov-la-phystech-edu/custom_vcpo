# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""MetricsAggregator._get_aggregation_type picks an aggregation from the metric
NAME. It used to do so by substring, which silently mis-bucketed the replay
metrics: "replay/minibatch_new_ratio" contains "min" inside "minibatch", so the
reuse-rate panels reported the window's worst update instead of its mean — the
exact signal the reuse-driven-drift analysis reads.

Run: pytest recipe/fully_async_policy/unittest/test_metric_aggregation_type_on_cpu.py
"""

import pytest

from recipe.fully_async_policy.detach_utils import MetricsAggregator


def _agg(name: str) -> str:
    aggregator = MetricsAggregator.__new__(MetricsAggregator)
    aggregator.aggregation_rules = {}
    return aggregator._get_aggregation_type(name)


class TestReplayMetricsAreAveraged:
    @pytest.mark.parametrize(
        "name",
        [
            "replay/minibatch_new",
            "replay/minibatch_replayed",
            "replay/minibatch_new_ratio",
            "replay/minibatch_staleness_hist",
        ],
    )
    def test_minibatch_is_not_read_as_min(self, name):
        assert _agg(name) == "avg"


class TestWordMatchesStillWork:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("staleness/gap_min", "min"),
            ("staleness/minimum_gap", "min"),
            ("perf/max_memory", "max"),
            ("perf/maximum_memory", "max"),
            ("rollout_is_geom_mean", "avg"),
            ("fully_async/processing_time/avg", "avg"),
            ("tokens/total", "sum"),
            ("actor/grad_norm_sum", "sum"),
            ("timing_s/step", "time_sum"),
        ],
    )
    def test_keyword_words_are_honored(self, name, expected):
        assert _agg(name) == expected

    def test_unknown_names_default_to_avg(self):
        assert _agg("replay/buffer_size") == "avg"
        assert _agg("actor/entropy") == "avg"

    def test_explicit_rules_win_over_the_heuristic(self):
        aggregator = MetricsAggregator.__new__(MetricsAggregator)
        aggregator.aggregation_rules = {"max": ["replay/minibatch_new"]}
        assert aggregator._get_aggregation_type("replay/minibatch_new") == "max"
