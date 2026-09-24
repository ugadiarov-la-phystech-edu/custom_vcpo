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
"""Unit tests for the token/logprob alignment in the partial-generation vLLM
server (recipe/fully_async_policy/vllm_rollout/vllm_async_server.py):

- align_token_logprobs: equal lengths, trailing extra logprobs (the openPangu
  2026-09-06 failure shape), trailing tokens without logprobs, empties, None,
  a sampled token missing from its dict, no input mutation
- vLLMHttpServerForPartial.generate_for_partial with a stubbed _generate_step:
  normal finish, mismatch on finish and on cancel (no exception, counters,
  rate-limited warning), paused fast path, cancel before the first token,
  dict cleanup on failure

Run: pytest recipe/fully_async_policy/unittest/test_vllm_partial_logprob_alignment_on_cpu.py
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from recipe.fully_async_policy.vllm_rollout import vllm_async_server as srv  # noqa: E402
from recipe.fully_async_policy.vllm_rollout.vllm_async_server import (  # noqa: E402
    align_token_logprobs,
    vLLMHttpServerForPartial,
)

ServerCls = (
    vLLMHttpServerForPartial.__ray_metadata__.modified_class
    if hasattr(vLLMHttpServerForPartial, "__ray_metadata__")
    else vLLMHttpServerForPartial
)


def _lp(token_id, value, extra=None):
    """One position's logprob dict: the sampled token plus optional other candidates."""
    d = {token_id: SimpleNamespace(logprob=value)}
    for tid, v in (extra or {}).items():
        d[tid] = SimpleNamespace(logprob=v)
    return d


def _completion(token_ids, logprobs, finish_reason="stop"):
    return SimpleNamespace(token_ids=list(token_ids), logprobs=logprobs, finish_reason=finish_reason)


def _request_output(token_ids, logprobs, finish_reason="stop"):
    return SimpleNamespace(outputs=[_completion(token_ids, logprobs, finish_reason)])


# ---------------------------------------------------------------- helper


def test_align_equal_lengths_picks_the_sampled_token_from_multi_entry_dicts():
    ids = [5, 7, 9]
    lps = [_lp(5, -0.1, {1: -3.0}), _lp(7, -0.2, {5: -2.0, 2: -4.0}), _lp(9, -0.3)]
    out_ids, out_lp, extra, missing = align_token_logprobs(ids, lps)
    assert out_ids == ids
    assert out_lp == pytest.approx([-0.1, -0.2, -0.3])
    assert (extra, missing) == (0, 0)


def test_align_ignores_trailing_extra_logprobs_the_production_shape():
    ids = [5, 7]
    lps = [_lp(5, -0.1), _lp(7, -0.2), _lp(11, -0.9)]  # one logprob beyond the last token
    out_ids, out_lp, extra, missing = align_token_logprobs(ids, lps)
    assert out_ids == [5, 7]
    assert out_lp == pytest.approx([-0.1, -0.2])
    assert (extra, missing) == (1, 0)
    lps3 = lps + [_lp(12, -0.5), _lp(13, -0.4)]
    assert align_token_logprobs(ids, lps3)[2] == 3


def test_align_drops_trailing_tokens_without_logprobs():
    ids = [5, 7, 9, 11]
    lps = [_lp(5, -0.1), _lp(7, -0.2)]
    out_ids, out_lp, extra, missing = align_token_logprobs(ids, lps)
    assert out_ids == [5, 7]
    assert len(out_ids) == len(out_lp) == 2
    assert (extra, missing) == (0, 2)


def test_align_handles_empties():
    assert align_token_logprobs([], []) == ([], [], 0, 0)
    assert align_token_logprobs([], [_lp(1, -1.0)]) == ([], [], 1, 0)
    assert align_token_logprobs([3], []) == ([], [], 0, 1)


def test_align_rejects_none_logprobs_and_a_missing_sampled_token():
    with pytest.raises(ValueError, match="logprobs=1"):
        align_token_logprobs([1, 2], None)
    with pytest.raises(KeyError):
        align_token_logprobs([1, 2], [_lp(1, -0.1), _lp(99, -0.2)])  # position 1 lacks token 2


def test_align_does_not_mutate_inputs_and_returns_fresh_lists():
    ids = [5, 7]
    lps = [_lp(5, -0.1), _lp(7, -0.2), _lp(8, -0.3)]
    ids_copy, n_lps = list(ids), len(lps)
    out_ids, _, _, _ = align_token_logprobs(ids, lps)
    out_ids.append(0)
    assert ids == ids_copy and len(lps) == n_lps


# ---------------------------------------------------------------- server


def _make_server(script):
    """A server whose _generate_step is scripted: script(request_id) is an async
    callable that fills self.req_output[request_id] (and may block forever)."""
    s = ServerCls.__new__(ServerCls)
    s.paused = False
    s.lock = asyncio.Lock()
    s.cancel_event = {}
    s.req_output = {}
    s.logprob_mismatch_events = 0
    s.logprob_mismatch_extra = 0
    s.logprob_mismatch_missing = 0

    async def _generate_step(prompt_ids, sampling_params, request_id, image_data=None):
        await script(s, request_id)

    s._generate_step = _generate_step
    return s


def _finishing(output):
    async def script(s, rid):
        s.req_output[rid] = output

    return script


def _blocking(output=None):
    """Sets the (partial) output, then waits until cancelled."""

    async def script(s, rid):
        if output is not None:
            s.req_output[rid] = output
        await asyncio.Event().wait()

    return script


def _run(coro):
    return asyncio.run(coro)


def test_server_normal_finish_returns_aligned_lists_and_cleans_up():
    s = _make_server(_finishing(_request_output([5, 7, 9], [_lp(5, -0.1), _lp(7, -0.2), _lp(9, -0.3)])))
    ids, lps, is_cancel = _run(s.generate_for_partial([1, 2], {}, "r1"))
    assert ids == [5, 7, 9]
    assert lps == pytest.approx([-0.1, -0.2, -0.3])
    assert is_cancel is False
    assert s.cancel_event == {} and s.req_output == {}
    assert (s.logprob_mismatch_events, s.logprob_mismatch_extra, s.logprob_mismatch_missing) == (0, 0, 0)


def test_server_mismatch_on_finish_no_longer_raises_and_is_counted(caplog):
    out = _request_output([5, 7], [_lp(5, -0.1), _lp(7, -0.2), _lp(11, -0.9)], finish_reason="stop")
    s = _make_server(_finishing(out))
    with caplog.at_level(logging.WARNING, logger=srv.logger.name):
        ids, lps, is_cancel = _run(s.generate_for_partial([1], {}, "r-mismatch"))
    assert ids == [5, 7] and lps == pytest.approx([-0.1, -0.2]) and is_cancel is False
    assert (s.logprob_mismatch_events, s.logprob_mismatch_extra, s.logprob_mismatch_missing) == (1, 1, 0)
    warnings = [r for r in caplog.records if "length mismatch" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "r-mismatch" in msg and "extra_logprobs=1" in msg and "is_cancel=False" in msg
    assert "finish_reason=stop" in msg and "len(token_ids)=2" in msg and "len(logprobs)=3" in msg
    assert s.cancel_event == {} and s.req_output == {}


def test_server_cancel_returns_partial_output_with_is_cancel():
    partial = _request_output([5, 7], [_lp(5, -0.1), _lp(7, -0.2)], finish_reason=None)
    s = _make_server(_blocking(partial))

    async def run():
        task = asyncio.create_task(s.generate_for_partial([1], {}, "r-cancel"))
        for _ in range(20):  # let the request register and set its partial output
            await asyncio.sleep(0)
        assert "r-cancel" in s.cancel_event
        await s.cancel()
        return await task

    ids, lps, is_cancel = _run(run())
    assert ids == [5, 7] and lps == pytest.approx([-0.1, -0.2])
    assert is_cancel is True
    assert s.paused is True
    assert s.cancel_event == {} and s.req_output == {}


def test_server_mismatch_on_cancel_is_aligned_and_flagged_cancelled(caplog):
    partial = _request_output([5, 7], [_lp(5, -0.1), _lp(7, -0.2), _lp(11, -0.9)], finish_reason=None)
    s = _make_server(_blocking(partial))

    async def run():
        task = asyncio.create_task(s.generate_for_partial([1], {}, "r-cm"))
        for _ in range(20):
            await asyncio.sleep(0)
        await s.cancel()
        return await task

    with caplog.at_level(logging.WARNING, logger=srv.logger.name):
        ids, lps, is_cancel = _run(run())
    assert ids == [5, 7] and lps == pytest.approx([-0.1, -0.2]) and is_cancel is True
    assert s.logprob_mismatch_events == 1 and s.logprob_mismatch_extra == 1
    msg = [r.getMessage() for r in caplog.records if "length mismatch" in r.getMessage()][0]
    assert "is_cancel=True" in msg and "finish_reason=None" in msg


def test_server_paused_fast_path_touches_nothing():
    s = _make_server(_finishing(_request_output([5], [_lp(5, -0.1)])))
    s.paused = True
    assert _run(s.generate_for_partial([1], {}, "r-paused")) == ([], [], True)
    assert s.cancel_event == {} and s.req_output == {}


def test_server_cancel_before_first_token_returns_empty_cancelled():
    s = _make_server(_blocking(None))  # never produces an output

    async def run():
        task = asyncio.create_task(s.generate_for_partial([1], {}, "r-early"))
        for _ in range(20):
            await asyncio.sleep(0)
        await s.cancel()
        return await task

    assert _run(run()) == ([], [], True)
    assert s.cancel_event == {} and s.req_output == {}


def test_server_failure_in_alignment_still_cleans_the_request_dicts():
    bad = _request_output([5, 7], [_lp(5, -0.1), _lp(99, -0.2)])  # sampled token 7 absent -> KeyError
    s = _make_server(_finishing(bad))
    with pytest.raises(KeyError):
        _run(s.generate_for_partial([1], {}, "r-bad"))
    assert s.cancel_event == {} and s.req_output == {}
    assert s.logprob_mismatch_events == 0


def test_server_mismatch_warnings_are_rate_limited(caplog):
    out = _request_output([5], [_lp(5, -0.1), _lp(6, -0.2)])
    s = _make_server(_finishing(out))
    n_requests = srv._MISMATCH_WARN_FIRST + 15
    with caplog.at_level(logging.WARNING, logger=srv.logger.name):
        for i in range(n_requests):
            _run(s.generate_for_partial([1], {}, f"r{i}"))
    warnings = [r for r in caplog.records if "length mismatch" in r.getMessage()]
    assert len(warnings) == srv._MISMATCH_WARN_FIRST  # 25 < 100: no periodic warning yet
    assert s.logprob_mismatch_events == n_requests
    assert s.logprob_mismatch_extra == n_requests
    # the periodic rule: event number _MISMATCH_WARN_EVERY warns again
    s.logprob_mismatch_events = srv._MISMATCH_WARN_EVERY - 1
    with caplog.at_level(logging.WARNING, logger=srv.logger.name):
        caplog.clear()
        _run(s.generate_for_partial([1], {}, "r-periodic"))
    assert len([r for r in caplog.records if "length mismatch" in r.getMessage()]) == 1


def test_server_init_declares_the_mismatch_counters():
    src = open(srv.__file__).read()
    init_body = src.split("def __init__")[1].split("async def _generate_step")[0]
    for name in ("logprob_mismatch_events", "logprob_mismatch_extra", "logprob_mismatch_missing"):
        assert f"self.{name} = 0" in init_body
