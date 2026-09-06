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
import asyncio
import logging
from typing import Any, Optional, Sequence

import ray
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _qwen2_5_vl_dedup_image_tokens,
    vLLMHttpServerBase,
    vLLMReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Mismatch warnings: every one of the first N events, then every Mth.
_MISMATCH_WARN_FIRST = 10
_MISMATCH_WARN_EVERY = 100


def align_token_logprobs(
    token_ids: Sequence[int], logprobs: Optional[Sequence[Any]]
) -> tuple[list[int], list[float], int, int]:
    """Pair a request's sampled token ids with their per-position logprobs.

    ``logprobs[i]`` is vLLM's dict ``{token_id: Logprob}`` for position ``i``
    (``logprobs=1`` is requested, but the dict may hold several candidates —
    the sampled token is always among them). Returns
    ``(token_ids, log_probs, extra, missing)`` where the two lists have equal
    length ``n = min(len(token_ids), len(logprobs))``, ``extra`` is the number
    of trailing logprob entries without a token (ignored) and ``missing`` the
    number of trailing tokens without a logprob (dropped, so the ids and the
    logprobs the IS correction consumes stay aligned).

    Why the lengths can differ: in cumulative streaming mode the
    ``CompletionOutput`` vLLM yields aliases the output processor's live
    ``token_ids`` / ``logprobs`` lists, which the engine keeps extending. The
    openPangu-7B replay run of 2026-09-06 hit ``len(logprobs) == len(token_ids)
    + 1`` on requests cut short at a weight-sync pause and lost the whole group
    to an IndexError; the ORZ / Qwen runs on the same code never did. Aligning
    keeps the group; the caller logs the shape so the trigger can be pinned down.

    ``logprobs is None`` means logprobs were not requested — a contract break,
    raised. A sampled token absent from its own position's dict is a genuine
    engine bug and stays a ``KeyError``.
    """
    if logprobs is None:
        raise ValueError("generate_for_partial requires per-token logprobs (sampling_params.logprobs=1), got None")
    n = min(len(token_ids), len(logprobs))
    log_probs = [float(logprobs[i][token_ids[i]].logprob) for i in range(n)]
    return list(token_ids[:n]), log_probs, len(logprobs) - n, len(token_ids) - n


@ray.remote(num_cpus=1)
class vLLMHttpServerForPartial(vLLMHttpServerBase):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)

        # for cancel LLMServer
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.req_output: dict[str, Optional[RequestOutput]] = {}
        # Token/logprob length-mismatch accounting (see align_token_logprobs):
        # events = requests whose lists disagreed, extra/missing = entries
        # ignored / tokens dropped in total.
        self.logprob_mismatch_events = 0
        self.logprob_mismatch_extra = 0
        self.logprob_mismatch_missing = 0

    async def _generate_step(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        max_tokens = self.config.max_model_len - len(prompt_ids)
        sampling_params["logprobs"] = 1
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        async for output in generator:
            self.req_output[request_id] = output
        assert self.req_output[request_id] is not None

    async def generate_for_partial(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> tuple[list[Any], list[Any], bool] | tuple[Sequence[int], list[float], Any]:
        async with self.lock:
            if self.paused:
                # After cancel, all tasks will return directly and wait for the next submission
                return [], [], True
            self.req_output[request_id]: Optional[RequestOutput] = None
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_ids, sampling_params, request_id, image_data)
            )

        done, pend = await asyncio.wait([generation_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            await task

        for task in pend:
            task.cancel()

        async with self.lock:
            try:
                output = self.req_output[request_id]
                if output is None:
                    return [], [], True
                completion = output.outputs[0]
                is_cancel = generation_handle not in done
                # The completion's lists are vLLM's live cumulative lists and
                # can disagree in length; align instead of indexing across
                # (a mismatch used to raise IndexError and lose the group).
                token_ids, log_probs, extra, missing = align_token_logprobs(completion.token_ids, completion.logprobs)
                if extra or missing:
                    self._note_logprob_mismatch(request_id, completion, is_cancel, token_ids, extra, missing)
            finally:
                self.cancel_event.pop(request_id, None)
                self.req_output.pop(request_id, None)
        return token_ids, log_probs, is_cancel

    def _note_logprob_mismatch(self, request_id, completion, is_cancel, token_ids, extra, missing) -> None:
        """Count a token/logprob length mismatch and log its shape, rate-limited
        (every one of the first _MISMATCH_WARN_FIRST events, then every
        _MISMATCH_WARN_EVERY-th): the fields are what a root-cause pass needs."""
        self.logprob_mismatch_events += 1
        self.logprob_mismatch_extra += extra
        self.logprob_mismatch_missing += missing
        n = self.logprob_mismatch_events
        if n <= _MISMATCH_WARN_FIRST or n % _MISMATCH_WARN_EVERY == 0:
            logger.warning(
                "[vLLMHttpServerForPartial] token/logprob length mismatch #%d on request %s: "
                "extra_logprobs=%d missing_logprobs=%d is_cancel=%s finish_reason=%s "
                "len(token_ids)=%d len(logprobs)=%d last_token_ids=%s (aligned to %d tokens)",
                n,
                request_id,
                extra,
                missing,
                is_cancel,
                getattr(completion, "finish_reason", None),
                len(completion.token_ids),
                len(completion.logprobs) if completion.logprobs is not None else -1,
                list(completion.token_ids[-3:]),
                len(token_ids),
            )

    async def cancel(self):
        async with self.lock:
            self.paused = True
            for request_id in self.cancel_event:
                self.cancel_event[request_id].set()

    async def resume(self):
        async with self.lock:
            self.paused = False


class FullyAsyncvLLMReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = vLLMHttpServerForPartial

    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])
