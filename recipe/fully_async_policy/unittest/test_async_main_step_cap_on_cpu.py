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
"""When the trainer stops at trainer.total_training_steps it returns before the rollouter, which
would otherwise keep generating, self-pause on its staleness quota once nobody drains the queue,
and never return — the job would hang. FullyAsyncTaskRunner._run_training_loop therefore cancels
the pending rollouter future when the trainer's future is the one that completed. These tests
cover the decision helper, the loop's use of it, and the Ray mechanism it relies on (ray.cancel
on a running async-actor task).

Run: pytest recipe/fully_async_policy/unittest/test_async_main_step_cap_on_cpu.py
"""

import asyncio
import inspect
import unittest

import pytest
import ray

from recipe.fully_async_policy import fully_async_main as main_mod
from recipe.fully_async_policy.fully_async_main import remaining_after_completion


class TestRemainingAfterCompletion(unittest.TestCase):
    def test_trainer_first_cancels_everything_pending(self):
        to_cancel, to_wait = remaining_after_completion("trainer", "trainer", ["rollouter"])
        self.assertEqual((to_cancel, to_wait), (["rollouter"], []))

    def test_rollouter_first_keeps_waiting_for_the_trainer(self):
        # the normal end: the rollouter's queue sentinel makes the trainer drain and return
        to_cancel, to_wait = remaining_after_completion("rollouter", "trainer", ["trainer"])
        self.assertEqual((to_cancel, to_wait), ([], ["trainer"]))

    def test_last_future_done_leaves_nothing(self):
        self.assertEqual(remaining_after_completion("trainer", "trainer", []), ([], []))
        self.assertEqual(remaining_after_completion("rollouter", "trainer", []), ([], []))

    def test_run_loop_uses_the_helper_and_never_waits_on_a_cancelled_future(self):
        src = inspect.getsource(main_mod.FullyAsyncTaskRunner._run_training_loop)
        self.assertIn("remaining_after_completion(future, trainer_future, remaining_futures)", src)
        self.assertIn("for remaining_future in to_cancel:", src)
        self.assertIn("ray.cancel(remaining_future)", src)
        # the loop keeps waiting only on what the helper returned as to_wait
        self.assertIn("futures = remaining_futures", src)
        # the success path still reports, the failure path still cancels and re-raises
        self.assertIn("One component completed successfully", src)
        self.assertIn("raise e", src)


def test_the_helpers_did_not_steal_the_ray_remote_decorators():
    """Both helpers sit right above a @ray.remote class; inserting a def between a decorator
    and its class silently turns the helper into a remote function and the class into a plain
    class (the trainer / task runner would no longer be actors)."""
    from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer, parse_max_train_steps

    assert hasattr(main_mod.FullyAsyncTaskRunner, "__ray_metadata__")
    assert hasattr(FullyAsyncTrainer, "__ray_metadata__")
    assert not hasattr(remaining_after_completion, "remote")
    assert not hasattr(parse_max_train_steps, "remote")


def test_ray_cancel_stops_a_running_async_actor_task():
    """The mechanism main relies on: ray.cancel on the rollouter's fit() future (an async-actor
    task) cancels the asyncio task so the future resolves as cancelled, and the actor itself
    survives. Uses a local single-CPU Ray instance."""
    try:
        ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"local Ray unavailable: {e}")
    try:

        @ray.remote
        class Forever:
            async def fit(self):
                await asyncio.Event().wait()
                return "finished"

            async def ping(self):
                return "pong"

        actor = Forever.remote()
        fut = actor.fit.remote()
        assert ray.get(actor.ping.remote(), timeout=120) == "pong"  # fit is running concurrently
        ray.cancel(fut)
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(fut, timeout=120)
        assert ray.get(actor.ping.remote(), timeout=120) == "pong"  # the actor is still alive
    finally:
        ray.shutdown()


if __name__ == "__main__":
    unittest.main()
