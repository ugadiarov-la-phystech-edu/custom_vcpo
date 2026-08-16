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

import json
import math
import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any
from collections import defaultdict

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
    process_structured_metrics
)
from recipe.fully_async_policy.ess_base_estimator import EssBaseEstimator
from recipe.fully_async_policy.message_queue import MessageQueueClient
from recipe.fully_async_policy.replay_buffer import ReplayBuffer
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer, make_opportunistic_minibatch_indices
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


# make_opportunistic_minibatch_indices moved to ray_trainer (imported above) so
# the fractional-ppo_epochs update path can use it too.


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(FullyAsyncRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # ==================== fully async config ====================

        self.message_queue_client = None
        self.param_synchronizer = None

        # Statistics
        # we start from step 1
        self.global_steps = 1
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        # Bookkeeping for cumulative_training_time (wall clock minus validation/checkpointing).
        # first_sample_time and validation time come from the rollouter via ValidateMetrics.
        self.cumulative_save_time = 0.0
        self.rollouter_first_sample_time = None
        self.rollouter_cumulative_validation_time = 0.0
        # Totals carried over from the run a checkpoint was resumed from (see
        # _save_timing_state/_restore_timing_state). The in-memory counters above
        # only cover the current process; adding these offsets keeps the
        # fully_async/timing/* metrics continuous across restarts.
        self.timing_wall_offset = 0.0
        self.timing_validation_offset = 0.0
        self.timing_save_offset = 0.0
        # Virtual timeline: cumulative_training_time reconstructs the wall clock
        # of an identical run with neither validation nor checkpointing, by
        # replaying the pipeline schedule with validation- and save-caused delays
        # deleted. Each step starts at max(virtual_free_time, batch virtual-ready
        # time from the rollouter's sample stamps) and advances by the step's
        # measured busy duration, excluding wait_last_valid stalls and
        # checkpoint-save time.
        self.virtual_free_time = None
        self.virtual_training_time_offset = 0.0  # restored from timing_state.json on resume
        self._step_virtual_start = None
        self._step_actual_start = None
        self._step_wait_valid_time = 0.0
        self._step_save_time = 0.0
        self.structured_metrics: dict[str, list[Any]] = defaultdict(list)

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        # Opportunistic PPO epochs: extra shuffled mini-batch updates on the
        # current batch while the queue does not yet hold a full next batch.
        opportunistic_cfg = config.async_training.get("opportunistic_epochs", None)
        self.opportunistic_enable = bool(opportunistic_cfg.get("enable", False)) if opportunistic_cfg else False
        self.opportunistic_max_extra_epochs = (
            int(opportunistic_cfg.get("max_extra_epochs", 3)) if opportunistic_cfg else 0
        )
        self.opportunistic_shuffle_seed = int(opportunistic_cfg.get("shuffle_seed", 1234)) if opportunistic_cfg else 0
        # Replay-buffer mode: the trainer keeps drained groups in a
        # version-aware buffer, composes each optimizer mini-batch as all
        # unseen groups (oldest first) plus a staleness-score-weighted sample
        # of already-used groups, syncs weights after every update, and evicts
        # groups staler than replay_buffer.staleness_threshold updates.
        replay_cfg = config.async_training.get("replay_buffer", None)
        self.replay_enable = bool(replay_cfg.get("enable", False)) if replay_cfg else False
        if self.replay_enable:
            self.replay_tau = float(replay_cfg.get("tau", 4.0))
            self.replay_staleness_threshold = int(replay_cfg.get("staleness_threshold", 8))
            # May be fractional (e.g. 1.5): the pause watermark is
            # requires_mini_batches * mini_size groups, while the warm-up runs
            # ceil(requires_mini_batches) fresh-chunk updates — so warm-up
            # alone always fills the buffer past the watermark and steady
            # state starts without a stall.
            self.replay_requires_mini_batches = float(replay_cfg.get("requires_mini_batches", 2))
            assert self.replay_requires_mini_batches >= 1, "replay_buffer.requires_mini_batches must be >= 1"
            self.replay_warmup_updates = math.ceil(self.replay_requires_mini_batches)
            self.replay_sampling_seed = int(replay_cfg.get("sampling_seed", 1234))
            assert self.trigger_parameter_sync_step == 1, (
                "replay_buffer mode syncs weights after every update: set "
                "async_training.trigger_parameter_sync_step=1"
            )
            assert self.require_batches == 1, (
                "replay_buffer mode composes one mini-batch per update: set async_training.require_batches=1"
            )
            assert bool(config.async_training.get("skip_recompute_old_log_prob", False)), (
                "replay_buffer mode trains against the cached behavior log-probs: set "
                "async_training.skip_recompute_old_log_prob=True"
            )
            assert opportunistic_cfg is None or not self.opportunistic_enable, (
                "replay_buffer mode supersedes opportunistic_epochs; disable it"
            )
            assert config.async_training.get("ppo_epochs", None) is None, (
                "replay_buffer mode supersedes async_training.ppo_epochs; leave it null"
            )
            assert str(config.algorithm.adv_estimator) == "grpo", (
                "replay_buffer mode freezes GRPO group advantages at insertion; "
                f"got adv_estimator={config.algorithm.adv_estimator}"
            )
            assert not config.algorithm.use_kl_in_reward, "replay_buffer mode does not support use_kl_in_reward"
            self.replay_buffer = ReplayBuffer(
                tau=self.replay_tau,
                staleness_threshold=self.replay_staleness_threshold,
                seed=self.replay_sampling_seed,
            )
            self.replay_updates_done = 0
            self.rollout_done = False
            # Auto-calibrated ESS reference: with ess_scaling.enable=True and
            # base_ess_ratio=null, the first update runs unscaled and its
            # measured (on-policy, staleness-0 warm-up) ESS ratio becomes the
            # base, passed to the actor via meta_info["ess_base_override"]
            # and persisted in replay_buffer.pt across restarts.
            actor_cfg = config.actor_rollout_ref.actor
            self.replay_ess_auto_base = bool(
                actor_cfg.get("update_policy_per_traj", False)
                and actor_cfg.ess_scaling.get("enable", False)
                and actor_cfg.ess_scaling.get("base_ess_ratio", None) is None
            )
            self.replay_ess_use_clipped = bool(actor_cfg.ess_scaling.get("use_clipped", False))
            self.replay_ess_base = None
            # Dynamic base estimator (mode=new_cohort): tracks the fresh
            # cohort's ESS as the brake reference instead of freezing the
            # first-update capture. The capture still runs and seeds the
            # estimator (warm start + default clamp ceiling).
            estimator_cfg = config.async_training.get("ess_base_estimator", None)
            estimator_mode = (
                str(estimator_cfg.get("mode", "first_update")) if estimator_cfg is not None else "first_update"
            )
            if estimator_mode == "new_cohort":
                assert self.replay_ess_auto_base, (
                    "ess_base_estimator.mode=new_cohort requires update_policy_per_traj=True, "
                    "ess_scaling.enable=True and ess_scaling.base_ess_ratio=null"
                )
                self.ess_base_estimator = EssBaseEstimator.from_config(estimator_cfg)
            else:
                assert estimator_mode == "first_update", (
                    f"unknown async_training.ess_base_estimator.mode: {estimator_mode}"
                )
                self.ess_base_estimator = None
        else:
            actor_cfg = config.actor_rollout_ref.actor
            assert not (
                actor_cfg.get("update_policy_per_traj", False)
                and actor_cfg.ess_scaling.get("enable", False)
                and actor_cfg.ess_scaling.get("base_ess_ratio", None) is None
            ), (
                "ess_scaling.base_ess_ratio=null (auto-calibration from the first update) is only "
                "supported in replay_buffer mode; set an explicit base_ess_ratio"
            )
            estimator_cfg = config.async_training.get("ess_base_estimator", None)
            assert estimator_cfg is None or str(estimator_cfg.get("mode", "first_update")) == "first_update", (
                "ess_base_estimator.mode=new_cohort is only supported in replay_buffer mode"
            )
            self.ess_base_estimator = None
        # Stop-the-world accounting modes: freeze the whole pipeline during
        # validation (trainer blocks instead of training ahead on backlog) and
        # generation during checkpoint saves, so both become pure time
        # translations of the pipeline and cumulative_training_time / the
        # trajectory match a no-validation-no-save run exactly.
        self.serialize_validation = bool(config.async_training.get("serialize_validation", False))
        self.pause_generation_during_save = bool(config.async_training.get("pause_generation_during_save", False))
        self.compute_prox_log_prob = self.config.async_training.compute_prox_log_prob
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        self.param_synchronizer = param_synchronizer

    def set_total_train_steps(self, total_train_steps):
        self.total_train_steps = total_train_steps
        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = self.message_queue_client.get_sample_sync()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds."
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        self._open_virtual_step(consumer_end, queue_samples)
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["trainer_param_version"] = self.current_param_version
        if "traj_uid" not in batch.non_tensor_batch and "uid" in batch.non_tensor_batch:
            uids = batch.non_tensor_batch.get("uid")
            batch.non_tensor_batch["traj_uid"] = np.array(
                [f"group-{uid}_traj-{idx}" for idx, uid in enumerate(uids)],
                dtype=object,
            )
            print(f"[FullyAsyncTrainer] Added trajectory UIDs to batch: {len(batch.non_tensor_batch['traj_uid'])}")
        if "reward_scalar" in batch.non_tensor_batch:
            reward_scalars = batch.non_tensor_batch.get("reward_scalar")
            avg_reward = np.mean(reward_scalars).item()
            print(f"[FullyAsyncTrainer] reward_scalar found with avg={avg_reward}")
        return 0, batch

    def _run_opportunistic_epochs(self, batch, metrics, timing_raw):
        """Opportunistic PPO epochs (rollout-bound elasticity).

        After the scheduled update, keep running extra mini-batch updates on the
        current batch for as long as the message queue holds fewer than
        required_samples — i.e. exactly while the trainer would otherwise idle
        waiting for generation. The queue depth is re-checked before every extra
        mini-batch, so an extra epoch shrinks (or never starts) the moment the
        next batch is ready; in trainer-bound runs the first check already fails
        and the feature is dormant. max_extra_epochs bounds reuse during
        rollouter pauses (validation / checkpoint saves), when "no new data"
        can hold for minutes. Groups are reshuffled across mini-batches each
        extra epoch (group-complete, equal-size — see
        make_opportunistic_minibatch_indices). Every extra update goes through
        the normal update_actor path, so with skip_recompute_old_log_prob the
        IS ratios are recomputed against the current policy per update. Note
        each extra update also advances the lr scheduler by one step (harmless
        for constant lr).
        """
        if not self.opportunistic_enable or self.opportunistic_max_extra_epochs <= 0:
            return
        extra_updates = 0
        extra_epochs_completed = 0
        queue_ready = False
        with marked_timer("opportunistic_extra", timing_raw, color="yellow"):
            rng = np.random.default_rng(self.opportunistic_shuffle_seed + self.global_steps)
            for extra_epoch in range(1, self.opportunistic_max_extra_epochs + 1):
                try:
                    minibatch_indices = make_opportunistic_minibatch_indices(
                        batch.non_tensor_batch["uid"], self.require_batches, rng
                    )
                except (KeyError, ValueError) as e:
                    # A malformed batch (missing uids, unequal group sizes, ...)
                    # must not kill the run over an *optional* extra pass.
                    print(
                        f"[FullyAsyncTrainer] opportunistic epochs skipped: cannot form mini-batches ({e})",
                        flush=True,
                    )
                    break
                epoch_metrics = defaultdict(list)
                for minibatch_idx in minibatch_indices:
                    if self.message_queue_client.get_queue_size_sync() >= self.required_samples:
                        queue_ready = True
                        break
                    subset = batch.select_idxs(minibatch_idx)
                    # select_idxs shares meta_info by reference; copy before stamping
                    subset.meta_info = dict(batch.meta_info)
                    subset.meta_info["global_token_num"] = subset.batch["attention_mask"].sum(dim=-1).tolist()
                    subset.meta_info["opportunistic_extra_epoch"] = extra_epoch
                    actor_output = self.actor_rollout_wg.update_actor(subset)
                    for key, value in reduce_metrics(actor_output.meta_info["metrics"]).items():
                        epoch_metrics[key].append(value)
                    extra_updates += 1
                # Per-extra-epoch actor/IS diagnostics (the scheduled pass keeps
                # the unprefixed keys); partial epochs log whatever ran.
                for key, values in epoch_metrics.items():
                    if key.startswith("actor/") or key.startswith("rollout_corr/"):
                        metrics[f"opportunistic/epoch_{extra_epoch}/{key}"] = float(np.mean(values))
                if queue_ready:
                    break
                extra_epochs_completed += 1
        metrics["opportunistic/extra_updates"] = extra_updates
        metrics["opportunistic/extra_epochs_completed"] = extra_epochs_completed
        metrics["opportunistic/extra_epochs"] = extra_updates / self.require_batches
        if extra_updates > 0:
            print(
                f"[FullyAsyncTrainer] opportunistic epochs at step {self.global_steps}: "
                f"{extra_updates} extra mini-batch updates "
                f"({extra_epochs_completed} full epochs, cap {self.opportunistic_max_extra_epochs})",
                flush=True,
            )

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    def _init_async_rollout_manager(self):
        pass

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        if self.replay_enable:
            return self._fit_replay()

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.max_steps_duration = 0

        # get validate data before training
        self._log_validation_data()

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            metrics = {}
            timing_raw = {}
            self._step_wait_valid_time = 0.0
            self._step_save_time = 0.0

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    epoch, batch = self._get_samples_from_queue()
                    if batch is None:
                        break
                    self._collect_metrics_from_samples(batch, metrics)
                batch.meta_info["rollout_corr_config"] = self.config.algorithm.get("rollout_correction", None)
                batch.meta_info["n_resp_per_rollout"] = self.config.actor_rollout_ref.rollout.n
                if (
                    self.config.actor_rollout_ref.actor.grad_baselining.enable
                    and self.config.actor_rollout_ref.actor.update_policy_per_traj
                ):
                    # Keep same rollout group on same DP rank when OPOB baselining is enabled.
                    batch.meta_info["dp_group_key"] = "uid"
                    batch.meta_info["dp_group_size"] = self.config.actor_rollout_ref.rollout.n
                batch, reward_extra_infos_dict = self._process_batch_common(
                    batch, metrics, timing_raw, self.local_trigger_step if self.compute_prox_log_prob else None
                )
                self._log_rollout(batch, reward_extra_infos_dict, timing_raw)
                self._run_opportunistic_epochs(batch, metrics, timing_raw)

            self._collect_metrics(batch, 0, metrics, timing_raw)
            structured_metrics = self.metrics_aggregator.add_step_metrics(
                metrics=metrics, sample_count=self.required_samples, timestamp=time.time(), structured_metrics=self.structured_metrics
            )
            if structured_metrics is not None:
                self.structured_metrics = structured_metrics
            # Trigger parameter synchronization after training step
            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
                f"local_trigger_step: {self.local_trigger_step} "
                f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
                f"{time_str}"
            )
            self._trigger_parameter_sync_after_step(global_steps=self.global_steps)
            # [NOTE] Skip self._log_validation_data() already logged in _trigger_parameter_sync_after_step
            self._check_save_checkpoint(timing_raw)
            self._advance_virtual_clock()
            self.global_steps += 1

        # final parameter sync and validate
        # 1. waiting remaining validate task
        ray.get(self.param_synchronizer.wait_last_valid.remote())
        self._log_validation_data()
        # 2. perform addtional parameter_sync and validate if trainer already updated
        if self.current_param_version % self.config.rollout.test_freq != 0 or self.local_trigger_step > 1:
            self._trigger_parameter_sync_after_step(validate=True, global_steps=self.global_steps)
            ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._log_validation_data()
        self.progress_bar.close()

        self._check_save_checkpoint(timing_raw)

    # ==================== replay-buffer training mode ====================

    def _drain_queue_into_buffer(self) -> int:
        """Move everything currently in the transport queue into the replay
        buffer without blocking. Returns the number of groups added."""
        drained = self.message_queue_client.get_available_samples_sync()
        added = 0
        for raw in drained:
            if raw is None:
                self.rollout_done = True
                continue
            rollout_sample = ray.cloudpickle.loads(raw)
            self.replay_buffer.add(rollout_sample, self.current_param_version)
            added += 1
        return added

    def _wait_one_sample_into_buffer(self) -> bool:
        """Block for one sample from the transport queue and add it to the
        buffer. Returns False when the termination sentinel arrived instead."""
        result = self.message_queue_client.get_sample_sync()
        if result is None:
            self.rollout_done = True
            return False
        sample, _ = result
        if sample is None:
            self.rollout_done = True
            return False
        self.replay_buffer.add(ray.cloudpickle.loads(sample), self.current_param_version)
        return True

    def _acquire_replay_minibatch(self):
        """Compose the next mini-batch of groups from the replay buffer.

        Warm-up (first ceil(requires_mini_batches) updates): wait until
        mini_size *unseen* groups are buffered and use exactly those, oldest
        first. Steady state: pause only while the buffer holds fewer than
        requires_mini_batches x mini_size groups (fractional values allowed),
        then compose all unseen groups (oldest first, capped) plus a
        score-weighted sample of used ones. Returns (entries, info) or
        (None, None) when generation has finished and the buffer cannot
        support another mini-batch."""
        mini_size = self.required_samples
        watermark = self.replay_requires_mini_batches * mini_size
        self._drain_queue_into_buffer()
        if self.replay_updates_done < self.replay_warmup_updates:
            while self.replay_buffer.new_count() < mini_size:
                if self.rollout_done:
                    print(
                        f"[FullyAsyncTrainer][Replay] rollout finished during warm-up with "
                        f"{self.replay_buffer.new_count()}/{mini_size} unseen groups; stopping"
                    )
                    return None, None
                self._wait_one_sample_into_buffer()
            entries = self.replay_buffer.take_oldest_new(mini_size)
            info = {
                "n_new": mini_size,
                "n_replayed": 0,
                "staleness": [e.staleness(self.current_param_version) for e in entries],
            }
        else:
            while self.replay_buffer.size() < watermark:
                if self.rollout_done:
                    print(
                        f"[FullyAsyncTrainer][Replay] rollout finished with buffer "
                        f"{self.replay_buffer.size()} < watermark {watermark}; stopping"
                    )
                    return None, None
                self._wait_one_sample_into_buffer()
            entries, info = self.replay_buffer.compose_minibatch(mini_size, self.current_param_version)
        # Open the virtual (no-validation-no-save) step: only the unseen
        # entries' arrival stamps gate this step — replayed groups were ready
        # long ago (a pure-replay mini-batch never waits on generation).
        consumer_end = time.time()
        self._open_virtual_step(consumer_end, [e.sample for e in entries if e.is_new])
        return entries, info

    def _replay_post_update_maintenance(self, entries, new_version: int) -> None:
        """Buffer maintenance after one replay update, at the model version the
        update just produced: retire the used groups' is_new flag BEFORE
        evicting — a just-trained group crossing the staleness threshold must
        count as evicted-seen, not evicted_unseen (that counter means
        generated-but-never-trained-on waste) — then evict too-stale groups
        and decay the survivors' scores."""
        self.replay_buffer.mark_used(entries)
        self.replay_buffer.evict(new_version)
        self.replay_buffer.recompute_scores(new_version)

    def _build_replay_batch(self, entries):
        """Assemble a training DataProto from buffered groups using the frozen
        insertion-time statistics: advantages broadcast from advantage_scalar,
        sparse token_level rewards from reward_scalar, cached behavior
        log-probs consumed via the skip_recompute_old_log_prob backward path
        (no reward / old-log-prob / advantage recomputation)."""
        rollout_samples = [e.sample for e in entries]
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                rollout_samples, self.tokenizer, self.config, self._balance_batch
            )
        else:
            batch = assemble_batch_from_rollout_samples(rollout_samples, self.tokenizer, self.config, None)
        batch.meta_info["trainer_param_version"] = self.current_param_version
        # Cohort membership for the dynamic ESS-base estimator: mark every row
        # of a not-yet-trained-on group. Stamped via group-uid membership (not
        # positional repeat) so balance_batch reordering cannot misalign it.
        new_group_uids = set()
        for entry in entries:
            if entry.is_new:
                entry_uids = entry.sample.full_batch.non_tensor_batch.get("uid")
                if entry_uids is not None and len(entry_uids):
                    new_group_uids.add(entry_uids[0])
        batch.non_tensor_batch["replay_is_new"] = np.array(
            [uid in new_group_uids for uid in batch.non_tensor_batch["uid"]], dtype=bool
        )
        if "traj_uid" not in batch.non_tensor_batch and "uid" in batch.non_tensor_batch:
            uids = batch.non_tensor_batch.get("uid")
            batch.non_tensor_batch["traj_uid"] = np.array(
                [f"group-{uid}_traj-{idx}" for idx, uid in enumerate(uids)],
                dtype=object,
            )

        response_mask = batch.batch["response_mask"]
        adv_scalars = torch.from_numpy(
            np.asarray(batch.non_tensor_batch["advantage_scalar"], dtype=np.float32)
        )
        advantages = adv_scalars.unsqueeze(-1) * response_mask.float()
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages

        reward_scalars = torch.from_numpy(
            np.asarray(batch.non_tensor_batch["reward_scalar"], dtype=np.float32)
        )
        token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        lengths = response_mask.sum(dim=-1).long()
        valid = lengths > 0
        rows = torch.arange(response_mask.shape[0])[valid]
        token_level_scores[rows, (lengths[valid] - 1)] = reward_scalars[valid]
        batch.batch["token_level_scores"] = token_level_scores
        batch.batch["token_level_rewards"] = token_level_scores

        batch.meta_info["skip_recompute_old_log_prob"] = True
        batch.meta_info["rollout_corr_config"] = self.config.algorithm.get("rollout_correction", None)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.meta_info["n_resp_per_rollout"] = self.config.actor_rollout_ref.rollout.n
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        if (
            self.config.actor_rollout_ref.actor.grad_baselining.enable
            and self.config.actor_rollout_ref.actor.update_policy_per_traj
        ):
            # Keep same rollout group on same DP rank when OPOB baselining is enabled.
            batch.meta_info["dp_group_key"] = "uid"
            batch.meta_info["dp_group_size"] = self.config.actor_rollout_ref.rollout.n
        if self.replay_ess_auto_base:
            # None until the first update's measurement is captured; the actor
            # skips LR scaling while the override is unresolved.
            batch.meta_info["ess_base_override"] = self.replay_ess_base
        return batch

    def _capture_ess_base(self, metrics):
        """Auto-calibration of ess_scaling.base_ess_ratio: capture the first
        update's measured ESS ratio (the staleness-0 warm-up mini-batch, i.e.
        the empirical on-policy rho_on) from the actor's structured
        staleness/ess entries. No-op once captured. The field matches
        ess_scaling.use_clipped so the reference and the scaling numerator
        measure the same quantity."""
        if self.replay_ess_base is not None:
            return
        key = "minibatch_ess_ratio_clipped" if self.replay_ess_use_clipped else "minibatch_ess_ratio"
        entries = metrics.get("staleness/ess") or []
        values = [float(e[key]) for e in entries if isinstance(e, dict) and e.get(key) is not None]
        if values:
            self.replay_ess_base = float(np.mean(values))
            print(
                f"[FullyAsyncTrainer][Replay] auto-calibrated ess_scaling.base_ess_ratio="
                f"{self.replay_ess_base:.4f} from the first update ({key})"
            )

    def _update_ess_base_estimator(self, metrics):
        """Feed this update's measured payloads (fresh-cohort weight moments
        and staleness buckets from the actor's structured staleness/ess
        entries) into the dynamic base estimator, refresh replay_ess_base for
        the next update's ess_base_override, and surface the estimator
        diagnostics as replay/ess_base_* scalars. Runs after
        _capture_ess_base so the first-update capture seeds the estimator
        (warm start + default clamp ceiling)."""
        estimator = self.ess_base_estimator
        estimator.seed(self.replay_ess_base)
        estimator.observe_entries(metrics.get("staleness/ess") or [])
        base = estimator.current_base()
        if base is not None:
            self.replay_ess_base = float(base)
        metrics.update(estimator.diagnostics())

    def _add_replay_metrics(self, metrics, info, new_version):
        """Item-17 metrics, computed after this update's eviction/rescoring at
        the post-update model version. Histogram lists go through the
        structured-metrics path (wandb images when enabled; scalars always)."""
        minibatch_staleness = info["staleness"]
        buffer_staleness = self.replay_buffer.staleness_list(new_version)
        metrics.update(
            {
                "replay/buffer_size": self.replay_buffer.size(),
                "replay/buffer_new": self.replay_buffer.new_count(),
                "replay/buffer_max_staleness": float(self.replay_buffer.max_staleness(new_version) or 0),
                "replay/minibatch_new": info["n_new"],
                "replay/minibatch_replayed": info["n_replayed"],
                "replay/minibatch_new_ratio": info["n_new"] / (info["n_new"] + info["n_replayed"]),
                "replay/minibatch_staleness_mean": float(np.mean(minibatch_staleness)),
                "replay/minibatch_staleness_max": float(np.max(minibatch_staleness)),
                "replay/evicted_cum": self.replay_buffer.evicted_total,
                "replay/evicted_unseen_cum": self.replay_buffer.evicted_unseen_total,
                "replay/total_added": self.replay_buffer.total_added,
                # Histograms (lists -> structured metrics, not scalar-reduced)
                "replay/minibatch_staleness_hist": [int(s) for s in minibatch_staleness],
                "replay/buffer_staleness_hist": [int(s) for s in buffer_staleness],
            }
        )
        if buffer_staleness:
            metrics["replay/buffer_staleness_mean"] = float(np.mean(buffer_staleness))
        # Per-traj actor updates report the effective (possibly ESS-scaled) lr
        # per mini-batch in the structured staleness/ess entries; surface the
        # mean next to the replay metrics so the brake is visible on the same
        # dashboard. No-op on the standard update path (no staleness/ess key).
        ess_entries = metrics.get("staleness/ess") or []
        scaled_lrs = [
            float(e["ess_scaled_lr"])
            for e in ess_entries
            if isinstance(e, dict) and e.get("ess_scaled_lr") is not None
        ]
        if scaled_lrs:
            metrics["replay/ess_scaled_lr"] = float(np.mean(scaled_lrs))
        # base_ess_ratio may evolve during training: prefer the value the actor
        # actually resolved and used this update (reported in the entries);
        # fall back to the trainer's captured auto-base.
        used_bases = [
            float(e["base_ess_ratio"])
            for e in ess_entries
            if isinstance(e, dict) and e.get("base_ess_ratio") is not None
        ]
        if used_bases:
            metrics["replay/ess_base"] = float(np.mean(used_bases))
        elif getattr(self, "replay_ess_base", None) is not None:
            metrics["replay/ess_base"] = self.replay_ess_base

    REPLAY_HIST_KEYS = ("replay/minibatch_staleness_hist", "replay/buffer_staleness_hist")

    def _log_tb_staleness_histograms(self, step: int):
        """Send the raw per-group staleness lists to the tensorboard backend
        only (SummaryWriter.add_histogram gives the native histogram view).
        The wandb image path in process_structured_metrics is unaffected, and
        the console logger never sees the raw lists. Must run before the
        structured-metrics reset in _trigger_parameter_sync_after_step."""
        if "tensorboard" not in self.config.trainer.logger:
            return
        tb_hist = {}
        for key in self.REPLAY_HIST_KEYS:
            values = self.structured_metrics.get(key)
            if values:
                tb_hist[key] = [float(v) for v in values]
        if tb_hist:
            self.logger.log(data=tb_hist, step=step, backend=["tensorboard"])

    def _fit_replay(self):
        """Replay-buffer training loop: one optimizer update per iteration,
        weight sync after every update, staleness-based eviction and score
        decay, warm-up on fresh groups. See replay_buffer.py for the buffer
        semantics."""
        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.max_steps_duration = 0
        self._log_validation_data()

        timing_raw = {}
        while True:
            metrics = {}
            timing_raw = {}
            self._step_wait_valid_time = 0.0
            self._step_save_time = 0.0

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    entries, info = self._acquire_replay_minibatch()
                    if entries is None:
                        break
                    batch = self._build_replay_batch(entries)
                    self._collect_metrics_from_samples(batch, metrics)
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                if self.replay_ess_auto_base:
                    self._capture_ess_base(metrics)
                if self.ess_base_estimator is not None:
                    self._update_ess_base_estimator(metrics)
                self._log_rollout(batch, {}, timing_raw)

            # Post-update buffer maintenance at the version this update just
            # produced (stamped by the sync below).
            new_version = self.current_param_version + 1
            self._replay_post_update_maintenance(entries, new_version)
            self.replay_updates_done += 1
            self._add_replay_metrics(metrics, info, new_version)

            self._collect_metrics(batch, 0, metrics, timing_raw)
            structured_metrics = self.metrics_aggregator.add_step_metrics(
                metrics=metrics,
                sample_count=self.required_samples,
                timestamp=time.time(),
                structured_metrics=self.structured_metrics,
            )
            if structured_metrics is not None:
                self.structured_metrics = structured_metrics
            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"[FullyAsyncTrainer][Replay] global_steps: {self.global_steps} "
                f"update: {self.replay_updates_done} "
                f"buffer: {self.replay_buffer.size()} "
                f"(new: {self.replay_buffer.new_count()}) {time_str}"
            )
            self._trigger_parameter_sync_after_step(global_steps=self.global_steps)
            self._check_save_checkpoint(timing_raw)
            self._advance_virtual_clock()
            self.global_steps += 1

        # final parameter sync and validate (same tail as the FIFO fit)
        ray.get(self.param_synchronizer.wait_last_valid.remote())
        self._log_validation_data()
        if self.current_param_version % self.config.rollout.test_freq != 0 or self.local_trigger_step > 1:
            self._trigger_parameter_sync_after_step(validate=True, global_steps=self.global_steps)
            ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._log_validation_data()
        self.progress_bar.close()

        self._check_save_checkpoint(timing_raw)

    def _check_save_checkpoint(self, timing_raw):
        if self.current_param_version == self.last_ckpt_version:
            return
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. The current step number is a multiple of the save frequency.
        # 3. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version
            self.cumulative_save_time += timing_raw.get("save_checkpoint", 0.0)
            # Saving is not training: exclude it from the virtual timeline's busy time.
            self._step_save_time += timing_raw.get("save_checkpoint", 0.0)

    def _save_checkpoint(self):
        if self.pause_generation_during_save:
            # Stop-the-world save: freeze generation for the entire save (the
            # long actor dist-ckpt below, not just the rollouter's queue
            # snapshot), so the save is a pure time translation of the pipeline
            # instead of building a queue surplus. The rollouter accounts the
            # full pause into cumulative_checkpoint_pause at resume.
            ray.get(self.param_synchronizer.pause_rollouter_for_save.remote())
        try:
            self._save_checkpoint_inner()
        finally:
            if self.pause_generation_during_save:
                ray.get(self.param_synchronizer.resume_rollouter_after_save.remote())

    def _save_checkpoint_inner(self):
        save_start = time.time()
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.param_synchronizer.rollouter_save_checkpoint.remote(local_global_step_folder))
        self._save_timing_state(local_global_step_folder, save_start)
        if self.replay_enable:
            replay_path = os.path.join(local_global_step_folder, "replay_buffer.pt")
            torch.save(self._replay_checkpoint_state(), replay_path)
            print(
                f"[FullyAsyncTrainer][Replay] Saved replay buffer "
                f"({self.replay_buffer.size()} groups) to {replay_path}"
            )
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def _save_timing_state(self, local_global_step_folder, save_start):
        """Persist the cumulative timing totals so a resumed run continues the
        fully_async/timing/* metrics instead of restarting them from zero.
        The snapshot is taken at save_start, so the in-progress save's own
        duration is excluded — it is exactly the state a resume reconstructs.
        Before the rollouter reports its first training-dataset draw the current
        segment has no measurable wall time, so the restored offsets are carried
        forward unchanged."""
        virtual_now = self._virtual_now(save_start)
        if self.rollouter_first_sample_time is not None:
            wall_time = save_start - self.rollouter_first_sample_time + self.timing_wall_offset
            validation_time = self.rollouter_cumulative_validation_time + self.timing_validation_offset
            save_time = self.cumulative_save_time + self.timing_save_offset
            if virtual_now is not None:
                virtual_training_time = (
                    virtual_now - self.rollouter_first_sample_time + self.virtual_training_time_offset
                )
            else:
                virtual_training_time = self.virtual_training_time_offset
        else:
            wall_time = self.timing_wall_offset
            validation_time = self.timing_validation_offset
            save_time = self.timing_save_offset
            virtual_training_time = self.virtual_training_time_offset
        timing_state = {
            "wall_time_since_first_sample": wall_time,
            "cumulative_validation_time": validation_time,
            "cumulative_save_time": save_time,
            "cumulative_training_time": virtual_training_time,
        }
        with open(os.path.join(local_global_step_folder, "timing_state.json"), "w") as f:
            json.dump(timing_state, f, indent=2)

    def _restore_timing_state(self, global_step_folder):
        timing_state_path = os.path.join(global_step_folder, "timing_state.json")
        if not os.path.exists(timing_state_path):
            print("[FullyAsyncTrainer] No timing_state.json in checkpoint; timing metrics restart from zero")
            return
        with open(timing_state_path) as f:
            timing_state = json.load(f)
        self.timing_wall_offset = timing_state.get("wall_time_since_first_sample", 0.0)
        self.timing_validation_offset = timing_state.get("cumulative_validation_time", 0.0)
        self.timing_save_offset = timing_state.get("cumulative_save_time", 0.0)
        # Checkpoints from before the virtual-clock metric only carry the naive
        # subtraction value; it is the best available continuation point.
        self.virtual_training_time_offset = timing_state.get(
            "cumulative_training_time",
            self.timing_wall_offset - self.timing_validation_offset - self.timing_save_offset,
        )
        print(
            f"[FullyAsyncTrainer] Restored timing state from {timing_state_path}: "
            f"cumulative_training_time resumes at {self.virtual_training_time_offset:.1f}s"
        )

    def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncTrainer] Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")
        self._restore_timing_state(global_step_folder)
        if self.replay_enable:
            replay_path = os.path.join(global_step_folder, "replay_buffer.pt")
            if os.path.exists(replay_path):
                replay_state = torch.load(replay_path, weights_only=False)
                self._load_replay_checkpoint_state(replay_state)
                print(
                    f"[FullyAsyncTrainer][Replay] Restored replay buffer "
                    f"({self.replay_buffer.size()} groups, {self.replay_updates_done} updates done) "
                    f"from {replay_path}"
                )
            else:
                print(f"[FullyAsyncTrainer][Replay] WARNING: no replay buffer state at {replay_path}, starting empty")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        return self.current_param_version

    def _replay_checkpoint_state(self) -> dict:
        return {
            "buffer": self.replay_buffer.state_dict(),
            "updates_done": self.replay_updates_done,
            "ess_base": self.replay_ess_base,
            "ess_base_estimator": (
                self.ess_base_estimator.state_dict()
                if getattr(self, "ess_base_estimator", None) is not None
                else None
            ),
        }

    def _load_replay_checkpoint_state(self, state: dict) -> None:
        self.replay_buffer.load_state_dict(state["buffer"])
        self.replay_updates_done = int(state.get("updates_done", 0))
        self.replay_ess_base = state.get("ess_base", None)
        estimator_state = state.get("ess_base_estimator", None)
        if getattr(self, "ess_base_estimator", None) is not None and estimator_state:
            self.ess_base_estimator.load_state_dict(estimator_state)
        if self.replay_ess_auto_base and self.replay_ess_base is None:
            print(
                "[FullyAsyncTrainer][Replay] WARNING: ess_scaling.base_ess_ratio=null (auto) but the "
                "checkpoint carries no stored value; it will be captured from the FIRST POST-RESUME "
                "update, which is NOT on-policy (mature buffer) — prefer an explicit base_ess_ratio "
                "when resuming from checkpoints predating this feature"
            )

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            samples_param_versions = batch.meta_info["rollout_param_versions"]
            stale_count = sum(1 for v in samples_param_versions if self.current_param_version - v >= 1)
            self.stale_samples_processed += stale_count
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_samples_processed": self.stale_samples_processed,
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    def _trigger_parameter_sync_after_step(self, validate: bool = False, global_steps: int = None):
        """
        Trigger parameter synchronization after training step
        This ensures rollouter always uses the latest trained parameters
        """
        if self.local_trigger_step < self.trigger_parameter_sync_step and not validate:
            self.local_trigger_step += 1
            return

        # Param Sync before validation
        timing_param_sync = {}
        with marked_timer("timing_s/wait_last_valid", timing_param_sync):
            ray.get(self.param_synchronizer.wait_last_valid.remote())
        # This wait exists only because of validation: exclude it from the
        # virtual (no-validation) timeline's busy time.
        self._step_wait_valid_time += timing_param_sync["timing_s/wait_last_valid"]

        # [NOTE] Log validation data before incrementing step
        self._log_validation_data()

        self.current_param_version += 1
        self.local_trigger_step = 1
        step_data = self.metrics_aggregator.get_aggregated_metrics()
        # wandb.Image media breaks non-wandb loggers (e.g. tensorboard's add_scalar)
        allow_media = "wandb" in self.config.trainer.logger
        step_data.update(process_structured_metrics(self.structured_metrics, allow_media=allow_media))
        self._log_tb_staleness_histograms(self.current_param_version)
        self._add_cumulative_time_metrics(step_data)
        self.structured_metrics = defaultdict(list)
        self.logger.log(
            data=step_data,
            step=self.current_param_version,
        )
        self.progress_bar.update(1)
        self.metrics_aggregator.reset()
        
        with marked_timer("timing_s/param_sync", timing_param_sync):
            ray.get(
                self.param_synchronizer.sync_weights.remote(
                    self.current_param_version, validate=validate, global_steps=global_steps
                )
            )
        if self.serialize_validation and (validate or self._sync_triggers_validation()):
            # Stop-the-world validation: block until the validation sweep this
            # sync just launched completes, instead of training ahead on
            # backlog. Generation is paused throughout, so the whole pipeline
            # freezes — a pure time translation of the no-validation schedule.
            # The stall is validation-caused idle: exclude it from the virtual
            # clock like every other wait_last_valid stall.
            with marked_timer("timing_s/wait_validation_serialized", timing_param_sync):
                ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._step_wait_valid_time += timing_param_sync["timing_s/wait_validation_serialized"]
        self.logger.log(data=timing_param_sync, step=self.current_param_version)

    def _sync_triggers_validation(self) -> bool:
        """Mirror of the rollouter's validation decision in update_param_version:
        validation fires when the just-synced version is a positive multiple of
        rollout.test_freq. (If the rollouter has no val_reward_fn the wait
        returns immediately, so a false positive costs nothing.)"""
        test_freq = self.config.rollout.test_freq
        return test_freq > 0 and self.current_param_version > 0 and self.current_param_version % test_freq == 0

    def _open_virtual_step(self, consumer_end: float, queue_samples: list):
        """Start this step on the virtual (no-validation-no-save) timeline: at
        max(trainer free, batch ready), where the batch is ready when its last
        sample would have arrived without the rollouter's validation and
        checkpoint-save pauses. Samples restored from an old-format queue
        snapshot may lack the stamps; fall back to the actual ready time (no
        pause correction) for them."""
        virtual_ready_times = [
            s.enqueue_time - s.validation_pause_before - getattr(s, "checkpoint_pause_before", 0.0)
            for s in queue_samples
            if getattr(s, "enqueue_time", None) is not None
        ]
        batch_virtual_ready = max(virtual_ready_times) if virtual_ready_times else consumer_end
        self._step_actual_start = consumer_end
        self._step_virtual_start = (
            max(self.virtual_free_time, batch_virtual_ready)
            if self.virtual_free_time is not None
            else batch_virtual_ready
        )

    def _virtual_now(self, now: float):
        """Current position on the virtual (no-validation-no-save) timeline.

        Mid-step, the step began at _step_virtual_start and has been busy for
        the actual elapsed time minus any wait_last_valid stall and checkpoint
        saving; between steps it is wherever the last step ended. None until
        the first batch arrives."""
        if self._step_virtual_start is not None:
            return (
                self._step_virtual_start
                + (now - self._step_actual_start)
                - self._step_wait_valid_time
                - self._step_save_time
            )
        return self.virtual_free_time

    def _advance_virtual_clock(self, now: float = None):
        """Close the current step on the virtual timeline (called after the
        checkpoint save, whose duration _virtual_now excludes)."""
        if self._step_virtual_start is None:
            return
        self.virtual_free_time = self._virtual_now(time.time() if now is None else now)
        self._step_virtual_start = None
        self._step_actual_start = None

    def _add_cumulative_time_metrics(self, step_data: dict, now: float = None):
        """cumulative_training_time: the wall clock (since the first
        training-dataset draw) that an identical run with *neither validation
        nor checkpointing* would have needed to reach this point. Reconstructed
        by replaying the pipeline schedule on a virtual timeline: each step
        starts at max(trainer free, batch ready), with sample-ready times
        shifted back by the rollouter's validation and checkpoint-save pauses,
        and trainer stalls on wait_last_valid and checkpoint saving excluded
        from the busy time. This makes the metric exact in rollout-bound,
        trainer-bound and balanced regimes alike (a naive wall - validation -
        save subtraction, derivable from the three component tags,
        over-subtracts whenever validation overlaps saves or backlog training).
        No-op until the rollouter reports its first training-dataset draw."""
        if self.rollouter_first_sample_time is None:
            return
        now = time.time() if now is None else now
        wall_time = now - self.rollouter_first_sample_time + self.timing_wall_offset
        validation_time = self.rollouter_cumulative_validation_time + self.timing_validation_offset
        save_time = self.cumulative_save_time + self.timing_save_offset
        step_data["fully_async/timing/wall_time_since_first_sample"] = wall_time
        step_data["fully_async/timing/cumulative_validation_time"] = validation_time
        step_data["fully_async/timing/cumulative_save_time"] = save_time
        virtual_now = self._virtual_now(now)
        if virtual_now is not None:
            step_data["fully_async/timing/cumulative_training_time"] = (
                virtual_now - self.rollouter_first_sample_time + self.virtual_training_time_offset
            )

    def _log_validation_data(self):
        """
        Log validation data
        """
        
        # [NOTE] Continue logging val_data until message queue client is empty
        while True: 
            val_data = self.message_queue_client.get_validate_sync()
            if not val_data:
                break

            val_metrics: ValidateMetrics = ray.cloudpickle.loads(val_data)
            if val_metrics.first_sample_time is not None:
                self.rollouter_first_sample_time = val_metrics.first_sample_time
            if val_metrics.cumulative_validation_time is not None:
                self.rollouter_cumulative_validation_time = val_metrics.cumulative_validation_time
            if val_metrics.metrics:
                self.logger.log(data=val_metrics.metrics, step=val_metrics.param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} val_metric step={val_metrics.param_version}\n"
                    f"Validation metrics: {val_metrics.metrics}"

                )
            self.logger.log(data=val_metrics.timing_raw, step=val_metrics.param_version)