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
import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any
from collections import defaultdict

import numpy as np
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
    process_structured_metrics
)
from recipe.fully_async_policy.message_queue import MessageQueueClient
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
        # Totals carried over from the resumed-from run
        # (_save/_restore_timing_state) keep the fully_async/timing/* metrics
        # continuous across restarts.
        self.timing_wall_offset = 0.0
        self.timing_validation_offset = 0.0
        self.timing_save_offset = 0.0
        # Virtual timeline: cumulative_training_time replays the pipeline
        # schedule with validation- and save-caused delays deleted. Each step
        # starts at max(virtual_free_time, batch virtual-ready time) and
        # advances by measured busy time minus wait_last_valid stalls and save
        # time.
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
        # Stop-the-world accounting: freeze the pipeline during validation and
        # generation during saves, so both become pure time translations and
        # the clock/trajectory match a no-validation-no-save run.
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
        """Opportunistic PPO epochs (rollout-bound elasticity): while the queue
        lacks a full next batch, replay extra shuffled group-complete mini-
        batch updates on the current batch (queue re-checked before every
        mini-batch; at most max_extra_epochs full epochs; runs before the
        parameter sync).
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
            # Stop-the-world save: freeze generation for the entire save (not
            # just the queue snapshot); the rollouter accounts the pause at
            # resume.
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
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def _save_timing_state(self, local_global_step_folder, save_start):
        """Persist the cumulative timing totals so a resumed run continues the
        virtual clock unchanged.
        """
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
            # Stop-the-world validation: block until the sweep completes
            # instead of training ahead on backlog; generation is paused, so
            # the whole pipeline freezes (a pure time translation of the no-
            # validation schedule).
            with marked_timer("timing_s/wait_validation_serialized", timing_param_sync):
                ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._step_wait_valid_time += timing_param_sync["timing_s/wait_validation_serialized"]
        self.logger.log(data=timing_param_sync, step=self.current_param_version)

    def _sync_triggers_validation(self) -> bool:
        """Mirror of the rollouter's validation decision in update_param_version;
        a false positive returns immediately and costs nothing.
        """
        test_freq = self.config.rollout.test_freq
        return test_freq > 0 and self.current_param_version > 0 and self.current_param_version % test_freq == 0

    def _open_virtual_step(self, consumer_end: float, queue_samples: list):
        """Start this step on the virtual (no-validation-no-save) timeline: at
        max(virtual free time, the batch's virtual-ready time from its
        stamps).
        """
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
        """Current position on the virtual timeline: step busy time so far minus
        validation waits and save time. Anchored when the first batch arrives.
        """
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
        """cumulative_training_time: the wall clock an identical run with neither
        validation nor checkpointing would have needed (virtual-clock replay).
        No-op until the rollouter reports its first training-dataset draw.
        """
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