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

import os
import random
import shutil

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name, get_torch_device


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads the following states in a SPMD way:
    - model
    - optimizer
    - lr_scheduler
    - extra_states

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig | CheckpointConfig = None,
    ):
        self.checkpoint_config = checkpoint_config
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    @property
    def should_save_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf
        model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_save_hf_metadata(self) -> bool:
        """
        Returns True if the huggingface config/tokenizer/generation config should be written next to the
        weights. They are needed whenever weights are saved in any form: a sharded checkpoint is merged
        against them later, and an 'hf_model' directory is only loadable (vLLM, from_pretrained) with them.
        """
        return self.should_save_model or self.should_save_hf_model

    @property
    def should_save_dist_checkpoint(self) -> bool:
        """
        Returns True if anything belongs in the distributed checkpoint. With 'hf_model' alone there is
        nothing to write, and saving anyway leaves an empty checkpoint directory that looks loadable.
        """
        return self.should_save_model or self.should_save_optimizer or self.should_save_extra

    @property
    def should_load_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None

    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    """
    Determine if checkpoint should be saved based on capacity esi expiration.

    Args:
        max_steps_duration: Max estimated time (seconds) required to complete one training step
        save_ckpt_duration: Estimated time (seconds) required to save checkpoint (default: 60)
        redundant_time: Additional buffer time (seconds) for unexpected delays (default: 0)
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws
    if exp_ts_mlp:
        try:
            import time

            remaining = float(exp_ts_mlp) - time.time()
        except ValueError:
            return False
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )
    elif exp_ts_aws:
        from datetime import datetime, timedelta

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))
        time_difference = expiration_time - datetime.now()
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60
        return time_difference < timedelta(minutes=threshold_minutes)
    else:
        return False


def restores_model_weights(load_contents) -> bool:
    """Return True if loading a checkpoint with these contents restores the model weights.

    A resume whose ``load_contents`` lacks 'model' loads nothing into the model: an 'hf_model'
    checkpoint holds the weights in huggingface format, which the checkpoint managers write but never
    read back. Such a resume silently continues from the pretrained weights, so callers should refuse
    it rather than let it look like it worked.
    """
    if load_contents is None:
        return True  # None means the manager's default, which includes 'model'
    return "model" in load_contents


def resync_optimizer_main_params(optimizer, *, loaded_model: bool, loaded_optimizer: bool) -> bool:
    """Refresh an optimizer's master parameter copies from the (just restored) model parameters.

    Megatron's mixed-precision and distributed optimizers keep fp32 master copies of the parameters,
    made when the optimizer is built - which happens before any checkpoint is loaded. Restoring the
    model but not the optimizer updates the low-precision model parameters in place and leaves those
    master copies holding the pre-load values, so the first ``optimizer.step()`` copies them back over
    the restored weights and the run silently reverts to the pretrained model.

    Optimizers without master copies (torch/FSDP) step on the model parameters themselves and expose no
    such hook, so they are left untouched.

    Returns:
        bool: True if a resync was performed.
    """
    if not loaded_model or loaded_optimizer or optimizer is None:
        return False
    reload_model_params = getattr(optimizer, "reload_model_params", None)
    if reload_model_params is None:
        return False
    reload_model_params()
    return True
