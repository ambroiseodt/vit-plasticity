r"""
Checkpoint manager for training and evaluation.

Notes
-----
When using the Checkpointer class, make sure that the checkpoint directory
is either empty or does not contain corrupted checkpoints to avoid errors.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
import re
import shutil
from asyncio import Future
from dataclasses import dataclass
from pathlib import Path, PosixPath
from types import TracebackType

import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ..distributed import get_rank, is_master_process
from ..utils import DataclassProtocol, json_serializable

logger = logging.getLogger("vitef")


# ------------------------------------------------------------------------------
# Checkpointing logic at training time
# ------------------------------------------------------------------------------


@dataclass
class CheckpointerConfig:
    r"""
    Checkpoint configuration.

    Parameters
    ----------
    period: int
        Number of updates between each checkpoint.
    n_kept: int
        Number of checkpoints to keep.
    path: str
        Path to the checkpoint directory (set automatically by the orchestrator).

    See also
    --------
    orchestrator.py
    """

    period: int = 0
    n_kept: int = 0
    path: str = ""

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.period > 0:
            assert self.path, "Path was not set."


class Checkpointer:
    r"""
    Checkpoint manager.

    Parameters
    ----------
    config: config class with
        period: int
            Number of updates between each checkpoint.
        n_kept: int
            Number of checkpoints to keep.
        path: str
            Path to the checkpoint directory.
    model: nn.Module
        Model to checkpoint.
    model_config: DataclassProtocol
        Model configuration.
    optimizer: Optimizer
        Optimizer to checkpoint.
    training_state: Stateful
        Training state objects to checkpoint.
    """

    folder_name = "{:010d}"
    re_folder = r"\d{10}"
    re_digits = re.compile(r"\d+")

    def __init__(
        self,
        config: CheckpointerConfig,
        model: nn.Module,
        model_config: DataclassProtocol = None,
        optimizer: Optimizer = None,
        training_state: Stateful = None,
    ):
        # Initialize from config
        self.period = config.period
        self.n_kept = config.n_kept
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Create alias for the objects to monitor
        self.model = model
        self.model_config = model_config
        self.optimizer = optimizer
        self.training_state = training_state

        # Initialize current state
        self.device_rank = get_rank()
        self.saved_step = 0
        self.step = 0
        self.process: Future = None

    def sync_step(self, step: int) -> None:
        r"""Synchronize the step with the given value."""
        self.saved_step = self.step = step

    def __enter__(self) -> "Checkpointer":
        r"""Enter the checkpoint context by loading the last checkpoint."""
        path = self.get_last_checkpoint_path(self.path)
        if path:
            self.load(path)
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        r"""Exit the checkpoint context by saving checkpoint if needed."""

        # Save checkpoint when exiting if not done already
        if self.saved_step != self.step:
            self.update()

        if self.process is not None:
            logger.info("Waiting for final checkpoint to complete.")
            self.process.result()

    def __call__(self) -> None:
        r"""Call update function periodically."""
        self.step += 1
        if self.period <= 0:
            return
        if self.step % self.period == 0:
            self.update()

    def update(self, eval_flag: str = "") -> None:
        r"""
        Checkpoint model, optimizer, scheduler and training state.

        Parameters
        ----------
        eval: str, default=""
            Whether to save the checkpoint for evaluation.
        """
        path = self.path / self.folder_name.format(self.step)
        path.mkdir(parents=False, exist_ok=True)

        # Add evaluation flag, if needed
        if eval_flag:
            (path / f"eval_{eval_flag}").touch()

        # Do not checkpoint twice
        if self.saved_step == self.step:
            return

        self.save(path)

        self._cleaning()
        self.saved_step = self.step

    def load(self, path: str) -> None:
        r"""
        Load checkpoint from path

        Parameters
        ----------
        path: str
            Path to the checkpoint to load.
        """

        logger.info(f"Loading checkpoint from {str(path)}.")
        state_dict = self.get_state_dict()
        dcp.load(state_dict=state_dict, checkpoint_id=path)

        logger.info("Loading model weights and optimizer state.")
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )

        logger.info("Loading training state.")
        self.training_state.load_state_dict(state_dict["training"])

    def save(self, path: str) -> None:
        r"""
        Save checkpoint to path.

        Parameters
        ----------
        path: str
            Path to save the checkpoint.
        """

        if self.process is not None:
            logger.info("Waiting for previous checkpoint to complete.")
            self.process.result()

        logger.info(f"Saving checkpoint at step {self.step} to {str(path)}.")
        state_dict = self.get_state_dict()
        self.process = dcp.async_save(state_dict, checkpoint_id=path)

        if self.model_config is not None and is_master_process():
            # JSON serializable alias
            model_config = json_serializable(self.model_config)
            with open(path / "params.json", "w") as f:
                json.dump(model_config, f)

    def get_state_dict(self) -> dict[str, dict]:
        r"""Return state dict of all tracked stateful objects."""
        model_sd, optimizer_sd = get_state_dict(model=self.model, optimizers=self.optimizer)
        state_dict = {"model": model_sd, "optim": optimizer_sd}
        state_dict |= {"training": self.training_state.state_dict()}
        return state_dict

    @classmethod
    def get_last_checkpoint_path(cls, path: str) -> str:
        r"""Get last existing checkpoint."""
        folders = cls._list_checkpoints(path)
        if folders:
            return max(folders, key=lambda p: cls._get_key_step(p.name))
        return ""

    def _cleaning(self) -> None:
        r"""Clean up old checkpoints."""
        if self.n_kept <= 0 or not is_master_process():
            return
        all_checkpoints = self._list_checkpoints(self.path)
        all_checkpoints.sort(key=lambda p: self._get_key_step(p.name))
        for prefix in all_checkpoints[: -self.n_kept]:
            if not any(prefix.glob("eval_*")):
                logger.info(f"Removing: {str(prefix)}")
                shutil.rmtree(prefix)

    @classmethod
    def _list_checkpoints(cls, path: str) -> list[PosixPath]:
        r"""List all existing checkpoints."""
        return [p for p in path.iterdir() if p.is_dir() and re.match(cls.re_folder, p.name)]

    @classmethod
    def _get_key_step(cls, name: str) -> int:
        return int(re.findall(cls.re_digits, name)[-1])
