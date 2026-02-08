r"""
Distributed computing manager.

Notes
-----
The current implementation in apps/vit does not require parallelization since models have reasonable
size. As such, the computing manager is a work in progress and is subject to potential bugs.
The code follows https://github.com/ambroiseodt/itl and https://github.com/facebookresearch/lingua
but we do not support Slurm or custom Triton kernels.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import os
import socket
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from logging import getLogger
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import DEVICE
from .utils import build_with_type_check

logger = getLogger("vitef")


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------


@lru_cache
def is_torchrun_job() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache
def is_distributed_job() -> bool:
    return is_torchrun_job()


@lru_cache
def get_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["RANK"])
    else:
        return 0


@lru_cache
def get_local_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


@lru_cache
def get_world_size() -> int:
    if is_torchrun_job():
        return int(os.environ["WORLD_SIZE"])
    else:
        return 1


@lru_cache
def is_master_process() -> bool:
    return get_rank() == 0


@lru_cache
def get_hostname() -> str:
    return socket.gethostname()


# ------------------------------------------------------------------------------
# OS Environment
# ------------------------------------------------------------------------------


@dataclass
class OsEnvironment:
    r"""Configuration for the OS environment."""

    OMP_NUM_THREADS: str = "1"


def set_os_environment(config: OsEnvironment) -> None:
    r"""
    Set OS environment variables based on configuration.
    """
    env_vars = asdict(config)
    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            logger.info(f"OS: Setting {name} to {value}")


@contextmanager
def clean_environment() -> Generator[None, None, None]:
    r"""Context that momentarily clean OS environment variables."""
    distrib_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    )
    os_environment = {x: os.environ.pop(x) for x in os.environ if x in distrib_names}
    try:
        yield
    finally:
        os.environ.update(os_environment)


# ------------------------------------------------------------------------------
# Distributed computing configuration and manager
# ------------------------------------------------------------------------------


@dataclass
class ComputingManagerConfig:
    r"""Computing manager configuration."""

    device: str = DEVICE
    backend: str = "cpu:gloo,cuda:nccl"
    dp: int = 0
    tp: int = 1

    # submanager
    os_environment: OsEnvironment = field(default_factory=OsEnvironment)

    def __post_init__(self) -> None:
        # Default device: cuda if available else cpu
        if not self.device:
            self.device = DEVICE

        # Ensure valid dp argument
        if not self.dp:
            self.dp = get_world_size() // self.tp


class ComputingManager:
    r"""
    Distributed computing manager that handles the initialization and destruction of the distributed environment.

    Parameters
    ----------
    config: confuguration class with
        device: str
            Device.
        backend: str
            torch.distributed built-in backend.
        dp: int
            Number of gpus to use for data parallel.
        tp: int
            Number of gpus to use for tensor parallel.
    """

    def __init__(self, config: ComputingManagerConfig):
        self.backend = config.backend
        self.device = torch.device(config.device)
        self.tp = config.tp
        self.dp = config.dp
        nb_devices = get_world_size()
        assert self.device.type == "cpu" or self.dp * self.tp == nb_devices, (
            f"DP * TP must equal the number of GPUs {self.tp} * {self.dp} != {nb_devices}"
        )
        set_os_environment(config.os_environment)
        self.tp_mesh: DeviceMesh
        self.dp_mesh: DeviceMesh

    def __enter__(self):
        r"""Initialize distributed environment"""
        if not is_distributed_job():
            self.dp = self.tp = 1
            self.tp_mesh = self.dp_mesh = None
            logger.info(f"Running on {self.device}")
            return self

        rank = get_rank()
        local_rank = get_local_rank()
        world_size = get_world_size()
        dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
        logger.info(f"Setting up device ranked {rank + 1} / {world_size}")
        self.device = torch.device(f"cuda:{local_rank}")

        logger.info("Creating device mesh")
        mesh = init_device_mesh(self.device.type, mesh_shape=(self.dp, self.tp), mesh_dim_names=("dp", "tp"))
        self.tp_mesh = mesh["tp"]
        self.dp_mesh = mesh["dp"]
        return self

    def build_model(self, model: nn.Module, tp_plan: dict[str, ParallelStyle] = None) -> nn.Module:
        r"""
        Initialize the model by casting it to the device, compiling and parallelizing it according to configuration.

        Parameters
        ----------
        model: nn.Module
            Model to be parallelized.
        tp_plan: dict
            Tensor parallelization plan.

        Notes
        -----
        Current work in progress to ensure the proper tensor and data parallel behavior.
        """
        model = model.to(device=self.device)

        if self.tp > 1:
            logger.info("Parallelizing model with tensor parallel")
            model = parallelize_module(model, device_mesh=self.tp_mesh, parallelize_plan=tp_plan)

        if self.dp > 1:
            if self.tp > 1:
                logger.info("Parallelizing model with fully sharded data parallel")
                model = FSDP(model, device_mesh=self.dp_mesh, use_orig_params=True)
            else:
                logger.info("Parallelizing model with data parallel")
                model = DDP(model)

        return model

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        r"""Exit distributed environment"""
        rank = get_rank()
        world_size = get_world_size()
        logger.info(f"Exiting distributed environment {rank + 1} / {world_size}")
        if is_distributed_job():
            dist.destroy_process_group()


def build_manager(config: dict[str, Any]) -> dict[str, Any]:
    r"""
    Build the computing manager to set the OS enviroment and implement data and tensor parallel when needed.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    manager: ComputingManager
        Distributed computing manager.
    """
    config_obj = build_with_type_check(ComputingManagerConfig, config)
    manager = ComputingManager(config=config_obj)

    return manager


# ------------------------------------------------------------------------------
# Access root model when wrapped in parallelization layers
# ------------------------------------------------------------------------------


def get_raw_model(model: nn.Module) -> nn.Module:
    r"""Recover the raw model if it is wrapped in distributed module."""
    if isinstance(model, DDP) or isinstance(model, FSDP):
        return get_raw_model(model.module)
    else:
        return model
