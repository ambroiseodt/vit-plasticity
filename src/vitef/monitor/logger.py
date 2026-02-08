r"""
Logging manager to recover training and evaluation metrics.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from traceback import format_exception
from types import TracebackType
from typing import Any, Literal

import torch.nn as nn

from ..distributed import get_hostname, get_rank, is_master_process

logger = logging.getLogger("vitef")

# ------------------------------------------------------------------------------
# Logger for training and evaluation metrics
# ------------------------------------------------------------------------------


@dataclass
class LoggerConfig:
    r"""
    Logger configuration (both for stdout and metrics).

    Parameters
    ----------
    period: int
        Number of updates between each logging.
    level: str
        Logging level.
    stdout_path: str
        Path to the stdout log directory (set automatically by the orchestrator).
    metric_path: str
        Path to the metrics log directory (set automatically by the orchestrator).

    See also
    --------
    orchestrator.py
    """

    period: int = 0
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    stdout_path: str = ""
    metric_path: str = ""

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.period > 0:
            assert self.stdout_path, "stdout_path was not set."
            assert self.metric_path, "metric_path was not set."
            self.level = self.level.upper()


# ------------------------------------------------------------------------------
# Logging Manager
# ------------------------------------------------------------------------------


class Logger:
    r"""
    Logger manager.

    Parameters
    ----------
    config: config class with
        period: int
            Number of updates between each logging.
        level: str
            Logging level.
        stdout_path: str
            Path to the stdout log directory.
        metric_path: str
            Path to the metrics log directory).
    eval: bool, default=False
        Whether in evaluation or training mode.
    """

    def __init__(self, config: LoggerConfig, eval: bool = False) -> None:
        # Metric file
        rank = get_rank()
        self.path = Path(config.metric_path)
        self.path.mkdir(parents=True, exist_ok=True)
        if eval:
            # Clean file if not empty
            self.metric = str(self.path / "eval.jsonl")
            with open(self.metric, "w") as f:
                f.close()

        else:
            self.metric = str(self.path / f"raw_{rank}.jsonl")

        # Stdout file
        path = Path(config.stdout_path)
        path.mkdir(parents=True, exist_ok=True)
        stdout_file = path / f"device_{rank}.log"

        # Configure stdout
        # ...remove existing handler
        logger.handlers.clear()

        # ...initialize logging stream
        log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
        log_level = getattr(logging, config.level)
        logger.setLevel(log_level)
        handler = logging.FileHandler(stdout_file, "a")
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # ...log to console
        if is_master_process():
            handler = logging.StreamHandler()
            handler.setFormatter(log_format)
            logger.addHandler(handler)
            logger.info(f"Logging to {path}")

        logger.info(f"Running on machine {get_hostname()}")

        # Start timer
        self.start_time = time.time()

    def __enter__(self) -> "Logger":
        r"""Open logging files."""
        self.metric = open(self.metric, "a")
        return self

    def __call__(self, metrics: dict[str, Any]) -> None:
        r"""Report metrics to file."""
        metrics |= {"ts": time.time() - self.start_time}
        print(json.dumps(metrics), file=self.metric, flush=True)
        logger.info({k: round(v, 5) for k, v in metrics.items()})

    def report_model_size(self, model: nn.Module) -> None:
        r"""Report the number of trainable parameters of a model."""
        if is_master_process():
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            with open(self.path / "info_model.jsonl", "a") as f:
                print(json.dumps({"model_params": n_params}), file=f, flush=True)
            logger.info(f"The model has {n_params} trainable parameters.")

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        r"""Close logging files. Log exceptions if any."""
        self.metric.close()
        if exc is not None:
            logger.error(f"Exception: {value}")
            logger.info("".join(format_exception(exc, value, tb)))
