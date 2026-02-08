r"""
Codebase to evaluate vision transformers.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"


import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vitef.config import DEVICE
from vitef.data import build_loader
from vitef.models import build_model
from vitef.monitor import Logger, Utility, build_eval_orchestrator

from .utils import EvalState

logger = logging.getLogger("vitef")

# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------


@torch.inference_mode()
def run_evaluation(model: nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
    r"""
    Run evaluation and return a dictionary of metrics.

    Parameters
    ----------
    model: nn.Module
        Model to evaluate.
    loader: DataLoader
        DataLoader to use for evaluation.
    device: str
        Device.

    Returns
    -------
    metrics: dict
        Dictionary of metrics containing evaluation accuracy and loss.
    """

    # Initialize evaluation state
    state = EvalState()

    # Setup evaluation mode
    model.eval()
    for x_batch, y_batch in loader:
        # Move to device
        x_batch = x_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        # Forward
        preds = model(x_batch)
        loss = F.cross_entropy(preds, y_batch)
        y_pred = preds.argmax(dim=-1)
        accuracy = (y_pred == y_batch).float().mean().item()

        # Update state
        state.accuracy += accuracy
        state.loss += loss.item()
        state.step += 1

    # Recover metric dictionnary
    metrics = {"eval_acc": state.accuracy / state.step, "eval_loss": state.loss / state.step}

    return metrics


# ------------------------------------------------------------------------------
# Evaluation configuration
# ------------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    r"""Evaluation config."""

    # Experiment
    log_dir: str = ""
    checkpoint_step: str | None = None

    # Data
    dataset_name: str | None = None
    batch_size: int = 512

    # Device
    device: str = DEVICE

    # Reproducibility
    seed: int = 42

    # Orchestrator
    logging_level: str = "INFO"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        r"""
        Set up the seed.
        """
        # Make log_dir a Path
        self.log_dir = Path(self.log_dir)

        # Reproducibility
        if self.seed is None:
            self.seed = 42


# ------------------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------------------


@torch.inference_mode()
def eval(config: EvaluationConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Set up orchestrator to handle logging and utility for reproducibility
        # ---------------------------------------------------------------------

        orchestrator_config = {
            "log_dir": config.log_dir,
            "checkpoint_step": config.checkpoint_step,
            "logging_level": config.logging_level,
            "seed": config.seed,
        }
        orchestrator = build_eval_orchestrator(config=orchestrator_config)
        metric_logger = Logger(config=orchestrator["logger"], eval=True)
        context_stack.enter_context(metric_logger)
        utils = Utility(config=orchestrator["utility"])
        context_stack.enter_context(utils)

        # Recover configurations and checkpoint directory
        exp_config = orchestrator["exp_config"]
        checkpoint_dir = orchestrator["checkpoint_dir"]
        with open(checkpoint_dir / "params.json") as f:
            model_config = json.load(f)

        # Set dataset to evaluate on if not given
        if config.dataset_name is None:
            config.dataset_name = exp_config["dataset_name"]

        # ---------------------------------------------------------------------
        # Build dataloader
        # ---------------------------------------------------------------------
        logger.info("Building dataloader.")
        loader_config = {
            "dataset_name": config.dataset_name,
            "batch_size": config.batch_size,
            "mode": "test",
            "size": exp_config["image_dim"][-1],
        }
        test_loader, n_classes = build_loader(config=loader_config, drop_last=False, return_n_classes=True)
        logger.info("Done building dataloader.")

        # ---------------------------------------------------------------------
        # Build model and recover checkpoints
        # ---------------------------------------------------------------------
        logger.info("Building model.")
        model = build_model(config=model_config, device=config.device, return_config=False)
        logger.info("Done building model.")

        # Load weights from checkpoints
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
        model.load_state_dict(state_dict["model"])

        # ---------------------------------------------------------------------
        # Evaluation loop
        # ---------------------------------------------------------------------
        metrics = run_evaluation(model=model, loader=test_loader, device=config.device)
        metric_logger({"test_acc": metrics["eval_acc"]})
        logger.info(f"Test accuracy: {metrics['eval_acc'] * 100:.2f}%")

    logger.info("Evaluation done.")


# %% Main
def main() -> None:
    r"""
    Launch an evaluation job from a configuration file specified by cli argument.

    The command line interface here uses OmegaConf
    (see https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments).

    The behavior here is as follows:
    1. EvaluationConfig is instantiated with its default values,
    2. Those default values are overridden with the ones in the provided config file,
    3. We override the result with the additional arguments provided through command line.

    Usage:
    To launch the evaluation according to your_config.yaml, run:

    ```bash
    python -m apps.vit.eval config=apps/vit/configs/your_config.yaml
    ```

    To overwrite some arguments via the cli, e.g., to set the device to cuda:0, run:

    ```bash
    python -m apps.vit.eval config=apps/vit/configs/your_config.yaml device=cuda:0
    ```
    """

    # Recover config from CLI
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)

    # Remove 'config' attribute from config as the underlying dataclass does not have it
    del cli_args.config

    # Recover default config and merge all of them
    default_config = OmegaConf.structured(EvaluationConfig())
    config = OmegaConf.merge(default_config, file_config, cli_args)
    config = OmegaConf.to_object(config)

    # Launch evaluation
    eval(config)


# %% CLI
if __name__ == "__main__":
    main()
# %%
