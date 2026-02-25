r"""
Codebase to finetune vision transformers while freezing some modules if specified.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

# %% Imports

import os

os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass

import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_

from vitef.config import DEVICE
from vitef.data import build_train_val_loader, make_iterable
from vitef.models import build_model
from vitef.monitor import Checkpointer, Logger, Utility, build_orchestrator
from vitef.optim import build_optimizer, build_scheduler

from .eval import run_evaluation
from .utils import TrainingState, freeze_model

logger = logging.getLogger("vitef")

# ------------------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    r"""Training config."""

    # Model
    model_name: str = "base"
    patch_size: int = 16
    image_dim: tuple = (3, 224, 224)
    components: list[str] | None = None

    # Data
    dataset_name: str = "cifar10"
    train_size: float = 0.8
    batch_size: int = 512
    val_batch_size: int = 512

    # Training
    n_steps: int = 10_000
    grad_acc_steps: int = 1
    grad_clip: float | None = None

    # Evaluation
    eval_period: int = 1000

    # Optimizer
    optimizer: str = "sgd"
    lr: float = 1e-3
    momentum: float = 0.9

    # Scheduler
    scheduler: str = "constant"
    min_factor: float = 0

    # Device
    device: str = DEVICE

    # Orchestrator
    log_dir: str = ""
    overwrite: bool = False
    logging_period: int = 10
    logging_level: str = "INFO"
    seed: int = 42
    utility_period: int = 1000

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        r"""
        Check the validity of arguments and set up the seed.
        """
        # Ensure that the evaluation period is valid
        if (self.eval_period <= 0) or (self.eval_period > self.n_steps):
            self.eval_period = self.n_steps

        # Reproducibility
        if self.seed is None:
            self.seed = 42


# ------------------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------------------


def train(config: TrainingConfig) -> None:
    r"""
    Run the training from the config file, saving the checkpoint of the best evaluated model.

    Parameters
    ----------
    config: TrainingConfig
        Configuration file to setup the model, the dataloaders and the training parameters.
    """

    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Set up orchestrator to handle logging, checkpointing and utility
        # ---------------------------------------------------------------------

        orchestrator_config = {
            "log_dir": config.log_dir,
            "overwrite": config.overwrite,
            "config_file": asdict(config),
            "checkpoint_period": 0,
            "checkpoint_n_kept": 1,
            "logging_period": config.logging_period,
            "logging_level": config.logging_level,
            "seed": config.seed,
            "utility_period": config.utility_period,
        }
        orchestrator = build_orchestrator(config=orchestrator_config)

        metric_logger = Logger(config=orchestrator["logger"], eval=False)
        context_stack.enter_context(metric_logger)

        utils = Utility(config=orchestrator["utility"])
        context_stack.enter_context(utils)

        # ---------------------------------------------------------------------
        # Build dataloader
        # ---------------------------------------------------------------------

        logger.info("Building dataloaders.")
        loader_config = {
            "dataset_name": config.dataset_name,
            "batch_size": config.batch_size,
            "val_batch_size": config.val_batch_size,
            "size": config.image_dim[-1],
        }
        train_loader, val_loader, n_classes = build_train_val_loader(
            config=loader_config, train_size=config.train_size, return_n_classes=True
        )
        logger.info("Done building dataloaders.")

        # ---------------------------------------------------------------------
        # Build model while freezing some components if specified
        # ---------------------------------------------------------------------
        logger.info("Building model.")
        model_config = {
            "implementation": "vit",
            "model_name": config.model_name,
            "pretrained": True,
            "in21k": True,
            "patch_size": config.patch_size,
            "image_dim": config.image_dim,
            "finetuning": True,
            "n_classes": n_classes,
        }

        model, model_config = build_model(
            config=model_config,
            device=config.device,
            return_config=True,
        )
        freeze_model(model=model, components=config.components)
        logger.info("Done building model.")

        # ---------------------------------------------------------------------
        # Build optimizer, scheduler and training state
        # ---------------------------------------------------------------------

        logger.info("Building optimizer.")
        optim_config = {
            "optimizer": config.optimizer,
            "lr": config.lr,
            "momentum": config.momentum,
        }
        optimizer = build_optimizer(config=optim_config, model=model)
        scheduler_config = {"scheduler": config.scheduler, "min_factor": config.min_factor}
        scheduler = build_scheduler(
            config=scheduler_config,
            optimizer=optimizer,
            n_steps=config.n_steps,
        )
        training_state = TrainingState(step=0, acc_step=0, scheduler=scheduler)
        logger.info("Done building optimizer.")

        # ---------------------------------------------------------------------
        # Build checkpoint
        # ---------------------------------------------------------------------

        checkpoint = Checkpointer(
            config=orchestrator["checkpointer"],
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            training_state=training_state,
        )
        context_stack.enter_context(checkpoint)

        # ---------------------------------------------------------------------
        # Recover the number of trainable parameters
        # ---------------------------------------------------------------------

        metric_logger.report_model_size(model)
        current_time, current_step = time.time(), training_state.step

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        # Synchronize checkpoint with the optimizer
        checkpoint.sync_step(training_state.step)

        # Aliases
        logging_period = orchestrator["logger"].period
        eval_period = config.eval_period
        best_eval_accuracy = 0.0
        best_eval_step = 0

        # Set training mode
        model.train()

        # Make training loader iterable
        train_loader = make_iterable(train_loader)
        iterator = iter(train_loader)

        # Loop over batches
        while training_state.step < config.n_steps:
            # Accumulation step
            training_state.acc_step += 1
            training_state.acc_step = training_state.acc_step % config.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data
            # -----------------------------------------------------------------
            x_batch, y_batch = next(iterator)
            if config.device != "cpu":
                x_batch = x_batch.pin_memory()
                y_batch = y_batch.pin_memory()
            x_batch = x_batch.to(device=config.device, non_blocking=True)
            y_batch = y_batch.to(device=config.device, non_blocking=True)

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            # Forward propagation
            preds = model(x_batch)
            loss = F.cross_entropy(preds, y_batch)

            # Rescale when using gradient accumulation (backprop on mean, not sum)
            loss = loss / config.grad_acc_steps

            # Backward propagation
            loss.backward()

            # Gradient accumulation
            if training_state.acc_step != 0:
                continue

            # Clip gradient norm
            grad_clip = config.grad_clip if config.grad_clip is not None else float("inf")
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            training_state.step += 1

            # Call monitor for garbage collection and checkpointing
            utils()
            checkpoint()

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            # Alias
            step = training_state.step

            if (logging_period > 0) and (step % logging_period == 0):
                logger.info(f"Metric logging at step {step}.")

                # Undo gradient accumulation scaling
                loss = loss.item() * config.grad_acc_steps

                # Optimization information
                lr = optimizer.param_groups[0]["lr"]
                grad_norm = grad_norm.item()

                # Time information
                elapsed_time = time.time() - current_time
                elapsed_steps = step - current_step
                current_time, current_step = time.time(), training_state.step

                # Save metrics
                metrics = {
                    "loss": loss,
                    "step": step,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "elapsed_steps": elapsed_steps,
                    "ts": elapsed_time,
                }
                metric_logger(metrics)

            # -----------------------------------------------------------------
            # Evaluation
            # -----------------------------------------------------------------
            if (eval_period > 0) and (step % eval_period == 0):
                # Saving metrics
                logger.info(f"Evaluation at step {step}.")
                metrics = run_evaluation(model=model, loader=val_loader, device=config.device)
                metrics |= {"step": step}
                metric_logger(metrics)

                # Checkpointing the current best model
                if metrics["eval_acc"] > best_eval_accuracy:
                    best_eval_accuracy = metrics["eval_acc"]
                    best_eval_step = metrics["step"]
                    logger.info(
                        f"Saving the current best model (validation accuracy of {best_eval_accuracy * 100:.2f}%)."
                    )
                    checkpoint.update()

        # Synchronize to ensure the last checkpoint is the best evaluated model
        checkpoint.sync_step(best_eval_step)

    logger.info("Training done.")


# %% Main
def main() -> None:
    r"""
    Launch a training job from a configuration file specified by cli argument.

    The command line interface here uses OmegaConf
    (see https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments).

    The behavior here is as follows:
    1. TrainingConfig is instantiated with its default values,
    2. Those default values are overridden with the ones in the provided config file,
    3. We override the result with the additional arguments provided through command line.

    Usage:
    To launch a training according to your_config.yaml, run:

    ```bash
    python -m apps.vit.train config=apps/vit/configs/your_config.yaml
    ``

    To overwrite some arguments via the cli, e.g., to set the device to cuda:0, run:

    ```bash
    python -m apps.vit.train config=apps/vit/configs/your_config.yaml device=cuda:0
    ```
    """
    # Recover config from CLI
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)

    # Remove 'config' attribute from config as the underlying dataclass does not have it
    del cli_args.config

    # Recover default config and merge all of them
    default_config = OmegaConf.structured(TrainingConfig())
    config = OmegaConf.merge(default_config, file_config, cli_args)
    config = OmegaConf.to_object(config)

    # Launch training
    train(config)


# %% CLI
if __name__ == "__main__":
    main()
# %%
