r"""
Codebase to perform linear probing on vision transformers.

Notes
-----
Our implementation allows to conduct linear probing on pretrained and finetuned models,
and the log_dir argument from the config file indicates the checkpoint to recover.
In addition, probes can be taken over any transformer components across layers .

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

# %% Imports

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import logging
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from vitef.config import DEVICE, SAVING_DIR
from vitef.data import build_loader, build_train_val_loader
from vitef.models import build_model
from vitef.monitor import Utility, build_eval_orchestrator
from vitef.utils import get_numpy, json_serializable

logger = logging.getLogger("vitef")

# Paths
PROBE_DIR = SAVING_DIR / "probes"

# ------------------------------------------------------------------------------
# Online linear probing
# ------------------------------------------------------------------------------


@torch.inference_mode()
def get_embeddings(model: nn.Module, loader: DataLoader, cls_pooling: bool, device: str) -> tuple:
    """
    Recover embeddings after each component of a ViT model accross its layers.

    Parameters
    ----------
    model: nn.Module
        Model to evaluate.
    loader: DataLoader
        DataLoader to use for evaluation.
    cls_pooling: bool
        Whether to pool the CLS token or do average pooling before the linear probing.
    device: str
        Device.

    Returns
    -------
    metrics: dict
        Dictionary of metrics containing evaluation accuracy and loss.
    """
    embeddings = {}
    labels = []
    model.eval()

    for x_batch, y_batch in tqdm(loader):
        # Move to device
        x_batch = x_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        # Recover probes
        probes = model.get_probes(x_batch)

        # Loop over components
        for key in probes.keys():
            # Pool CLS token or mean pooling
            if cls_pooling:
                emb = probes[key][:, 0, :]
            else:
                emb = probes[key].mean(axis=1)

            # Recover embeddings
            if key in embeddings:
                embeddings[key].append(get_numpy(emb))
            else:
                embeddings[key] = [get_numpy(emb)]

        # Recover labels
        labels.append(get_numpy(y_batch))

    # Concatenate
    for key, value in embeddings.items():
        value = np.concatenate(value)
        value /= np.linalg.norm(value, axis=-1, keepdims=True)
        embeddings[key] = value

    labels = np.concatenate(labels)

    return embeddings, labels


def run_linear_probing(
    model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cls_pooling: bool, device: str, seed: int
) -> dict[str, Any]:
    r"""
    Run linear probing with Logistic Regression and return a dictionary of metrics.

    We note that the random state of the logistic regression is used only when
    the solver is among  'sag', 'saga' or 'liblinear' to shuffle the data.

    Parameters
    ----------
    model: nn.Module
        Model to evaluate.
    loader: DataLoader
        DataLoader to use for evaluation.
    cls_pooling: bool
        Whether to pool the CLS token or do average pooling before the linear probing.
    device: str
        Device.

    Returns
    -------
    metrics: dict
        Dictionary of metrics containing evaluation accuracy and loss.
    """

    # Embeddings
    train_embeddings, train_labels = get_embeddings(
        model=model, loader=train_loader, cls_pooling=cls_pooling, device=device
    )
    test_embeddings, test_labels = get_embeddings(
        model=model, loader=test_loader, cls_pooling=cls_pooling, device=device
    )

    # Linear classifier
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=seed),
    )

    # Train and test
    metrics = {}
    for key in train_embeddings:
        clf.fit(train_embeddings[key], train_labels.ravel())
        acc_test = clf.score(test_embeddings[key], test_labels.ravel())
        metrics[key] = acc_test
    return metrics


# ------------------------------------------------------------------------------
# Linear probing configuration
# ------------------------------------------------------------------------------


@dataclass
class LinearProbingConfig:
    r"""Linear probing config."""

    # Experiment
    log_dir: str = ""
    checkpoint_step: str | None = None
    finetuned: bool = True
    cls_pooling: bool = False

    # Data
    dataset_name: str = "cifar10"
    train_size: float = 0.8
    batch_size: int = 512
    val_batch_size: int = 512
    test_batch_size: int = 512

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
        # Logging directory
        self.log_dir = Path(self.log_dir)

        # Reproducibility
        if self.seed is None:
            self.seed = 42


# ------------------------------------------------------------------------------
# Main linear probing function
# ------------------------------------------------------------------------------


def linear_probing(config: LinearProbingConfig) -> None:
    r"""
    Run the linear probing from the config file.

    Parameters
    ----------
    config: LinearProbingConfig
        Configuration file to setup the model, the dataloaders and the linear classifier.
    """

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
        utils = Utility(config=orchestrator["utility"])
        context_stack.enter_context(utils)

        # Recover configurations and checkpoint directory
        exp_config = orchestrator["exp_config"]
        checkpoint_dir = orchestrator["checkpoint_dir"]
        with open(checkpoint_dir / "params.json") as f:
            model_config = json.load(f)

        # Model only pretrained on ImageNet-21k
        if not config.finetuned:
            model_config["pretrained"] = True
            model_config["in21k"] = True

        # ---------------------------------------------------------------------
        # Build model and recover checkpoints
        # ---------------------------------------------------------------------
        logger.info("Building model.")
        model = build_model(config=model_config, device=config.device, return_config=False)
        logger.info("Done building model.")

        # Load weights from finetuned model
        if config.finetuned:
            state_dict = {"model": model.state_dict()}
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
            model.load_state_dict(state_dict["model"])

        # ---------------------------------------------------------------------
        # Build dataloader
        # ---------------------------------------------------------------------
        logger.info("Building dataloaders.")
        loader_config = {
            "dataset_name": config.dataset_name,
            "batch_size": config.batch_size,
            "val_batch_size": config.val_batch_size,
            "size": exp_config["image_dim"][-1],
        }
        train_loader, val_loader, n_classes = build_train_val_loader(
            config=loader_config, train_size=config.train_size, return_n_classes=True
        )

        test_loader_config = {
            "dataset_name": config.dataset_name,
            "batch_size": config.test_batch_size,
            "mode": "test",
            "size": exp_config["image_dim"][-1],
        }
        test_loader = build_loader(config=test_loader_config, drop_last=False)
        logger.info("Done building dataloaders.")

        # ---------------------------------------------------------------------
        # Saving config
        # ---------------------------------------------------------------------
        if config.cls_pooling:
            if config.finetuned:
                save_dir = PROBE_DIR / f"{config.log_dir}_cls_pooling"
            else:
                save_dir = PROBE_DIR / f"vit_{config.dataset_name}_seed_{exp_config['seed']}_pretrained_cls_pooling"
        else:
            if config.finetuned:
                save_dir = PROBE_DIR / config.log_dir
            else:
                save_dir = PROBE_DIR / f"vit_{config.dataset_name}_seed_{exp_config['seed']}_pretrained"

        save_dir.mkdir(exist_ok=True, parents=True)
        with open(save_dir / "config.json", "w") as f:
            # JSON serializable alias
            exp_config = json_serializable(asdict(config))
            json.dump(exp_config, f, indent=4)

        # ---------------------------------------------------------------------
        # Linear probing
        # ---------------------------------------------------------------------
        metrics = run_linear_probing(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            cls_pooling=config.cls_pooling,
            device=config.device,
            seed=config.seed,
        )
        with open(save_dir / "linear_probing.json", "w") as f:
            json.dump(metrics, f, indent=4)

    logger.info("Linear probing done.")


# %% Main
def main() -> None:
    r"""
    Launch a linear probing job from a configuration file specified by cli argument.

    The command line interface here uses OmegaConf
    (see https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments).

    The behavior here is as follows:
    1. LinearProbingConfig is instantiated with its default values,
    2. Those default values are overridden with the ones in the provided config file,
    3. We override the result with the additional arguments provided through command line.

    Usage:
    To launch a linear probing job according to your_config.yaml, run:

    ```bash
    python -m apps.vit.linear_probing config=apps/vit/configs/your_config.yaml
    ``

    To overwrite some arguments via the cli, e.g., to set the device to cuda:0, run:

    ```bash
    python -m apps.vit.linear_probing config=apps/vit/configs/your_config.yaml device=cuda:0
    ```
    """
    # Recover config from CLI
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)

    # Remove 'config' attribute from config as the underlying dataclass does not have it
    del cli_args.config

    # Recover default config and merge all of them
    default_config = OmegaConf.structured(LinearProbingConfig())
    config = OmegaConf.merge(default_config, file_config, cli_args)
    config = OmegaConf.to_object(config)

    # Launch linear probing
    linear_probing(config)


# %% CLI
if __name__ == "__main__":
    main()
# %%
