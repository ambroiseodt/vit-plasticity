r"""
Codebase to analyze the plasticity of ViT components.

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
import pickle
from dataclasses import asdict, dataclass

import fire
import torch

from vitef.config import DEVICE, SAVING_DIR, set_seed
from vitef.data import build_loader, make_iterable
from vitef.models import build_model
from vitef.utils import get_numpy, get_valid_tensor, json_serializable, update_dict

logger = logging.getLogger("vitef")

# Paths
SAVE_DIR = SAVING_DIR / "analysis"


# ------------------------------------------------------------------------------
# Distance between hidden representations
# ------------------------------------------------------------------------------


def distance(x: torch.Tensor, y: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    r"""
    Compute the Frobenius distance between two sequences of tokens (seen as 2D matrices).

    Parameters:
    ----------
    x: torch.Tensor
        Batch of sequences of tokens clouds of dimension (N, n, d).
    y: torch.Tensor
        Batch of sequences of tokens clouds of dimension (N, n, d).
    reduction: str
        Specifies the reduction to apply to the output. Options are "none", "mean" and "sum".
        "none": no reduction will be applied, "mean": the mean of the output is taken,
        "sum": the output will be summed.

    Returns:
    -------
    dist: torch.Tensor
        Euclidean distance between two sets of points. If inputs have no batch dimension or
        if reduction is "mean" or "sum", the output is a scalar. Otherwise, if reduction
        is "none", the output is of dimension (N,).
    """

    # Convert to tensor with batch dimension if necessary
    x = get_valid_tensor(x)
    y = get_valid_tensor(y)

    # Compute distance
    dist = ((x - y) ** 2).flatten(start_dim=1).sum(dim=-1).sqrt()

    # Reduction
    match reduction.lower():
        case "none":
            pass
        case "mean":
            dist = dist.mean()
        case "sum":
            dist = dist.sum()
        case _:
            raise ValueError(f"Unknown reduction'{reduction}'. Choose between 'none', 'mean' or 'sum'.")

    return dist


# ------------------------------------------------------------------------------
# Analyze the plasticity of ViT components across layers
# ------------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    r"""
    Plasticity analysis config file
    """

    # Model
    model_name: str = "base"
    pretrained: bool = True
    patch_size: int = 16
    image_dim: tuple = (3, 224, 224)

    # Data
    dataset_name: str = "cifar10"
    batch_size: int = 128
    n_steps: int = 100

    # Randomness
    seed: int = 42

    # Device
    device: str = DEVICE

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.n_steps is None:
            self.n_steps = 1
        if self.seed is None:
            self.seed = 42


def analysis(config: AnalysisConfig) -> None:
    # ---------------------------------------------------------------------
    # Set seed for reproducibility
    # ---------------------------------------------------------------------

    set_seed(config.seed)

    # ---------------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------------

    logger.info("Building model.")
    model_config = {
        "implementation": "vit",
        "model_name": config.model_name,
        "pretrained": config.pretrained,
        "in21k": True,
        "patch_size": config.patch_size,
        "image_dim": config.image_dim,
    }

    model = build_model(config=model_config, device=config.device)
    logger.info("Done building model.")

    # ---------------------------------------------------------------------
    # Build dataloader
    # ---------------------------------------------------------------------

    logger.info("Building dataloaders.")

    # Subset of the pretraining data
    loader_config = {
        "dataset_name": "imagenet",
        "batch_size": config.batch_size,
        "mode": "val",
        "size": config.image_dim[-1],
    }
    loader1 = build_loader(config=loader_config)

    # Subset of the downstream data
    loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.batch_size,
        "mode": "test",
        "size": config.image_dim[-1],
    }
    loader2 = build_loader(config=loader_config)
    logger.info("Done building dataloaders.")

    # ---------------------------------------------------------------------
    # Set saving paths
    # ---------------------------------------------------------------------

    # Set config name
    config_name = f"analysis_{model.model_name}_pretrained_{config.pretrained}"
    config_name += f"_{config.dataset_name}"

    # Save config
    save_dir = SAVE_DIR / config_name
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        # JSON serializable alias
        exp_config = json_serializable(asdict(config))
        json.dump(exp_config, f, indent=4)

    # ---------------------------------------------------------------------
    # Recover layers outputs
    # ---------------------------------------------------------------------

    # Recover statitics
    model.eval()
    distances = {}
    step = 0

    # Make loaders iterable
    loader1, loader2 = map(lambda t: make_iterable(t), (loader1, loader2))
    iterator1 = iter(loader1)
    iterator2 = iter(loader2)

    # Loop of batches of data
    while step < config.n_steps:
        # Recover data
        x_batch1, _ = next(iterator1)
        x_batch2, _ = next(iterator2)

        # Move to device
        if config.device != "cpu":
            x_batch1 = x_batch1.pin_memory()
            x_batch2 = x_batch2.pin_memory()
        x_batch1 = x_batch1.to(device=config.device, non_blocking=True)
        x_batch2 = x_batch2.to(device=config.device, non_blocking=True)

        # Block decomposition
        outputs1 = model.get_decomposition(x_batch1)
        outputs2 = model.get_decomposition(x_batch2)

        # Free memory
        del x_batch1, x_batch2

        # Loop over layers
        for key in list(outputs1.keys()):
            # Recover the layer outputs
            z1 = outputs1.pop(key).to(config.device)
            z2 = outputs2.pop(key).to(config.device)

            # Compute the distance between the two layer outputs
            dist = distance(z1, z2, reduction="none")

            # Update dictionary and free memory
            value = get_numpy(dist)
            update_dict(value, dict_object=distances, key=key)
            del z1, z2

        # Free memory
        del outputs1, outputs2

        # Track progress
        if step % 10 == 0:
            print(f"Iteration {step}")

        # Update step
        step += 1

    # Saving results
    logger.info(f"Saving results in {save_dir}.")
    pickle.dump(distances, open(save_dir / "distances.pkl", "wb"))


def run_analysis(
    model_name: str = "base",
    pretrained: bool = True,
    patch_size: int = 16,
    image_dim: tuple = (3, 224, 224),
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    n_steps: int = 100,
    device: str = DEVICE,
) -> None:
    config = AnalysisConfig(
        model_name=model_name,
        pretrained=pretrained,
        patch_size=patch_size,
        image_dim=image_dim,
        dataset_name=dataset_name,
        batch_size=batch_size,
        n_steps=n_steps,
        device=device,
    )
    logger.info(f"Running experiments with {config=}.")
    analysis(config=config)


# %% Main
def main() -> None:
    r"""
    Launch the analysis from a configuration file specified by cli argument.

    Usage:
    ```
    To launch an experiment, for instance the analysis on GPU 0, run:
    ```bash
    python -m apps.vit.analysis run --device "cuda:0"
    ```
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"run": run_analysis})


# %% CLI

if __name__ == "__main__":
    main()
# %%
