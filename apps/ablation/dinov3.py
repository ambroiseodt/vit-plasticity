r"""
Codebase to analyze the plasticity of Dinov3 components.

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
import torch.nn as nn
from transformers import DINOv3ViTModel, PreTrainedModel

from vitef.config import DEVICE, SAVING_DIR, set_seed
from vitef.data import build_loader, make_iterable
from vitef.utils import get_numpy, get_valid_tensor, json_serializable, move_to_cpu, update_dict

logger = logging.getLogger("vitef")

# Paths
SAVE_DIR = SAVING_DIR / "analysis/ablation"


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------


@dataclass
class Dinov3Config:
    r"""
    Dinov3 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "vits16", "vitb16", "vitl16", "vit7b16", "vits16plus" and "vith16plus".
    device: str
        Device to use.
    """

    model_size: str = "base"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["vits16", "vitb16", "vitl16", "vit7b16", "vits16plus", "vith16plus"]:
            raise ValueError(f"Invalid model size {self.model_size}.")


class Dinov3(nn.Module):
    def __init__(self, config: Dinov3Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"facebook/dinov3-{config.model_size}-pretrain-lvd1689m"
        self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "facebook/dinov3-vith16plus-pretrain-lvd1689m".
        device: str
            Device to use.

        Notes
        -----
        The model is set to evaluation mode after loading (dropout disabled).
        Call model.train() explicitly to re-enable training mode.
        """
        available_models = [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = DINOv3ViTModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    @torch.inference_mode()
    def get_decomposition(self, x: torch.Tensor) -> dict:
        r"""
        Recover the outputs of ViT components across layers for the same input.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, *)
            Batch of input data to be patched and embedded in sequences of tokens of dimension E.

        Returns
        -------
        outputs: Dictionnary of output of each model's layer.
        """
        outputs = {}

        # Embedding layer
        out = self.model.embeddings(x)
        position_embeddings = self.model.rope_embeddings(x)
        outputs["embedding"] = move_to_cpu(out)

        # Transformer blocks
        layers = self.model.layer
        for i, block in enumerate(layers):
            # LN1
            block_out = block.norm1(out)
            key = "attn_norm"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # MHA
            block_out, _ = block.attention(out, position_embeddings=position_embeddings)
            key = "attn"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # LN2
            block_out = block.norm2(out)
            key = "ffn_norm"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # FC1
            block_out = block.mlp.up_proj(out)
            key = "ffn_fc1"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # FC2
            zero_out = torch.zeros_like(out)

            # For Dinov3-7B, the intermediate dimension is twice as big as the embedding dimension
            if "vit7b16" in self.model_name:
                expanded_out = torch.cat((out, zero_out), dim=-1)

            # Otherwise, the intermediate dimension is 4 times the embedding dimension
            else:
                expanded_out = torch.cat((out, zero_out, zero_out, zero_out), dim=-1)
            block_out = block.mlp.down_proj(expanded_out)
            key = "ffn_fc2"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

        return outputs


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
    model_size: str = "vitb16"
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
    model_config = Dinov3Config(model_size=config.model_size, device=config.device)
    model = Dinov3(config=model_config)
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

    # Save config
    config_name = f"analysis_dinov3_{config.model_size}_{config.dataset_name}"
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
    model_size: str = "base",
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    n_steps: int = 100,
    device: str = DEVICE,
) -> None:
    config = AnalysisConfig(
        model_size=model_size,
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
