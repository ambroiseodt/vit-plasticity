r"""
Codebase to analyze the plasticity of GPT2 components.

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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding, GPT2Model, PreTrainedModel

from vitef.config import DEVICE, SAVING_DIR, set_seed
from vitef.data import make_iterable
from vitef.utils import get_numpy, get_valid_tensor, json_serializable, move_to_cpu, update_dict

logger = logging.getLogger("vitef")

# Paths
SAVE_DIR = SAVING_DIR / "analysis/ablation"


# ------------------------------------------------------------------------------
# Text loader
# ------------------------------------------------------------------------------
def build_text_loader(config: dict) -> DataLoader:
    r"""
    Build a DataLoader for a text dataset.

    Supported datasets
    ------------------
    "wikitext"
        WikiText-103-raw-v1. Long-form encyclopedic prose, close to LLM
        pretraining corpora. Analogous to ImageNet in the DINOv3 study.
    "ag_news"
        AG News Corpus. Short, topic-diverse news snippets across 4 categories.
        Analogous to CIFAR-10 in the DINOv3 study.

    Parameters
    ----------
    config: dict
        Configuration dictionary with keys:
            dataset_name : str   — "wikitext" or "ag_news".
            batch_size   : int   — Number of sequences per batch.
            mode         : str   — Dataset split: "train", "val" or "test".
            seq_len      : int   — Number of tokens per sequence (pad/truncate).
            tokenizer            — HuggingFace tokeniser.

    Returns
    -------
    loader : DataLoader
    """
    from datasets import load_dataset

    dataset_name = config["dataset_name"]
    batch_size = config["batch_size"]
    mode = config["mode"]
    seq_len = config["seq_len"]
    tokenizer = config["tokenizer"]

    # Map mode to HuggingFace split names
    split_map = {"train": "train", "val": "validation", "test": "test"}
    split = split_map[mode]

    if dataset_name == "wikitext":
        raw = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        texts = [row["text"] for row in raw if row["text"].strip()]
    elif dataset_name == "ag_news":
        # AG News has no validation split; fall back to test
        if split == "validation":
            split = "test"
        raw = load_dataset("ag_news", split=split)
        texts = [row["text"] for row in raw]
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose 'wikitext' or 'ag_news'.")

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt",
    )

    class _TextDataset(torch.utils.data.Dataset):
        def __init__(self, enc: BatchEncoding):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx: int):
            return self.input_ids[idx], self.attention_mask[idx]

    dataset = _TextDataset(encodings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


# ------------------------------------------------------------------------------
# Gemma3 model
# ------------------------------------------------------------------------------
@dataclass
class Gpt2Config:
    r"""
    GPT2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "base", "medium", "large" and "xl".
    device: str
        Device to use.
    """

    model_size: str = "1b"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.model_size not in ["base", "medium", "large", "xl"]:
            raise ValueError(f"Invalid model size {self.model_size}.")


class Gpt2(nn.Module):
    def __init__(self, config: Gpt2Config):
        super().__init__()
        self.device = config.device
        self.model_name = "openai-community/gpt2"
        if config.model_size != "base":
            self.model_name += f"-{config.model_size}"
        self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "openai-community/gpt2",
                "openai-community/gpt2-medium",
                "openai-community/gpt2-large",
                "openai-community/gpt2-xl".
        device: str
            Device to use.

        Notes
        -----
        The model is set to evaluation mode after loading (dropout disabled).
        Call model.train() explicitly to re-enable training mode.
        """
        available_models = [
            "openai-community/gpt2",
            "openai-community/gpt2-medium",
            "openai-community/gpt2-large",
            "openai-community/gpt2-xl",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = GPT2Model.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    @torch.inference_mode()
    def get_decomposition(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        r"""
        Recover the outputs of transformer components across layers for the same input.

        Parameters
        ----------
        input_ids: torch.Tensor of dimension (N, L)
            Batch of tokenised sequences.
        attention_mask: torch.Tensor of dimension (N, L)
            Attention mask (1 = real token, 0 = padding).

        Returns
        -------
        outputs: Dictionary of output of each model's layer.
        """
        outputs = {}

        # Embedding layer with token and position embedding
        inputs_embeds = self.model.wte(input_ids)
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        position_embeds = self.model.wpe(position_ids)
        out = inputs_embeds + position_embeds.to(inputs_embeds.device)
        outputs["embedding"] = move_to_cpu(out)

        # Transformer blocks
        layers = self.model.h
        for i, block in enumerate(layers):
            # LN1
            block_out = block.ln_1(out)
            key = "attn_norm"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # MHA
            block_out, _ = block.attn(hidden_states=block_out, attention_mask=None)
            key = "attn"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # LN2
            block_out = block.ln_2(out)
            key = "ffn_norm"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # FC1
            block_out = block.mlp.c_fc(out)
            key = "ffn_fc1"
            outputs[f"block{i}" + "_" + key] = move_to_cpu(block_out)

            # FC2
            zero_out = torch.zeros_like(out)
            zero_list = [zero_out] * (block_out.shape[-1] // out.shape[-1] - 1)
            expanded_out = torch.cat((out, *zero_list), dim=-1)
            block_out = block.mlp.c_proj(expanded_out)
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
        Batch of sequences of tokens of dimension (N, L, d).
    y: torch.Tensor
        Batch of sequences of tokens of dimension (N, L, d).
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
# Analyze the plasticity of transformer components across layers
# ------------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    r"""
    Plasticity analysis config file
    """

    # Model
    model_size: str = "1b"
    seq_len: int = 128

    # Data
    dataset_name: str = "ag_news"
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
    model_config = Gpt2Config(model_size=config.model_size, device=config.device)
    model = Gpt2(config=model_config)
    logger.info("Done building model.")

    # ---------------------------------------------------------------------
    # Build dataloader
    # ---------------------------------------------------------------------

    logger.info("Building dataloaders.")

    # Subset of the pretraining data: WikiText-103 (analogous to ImageNet)
    loader_config = {
        "dataset_name": "wikitext",
        "batch_size": config.batch_size,
        "mode": "test",
        "seq_len": config.seq_len,
        "tokenizer": model.tokenizer,
    }
    loader1 = build_text_loader(config=loader_config)

    # Subset of the downstream data: AG News (analogous to CIFAR-10)
    loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.batch_size,
        "mode": "test",
        "seq_len": config.seq_len,
        "tokenizer": model.tokenizer,
    }
    loader2 = build_text_loader(config=loader_config)
    logger.info("Done building dataloaders.")

    # ---------------------------------------------------------------------
    # Set saving paths
    # ---------------------------------------------------------------------

    # Save config
    config_name = f"analysis_gpt2_{config.model_size}_{config.dataset_name}"
    save_dir = SAVE_DIR / config_name
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        # JSON serializable alias
        exp_config = json_serializable(asdict(config))
        json.dump(exp_config, f, indent=4)

    # ---------------------------------------------------------------------
    # Recover layers outputs
    # ---------------------------------------------------------------------

    # Recover statistics
    model.eval()
    distances = {}
    slopes = {}
    step = 0

    # Make loaders iterable
    loader1, loader2 = map(lambda t: make_iterable(t), (loader1, loader2))
    iterator1 = iter(loader1)
    iterator2 = iter(loader2)

    # Loop over batches of data
    while step < config.n_steps:
        # Recover data
        input_ids1, mask1 = next(iterator1)
        input_ids2, mask2 = next(iterator2)

        # Move to device
        if config.device != "cpu":
            input_ids1 = input_ids1.pin_memory()
            mask1 = mask1.pin_memory()
            input_ids2 = input_ids2.pin_memory()
            mask2 = mask2.pin_memory()
        input_ids1 = input_ids1.to(device=config.device, non_blocking=True)
        mask1 = mask1.to(device=config.device, non_blocking=True)
        input_ids2 = input_ids2.to(device=config.device, non_blocking=True)
        mask2 = mask2.to(device=config.device, non_blocking=True)

        # Block decomposition
        outputs1 = model.get_decomposition(input_ids1, mask1)
        outputs2 = model.get_decomposition(input_ids2, mask2)

        # Free memory
        del input_ids1, mask1, input_ids2, mask2

        # Loop over layers
        # We iterate over consecutive (input, output) pairs to compute the slope.
        # For each component, the slope is |f(x) - f(y)| / |x - y| where
        # x, y are the component inputs (previous key) and f(x), f(y) are the outputs.
        keys = list(outputs1.keys())
        for key in keys:
            # Recover the layer outputs
            z1 = outputs1.pop(key).to(config.device)
            z2 = outputs2.pop(key).to(config.device)

            # Compute the distance between the two layer outputs
            dist = distance(z1, z2, reduction="none")

            # Update dictionary and free memory (convert dist in float32 before calling numpy)
            value = get_numpy(dist.float())
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
    pickle.dump(slopes, open(save_dir / "slopes.pkl", "wb"))


def run_analysis(
    model_size: str = "1b",
    dataset_name: str = "ag_news",
    seq_len: int = 128,
    batch_size: int = 128,
    n_steps: int = 100,
    device: str = DEVICE,
) -> None:
    config = AnalysisConfig(
        model_size=model_size,
        dataset_name=dataset_name,
        seq_len=seq_len,
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
