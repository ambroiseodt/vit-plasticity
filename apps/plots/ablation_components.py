r"""
Plotting functions related to ablations runs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vitef.config import FIGURE_DIR, RESULT_DIR, SAVING_DIR
from vitef.utils import load_jsonl_to_numpy

logger = logging.getLogger("vitef")

# Paths
RUNS_DIR = SAVING_DIR / "runs/ablation"

# Trainable components in the ViT
VIT_COMPONENTS = ["LN1", "MHA", "LN2", "FC1", "FC2"]

# Trainable components in the ViT
VIT_COMPONENTS_MAP = {
    "all": "All",
    "attn_norm": "LN1",
    "mha": "MHA",
    "ffn_norm": "LN2",
    "ffn_fc1": "FC1",
    "ffn_fc2": "FC2",
}

# Learning rates
LR_VALUES = {
    "cifar10": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar100": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_contrast_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_gaussian_noise_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_motion_blur_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_snow_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_speckle_noise_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "domainnet_clipart": ["3e-3", "1e-2", "3e-2", "6e-2"],
    "domainnet_sketch": ["3e-3", "1e-2", "3e-2", "6e-2"],
    "flowers102": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "pet": ["1e-3", "3e-3", "1e-2", "3e-2"],
}

# Dataset names
DATASET_MAP = {
    "cifar10": "Cifar10",
    "cifar100": "Cifar100",
    "cifar10_c_contrast_5": "Contrast",
    "cifar10_c_gaussian_noise_5": "Gaussian Noise",
    "cifar10_c_motion_blur_5": "Motion Blur",
    "cifar10_c_snow_5": "Snow",
    "cifar10_c_speckle_noise_5": "Speckle Noise",
    "domainnet_clipart": "Clipart",
    "domainnet_sketch": "Sketch",
    "pet": "Pet",
    "flowers102": "Flowers102",
}

# Figure golden ratio (from ICML style file)
WIDTH = 6
HEIGHT = 5
FONTSIZE = 15
FONTSIZE_LEGEND = 15
LINEWIDTH = 5
GD_LINEWIDTH = 2.5
RED_LINEWIDTH = 2.5
ERR_LINEWIDTH = 2
ALPHA_GRID = 0.8
COLORS = {
    "LN1": "#daa4ac",
    "MHA": "#37abb5",
    "LN2": "#b153a1",
    "FC1": "#a291e1",
    "FC2": "#858ec2",
}

# Visual parameters
palette = sns.cubehelix_palette()
custom_params = {"axes.grid": False}
sns.set_theme(style="ticks", palette=palette, rc=custom_params)
sns.set_context("talk")
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

# ----------------------------------------------------------------------------
# Utils to aggregate results
# ----------------------------------------------------------------------------


def get_single_exp(dataset_name: str, seed: int, lr: str, prefix: str = "vit") -> tuple:
    r"""Recover training and evaluation information for a given run."""

    # Get log_dir name
    log_dir = f"{prefix}_{dataset_name}_seed_{seed}_lr_{lr}_comp_mha_fc1"
    log_dir = RUNS_DIR / log_dir

    # Recover experiment configuration, model information and evaluation file
    with open(log_dir / "config.json") as f:
        exp_config = json.load(f)

    with open(log_dir / "metrics" / "info_model.jsonl") as f:
        info_model = json.load(f)

    with open(log_dir / "metrics" / "eval.jsonl") as f:
        eval_file = json.load(f)

    # Recover the training step of the checkpoint evaluated
    checkpoint_dir = Path(log_dir / "checkpoints")
    iterator = checkpoint_dir.iterdir()
    *_, last = iterator
    checkpoint_step = last.parts[-1]

    # Keep only the element
    eval_data = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "max_n_steps": exp_config["n_steps"],
        "lr": float(lr),
        "trainable_components": "mha_fc1",
        "model_size": info_model["model_params"],
        "n_step": checkpoint_step,
        "test_acc": eval_file["test_acc"],
    }

    # Recover runs
    data_keys = ["loss", "step", "grad_norm", "eval_loss", "eval_acc"]
    data = load_jsonl_to_numpy(log_dir / "metrics" / "raw_0.jsonl", keys=data_keys)

    # Index for training and evaluation
    not_training = np.isnan(data["loss"].astype(float))
    not_eval = np.isnan(data["eval_loss"].astype(float))

    # Recover training runs
    train_steps = data["step"][~not_training]
    train_loss = data["loss"][~not_training]
    grad_norms = data["grad_norm"][~not_training]
    training_runs = [train_steps, train_loss, grad_norms]

    # Recover validation runs
    val_steps = data["step"][~not_eval]
    val_loss = data["eval_loss"][~not_eval]
    val_acc = data["eval_acc"][~not_eval]
    validation_runs = [val_steps, val_loss, val_acc]

    return training_runs, validation_runs, eval_data


def get_evals_csv(dataset_name: str, seeds: list, lrs: list) -> None:
    r"""Recover and aggreate evaluation results for a given dataset."""
    all_results = []
    keys = [
        "dataset_name",
        "seed",
        "max_n_steps",
        "lr",
        "trainable_components",
        "model_size",
        "n_step",
        "test_acc",
    ]

    # Aggregate results for the mha + fc1
    for seed in seeds:
        for lr in lrs:
            results = {}
            _, _, eval_data = get_single_exp(dataset_name=dataset_name, seed=seed, lr=lr)
            for key in keys:
                results[key] = eval_data[key]
            all_results.append(results)

    # Save results
    df = pd.DataFrame(all_results)
    results_path = RESULT_DIR / "ablation/finetuning/components"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"{dataset_name}.csv"
    df.to_csv(path)


def get_data(dataset_name: str, folder: str) -> pd.DataFrame:
    r"""Load data from csv file."""
    path = RESULT_DIR / folder / f"{dataset_name}.csv"
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    r"""Save figure in pdf format."""
    figure_path = FIGURE_DIR / "ablation/finetuning"
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def table_results(dataset_names: list, seeds: list) -> None:
    r"""
    Recover finetuning and zero-shot performance. The zero-shot is obtained by doing
    linear probing on the attention representation of the last layer.
    """

    # ViT-Large
    acc_mean = {}
    acc_std = {}
    relative_gain = {}
    for dataset_name in dataset_names:
        # Finetuning results
        data = get_data(dataset_name, folder="ablation/finetuning/components")
        acc_mean[dataset_name] = {}
        acc_std[dataset_name] = {}
        relative_gain[dataset_name] = {}
        trainable_component = "mha_fc1"
        values = []
        for lr in LR_VALUES[dataset_name]:
            for seed in seeds:
                root_ind = (
                    (data["lr"] == float(lr))
                    & (data["seed"] == int(seed))
                    & (data["trainable_components"] == trainable_component)
                )
                test_acc = np.asarray(data[root_ind]["test_acc"])
                values.append(test_acc)
            acc_mean[dataset_name] = np.asarray(values).mean()
            acc_std[dataset_name] = np.asarray(values).std()

    print("Finetuning")
    for dataset_name in dataset_names:
        print(dataset_name)
        trainable_component = "mha_fc1"
        print(
            trainable_component,
            f"{np.round(acc_mean[dataset_name] * 100, 1)}",
            f"{np.round(acc_std[dataset_name] * 100, 1)}",
        )
        print("\n")


# ----------------------------------------------------------------------------
# Results functions
# ----------------------------------------------------------------------------


def get_csv_results() -> None:
    dataset_names = ["domainnet_clipart"]
    seeds = [0]
    for dataset_name in dataset_names:
        lrs = LR_VALUES[dataset_name]
        get_evals_csv(dataset_name=dataset_name, seeds=seeds, lrs=lrs)


def get_table_results() -> None:
    dataset_names = ["domainnet_clipart"]
    seeds = [0]
    table_results(dataset_names=dataset_names, seeds=seeds)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"csv": get_csv_results, "table": get_table_results})


# %% CLI
if __name__ == "__main__":
    main()
# %%
