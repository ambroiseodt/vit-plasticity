r"""
Plotting functions related to finetuning runs.

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
import numpy as np
import pandas as pd

from vitef.config import RESULT_DIR, SAVING_DIR
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

# ----------------------------------------------------------------------------
# Utils to aggregate results
# ----------------------------------------------------------------------------


def get_single_exp(dataset_name: str, seed: int, lr: str, comp: int, prefix: str = "vit") -> tuple:
    r"""Recover training and evaluation information for a given run."""

    # Get log_dir name
    log_dir = f"{prefix}_{dataset_name}_seed_{seed}_lr_{lr}_comp_{comp}"
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

    # Recover model information and evaluation results
    all_components = ["emb", "attn_norm", "mha", "ffn_norm", "ffn_fc1", "ffn_fc2"]
    trainable_components = [x for x in all_components if x not in exp_config["components"]]
    if trainable_components == all_components:
        trainable_components = ["all"]

    # Keep only the element
    trainable_components = trainable_components[0]
    eval_data = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "max_n_steps": exp_config["n_steps"],
        "lr": float(lr),
        "trainable_components": trainable_components,
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

    # Aggregate results for the 6 configurations (all, attn_norm, mha, ffn_norm, ffn_fc1, ffn_fc2)
    comps = [0, 2, 3, 4, 5, 6]
    for seed in seeds:
        for lr in lrs:
            for comp in comps:
                results = {}
                _, _, eval_data = get_single_exp(dataset_name=dataset_name, seed=seed, lr=lr, comp=comp)
                for key in keys:
                    results[key] = eval_data[key]
                all_results.append(results)

    # Save results
    df = pd.DataFrame(all_results)
    results_path = RESULT_DIR / "ablation/finetuning/low_data"
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
        data = get_data(dataset_name, folder="ablation/finetuning/low_data")
        acc_mean[dataset_name] = {}
        acc_std[dataset_name] = {}
        relative_gain[dataset_name] = {}

        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            values = []
            if "domain" in dataset_name:
                lr_values = ["3e-3", "1e-2", "3e-2", "6e-2"]
            else:
                lr_values = ["3e-3"]  # ["1e-3", "3e-3", "1e-2"]
            for lr in lr_values:
                for seed in seeds:
                    root_ind = (
                        (data["lr"] == float(lr))
                        & (data["seed"] == int(seed))
                        & (data["trainable_components"] == trainable_component)
                    )
                    test_acc = np.asarray(data[root_ind]["test_acc"])
                    values.append(test_acc)
            acc_mean[dataset_name][i] = np.asarray(values).mean()
            acc_std[dataset_name][i] = np.asarray(values).std()

    print("Finetuning")
    ordered_index = [0, 1, 3, 5, 4, 2]
    for dataset_name in dataset_names:
        print(dataset_name)
        for i in ordered_index:
            trainable_component = list(VIT_COMPONENTS_MAP.keys())[i]
            print(
                trainable_component,
                f"{np.round(acc_mean[dataset_name][i] * 100, 1)}",
                f"{np.round(acc_std[dataset_name][i] * 100, 1)}",
            )
        print("\n")

    # Get average results
    mean_acc = {}
    mean_std = {}
    for dataset_name in dataset_names:
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            if trainable_component not in mean_acc:
                mean_acc[trainable_component] = [acc_mean[dataset_name][i]]
                mean_std[trainable_component] = [acc_std[dataset_name][i]]
            else:
                mean_acc[trainable_component].append(acc_mean[dataset_name][i])
                mean_std[trainable_component].append(acc_std[dataset_name][i])

    print("Average accuracy")
    for trainable_component in VIT_COMPONENTS_MAP.keys():
        print(
            trainable_component,
            np.round(np.mean(mean_acc[trainable_component]) * 100, 1),
            np.round(np.mean(mean_std[trainable_component]) * 100, 1),
        )

    print("\n")


# ----------------------------------------------------------------------------
# Results functions
# ----------------------------------------------------------------------------


def get_csv_results() -> None:
    dataset_names = [
        "cifar10ld",
        "cifar100ld",
        "cifar10ld_c_motion_blur_5",
        "cifar10ld_c_contrast_5",
        "cifar10ld_c_snow_5",
        "cifar10ld_c_speckle_noise_5",
        "domainld_clipart",
        "domainld_sketch",
        # "flold",
        "ptld",
    ]
    seeds = [0]
    for dataset_name in dataset_names:
        if "domain" in dataset_name:
            lrs = ["3e-3", "1e-2", "3e-2", "6e-2"]
        else:
            lrs = ["1e-3", "3e-3", "1e-2", "3e-2"]
        get_evals_csv(dataset_name=dataset_name, seeds=seeds, lrs=lrs)


def get_table_results() -> None:
    dataset_names = [
        "cifar10ld_c_motion_blur_5",
        "cifar10ld_c_snow_5",
        "cifar10ld_c_speckle_noise_5",
        "ptld",
    ]
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
