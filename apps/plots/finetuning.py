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
import pickle
import re
from pathlib import Path

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from vitef.config import FIGURE_DIR, RESULT_DIR, SAVING_DIR
from vitef.utils import load_jsonl_to_numpy

logger = logging.getLogger("vitef")

# Paths
RUNS_DIR = SAVING_DIR / "runs"
PROBES_DIR = SAVING_DIR / "probes"
ANALYSIS_DIR = SAVING_DIR / "analysis"

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


# Parameter for better visual
ymin_manual = -0.5


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

    # Aggregate results for the 7 configurations (all, emb, attn_norm, mha, ffn_norm, ffn_fc1, ffn_fc2)
    comps = range(7)
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
    results_path = RESULT_DIR / "finetuning"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"{dataset_name}.csv"
    df.to_csv(path)


def get_runs(dataset_name: str, seeds: list, lrs: list) -> dict:
    r"""Recover the training and validation runs for a given dataset."""
    all_runs = {}

    # Aggregate results for the 6 configurations of interest
    index_map = {0: "all", 2: "attn_norm", 3: "mha", 4: "ffn_norm", 5: "ffn_fc1", 6: "ffn_fc2"}
    for lr in lrs:
        all_runs[lr] = {}
        for comp in index_map.keys():
            trainable_component = index_map[comp]
            all_runs[lr][trainable_component] = {}
            for seed in seeds:
                all_runs[lr][trainable_component][seed] = {
                    "model_size": None,
                    "train_steps": None,
                    "train_loss": None,
                    "grad_norm": None,
                    "val_steps": None,
                    "val_loss": None,
                    "val_acc": None,
                }
                training_runs, validation_runs, eval_data = get_single_exp(
                    dataset_name=dataset_name, seed=seed, lr=lr, comp=comp
                )
                if all_runs[lr][trainable_component][seed]["model_size"] is None:
                    all_runs[lr][trainable_component][seed]["model_size"] = eval_data["model_size"]
                    all_runs[lr][trainable_component][seed]["trainable_components"] = eval_data["trainable_components"]
                train_steps, train_loss, grad_norms = training_runs
                val_steps, val_loss, val_acc = validation_runs
                all_runs[lr][trainable_component][seed]["train_steps"] = train_steps
                all_runs[lr][trainable_component][seed]["train_loss"] = train_loss
                all_runs[lr][trainable_component][seed]["grad_norm"] = grad_norms
                all_runs[lr][trainable_component][seed]["val_steps"] = val_steps
                all_runs[lr][trainable_component][seed]["val_loss"] = val_loss
                all_runs[lr][trainable_component][seed]["val_acc"] = val_acc
    return all_runs


def get_single_exp_linear_probing(
    dataset_name: str, seed: int, lr: str, comp: int, prefix: str = "vit", finetuned: bool = False
) -> tuple:
    r"""Recover linear probing results for a given run."""

    corruption_dataset_map = {
        "cifar10_c_contrast_5": "cifar10_c-corruption-contrast-severity-5",
        "cifar10_c_gaussian_noise_5": "cifar10_c-corruption-gaussian_noise-severity-5",
        "cifar10_c_motion_blur_5": "cifar10_c-corruption-motion_blur-severity-5",
        "cifar10_c_snow_5": "cifar10_c-corruption-snow-severity-5",
        "cifar10_c_speckle_noise_5": "cifar10_c-corruption-speckle_noise-severity-5",
    }

    domainnet_dataset_map = {
        "domainnet_clipart": "domainnet-clipart",
        "domainnet_sketch": "domainnet-sketch",
    }

    # Get log_dir name
    if finetuned:
        log_dir = f"{prefix}_{dataset_name}_seed_{seed}_lr_{lr}_comp_{comp}"
    else:
        if "cifar10_c" in dataset_name:
            dataset_name = corruption_dataset_map[dataset_name]
        elif "domainnet" in dataset_name:
            dataset_name = domainnet_dataset_map[dataset_name]
        log_dir = f"{prefix}_{dataset_name}_seed_0_pretrained"
    log_dir = PROBES_DIR / log_dir

    with open(log_dir / "linear_probing.json") as f:
        results_file = json.load(f)

    # Recover model information and evaluation results
    all_components = ["attn_norm", "attn", "attn_res", "ffn_norm", "ffn_fc1", "ffn_activation", "ffn_fc2", "ffn_res"]
    if finetuned:
        if comp == 0:
            trainable_components = ["all"]
        else:
            trainable_components = [all_components[comp - 1]]
    else:
        trainable_components = ["none"]

    # Keep only the element
    trainable_components = trainable_components[0]
    meta_data = {
        "dataset_name": dataset_name,
        "trainable_components": trainable_components,
    }
    if finetuned:
        meta_data = meta_data | {"seed": int(seed), "lr": float(lr)}

    results = []
    for key in results_file.keys():
        block, comp = key.split("_", 1)
        block = block.split("block", 1)[-1]
        result = meta_data | {"block": block, "component": comp, "test_acc": results_file[key]}
        results.append(result)
    return results


def get_linear_probing_csv(dataset_name: str, lrs: list) -> None:
    r"""Recover and aggreate linear probing results for a given dataset."""

    # Pretrained model
    all_results = get_single_exp_linear_probing(
        dataset_name=dataset_name, seed=None, lr=None, comp=None, finetuned=False
    )
    df = pd.DataFrame(all_results)
    results_path = RESULT_DIR / "linear_probing"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"{dataset_name}.csv"
    df.to_csv(path)


def get_data(dataset_name: str, folder: str) -> pd.DataFrame:
    r"""Load data from csv file."""
    path = RESULT_DIR / folder / f"{dataset_name}.csv"
    df = pd.read_csv(path)
    return df


def get_sensitivity(path: str) -> dict:
    r"""
    Recover the sensitivity measure of ViT components (computed as the average rate of change).

    Parameters
    ----------
    path: str
        Path to the experiment.
    """
    # Loop over results files
    distances = pickle.load(open(path / "distances.pkl", "rb"))
    keys = list(distances.keys())

    # Recover the input and remove the corresponding key
    input_key = "embedding"
    inputs = np.asarray(distances[input_key].flatten())
    keys.remove(input_key)

    # Loop over ViT components
    dict_df = {}
    for _, layer in enumerate(keys):
        # Compute the ratio of distances
        values = np.asarray(distances[layer].flatten())
        ratio = values / inputs

        # Recover block name and layer index
        substring, component = layer.split("_", 1)

        # Update dictionary
        if component in dict_df:
            dict_df[component].append(ratio)
        else:
            dict_df[component] = [ratio]

    return dict_df


def get_config_sensitivity(dataset_name: str, model_name: str, pretrained: bool = True) -> str:
    r"""Return configuration"""

    corruption_dataset_map = {
        "cifar10_c_contrast_5": "cifar10_c-corruption-contrast-severity-5",
        "cifar10_c_gaussian_noise_5": "cifar10_c-corruption-gaussian_noise-severity-5",
        "cifar10_c_motion_blur_5": "cifar10_c-corruption-motion_blur-severity-5",
        "cifar10_c_snow_5": "cifar10_c-corruption-snow-severity-5",
        "cifar10_c_speckle_noise_5": "cifar10_c-corruption-speckle_noise-severity-5",
    }

    domainnet_dataset_map = {
        "domainnet_clipart": "domainnet-clipart",
        "domainnet_sketch": "domainnet-sketch",
    }

    if "cifar10_c" in dataset_name:
        dataset_name = corruption_dataset_map[dataset_name]
    elif "domainnet" in dataset_name:
        dataset_name = domainnet_dataset_map[dataset_name]

    # ViT model name
    if model_name == "huge":
        vit_model_name = f"vit-{model_name}-patch14-224-in21k"
    else:
        vit_model_name = f"vit-{model_name}-patch16-224-in21k"

    # Config name
    config_name = f"analysis_{vit_model_name}_pretrained_{pretrained}"
    config_name += f"_{dataset_name}"

    return config_name


# ----------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    r"""Save figure in pdf format."""
    figure_path = FIGURE_DIR / "finetuning"
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def table_results(dataset_names: list, seeds: list) -> None:
    r"""
    Recover finetuning and zero-shot performance. The zero-shot is obtained by doing
    linear probing on the attention representation of the last layer.
    """
    mean_linear_probing = []
    acc_mean = {}
    acc_std = {}
    relative_gain = {}

    print("Linear probing")
    for dataset_name in dataset_names:
        # Linear probing results
        linear_prob_pretrained = get_data(dataset_name, folder="linear_probing")
        root_ind = (linear_prob_pretrained["block"] == 11) & (linear_prob_pretrained["component"] == "ffn_res")
        linear_prob_acc = linear_prob_pretrained[root_ind]["test_acc"].iloc[0]
        mean_linear_probing.append(linear_prob_acc)
        print(f"{dataset_name}: {np.round(linear_prob_acc * 100, 2)}")

        # Finetuning resultst
        data = get_data(dataset_name, folder="finetuning")
        acc_mean[dataset_name] = {}
        acc_std[dataset_name] = {}
        relative_gain[dataset_name] = {}

        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            best_acc = 0
            std = 0
            for lr in LR_VALUES[dataset_name]:
                values = []
                for seed in seeds:
                    root_ind = (
                        (data["lr"] == float(lr))
                        & (data["seed"] == int(seed))
                        & (data["trainable_components"] == trainable_component)
                    )
                    test_acc = np.asarray(data[root_ind]["test_acc"])
                    values.append(test_acc)
                std_temp = np.asarray(values).std()
                values = np.asarray(values).mean()
                if values > best_acc:
                    best_acc = values
                    std = std_temp
            acc_mean[dataset_name][i] = best_acc
            acc_std[dataset_name][i] = std
            relative_gain[dataset_name][i] = (best_acc - linear_prob_acc) / linear_prob_acc

    print(f"Average: {np.round(np.mean(mean_linear_probing) * 100, 2)}\n")

    print("Finetuning")
    ordered_index = [0, 1, 3, 5, 4, 2]
    for dataset_name in dataset_names:
        print(dataset_name)
        for i in ordered_index:
            trainable_component = list(VIT_COMPONENTS_MAP.keys())[i]
            print(
                trainable_component,
                f"{np.round(acc_mean[dataset_name][i] * 100, 2)}",
                f"{np.round(acc_std[dataset_name][i] * 100, 2)}",
            )
        print("\n")

    # Get average results
    mean_acc = {}
    mean_relative_gain = {}
    for dataset_name in dataset_names:
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            if trainable_component not in mean_acc:
                mean_acc[trainable_component] = [acc_mean[dataset_name][i]]
                mean_relative_gain[trainable_component] = [relative_gain[dataset_name][i]]
            else:
                mean_acc[trainable_component].append(acc_mean[dataset_name][i])
                mean_relative_gain[trainable_component].append(relative_gain[dataset_name][i])

    print("Average accuracy")
    for trainable_component in VIT_COMPONENTS_MAP.keys():
        print(trainable_component, np.round(np.mean(mean_acc[trainable_component]) * 100, 2))

    print("\n")

    print("Average relative gain")
    for trainable_component in VIT_COMPONENTS_MAP.keys():
        print(trainable_component, np.round(np.mean(mean_relative_gain[trainable_component]) * 100, 2))

    print("\n")
    print("Finetuning performance gap")
    acc_gap = {}
    for dataset_name in dataset_names:
        data = get_data(dataset_name, folder="finetuning")
        acc_gap[dataset_name] = {}
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            acc_gap[dataset_name][i] = []
            for lr in LR_VALUES[dataset_name]:
                values = []
                for seed in seeds:
                    root_ind = (
                        (data["lr"] == float(lr))
                        & (data["seed"] == int(seed))
                        & (data["trainable_components"] == trainable_component)
                    )
                    test_acc = np.asarray(data[root_ind]["test_acc"])
                    values.append(test_acc)
                acc_gap[dataset_name][i].append(np.asarray(values).mean())

    # Get average performance gap
    mean_gap = {}
    for dataset_name in dataset_names:
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            values = np.asarray(acc_gap[dataset_name][i])
            gap = values.max() - values.min()
            if trainable_component not in mean_gap:
                mean_gap[trainable_component] = [gap]
            else:
                mean_gap[trainable_component].append(gap)

    for trainable_component in VIT_COMPONENTS_MAP.keys():
        print(trainable_component, np.round(np.mean(mean_gap[trainable_component]) * 100, 2))


def stat_results(data: str) -> None:
    r"""Apply Student's t-test and Wilcoxon on table of results."""

    # Null-hypothesis: MHA is better than all the other components
    comparisons = [
        ("MHA", "FC1"),
        ("MHA", "FC2"),
        ("MHA", "LN2"),
        ("MHA", "LN1"),
    ]

    # Parse data into DataFrame
    rows = []
    val_pattern = re.compile(r"(\d+\.\d+)(?:±(\d+\.\d+))?")

    for line in data.strip().split("\n"):
        first_val_match = val_pattern.search(line)
        dataset_name = line[: first_val_match.start()].strip()
        matches = val_pattern.findall(line)

        row = {"Dataset": dataset_name}
        for i, (mean, _) in enumerate(matches):
            row[f"Method_{i + 1}"] = float(mean)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Define mapping: MHA=1, FC1=2, FC2=3, LN1=4, LN2=5
    mapping = {
        "MHA": "Method_1",
        "FC1": "Method_2",
        "FC2": "Method_3",
        "LN2": "Method_4",
        "LN1": "Method_5",
    }

    # Start statistical tests
    print("--- Statistical Test Results (MHA vs Others) ---")
    print(f"{'Comparison':<15} | {'Mean Diff':<10} | {'T-Test p':<10} | {'Wilcoxon p':<10}")
    print("-" * 55)

    for ref, comp in comparisons:
        group1 = df[mapping[ref]]
        group2 = df[mapping[comp]]

        # Paired T-test (assumes normality)
        _, t_p = stats.ttest_rel(group1, group2)

        # Wilcoxon Signed-Rank Test (non-parametric alternative)
        _, w_p = stats.wilcoxon(group1, group2, alternative="greater")

        mean_diff = group1.mean() - group2.mean()

        print(f"{ref} vs {comp:<11} | {mean_diff:>9.4f}% | {t_p:>10.4f} | {w_p:>10.4f}")

    print("\nNote: p < 0.05 is typically considered statistically significant.")


def get_intro(
    dataset_names: list,
    seeds: list,
    save: bool = False,
    ncol: int = 5,
) -> None:
    r"""Plot plasticity distribution and best finetuning performance for each component."""
    results = {}
    ncols = 2
    width = 4
    height = 3.5
    figsize = (ncols * width, height)
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Distribution of plasticity
    ax = axes[0]
    plasticity_rank = [5, 1, 4, 2, 3]
    yname = "Rates of Change"
    dict_df = {"": [], yname: []}
    model_name = "base"
    for dataset_name in dataset_names:
        config = get_config_sensitivity(dataset_name=dataset_name, model_name=model_name, pretrained=True)
        dict_values = get_sensitivity(path=ANALYSIS_DIR / config)
        for j, key in enumerate(dict_values.keys()):
            ratio = np.asarray(dict_values[key])
            mean = np.mean(ratio, axis=-1)
            rank = plasticity_rank[j]
            for val in mean:
                dict_df[""].append(rank)
                dict_df[yname].append(val)
    colors = [COLORS[key] for key in ["MHA", "FC1", "FC2", "LN2", "LN1"]]
    sns.boxplot(
        data=dict_df,
        x="",
        y=yname,
        hue="",
        palette=colors,
        legend=False,
        boxprops={"edgecolor": "#333333", "linewidth": 0.5},
        whiskerprops={"color": "#333333", "linewidth": 0.5, "linestyle": "--"},
        capprops={"color": "#333333", "linewidth": 0.5},
        medianprops={"color": "#333333", "linewidth": 0.5},
        showfliers=False,
        ax=ax,
    )

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(range(1, 6))
    yticks = [1, 6, 11]
    ax.set_ylim(ymin_manual, yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(r"Theoretical Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Finetuning
    ax = axes[1]
    for dataset_name in dataset_names:
        # Linear probing results
        linear_prob_pretrained = get_data(dataset_name, folder="linear_probing")
        root_ind = (linear_prob_pretrained["block"] == 11) & (linear_prob_pretrained["component"] == "ffn_res")
        linear_prob_acc = linear_prob_pretrained[root_ind]["test_acc"].iloc[0]

        # Finetuning results
        data = get_data(dataset_name, folder="finetuning")
        results[dataset_name] = {}
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            best_values = 0
            for lr in LR_VALUES[dataset_name]:
                values = []
                for seed in seeds:
                    root_ind = (
                        (data["lr"] == float(lr))
                        & (data["seed"] == int(seed))
                        & (data["trainable_components"] == trainable_component)
                    )
                    test_acc = np.asarray(data[root_ind]["test_acc"])
                    ratio = (test_acc - linear_prob_acc) / linear_prob_acc * 100
                    values.append(ratio)
                mean_values = np.asarray(values).mean()

                # Recover best values and correspinding std index
                if mean_values > best_values:
                    best_values = mean_values

            # Recover best values
            results[dataset_name][i] = best_values

    # Get average results
    mean_values = {}
    for dataset_name in dataset_names:
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            mean = results[dataset_name][i]
            if trainable_component not in mean_values:
                mean_values[trainable_component] = [mean]
            else:
                mean_values[trainable_component].append(mean)

    # Show batplot
    yname = "Relative Gain (%)"
    dict_df = {"": [], yname: []}

    # Reorder component according to plasticity rank
    ordered_components = ["mha", "ffn_fc1", "ffn_fc2", "ffn_norm", "attn_norm"]
    comps = [VIT_COMPONENTS_MAP[val] for val in ordered_components]
    palette = [COLORS[comp] for comp in comps]
    for key in ordered_components:
        dict_df[""].append(VIT_COMPONENTS_MAP[key])
        dict_df[yname].append(np.mean(mean_values[key]))
    df = pd.DataFrame(dict_df)
    sns.barplot(data=df, x="", y=yname, hue="", palette=palette, legend=True, ax=ax)

    # Visualization
    ax.legend().remove()
    ax.yaxis.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(range(1, 6))
    ax.set_ylim(45.8, 48.8)
    yticks = [45.8, 47.3, 48.8]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(r"Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Relative Gain ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend with reordered labels
    lines_labels = [fig.axes[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=12.5,
    )

    plt.tight_layout()
    if save:
        figname = "intro"
        save_plot(figname=figname)
    plt.show()


def get_best_performance(
    dataset_names: list,
    seeds: list,
    save: bool = False,
    ncol: int = 5,
) -> None:
    r"""Plot best finetuning performance for each component."""
    results = {}
    figsize = (WIDTH, HEIGHT)
    fig = plt.figure(figsize=figsize)

    # Finetuning
    for dataset_name in dataset_names:
        data = get_data(dataset_name, folder="finetuning")
        results[dataset_name] = {}
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            best_values = 0
            temp_std = []
            index = 0
            for j, lr in enumerate(LR_VALUES[dataset_name]):
                values = []
                for seed in seeds:
                    root_ind = (
                        (data["lr"] == float(lr))
                        & (data["seed"] == int(seed))
                        & (data["trainable_components"] == trainable_component)
                    )
                    test_acc = np.asarray(data[root_ind]["test_acc"]) * 100
                    values.append(test_acc)
                mean_values = np.asarray(values).mean()
                temp_std.append(np.asarray(values).std())

                # Recover best values and correspinding std index
                if mean_values > best_values:
                    best_values = mean_values
                    index = j

            # Recover best values and corresponding std over seeds
            results[dataset_name][i] = (best_values, temp_std[index])

    # Get average results
    mean_values = {}
    std_values = {}
    for dataset_name in dataset_names:
        for i, trainable_component in enumerate(VIT_COMPONENTS_MAP.keys()):
            mean, std = results[dataset_name][i]
            if trainable_component not in mean_values:
                mean_values[trainable_component] = [mean]
                std_values[trainable_component] = [std]
            else:
                mean_values[trainable_component].append(mean)
                std_values[trainable_component].append(std)

    # Show batplot
    yname = "Accuracy (%)"
    dict_df = {"": [], yname: []}

    # Reorder component according to plasticity rank
    ordered_components = ["mha", "ffn_fc1", "ffn_fc2", "ffn_norm", "attn_norm"]
    comps = [VIT_COMPONENTS_MAP[val] for val in ordered_components]
    palette = [COLORS[comp] for comp in comps]
    for key in ordered_components:
        dict_df[""].append(VIT_COMPONENTS_MAP[key])
        dict_df[yname].append(np.mean(mean_values[key]))
    df = pd.DataFrame(dict_df)
    ax = sns.barplot(data=df, x="", y=yname, hue="", palette=palette, legend=True, errorbar=("sd"))

    # Recover pooled standard deviation over seeds
    pooled_std = {key: np.sqrt(np.mean(np.square(std_values[key]))) for key in ordered_components}

    # Recover pooled standard error over seeds
    pooled_se = np.asarray(list(pooled_std.values())) / np.sqrt(len(seeds))

    # In the plotting section, change yerr:
    for rank in range(len(pooled_se)):
        plt.errorbar(
            x=rank,
            y=df.loc[rank, yname],
            yerr=pooled_se[rank],
            fmt="none",
            color="#333333",
            capsize=0,
            linewidth=ERR_LINEWIDTH,
        )

    # Visualization
    ax = fig.axes[0]
    ax.legend().remove()
    ax.yaxis.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(range(1, 6))
    ax.set_ylim(89, 91)
    yticks = [89, 90, 91]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(r"Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend with reordered labels
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]

    leg = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=12.5,
    )

    # Manually change the line width
    for handle in leg.legend_handles:
        try:
            handle.set_markersize(10)
        except AttributeError:
            pass

    plt.tight_layout()
    if save:
        figname = "finetuning_all"
        save_plot(figname=figname)
    plt.show()


def get_robustness_all(
    dataset_names: list,
    seeds: list,
    save: bool = False,
    ncol: int = 6,
) -> None:
    r"""Plot results for each component over learning rates and initializations."""
    nrows = 4
    ncols = 3
    width = 5
    height = 4
    figsize = (ncols * width, nrows * height)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 6)  # Use 6 units to allow half-column shifts

    # Results
    for i, dataset_name in enumerate(dataset_names):
        # Rows 1 & 2: Each plot takes 2 units (3 plots * 2 = 6 units)
        if i < 9:
            if i < 3:
                j = 0
            elif i >= 3 and i < 6:
                j = 1
                i -= 3
            elif i >= 6:
                j = 2
                i -= 6
            ax = fig.add_subplot(gs[j, i * 2 : (i + 1) * 2])
            if i == 1 and j == 0:
                ax_line = ax

        # Row 3: Each plot takes 2 units, but we start at index 1 to center them
        # (1 + 2 + 2 + 1 = 6 units used, leaving 1 unit on each side)
        else:
            j = 3
            i -= 9
            ax = fig.add_subplot(gs[j, (i * 2) + 1 : (i * 2) + 3])

        data = get_data(dataset_name, folder="finetuning")
        results = {}
        for trainable_component in VIT_COMPONENTS_MAP.keys():
            results[trainable_component] = []
            for lr in LR_VALUES[dataset_name]:
                for seed in seeds:
                    root_ind = (
                        (data["lr"] == float(lr))
                        & (data["seed"] == int(seed))
                        & (data["trainable_components"] == trainable_component)
                    )
                    test_acc = data[root_ind]["test_acc"].iloc[0]
                    results[trainable_component].append(test_acc * 100)

        full = np.asarray(results.pop("all")).mean()

        # Show batplot
        yname = "Accuracy (%)"
        dict_df = {"": [], yname: []}
        ordered_components = ["mha", "ffn_fc1", "ffn_fc2", "ffn_norm", "attn_norm"]
        comps = [VIT_COMPONENTS_MAP[val] for val in ordered_components]
        palette = [COLORS[comp] for comp in comps]
        for key in ordered_components:
            values = results[key]
            for val in values:
                dict_df[""].append(VIT_COMPONENTS_MAP[key])
                dict_df[yname].append(val)
        df = pd.DataFrame(dict_df)
        sns.boxplot(
            data=df,
            x="",
            y=yname,
            hue="",
            ax=ax,
            # linewidth=0.1,
            palette=palette,
            legend=True,
            boxprops={"edgecolor": "#333333", "linewidth": 0.5},
            whiskerprops={"color": "#333333", "linewidth": 0.5, "linestyle": "--"},
            capprops={"color": "#333333", "linewidth": 0.5},
            medianprops={"color": "#333333", "linewidth": 0.5},
            showfliers=False,
        )
        tol = 0.41
        xmin = 0
        xmax = 4
        if i == 1 and j == 0:
            line = ax.hlines(
                full,
                xmin=xmin - tol,
                xmax=xmax + tol,
                color="tab:red",
                linestyle="--",
                label="full finetuning",
                lw=RED_LINEWIDTH,
            )
        else:
            ax.hlines(
                full,
                xmin=xmin - tol,
                xmax=xmax + tol,
                color="tab:red",
                linestyle="--",
                lw=RED_LINEWIDTH,
            )

        # Visualization
        ax.legend().remove()
        ax.yaxis.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_title(f"{DATASET_MAP[dataset_name]} \n")
        ax.set_xticks(range(5))
        ax.set_xticklabels(range(1, 6))
        ymin, ymax = ax.get_ylim()
        N = 3 if dataset_name != "flowers102" else 2
        yticks = np.linspace(ymin, ymax, N)
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))
        ax.set_xlabel(r"Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
        ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend with reordered labels
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=FONTSIZE,
    )

    second_legend = ax_line.legend(
        handles=[line],
        loc="upper center",
        bbox_to_anchor=(0.34, 0.15),
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        framealpha=0,
        handlelength=1.5,
    )

    # Manually add the first legend back to the plot
    ax_line.add_artist(second_legend)

    plt.tight_layout()
    if save:
        figname = "robustness_all"
        save_plot(figname=figname)
    plt.show()


def get_training_evolution(
    dataset_name: list,
    seed: int,
    save: bool = False,
    ncol: int = 6,
) -> None:
    r"""Plot gradient norms evolution for each component."""
    lrs = LR_VALUES[dataset_name]
    nrows = 2
    ncols = len(lrs)
    width = 4
    height = width
    figsize = (ncols * width, nrows * height)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey="row")
    all_runs = get_runs(dataset_name=dataset_name, seeds=[seed], lrs=lrs)

    # Training steps for x-axis visualization
    steps_range = {
        "cifar10": [0, 5000, 10000],
        "cifar100": [0, 5000, 10000],
        "cifar10_c_contrast_5": [0, 5000, 10000],
        "cifar10_c_gaussian_noise_5": [0, 5000, 10000],
        "cifar10_c_motion_blur_5": [0, 5000, 10000],
        "cifar10_c_snow_5": [0, 5000, 10000],
        "cifar10_c_speckle_noise_5": [0, 5000, 10000],
        "domainnet_clipart": [0, 10000, 20000],
        "domainnet_sketch": [0, 10000, 20000],
        "flowers102": [0, 2500, 5000],
        "pet": [0, 2000, 4000],
    }

    # Gradient norm for y-axis visualization
    gd_range = {
        "cifar10": [0.2, 0.7, 1.2],
        "cifar100": [0.3, 1.1, 1.9],
        "cifar10_c_contrast_5": [0.3, 0.9, 1.5],
        "cifar10_c_gaussian_noise_5": [0.3, 1.1, 1.9],
        "cifar10_c_motion_blur_5": [0.3, 0.9, 1.5],
        "cifar10_c_snow_5": [0.3, 0.8, 1.3],
        "cifar10_c_speckle_noise_5": [0.3, 1.0, 1.7],
        "domainnet_clipart": [0.3, 0.9, 1.5],
        "domainnet_sketch": [0.3, 1.0, 1.9],
        "flowers102": [0, 0.4, 0.8],
        "pet": [0.1, 0.5, 0.9],
    }

    # Validation loss for y-axis visualization
    loss_range = {
        "cifar10": [0.0, 0.1],
        "cifar100": [0.2, 0.4, 0.8],
        "cifar10_c_contrast_5": [0.0, 0.3],
        "cifar10_c_gaussian_noise_5": [0.2, 0.6, 1.0],
        "cifar10_c_motion_blur_5": [0.1, 0.4, 0.7],
        "cifar10_c_snow_5": [0.1, 0.3, 0.5],
        "cifar10_c_speckle_noise_5": [0.2, 0.6, 1.0],
        "domainnet_clipart": [0.8, 1.3, 1.8],
        "domainnet_sketch": [1.3, 1.8, 2.3],
        "flowers102": [0, 0.3, 0.6],
        "pet": [0.0, 0.4, 0.8],
    }

    ordered_components = ["mha", "ffn_fc1", "ffn_fc2", "ffn_norm", "attn_norm"]

    # Gradient norm
    for i, lr in enumerate(lrs):
        ax = axes[0, i]
        for trainable_component in ordered_components:
            grad_norms = all_runs[lr][trainable_component][seed]["grad_norm"]
            steps = all_runs[lr][trainable_component][seed]["train_steps"]
            ax.plot(
                steps,
                grad_norms,
                color=COLORS[VIT_COMPONENTS_MAP[trainable_component]],
                lw=GD_LINEWIDTH,
                label=VIT_COMPONENTS_MAP[trainable_component],
            )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)

        # Fix x-axis ticks
        xticks = steps_range[dataset_name]
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.array(xticks, dtype=int))

        # Fix y-axis ticks
        yticks = np.asarray(gd_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_title(r"$\eta=$" + f"{lr}\n")
        ax.set_xlabel("Training Steps", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel("Gradient Norm", fontsize=FONTSIZE)
        sns.despine(fig, ax, trim=True, right=True, offset=10)

        ax = axes[1, i]
        for trainable_component in ordered_components:
            grad_norms = all_runs[lr][trainable_component][seed]["val_loss"]
            steps = all_runs[lr][trainable_component][seed]["val_steps"]
            ax.plot(
                steps,
                grad_norms,
                color=COLORS[VIT_COMPONENTS_MAP[trainable_component]],
                lw=GD_LINEWIDTH,
                label=VIT_COMPONENTS_MAP[trainable_component],
            )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)

        # Fix x-axis ticks
        xticks = steps_range[dataset_name]
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.array(xticks, dtype=int))

        # Fix y-axis ticks
        yticks = np.asarray(loss_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=float))

        ax.set_xlabel("Training Steps", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel("Validation Loss", fontsize=FONTSIZE)
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend with reordered labels
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]

    leg = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=FONTSIZE,
    )

    # Manually change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(LINEWIDTH)

    plt.tight_layout()
    if save:
        figname = f"training_evolution_{dataset_name}_seed_{seed}"
        save_plot(figname=figname)
    plt.show()


def get_robustness_training_domainnet_sketch(
    save: bool = False,
    ncol: int = 6,
) -> None:
    r"""Plot results robustness and training evolution for DomainNet Sketch."""
    ncols = 3
    width = 4
    height = width / 1.2
    ncols = 3
    figsize = (ncols * width, height)
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)
    ordered_components = ["mha", "ffn_fc1", "ffn_fc2", "ffn_norm", "attn_norm"]
    dataset_name = "domainnet_sketch"

    # Robustness on all seeds and learning rates
    ax = axes[0]
    seeds = [0, 42, 3407]
    data = get_data(dataset_name, folder="finetuning")
    results = {}
    for trainable_component in VIT_COMPONENTS_MAP.keys():
        results[trainable_component] = []
        for lr in LR_VALUES[dataset_name]:
            for seed in seeds:
                root_ind = (
                    (data["lr"] == float(lr))
                    & (data["seed"] == int(seed))
                    & (data["trainable_components"] == trainable_component)
                )
                test_acc = data[root_ind]["test_acc"].iloc[0]
                results[trainable_component].append(test_acc * 100)

    # Show batplot
    yname = "Accuracy (%)"
    dict_df = {"": [], yname: []}
    comps = [VIT_COMPONENTS_MAP[val] for val in ordered_components]
    palette = [COLORS[comp] for comp in comps]
    for key in ordered_components:
        values = results[key]
        for val in values:
            dict_df[""].append(VIT_COMPONENTS_MAP[key])
            dict_df[yname].append(val)
    df = pd.DataFrame(dict_df)
    sns.boxplot(
        data=df,
        x="",
        y=yname,
        hue="",
        ax=ax,
        # linewidth=0.1,
        palette=palette,
        legend=True,
        showfliers=False,
        boxprops={"edgecolor": "#333333", "linewidth": 0.5},
        whiskerprops={"color": "#333333", "linewidth": 0.5, "linestyle": "--"},
        capprops={"color": "#333333", "linewidth": 0.5},
        medianprops={"color": "#333333", "linewidth": 0.5},
    )

    # Visualization
    ax.legend().remove()
    ax.yaxis.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(range(1, 6))
    ymin, ymax = ax.get_ylim()
    N = 3 if dataset_name != "flowers102" else 2
    yticks = np.linspace(ymin, ymax, N)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.array(yticks, dtype=int))
    ax.set_xlabel(r"Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Focus on a single run with the lr achieving the best result
    seed = 42
    lr = "1e-2"
    all_runs = get_runs(dataset_name=dataset_name, seeds=[seed], lrs=[lr])

    # Training steps for x-axis visualization
    steps_range = {
        "cifar10": [0, 5000, 10000],
        "cifar100": [0, 5000, 10000],
        "cifar10_c_contrast_5": [0, 5000, 10000],
        "cifar10_c_gaussian_noise_5": [0, 5000, 10000],
        "cifar10_c_motion_blur_5": [0, 5000, 10000],
        "cifar10_c_snow_5": [0, 5000, 10000],
        "cifar10_c_speckle_noise_5": [0, 5000, 10000],
        "domainnet_clipart": [0, 10000, 20000],
        "domainnet_sketch": [0, 10000, 20000],
        "flowers102": [0, 2500, 5000],
        "pet": [0, 2000, 4000],
    }

    # Gradient norm for y-axis visualization
    gd_range = {
        "cifar10": [0.2, 0.7, 1.2],
        "cifar100": [0.3, 1.1, 1.9],
        "cifar10_c_contrast_5": [0.3, 0.9, 1.5],
        "cifar10_c_gaussian_noise_5": [0.3, 1.1, 1.9],
        "cifar10_c_motion_blur_5": [0.3, 0.9, 1.5],
        "cifar10_c_snow_5": [0.3, 0.8, 1.3],
        "cifar10_c_speckle_noise_5": [0.3, 1.0, 1.7],
        "domainnet_clipart": [0.3, 0.9, 1.5],
        "domainnet_sketch": [0.2, 1.1, 2.0],
        "flowers102": [0, 0.4, 0.8],
        "pet": [0.1, 0.5, 0.9],
    }

    # Validation loss for y-axis visualization
    loss_range = {
        "cifar10": [0.0, 0.1],
        "cifar100": [0.2, 0.4, 0.8],
        "cifar10_c_contrast_5": [0.0, 0.3],
        "cifar10_c_gaussian_noise_5": [0.2, 0.6, 1.0],
        "cifar10_c_motion_blur_5": [0.1, 0.4, 0.7],
        "cifar10_c_snow_5": [0.1, 0.3, 0.5],
        "cifar10_c_speckle_noise_5": [0.2, 0.6, 1.0],
        "domainnet_clipart": [0.8, 1.3, 1.8],
        "domainnet_sketch": [1.3, 1.8, 2.3],
        "flowers102": [0, 0.3, 0.6],
        "pet": [0.0, 0.4, 0.8],
    }

    ax = axes[1]
    for trainable_component in ordered_components:
        grad_norms = all_runs[lr][trainable_component][seed]["grad_norm"]
        steps = all_runs[lr][trainable_component][seed]["train_steps"]
        ax.plot(
            steps,
            grad_norms,
            color=COLORS[VIT_COMPONENTS_MAP[trainable_component]],
            lw=GD_LINEWIDTH,
            label=VIT_COMPONENTS_MAP[trainable_component],
        )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)

        # Fix x-axis ticks
        xticks = steps_range[dataset_name]
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.array(xticks, dtype=int))

        # Fix y-axis ticks
        yticks = np.asarray(gd_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel("Training Steps", fontsize=FONTSIZE)
        ax.set_ylabel("Gradient Norm", fontsize=FONTSIZE)
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Validation loss
    ax = axes[2]
    for trainable_component in ordered_components:
        grad_norms = all_runs[lr][trainable_component][seed]["val_loss"]
        steps = all_runs[lr][trainable_component][seed]["val_steps"]
        ax.plot(
            steps,
            grad_norms,
            color=COLORS[VIT_COMPONENTS_MAP[trainable_component]],
            lw=GD_LINEWIDTH,
            label=VIT_COMPONENTS_MAP[trainable_component],
        )

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)

    # Fix x-axis ticks
    xticks = steps_range[dataset_name]
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.array(xticks, dtype=int))

    # Fix y-axis ticks
    yticks = np.asarray(loss_range[dataset_name])
    ax.set_ylim(yticks.min(), yticks.max())
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.array(yticks, dtype=float))

    ax.set_xlabel("Training Steps", fontsize=FONTSIZE)
    ax.set_ylabel("Validation Loss", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend with reordered labels
    lines_labels = [fig.axes[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]

    leg = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=FONTSIZE,
    )

    # Manually change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(LINEWIDTH)

    plt.tight_layout()
    if save:
        figname = f"robustness_training_{dataset_name}"
        save_plot(figname=figname)
    plt.show()


# ----------------------------------------------------------------------------
# Results functions
# ----------------------------------------------------------------------------


def get_csv_results() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    for dataset_name in dataset_names:
        lrs = LR_VALUES[dataset_name]
        get_evals_csv(dataset_name=dataset_name, seeds=seeds, lrs=lrs)
        get_linear_probing_csv(dataset_name=dataset_name, lrs=lrs)


def get_table_results() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    table_results(dataset_names=dataset_names, seeds=seeds)


def get_statistical_test() -> None:
    data = """
    Cifar10 98.91±0.07 99.09±0.05 98.91±0.06 98.72±0.05 98.67±0.03 99.02±0.02 91.95
    Cifar100 92.65±0.07 92.85±0.07 92.31±0.11 91.93±0.11 91.43±0.07 92.74±0.05 65.43
    Contrast 97.09±0.11 97.06±0.08 96.28±0.11 96.67±0.20 96.89±0.19 97.23±0.18 73.25
    Gaussian Noise 89.41±0.53 89.49±0.16 88.49±0.51 89.55±0.04 88.99±0.24 87.14±1.16 49.20
    Motion Blur 94.72±0.21 94.53±0.06 94.04±0.16 93.95±0.34 93.25±0.29 94.67±0.14 59.70
    Snow 95.47±0.13 95.52±0.20 95.27±0.29 95.51±0.11 95.15±0.10 95.42±0.13 59.25
    Speckle Noise 90.07±0.32 89.85±0.34 89.22±0.31 89.71±0.17 89.74±0.31 89.58±0.43 51.15
    Clipart 77.31±0.41 76.47±0.24 76.54±0.17 74.37±0.08 74.65±0.16 78.50±0.49 42.76
    Sketch 69.23±0.05 69.31±0.18 69.49±0.20 65.27±0.15 65.76±0.10 71.30±0.26 29.08
    Flowers102 99.03±0.08 99.05±0.06 98.86±0.06 99.21±0.07 98.99±0.20 99.15±0.05 96.34
    Pet 94.37±0.13 94.26±0.26 93.98±0.20 94.39±0.13 94.46±0.11 94.57±0.29 88.33
    """
    stat_results(data=data)


def plot_figures() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    save = True
    get_intro(dataset_names=dataset_names, seeds=seeds, save=save)

    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    save = True
    get_best_performance(dataset_names=dataset_names, seeds=seeds, save=save)

    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    save = True
    get_robustness_all(dataset_names=dataset_names, seeds=seeds, save=save)

    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    save = True
    for seed in [0, 42, 3407]:
        for dataset_name in dataset_names:
            get_training_evolution(dataset_name=dataset_name, seed=seed, save=save)

    save = True
    get_robustness_training_domainnet_sketch(save=save)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"csv": get_csv_results, "table": get_table_results, "stat": get_statistical_test, "plot": plot_figures})


# %% CLI
if __name__ == "__main__":
    main()
# %%
