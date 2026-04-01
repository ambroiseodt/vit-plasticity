r"""
Plotting functions related to vision transformer plasticity.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import pickle

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vitef.config import FIGURE_DIR, SAVING_DIR

# Paths
SAVE_DIR = SAVING_DIR / "analysis/ablation"

# Parameters
VIT_COMPONENTS = ["LN1", "MHA", "LN2", "FC1", "FC2"]
N_LAYERS = {"base": 12, "large": 24, "huge": 32}
MODEL_NAMES = {"base": "ViT-Base", "large": "ViT-Large", "huge": "ViT-Huge"}

# Figure golden ratio (from ICML style file)
WIDTH = 4
HEIGHT = WIDTH
FONTSIZE = 15
FONTSIZE_LEGEND = 15
LINEWIDTH = 5
MARKER_SIZE = 150
RED_LINEWIDTH = 3
ALPHA_GRID = 0.8
ALPHA_CI = 0.8
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

# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    figure_path = FIGURE_DIR / "ablation/analysis"
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def get_plasticity(path: str) -> dict:
    r"""
    Recover the plasticity measure of ViT components (computed as the average rate of change).

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


def get_config(dataset_name: str, model_name: str) -> str:
    r"""Return configuration"""

    # Config name
    config_name = f"analysis_{model_name}_{dataset_name}"

    return config_name


def get_vit_config(dataset_name: str, model_name: str) -> str:
    r"""Return configuration"""

    # ViT model name
    if model_name == "huge":
        vit_model_name = f"vit-{model_name}-patch14-224-in21k"
    else:
        vit_model_name = f"vit-{model_name}-patch16-224-in21k"

    # Config name
    config_name = f"analysis_{vit_model_name}_pretrained_True"
    config_name += f"_{dataset_name}"

    return config_name


def table_results() -> None:
    # ViT
    dataset_name = "cifar10"
    model_name = "base"
    print(model_name)
    config = get_vit_config(dataset_name=dataset_name, model_name=model_name)
    dict_df = get_plasticity(path=SAVING_DIR / "analysis" / config)
    for key in dict_df.keys():
        ratio = np.mean(np.asarray(dict_df[key]))
        print(key, ratio)

    for model_name in ["dinov3_vit7b16"]:
        dataset_name = "cifar10"
        print(model_name)
        config = get_config(dataset_name=dataset_name, model_name=model_name)
        dict_df = get_plasticity(path=SAVE_DIR / config)
        for key in dict_df.keys():
            ratio = np.mean(np.asarray(dict_df[key]))
            print(key, ratio)

    # GPT2
    dataset_name = "ag_news"
    model_name = "gpt2_base"
    print(model_name)
    config = get_config(dataset_name=dataset_name, model_name=model_name)
    dict_df = get_plasticity(path=SAVE_DIR / config)
    for key in dict_df.keys():
        ratio = np.mean(np.asarray(dict_df[key]))
        print(key, ratio)


def get_plasticity_ablation(
    save: bool = False,
    ncol: int = 6,
) -> None:
    r"""Plot the plasticity of ViT components."""
    # Figure parameters
    width = 4.2
    height = 4
    ncols = 3
    nrows = 2
    figsize = (ncols * width, nrows * height)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Plot theory validation for ViT
    ax = axes[0, 0]
    model_name = "base"
    dataset_name = "cifar10"
    ax.set_title("ViT \n", fontsize=19)
    config = get_vit_config(dataset_name=dataset_name, model_name=model_name)
    dict_df = get_plasticity(path=SAVING_DIR / "analysis" / config)
    plasticity_rank = [5, 1, 4, 2, 3]
    plasticity_rank = [5, 1, 4, 2, 3]
    yname = "Rates of Change"
    df = {"": [], yname: []}
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        rank = plasticity_rank[j]
        for val in mean:
            df[""].append(rank)
            df[yname].append(val)
    colors = [COLORS[key] for key in ["MHA", "FC1", "FC2", "LN2", "LN1"]]
    sns.boxplot(
        data=df,
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
    box_yticks = [1, 6, 11]
    ax.set_ylim(ymin_manual, box_yticks[-1])
    ax.set_yticks(box_yticks)
    ax.set_yticklabels(box_yticks)
    ax.set_xlabel(r"Theoretical Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot evolution over layers for ViT
    ax = axes[1, 0]
    ax.set_title("ViT \n", fontsize=19)
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        std = np.std(ratio, axis=-1)
        ci = 1.96 * std / np.sqrt(ratio.shape[-1])
        trainable_component = VIT_COMPONENTS[j]
        n_layers = len(mean)
        x_range = np.arange(n_layers) / (n_layers - 1) * 100
        ax.plot(
            x_range,
            mean,
            linewidth=LINEWIDTH,
            color=COLORS[trainable_component],
            label=VIT_COMPONENTS[j],
        )
        ax.fill_between(x_range, mean - ci, mean + ci, color=COLORS[trainable_component], alpha=ALPHA_CI)

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_xticks([0, 50, 100])
        curve_yticks = [1, 12, 23]
        ax.set_ylim(ymin_manual, curve_yticks[-1])
        ax.set_yticks(curve_yticks)
        ax.set_yticklabels(np.array(curve_yticks, dtype=int))
        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot theory validation for Dinov3-7B
    ax = axes[0, 1]
    model_name = "dinov3_vit7b16"
    dataset_name = "cifar10"
    ax.set_title("DINOv3 \n", fontsize=19)
    config = get_config(dataset_name=dataset_name, model_name=model_name)
    dict_df = get_plasticity(path=SAVE_DIR / config)
    plasticity_rank = [5, 1, 4, 2, 3]
    plasticity_rank = [5, 1, 4, 2, 3]
    yname = "Rates of Change"
    df = {"": [], yname: []}
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        rank = plasticity_rank[j]
        for val in mean:
            df[""].append(rank)
            df[yname].append(val)
    colors = [COLORS[key] for key in ["MHA", "FC1", "FC2", "LN2", "LN1"]]
    sns.boxplot(
        data=df,
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
    box_yticks = [1, 10, 19]
    ax.set_ylim(ymin_manual, box_yticks[-1])
    ax.set_yticks(box_yticks)
    ax.set_yticklabels(np.array(box_yticks, dtype=int))
    ax.set_xlabel(r"Theoretical Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot evolution over layers for Dinov3-7B
    ax = axes[1, 1]
    model_name = "dinov3_vit7b16"
    ax.set_title("DINOv3 \n", fontsize=19)
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        std = np.std(ratio, axis=-1)
        ci = 1.96 * std / np.sqrt(ratio.shape[-1])
        trainable_component = VIT_COMPONENTS[j]
        n_layers = len(mean)
        x_range = np.arange(n_layers) / (n_layers - 1) * 100
        ax.plot(
            x_range,
            mean,
            linewidth=LINEWIDTH,
            color=COLORS[trainable_component],
            label=VIT_COMPONENTS[j],
        )
        ax.fill_between(x_range, mean - ci, mean + ci, color=COLORS[trainable_component], alpha=ALPHA_CI)

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_xticks([0, 50, 100])
        curve_yticks = [1, 13, 25]
        ax.set_ylim(ymin_manual, curve_yticks[-1])
        ax.set_yticks(curve_yticks)
        ax.set_yticklabels(np.array(curve_yticks, dtype=int))
        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot theory validation for GPT2
    ax = axes[0, 2]
    model_name = "gpt2_base"
    dataset_name = "ag_news"
    ax.set_title("GPT-2 \n", fontsize=19)
    config = get_config(dataset_name=dataset_name, model_name=model_name)
    dict_df = get_plasticity(path=SAVE_DIR / config)
    plasticity_rank = [5, 1, 4, 2, 3]
    plasticity_rank = [5, 1, 4, 2, 3]
    yname = "Rates of Change"
    df = {"": [], yname: []}
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        rank = plasticity_rank[j]
        for val in mean:
            df[""].append(rank)
            df[yname].append(val)
    colors = [COLORS[key] for key in ["MHA", "FC1", "FC2", "LN2", "LN1"]]
    sns.boxplot(
        data=df,
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
    box_yticks = [1, 8, 15]
    ax.set_ylim(ymin_manual, box_yticks[-1])
    ax.set_yticks(box_yticks)
    ax.set_yticklabels(np.array(box_yticks, dtype=int))
    ax.set_xlabel(r"Theoretical Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot evolution over layers for GPT2
    ax = axes[1, 2]
    model_name = "gpt2"
    ax.set_title("GPT-2 \n", fontsize=19)
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        std = np.std(ratio, axis=-1)
        ci = 1.96 * std / np.sqrt(ratio.shape[-1])
        trainable_component = VIT_COMPONENTS[j]
        n_layers = len(mean)
        x_range = np.arange(n_layers) / (n_layers - 1) * 100
        ax.plot(
            x_range,
            mean,
            linewidth=LINEWIDTH,
            color=COLORS[trainable_component],
            label=VIT_COMPONENTS[j],
        )
        ax.fill_between(x_range, mean - ci, mean + ci, color=COLORS[trainable_component], alpha=ALPHA_CI)

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(ymin_manual, curve_yticks[-1])
        curve_yticks = [1, 13, 25]
        ax.set_yticks(curve_yticks)
        ax.set_yticklabels(np.array(curve_yticks, dtype=int))
        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    ax = axes[1, 0]
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    ordered_index = [1, 3, 4, 2, 0]
    lines = [lines[i] for i in ordered_index]
    labels = [labels[i] for i in ordered_index]
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

    plt.tight_layout()
    if save:
        figname = "plasticity_ablation"
        save_plot(figname=figname)
    plt.show()


def get_table_results() -> None:
    table_results()


def plot_figures() -> None:
    save = True
    get_plasticity_ablation(save=save)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "table": get_table_results,
            "plot": plot_figures,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
