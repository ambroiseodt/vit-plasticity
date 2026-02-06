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
SAVE_DIR = SAVING_DIR / "analysis"

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
    figure_path = FIGURE_DIR / "analysis"
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


def get_config(dataset_name: str, model_name: str, pretrained: bool = True) -> str:
    r"""Return configuration"""

    # ViT model name
    if model_name == "huge":
        vit_model_name = f"vit-{model_name}-patch14-224-in21k"
    else:
        vit_model_name = f"vit-{model_name}-patch16-224-in21k"

    # Config name
    config_name = f"analysis_{vit_model_name}_pretrained_{pretrained}"
    config_name += f"_{dataset_name}"

    return config_name


def get_all_plasticity(
    dataset_name: str,
    pretrained: bool,
    save: bool = False,
    ncol: int = 6,
) -> None:
    r"""Plot the plasticity of ViT components."""
    # Figure parameters
    width = 4
    height = 4
    figsize = (3 * width, height)
    fig, axes = plt.subplots(ncols=3, figsize=figsize)

    # Plot theory validation
    ax = axes[0]
    model_name = "base"
    ax.set_title(f"{MODEL_NAMES[model_name]}")
    config = get_config(dataset_name=dataset_name, model_name=model_name, pretrained=True)
    dict_df = get_plasticity(path=SAVE_DIR / config)
    plasticity_rank = [5, 1, 4, 2, 3]
    # Distribution of plasticity
    ax = axes[0]
    plasticity_rank = [5, 1, 4, 2, 3]
    yname = "Rates of Change"
    df = {"": [], yname: []}
    model_name = "base"
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
    yticks = [1, 6, 11]
    ax.set_ylim(ymin_manual, yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(r"Theoretical Plasticity Rank ($\downarrow$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot evolution over layers for ViT_Base
    ax = axes[1]
    model_name = "base"
    ax.set_title(f"{MODEL_NAMES[model_name]}")
    config = get_config(dataset_name=dataset_name, model_name=model_name, pretrained=True)
    dict_df = get_plasticity(path=SAVE_DIR / config)
    n_layers = N_LAYERS[model_name]
    x_range = np.arange(n_layers) / (n_layers - 1) * 100
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        std = np.std(ratio, axis=-1)
        ci = 1.96 * std / np.sqrt(ratio.shape[-1])
        trainable_component = VIT_COMPONENTS[j]
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
        yticks = [1, 12, 23]
        ax.set_ylim(ymin_manual, yticks[-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))
        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Plot evolution over layers for ViT-Huge
    ax = axes[2]
    model_name = "huge"
    ax.set_title(f"{MODEL_NAMES[model_name]}")
    config = get_config(dataset_name=dataset_name, model_name=model_name, pretrained=True)
    dict_df = get_plasticity(path=SAVE_DIR / config)
    n_layers = N_LAYERS[model_name]
    x_range = np.arange(n_layers) / (n_layers - 1) * 100
    for j, key in enumerate(dict_df.keys()):
        ratio = np.asarray(dict_df[key])
        mean = np.mean(ratio, axis=-1)
        std = np.std(ratio, axis=-1)
        ci = 1.96 * std / np.sqrt(ratio.shape[-1])
        trainable_component = VIT_COMPONENTS[j]
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
        yticks = [1, 12, 23]
        ax.set_ylim(ymin_manual, yticks[-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))
        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        ax.set_ylabel(r"Plasticity $\mathscr{P}(f)$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[2].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    ordered_index = [1, 3, 4, 2, 0]
    lines = [lines[i] for i in ordered_index]
    labels = [labels[i] for i in ordered_index]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
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
        figname = f"plasticity_{dataset_name}"
        save_plot(figname=figname)
    plt.show()


def plot_figures() -> None:
    pretrained = True
    save = True
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c-corruption-contrast-severity-5",
        "cifar10_c-corruption-gaussian_noise-severity-5",
        "cifar10_c-corruption-motion_blur-severity-5",
        "cifar10_c-corruption-snow-severity-5",
        "cifar10_c-corruption-speckle_noise-severity-5",
        "domainnet-clipart",
        "domainnet-sketch",
        "flowers102",
        "pet",
    ]
    for dataset_name in dataset_names:
        get_all_plasticity(
            dataset_name=dataset_name,
            pretrained=pretrained,
            save=save,
        )


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "plot": plot_figures,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
