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
        ratio = np.mean(np.asarray(dict_df[key]), axis=-1)
        ratio = np.mean(ratio)
        print(key, ratio)

    for model_name in ["dinov3_vit7b16", "dinov3_vits16"]:
        dataset_name = "cifar10"
        print(model_name)
        config = get_config(dataset_name=dataset_name, model_name=model_name)
        dict_df = get_plasticity(path=SAVE_DIR / config)
        for key in dict_df.keys():
            ratio = np.mean(np.asarray(dict_df[key]), axis=-1)
            ratio = np.mean(ratio)
            print(key, ratio)


def get_table_results() -> None:
    table_results()


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
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
