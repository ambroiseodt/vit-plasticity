r"""
Plotting functions related to the plasticity upper bounds.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import math

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from vitef.config import FIGURE_DIR, SAVING_DIR
from vitef.data import build_loader, make_iterable
from vitef.models import build_model

# Paths
SAVE_DIR = SAVING_DIR / "analysis"

# Parameters
VIT_COMPONENTS = ["LN1", "MHA", "LN2", "FC1", "FC2"]
N_LAYERS = {"base": 12, "large": 24, "huge": 32}
MODEL_NAMES = {"base": "ViT-Base", "large": "ViT-Large", "huge": "ViT-Huge"}

# Figure golden ratio (from ICML style file)
WIDTH = 6
HEIGHT = 4
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
# Utils functions
# ------------------------------------------------------------------------------


def get_radius(
    model_name: str, patch_size: int, dataset_name: str, batch_size: int, max_steps: int, device: str = "cuda:0"
) -> None:
    r"""Estimate the average token norm over embedded images of a dataset. For Cifar10, we obtain r=19.4"""

    # Compute the average token norm
    loader_config = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "mode": "test",
        "size": 224,
    }
    loader = build_loader(config=loader_config, drop_last=False)
    model_config = {
        "implementation": "vit",
        "model_name": model_name,
        "pretrained": True,
        "in21k": True,
        "patch_size": patch_size,
        "image_dim": (3, 224, 224),
    }
    model = build_model(config=model_config, device=device)
    embedding_layer = model.model.embedding

    # Make loaders iterable
    loader = make_iterable(loader)
    iterator = iter(loader)

    # Loop of batches of data
    step = 0
    mean_norm = []
    while step < max_steps:
        x_batch, _ = next(iterator)
        emb_batch = embedding_layer(x_batch.to(device))
        token_norms = (emb_batch**2).sum(dim=-1)
        avg_token_norm = token_norms.mean(dim=-1).sqrt().mean()
        mean_norm.append(avg_token_norm.item())
        step += 1
    mean_norm = np.mean(mean_norm)
    print("The radius of the token embedding space is: r =", np.round(np.mean(mean_norm), 2))


# Compute the plasticity upper bounds
def norm_ub(model_name: str, device: str, patch_size: int) -> float:
    """Compute the upper bound on the sensitivity of layer norms."""

    model_config = {
        "implementation": "vit",
        "model_name": model_name,
        "pretrained": True,
        "in21k": True,
        "patch_size": patch_size,
        "image_dim": (3, 224, 224),
    }

    model = build_model(config=model_config, device=device)
    attn_norm_ub = []
    ffn_norm_ub = []
    for block in model.model.blocks:
        for name, param in block.named_parameters():
            if "attn_norm" in name and "weight" in name:
                A = param.data
                S = torch.max(A)
                attn_norm_ub.append(S.item())

            if "ffn_norm" in name and "weight" in name:
                A = param.data
                S = torch.max(A)
                ffn_norm_ub.append(S.item())

    return attn_norm_ub, ffn_norm_ub


def linear_ub(model_name: str, device: str, patch_size: int) -> float:
    """Compute the upper bound on the sensitivity of linear layers."""

    model_config = {
        "implementation": "vit",
        "model_name": model_name,
        "pretrained": True,
        "in21k": True,
        "patch_size": patch_size,
        "image_dim": (3, 224, 224),
    }

    model = build_model(config=model_config, device=device)
    fc1_ub = []
    fc2_ub = []
    for block in model.model.blocks:
        for name, param in block.named_parameters():
            if "fc1" in name and "weight" in name:
                A = param.data
                S = torch.linalg.svdvals(A)
                S = torch.max(S)
                fc1_ub.append(S.item())

            if "fc2" in name and "weight" in name:
                A = param.data
                S = torch.linalg.svdvals(A)
                S = torch.max(S)
                fc2_ub.append(S.item())

    return fc1_ub, fc2_ub


def attention_ub(model_name: str, device: str, patch_size: int, r: float) -> float:
    """Compute the upper bound on the sensitivity of the multihead self-attention module."""

    model_config = {
        "implementation": "vit",
        "model_name": model_name,
        "pretrained": True,
        "in21k": True,
        "patch_size": patch_size,
        "image_dim": (3, 224, 224),
    }
    model = build_model(config=model_config, device=device)
    mha_ub = []
    n_heads = {"base": 12, "large": 16, "huge": 16}[model_name]
    emb_dim = {"base": 768, "large": 1024, "huge": 1280}[model_name]
    seq_len = {14: 257, 16: 197}[patch_size]
    emb_dim_head = int(emb_dim // n_heads)
    for block in model.model.blocks:
        O_h, V_h, QK_h = [], [], []
        for name, param in block.named_parameters():
            if "attn.output" in name and "weight" in name:
                A = param.data
                for i in range(n_heads):
                    A_h = A[:, i * emb_dim_head : (i + 1) * emb_dim_head]
                    O_h.append(torch.linalg.svdvals(A_h).max().item())

            if "attn.qkv_mat" in name and "weight" in name:
                A = param.data
                q, k, v = A.chunk(3, dim=0)
                for i in range(n_heads):
                    q_h = q[:, i * emb_dim_head : (i + 1) * emb_dim_head]
                    k_h = k[:, i * emb_dim_head : (i + 1) * emb_dim_head]
                    v_h = v[:, i * emb_dim_head : (i + 1) * emb_dim_head]
                    qk_h = q_h @ k_h.transpose(-1, -2) / math.sqrt(len(q_h[0]))
                    V_h.append(torch.linalg.svdvals(v_h).max().item())
                    QK_h.append(torch.linalg.svdvals(qk_h).max().item())
        comp = 0
        for i in range(n_heads):
            comp += O_h[i] * V_h[i] * np.sqrt(3 * seq_len + (12 * seq_len + 3) * r**4 * QK_h[i] ** 2)
        mha_ub.append(comp)

    return mha_ub


def get_theoretical_bounds(
    model_name: str,
    patch_size: int,
    device: str = "cuda:0",
    r: float = 19.4,
) -> None:
    """Recover the theoretical bounds on plasticity of ViT components."""
    # Compute the plasticity upper bounds
    LN1, LN2 = norm_ub(model_name=model_name, device=device, patch_size=patch_size)
    FC1, FC2 = linear_ub(model_name=model_name, device=device, patch_size=patch_size)
    MHA = attention_ub(model_name=model_name, device=device, patch_size=patch_size, r=r)
    return LN1, MHA, LN2, FC1, FC2


# ------------------------------------------------------------------------------
# Utils functions
# ------------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    figure_path = FIGURE_DIR / "theory"
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def plot_theoretical_bounds(
    model_name: str,
    patch_size: int,
    device: str = "cuda:0",
    r: float = 19.4,
    save: bool = False,
    ncol: int = 6,
) -> None:
    """Plot the theoretical bounds on plasticity of ViT components."""
    figsize = (WIDTH, HEIGHT)
    fig = plt.figure(figsize=figsize)
    n_layers = N_LAYERS[model_name]
    x_range = np.arange(n_layers) / (n_layers - 1) * 100
    bounds = get_theoretical_bounds(model_name=model_name, patch_size=patch_size, device=device, r=r)
    for j, trainable_component in enumerate(VIT_COMPONENTS):
        plt.plot(x_range, bounds[j], label=trainable_component, color=COLORS[trainable_component], linewidth=LINEWIDTH)

    # Plotting
    ax = fig.axes[0]
    ax.set_yscale("log")

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([1e0, 1e4, 1e8])
    ax.set_yticklabels([r"$10^0$", r"$10^4$", r"$10^8$"])
    ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
    ax.set_ylabel("Plasticity Upper Bound", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
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
        figname = "theoretical_bounds"
        save_plot(figname=figname)
    plt.show()


def print_radius() -> None:
    dataset_name = "cifar10"
    batch_size = 16
    patch_size = 16
    model_name = "base"
    device = "cuda:0"
    max_steps = 1000
    get_radius(
        model_name=model_name,
        patch_size=patch_size,
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_steps=max_steps,
        device=device,
    )


def plot_figures() -> None:
    model_name = "base"
    patch_size = 16
    save = True
    plot_theoretical_bounds(model_name=model_name, patch_size=patch_size, save=save)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"radius": print_radius, "plot": plot_figures})


# %% CLI
if __name__ == "__main__":
    main()
# %%
