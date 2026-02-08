r"""
Plotting functions to the illustrate the benefits of plasticity during gradient descent.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import glob
import logging
import os
import pickle
import re

import fire
import fitz
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.collections import PathCollection
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from vitef.config import FIGURE_DIR, set_seed
from vitef.data import build_loader

os.environ["OMP_NUM_THREADS"] = "1"


from vitef.config import SAVING_DIR

logger = logging.getLogger("vitef")

# Paths
SAVE_DIR = SAVING_DIR / "loss_landscape"

# Reproducibility
set_seed(42)

# Figure golden ratio (from ICML style file)
FONTSIZE = 15
FONTSIZE_LEG = 12

# ----------------------------------------------------------------------------
# Generate random orthogonal basis in parameter space using PCA
# ----------------------------------------------------------------------------


def get_pca_basis(
    params: list, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, n_steps: int, lr: float, device: str
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    r"""
    Performs a short optimization run to capture the gradient descent trajectory,
    and apply PCA to find the two main directions of variation.
    """

    trajectory = []
    optimizer = torch.optim.SGD(params, lr=lr)
    base_params = [p.data.clone() for p in params]

    # Collect weight during training to create trajectory
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        flatten_params = torch.cat([p.data.view(-1) for p in params]).cpu().numpy()
        trajectory.append(flatten_params)

    # Reset weights to original state after scanning
    for p, w in zip(params, base_params, strict=False):
        p.data.copy_(w)

    # PCA over the trajectory
    trajectory = np.array(trajectory)
    pca = PCA(n_components=2)
    pca.fit(trajectory)

    # Create orthogonal directions from PCA
    pca_dx = torch.from_numpy(pca.components_[0]).float()
    pca_dy = torch.from_numpy(pca.components_[1]).float()
    pca_dx, pca_dy = map(lambda x: x.to(device), (pca_dx, pca_dy))

    dx, dy = [], []
    pointer = 0
    for p in params:
        numel = p.numel()

        # Reshape PCA vectors back to the paraameter shape
        dx.append(pca_dx[pointer : pointer + numel].view_as(p))
        dy.append(pca_dy[pointer : pointer + numel].view_as(p))
        pointer += numel

    return dx, dy


# ----------------------------------------------------------------------------
# Get rates of change and loss landscape
# ----------------------------------------------------------------------------
def get_rates_of_change(
    dataset_name: str,
    batch_size: int,
    trainable_component: str,
    block: int,
    n_steps: int,
    lr: float,
    resolution: float,
    grid_range: float,
    device: str,
) -> np.array:
    r"""Compute the rates of change for a given component over a grid in the gradient directions."""

    # Get dataset
    loader_config = {
        "dataset_name": "cifar10",
        "batch_size": batch_size,
        "mode": "test",
        "size": 224,
    }
    loader = build_loader(config=loader_config)

    # Setup model
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = model.to(device)

    # Get one batch for the analysis
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = map(lambda x: x.to(device), (x_batch, y_batch))

    # 2. Select Parameters & Setup Functional Probe
    name_filter = {"ln1": "norm1", "mha": "attn", "ln2": "norm2", "fc1": "fc1", "fc2": "fc2"}[trainable_component]
    target_params = [p for n, p in model.named_parameters() if name_filter in n]
    base_weights = [p.data.clone() for p in target_params]
    for p in model.parameters():
        p.requires_grad = False
    for p in target_params:
        p.requires_grad = True

    # Extract the specific module for functional probing
    module, name = next((m, n) for n, m in model.named_modules() if (name_filter in n and f"blocks.{block}." in n))

    # Recover PCA basis in parameter space
    p_dx, p_dy = get_pca_basis(target_params, model, x_batch, y_batch, n_steps=n_steps, lr=lr, device=device)

    with torch.no_grad():
        # Embed input images to feature space
        emb = model.patch_embed(x_batch)
        emb = model._pos_embed(emb)

        # Use the first sample in the batch as our reference 'x' for the grid
        feat_input = emb[0:1].detach().requires_grad_(True)

    # Recover the direction in feature space where the component is the most sensitive
    feat_out = module(feat_input)
    feat_out.norm().backward()
    f_dx = feat_input.grad.detach()
    f_dx /= f_dx.norm()

    # Create an orthogonal direction using Rademacher noise
    f_dy = torch.sign(torch.randn_like(f_dx))
    f_dy -= torch.sum(f_dy * f_dx) * f_dx
    f_dy /= f_dy.norm()

    # Recover surfaces
    u_coords = np.linspace(-grid_range, grid_range, resolution)
    v_coords = np.linspace(-grid_range, grid_range, resolution)
    Z_loss = np.zeros((resolution, resolution))
    Z_func = np.zeros((resolution, resolution))
    with torch.no_grad():
        f_x = module(feat_input)

    print(f"Mapping Surfaces for {trainable_component.upper()}...")
    for i, u in enumerate(u_coords):
        for j, v in enumerate(v_coords):
            # Loss
            for p, w, x, y in zip(target_params, base_weights, p_dx, p_dy, strict=False):
                p.data = w + u * x + v * y
            Z_loss[j, i] = F.cross_entropy(model(x_batch), y_batch).item()

            # Functional rate of change ||f(x)-f(y)|| / ||x-y||
            delta = u * f_dx + v * f_dy
            dist_in = max(torch.norm(delta).item(), 1e-8)
            f_y = module(feat_input + delta)
            Z_func[j, i] = max((torch.norm(f_y - f_x) / dist_in).item(), 1e-8)

    # Reset weights for trajectory
    for p, w in zip(target_params, base_weights, strict=False):
        p.data.copy_(w)

    # Recover gradient descent trajectory in parameter space
    optimizer = torch.optim.SGD(target_params, lr=lr)
    trajectory = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        disp = [p.data - w for p, w in zip(target_params, base_weights, strict=False)]
        tu = sum(torch.sum(d * x) for d, x in zip(disp, p_dx, strict=False)).item()
        tv = sum(torch.sum(d * y) for d, y in zip(disp, p_dy, strict=False)).item()
        trajectory.append((tu, tv))
        print(f"Step {_ + 1}/{n_steps}: Loss={loss.item():.4f}")

    return Z_loss, Z_func, u_coords, v_coords, trajectory


# ----------------------------------------------------------------------------
# Results function
# ----------------------------------------------------------------------------
def get_analysis(
    dataset_name: str,
    batch_size: int,
    trainable_component: str,
    block: int,
    n_steps: int,
    lr: float,
    resolution: float,
    grid_range: float,
    device: str,
) -> None:
    Z_loss, Z_func, u_coords, v_coords, trajectory = get_rates_of_change(
        dataset_name=dataset_name,
        batch_size=batch_size,
        trainable_component=trainable_component,
        block=block,
        n_steps=n_steps,
        lr=lr,
        resolution=resolution,
        grid_range=grid_range,
        device=device,
    )

    # Saving results
    save_dir = SAVE_DIR / f"{trainable_component}_block_{block}"
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving results in {save_dir}.")
    pickle.dump(Z_loss, open(save_dir / "loss.pkl", "wb"))
    pickle.dump(Z_func, open(save_dir / "func.pkl", "wb"))
    pickle.dump(u_coords, open(save_dir / "u_coords.pkl", "wb"))
    pickle.dump(v_coords, open(save_dir / "v_coords.pkl", "wb"))
    pickle.dump(trajectory, open(save_dir / "traj.pkl", "wb"))


# ----------------------------------------------------------------------------
# Plotting function
# ----------------------------------------------------------------------------


def save_plot(figname: str, folder: str = None, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    figure_path = FIGURE_DIR / "loss_landscape"
    if folder:
        figure_path = figure_path / folder
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def save_results() -> None:
    device = "cuda:3"
    dataset_name = "cifar10"
    batch_size = 4
    n_steps = 20
    lr = 1e-3
    resolution = 20
    grid_range = 0.5
    block = 0
    for trainable_component in ["ln1", "fc1", "mha"]:
        get_analysis(
            dataset_name=dataset_name,
            batch_size=batch_size,
            trainable_component=trainable_component,
            block=block,
            n_steps=n_steps,
            lr=lr,
            resolution=resolution,
            grid_range=grid_range,
            device=device,
        )


def get_results(
    save: bool = False,
    ncol: int = 6,
) -> None:
    # Figure parameters
    width = 4
    height = 4
    nrows = 2
    ncols = 2
    figsize = (ncols * width, nrows * height)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.3)

    # Recover results LN1
    block = 0
    trainable_component = "ln1"
    path = SAVE_DIR / f"{trainable_component}_block_{block}"
    Z_loss_ln1 = pickle.load(open(path / "loss.pkl", "rb"))
    Z_func_ln1 = pickle.load(open(path / "func.pkl", "rb"))
    u_coords_ln1 = pickle.load(open(path / "u_coords.pkl", "rb"))
    v_coords_ln1 = pickle.load(open(path / "v_coords.pkl", "rb"))
    trajectory_ln1 = pickle.load(open(path / "traj.pkl", "rb"))

    # Recover results MHA
    block = 0
    trainable_component = "mha"
    path = SAVE_DIR / f"{trainable_component}_block_{block}"
    Z_loss_mha = pickle.load(open(path / "loss.pkl", "rb"))
    Z_func_mha = pickle.load(open(path / "func.pkl", "rb"))
    u_coords_mha = pickle.load(open(path / "u_coords.pkl", "rb"))
    v_coords_mha = pickle.load(open(path / "v_coords.pkl", "rb"))
    trajectory_mha = pickle.load(open(path / "traj.pkl", "rb"))

    # Recover surgace range
    z1_min, z1_max = Z_func_ln1.min(), Z_func_ln1.max()
    z2_min, z2_max = Z_func_mha.min(), Z_func_mha.max()
    row1_min = min(z1_min, z2_min)
    row1_max = max(z1_max, z2_max)

    # Recover loss range
    z1_min, z1_max = Z_loss_ln1.min(), Z_loss_ln1.max()
    z2_min, z2_max = Z_loss_mha.min(), Z_loss_mha.max()
    row2_min = min(z1_min, z2_min)
    row2_max = max(z1_max, z2_max)

    # Rescaling for visualization purposes
    Z_func_ln1 /= row1_max
    Z_func_mha /= row1_max
    row1_min /= row1_max
    row1_max /= row1_max

    # Functional rates of change
    cmap = "viridis"
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    X_ln1, Y_ln1 = np.meshgrid(u_coords_ln1, v_coords_ln1)
    ax1.plot_surface(
        X_ln1,
        Y_ln1,
        Z_func_ln1,
        vmin=row1_min,
        vmax=row1_max,
        cmap=cmap,
        rcount=100,
        ccount=100,
        antialiased=False,
    )

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    X_mha, Y_mha = np.meshgrid(u_coords_mha, v_coords_mha)
    surf2 = ax2.plot_surface(
        X_mha,
        Y_mha,
        Z_func_mha,
        vmin=row1_min,
        vmax=row1_max,
        cmap=cmap,
        rcount=100,
        ccount=100,
        antialiased=False,
    )

    for ax in [ax1, ax2]:
        ax.set_zlim(row1_min, row1_max)
        ax.set_box_aspect((1, 1, 0.8))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_xticks([-0.4, 0.0, 0.4])
        ax.set_yticks([-0.4, 0.0, 0.4])
        ax.set_proj_type("ortho")
        ax.view_init(elev=25, azim=-135)

        # Grid
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update(
                {
                    "color": (0, 0, 0, 0.15),
                    "linewidth": 0.2,
                    "linestyle": "-",
                }
            )
    ax1.set_zlabel("Rate of Change" + r"$\quad \frac{\|f(x)-f(y)\|}{\|x-y\|}$", fontsize=FONTSIZE, labelpad=10)
    ax1.set_title("Smooth LayerNorm", fontsize=FONTSIZE)
    ax2.set_title("Non-Smooth Attention", fontsize=FONTSIZE)

    # Add Colorbar for Row 1
    pos1 = ax2.get_position()
    cax1 = fig.add_axes([pos1.x1 + 0.04, pos1.y0 + 0.02, 0.02, pos1.height - 0.04])
    cbar1 = fig.colorbar(surf2, cax=cax1)
    cbar1.set_ticks([0.3, 0.6, 0.9])
    cbar1.set_ticklabels([0.3, 0.6, 0.9])

    # Loss landscape
    cmap = "magma"
    marker = "*"
    ms = 280
    lw = 1
    color = "#547398"

    # Rescaling for visualization purposes
    Z_loss_ln1 /= row2_max
    Z_loss_mha /= row2_max
    row2_min /= row2_max
    row2_max /= row2_max

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.contourf(
        X_ln1,
        Y_ln1,
        Z_loss_ln1,
        levels=40,
        cmap=cmap,
        vmin=row2_min,
        vmax=row2_max,
    )
    tx, ty = zip(*[(0, 0)] + trajectory_ln1, strict=False)
    ax3.plot(tx, ty, "w-o", lw=lw, markersize=4)
    ax3.scatter(tx[-1], ty[-1], marker=marker, color=color, s=ms, zorder=5, edgecolors="#3a4b71", lw=0.05)

    ax4 = fig.add_subplot(gs[1, 1])
    cont4 = ax4.contourf(
        X_mha,
        Y_mha,
        Z_loss_mha,
        levels=40,
        cmap=cmap,
        vmin=row2_min,
        vmax=row2_max,
    )
    tx, ty = zip(*[(0, 0)] + trajectory_mha, strict=False)
    ax4.plot(tx, ty, "w-o", lw=lw, markersize=4, label="SGD updates")
    ax4.scatter(
        tx[-1], ty[-1], marker=marker, color=color, s=ms, zorder=5, label="final step", edgecolors="#3a4b71", lw=0.05
    )

    for ax in [ax3, ax4]:
        ax.set_aspect("equal")
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        xlims = [-0.25, 0.25]
        ylims = [-0.25, 0.25]
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xticks([-0.2, 0.0, 0.2])
        ax.set_yticks([-0.2, 0.0, 0.2])

    ax4.sharey(ax3)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax4.tick_params(axis="y", which="both", length=0)

    # Colorbar
    pos2 = ax4.get_position()
    cax2 = fig.add_axes([pos2.x1 + 0.04, pos2.y0, 0.02, pos2.height])
    cbar2 = fig.colorbar(cont4, cax=cax2)
    cbar2.set_ticks([0.0, 0.5, 1.0])
    cbar2.set_ticklabels([0.0, 0.5, 1.0])

    # Legend
    ax3.set_ylabel("Loss Landscape", fontsize=FONTSIZE)
    leg = ax4.legend(fontsize=FONTSIZE_LEG, frameon=False)
    texts = leg.get_texts()
    for text in texts:
        text.set_color("white")

    # Override marker size for second marker
    handle = leg.legend_handles[-1]
    if isinstance(handle, PathCollection):
        handle.set_sizes([200])

    if save:
        figname = "loss_landscape"
        save_plot(figname=figname)
    plt.show()


def get_frames(save: bool) -> None:

    # Visualization
    lw_grid = 0.2
    lw_surface = 0.1
    lw_contour = 0.2
    alpha_surface = 0.3
    alpha_contour = 0.8
    levels = 40
    n_frames = 20
    rcount = 100
    ccount = 100

    # Figure parameters
    width = 4
    height = 4
    nrows = 2
    ncols = 2
    figsize = (ncols * width, nrows * height)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.3)

    # Recover results LN1
    block = 0
    trainable_component = "ln1"
    path = SAVE_DIR / f"{trainable_component}_block_{block}"
    Z_loss_ln1 = pickle.load(open(path / "loss.pkl", "rb"))
    Z_func_ln1 = pickle.load(open(path / "func.pkl", "rb"))
    u_coords_ln1 = pickle.load(open(path / "u_coords.pkl", "rb"))
    v_coords_ln1 = pickle.load(open(path / "v_coords.pkl", "rb"))
    trajectory_ln1 = pickle.load(open(path / "traj.pkl", "rb"))

    # Recover results MHA
    block = 0
    trainable_component = "mha"
    path = SAVE_DIR / f"{trainable_component}_block_{block}"
    Z_loss_mha = pickle.load(open(path / "loss.pkl", "rb"))
    Z_func_mha = pickle.load(open(path / "func.pkl", "rb"))
    u_coords_mha = pickle.load(open(path / "u_coords.pkl", "rb"))
    v_coords_mha = pickle.load(open(path / "v_coords.pkl", "rb"))
    trajectory_mha = pickle.load(open(path / "traj.pkl", "rb"))

    # Recover surgace range
    z1_min, z1_max = Z_func_ln1.min(), Z_func_ln1.max()
    z2_min, z2_max = Z_func_mha.min(), Z_func_mha.max()
    row1_min = min(z1_min, z2_min)
    row1_max = max(z1_max, z2_max)

    # Recover loss range
    z1_min, z1_max = Z_loss_ln1.min(), Z_loss_ln1.max()
    z2_min, z2_max = Z_loss_mha.min(), Z_loss_mha.max()
    row2_min = min(z1_min, z2_min)
    row2_max = max(z1_max, z2_max)

    # Rescaling for visualization purposes
    Z_func_ln1 /= row1_max
    Z_func_mha /= row1_max
    row1_min /= row1_max
    row1_max /= row1_max

    # Functional rates of change
    cmap = "viridis"
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    X_ln1, Y_ln1 = np.meshgrid(u_coords_ln1, v_coords_ln1)
    surf1 = ax1.plot_surface(
        X_ln1,
        Y_ln1,
        Z_func_ln1,
        vmin=row1_min,
        vmax=row1_max,
        cmap=cmap,
        rcount=rcount,
        ccount=ccount,
        antialiased=False,
        shade=True,
    )
    surf1.set_edgecolor((1, 1, 1, alpha_surface))
    surf1.set_linewidth(lw_surface)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    X_mha, Y_mha = np.meshgrid(u_coords_mha, v_coords_mha)
    surf2 = ax2.plot_surface(
        X_mha,
        Y_mha,
        Z_func_mha,
        vmin=row1_min,
        vmax=row1_max,
        cmap=cmap,
        rcount=rcount,
        ccount=ccount,
        antialiased=False,
        zorder=4,
        shade=True,
    )
    surf1.set_edgecolor((1, 1, 1, alpha_surface))
    surf2.set_linewidth(lw_surface)

    for ax in [ax1, ax2]:
        ax.set_zlim(row1_min, row1_max)
        ax.set_box_aspect((1, 1, 0.8))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_xticks([-0.4, 0.0, 0.4])
        ax.set_yticks([-0.4, 0.0, 0.4])
        ax.set_proj_type("ortho")
        ax.view_init(elev=25, azim=-135)

        # Grid
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update(
                {
                    "color": (0, 0, 0, 0.35),
                    "linewidth": lw_grid,
                    "linestyle": "-",
                }
            )
    ax1.set_zlabel("Rate of Change" + r"$\quad \frac{\|f(x)-f(y)\|}{\|x-y\|}$", fontsize=FONTSIZE, labelpad=10)
    ax1.set_title("Smooth LayerNorm", fontsize=FONTSIZE)
    ax2.set_title("Non-Smooth Attention", fontsize=FONTSIZE)

    # Add Colorbar for Row 1
    pos1 = ax2.get_position()
    cax1 = fig.add_axes([pos1.x1 + 0.04, pos1.y0 + 0.02, 0.02, pos1.height - 0.04])
    cbar1 = fig.colorbar(surf2, cax=cax1)
    cbar1.set_ticks([0.3, 0.6, 0.9])
    cbar1.set_ticklabels([0.3, 0.6, 0.9])

    # Loss landscape
    cmap = "magma"
    marker = "*"
    ms = 280
    lw = 1
    color = "#547398"

    # Rescaling for visualization purposes
    Z_loss_ln1 /= row2_max
    Z_loss_mha /= row2_max
    row2_min /= row2_max
    row2_max /= row2_max

    ax3 = fig.add_subplot(gs[1, 0])
    cont3 = ax3.contourf(
        X_ln1,
        Y_ln1,
        Z_loss_ln1,
        levels=levels,
        cmap=cmap,
        vmin=row2_min,
        vmax=row2_max,
    )
    ax3.contour(
        X_ln1, Y_ln1, Z_loss_ln1, levels=cont3.levels, colors="white", linewidths=lw_contour, alpha=alpha_contour
    )
    tx3, ty3 = zip(*[(0, 0)] + trajectory_ln1, strict=False)
    ax3.scatter(tx3[-1], ty3[-1], marker=marker, color=color, s=ms, zorder=5, edgecolors="#3a4b71", lw=0.05)

    ax4 = fig.add_subplot(gs[1, 1])

    cont4 = ax4.contourf(X_mha, Y_mha, Z_loss_mha, levels=levels, cmap=cmap, vmin=row2_min, vmax=row2_max)
    ax4.contour(
        X_mha, Y_mha, Z_loss_mha, levels=cont4.levels, colors="white", linewidths=lw_contour, alpha=alpha_contour
    )
    tx4, ty4 = zip(*[(0, 0)] + trajectory_mha, strict=False)
    ax4.scatter(
        tx4[-1],
        ty4[-1],
        marker=marker,
        color=color,
        s=ms,
        zorder=5,
        label="final step",
        edgecolors="#3a4b71",
        lw=0.05,
    )

    for ax in [ax3, ax4]:
        ax.set_aspect("equal")
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        xlims = [-0.25, 0.25]
        ylims = [-0.25, 0.25]
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xticks([-0.2, 0.0, 0.2])
        ax.set_yticks([-0.2, 0.0, 0.2])

    ax4.sharey(ax3)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax4.tick_params(axis="y", which="both", length=0)

    # Colorbar
    pos2 = ax4.get_position()
    cax2 = fig.add_axes([pos2.x1 + 0.04, pos2.y0, 0.02, pos2.height])
    cbar2 = fig.colorbar(cont4, cax=cax2)
    cbar2.set_ticks([0.0, 0.5, 1.0])
    cbar2.set_ticklabels([0.0, 0.5, 1.0])

    # Animated gradient descent
    (path_line1,) = ax3.plot([], [], "w-o", lw=lw, markersize=4, label="SGD updates")
    (current_point1,) = ax3.plot([], [], "w-o", lw=lw, markersize=4)

    (path_line2,) = ax4.plot([], [], "w-o", lw=lw, markersize=4, label="SGD updates")
    (current_point2,) = ax4.plot([], [], "w-o", lw=lw, markersize=4)

    for frame in range(n_frames):
        # Update first contour plot
        if not frame:
            x_data, y_data = [tx3[0]], [ty3[0]]
        else:
            x_data, y_data = tx3[:frame], ty3[:frame]
        path_line1.set_data(x_data, y_data)
        current_point1.set_data([x_data[-1]], [y_data[-1]])

        # Update second contour plot
        if not frame:
            x_data, y_data = [tx4[0]], [ty4[0]]
        else:
            x_data, y_data = tx4[:frame], ty4[:frame]
        path_line2.set_data(x_data, y_data)
        current_point2.set_data([x_data[-1]], [y_data[-1]])

        # Legend
        ax3.set_ylabel("Loss Landscape", fontsize=FONTSIZE)
        leg = ax4.legend(fontsize=FONTSIZE_LEG, frameon=False)
        texts = leg.get_texts()
        for text in texts:
            text.set_color("white")

        # Override marker size for second marker
        handle = leg.legend_handles[-1]
        if isinstance(handle, PathCollection):
            handle.set_sizes([200])

        if save:
            figname = f"frame_{frame}"
            folder = "frames"
            save_plot(figname=figname, folder=folder, dpi=300)

        plt.show()


def interpolate_frames(frames: list, n_steps: int = 5) -> list:
    r"""Generates intermediate frames between each pair of frames."""
    interpolated = []
    for i in range(len(frames) - 1):
        f1 = frames[i].astype(np.float32)
        f2 = frames[i + 1].astype(np.float32)

        # Linear interpolation
        for step in range(n_steps):
            alpha = step / n_steps
            blended = (1 - alpha) * f1 + alpha * f2
            interpolated.append(blended.astype(np.uint8))
    interpolated.append(frames[-1])

    return interpolated


def plot_figures() -> None:
    save = True
    get_results(save=save)


def plot_frames() -> None:
    save = True
    get_frames(save=save)


def plot_gif() -> None:
    folder_path = FIGURE_DIR / "loss_landscape" / "frames"
    file_pattern = "frame_*.pdf"
    search_path = os.path.join(folder_path, file_pattern)
    pdf_files = glob.glob(search_path)
    pdf_files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", f)])
    original_frames = []
    total_duration = 5000

    # Extraction
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        page = doc.load_page(0)
        zoom = 4.5
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, annots=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        original_frames.append(img)
        doc.close()

    # Interpolation
    n_steps = 5
    smooth_frames = interpolate_frames(original_frames, n_steps=n_steps)
    duration = total_duration / len(smooth_frames)

    # Save
    iio.imwrite(folder_path / "loss_landscape.gif", smooth_frames, duration=duration, loop=0, quantizer="nq")
    print(f"GIF generated with {len(smooth_frames)} frames.")


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"save": save_results, "plot": plot_figures, "frames": plot_frames, "gif": plot_gif})


# %% CLI
if __name__ == "__main__":
    main()
# %%
