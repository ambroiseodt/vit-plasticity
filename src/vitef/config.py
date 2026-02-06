r"""
Configuration file.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import subprocess
from pathlib import Path

import numpy as np
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
ROOT_DIR = Path(__file__).parents[2].resolve()
DATASET_DIR = ROOT_DIR / Path("datasets/")
FIGURE_DIR = ROOT_DIR / Path("figures/")
MODEL_DIR = ROOT_DIR / Path("checkpoints/")
RESULT_DIR = ROOT_DIR / Path("results/")
SAVING_DIR = ROOT_DIR / Path("savings/")


# Task-specific suffices
CV_SUFFIX = "computer_vision"
NLP_SUFFIX = "nlp"
TS_SUFFIX = "time_series"

# Tex available
USETEX = not subprocess.run(["which", "pdflatex"], stdout=subprocess.DEVNULL).returncode
USETEX = False


def set_seed(seed: int) -> np.random.default_rng:
    r"""Set seed for reprodubilty.

    Parameters
    ----------
    seed: int
        Seed value.

    Returns
    -------
    RNG: Generator
        Random number generator.
    """
    np.random.seed(seed=seed)
    RNG = np.random.default_rng(seed)
    torch.manual_seed(seed=seed)

    return RNG
