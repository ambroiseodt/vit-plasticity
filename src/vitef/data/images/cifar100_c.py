r"""
Cifar100-C Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from ...config import DATASET_DIR
from ...utils import deterministic_split


@dataclass
class Cifar100CDatasetConfig:
    r"""
    Cifar100-C configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data.
    corruption_type: str, default="brightness"
        Type of corruption applied on the data. Options are:
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur".
    corruption_severity: int, default=1
        Severity of the corruption applied on the data.
        Allowed values are 1, 2, 3, 4 and 5.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: str | None = None
    corruption_type: str = "brightness"
    corruption_severity: int = 1
    mode: str = "train"
    transform: Any | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Check if corruption type is valid
        assert self.corruption_type in [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur",
        ], f"Invalid corruption type {self.corruption_type}. See documentation for options."

        # Check if corruption severity is valid
        assert self.corruption_severity in [
            1,
            2,
            3,
            4,
            5,
        ], f"Invalid severity {self.severity}. Options are from 1 to 5."

        # Valid mode
        assert self.mode in ["train", "test"], f"Invalid mode {self.mode}. Options are 'train' and 'test."

        # Set default values
        if self.save_dir is None:
            self.save_dir = DATASET_DIR / "cifar100_c"


class Cifar100CDataset(Dataset):
    r"""
    Cifar100-C dataset from [1]_.

    It consits of 15 corruptions with 5 level of severity of the CIFAR-10 test set.
    To use it for finetuning, we split it into deterministic training and test sets.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data.
        corruption_type: str, default="brightness"
            Type of corruption applied on the data.
        corruption_severity: int
            Severity of the corruption applied on the data.
        mode: str
            Mode. Options are "train" and "test".
        transform: nn.Module
            Transformation to apply to the images.

    Notes
    ----------
    Cifar100-C was created by Dan Hendrycks and Thomas Dietterich.
    Official website: https://github.com/hendrycks/robustness.

    Download
    --------
    To download the data, run the following command from the root of the repository:
    ```bash
    $ wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    ```
    Move the downloaded file to the desired location, e.g, datasets/cifar100_c, and extract it with
    ```bash
    $ tar xvf CIFAR-100-C.tar -C . --strip-components 1
    ```
    References
    ----------
    .. [1] D. Hendrycks and t. Dietterich. Benchmarking Neural Network Robustness to Common
           Corruptions and Surface Variations. In ICLR 2019
    """

    def __init__(self, config: Cifar100CDatasetConfig):
        super().__init__()
        train = True if config.mode == "train" else False

        # Recover path to data corresponding to corruption and severity
        path_images = config.save_dir / f"{config.corruption_type}.npy"
        path_labels = config.save_dir / "labels.npy"

        # Recover index of the images
        start = (config.corruption_severity - 1) * 10_000
        end = start + 10_000

        # Recover images and corresponding labels
        data = np.load(path_images)[start:end]
        targets = np.array([int(label) for label in np.load(path_labels)[start:end]])

        # Create deterministic train and test sets (80/20)
        indices, n_train = deterministic_split(data=data, train_size=0.8)
        if train:
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]

        # Recover images and corresponding labels
        indices = np.sort(indices)
        self.data = data[indices]
        self.targets = targets[indices]
        self.n_classes = 100

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = F.to_pil_image(self.data[idx])
        label = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.data)} images."
