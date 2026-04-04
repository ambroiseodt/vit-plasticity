r"""
Cifar100 Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from ...config import DATASET_DIR


@dataclass
class Cifar100ldDatasetConfig:
    r"""
    Cifar100 configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: str | None = None
    mode: str = "train"
    transform: Any | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        assert self.mode in ["train", "test"], f"Invalid mode {self.mode}. Options are 'train' and 'test."
        if self.save_dir is None:
            self.save_dir = DATASET_DIR / "cifar100"


class Cifar100ldDataset(Dataset):
    r"""
    Cifar100 dataset from [1]_.

    It consits of 60_000 32x32 color images in 100 classes, with 600 images per class.
    There are 50_000 training images and 10_000 test images.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exit.
        mode: str
            Mode. Options are "train" and "test".
        transform: nn.Module
            Transformation to apply to the images.

    Notes
    ----------
    CIFAR-100 was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
    Official website: https://www.cs.toronto.edu/~kriz/cifar.html.

    References
    ----------
    .. [1] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical Report, 2009
    """

    def __init__(self, config: Cifar100ldDatasetConfig):
        super().__init__()
        train = True if config.mode == "train" else False
        dataset = torchvision.datasets.CIFAR100(
            root=config.save_dir,
            train=train,
            download=True,
        )

        # Recover images and corresponding labels
        self.data = np.asarray(dataset.data)
        self.targets = np.asarray(dataset.targets)
        self.n_classes = 100

        # Deterministic low-data subsampling
        sample_size = 1000
        if train:
            samples_per_class = sample_size // self.n_classes
            indices = []
            st0 = np.random.get_state()
            np.random.seed(42)
            for c in range(self.n_classes):
                class_indices = np.where(self.targets == c)[0]
                class_indices = np.random.permutation(class_indices)
                indices.extend(class_indices[:samples_per_class])
            np.random.set_state(st0)
            self.data = self.data[indices]
            self.targets = self.targets[indices]

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
