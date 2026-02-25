r"""
Cifar10 Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from typing import Any

import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from ...config import DATASET_DIR


@dataclass
class Cifar10DatasetConfig:
    r"""
    Cifar10 configuration file.

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
            self.save_dir = DATASET_DIR / "cifar10"


class Cifar10Dataset(Dataset):
    r"""
    Cifar10 dataset from [1]_.

    It consits of 60_000 32x32 color images in 10 classes, with 6_000 images per class.
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
    CIFAR-10 was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
    Official website: https://www.cs.toronto.edu/~kriz/cifar.html.

    References
    ----------
    .. [1] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical Report, 2009
    """

    def __init__(self, config: Cifar10DatasetConfig):
        super().__init__()
        train = True if config.mode == "train" else False
        dataset = torchvision.datasets.CIFAR10(
            root=config.save_dir,
            train=train,
            download=True,
        )

        # Recover images and corresponding labels
        self.data = dataset.data
        self.targets = dataset.targets
        self.n_classes = 10

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
