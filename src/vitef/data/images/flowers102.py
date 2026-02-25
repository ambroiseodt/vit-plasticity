r"""
Oxford Flowers-102 Dataset.

Notes
-----
To load the Flowers102 dataset, scipy is needed.


License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from typing import Any

import torchvision
from PIL import Image
from torch.utils.data import Dataset

from ...config import DATASET_DIR


@dataclass
class Flowers102DatasetConfig:
    r"""
    Oxford Flowers-102 configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train", "val", and "test".
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
        assert self.mode in ["train", "val", "test"], f"Invalid mode {self.mode}. Options are 'train', 'val' and 'test."
        if self.save_dir is None:
            self.save_dir = DATASET_DIR / "flowers102"


class Flowers102Dataset(Dataset):
    r"""
    Oxford 102 Flower dataset from [1]_.

    It consits of images of 102 flower categories with each class containing of between 40 and 258 images.
    The flowers were chosen to be flowers commonly occurring in the United Kingdom.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exit.
        mode: str
            Mode. Options are "train", "val", and "test".
        transform: nn.Module
            Transformation to apply to the images.

    Notes
    ----------
    Oxford 102 Flower was created by Maria-Elena Nilsback and Andrew Zisserman.
    Official website: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.

    References
    ----------
    .. [1] M-E. Nilsback and A. Zisserman. Automated Flower Classification
           over a Large Number of Classes. In ICVGIP 2008
    """

    def __init__(self, config: Flowers102DatasetConfig):
        super().__init__()
        split = config.mode
        dataset = torchvision.datasets.Flowers102(
            root=config.save_dir,
            split=split,
            download=True,
        )

        # Recover dataset
        self.samples = dataset._image_files
        self.targets = dataset._labels
        self.n_classes = 102

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        sample = Image.open(path).convert("RGB")
        label = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
