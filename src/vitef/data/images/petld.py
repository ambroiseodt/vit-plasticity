r"""
Oxford-IIIT Pet Dataset.

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
from PIL import Image
from torch.utils.data import Dataset

from ...config import DATASET_DIR


@dataclass
class OxfordIIITPetDatasetConfig:
    r"""
    Oxford-IIIT Pet configuration file.

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
            self.save_dir = DATASET_DIR / "pet"


class OxfordIIITPetDataset(Dataset):
    r"""
    Oxford-IIIT Pet dataset from [1]_.

    It consists of 37 category of pets with around 200 images for each class.
    The images have a large variations in scale, pose and lighting.

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
    Oxford-IIIT Pet was created by Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, C. V. Jawahar
    Official website: https://www.robots.ox.ac.uk/~vgg/data/pets/.

    References
    ----------
    .. [1] O. M. Parkhi et al. Cats and Dogs. In CVPR 2012
    """

    def __init__(self, config: OxfordIIITPetDatasetConfig):
        super().__init__()
        split = "trainval" if config.mode == "train" else "test"
        dataset = torchvision.datasets.OxfordIIITPet(
            root=config.save_dir,
            split=split,
            download=True,
        )

        # Recover dataset
        self.samples = np.asarray(dataset._images)
        self.targets = np.asarray(dataset._labels)
        self.n_classes = 37

        # Deterministic low-data subsampling
        sample_size = 1000
        if config.mode == "train":
            samples_per_class = sample_size // self.n_classes
            indices = []

            st0 = np.random.get_state()
            np.random.seed(42)
            for c in range(self.n_classes):
                class_indices = np.where(self.targets == c)[0]
                if len(class_indices) > 0:
                    class_indices = np.random.permutation(class_indices)
                    indices.extend(class_indices[:samples_per_class])
            np.random.set_state(st0)

            # Sort indices to keep file access somewhat organized
            indices = np.sort(indices)
            self.samples = self.samples[indices]
            self.targets = self.targets[indices]

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
