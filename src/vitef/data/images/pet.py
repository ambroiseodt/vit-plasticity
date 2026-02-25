r"""
Oxford-IIIT Pet Dataset.

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
        self.samples = dataset._images
        print(Image.open(self.samples[0]).convert("RGB"))
        self.targets = dataset._labels
        self.n_classes = 37

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
