r"""
DomainNet Dataset.

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
from ...utils import deterministic_split


@dataclass
class DomainNetDatasetConfig:
    r"""
    DomainNet configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data.
    domain: str, default="clipart"
        Type of corruption applied on the data. Options are:
            "clipart",
            "infograph",
            "painting",
            "quickdraw",
            "real",
            "sketch".
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: str = None
    domain: str = "clipart"
    mode: str = "train"
    transform: Any = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Check if domain is valid
        assert self.domain in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"], (
            f"Invalid corruption type {self.domain}. See documentation for options."
        )

        # Valid mode
        assert self.mode in ["train", "test"], f"Invalid mode {self.mode}. Options are 'train' and 'test."

        # Set default values
        if self.save_dir is None:
            self.save_dir = DATASET_DIR / "domainnet"


class DomainNetDataset(Dataset):
    r"""
    DomainNet dataset from [1]_.

    It consits of images of 345 categories of objects from 6 different domains.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data.
        domain: str
            Type of images.
        mode: str
            Mode. Options are "train" and "test".
        transform: nn.Module
            Transformation to apply to the images.

    Notes
    ----------
    Official website: http://csr.bu.edu/ftp/visda/2019/multi-source

    Download
    --------
    To download the data, run the following command (replacing domain by infograph, quickdraw, real, or sketch):
    ```bash
    $ wget http://csr.bu.edu/ftp/visda/2019/multi-source/domain.zip
    ```
    For clipart and painting domains, run the following command;
    ```bash
    $ wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/domain.zip
    ```
    Move the downloaded files to the desired location, e.g, datasets/domainnet, and extract them with
    ```bash
    $ unzip domain.zip
    ```

    References
    ----------
    .. [1] X. Peng et al. Moment matching for multi-source domain adaptation. In ICCV 2019
    """

    def __init__(self, config: DomainNetDatasetConfig):
        super().__init__()
        train = True if config.mode == "train" else False

        # Recover dataset
        dataset = torchvision.datasets.ImageFolder(root=config.save_dir / config.domain)
        samples = np.asarray(dataset.imgs)

        # Create deterministic train and test sets (80/20)
        indices, n_train = deterministic_split(data=samples, train_size=0.8)
        if train:
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]

        # Recover images and corresponding labels
        indices = np.sort(indices)
        self.samples = samples[indices]
        self.n_classes = 345

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        sample = Image.open(path).convert("RGB")
        label = int(label)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
