r"""
ImageNet Dataset.

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
class ImageNetDatasetConfig:
    r"""
    ImageNet configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "val".
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
        assert self.mode in ["train", "val"], f"Invalid mode {self.mode}. Options are 'train' and 'val'."
        if self.save_dir is None:
            self.save_dir = DATASET_DIR / "imagenet"


class ImageNetDataset(Dataset):
    r"""
    ImageNet dataset from [1]_.

    It consits of a subset of 10 easily classified classes from ImageNet.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exit.
        mode: str
            Mode. Options are "train" and "val".
        transform: nn.Module
            Transformation to apply to the images.

    Notes
    ----------
    Official website: https://www.image-net.org.

    Download
    --------
    To download the training or validation data, run the following command from the root of the repository
    (replacing subset by train or val):
    ```bash
    $ wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_subset.tar --no-check-certificate
    ```
    To extract the images, run the following command from the root of the repository:
    ```bash
    $ bash apps/vit/scripts/extract_imagenet.sh
    ```

    References
    ----------
    .. [1] J. Deng et al. Imagenet: A large-scale hierarchical image database. In CVPR 2009
    """

    def __init__(self, config: ImageNetDatasetConfig):
        super().__init__()
        split = "train" if config.mode == "train" else "val"
        dataset = torchvision.datasets.ImageFolder(root=config.save_dir / split)

        # Recover dataset
        self.samples = dataset.imgs
        self.n_classes = 1000

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
