r"""
Utils functions to build dataloaders.

Notes
-----
Loading data can be slow if the num_workers variable is not properly set. Small values are not suitable for large
datasets and large values increase the CPU usage (see https://lightning.ai/docs/pytorch/stable/advanced/speed.html).
In the default DataLoader mode with num_workers=0 where the main process loads the data, we launch runs with
OMP_NUM_THREADS=1 to avoid unnecessary parallelization, following https://github.com/facebookresearch/lingua.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from itertools import repeat
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from vitef.utils import build_with_type_check

logger = logging.getLogger("vitef")

# Datasets with predefined validation sets
PREDEFINED_VAL_DATASETS = ["flowers102"]

# ------------------------------------------------------------------------------
# Utils functions and classes to build dataloaders
# ------------------------------------------------------------------------------


class DatasetFromSubset(DataLoader):
    r"""Build a Dataset object from a Subset object."""

    def __init__(self, subset: Dataset, transform: Compose = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def make_iterable(dataloader: DataLoader) -> None:
    r"""Convert a dataloader into an iterable dataloader."""
    for loader in repeat(dataloader):
        yield from loader


# ------------------------------------------------------------------------------
# Main function to build a dataloader
# ------------------------------------------------------------------------------


def build_loader(
    config: dict[str, Any],
    drop_last: bool = True,
    force_shuffle: bool = False,
    return_n_classes: bool = False,
) -> DataLoader:
    r"""
    Initialize dataloader based on the specified implementation given in the config file.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    drop_last: bool, default=True
        Whether to ignore the last batch if the dataset size is not divisible by the batch size.
        For datasets with few samples, if a large batch size is used, drop_last set to true may
        result in a significant portion of the dataset being discarded.
    force_shuffle: bool, default=False
        Whether to force shuffling the data, independently of the mode.
    return_n_classes: bool, default=False
        Whether to return the number of classes in the dataset.

    Returns
    -------
    loader: Dataloader
        An instance of the specified dataloader implementation.
    """

    # Argument parsing
    batch_size = config.pop("batch_size", 128)
    size = config.pop("size", 224)
    mode = config["mode"]

    # Set data transform
    config = config | dict(transform=build_transform(size=size, mode=mode))

    # Create dataset
    dataset = build_dataset(config)

    # Create dataloader
    shuffle = force_shuffle or (mode == "train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    # Recover number of classes
    if return_n_classes:
        return loader, dataset.n_classes

    return loader


# ------------------------------------------------------------------------------
# Main functions to build training and validation dataloaders
# ------------------------------------------------------------------------------


def build_train_val_loader(
    config: dict[str, Any],
    train_size: float = 0.8,
    return_n_classes: bool = False,
) -> tuple[DataLoader, DataLoader]:
    r"""
    Initialize training and validation dataloaders based on the specified implementation given in the config file.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    train_size: float, default=0.8
        Proportion of the dataset to include in the training set.
        If a pre-defined validation set exists, it will be used directly without splitting the training set.
    return_n_classes: bool, default=False
        Whether to return the number of classes in the dataset.

    Returns
    -------
    tuple of DataLoaders
        train_loader: Dataloader
            Training dataloader.
        val_loader: Dataloader
            Validation dataloader.
    """

    # Argument parsing
    batch_size = config.pop("batch_size", 128)
    val_batch_size = config.pop("val_batch_size", 128)
    size = config.pop("size", 224)

    # Check if validation set is pre-defined
    if config["dataset_name"] in PREDEFINED_VAL_DATASETS:
        logger.info("Validation set is pre-defined and used directly without splitting the training set.")

        # Training
        mode = "train"
        train_config = config | dict(mode=mode, transform=build_transform(size=size, mode=mode))
        train_set = build_dataset(train_config)
        n_classes = train_set.n_classes

        # Validation
        mode = "val"
        val_config = config | dict(mode=mode, transform=build_transform(size=size, mode=mode))
        val_set = build_dataset(val_config)

    # Otherwise, split training set into training and validation sets
    else:
        logger.info("Validation set is not pre-defined, thus training set is split into training and validation sets.")
        config = config | dict(mode="train", transform=None)
        train_set = build_dataset(config)
        n_classes = train_set.n_classes

        # Split training and validation sets
        n_train = int(train_size * len(train_set))
        train_set, val_set = random_split(train_set, [n_train, len(train_set) - n_train])

        # Apply training transform to the training set
        train_transform = build_transform(size=size, mode="train")
        train_set = DatasetFromSubset(subset=train_set, transform=train_transform)

        # Apply validation transform to the validation set
        val_transform = build_transform(size=size, mode="val")
        val_set = DatasetFromSubset(subset=val_set, transform=val_transform)

    # Create training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=False)

    # Recover number of classes
    if return_n_classes:
        return train_loader, val_loader, n_classes

    return train_loader, val_loader


# ------------------------------------------------------------------------------
# Main function to build a dataset
# ------------------------------------------------------------------------------


def build_dataset(config: dict[str, Any]) -> Dataset:
    r"""
    Initialize the dataset based on the specified implementation given in the config file.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    dataset: Dataset
        An instance of the specified dataset implementation.
    """
    # argument parsing
    dataset_name = config.pop("dataset_name", "cifar10")
    match dataset_name.lower():
        case "cifar10":
            from vitef.data.images.cifar10 import Cifar10Dataset, Cifar10DatasetConfig

            dataset_type = Cifar10Dataset
            config_obj = build_with_type_check(Cifar10DatasetConfig, config)

        case x if "cifar10_c" in x:
            from vitef.data.images.cifar10_c import Cifar10CDataset, Cifar10CDatasetConfig

            dataset_type = Cifar10CDataset

            # Dataset name format should be cifar10_c-corruption-brightness-severity-1
            substring = dataset_name.split("cifar10_c", 1)[-1]
            substring = substring.split("-corruption-", 1)[-1]
            config["corruption_type"], config["corruption_severity"] = substring.split("-severity-", 1)
            config_obj = build_with_type_check(Cifar10CDatasetConfig, config)

        case "cifar100":
            from vitef.data.images.cifar100 import Cifar100Dataset, Cifar100DatasetConfig

            dataset_type = Cifar100Dataset
            config_obj = build_with_type_check(Cifar100DatasetConfig, config)

        case x if "cifar100_c" in x:
            from vitef.data.images.cifar100_c import Cifar100CDataset, Cifar100CDatasetConfig

            dataset_type = Cifar100CDataset

            # Dataset name format should be cifar100_c-corruption-brightness-severity-1
            substring = dataset_name.split("cifar100_c", 1)[-1]
            substring = substring.split("-corruption-", 1)[-1]
            config["corruption_type"], config["corruption_severity"] = substring.split("-severity-", 1)
            config_obj = build_with_type_check(Cifar100CDatasetConfig, config)

        case x if "domainnet" in x:
            from vitef.data.images.domainnet import DomainNetDataset, DomainNetDatasetConfig

            dataset_type = DomainNetDataset

            # Dataset name format should be domainnet-clipart
            config["domain"] = dataset_name.split("domainnet-", 1)[-1]
            config_obj = build_with_type_check(DomainNetDatasetConfig, config)

        case "flowers102":
            from vitef.data.images.flowers102 import Flowers102Dataset, Flowers102DatasetConfig

            dataset_type = Flowers102Dataset
            config_obj = build_with_type_check(Flowers102DatasetConfig, config)

        case "imagenet":
            from vitef.data.images.imagenet import ImageNetDataset, ImageNetDatasetConfig

            dataset_type = ImageNetDataset
            config_obj = build_with_type_check(ImageNetDatasetConfig, config)

        case x if "imagenet_c" in x:
            from vitef.data.images.imagenet_c import ImageNetCDataset, ImageNetCDatasetConfig

            dataset_type = ImageNetCDataset

            # Dataset name format should be imagenet_c-corruption-brightness-severity-1
            substring = dataset_name.split("imagenet_c", 1)[-1]
            substring = substring.split("-corruption-", 1)[-1]
            config["corruption_type"], config["corruption_severity"] = substring.split("-severity-", 1)
            config_obj = build_with_type_check(ImageNetCDatasetConfig, config)

        case "pet":
            from vitef.data.images.pet import OxfordIIITPetDataset, OxfordIIITPetDatasetConfig

            dataset_type = OxfordIIITPetDataset
            config_obj = build_with_type_check(OxfordIIITPetDatasetConfig, config)

        case _:
            raise ValueError(f"Dataset name {dataset_name} not found.")

    # Build dataset
    dataset = dataset_type(config_obj)

    return dataset


# ------------------------------------------------------------------------------
# Main function to build transforms
# ------------------------------------------------------------------------------


def build_transform(size: int, mode: str) -> Compose:
    r"""
    Image preprocessing following [1]_.

    Following the literature, the image are normalized using ImageNet mean and standard deviation.

    Parameters
    ----------
    size: int
        Size of the image after resizing.
    mode: str
        Mode. Options are "train", "val" and "test".

    Returns
    -------
    transform: Compose
        List of Transforms objects.

    References
    ----------
    .. [1] A. Kolesnikov et al. Big transfer (bit): General visual representation learning. In ECCV 2020
    """
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    match mode.lower():
        case "train":
            transform = Compose(
                [
                    RandomResizedCrop(size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    normalize,
                ]
            )
        case "val":
            transform = Compose(
                [
                    Resize(size),
                    CenterCrop(size),
                    ToTensor(),
                    normalize,
                ]
            )
        case "test":
            transform = Compose(
                [
                    Resize(size),
                    CenterCrop(size),
                    ToTensor(),
                    normalize,
                ]
            )
        case _:
            raise ValueError(f"Mode {mode} not found. Options are 'train', 'val' and 'test'.")

    return transform
