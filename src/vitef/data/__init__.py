r"""
Codebase to preprocess data, notably images, but other modalities can be added similarly.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .images import (
    Cifar10CDataset,
    Cifar10CDatasetConfig,
    Cifar10Dataset,
    Cifar10DatasetConfig,
    Cifar100CDataset,
    Cifar100CDatasetConfig,
    Cifar100Dataset,
    Cifar100DatasetConfig,
    DomainNetDataset,
    DomainNetDatasetConfig,
    Flowers102Dataset,
    Flowers102DatasetConfig,
    ImageNetCDataset,
    ImageNetCDatasetConfig,
    ImageNetDataset,
    ImageNetDatasetConfig,
    OxfordIIITPetDataset,
    OxfordIIITPetDatasetConfig,
    build_loader,
    build_train_val_loader,
    make_iterable,
)

__all__ = [
    "Cifar10DatasetConfig",
    "Cifar10Dataset",
    "Cifar10CDatasetConfig",
    "Cifar10CDataset",
    "Cifar100DatasetConfig",
    "Cifar100Dataset",
    "Cifar100CDatasetConfig",
    "Cifar100CDataset",
    "DomainNetDataset",
    "DomainNetDatasetConfig",
    "ImageNetDatasetConfig",
    "ImageNetDataset",
    "ImageNetCDataset",
    "ImageNetCDatasetConfig",
    "Flowers102DatasetConfig",
    "Flowers102Dataset",
    "OxfordIIITPetDatasetConfig",
    "OxfordIIITPetDataset",
    "build_loader",
    "build_train_val_loader",
    "make_iterable",
]
