r"""
Utils functions.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import copy
import json
import logging
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import PosixPath
from typing import Any, Literal, Protocol, TypeVar, Union, get_args, get_origin, runtime_checkable

import numpy as np
import torch

T = TypeVar("T")

logger = logging.getLogger("vitef")


# ------------------------------------------------------------------------------
# Type hint for dataclass objects
# ------------------------------------------------------------------------------


@runtime_checkable
@dataclass
class DataclassProtocol(Protocol):
    pass


# ------------------------------------------------------------------------------
# Configuration type
# ------------------------------------------------------------------------------


def build_with_type_check(object_type: type[T], data: dict[str, Any], inplace: bool = True) -> Any:
    r"""
    Recursively initializes a typed object from a nested dictionary.
    """
    if not inplace:
        data = copy.deepcopy(data)

    # Trivial cases
    if data is None or object_type is Any:
        return data
    args = get_args(object_type)

    # Dataclasses
    if is_dataclass(object_type):
        field_values = {}
        for data_field in fields(object_type):
            if not data_field.init:
                continue
            fname, ftype = data_field.name, data_field.type
            if fname in data:
                value = data.pop(fname)
                field_values[fname] = build_with_type_check(ftype, value)
            else:
                logger.debug(f"Field '{fname}' not found in {object_type}.")
        for fname in data:
            logger.warning(f"Field '{fname}' ignored when initializing {object_type}.")
        return object_type(**field_values)

    # List
    elif get_origin(object_type) is list and len(args) == 1:
        return [build_with_type_check(args[0], item) for item in data]

    # Dict
    elif get_origin(object_type) is dict and len(args) == 2:
        return {build_with_type_check(args[0], k): build_with_type_check(args[1], v) for k, v in data.items()}

    # Union
    elif get_origin(object_type) is Union:
        for arg in args:
            try:
                return build_with_type_check(arg, data)
            except (TypeError, ValueError):
                continue

    # Literal
    elif get_origin(object_type) is Literal:
        if data not in args:
            raise ValueError(f"Value '{data}' is not a valid literal for {object_type}.")
        return data

    # Primitive types
    try:
        return object_type(data)
    except (TypeError, ValueError):
        logger.warning(f"Initializing {object_type}:{data} without type checking.")
        return data


# ------------------------------------------------------------------------------
# JSONL Loading Utilities
# ------------------------------------------------------------------------------


def get_jsonl_keys(path: str, readall: bool = True) -> list[str]:
    r"""
    Get keys from a jsonl file.

    Parameters
    ----------
    path: str
        Path to the jsonl file.
    readall: bool, defaul=True
        Whether to read all lines of the file or the first one only.

    Returns
    -------
    keys: list
        List of keys in the jsonl file.
    """
    keys = set()
    with open(os.path.expandvars(path)) as f:
        for lineno, line in enumerate(f, start=1):
            try:
                keys |= json.loads(line).keys()
            except json.JSONDecodeError as e:
                logger.warning(f"Error reading line {lineno}: {e}")
            if not readall:
                break
    return list(keys)


def load_jsonl_to_numpy(path: str, keys: list[str] = None) -> dict[str, np.ndarray]:
    r"""
    Convert a jsonl file to a dictionnary of numpy array.

    Parameters
    ----------
    path: str
        Path to the jsonl file.
    keys: list
        List of keys in the jsonl file.

    Returns
    -------
    result_dict:dict
        Dictionnary of numpy arrays containing the data from the jsonl file.
    """
    if keys is None:
        keys = get_jsonl_keys(path, readall=True)

    data: dict[str, list] = {key: [] for key in keys}
    with open(os.path.expandvars(path)) as f:
        # read jsonl as a csv with missing values
        for lineno, line in enumerate(f, start=1):
            try:
                values: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Error reading line {lineno}: {e}")
            for key in keys:
                data[key].append(values.get(key, None))
    result_dict = {k: np.array(v) for k, v in data.items()}
    return result_dict


# ------------------------------------------------------------------------------
# Object type
# ------------------------------------------------------------------------------


def get_valid_tensor(x: Any) -> torch.Tensor:
    r"""Convert to tensor with batch dimension if necessary."""
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x


def get_numpy(x: torch.Tensor) -> np.array:
    r"""Detach tensor from graph to work on cpu and convert to numpy."""
    x = x.detach().cpu().numpy()
    if not x.ndim:
        x = np.expand_dims(x, axis=0)
    return x


def move_to_cpu(x: torch.Tensor) -> torch.Tensor:
    r"""Detach tensor from graph to work on cpu."""
    return x.detach().cpu()


def json_serializable(object_dict: dict) -> dict:
    r"""Convert dictionnary values in JSON serializable objects."""
    for key, value in object_dict.items():
        invalid_type = type(value) in [PosixPath, torch.device]
        object_dict[key] = str(value) if invalid_type else value
    return object_dict


# ------------------------------------------------------------------------------
#   Update object
# ------------------------------------------------------------------------------


def update_dict(value: np.array, dict_object: dict, key: Any) -> None:
    r"""Update a dictionary with a new value."""
    if key in dict_object.keys():
        dict_object[key] = np.concatenate((dict_object[key], value), axis=0)
    else:
        dict_object[key] = value


# ------------------------------------------------------------------------------
#   Create deterministic subsets and split of data
# ------------------------------------------------------------------------------


def deterministic_split(data: np.array, train_size: float = 0.8) -> tuple:
    r"""Create a deterministic split of the original data."""
    n_samples = len(data)
    n_train = int(train_size * n_samples)
    st0 = np.random.get_state()
    np.random.seed(42)
    indices = np.random.permutation(range(n_samples))
    np.random.set_state(st0)

    return indices, n_train
