r"""
Utils functions to build transformer models.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import asdict
from typing import Any

import torch.nn as nn

from ..config import DEVICE
from ..utils import build_with_type_check

# ------------------------------------------------------------------------------
# Main function to build the model
# ------------------------------------------------------------------------------


def build_model(config: dict[str, Any], device: str = DEVICE, return_config: bool = False) -> nn.Module:
    r"""
    Initialize model based on the specified implementation given in the config file.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    device: str, default=DEVICE
        Device on which to move the model.
    return_config: bool, default=False
        Whether to return the model configuration.

    Returns
    -------
    model: nn.Module
        An instance of the specified model implementation.
    """

    # Argument parsing
    implementation = config.pop("implementation", "vit")

    match implementation.lower():
        case "gpt2":
            from vitef.models.gpt2 import GPT2, GPT2Config

            model_type = GPT2
            config_obj = build_with_type_check(GPT2Config, config)

        case "patchtst":
            from vitef.models.patchtst import PatchTST, PatchTSTConfig

            model_type = PatchTST
            config_obj = build_with_type_check(PatchTSTConfig, config)

        case "transformer":
            from vitef.models.transformer import Transformer, TransformerConfig

            model_type = Transformer
            config_obj = build_with_type_check(TransformerConfig, config)

        case "vit":
            from vitef.models.vit import ViT, ViTConfig

            model_type = ViT
            config_obj = build_with_type_check(ViTConfig, config)

        case _:
            raise ValueError(f"Implementation {implementation} not found.")

    # Initialize the model
    model = model_type(config_obj)

    # Move to device
    model = model.to(device=device)

    # Return model and config
    if return_config:
        return model, asdict(config_obj)

    return model
