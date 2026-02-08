r"""
PatchTST model following [1]_.

Comments abbreviations:
    N: batch size
    L: sequence length
    E: embedding dimension
    H: number of heads
    D: downsampling factor in attention

References
----------
.. [1] Y. Nie et al. A Time Series is Worth 64 Words: Long-Term Forecasting
        With Transformers. In ICLR 2023.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import os
from dataclasses import dataclass
from pathlib import PosixPath

import torch
import torch.nn as nn

from ..config import MODEL_DIR
from .transformer.architecture import Transformer, TransformerConfig

logger = logging.getLogger("vitef")


@dataclass
class PatchTSTConfig:
    r"""
    PatchTST configuration file.

    Parameters
    ----------
    model_name: str
        Type of model to use.
    pretrained: bool
        If True, load the pretrained weights from save_dir if available or from HuggingFace.
    save_dir: str
        If provided, save the model to this directory.
    patch_size: int
        Size of the patches to be extracted from the time series.
    stride: int
        Stride.
    length: int
        Length of the time series.
    forecasting_horizon: int
        Forecasting horizon.
    """

    model_name: str = "base"
    pretrained: bool = False
    save_dir: str = None
    patch_size: int = 16
    stride: int = 8
    length: int = 512
    forecasting_horizon: int = 96

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Save directory
        if self.save_dir is None:
            self.save_dir = MODEL_DIR / "patchtst"


class PatchTST(nn.Module):
    r"""
    PatchTST implementation following [1]_.

    Inputs are batches of time series represented as tensors of dimension (batch_size, seq_length).

    Parameters
    ----------
    config: configuration class with
        model_name: str
            Type of model to use.
        pretrained: bool
            If True, load the pretrained weights from save_dir if available.
        save_dir: str
            If provided, save the model to this directory.
        patch_size: int
            Size of the patches to be extracted from the time series.
        stride: int
            Stride.
        length: int
            Length of the time series.
        forecasting_horizon: int
            Forecasting horizon.

    Notes
    -----
    Implementation inspired by https://github.com/yuqinie98/PatchTST.

    References
    ----------
    .. [1] Y. Nie et al. A Time Series is Worth 64 Words: Long-Term Forecasting
        With Transformers. In ICLR 2023.
    """

    def __init__(self, patchtst_config: PatchTSTConfig):
        super().__init__()

        # Set model name
        self.model_name = (
            f"patchtst-{patchtst_config.model_name.lower()}-patch{patchtst_config.patch_size}-{patchtst_config.length}"
        )

        # Build Transformer configuration file
        config_args = {
            "base": dict(emb_dim=128, n_heads=16, n_layers=3, ffn_dim=256),  # 1M params
        }[patchtst_config.model_name]

        config_args = config_args | dict(
            length=patchtst_config.length,
            patch_type="time_series",
            patch_size=patchtst_config.patch_size,
            stride=patchtst_config.stride,
            emb_type="linear",
            pos_emb=True,
            freeze_pos=False,
            emb_dropout=0.0,
            attn_bias=True,
            attn_dropout=0.0,
            flash=False,
            causal=False,
            activation="gelu",
            ffn_bias=True,
            ffn_dropout=0.0,
            norm="batch",
            norm_bias=True,
            norm_eps=1e-5,
            pre_norm=False,
            cls_token=False,
            output_type="forecasting",
            weight_tying=False,
            output_dropout=0.0,
            forecasting_horizon=patchtst_config.forecasting_horizon,
        )

        config = TransformerConfig(**config_args)

        # Initialize the transformer
        self.model = Transformer(config)

        # Bind the forward method to the inner Transformer class
        self.forward = self.model.forward

        # Bind methods to the inner Transformer class
        self.get_decomposition = self.model.get_decomposition
        self.get_probes = self.model.get_probes

        # Load pretrained weights
        if patchtst_config.pretrained:
            self.save_dir = patchtst_config.save_dir
            self.config = config
            if self.save_dir is not None:
                self.save_dir = PosixPath(self.save_dir)
                save_path = self.save_dir / f"{self.model_name}.pt"

                # Load weights if available
                if os.path.exists(save_path):
                    logger.info(f"Loading {self.model_name} model from {save_path}.")
                    self.model.load_state_dict(torch.load(save_path))

                # Else, use the random initialization
                else:
                    logger.info(f"Pretrained weights for {self.model_name} not found. Using random initialization.")

    def _save_weights(self):
        r"""Save the model's weights."""
        if self.save_dir is not None:
            assert isinstance(self.save_dir, PosixPath), "save_dir must be a PosixPath object."
            self.save_dir.mkdir(exist_ok=True, parents=True)
            save_path = self.save_dir / f"{self.model_name}.pt"
            logger.info(f"Saving {self.model_name} model to {save_path}")
            torch.save(self.model.state_dict(), save_path)

    def __repr__(self):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model with {n_params} trainable parameters."
