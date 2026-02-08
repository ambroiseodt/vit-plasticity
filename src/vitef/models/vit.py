r"""
Vision Transformer model following [1]_.

Comments abbreviations:
    N: batch size
    L: sequence length
    E: embedding dimension
    H: number of heads
    D: downsampling factor in attention

References
----------
.. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
       for Image Recognition at Scale. ICLR 2021.

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

from vitef.config import MODEL_DIR

from .transformer.architecture import Transformer, TransformerConfig

logger = logging.getLogger("vitef")


@dataclass
class ViTConfig:
    r"""
    ViT configuration file.

    Parameters
    ----------
    model_name: str
        Type of model to use. Options are "base", "large", and "huge".
    pretrained: bool
        If True, load the pretrained weights from save_dir if available or from HuggingFace.
    in21k: bool
        If True, the ViT model is among the ones only pre-trained on ImageNet-21k but not finetuned on ImageNet-1k.
    save_dir: str
        If provided, save the model to this directory.
    patch_size: int
        Size of the patches to be extracted from the image.
    image_dim: tuple
        Image dimensions (channels, height, width).
    finetuning: bool
        If True, the model is prepared for finetuning with a new classification head.
    n_classes: int
        Number of classes for the classification head. Defaults to 1000 for ImageNet-1k.
    """

    model_name: str = "base"
    pretrained: bool = False
    in21k: bool = False
    save_dir: str = None
    patch_size: int = 16
    image_dim: tuple = (3, 224, 224)
    finetuning: bool = False
    n_classes: int = 1000

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Save directory
        if self.save_dir is None:
            self.save_dir = MODEL_DIR / "vit"


class ViT(nn.Module):
    r"""
    Vision Transformer implementation following [1]_.

    The current implementation allows for pre-training and finetuning of the model on image classification tasks.
    Inputs are batches of images represented as tensors of dimension (batch_size, n_channels, height, width).

    Parameters
    ----------
    config: configuration class with
        model_name: str
            Type of model to use. Options are "base", "large", and "huge".
        pretrained: bool
            If True, load the pretrained weights from save_dir if available or from HuggingFace.
        in21k: bool
            If True, the considered model was only pre-trained on ImageNet-21k without finetuning.
        save_dir: str
            If provided, save the model to this directory.
        patch_size: int
            Size of the patches to be extracted from the image.
        image_dim: tuple
            Image dimensions (n_channels, height, width).
        finetuning: bool
            If True, the model is prepared for finetuning with a new classification head and
            positional embedding are interpolaged if the image resolution is higher than the pretraining one.
        n_classes: int
            Number of classes for the classification head.

    Notes
    -----
    Implementation inspired by https://github.com/huggingface/transformers.

    References
    ----------
    .. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
            for Image Recognition at Scale. ICLR 2021.
    """

    def __init__(self, vit_config: ViTConfig):
        super().__init__()

        # Set model name
        self.model_name = f"vit-{vit_config.model_name.lower()}-patch{vit_config.patch_size}-{vit_config.image_dim[-1]}"
        if vit_config.in21k:
            self.model_name += "-in21k"

        # Build Transformer configuration file
        config_args = {
            "base": dict(emb_dim=768, n_heads=12, n_layers=12, ffn_dim=3072),  # 86M params
            "large": dict(emb_dim=1024, n_heads=16, n_layers=24, ffn_dim=4096),  # 307M params
            "huge": dict(emb_dim=1280, n_heads=16, n_layers=32, ffn_dim=5120),  # 632M params
        }[vit_config.model_name]

        config_args = config_args | dict(
            image_dim=vit_config.image_dim,
            patch_type="computer_vision",
            image_patch="hybrid",
            patch_size=vit_config.patch_size,
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
            norm="layer",
            norm_bias=True,
            norm_eps=1e-12,
            pre_norm=True,
            cls_token=True,
            pool="cls",
            output_type="classification",
            weight_tying=False,
            output_dropout=0.0,
            n_classes=1000 if not vit_config.in21k else 2,
        )

        config = TransformerConfig(**config_args)

        # Initialize the transformer
        self.model = Transformer(config)

        # Save config
        self.config = config

        # Bind the forward method to the inner Transformer class
        self.forward = self.model.forward

        # Bind methods to the inner Transformer class
        self.get_decomposition = self.model.get_decomposition
        self.get_probes = self.model.get_probes

        # Load pretrained weights
        if vit_config.pretrained:
            self.save_dir = vit_config.save_dir

            # Load weights if available
            available_models = [
                "vit-base-patch16-224",
                "vit-base-patch16-384",
                "vit-base-patch32-384",
                "vit-base-patch16-224-in21k",
                "vit-base-patch32-224-in21k",
                "vit-large-patch16-224",
                "vit-large-patch16-384",
                "vit-large-patch32-384",
                "vit-large-patch16-224-in21k",
                "vit-large-patch32-224-in21k",
                "vit-huge-patch14-224-in21k",
            ]
            if self.model_name in available_models:
                self.load_pretrained_weights()
                logger.info(f"Pretrained weights successfully loaded for {self.model_name}.")

            # Else, use the random initialization
            else:
                logger.info(f"Pretrained weights for {self.model_name} not found. Using random initialization.")

        # Prepare for finetuning
        if vit_config.finetuning:
            # Update the number of classes
            self.config.n_classes = vit_config.n_classes

            # Set finetuning mode
            self.set_finetuning_mode()
            logger.info(f"Initialize new classification head with {self.config.n_classes} classes for finetuning.")

    def load_pretrained_weights(self) -> None:
        r"""Load weights into the model."""

        # Try to load weights from save_path
        if self.save_dir is not None:
            self.save_dir = PosixPath(self.save_dir)
            save_path = self.save_dir / f"{self.model_name}.pt"
            if os.path.exists(save_path):
                logger.info(f"Loading {self.model_name} model from {save_path}")
                self.model.load_state_dict(torch.load(save_path))
                return

        # Otherwise load from HuggingFace
        logger.info(f"Loading {self.model_name} model from HuggingFace Transformers library.")
        self.n_layers = self.config.n_layers
        self.emb_dim = self.config.emb_dim
        self._load_from_huggingface()

        # Save weights
        self._save_weights()

    def set_finetuning_mode(self) -> None:
        r"""Initialize a new classification head with the desired number of classes for finetuning."""
        self.model.output.output_layer.output = nn.Linear(self.config.emb_dim, self.config.n_classes)

    def _load_from_huggingface(self) -> None:
        r"""
        Load weights into the model from the HuggingFace Transformers library.

        Parameters
        ----------
        model: str
            Name of the model. Options are:
                "vit-base-patch16-224",
                "vit-base-patch16-384",
                "vit-base-patch32-384",
                "vit-base-patch16-224-in21k",
                "vit-base-patch32-224-in21k",
                "vit-large-patch16-224",
                "vit-large-patch16-384",
                "vit-large-patch32-384",
                "vit-large-patch16-224-in21k",
                "vit-large-patch32-224-in21k",
                "vit-huge-patch14-224-in21k".
        """
        from transformers import ViTForImageClassification

        vit_state_dict = ViTForImageClassification.from_pretrained(f"google/{self.model_name}").state_dict()
        local_state_dict = self.model.state_dict()
        correspondence = {
            "embedding.cls_token": "vit.embeddings.cls_token",
            "embedding.patching.patching.0.weight": "vit.embeddings.patch_embeddings.projection.weight",
            "embedding.patching.patching.0.bias": "vit.embeddings.patch_embeddings.projection.bias",
            "embedding.pos_emb": "vit.embeddings.position_embeddings",
            "output.output_layer.output_norm.weight": "vit.layernorm.weight",
            "output.output_layer.output_norm.bias": "vit.layernorm.bias",
            "output.output_layer.output.weight": "classifier.weight",
            "output.output_layer.output.bias": "classifier.bias",
        }
        special = {}
        for layer in range(self.n_layers):
            correspondence = correspondence | {
                f"blocks.{layer}.attn_norm.weight": f"vit.encoder.layer.{layer}.layernorm_before.weight",
                f"blocks.{layer}.attn_norm.bias": f"vit.encoder.layer.{layer}.layernorm_before.bias",
                f"blocks.{layer}.attn.output.weight": f"vit.encoder.layer.{layer}.attention.output.dense.weight",
                f"blocks.{layer}.attn.output.bias": f"vit.encoder.layer.{layer}.attention.output.dense.bias",
                f"blocks.{layer}.ffn_norm.weight": f"vit.encoder.layer.{layer}.layernorm_after.weight",
                f"blocks.{layer}.ffn_norm.bias": f"vit.encoder.layer.{layer}.layernorm_after.bias",
                f"blocks.{layer}.ffn.fc1.weight": f"vit.encoder.layer.{layer}.intermediate.dense.weight",
                f"blocks.{layer}.ffn.fc1.bias": f"vit.encoder.layer.{layer}.intermediate.dense.bias",
                f"blocks.{layer}.ffn.fc2.weight": f"vit.encoder.layer.{layer}.output.dense.weight",
                f"blocks.{layer}.ffn.fc2.bias": f"vit.encoder.layer.{layer}.output.dense.bias",
            }
            special = special | {
                f"blocks.{layer}.attn.qkv_mat.weight": [
                    f"vit.encoder.layer.{layer}.attention.attention.query.weight",
                    f"vit.encoder.layer.{layer}.attention.attention.key.weight",
                    f"vit.encoder.layer.{layer}.attention.attention.value.weight",
                ],
                f"blocks.{layer}.attn.qkv_mat.bias": [
                    f"vit.encoder.layer.{layer}.attention.attention.query.bias",
                    f"vit.encoder.layer.{layer}.attention.attention.key.bias",
                    f"vit.encoder.layer.{layer}.attention.attention.value.bias",
                ],
            }
        for k in correspondence:
            local_state_dict[k] = vit_state_dict[correspondence[k]]
        for k in special:
            local_state_dict[k] = torch.cat(tuple([vit_state_dict[key] for key in special[k]]))
        self.model.load_state_dict(local_state_dict)

    def _save_weights(self) -> None:
        r"""Save the model's weights."""
        if self.save_dir is not None:
            assert isinstance(self.save_dir, PosixPath), "save_dir must be a PosixPath object."
            self.save_dir.mkdir(exist_ok=True, parents=True)
            save_path = self.save_dir / f"{self.model_name}.pt"
            logger.info(f"Saving {self.model_name} model to {save_path}")
            torch.save(self.model.state_dict(), save_path)

    def __repr__(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model with {n_params} trainable parameters."
