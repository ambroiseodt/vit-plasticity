r"""
GPT2 model following [1]_.

Comments abbreviations:
    N: batch size
    L: sequence length
    E: embedding dimension
    H: number of heads
    D: downsampling factor in attention

References
----------
.. [1] A. Radford et al. Language Models are Unsupervised Multitask Learners. arxiv 2019

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

import tiktoken
import torch
import torch.nn as nn

from ..config import MODEL_DIR
from .transformer.architecture import Transformer, TransformerConfig

logger = logging.getLogger("vitef")


@dataclass
class GPT2Config:
    r"""
    GPT2 configuration file.

    Parameters
    ----------
    model_name: str
        Type of model to use. Options are "base", "medium", "large" and "xl".
    pretrained: bool
        If True, load the pretrained weights from save_dir if available or from HuggingFace.
    save_dir: str
        If provided, save the model to this directory.
    """

    model_name: str = "gpt2"
    pretrained: bool = False
    save_dir: str = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Save directory
        if self.save_dir is None:
            self.save_dir = MODEL_DIR / "gpt2"


class GPT2(nn.Module):
    r"""
    GPT2 implementation following [1]_.

    Parameters
    ----------
    config: configuration class with
        model_name: str
            Type of model to use. Options are "nano", "base", "medium", "large" and "xl".
        pretrained: bool
            If True, load the pretrained weights from save_dir if available or from HuggingFace.
        save_dir: str
            If provided, save the model to this directory.

    Notes
    -----
    Implementation inspired by https://github.com/huggingface/transformers.

    References
    ----------
    .. [1] A. Radford et al. Language Models are Unsupervised Multitask Learners. arxiv 2019
    """

    def __init__(self, gpt2_config: GPT2Config):
        super().__init__()

        # Set model name
        self.model_name = "gpt2-" + gpt2_config.model_name.lower() if gpt2_config.model_name != "base" else "gpt2"

        # Build Transformer configuration file
        config_args = {
            "base": dict(emb_dim=768, n_heads=12, n_layers=12),  # 124M params
            "medium": dict(emb_dim=1024, n_heads=16, n_layers=24),  # 350M params
            "large": dict(emb_dim=1280, n_heads=20, n_layers=36),  # 774M params
            "xl": dict(emb_dim=1600, n_heads=25, n_layers=48),  # 1558M params
        }[gpt2_config.model_name]

        config_args = config_args | dict(
            patch_type=None,
            vocab_size=50_257,
            emb_type="dict",
            pos_emb=True,
            freeze_pos=False,
            seq_len=1024,
            emb_dropout=0.0,
            attn_bias=True,
            attn_dropout=0.0,
            flash=False,
            causal=True,
            activation="gelu",
            ffn_bias=True,
            ffn_dropout=0.0,
            norm="layer",
            norm_bias=True,
            norm_eps=1e-5,
            pre_norm=True,
            cls_token=False,
            output_type="sequence_to_sequence",
            weight_tying=True,
            output_dropout=0.0,
        )

        config = TransformerConfig(**config_args)

        # Initialize the transformer
        self.model = Transformer(config)

        # Bind the forward method to the inner Transformer class
        self.forward = self.model.forward

        # Bind methods to the inner Transformer class
        self.get_decomposition = self.model.get_decomposition
        self.get_probes = self.model.get_probes

        # Load tokenizer
        logger.info(f"Loading {self.model_name} tokenizer from Tiktoken")
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Load pretrained weights
        if gpt2_config.pretrained:
            self.save_dir = gpt2_config.save_dir
            self.config = config

            # Load weights if available
            available_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
            if self.model_name in available_models:
                self._load_pretrained_weights()
                logger.info(f"Pretrained weights successfully loaded for {self.model_name}.")

            # Else, use the random initialization
            else:
                logger.info(f"Pretrained weights for {self.model_name} not found. Using random initialization.")

    def _load_pretrained_weights(self):
        r"""Load weights into the model from HuggingFace Transformers library."""

        # Try to load weights from save_path
        if self.save_dir is not None:
            assert isinstance(self.save_dir, PosixPath), "save_dir must be a PosixPath object"
            save_path = self.save_dir / f"{self.model_name}.pt"
            if os.path.exists(save_path):
                logger.info(f"Loading {self.model_name} model from {save_path}")
                self.model.load_state_dict(torch.load(save_path))
                return

        # Otherwise load from HuggingFace
        logger.info(f"Loading {self.model_name} model from HuggingFace Transformers library")
        self.n_layers = self.config.n_layers
        self.emb_dim = self.config.emb_dim
        self._load_from_huggingface()

        # Save weights
        self._save_weights()

    def _load_from_huggingface(self):
        r"""
        Load weights into the model from HuggingFace Transformers library.

        Parameters
        ----------
        model: str
            Name of the model. Options are "gpt2", "gpt2-medium", "gpt2-large", and "gpt2-xl".
        """
        from transformers import GPT2LMHeadModel

        gpt_state_dict = GPT2LMHeadModel.from_pretrained(self.model_name).state_dict()
        local_state_dict = self.model.state_dict()
        correspondence = {
            "embedding.token_emb.weight": "transformer.wte.weight",
            "embedding.pos_emb": "transformer.wpe.weight",
            "output.output_layer.output_norm.weight": "transformer.ln_f.weight",
            "output.output_layer.output_norm.bias": "transformer.ln_f.bias",
            "output.output_layer.output.weight": "lm_head.weight",
        }
        unsqueeze = ["transformer.wpe.weight"]
        transposed = []
        for layer in range(self.n_layers):
            correspondence = correspondence | {
                f"blocks.{layer}.attn_norm.weight": f"transformer.h.{layer}.ln_1.weight",
                f"blocks.{layer}.attn_norm.bias": f"transformer.h.{layer}.ln_1.bias",
                f"blocks.{layer}.attn.qkv_mat.weight": f"transformer.h.{layer}.attn.c_attn.weight",
                f"blocks.{layer}.attn.qkv_mat.bias": f"transformer.h.{layer}.attn.c_attn.bias",
                f"blocks.{layer}.attn.output.weight": f"transformer.h.{layer}.attn.c_proj.weight",
                f"blocks.{layer}.attn.output.bias": f"transformer.h.{layer}.attn.c_proj.bias",
                f"blocks.{layer}.ffn_norm.weight": f"transformer.h.{layer}.ln_2.weight",
                f"blocks.{layer}.ffn_norm.bias": f"transformer.h.{layer}.ln_2.bias",
                f"blocks.{layer}.ffn.fc1.weight": f"transformer.h.{layer}.mlp.c_fc.weight",
                f"blocks.{layer}.ffn.fc1.bias": f"transformer.h.{layer}.mlp.c_fc.bias",
                f"blocks.{layer}.ffn.fc2.weight": f"transformer.h.{layer}.mlp.c_proj.weight",
                f"blocks.{layer}.ffn.fc2.bias": f"transformer.h.{layer}.mlp.c_proj.bias",
            }
            transposed = transposed + [
                f"transformer.h.{layer}.attn.c_attn.weight",
                f"transformer.h.{layer}.attn.c_proj.weight",
                f"transformer.h.{layer}.mlp.c_fc.weight",
                f"transformer.h.{layer}.mlp.c_proj.weight",
            ]
        for k in correspondence:
            if correspondence[k] in transposed:
                local_state_dict[k] = gpt_state_dict[correspondence[k]].T
            elif correspondence[k] in unsqueeze:
                local_state_dict[k] = torch.unsqueeze(gpt_state_dict[correspondence[k]], dim=0)
            else:
                local_state_dict[k] = gpt_state_dict[correspondence[k]]
        self.model.load_state_dict(local_state_dict)

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
