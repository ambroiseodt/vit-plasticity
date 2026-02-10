r"""
Transformer architecture with decomposition methods to recover hidden representations.

Comments abbreviations:
    N: batch size
    L: sequence length
    E: embedding dimension
    h: number of heads
    D: downsampling factor in attention

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ...utils import move_to_cpu
from .utils import (
    BatchNorm,
    ClassificationLayer,
    ForecastingLayer,
    LayerNorm,
    PatchImages,
    PatchTimeSeries,
    RMSNorm,
    Seq2SeqLayer,
)

logger = logging.getLogger("vitef")


# -------------------------------------------------------------------------------
# Transformer config
# -------------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    r"""Transformer configuration."""

    # Data parameters
    image_dim: tuple = (3, 224, 224)
    length: int = 512

    # Patching parameters
    patch_type: str = None
    image_patch: str = "hybrid"
    patch_size: int = 16
    stride: int = 8

    # Embedding parameters
    vocab_size: int = -1
    emb_type: str = "dict"
    emb_dim: int = -1
    pos_emb: bool = True
    freeze_pos: bool = False
    seq_len: int = -1
    emb_dropout: float = None

    # Attention parameters
    n_heads: int = -1
    attn_bias: bool = False
    attn_dropout: float = None
    flash: bool = False
    causal: bool = False

    # Feed-forward parameters
    activation: float = "gelu"
    ffn_dim: bool = None
    ffn_bias: bool = False
    ffn_dropout: float = None

    # Transformer block parameter
    norm: str = "layer"
    norm_bias: bool = False
    norm_eps: float = 1e-5
    pre_norm: bool = True

    # Transformer parameters
    n_layers: int = -1
    dropout: float = 0.0

    # Task-specific parameters
    cls_token: bool = False
    output_type: str = "sequence_to_sequence"
    weight_tying: bool = True
    output_dropout: float = None
    n_classes: int = -1
    forecasting_horizon: int = -1

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Hidden feed-forward dimension
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.emb_dim

        # Flash attention in PyTorch
        if self.flash is None:
            self.flash = torch.__version__ >= "2"

        # Single dropout parameter
        if self.emb_dropout is None:
            self.emb_dropout = self.dropout
        if self.attn_dropout is None:
            self.attn_dropout = self.dropout
        if self.ffn_dropout is None:
            self.ffn_dropout = self.dropout
        if self.output_dropout is None:
            self.output_dropout = self.dropout


# -------------------------------------------------------------------------------
# Attention layers
# -------------------------------------------------------------------------------


class SelfAttention(nn.Module):
    r"""
    Multihead self-attention layer following [1]_.

    This implementation supports both vanilla and causal (masked attention).

    Parameters
    ----------
    config: configuration class with
        emb_dim: int
            Embedding dimension of the tokens.
       n_heads: int
            Number of attention heads (should divide emb_dim).
        attn_bias: bool
            Whether to use bias in attention.
        attn_dropout: float
            Dropout probability.
        flash: bool
            Whether to use flash attention. Note that the speed benefits
            are more important for large models with long sequence lengths.
            See also `Flash Attention <https://arxiv.org/abs/2205.14135>`_.
        causal: bool
            Whether to use causal attention mask.

    See also
    --------
    AttentionInitialization

    References
    ----------
    .. [1] A. Vaswani et al. Attention Is All You Need. In NIPS 2017.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        assert config.emb_dim % config.n_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.h = config.n_heads

        # Attention initialization
        self.qkv_mat = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.attn_bias)
        self.output = nn.Linear(config.emb_dim, config.emb_dim, bias=config.attn_bias)

        # Flash attention implementation and attention mask
        self.flash = config.flash

        # Causal attention if needed
        self.causal = config.causal
        if self.causal:
            L = config.seq_len
            mask = torch.ones(L, L)
            mask = torch.tril(mask, diagonal=0)
            self.register_buffer("mask", mask.view(1, 1, L, L) == 0)

        # Drop-out regularization
        self.dropout = config.attn_dropout

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        r"""Multihead self-attention (MHA).

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.
        verbose: bool, default=False
            Whether to return the attention matrix.

        Returns
        -------
        z: torch.Tensor of dimension (N, L, E)
            Batch of MHA embeddings.
        """
        # Query, key, value: (N, L, E) @ (E, 3 * E) -> (N, L, 3 * E) -> (N, L, E) * 3
        q, k, v = self.qkv_mat(x).chunk(3, dim=-1)

        # Recover number of heads and dimension of each head layer
        N, L, E = q.size()
        h, dim = self.h, E // self.h

        # Reformating: (N, L, E) -> (N, h, L, E / h)
        q, k, v = map(lambda t: t.view(N, -1, h, dim).transpose(1, 2), (q, k, v))

        if not self.flash or verbose:
            # Vanilla implementation
            # (N, h, L, E / h) @ (N, H, E / h, L) -> (N, h, L, L)
            attn = q @ k.transpose(-1, -2) / math.sqrt(dim)

            if self.causal:
                # Masking the attention matrix
                attn = attn.masked_fill(self.mask[..., :L, :L], float("-inf"))

            attn = F.softmax(attn, dim=-1)

            # (N, h, L, L) @ (N, h, L, E / H) -> (N, h, L, E / h)
            z = attn @ v
        else:
            # Fast implementation based on fused kernel
            dropout_p = self.dropout if self.training else 0
            z = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=self.causal)

        # Reformating: (N, h, L, E / h) -> (N, L, h, E / h) -> (N, L, E)
        z = rearrange(z, "N h L dim -> N L (h dim)")

        # Output layer: (N, L, E) @ (E, E) -> (N, L, E)
        z = F.dropout(self.output(z), p=self.dropout, training=self.training)
        if verbose:
            return z, attn
        return z


# --------------------------------------------------------------------------------
# Feed-forward Layers
# --------------------------------------------------------------------------------


class FeedForward(nn.Module):
    r"""Feed-forward network in transformer architecture.

    Parameters
    ----------
    config: configuration class with
        emb_dim: int
            Embedding dimension of the tokens.
        ffn_dim: int
            Hidden dimension of the MLP.
        activation: str
            Activation function. Options are "relu", "gelu".
        ffn_bias: bool
            Whether to use bias in the MLP.
        ffn_dropout: float
            Dropout probability.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # MLP initialization
        self.fc1 = nn.Linear(config.emb_dim, config.ffn_dim, bias=config.ffn_bias)
        self.fc2 = nn.Linear(config.ffn_dim, config.emb_dim, bias=config.ffn_bias)

        # Parsing the activation function
        activation = config.activation.lower()
        self.activation = getattr(F, activation, None)
        if self.activation is None:
            raise ValueError(f"Unknown activation function '{config.activation}'")

        # Drop-out regularization
        self.dropout = config.ffn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Feed-forward network (FFN).

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens after the MHA block.

        Returns
        -------
        out: torch.Tensor of dimension (N, L, E)
            Batch of FF outputs.
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# --------------------------------------------------------------------------------
# Transformer Block
# --------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    r"""
    Transformer block.

    Parameters
    ----------
    config: configuration class with
        emb_dim: int
            Embedding dimension of the tokens.
        pre_norm: bool
            Whether to apply layer normalization before MHA and FFN layers or after.
            See also `On Layer Normalization <https://arxiv.org/pdf/2002.04745>`_.
        norm: str
            Type of normalization layer. Options are "layer", "rms".
        norm_bias: bool
            Whether to use bias in the layer normalization.
        norm_eps: float
            Epsilon parameter for layer normalization.
        and the parameters to initialize SelfAttention and FeedForward.

    See also
    --------
    SelfAttention
    FeedForward
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        match config.norm.lower():
            case "batch":
                NormLayer = BatchNorm
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case _:
                raise ValueError(
                    f"Unknown normalization layer '{config.norm}'. Choose between 'batch', 'layer', and 'rms'."
                )

        self.attn_norm = NormLayer(config.emb_dim, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = SelfAttention(config)
        self.ffn_norm = NormLayer(config.emb_dim, eps=config.norm_eps, bias=config.norm_bias)
        self.ffn = FeedForward(config)
        self.pre_norm = config.pre_norm

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        r"""
        Transformer block.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.
        verbose: bool, default=False
            Whether to return the attention matrices along with the Transformer outputs.

        Returns
        -------
        out: torch.Tensor of dimension (N, L, E)
            Batch of outputs of the transformer block.
        """
        if self.pre_norm:
            out = self.attn(self.attn_norm(x), verbose=verbose)
            if verbose:
                out, att = out
            out = x + out
            out = out + self.ffn(self.ffn_norm(out))
        else:
            out = self.attn(x, verbose=verbose)
            if verbose:
                out, att = out
            out = self.attn_norm(x + out)
            out = self.ffn_norm(out + self.ffn(out))
        if verbose:
            return out, att
        return out

    @torch.inference_mode()
    def _decompose(self, x: torch.Tensor) -> dict:
        r"""
        Recover the outputs of each components of a Transformer block for the same input.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.

        Returns
        -------
        outputs: Dictionnary of outputs for each transformer block's components.
        """
        outputs = {}

        # Layer normalization
        outputs["attn_norm"] = move_to_cpu(self.attn_norm(x))

        # Attention block
        outputs["attn"] = move_to_cpu(self.attn(x))

        # Layer normalization
        outputs["ffn_norm"] = move_to_cpu(self.ffn_norm(x))

        # Feed-forward first layer
        outputs["ffn_fc1"] = move_to_cpu(self.ffn.fc1(x))

        # Feed-forward second layer (it takes as input tensors of higher dimensions)
        zero_x = torch.zeros_like(x)
        expanded_x = torch.cat((x, zero_x, zero_x, zero_x), dim=-1)
        outputs["ffn_fc2"] = move_to_cpu(self.ffn.fc2(expanded_x))

        return outputs

    @torch.inference_mode()
    def _probes(self, x: torch.Tensor) -> tuple[torch.tensor, dict]:
        r"""
        Recover the hidden representation after each component in a transformer block.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.

        Returns
        -------
        probes: dict
            Dictionnary of hidden representation of each component of the block.
        """
        probes = {}
        if self.pre_norm:
            # Attention norm
            out = self.attn_norm(x)
            probes["attn_norm"] = move_to_cpu(out)

            # Attention
            out = self.attn(out)
            probes["attn"] = move_to_cpu(out)

            # Residual connection
            out_res = x + out
            probes["attn_res"] = move_to_cpu(out_res)

            # Feedforward norm
            out = self.ffn_norm(out_res)
            probes["ffn_norm"] = move_to_cpu(out)

            # FC1
            out = self.ffn.fc1(out)
            probes["ffn_fc1"] = move_to_cpu(out)

            # Activation
            out = self.ffn.activation(out)
            probes["ffn_activation"] = move_to_cpu(out)

            # FC2
            out = self.ffn.fc2(out)
            probes["ffn_fc2"] = move_to_cpu(out)

            # Residual connection
            out = out_res + out
            probes["ffn_res"] = move_to_cpu(out)

        else:
            # Attention
            out = self.attn(x)
            probes["attn"] = move_to_cpu(out)

            # Residual connection
            out = x + out
            probes["attn_res"] = move_to_cpu(out)

            # Attention norm
            out_res = self.attn_norm(out)
            probes["attn_norm"] = move_to_cpu(out)

            # FC1
            out = self.ffn.fc1(out_res)
            probes["ffn_fc1"] = move_to_cpu(out)

            # Activation
            out = self.ffn.activation(out)
            probes["ffn_activation"] = move_to_cpu(out)

            # FC2
            out = self.ffn.fc2(out)
            probes["ffn_fc2"] = move_to_cpu(out)

            # Residual connection
            out = out_res + out
            probes["ffn_res"] = move_to_cpu(out)

            # FFN norm
            out = self.ffn_norm(out)
            probes["ffn_norm"] = move_to_cpu(out)

        return out, probes


# --------------------------------------------------------------------------------
# Embedding Module
# --------------------------------------------------------------------------------


class Embedding(nn.Module):
    r"""
    Embedding layer.

    This module implement both token and positional embeddings and support
    embedding of continuous data (e.g., images) via the "linear" embedding type
    and discrete data (e.g., text) via the "dict" embedding type. It also supports
    patching of data, e.g., for computer vision or time series data.

    Parameters
    ----------
    config: configuration class with
        image_dim: tuple
            Tuple of number of channels, height and width of the input images.
        length: int
            Length of the input time series.
        patch_type: Optional[str]
            Type of patching to use. Options are "computer_vision" or "time_series" with
            typical use cases in ViT [4]_ and PatchTST [5]_, respectively.
            If set to None, no patching is used.
        image_patch: str
            Type of image patch to extract. Can be either "raw" or "hybrid".
            If "hybrid", tokens are directly embed in dimension emb_dim so
            the token embedding layer is not needed.
        patch_size: int
            Patch size P.
        stride: int
            Stride S.
        cls_token: bool
            Whether to use a classification token.
        vocab_size: int
            Corresponds to the vocabulary size for discrete data and
            to the dimension of input tokens for continuous data.
        emb_type: str
            Type of token embedding: either "dict" which amounts to amounts to a lookup table that store
            words embeddings, or "linear" which amounts to a trainable linear projection.
        emb_dim: int
            Embedding dimension of the tokens.
        pos_emb: bool
            Whether to use positional embedding.
        freeze_pos: bool
            Whether to have learnable positional embedding.
        seq_len: int
            Maximum sequence length (required if pos_emb is True).
        emb_dropout: float
            Dropout probability for the embeddings layer.

    References
    ----------
    .. [4] A. Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers
           for Image Recognition at Scale. In ICLR 2021.

    .. [5] Y. Nie et al. A Time Series is Worth 64 Words: Long-Term Forecasting
           With Transformers. In ICLR 2023.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # Patching
        if config.patch_type:
            match config.patch_type.lower():
                case "computer_vision":
                    PatchingLayer = PatchImages(
                        image_dim=config.image_dim,
                        image_patch=config.image_patch,
                        patch_size=config.patch_size,
                        emb_dim=config.emb_dim,
                    )
                case "time_series":
                    PatchingLayer = PatchTimeSeries(
                        length=config.length,
                        patch_size=config.patch_size,
                        stride=config.stride,
                    )
                case _:
                    raise ValueError(
                        f"Unknown patching layer '{config.patch_type}'. Choose between"
                        "'computer_vision' and 'time_series'."
                    )
            self.patching = PatchingLayer

            # Update sequence length and vocabulary size
            config.seq_len = self.patching.n_patches
            config.vocab_size = self.patching.patch_dim
            logger.info(f"Each sequence is of length {config.seq_len} and tokens of dimension {config.vocab_size}.")

        else:
            self.patching = None

        # Classification token
        if config.cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.emb_dim))
            config.seq_len += 1
            logger.info(
                f"Sequence length was increased by one, reaching {config.seq_len}, to take into account the cls token."
            )
        else:
            self.cls_token = None

        # Token embedding
        match config.emb_type.lower():
            case "dict":
                EmbeddingLayer = nn.Embedding
            case "linear":
                EmbeddingLayer = nn.Linear
            case _:
                raise ValueError(f"Unknown embedding layer '{config.emb_type}'. Choose between 'dict' and 'linear'")

        self.token_emb = EmbeddingLayer(config.vocab_size, config.emb_dim)

        # Specific case of hybrid patching in ViT
        if config.patch_type:
            if all(
                map(
                    lambda t: t[0].lower() == t[1],
                    ((config.patch_type, "computer_vision"), (config.image_patch, "hybrid")),
                )
            ):
                self.token_emb = nn.Identity()

        # Position embedding
        if config.pos_emb:
            self.L = config.seq_len
            self.pos_dim = config.emb_dim
            self.pos_emb = nn.Parameter(torch.randn(1, self.L, self.pos_dim)).requires_grad_(
                False if config.freeze_pos else True
            )
        else:
            self.pos_emb = None

        # Dropout regularization
        self.dropout = config.emb_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Embedding layer.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, *)
            Batch of raw data to be processed as sequences of tokens of dimension emb_dim.

        Returns
        -------
        out: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.
        """

        # Patching
        if self.patching is not None:
            x = self.patching(x)

        # Token embedding
        out = self.token_emb(x)

        # Cls token
        if self.cls_token is not None:
            N = out.shape[0]
            cls_tokens = repeat(self.cls_token, "1 1 d -> N 1 d", N=N)
            out = torch.cat((cls_tokens, out), dim=1)

        # Position embedding
        if self.pos_emb is not None:
            L = out.shape[1]
            assert L <= self.L, f"Input sequence length {L} is longer than the maximum sequence length {self.L}"
            out[..., : self.pos_dim] = out[..., : self.pos_dim] + self.pos_emb[:L]

        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# --------------------------------------------------------------------------------
# Output Module
# --------------------------------------------------------------------------------


class Output(nn.Module):
    r"""
    Output layer.

    This module implement the task-specific output layer through either an unembedding head,
    typically used for next-token prediction, a classification head or a forecasting head.

    Parameters
    ----------
    config: configuration class with
        vocab_size: int
            Corresponds to the vocabulary size for discrete data and
            to the dimension of input tokens for continuous data.
        emb_type: str
            Type of token embedding. Options are "dict" which amounts to a lookup table that store
            words embeddings, and "linear" which amounts to a trainable linear projection.
        emb_dim: int
            Embedding dimension of the tokens.
        output_type: str
            Type of output. Options are "classification", "forecasting" and "sequence_to_sequence".
        output_dropout: float
            Dropout probability for the output layer.
        n_classes: int
            Number of classes.
        forecasting_horizon: int
            Forecasting horizon.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # Task-specific output layer
        self.output_type = config.output_type
        match self.output_type.lower():
            case "classification":
                OutputLayer = ClassificationLayer(
                    emb_dim=config.emb_dim,
                    n_classes=config.n_classes,
                    norm=config.norm,
                    norm_eps=config.norm_eps,
                    norm_bias=config.norm_bias,
                    dropout=config.output_dropout,
                )
            case "forecasting":
                OutputLayer = ForecastingLayer(
                    seq_len=config.seq_len,
                    emb_dim=config.emb_dim,
                    forecasting_horizon=config.forecasting_horizon,
                    dropout=config.output_dropout,
                )
            case "sequence_to_sequence":
                OutputLayer = Seq2SeqLayer(
                    emb_dim=config.emb_dim,
                    vocab_size=config.vocab_size,
                    norm=config.norm,
                    norm_eps=config.norm_eps,
                    norm_bias=config.norm_bias,
                    dropout=config.output_dropout,
                )
            case _:
                raise ValueError(
                    f"Unknown output '{config.output_type}'. Choose between 'classification', 'forecasting'",
                    "and 'sequence_to_sequence'.",
                )
        self.output_layer = OutputLayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Output layer.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens processed by the Transformer.

        Returns
        -------
        out: torch.Tensor of dimension (N, *)
            Batch of outputs. The dimension depends on the task-specific output layer.
                - For classification, the output has dimension (N, n_classes).
                - For forecasting, the output has dimension (N, forecasting_horizon).
                - For sequence_to_sequence, the output has dimension (N, L, vocab_size).
        """
        out = self.output_layer(x)
        return out

    def apply_weight_tying(self, embedding_layer: nn.Module) -> None:
        r"""Tying token embedding and un-embedding weights for sequence-to-sequence task."""
        if hasattr(self.output_layer, "apply_weight_tying"):
            self.output_layer.apply_weight_tying(embedding_layer=embedding_layer)


# --------------------------------------------------------------------------------
# Transformer Architecture
# --------------------------------------------------------------------------------


class Transformer(nn.Module):
    r"""
    Multihead multi-layer transformer.

    This implementation supports both encoder-only and decoder-only transformers
    with task-specific outputs which allows to easily reimplement common models
    such as GPT2, Mistral, ViT or PatchTST.

    Parameters
    ----------
    config: configuration class with
        n_layers: int
            Number of transformer blocks.
        weight_tying: bool
            Whether to use weight tying between the token embedding and the output layer.
            See also `Weight Tying <https://arxiv.org/pdf/1608.05859>`_.
        output_type: str
            Type of output. Options are "classification", "forecasting" and "sequence_to_sequence".
    See also
    --------
    Embedding
    TransformerBlock
    Output
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # Embedding
        self.embedding = Embedding(config)

        # Transformer model
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Output layer
        self.output = Output(config)

        # Tying token embedding and un-embedding for sequence-to-sequence task
        if config.weight_tying and (config.output_type.lower() == "sequence_to_sequence"):
            self.output.apply_weight_tying(embedding_layer=self.embedding)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        r"""
        Transformer model.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, *)
            Batch of raw data to be processed as sequences of tokens of dimension emb_dim.
        verbose: bool, default=False
            Whether to return the attention matrices along with the Transformer outputs.

        Returns
        -------
        out: torch.Tensor of dimension (N, *)
            Batch of outputs. The dimension depends on the task-specific output layer.
                - For sequence-to-sequence, the output has dimension (N, L, vocab_size).
                - For classification, the output has dimension (N, n_classes).
                - For forecasting, the output has dimension (N, forecasting_horizon).
        """
        out = self.embedding(x)
        attentions = []
        for block in self.blocks:
            out = block(out, verbose=verbose)
            if verbose:
                out, att = out
                attentions.append(att)
        out = self.output(out)
        if verbose:
            attentions = torch.stack(attentions)
            return out, attentions
        return out

    @torch.inference_mode()
    def get_decomposition(self, x: torch.Tensor) -> dict:
        r"""
        Recover the outputs of ViT components across layers for the same input.

        Since transformers are sequence-to-sequence models, those outputs are
        sequences of tokens that can be seen as discrete probability measures [2, 3]
        for the embedding layer and the components of each transformer block.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, *)
            Batch of input data to be patched and embedded in sequences of tokens of dimension E.

        Returns
        -------
        outputs: Dictionnary of output of each model's layer.

        References
        ----------
        .. [2] G. de Bie et al. Stochastic Deep Networks. In ICML 2019.
        .. [3] M. Sander et al. Sinkformers: Transformers with Doubly
               Stochastic Attention. In AISTATS 2022.
        """
        outputs = {}

        # Embedding layer
        out = self.embedding(x)
        outputs["embedding"] = move_to_cpu(out)

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            # Recover decomposition
            block_output = block._decompose(out)
            for key in block_output.keys():
                outputs[f"block{i}" + "_" + key] = block_output[key]

        return outputs

    @torch.inference_mode()
    def get_probes(self, x: torch.Tensor) -> dict:
        r"""
        Recover the successive hidden representation of ViT components across layers.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, *)
            Batch of input data to be patched and embedded in sequences of tokens of dimension E.

        Returns
        -------
        probes: Dictionnary of output of each component's hidden representation over layers.
        """
        probes = {}

        # Embedding layer
        out = self.embedding(x)

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            # Recover decomposition
            out, block_probes = block._probes(out)
            for key in block_probes.keys():
                probes[f"block{i}" + "_" + key] = block_probes[key]

        return probes
