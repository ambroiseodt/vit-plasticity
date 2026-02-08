r"""
Utils for transformer models.

Comments abbreviations:
    N: batch size
    H: image height
    W: image width
    C: number of channels
    T: length of the time series
    P: patch size
    S: stride
    L: sequence length (or number of patches)
    E: embedding dimension
    h: number of heads
    F: forecasting horizon
    V: vocabulary size

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# --------------------------------------------------------------------------------
# Patching Layers
# --------------------------------------------------------------------------------


class PatchImages(nn.Module):
    r"""
    Patch images into sequences of tokens following [1]_.

    This module supports the raw patching and the hybrid one using a CNN.
    The raw patching splits the image into non-overlapping patches of size P.
    The hybrid one extract the patches from a CNN, directly embedding the tokens to emb_dim.
    The output is a sequence of n_patch tokens of dimension either P**2 * C or emb_dim.

    Parameters
    ----------
    image_dim: tuple
        Number of channels, height and width of the input images.
    image_patch: str
        Type of patching to use on images. Can be either "raw" or "hybrid".
        If "hybrid", tokens are directly embed in dimension emb_dim so
        the token embedding layer is not needed.
    emb_dim: int
        Embedding dimension of the tokens.
    patch_size: int
        Patch size P.
    n_patches: int
        Number of patches, equal to (H * W) // P**2.
    patch_dim: int
        Size of the flattened patches, equal to P**2 * C

    Notes
    -----
    Implementation inspired by https://github.com/huggingface/transformers.

    References
    ----------
    .. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
            for Image Recognition at Scale. ICLR 2021.
    """

    def __init__(self, image_dim: tuple, image_patch: str, patch_size: int, emb_dim: int):
        super().__init__()
        n_channels, height, width = image_dim

        # Check image dimensions
        is_divisible = (height % patch_size == 0) and (width % patch_size == 0)
        assert is_divisible, "Image dimensions must be divisible by the patch size."

        # Recover patch dimensions
        self.n_patches = height * width // (patch_size**2)
        self.patch_dim = patch_size**2 * n_channels

        match image_patch.lower():
            case "raw":
                patching = Rearrange("N (h P1) (w P2) C-> N (h w) (P1 P2 C)", P1=patch_size, P2=patch_size)
            case "hybrid":
                patching = nn.Sequential(
                    nn.Conv2d(in_channels=n_channels, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size),
                    nn.Flatten(start_dim=2),
                )
            case _:
                raise ValueError(f"Unknown patching type '{image_patch}'. Choose between 'raw' and 'hybrid'.")
        self.patching = patching

    def forward(self, x: torch.tensor) -> torch.Tensor:
        r"""
        Patching layer for images.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of dimension (N, H, W, C).

        Returns
        -------
        patched_x: torch.Tensor
            Output tensor which dimension depends on patch_type.
                - If patch_type is "raw", output dimension is (N, n_patches, P**2 * C).
                - If patch_type is "hybrid", output dimension is (N, n_patches, emb_dim).
        """
        patched_x = self.patching(x).transpose(1, 2)
        return patched_x


class PatchTimeSeries(nn.Module):
    r"""
    Patch time series into sequences of tokens following [2]_.

    The patching is done by padding time series with S repeated numbers of their last entry
    and then splitting the time series into non-overlapping patches of size P.
    The output is a sequence of n_patches tokens of dimension either P.

    Parameters
    ----------
    patch_size: int
        Patch size P.
    stride: int
        Stride S.
    length: int
        Length of the input time series.
    n_patches: int
        Number of patches, equal to floor((T - P) / S) + 2.
    patch_dim: int
        Size of the patches, equal to P.

    Notes
    -----
    Implementation inspired by https://github.com/yuqinie98/PatchTST.

    References
    ----------
    .. [2] Y. Nie et al. A Time Series is Worth 64 Words: Long-Term Forecasting
        With Transformers. In ICLR 2023.
    """

    def __init__(self, length: int, patch_size: int, stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_patches = math.floor((length - patch_size) / stride) + 2
        self.patch_dim = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Patching layer for time series.

        x: torch.Tensor
            Input tensor of dimension (N, T).

        Returns
        ----------
        patched_x: torch.Tensor
            Output tensor of dimension (N, n_patches, P).
        """

        # Padding: (N, T) -> (N, T + S)
        padding_layer = nn.ReplicationPad1d((0, self.stride))
        x = padding_layer(x)

        # Patching: (N, T + S) -> (N, n_patches, P)
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        return patched_x


# --------------------------------------------------------------------------------
# Normalization Layers
# --------------------------------------------------------------------------------


class BatchNorm(nn.Module):
    r"""
    Batch normalization layer following [1]_.

    Parameters
    ----------
    fan_in: int
        Input dimension.
    bias: bool, default=False.
        Whether to use a bias in the layer normalization.
    eps: float, default=1e-5.
        Epsilon value for numerical stability.
        If the value is too small compared to float precision, the input should be converted
        to float before computing the norm with x --> x.float(). The obtained output should
        then be converted back to the original type with norm --> norm.type_as(x).

    Notes
    -----
    Implementation inspired by https://github.com/yuqinie98/PatchTST.

    References
    ----------
    .. [1] Y. Nie et al. A Time Series is Worth 64 Words: Long-Term Forecasting
        With Transformers. In ICLR 2023.
    """

    def __init__(self, fan_in: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(fan_in, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Batch normalization.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.

        Returns
        -------
        out: torch.Tensor of dimension (N, L, E)
            Batch of normalized sequences of tokens.
        """
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        return x.transpose(1, 2)


if torch.__version__ < "2.1":

    class LayerNorm(nn.Module):
        r"""
        LayerNorm normalization layer following [2]_.

        Parameters
        ----------
        fan_in: int
            Input dimension.
        bias: bool, default=False.
            Whether to use a bias in the layer normalization.
        eps: float, default=1e-5.
            Epsilon value for numerical stability.
            If the value is too small compared to float precision, the input should be converted
            to float before computing the norm with x --> x.float(). The obtained output should
            then be converted back to the original type with out --> out.type_as(x).

        Notes
        -----
        Pytorch 2.0.1 does not have LayerNorm without bias. This implementation is a workaround.
        Implementation inspired by https://github.com/facebookresearch/pal.

        References
        ----------
        .. [2] Jimmy Lei Ba et al. Layer normalization. arXiv preprint, 2016.

        """

        def __init__(self, fan_in: int, eps: float = 1e-5, bias: bool = False):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(fan_in))
            self.bias = nn.Parameter(torch.zeros(fan_in)) if bias else None
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            r"""
            Layer normalization.

            Parameters
            ----------
            x: torch.Tensor of dimension (N, L, E)
                Batch of sequences of tokens.

            Returns
            -------
            out: torch.Tensor of dimension (N, L, E)
                Batch of normalized sequences of tokens.
            """
            out = F.layer_norm(
                x,
                normalized_shape=self.weight.shape,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )

            return out

else:
    LayerNorm = nn.LayerNorm


class RMSNorm(nn.Module):
    r"""
    RMSNorm normalization layer following [3]_.

    Parameters
    ----------
    fan_in: int
        Input dimension.
    bias: bool, default=False.
        Whether to use a bias in the layer normalization.
    eps: float, default=1e-5.
        Epsilon value for numerical stability.
        If the value is too small compared to float precision, the input should be converted
        to float before computing the norm with x --> x.float(). The obtained output should
        then be converted back to the original type with norm --> norm.type_as(x).

    Notes
    -----
    This is the normalization used in Mistral models.
    Implementation inspired by https://github.com/facebookresearch/pal.

    References
    ----------
    .. [3] Bia Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In NeurIPS 2019.
    """

    def __init__(self, fan_in: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(fan_in))
        self.bias = nn.Parameter(torch.zeros(fan_in)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        RMS normalization.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens.

        Returns
        -------
        out: torch.Tensor of dimension (N, L, E)
            Batch of normalized sequences of tokens.
        """
        norm = (x**2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        out = x / norm
        out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out.type_as(x)


# --------------------------------------------------------------------------------
# Task-specific Layers
# --------------------------------------------------------------------------------


class ClassificationLayer(nn.Module):
    r"""
    Classification with the transformer using the CLS token following [1]_.

    Parameters
    ----------
    emb_dim: int
        Embedding dimension of the tokens.
    n_classes: int
        Number of classes.
    Notes
    -----
    Implementation inspired by https://github.com/huggingface/transformers.

    References
    ----------
    .. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
            for Image Recognition at Scale. ICLR 2021.
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        norm: str,
        norm_eps: float,
        norm_bias: bool,
        dropout: float,
    ):
        super().__init__()

        match norm.lower():
            case "batch":
                NormLayer = BatchNorm
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case _:
                raise ValueError(f"Unknown normalization layer '{norm}'. Choose between 'batch', 'layer', and 'rms'.")

        self.output_norm = NormLayer(emb_dim, eps=norm_eps, bias=norm_bias)
        self.dropout = dropout
        self.output = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Classification layer.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens processed by the Transformer.

        Returns
        -------
        out: torch.Tensor of dimension (N, n_classes)
            Batch of logits each of dimension n_classes.
        """

        # Normalization
        x = self.output_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classifier applied on the CLS token
        out = self.output(x[:, 0, :])

        return out


class ForecastingLayer(nn.Module):
    r"""
    Forecasting with transformer layer following [2].

    Parameters
    ----------
    emb_dim: int
        Embedding dimension of the tokens.
    seq_len: int
        Maximum sequence length (it corresponds to the number of patches).
    forecasting_horizon: int
        Forecasting horizon.

    Notes
    -----
    Implementation inspired by https://github.com/yuqinie98/PatchTST.

    References
    ----------
    .. [2] Y. Nie et al. A Time Series is Worth 64 Words: Long-Term Forecasting
        With Transformers. In ICLR 2023.
    """

    def __init__(
        self,
        seq_len: int,
        emb_dim: int,
        forecasting_horizon: int,
        dropout: float,
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.output = nn.Linear(seq_len * emb_dim, forecasting_horizon)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forecasting layer.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens processed by the Transformer.

        Returns
        -------
        out: torch.Tensor of dimension (N, forecasting_horizon)
            Batch of predicted time series.
        """

        # Flatten: (N, L, E) -> (N, L * E)
        x = self.flatten(x)

        # Project to the forecasting horizon: (N, L * E) -> (N, F)
        out = self.output(x)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class Seq2SeqLayer(nn.Module):
    r"""
    Sequence-to-sequence layer projecting tokens back to the original vocabulary space.

    This module is used for language modeling, e.g, for next-token predictions tasks [3]_ or
    to solve mathematical tasks like [4]_.

    Parameters
    ----------
    emb_dim: int
        Embedding dimension of the tokens.
    vocab_size: int
        Vocabulary size.

    Notes
    -----
    Implementation inspired by https://github.com/facebookresearch/pal and
    https://github.com/huggingface/transformers.

    References
    ----------
    .. [3] A. Radford et al. Language Models are Unsupervised Multitask Learners. arXiv 2019
    .. [4] A. Odonnat et al. Easing Optimization Paths: A Circuit Perspective. In ICASSP 2025.
    """

    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        norm: str,
        norm_eps: float,
        norm_bias: bool,
        dropout: float,
    ):
        super().__init__()

        match norm.lower():
            case "batch":
                NormLayer = BatchNorm
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case _:
                raise ValueError(f"Unknown normalization layer '{norm}'. Choose between 'batch', 'layer', and 'rms'.")

        self.output_norm = NormLayer(emb_dim, eps=norm_eps, bias=norm_bias)
        self.dropout = dropout
        self.output = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Sequence-to-sequence layer.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Batch of sequences of tokens processed by the Transformer.

        Returns
        -------
        out: torch.Tensor of dimension (N, L, V)
            Batch of sequences of tokens in the original vocabulary space.
        """

        # Normalization
        x = self.output_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Project back to the vocabulary space: (N, L, E) -> (N, L, V)
        out = self.output(x)

        return out

    def apply_weight_tying(self, embedding_layer: nn.Module) -> None:
        r"""Tying token embedding and un-embedding weights for sequence-to-sequence task."""
        self.output.weight.data = embedding_layer.token_emb.weight.data
