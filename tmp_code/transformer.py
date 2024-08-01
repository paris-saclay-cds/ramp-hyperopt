import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
import numpy as np
import transformers
from typing import Optional
from ramphy import Hyperparameter

num_layers = Hyperparameter(dtype="int", default=1, values=[1, 2, 3, 5])
num_heads = Hyperparameter(dtype="int", default=1, values=[1, 2, 5, 10])
ff_size = Hyperparameter(dtype="int", default=128, values=[128, 256, 512, 1024])
activation = Hyperparameter(dtype="str", default="gelu", values=["gelu", "relu"])
use_revin = Hyperparameter(dtype="bool", default=True, values=[True, False])

NUM_LAYERS = int(num_layers)
NUM_HEADS = int(num_heads)
FF_SIZE = int(ff_size)
ACTIVATION = str(activation)
USE_REVIN = bool(use_revin)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, feat_size: int, num_heads: int, ff_size: int, activation: str = "gelu", dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Attention
        self._self_attention = nn.MultiheadAttention(embed_dim=feat_size, num_heads=num_heads, dropout=dropout)
        self.dropout_att = nn.Dropout(dropout)
        self.norm_att = nn.LayerNorm(feat_size)

        # FF block
        self.norm_ff = nn.LayerNorm(feat_size)
        self.linear1 = nn.Linear(feat_size, ff_size)
        self.linear2 = nn.Linear(ff_size, feat_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU
        elif activation == "relu":
            self.activation = nn.ReLU
        else:
            raise ValueError("Activation has to be either ReLU or GeLU")

    def forward(self, x: torch.Tensor, att_mask: Optional[torch.Tensor]) -> torch.Tensor:
        att = self.self_attention(self.norm_att(x), att_mask=att_mask)
        x = x + att
        x = x + self.ff_block(self.norm_ff(x))
        return x

    def self_attention(self, x: torch.Tensor, att_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self._self_attention(x, x, x, att_mask=att_mask, is_causal=False)[0]
        return self.dropout_att(x)

    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Transformer(nn.Module):
    def __init__(
        self, num_layers: int, input_size: int, num_heads: int, ff_size: int, activation: str = "gelu"
    ) -> None:
        super().__init__()
        if activation not in ["relu", "gelu"]:
            raise ValueError("Only use Relu or Gelu")

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            batch_first=True,
            norm_first=True,  # Apparently this is better
            activation=activation,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, memory=x)


class Regressor(BaseEstimator):
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO tokenize the dataset to get the right input_size
        input_size = len(self.metadata["features"])  # check this

        if USE_REVIN:
            self.revin = RevIN(num_features=input_size)

        self.transformer = Transformer(
            num_layers=NUM_LAYERS, input_size=input_size, num_heads=NUM_HEADS, ff_size=FF_SIZE, activation=ACTIVATION
        )


# class Regressor(BaseEstimator):
#     def __init__(self, metadata) -> None:
#         super().__init__()
#         self.metadata = metadata
