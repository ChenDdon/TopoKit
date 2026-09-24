"""Compact pre-norm encoder; imported only when PyTorch is requested."""
import torch
from torch import nn
from torch.nn import functional as F

from .encoding import position_encoding


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_heads
        self.attention_dropout = c.attention_dropout
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.layer_norm_eps)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.attention_output = nn.Linear(c.d_model, c.d_model)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.layer_norm_eps)
        self.fc1 = nn.Linear(c.d_model, c.d_model * c.ff_multiplier)
        self.fc2 = nn.Linear(c.d_model * c.ff_multiplier, c.d_model)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attention_dropout if self.training else 0.0)
        x = x + self.dropout(self.attention_output(y.transpose(1, 2).reshape(b, n, d)))
        return x + self.dropout(self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class TopoFormerMini(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.projection = nn.Linear(c.n_statistics * c.n_channels, c.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.d_model))
        self.register_buffer("positions", torch.from_numpy(position_encoding(c.n_scales, c.d_model)))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = nn.LayerNorm(c.d_model, eps=c.layer_norm_eps)
        self.pooler = nn.Linear(c.d_model, c.d_model)
        self.regressor = nn.Linear(c.d_model, 1)
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != self.config.feature_shape:
            raise ValueError(f"expected batch x {self.config.feature_shape}")
        b = features.shape[0]
        tokens = features.permute(0, 2, 1, 3).reshape(b, self.config.n_scales, -1)
        x = torch.cat((self.cls_token.expand(b, -1, -1), self.projection(tokens)), dim=1)
        x = x + self.positions
        for block in self.blocks:
            x = block(x)
        return self.regressor(torch.tanh(self.pooler(self.norm(x)[:, 0]))).squeeze(-1)
