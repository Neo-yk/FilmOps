# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
from typing import Optional
import torch
from torch import nn, Tensor

import torch.nn.functional as F
import math

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: Optional[float] = None, init_proj_std: Optional[float] = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def scaled_dot_product_attention(
        self, query, key, value, 
        attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        # Validate input shapes
        assert query.dim() == key.dim() == value.dim(), "Q, K, V must have same dimensionality"
        assert query.size(-1) == key.size(-1), "Embedding dim of Q and K must match"
        
        # Compute attention scores
        d_k = query.size(-1)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply causal mask (if requested and no attn_mask provided)
        if is_causal and attn_mask is None:
            L, S = query.size(-2), key.size(-2)
            causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
            # Expand mask dimensions to match attn_scores
            while causal_mask.dim() < attn_scores.dim():
                causal_mask = causal_mask.unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply provided attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            else:  # Float mask (additive)
                attn_scores = attn_scores + attn_mask
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        
        # Weighted sum of values
        return torch.matmul(attn_weights, value)


    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        # x = nn.functional.scaled_dot_product_attention(
        x = self.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
