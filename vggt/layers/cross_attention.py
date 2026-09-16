"""Reusable cross-attention layer with distinct query and key/value tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossAttention(nn.Module):
    """Apply biased multi-head attention from queries to a separate context.

    Query, key, value, and output projections include bias and use no dropout.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        """Initialize independent biased projections for one cross-attention site."""
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.value = nn.Linear(dim, dim, bias=True)
        self.output = nn.Linear(dim, dim, bias=True)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """Attend each ``[B, N, dim]`` query to every ``[B, M, dim]`` context token.

        Raises:
            ValueError: If ranks, batches, or widths are incompatible.
        """
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError("query and context must be rank-3 tensors")
        if query.shape[0] != context.shape[0]:
            raise ValueError("query and context batch sizes must match")
        if query.shape[-1] != self.dim or context.shape[-1] != self.dim:
            raise ValueError(f"query and context widths must equal {self.dim}")
        batch_size, query_tokens, _ = query.shape
        context_tokens = context.shape[1]
        q = self.query(query).reshape(batch_size, query_tokens, self.num_heads, self.head_dim)
        k = self.key(context).reshape(batch_size, context_tokens, self.num_heads, self.head_dim)
        v = self.value(context).reshape(batch_size, context_tokens, self.num_heads, self.head_dim)
        weights = torch.matmul(
            q.transpose(1, 2) * self.scale,
            k.transpose(1, 2).transpose(-2, -1),
        ).softmax(dim=-1)
        attended = torch.matmul(weights, v.transpose(1, 2))
        return self.output(
            attended.transpose(1, 2).reshape(batch_size, query_tokens, self.dim)
        )
