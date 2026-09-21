"""Private shared implementation for E01a recurrent-memory read adaptors."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from vggt.layers.cross_attention import CrossAttention


class _ReadAdaptorBlock(nn.Module):
    """Apply one pre-normalized memory cross-attention residual update."""

    def __init__(self, memory_dim: int, num_heads: int) -> None:
        """Create unshared norms and biased cross-attention for one block."""
        super().__init__()
        self.query_norm = nn.LayerNorm(memory_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.cross_attention = CrossAttention(memory_dim, num_heads)

    def forward(self, queries: Tensor, read_memory: Tensor) -> Tensor:
        """Return queries plus their cross-attention update from all memory tokens."""
        return queries + self.cross_attention(
            self.query_norm(queries), self.memory_norm(read_memory)
        )


class _ReadAdaptorBase(nn.Module):
    """Project features through independent E01a memory cross-attention blocks.

    Subclasses validate and flatten their public feature layout, then use
    ``_adapt_flattened`` to preserve its original ordering on reconstruction.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 2048,
        memory_dim: int = 512,
        num_frames: int = 8,
        memory_tokens: int = 16,
        num_blocks: int = 3,
        num_heads: int = 8,
    ) -> None:
        """Initialize a configurable E01a adaptor and its scalar residual gate.

        Raises:
            ValueError: If dimensions are nonpositive or the memory width does
                not divide equally into attention heads.
        """
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "memory_dim": memory_dim,
            "num_frames": num_frames,
            "memory_tokens": memory_tokens,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if memory_dim % num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.num_frames = num_frames
        self.memory_tokens = memory_tokens
        self.num_heads = num_heads
        self.input_projection = nn.Linear(feature_dim, memory_dim, bias=True)
        self.blocks = nn.ModuleList(
            [_ReadAdaptorBlock(memory_dim, num_heads) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Linear(memory_dim, feature_dim, bias=True)
        self.residual_gate = nn.Parameter(torch.tensor(-2.1972246))

    def _validate_common(self, features: Tensor, read_memory: Tensor) -> None:
        """Validate shared batch, memory, device, and dtype adaptor requirements.

        Raises:
            ValueError: If tensors have incompatible batches, memory shape,
                device, or dtype relative to this adaptor's parameters.
        """
        if read_memory.ndim != 3:
            raise ValueError("read_memory must be rank 3 [B, memory_tokens, memory_dim]")
        if features.shape[0] != read_memory.shape[0]:
            raise ValueError("feature and read_memory batch sizes must match")
        if read_memory.shape[1] != self.memory_tokens:
            raise ValueError(f"read_memory memory_tokens must equal {self.memory_tokens}")
        if read_memory.shape[2] != self.memory_dim:
            raise ValueError(f"read_memory memory width must equal {self.memory_dim}")
        parameter = self.input_projection.weight
        if features.device != parameter.device or read_memory.device != parameter.device:
            raise ValueError("feature and read_memory device must match adaptor parameters")
        if features.dtype != parameter.dtype or read_memory.dtype != parameter.dtype:
            raise ValueError("feature and read_memory dtype must match adaptor parameters")

    def _adapt_flattened(self, features: Tensor, read_memory: Tensor) -> Tensor:
        """Return a gated width-preserving update for flattened feature queries.

        Args:
            features: Original features shaped ``[B, queries, feature_dim]``.
            read_memory: Incoming state shaped ``[B, memory_tokens, memory_dim]``.

        Returns:
            Adapted features in the exact input shape without modifying either
            argument or detaching the recurrent-memory graph.
        """
        internal = self.input_projection(features)
        for block in self.blocks:
            internal = block(internal, read_memory)
        update = self.output_projection(internal)
        return features + torch.sigmoid(self.residual_gate) * update
