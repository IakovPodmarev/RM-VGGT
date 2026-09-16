"""Standalone E01a recurrent memory writer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vggt.layers.attention import Attention
from vggt.layers.cross_attention import CrossAttention


class MemoryWriterBlock(nn.Module):
    """Perform one E01a self-, camera-cross-, and patch-cross-attention block."""

    def __init__(self, memory_dim: int, num_heads: int) -> None:
        """Initialize the block's unshared pre-normalized attention sublayers."""
        super().__init__()
        self.self_norm = nn.LayerNorm(memory_dim)
        self.self_attention = Attention(memory_dim, num_heads=num_heads, qkv_bias=True, proj_bias=True, attn_drop=0.0, proj_drop=0.0)
        self.camera_query_norm = nn.LayerNorm(memory_dim)
        self.camera_context_norm = nn.LayerNorm(memory_dim)
        self.camera_cross_attention = CrossAttention(memory_dim, num_heads)
        self.patch_query_norm = nn.LayerNorm(memory_dim)
        self.patch_context_norm = nn.LayerNorm(memory_dim)
        self.patch_cross_attention = CrossAttention(memory_dim, num_heads)

    def forward(self, write_tokens: Tensor, camera_context: Tensor, patch_context: Tensor) -> Tensor:
        """Update all write tokens from self, camera, then patch context."""
        write_tokens = write_tokens + self.self_attention(self.self_norm(write_tokens))
        write_tokens = write_tokens + self.camera_cross_attention(self.camera_query_norm(write_tokens), self.camera_context_norm(camera_context))
        return write_tokens + self.patch_cross_attention(self.patch_query_norm(write_tokens), self.patch_context_norm(patch_context))


class MemoryWriter(nn.Module):
    """Produce E01a recurrent memory from one segment's frozen layer-23 features.

    Defaults implement eight input frames, eight slots per typed group, a
    37-by-37 patch grid pooled to 16-by-16, width 2048 observations, width 512
    state, three blocks, and four attention heads. Type and slot embeddings
    affect attention representations only and are never accumulated into state.
    """

    def __init__(self, *, feature_dim: int = 2048, memory_dim: int = 512, num_frames: int = 8, memory_slots_per_type: int = 8, patch_grid_size: int = 37, pooled_grid_size: int = 16, num_blocks: int = 3, num_heads: int = 4) -> None:
        """Initialize writer parameters and reject incompatible E01a geometry.

        Raises:
            ValueError: If dimensions are nonpositive, incompatible with heads,
            or cannot allocate equal sine/cosine channels to both spatial axes.
        """
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "memory_dim": memory_dim,
            "num_frames": num_frames,
            "memory_slots_per_type": memory_slots_per_type,
            "patch_grid_size": patch_grid_size,
            "pooled_grid_size": pooled_grid_size,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if memory_dim % num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        if memory_dim % 4:
            raise ValueError("memory_dim must allocate four equal positional parts")
        self.feature_dim, self.memory_dim, self.num_frames = feature_dim, memory_dim, num_frames
        self.memory_slots_per_type = memory_slots_per_type
        self.total_memory_slots = memory_slots_per_type * 2
        self.patch_grid_size, self.pooled_grid_size = patch_grid_size, pooled_grid_size
        self.initial_memory_bank = nn.Parameter(torch.empty(1, self.total_memory_slots, memory_dim))
        self.write_queries = nn.Parameter(torch.empty(1, self.total_memory_slots, memory_dim))
        self.memory_type_embedding = nn.Parameter(torch.empty(2, memory_dim))
        self.query_type_embedding = nn.Parameter(torch.empty(2, memory_dim))
        self.slot_embedding = nn.Parameter(torch.empty(self.total_memory_slots, memory_dim))
        self.temporal_embedding = nn.Parameter(torch.empty(num_frames, memory_dim))
        self.camera_projection = nn.Linear(feature_dim, memory_dim, bias=True)
        self.patch_projection = nn.Linear(feature_dim, memory_dim, bias=True)
        self.blocks = nn.ModuleList([MemoryWriterBlock(memory_dim, num_heads) for _ in range(num_blocks)])
        self.candidate_mlp = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, memory_dim * 4), nn.GELU(), nn.Linear(memory_dim * 4, memory_dim))
        self.gate_mlp = nn.Sequential(nn.LayerNorm(memory_dim * 2), nn.Linear(memory_dim * 2, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1))
        self.register_buffer("patch_position_embedding", self._make_patch_position_embedding(pooled_grid_size, memory_dim), persistent=False)
        self._reset_parameters()

    @staticmethod
    def _make_patch_position_embedding(grid_size: int, memory_dim: int) -> Tensor:
        """Construct the approved float32 row-major 2D sine/cosine encoding."""
        frequencies = memory_dim // 4
        omega = 1.0 / (10000 ** (torch.arange(frequencies, dtype=torch.float32) / frequencies))
        rows = torch.arange(grid_size, dtype=torch.float32).repeat_interleave(grid_size)
        columns = torch.arange(grid_size, dtype=torch.float32).repeat(grid_size)
        return torch.cat((torch.sin(rows[:, None] * omega), torch.cos(rows[:, None] * omega), torch.sin(columns[:, None] * omega), torch.cos(columns[:, None] * omega)), dim=-1)

    def _reset_parameters(self) -> None:
        """Initialize learned tokens and the requested approximately one-third gate."""
        for parameter in (self.initial_memory_bank, self.write_queries, self.memory_type_embedding, self.query_type_embedding, self.slot_embedding, self.temporal_embedding):
            nn.init.normal_(parameter, std=0.02)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -0.708)

    def initial_memory(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Return independent typed initial state for a requested batch.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        bank = self.initial_memory_bank
        return bank.to(device=device or bank.device, dtype=dtype or bank.dtype).expand(batch_size, -1, -1).clone()

    def _memory_attention_tokens(self, memory: Tensor, type_index: int) -> Tensor:
        """Add non-recurrent type and slot representations for one memory group."""
        start = type_index * self.memory_slots_per_type
        return memory + self.memory_type_embedding[type_index] + self.slot_embedding[start : start + self.memory_slots_per_type]

    def _build_contexts(self, camera_features: Tensor, patch_features: Tensor, read_memory: Tensor) -> tuple[Tensor, Tensor]:
        """Project observations and append only their matching typed memory tokens."""
        batch_size = camera_features.shape[0]
        camera_observations = self.camera_projection(camera_features) + self.temporal_embedding
        patch_grid = patch_features.reshape(batch_size * self.num_frames, self.patch_grid_size, self.patch_grid_size, self.feature_dim)
        pooled = F.adaptive_avg_pool2d(patch_grid.permute(0, 3, 1, 2), (self.pooled_grid_size, self.pooled_grid_size))
        patch_observations = self.patch_projection(pooled.permute(0, 2, 3, 1).reshape(batch_size, self.num_frames, self.pooled_grid_size**2, self.feature_dim))
        positions = self.patch_position_embedding.to(device=patch_observations.device, dtype=patch_observations.dtype)
        patch_observations = patch_observations + self.temporal_embedding[:, None, :] + positions[None, None, :, :]
        patch_observations = patch_observations.reshape(batch_size, self.num_frames * self.pooled_grid_size**2, self.memory_dim)
        return (torch.cat((camera_observations, self._memory_attention_tokens(read_memory[:, :self.memory_slots_per_type], 0)), dim=1), torch.cat((patch_observations, self._memory_attention_tokens(read_memory[:, self.memory_slots_per_type:], 1)), dim=1))

    def _validate_inputs(self, camera_features: Tensor, patch_features: Tensor, read_memory: Tensor) -> None:
        """Reject rank, batch, width, frame, patch-grid, and memory-shape violations."""
        if camera_features.ndim != 3:
            raise ValueError("camera_features must be rank 3 [B, frames, feature_dim]")
        if patch_features.ndim != 4:
            raise ValueError("patch_features must be rank 4 [B, frames, patches, feature_dim]")
        if read_memory.ndim != 3:
            raise ValueError("read_memory must be rank 3 [B, memory slots, memory_dim]")
        if camera_features.shape[0] != patch_features.shape[0] or camera_features.shape[0] != read_memory.shape[0]:
            raise ValueError("camera_features, patch_features, and read_memory batch sizes must match")
        if camera_features.shape[1] != self.num_frames or patch_features.shape[1] != self.num_frames:
            raise ValueError(f"num_frames must equal {self.num_frames}")
        if camera_features.shape[-1] != self.feature_dim or patch_features.shape[-1] != self.feature_dim:
            raise ValueError(f"feature_dim must equal {self.feature_dim}")
        if patch_features.shape[2] != self.patch_grid_size**2:
            raise ValueError(f"patch_features must contain patch_grid_size squared ({self.patch_grid_size**2}) tokens")
        if read_memory.shape[1] != self.total_memory_slots:
            raise ValueError(f"read_memory memory slots must equal {self.total_memory_slots}")
        if read_memory.shape[-1] != self.memory_dim:
            raise ValueError(f"read_memory width must equal {self.memory_dim}")

    def forward(self, camera_features: Tensor, patch_features: Tensor, read_memory: Tensor) -> tuple[Tensor, Tensor]:
        """Write one E01a transition without detaching or mutating inputs.

        Returns:
            ``next_memory`` and ``keep_gate`` with default shapes ``[B, 16, 512]``
            and ``[B, 16, 1]`` respectively.
        Raises:
            ValueError: If inputs do not match configured writer geometry.
        """
        self._validate_inputs(camera_features, patch_features, read_memory)
        camera_context, patch_context = self._build_contexts(camera_features, patch_features, read_memory)
        type_indices = torch.arange(2, device=read_memory.device).repeat_interleave(self.memory_slots_per_type)
        write_tokens = self.write_queries.expand(read_memory.shape[0], -1, -1) + self.query_type_embedding[type_indices] + self.slot_embedding
        for block in self.blocks:
            write_tokens = block(write_tokens, camera_context, patch_context)
        candidate = self.candidate_mlp(write_tokens)
        keep_gate = torch.sigmoid(self.gate_mlp(torch.cat((read_memory, candidate), dim=-1)))
        return keep_gate * read_memory + (1.0 - keep_gate) * candidate, keep_gate
