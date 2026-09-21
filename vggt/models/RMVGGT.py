"""RM-VGGT frozen-aggregator recurrent-memory model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.models.aggregator import Aggregator
from vggt.rm_adaptor.camera_read_adaptor import CameraReadAdaptor
from vggt.rm_adaptor.depth_read_adaptor import DepthReadAdaptor
from vggt.rm_adaptor.memory_writer import MemoryWriter


class RMVGGT(nn.Module):
    """Compose frozen VGGT features with recurrent memory readers and writer.

    The model processes exactly one normalized segment at a time and owns the
    aggregator, prediction heads, writer, and two read adaptors. It does not
    retain recurrent state: callers supply incoming memory and receive outgoing
    memory separately.
    """

    def __init__(self, aggregator: nn.Module | None = None, camera_head: nn.Module | None = None, depth_head: nn.Module | None = None, memory_writer: MemoryWriter | None = None, camera_read_adaptor: CameraReadAdaptor | None = None, depth_read_adaptor: DepthReadAdaptor | None = None, *, img_size: int = 518, patch_size: int = 14, embed_dim: int = 1024) -> None:
        """Initialize injected or default modules and permanently freeze encoding.

        Default modules preserve ``aggregator.*``, ``camera_head.*``, and
        ``depth_head.*`` state-dict prefixes. Point and tracking heads are not
        constructed. The aggregator is always evaluation-mode and frozen.
        """
        super().__init__()
        feature_dim = embed_dim * 2
        self.aggregator = aggregator or Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = camera_head or CameraHead(dim_in=feature_dim)
        self.depth_head = depth_head or DPTHead(dim_in=feature_dim, output_dim=2, activation="exp", conf_activation="expp1", patch_size=patch_size)
        self.memory_writer = memory_writer or MemoryWriter(feature_dim=feature_dim)
        self.camera_read_adaptor = camera_read_adaptor or CameraReadAdaptor(feature_dim=feature_dim)
        self.depth_read_adaptor = depth_read_adaptor or DepthReadAdaptor(feature_dim=feature_dim)
        for parameter in self.aggregator.parameters():
            parameter.requires_grad_(False)
        self.aggregator.eval()

    def train(self, mode: bool = True) -> "RMVGGT":
        """Set trainable modules' mode while permanently retaining aggregator eval mode."""
        super().train(mode)
        self.aggregator.eval()
        return self

    def initial_memory(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Delegate independent initial recurrent state creation to the writer."""
        return self.memory_writer.initial_memory(batch_size, device=device, dtype=dtype)

    def _validate_cache(self, cached_features: list[Tensor | None], images: Tensor, patch_start_idx: int, read_memory: Tensor) -> Tensor:
        """Validate sparse cache, segment geometry, and recurrent tensor compatibility.

        Raises:
            ValueError: If required cache entries, rank, batch/frame geometry,
            patch start, patch count, memory shape, device, or dtype are invalid.
        """
        if not isinstance(cached_features, list) or len(cached_features) <= 23:
            raise ValueError("cached_features must contain layer 23")
        for index in (4, 11, 17, 23):
            if cached_features[index] is None:
                raise ValueError(f"cached_features layer {index} is required")
        if images.ndim != 5:
            raise ValueError("images must be rank 5 [B, frames, channels, height, width]")
        layer23 = cached_features[23]
        if layer23.ndim != 4:
            raise ValueError("layer 23 cache must be rank 4 [B, frames, tokens, feature_dim]")
        if layer23.shape[:2] != images.shape[:2]:
            raise ValueError("cache and images batch/frame counts must match")
        if not isinstance(patch_start_idx, int) or isinstance(patch_start_idx, bool) or not 0 < patch_start_idx < layer23.shape[2]:
            raise ValueError("patch_start_idx must select a nonempty patch token range")
        if layer23.shape[-1] != self.memory_writer.feature_dim:
            raise ValueError("layer 23 feature width must match memory writer feature_dim")
        if layer23.shape[2] - patch_start_idx != self.memory_writer.patch_grid_size**2:
            raise ValueError("layer 23 patch count is incompatible with memory writer")
        if read_memory.ndim != 3 or read_memory.shape[1:] != (self.memory_writer.total_memory_slots, self.memory_writer.memory_dim):
            raise ValueError("read_memory shape is incompatible with memory writer")
        if layer23.device != images.device or read_memory.device != layer23.device:
            raise ValueError("cache, images, and read_memory device must match")
        if layer23.dtype != images.dtype or read_memory.dtype != layer23.dtype:
            raise ValueError("cache, images, and read_memory dtype must match")
        for name, module in (
            ("memory_writer", self.memory_writer),
            ("camera_read_adaptor", self.camera_read_adaptor),
            ("depth_read_adaptor", self.depth_read_adaptor),
            ("camera_head", self.camera_head),
            ("depth_head", self.depth_head),
        ):
            parameter = next(module.parameters(), None)
            if parameter is None:
                continue
            if layer23.device != parameter.device:
                raise ValueError(f"cache device must match {name} parameters")
            if layer23.dtype != parameter.dtype:
                raise ValueError(f"cache dtype must match {name} parameters")
        return layer23

    def encode_segment(self, images: Tensor) -> tuple[list[Tensor | None], int]:
        """Run one segment through the frozen aggregator without autograd history.

        Args:
            images: Already-normalized segment shaped ``[B, S, 3, H, W]``.

        Returns:
            Sparse cached feature list and patch-token start index.

        Raises:
            ValueError: If image rank or aggregator cache structure is invalid.
        """
        if images.ndim != 5:
            raise ValueError("images must be rank 5 [B, frames, channels, height, width]")
        self.aggregator.eval()
        with torch.no_grad():
            cached_features, patch_start_idx = self.aggregator(images)
        if not isinstance(cached_features, list) or len(cached_features) <= 23:
            raise ValueError("aggregator cache must contain layer 23")
        return cached_features, patch_start_idx

    def forward(
        self,
        images: Tensor,
        read_memory: Tensor,
    ) -> tuple[dict[str, Tensor | list[Tensor]], Tensor, dict[str, Tensor]]:
        """Process one prepared segment using explicitly supplied memory.

        Args:
            images: Prepared segment images shaped ``[B, S, 3, H, W]``.
            read_memory: Incoming recurrent state for this segment.

        Returns:
            Prediction mapping, outgoing recurrent state, and diagnostics from
            the lower-level segment computation.

        Invariants:
            This public boundary encodes exactly one segment and delegates its
            trainable computation to ``forward_segment``. It neither creates,
            stores, detaches, nor mutates recurrent memory, and performs no
            normalization, device transfer, loss, backward, or optimizer work.
        """
        cached_features, patch_start_idx = self.encode_segment(images)
        return self.forward_segment(
            cached_features,
            images,
            patch_start_idx,
            read_memory,
        )

    def forward_segment(self, cached_features: list[Tensor | None], images: Tensor, patch_start_idx: int, read_memory: Tensor) -> tuple[dict[str, Tensor | list[Tensor]], Tensor, dict[str, Tensor]]:
        """Predict one segment and return outgoing memory plus gate diagnostics.

        Current prediction heads consume adapted features conditioned only on
        incoming memory. The writer receives original layer-23 features and
        returns outgoing memory that is not passed to either prediction head.
        """
        layer23 = self._validate_cache(cached_features, images, patch_start_idx, read_memory)
        camera_features = layer23[:, :, 0, :]
        patch_features = layer23[:, :, patch_start_idx:, :]
        next_memory, memory_keep_gate = self.memory_writer(camera_features, patch_features, read_memory)
        adapted_camera = self.camera_read_adaptor(camera_features, read_memory)
        adapted_patches = self.depth_read_adaptor(patch_features, read_memory)
        camera_cache = list(cached_features)
        camera_layer23 = layer23.clone()
        camera_layer23[:, :, 0, :] = adapted_camera
        camera_cache[23] = camera_layer23
        depth_cache = list(cached_features)
        depth_layer23 = layer23.clone()
        depth_layer23[:, :, patch_start_idx:, :] = adapted_patches
        depth_cache[23] = depth_layer23
        pose_enc_list = self.camera_head(camera_cache)
        depth, depth_conf = self.depth_head(depth_cache, images=images, patch_start_idx=patch_start_idx)
        predictions: dict[str, Tensor | list[Tensor]] = {
            "pose_enc": pose_enc_list[-1],
            "pose_enc_list": pose_enc_list,
            "depth": depth,
            "depth_conf": depth_conf,
        }
        diagnostics = {
            "memory_keep_gate": memory_keep_gate,
            "camera_residual_gate": torch.sigmoid(self.camera_read_adaptor.residual_gate),
            "depth_residual_gate": torch.sigmoid(self.depth_read_adaptor.residual_gate),
        }
        return predictions, next_memory, diagnostics
