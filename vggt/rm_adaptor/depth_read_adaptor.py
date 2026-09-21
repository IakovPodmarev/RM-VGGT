"""E01a recurrent-memory read adaptor for final VGGT patch features."""

from __future__ import annotations

from torch import Tensor

from ._read_adaptor import _ReadAdaptorBase


class DepthReadAdaptor(_ReadAdaptorBase):
    """Expose all incoming memory slots without changing patch-token layout.

    Frame and patch axes are flattened only for cross-attention and restored in
    their exact frame-major, patch-major order before returning.
    """

    def forward(self, patch_features: Tensor, read_memory: Tensor) -> Tensor:
        """Return patch features with a gated E01a memory-attention update.

        Args:
            patch_features: Features shaped ``[B, S, N, 2048]`` by default.
            read_memory: Incoming state shaped ``[B, 16, 512]`` by default.

        Returns:
            Adapted patch features with the same shape and token order.

        Raises:
            ValueError: If ranks, frame count, feature width, memory shape,
                device, or dtype are incompatible with this adaptor.
        """
        if patch_features.ndim != 4:
            raise ValueError("patch_features must be rank 4 [B, frames, patches, feature_dim]")
        if patch_features.shape[1] != self.num_frames:
            raise ValueError(f"patch_features num_frames must equal {self.num_frames}")
        if patch_features.shape[3] != self.feature_dim:
            raise ValueError(f"patch_features feature_dim must equal {self.feature_dim}")
        self._validate_common(patch_features, read_memory)
        batch_size, frames, patches, channels = patch_features.shape
        flattened = patch_features.reshape(batch_size, frames * patches, channels)
        return self._adapt_flattened(flattened, read_memory).reshape_as(patch_features)
