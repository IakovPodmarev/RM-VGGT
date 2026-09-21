"""E01a recurrent-memory read adaptor for final VGGT camera features."""

from __future__ import annotations

from torch import Tensor

from ._read_adaptor import _ReadAdaptorBase


class CameraReadAdaptor(_ReadAdaptorBase):
    """Expose all incoming memory slots to one camera token per frame.

    The module preserves the camera head's ``[B, S, feature_dim]`` layout and
    does not invoke a camera head or mutate the supplied features or memory.
    """

    def forward(self, camera_features: Tensor, read_memory: Tensor) -> Tensor:
        """Return camera features with a gated E01a memory-attention update.

        Args:
            camera_features: Features shaped ``[B, S, 2048]`` by default.
            read_memory: Incoming state shaped ``[B, 16, 512]`` by default.

        Returns:
            Adapted camera features with the same shape as ``camera_features``.

        Raises:
            ValueError: If ranks, frame count, feature width, memory shape,
                device, or dtype are incompatible with this adaptor.
        """
        if camera_features.ndim != 3:
            raise ValueError("camera_features must be rank 3 [B, frames, feature_dim]")
        if camera_features.shape[1] != self.num_frames:
            raise ValueError(f"camera_features num_frames must equal {self.num_frames}")
        if camera_features.shape[2] != self.feature_dim:
            raise ValueError(f"camera_features feature_dim must equal {self.feature_dim}")
        self._validate_common(camera_features, read_memory)
        return self._adapt_flattened(camera_features, read_memory)
