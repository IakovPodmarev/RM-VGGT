"""Complete recurrent episode forward for optional DDP wrapping."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from torch import Tensor, nn

from training.recurrent_loss import RecurrentLossResult, compute_recurrent_losses
from training.recurrent_sequence import RecurrentSequenceResult, run_recurrent_sequence


class RecurrentEpisodeModule(nn.Module):
    """Expose one differentiable full-episode objective to DDP.

    The underlying segment model retains its original parameter names when
    used directly for optimizer groups, clipping, and checkpoints.
    """

    def __init__(self, model: nn.Module, *, num_segments: int, segment_frames: int) -> None:
        """Keep one segment model and its required recurrent schedule."""
        super().__init__()
        self.segment_model = model
        self.num_segments = num_segments
        self.segment_frames = segment_frames

    def forward(
        self,
        segments: Iterable[Mapping[str, Any]],
        loss_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Tensor]],
    ) -> tuple[Tensor, RecurrentSequenceResult, RecurrentLossResult, list[Mapping[str, Any]]]:
        """Consume segments lazily, carry memory, and return their mean loss.

        The first tensor is the complete episode objective visible to DDP.
        Sequence, losses, and consumed targets remain connected for the caller;
        this method performs no backward, clipping, or optimizer action.
        Schedule and segment validation are delegated to the existing helpers.
        """
        recorded: list[Mapping[str, Any]] = []

        def stream() -> Iterable[Mapping[str, Any]]:
            """Record each mapping only as sequence execution requests it."""
            for segment in segments:
                recorded.append(segment)
                yield segment

        sequence = run_recurrent_sequence(
            self.segment_model, stream(),
            num_segments=self.num_segments, segment_frames=self.segment_frames,
        )
        losses = compute_recurrent_losses(sequence, recorded, loss_fn, num_segments=self.num_segments)
        return losses.objective, sequence, losses, recorded
