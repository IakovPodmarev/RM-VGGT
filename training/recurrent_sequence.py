"""Stateless orchestration for ordered recurrent segment processing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class RecurrentSequenceResult:
    """Outputs and ordered memory states from one recurrent sequence.

    Attributes:
        predictions: One prediction mapping per processed segment.
        diagnostics: One diagnostic mapping per processed segment.
        memory_states: Initial state followed by one outgoing state per segment.
    """

    predictions: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    memory_states: list[Tensor]

    @property
    def final_memory(self) -> Tensor:
        """Return the final outgoing state without copying or detaching it."""
        return self.memory_states[-1]


def run_recurrent_sequence(
    model: Any,
    segments: Iterable[Mapping[str, Any]],
    *,
    num_segments: int = 3,
    segment_frames: int = 8,
) -> RecurrentSequenceResult:
    """Process prepared segments in strict order with an external memory state.

    Args:
        model: Segment model exposing ``initial_memory(batch_size, device, dtype)``
            and supporting normal module invocation as
            ``model(images=..., read_memory=...)``.
        segments: Ordered iterable of independently prepared segment mappings.
            When ``seq_name`` identity metadata is supplied, it must contain
            one ordered identity per batch element and remain identical across
            all consumed mappings.
        num_segments: Required positive number of mappings to consume.
        segment_frames: Required positive image frame count per mapping.

    Returns:
        Per-segment predictions and diagnostics plus the initial and outgoing
        memory states in temporal order.

    Raises:
        TypeError: If the iterable, a yielded segment mapping, or its images
            are invalid.
        ValueError: If the schedule, a sized iterable length, segment geometry,
            tensor compatibility, batch identity, or supplied temporal metadata
            is invalid.

    Invariants:
        Sized length and schedule checks happen before consumption. Each yielded
        segment is validated immediately before its model call, so later input
        is not requested before earlier processing completes. The function is
        stateless: it creates one initial state per invocation, calls the model
        in input order, carries only returned memory between calls, and never
        detaches, mutates, normalizes, casts, transfers, losses, backpropagates,
        or optimizes tensors.
    """
    _validate_schedule(num_segments, segment_frames)
    if isinstance(segments, (str, bytes, Mapping)) or not isinstance(segments, Iterable):
        raise TypeError("segments must be an ordered iterable of segment mappings")
    if isinstance(segments, Sized) and len(segments) != num_segments:
        raise ValueError(
            f"segments must contain exactly {num_segments} mappings, got {len(segments)}"
        )

    iterator = iter(segments)
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    memory_states: list[Tensor] = []
    reference_images: Tensor | None = None
    metadata_supplied: bool | None = None
    identity_supplied: bool | None = None
    reference_identity: tuple[str, ...] | None = None
    memory: Tensor | None = None

    for segment_position in range(num_segments):
        try:
            segment = next(iterator)
        except StopIteration as error:
            raise ValueError(
                f"segments must contain exactly {num_segments} mappings"
            ) from error
        images, metadata_supplied = _validate_segment(
            segment,
            segment_position=segment_position,
            segment_frames=segment_frames,
            reference_images=reference_images,
            metadata_supplied=metadata_supplied,
        )
        reference_identity, identity_supplied = _validate_batch_identity(
            segment,
            batch_size=images.shape[0],
            reference_identity=reference_identity,
            identity_supplied=identity_supplied,
        )
        if reference_images is None:
            reference_images = images
            memory = model.initial_memory(
                images.shape[0], device=images.device, dtype=images.dtype
            )
            memory_states.append(memory)

        predictions_for_segment, memory, diagnostics_for_segment = model(
            images=images,
            read_memory=memory,
        )
        predictions.append(predictions_for_segment)
        diagnostics.append(diagnostics_for_segment)
        memory_states.append(memory)

    return RecurrentSequenceResult(
        predictions=predictions,
        diagnostics=diagnostics,
        memory_states=memory_states,
    )


def _validate_schedule(num_segments: int, segment_frames: int) -> None:
    """Reject nonpositive or noninteger recurrent schedule dimensions.

    Args:
        num_segments: Requested number of ordered segment calls.
        segment_frames: Required frame count in every segment image tensor.

    Raises:
        ValueError: If either dimension is not a positive integer.
    """
    for name, value in (("num_segments", num_segments), ("segment_frames", segment_frames)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _validate_batch_identity(
    segment: Mapping[str, Any],
    *,
    batch_size: int,
    reference_identity: tuple[str, ...] | None,
    identity_supplied: bool | None,
) -> tuple[tuple[str, ...] | None, bool]:
    """Validate optional ordered batch identities without changing caller data.

    Args:
        segment: Validated prepared mapping for the current temporal position.
        batch_size: Number of image rows in the current mapping.
        reference_identity: Ordered identities accepted from the first segment.
        identity_supplied: Whether preceding segments supplied ``seq_name``.

    Returns:
        The reference identities and the established identity-presence flag.

    Raises:
        ValueError: If ``seq_name`` is inconsistently supplied, is not one
            string per batch element, or changes element identity or order.

    Invariants:
        Presence is optional for an entire sequence, but once supplied it is
        required and identical for every later segment.
    """
    has_identity = "seq_name" in segment
    if identity_supplied is None:
        identity_supplied = has_identity
    elif identity_supplied != has_identity:
        raise ValueError("batch identity metadata must be supplied consistently")
    if not has_identity:
        return reference_identity, identity_supplied

    identities = segment["seq_name"]
    if (
        isinstance(identities, (str, bytes))
        or not isinstance(identities, Sequence)
        or len(identities) != batch_size
        or not all(isinstance(identity, str) for identity in identities)
    ):
        raise ValueError("batch identity metadata must contain one string per batch element")
    current_identity = tuple(identities)
    if reference_identity is None:
        return current_identity, identity_supplied
    if current_identity != reference_identity:
        raise ValueError("batch identity metadata must preserve element identity and order")
    return reference_identity, identity_supplied


def _validate_segment(
    segment: object,
    *,
    segment_position: int,
    segment_frames: int,
    reference_images: Tensor | None,
    metadata_supplied: bool | None,
) -> tuple[Tensor, bool]:
    """Validate one just-in-time segment before its model invocation.

    Args:
        segment: Candidate prepared mapping yielded by the caller.
        segment_position: Zero-based temporal position in the consumed stream.
        segment_frames: Required frame count.
        reference_images: Images from the first accepted segment, if any.
        metadata_supplied: Whether preceding segments supplied complete metadata.

    Returns:
        The original images tensor and the established metadata-presence flag.

    Raises:
        TypeError: If the mapping or images tensor is missing or invalid.
        ValueError: If geometry, device, dtype, or temporal metadata differs
            from the accepted stream contract.

    Invariants:
        This function never copies, casts, transfers, or mutates caller data.
    """
    if not isinstance(segment, Mapping):
        raise TypeError(f"segment {segment_position} must be a mapping")
    if "images" not in segment:
        raise TypeError(f"segment {segment_position} is missing images")
    images = segment["images"]
    if not torch.is_tensor(images):
        raise TypeError(f"segment {segment_position} images must be a tensor")
    if images.ndim != 5:
        raise ValueError("segment images must be rank 5 [B, frames, channels, height, width]")
    if images.shape[1] != segment_frames:
        raise ValueError(f"segment images must contain {segment_frames} frames")
    if images.shape[2] != 3:
        raise ValueError("segment images must have three channels")
    if reference_images is not None:
        if images.shape[0] != reference_images.shape[0]:
            raise ValueError("segment image batch size must remain unchanged")
        if images.shape[3:] != reference_images.shape[3:]:
            raise ValueError("segment image spatial size must remain unchanged")
        if images.device != reference_images.device:
            raise ValueError("segment image device must remain unchanged")
        if images.dtype != reference_images.dtype:
            raise ValueError("segment image dtype must remain unchanged")

    metadata_fields = ("segment_index", "frame_start", "frame_stop")
    present_fields = tuple(field in segment for field in metadata_fields)
    has_any_metadata = any(present_fields)
    has_complete_metadata = all(present_fields)
    if has_any_metadata and not has_complete_metadata:
        raise ValueError("segment metadata must include segment_index, frame_start, and frame_stop")
    if metadata_supplied is None:
        metadata_supplied = has_complete_metadata
    elif metadata_supplied != has_complete_metadata:
        raise ValueError("segment metadata must be supplied consistently for every segment")
    if has_complete_metadata:
        expected_range = (
            segment_position,
            segment_position * segment_frames,
            (segment_position + 1) * segment_frames,
        )
        actual_range = tuple(segment[field] for field in metadata_fields)
        if actual_range != expected_range:
            raise ValueError("segment metadata must match the ordered contiguous schedule")
    return images, metadata_supplied
