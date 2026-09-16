"""CPU-only episode splitting and normalization for fixed streaming episodes."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


FRAME_INDEXED_FIELDS = (
    "ids",
    "images",
    "depths",
    "extrinsics",
    "intrinsics",
    "cam_points",
    "world_points",
    "point_masks",
)
"""Fields whose dimension 1 is the explicit batched frame axis."""

_REQUIRED_NORMALIZATION_FIELDS = (
    "extrinsics",
    "cam_points",
    "world_points",
    "depths",
    "point_masks",
)
_SEGMENT_METADATA_FIELDS = ("segment_index", "frame_start", "frame_stop")


def validate_segment_dimensions(
    total_frames: int,
    segment_frames: int,
    num_segments: int,
) -> None:
    """Validate a positive, exactly partitionable streaming schedule.

    Args:
        total_frames: Number of frames in one raw episode.
        segment_frames: Number of consecutive frames in one segment.
        num_segments: Expected number of non-overlapping segments.

    Returns:
        ``None`` when the dimensions are positive and partition exactly.

    Raises:
        ValueError: If a dimension is not a positive integer or if
            ``total_frames != segment_frames * num_segments``.

    Invariants:
        Callers invoke this after Hydra composition because composing a plain
        YAML config does not instantiate its ``_target_`` nodes.
    """
    dimensions = {
        "total_frames": total_frames,
        "segment_frames": segment_frames,
        "num_segments": num_segments,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in dimensions.values()
    ):
        raise ValueError(
            "total_frames, segment_frames, and num_segments must be integers"
        )
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError(
            "total_frames, segment_frames, and num_segments must be positive"
        )
    if total_frames != segment_frames * num_segments:
        raise ValueError(
            "total_frames must equal segment_frames * num_segments, got "
            f"{total_frames} != {segment_frames} * {num_segments}"
        )


def split_episode(
    raw_episode: Mapping[str, Any], total_frames: int = 24, segment_frames: int = 8
) -> list[dict[str, Any]]:
    """Split one raw CPU episode into ordered, isolated segment batches.

    Args:
        raw_episode: Batched CPU episode. Only fields in
            ``FRAME_INDEXED_FIELDS`` are sliced on dimension 1; all other
            fields are copied as episode metadata. Every contracted field must
            be a CPU tensor with ``total_frames`` entries on dimension 1.
        total_frames: Required episode length.
        segment_frames: Required non-overlapping segment length.

    Returns:
        New CPU dictionaries in strict temporal order. Each includes
        ``segment_index``, ``frame_start``, and exclusive ``frame_stop`` metadata.

    Raises:
        TypeError: If the episode is not a mapping or contracted fields are not
            tensors.
        ValueError: If dimensions, field lengths, metadata names, or CPU
            residency are invalid.

    Invariants:
        Splitting precedes normalization, device transfer, and model execution.
        The caller episode is never mutated, and returned segments share no
        mutable tensor or metadata state with it or each other.
    """
    if not isinstance(raw_episode, Mapping):
        raise TypeError("raw_episode must be a mapping")
    if not isinstance(total_frames, int) or not isinstance(segment_frames, int):
        raise ValueError("total_frames and segment_frames must be integers")
    if total_frames <= 0 or segment_frames <= 0 or total_frames % segment_frames:
        raise ValueError("total_frames must be a positive multiple of segment_frames")
    num_segments = total_frames // segment_frames
    validate_segment_dimensions(total_frames, segment_frames, num_segments)
    _validate_episode_fields(raw_episode, total_frames)
    if any(field in raw_episode for field in _SEGMENT_METADATA_FIELDS):
        raise ValueError(
            "raw_episode must not contain reserved segment metadata fields"
        )

    segments = []
    for segment_index in range(num_segments):
        frame_start = segment_index * segment_frames
        frame_stop = frame_start + segment_frames
        segment = {
            key: value[:, frame_start:frame_stop].clone()
            if key in FRAME_INDEXED_FIELDS
            else _copy_value(value)
            for key, value in raw_episode.items()
        }
        segment.update(
            segment_index=segment_index,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )
        segments.append(segment)
    return segments


def normalize_segment(raw_segment_batch: Mapping[str, Any]) -> dict[str, Any]:
    """Independently normalize one isolated CPU segment.

    Args:
        raw_segment_batch: A segment returned by ``split_episode`` with camera,
            point, depth, and validity-mask tensors on CPU.

    Returns:
        A new CPU segment with normalized extrinsics, camera points, world
        points, and depths. Other tensors and metadata are independently copied.

    Raises:
        TypeError: If the segment is not a mapping or required fields are not
            tensors.
        ValueError: If fields are missing, have inconsistent or empty frame
            dimensions, or any contained tensor is not on CPU.

    Invariants:
        ``normalize_camera_extrinsics_and_points_batch`` is the sole geometry
        implementation. Its first camera and valid-point scale are applied only
        to this segment; it cannot observe future-segment data.
    """
    if not isinstance(raw_segment_batch, Mapping):
        raise TypeError("raw_segment_batch must be a mapping")
    extrinsics = raw_segment_batch.get("extrinsics")
    if not torch.is_tensor(extrinsics) or extrinsics.ndim < 2:
        raise TypeError("raw_segment_batch field 'extrinsics' must be a batched tensor")
    frame_count = extrinsics.shape[1]
    if frame_count <= 0:
        raise ValueError("raw_segment_batch must contain at least one frame")
    _validate_episode_fields(raw_segment_batch, frame_count)
    missing = [
        key for key in _REQUIRED_NORMALIZATION_FIELDS if key not in raw_segment_batch
    ]
    if missing:
        raise ValueError(
            f"raw_segment_batch is missing normalization fields: {missing}"
        )

    normalized = {key: _copy_value(value) for key, value in raw_segment_batch.items()}
    extrinsics, cam_points, world_points, depths = (
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=normalized["extrinsics"],
            cam_points=normalized["cam_points"],
            world_points=normalized["world_points"],
            depths=normalized["depths"],
            point_masks=normalized["point_masks"],
        )
    )
    normalized.update(
        extrinsics=extrinsics,
        cam_points=cam_points,
        world_points=world_points,
        depths=depths,
    )
    return normalized


def _validate_episode_fields(episode: Mapping[str, Any], frame_count: int) -> None:
    """Validate the explicit temporal-field and CPU-residency contract."""
    missing = [key for key in FRAME_INDEXED_FIELDS if key not in episode]
    if missing:
        raise ValueError(f"episode is missing frame-indexed fields: {missing}")
    for key in FRAME_INDEXED_FIELDS:
        value = episode[key]
        if not torch.is_tensor(value):
            raise TypeError(f"episode field {key!r} must be a tensor")
        if value.ndim < 2:
            raise ValueError(f"episode field {key!r} must have a batched frame axis")
        if value.shape[1] != frame_count:
            raise ValueError(
                f"episode field {key!r} has {value.shape[1]} frames, expected {frame_count}"
            )
    _validate_cpu_residency(episode)


def _validate_cpu_residency(value: Any) -> None:
    """Raise when any tensor in a nested episode value is not CPU-resident."""
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError("episode preprocessing accepts CPU tensors only")
    elif isinstance(value, Mapping):
        for nested_value in value.values():
            _validate_cpu_residency(nested_value)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            _validate_cpu_residency(nested_value)


def _copy_value(value: Any) -> Any:
    """Clone tensors and deep-copy metadata to prevent mutable-state sharing."""
    return value.clone() if torch.is_tensor(value) else deepcopy(value)
