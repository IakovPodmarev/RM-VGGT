"""Inspect raw sequential episodes through the existing preparation boundary."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from data.episode import FRAME_INDEXED_FIELDS, normalize_segment, split_episode
    from data.datasets.vkitti_sequence import SequentialVKittiEpisodeSource
    from recurrent_trainer import iter_prepared_segments
except ModuleNotFoundError:
    from training.data.episode import FRAME_INDEXED_FIELDS, normalize_segment, split_episode
    from training.data.datasets.vkitti_sequence import SequentialVKittiEpisodeSource
    from training.recurrent_trainer import iter_prepared_segments


RawEpisode = Mapping[str, Any]
EpisodeSource = Callable[[int], Iterable[RawEpisode]]


def inspect_sequential_episodes(
    train_source: EpisodeSource,
    validation_source: EpisodeSource,
    *,
    epoch: int = 0,
) -> dict[str, dict[str, Any]]:
    """Inspect one training and one validation episode without training.

    The report gives stream identity, frame range, shapes, finite targets, and
    segment metadata. Each segment is checked against independent use of the
    existing CPU normalization helper. Empty or malformed sources raise
    ValueError. No model, logger, optimizer, or dataset writer is constructed.
    """
    report: dict[str, dict[str, Any]] = {}
    for phase, source in (("train", train_source), ("validation", validation_source)):
        raw = next(iter(source(epoch)), None)
        if raw is None:
            raise ValueError(f"{phase} source has no complete episode")
        ids = raw["ids"]
        if ids.shape != (1, 24) or not torch.equal(ids[0, 1:] - ids[0, :-1], torch.ones(23, dtype=ids.dtype)):
            raise ValueError(f"{phase} episode must have 24 strictly consecutive frame IDs")
        image_size = getattr(source, "image_size", 518)
        if raw["images"].shape != (1, 24, 3, image_size, image_size):
            raise ValueError(f"{phase} episode has unexpected deterministic image size")
        repeated = next(iter(source(epoch)), None)
        if repeated is None or repeated["seq_name"] != raw["seq_name"]:
            raise ValueError(f"{phase} episode identity is not stable")
        for field in ("ids", "images", "depths", "extrinsics", "intrinsics"):
            if not torch.equal(raw[field], repeated[field]):
                raise ValueError(f"{phase} episode processing is not deterministic for {field}")
        for field in ("depths", "extrinsics", "intrinsics", "cam_points", "world_points"):
            if not torch.isfinite(raw[field]).all():
                raise ValueError(f"{phase} episode has nonfinite {field}")
        raw_segments = split_episode(raw)
        prepared = list(iter_prepared_segments(raw, device=torch.device("cpu")))
        if len(raw_segments) != 3 or len(prepared) != 3:
            raise ValueError(f"{phase} episode must prepare three segments")
        identity = raw["seq_name"]
        for index, (segment, ready) in enumerate(zip(raw_segments, prepared)):
            if segment["segment_index"] != index or ready["segment_index"] != index:
                raise ValueError(f"{phase} segment order is invalid")
            if segment["seq_name"] != identity or ready["seq_name"] != identity:
                raise ValueError(f"{phase} batch identity changed between segments")
            if any(segment[field].shape[1] != 8 for field in FRAME_INDEXED_FIELDS):
                raise ValueError(f"{phase} segment has an unsliced frame field")
            expected = normalize_segment(segment)
            for field in ("extrinsics", "cam_points", "world_points", "depths"):
                torch.testing.assert_close(ready[field], expected[field])
        report[phase] = {
            "identity": identity[0],
            **raw["episode_metadata"],
            "start_frame_id": int(ids[0, 0]),
            "end_frame_id": int(ids[0, -1]),
            "frame_ids": ids[0].tolist(),
            "image_shape": list(raw["images"].shape),
            "depth_shape": list(raw["depths"].shape),
            "extrinsic_shape": list(raw["extrinsics"].shape),
            "intrinsic_shape": list(raw["intrinsics"].shape),
            "segment_indices": [segment["segment_index"] for segment in prepared],
            "segment_frame_counts": [segment["images"].shape[1] for segment in prepared],
            "independently_normalized": True,
            "deterministic_processing": True,
        }
    return report


def main() -> None:
    """Inspect a configured VKITTI root from argparse without launching training."""
    parser = argparse.ArgumentParser(description="Inspect sequential VKITTI episodes")
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    train = SequentialVKittiEpisodeSource(
        args.dataset_root, ("Scene01", "Scene02", "Scene06", "Scene18"),
        training=True, seed=42,
    )
    validation = SequentialVKittiEpisodeSource(
        args.dataset_root, ("Scene20",), training=False, seed=42,
    )
    for phase, details in inspect_sequential_episodes(train, validation).items():
        print(f"{phase}: {details}")


if __name__ == "__main__":
    main()
