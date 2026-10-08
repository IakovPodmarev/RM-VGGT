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
    from training.data.episode import (
        FRAME_INDEXED_FIELDS,
        normalize_segment,
        split_episode,
    )
    from training.data.datasets.vkitti_sequence import SequentialVKittiEpisodeSource
    from training.recurrent_trainer import iter_prepared_segments


RawEpisode = Mapping[str, Any]
EpisodeSource = Callable[[int], Iterable[RawEpisode]]


def inspect_sequential_episodes(
    train_source: EpisodeSource,
    validation_source: EpisodeSource,
    *,
    total_frames: int,
    segment_frames: int,
    num_segments: int,
    epoch: int = 0,
) -> dict[str, dict[str, Any]]:
    """Inspect one training and one validation episode without training.

    The report gives stream identity, frame range, shapes, finite targets, and
    segment metadata for the configured frame counts. Each segment is checked
    against independent use of the
    existing CPU normalization helper. Empty or malformed sources raise
    ValueError. No model, logger, optimizer, or dataset writer is constructed.
    """
    report: dict[str, dict[str, Any]] = {}
    for phase, source in (("train", train_source), ("validation", validation_source)):
        raw = next(iter(source(epoch)), None)
        if raw is None:
            raise ValueError(f"{phase} source has no complete episode")
        ids = raw["ids"]
        if ids.shape != (1, total_frames) or not torch.equal(
            ids[0, 1:] - ids[0, :-1], torch.ones(total_frames - 1, dtype=ids.dtype)
        ):
            raise ValueError(
                f"{phase} episode must have {total_frames} strictly consecutive frame IDs"
            )
        image_size = getattr(source, "image_size", 518)
        if raw["images"].shape != (1, total_frames, 3, image_size, image_size):
            raise ValueError(f"{phase} episode has unexpected deterministic image size")
        repeated = next(iter(source(epoch)), None)
        if repeated is None or repeated["seq_name"] != raw["seq_name"]:
            raise ValueError(f"{phase} episode identity is not stable")
        for field in ("ids", "images", "depths", "extrinsics", "intrinsics"):
            if not torch.equal(raw[field], repeated[field]):
                raise ValueError(
                    f"{phase} episode processing is not deterministic for {field}"
                )
        for field in (
            "depths",
            "extrinsics",
            "intrinsics",
            "cam_points",
            "world_points",
        ):
            if not torch.isfinite(raw[field]).all():
                raise ValueError(f"{phase} episode has nonfinite {field}")
        raw_segments = split_episode(
            raw, total_frames=total_frames, segment_frames=segment_frames
        )
        prepared = list(
            iter_prepared_segments(
                raw,
                device=torch.device("cpu"),
                total_frames=total_frames,
                segment_frames=segment_frames,
            )
        )
        if len(raw_segments) != num_segments or len(prepared) != num_segments:
            raise ValueError(f"{phase} episode must prepare {num_segments} segments")
        identity = raw["seq_name"]
        for index, (segment, ready) in enumerate(zip(raw_segments, prepared)):
            if segment["segment_index"] != index or ready["segment_index"] != index:
                raise ValueError(f"{phase} segment order is invalid")
            if segment["seq_name"] != identity or ready["seq_name"] != identity:
                raise ValueError(f"{phase} batch identity changed between segments")
            if any(
                segment[field].shape[1] != segment_frames
                for field in FRAME_INDEXED_FIELDS if field in segment
            ):
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
            "segment_frame_counts": [
                segment["images"].shape[1] for segment in prepared
            ],
            "independently_normalized": True,
            "deterministic_processing": True,
        }
    return report


def main() -> None:
    """Inspect the resolved train and validation sources without training.

    Use the configured scene splits, start strides, seed, frame count, and
    image size so this CLI reports the same episode schedule as the trainer.
    """
    parser = argparse.ArgumentParser(description="Inspect sequential VKITTI episodes")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="e01a_frozen_aggregator_streaming")
    args = parser.parse_args()
    try:
        from launch import load_config
    except ModuleNotFoundError:
        from training.launch import load_config
    cfg = load_config(args.config)
    sequence = cfg.sequence
    train = SequentialVKittiEpisodeSource(
        args.dataset_root,
        cfg.dataset_split.train,
        training=True,
        seed=cfg.seed_value,
        total_frames=sequence.total_frames,
        episodes_windows_stride=cfg.episode_sources.train.episodes_windows_stride,
        image_size=cfg.img_size,
    )
    validation = SequentialVKittiEpisodeSource(
        args.dataset_root,
        cfg.dataset_split.validation,
        training=False,
        seed=cfg.seed_value,
        total_frames=sequence.total_frames,
        episodes_windows_stride=cfg.episode_sources.validation.episodes_windows_stride,
        image_size=cfg.img_size,
    )
    for phase, details in inspect_sequential_episodes(
        train,
        validation,
        total_frames=sequence.total_frames,
        segment_frames=sequence.segment_frames,
        num_segments=sequence.num_segments,
    ).items():
        print(f"{phase}: {details}")


if __name__ == "__main__":
    main()
