# ruff: noqa: E402

from copy import deepcopy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from launch import load_config
from data.episode import (
    FRAME_INDEXED_FIELDS,
    normalize_segment,
    split_episode,
    validate_segment_dimensions,
)


def _make_episode() -> dict[str, object]:
    """Create one CPU episode with frame-unique values and segment-unique scales."""
    frames = 24
    frame_ids = torch.arange(frames).view(1, frames)
    images = frame_ids.float().view(1, frames, 1, 1, 1).expand(-1, -1, 3, 2, 2).clone()
    extrinsics = torch.zeros(1, frames, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(1, frames, -1, -1).clone()
    scales = torch.tensor([2.0] * 8 + [4.0] * 8 + [8.0] * 8).view(1, frames, 1, 1, 1)
    world_points = torch.zeros(1, frames, 2, 2, 3)
    world_points[..., 0] = scales.squeeze(-1)
    cam_points = world_points.clone()
    depths = scales.squeeze(-1).expand(-1, -1, 2, 2).clone()
    return {
        "seq_name": ["synthetic-sequence"],
        "episode_metadata": {"camera": "left"},
        "ids": frame_ids,
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "cam_points": cam_points,
        "world_points": world_points,
        "point_masks": torch.ones(1, frames, 2, 2, dtype=torch.bool),
    }


def test_streaming_config_composes_with_fixed_values():
    """The experiment config exposes the fixed streaming schedule and head flags."""
    cfg = load_config("e01a_frozen_aggregator_streaming")

    assert cfg.exp_name == "e01a"
    assert cfg.img_size == 518
    assert dict(cfg.e01a) == {
        "experiment_id": "E01a",
        "total_frames": 24,
        "segment_frames": 8,
        "num_segments": 3,
        "backprop_mode": "full",
        "memory_enabled": True,
        "frozen_aggregator": True,
        "batch_size": 1,
        "episode_batch_size": 1,
    }
    assert cfg.accum_steps == 1
    assert (cfg.model.enable_camera, cfg.model.enable_depth) == (True, True)
    assert (cfg.model.enable_point, cfg.model.enable_track) == (False, False)

def test_load_config_rejects_inconsistent_e01a_dimensions():
    """The launch loading boundary rejects invalid E01a overrides before training."""
    with pytest.raises(
        ValueError,
        match="total_frames must equal segment_frames \\* num_segments",
    ):
        load_config(
            "e01a_frozen_aggregator_streaming",
            overrides=[
                "e01a.total_frames=24",
                "e01a.segment_frames=6",
                "e01a.num_segments=3",
            ],
        )


def test_load_config_preserves_default_composition():
    """The launch loading boundary leaves the default configuration unchanged."""
    cfg = load_config("default")

    assert cfg.exp_name == "exp001"



def test_dimension_validator_accepts_exact_partitions_and_rejects_others():
    """Validation depends on the dimension equation rather than literal values."""
    validate_segment_dimensions(total_frames=24, segment_frames=8, num_segments=3)
    validate_segment_dimensions(total_frames=12, segment_frames=3, num_segments=4)

    with pytest.raises(ValueError, match="total_frames"):
        validate_segment_dimensions(total_frames=24, segment_frames=7, num_segments=3)
    with pytest.raises(ValueError, match="positive"):
        validate_segment_dimensions(total_frames=0, segment_frames=8, num_segments=3)


def test_split_episode_slices_every_contracted_field_in_strict_order():
    """All contracted fields preserve the three ordered, non-overlapping ranges."""
    segments = split_episode(_make_episode())

    assert len(segments) == 3
    for index, segment in enumerate(segments):
        start = index * 8
        assert segment["segment_index"] == index
        assert (segment["frame_start"], segment["frame_stop"]) == (start, start + 8)
        torch.testing.assert_close(
            segment["ids"], torch.arange(start, start + 8).view(1, 8)
        )
        for field in FRAME_INDEXED_FIELDS:
            assert segment[field].shape[1] == 8


def test_split_episode_accepts_other_exact_partitions():
    """The splitter derives its segment count from valid caller dimensions."""
    episode = _make_episode()
    for field in FRAME_INDEXED_FIELDS:
        episode[field] = episode[field][:, :12].clone()

    segments = split_episode(episode, total_frames=12, segment_frames=3)

    assert len(segments) == 4
    assert [segment["segment_index"] for segment in segments] == [0, 1, 2, 3]
    assert [
        (segment["frame_start"], segment["frame_stop"]) for segment in segments
    ] == [(0, 3), (3, 6), (6, 9), (9, 12)]


def test_normalization_is_segment_local_and_keeps_first_camera_identity():
    """Each segment's valid points alone establish unit scale and its camera origin."""
    normalized = [
        normalize_segment(segment) for segment in split_episode(_make_episode())
    ]

    for segment in normalized:
        expected = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        )
        torch.testing.assert_close(
            segment["extrinsics"][:, 0], expected, atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            segment["world_points"].norm(dim=-1),
            torch.ones(1, 8, 2, 2),
            atol=1e-4,
            rtol=0,
        )
        assert all(
            not value.is_cuda for value in segment.values() if torch.is_tensor(value)
        )


def test_future_and_mutation_isolation():
    """Future changes and returned-segment mutations cannot affect segment zero or raw input."""
    baseline = _make_episode()
    future_changed = deepcopy(baseline)
    future_changed["images"][:, 8:] += 99
    future_changed["world_points"][:, 8:] *= 100

    baseline_zero = normalize_segment(split_episode(baseline)[0])
    changed_segments = split_episode(future_changed)
    changed_zero = normalize_segment(changed_segments[0])
    for field in FRAME_INDEXED_FIELDS:
        torch.testing.assert_close(baseline_zero[field], changed_zero[field])

    raw_images = baseline["images"].clone()
    split = split_episode(baseline)
    split[0]["images"].zero_()
    split[0]["episode_metadata"]["camera"] = "changed"
    assert torch.equal(baseline["images"], raw_images)
    assert split[1]["images"].any()
    assert baseline["episode_metadata"]["camera"] == "left"
    assert split[1]["episode_metadata"]["camera"] == "left"
