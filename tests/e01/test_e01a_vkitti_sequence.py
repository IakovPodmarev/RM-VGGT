"""Focused contracts for the sequential VKITTI episode source."""

from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.datasets.vkitti_sequence import SequentialVKittiEpisodeSource
from data.episode import FRAME_INDEXED_FIELDS, split_episode
from inspect_sequential_episodes import inspect_sequential_episodes
from launch import load_config
from training.recurrent_trainer import RecurrentTrainer, iter_prepared_segments


def write_stream(
    root: Path,
    scene: str,
    variation: str,
    camera: int,
    ids: list[int],
    *,
    missing_depth: set[int] | None = None,
    missing_intrinsic: set[int] | None = None,
    duplicate_intrinsic: int | None = None,
    nonfinite_intrinsic: int | None = None,
) -> None:
    """Create a small VKITTI stream whose files and rows use real frame IDs."""
    missing_depth = missing_depth or set()
    missing_intrinsic = missing_intrinsic or set()
    base = root / scene / variation
    rgb = base / "frames" / "rgb" / f"Camera_{camera}"
    depth = base / "frames" / "depth" / f"Camera_{camera}"
    rgb.mkdir(parents=True)
    depth.mkdir(parents=True)
    extrinsics, intrinsics = [], []
    for frame_id in ids:
        image = np.full((30, 36, 3), frame_id, dtype=np.uint8)
        depth_map = np.full((30, 36), 250, dtype=np.uint16)
        assert cv2.imwrite(str(rgb / f"rgb_{frame_id:05d}.jpg"), image)
        if frame_id not in missing_depth:
            assert cv2.imwrite(str(depth / f"depth_{frame_id:05d}.png"), depth_map)
        matrix = np.eye(4)
        matrix[0, 3] = frame_id + camera * 1000
        extrinsics.append([frame_id, camera, *matrix.ravel()])
        if frame_id not in missing_intrinsic:
            fx = np.nan if frame_id == nonfinite_intrinsic else 30.0 + camera * 10.0
            intrinsics.append([frame_id, camera, fx, 31.0, 18.0, 15.0])
            if frame_id == duplicate_intrinsic:
                intrinsics.append([frame_id, camera, 30.0, 31.0, 18.0, 15.0])
    for name, rows, header in (
        ("extrinsic.txt", extrinsics, "frame camera matrix"),
        ("intrinsic.txt", intrinsics, "frame camera fx fy cx cy"),
    ):
        path = base / name
        values = np.asarray(rows)
        if path.exists():
            values = np.concatenate((np.loadtxt(path, skiprows=1, ndmin=2), values))
        np.savetxt(path, values, header=header, comments="")


def source(root: Path | None, scenes: list[str], training: bool) -> SequentialVKittiEpisodeSource:
    """Construct the small deterministic source shared by source-boundary tests."""
    return SequentialVKittiEpisodeSource(root, scenes, training=training, seed=17, total_frames=24, image_size=14)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Populate sorted train and validation streams with overlapping windows."""
    for scene in ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20"):
        for variation in ("clone", "fog"):
            for camera in (0, 1):
                write_stream(tmp_path, scene, variation, camera, list(range(30)))
    return tmp_path


def identities(item: SequentialVKittiEpisodeSource, epoch: int) -> list[str]:
    """Collect source order while avoiding retention of raw tensor payloads."""
    return [episode["seq_name"][0] for episode in item(epoch)]


def test_config_composes_source_targets() -> None:
    """Configured sources use the capability-named production target."""
    cfg = load_config("e01a_frozen_aggregator_streaming")

    assert cfg.episode_sources.train._target_ == "data.datasets.vkitti_sequence.SequentialVKittiEpisodeSource"
    assert cfg.episode_sources.validation._target_ == "data.datasets.vkitti_sequence.SequentialVKittiEpisodeSource"
    assert (cfg.sequence.total_frames, cfg.sequence.segment_frames, cfg.sequence.num_segments) == (15, 5, 3)


def test_scene_filtering_ordering_and_cpu_raw_contract(root: Path) -> None:
    """Sources filter scenes, deterministically order windows, and preserve raw IDs."""
    train = source(root, ["Scene01", "Scene02", "Scene06", "Scene18"], True)
    validation = source(root, ["Scene20"], False)
    raw = next(iter(validation(0)))

    assert {item["episode_metadata"]["scene"] for item in train(0)} == {"Scene01", "Scene02", "Scene06", "Scene18"}
    assert {item["episode_metadata"]["scene"] for item in validation(0)} == {"Scene20"}
    assert identities(train, 3) == identities(train, 3)
    assert identities(train, 0) != identities(train, 1)
    assert identities(validation, 0) == identities(validation, 2)
    assert raw["images"].shape == (1, 24, 3, 14, 14)
    assert raw["depths"].shape == (1, 24, 14, 14)
    assert torch.all(raw["depths"][raw["depths"] > 0] == 2.5)
    assert raw["extrinsics"].shape == (1, 24, 3, 4)
    assert raw["intrinsics"].shape == (1, 24, 3, 3)
    assert raw["original_sizes"].shape == (1, 24, 2)
    assert raw["ids"].tolist() == [list(range(24))]
    assert len(raw["seq_name"]) == 1
    assert all(value.device.type == "cpu" for value in raw.values() if torch.is_tensor(value))


@pytest.mark.parametrize(
    "ids,missing_depth,missing_intrinsic,duplicate",
    [
        (list(range(23)), set(), set(), None),
        (list(range(12)) + list(range(13, 25)), set(), set(), None),
        (list(range(24)), {9}, set(), None),
        (list(range(24)), set(), {9}, None),
        (list(range(24)), set(), set(), 9),
    ],
)
def test_manifest_rejects_invalid_stream_local_windows(
    tmp_path: Path, ids: list[int], missing_depth: set[int], missing_intrinsic: set[int], duplicate: int | None
) -> None:
    """Gaps, incomplete files, and duplicate records cannot yield partial episodes."""
    write_stream(tmp_path, "Scene01", "clone", 0, ids, missing_depth=missing_depth, missing_intrinsic=missing_intrinsic, duplicate_intrinsic=duplicate)

    assert list(source(tmp_path, ["Scene01"], False)(0)) == []


def test_camera_identity_read_only_processing_and_split_boundary(root: Path) -> None:
    """Frame IDs drive camera lookup, no root writes occur, and every field splits."""
    before = sorted(path.relative_to(root) for path in root.rglob("*"))
    raw = next(iter(source(root, ["Scene20"], False)(0)))
    after = sorted(path.relative_to(root) for path in root.rglob("*"))
    same = next(iter(source(root, ["Scene20"], False)(9)))
    segments = split_episode(raw, total_frames=24, segment_frames=8)
    prepared = list(iter_prepared_segments(raw, device=torch.device("cpu"), total_frames=24, segment_frames=8))

    assert after == before
    assert torch.allclose(raw["extrinsics"][0, :, 0, 3], torch.arange(24, dtype=torch.float32))
    torch.testing.assert_close(raw["images"], same["images"])
    torch.testing.assert_close(raw["intrinsics"], same["intrinsics"])
    assert [segment["segment_index"] for segment in segments] == [0, 1, 2]
    assert all(segment["original_sizes"].shape[1] == 8 for segment in segments)
    assert all(segment[field].shape[1] == 8 for segment in segments for field in FRAME_INDEXED_FIELDS if field in segment)
    assert [segment["seq_name"] for segment in prepared] == [raw["seq_name"]] * 3


def test_errors_inspection_and_configured_source(root: Path, tmp_path: Path) -> None:
    """Roots and targets fail contextually while inspector uses no model boundary."""
    with pytest.raises((ValueError, FileNotFoundError), match="dataset root"):
        list(source(None, ["Scene01"], False)(0))
    with pytest.raises((ValueError, FileNotFoundError), match="does not exist"):
        list(source(Path("/definitely/not/a/vkitti/root"), ["Scene01"], False)(0))
    invalid_root = tmp_path / "invalid"
    write_stream(invalid_root, "Scene01", "clone", 0, list(range(24)), nonfinite_intrinsic=9)
    with pytest.raises(ValueError, match="Scene01.*clone.*Camera_0.*9"):
        list(source(invalid_root, ["Scene01"], False)(0))

    report = inspect_sequential_episodes(source(root, ["Scene01"], True), source(root, ["Scene20"], False), total_frames=24, segment_frames=8, num_segments=3)
    assert report["train"]["segment_indices"] == [0, 1, 2]
    assert report["train"]["deterministic_processing"] is True
    configured = RecurrentTrainer._configured_source(
        {"_target_": "data.datasets.vkitti_sequence.SequentialVKittiEpisodeSource", "dataset_root": str(root), "scenes": ["Scene20"], "training": False, "seed": 17, "total_frames": 24, "image_size": 14}
    )
    assert next(iter(configured(0)))["images"].shape[1] == 24

def test_both_camera_tables_are_selected_by_frame_and_camera(root: Path) -> None:
    """Both cameras are eligible and retain their distinct keyed extrinsics."""
    episodes = list(source(root, ["Scene20"], False)(0))
    camera_zero = next(item for item in episodes if item["episode_metadata"]["camera"] == "Camera_0")
    camera_one = next(item for item in episodes if item["episode_metadata"]["camera"] == "Camera_1")
    torch.testing.assert_close(camera_zero["extrinsics"][0, :, 0, 3], torch.arange(24, dtype=torch.float32))
    torch.testing.assert_close(camera_one["extrinsics"][0, :, 0, 3], torch.arange(1000, 1024, dtype=torch.float32))
    assert camera_one["intrinsics"][0, 0, 0, 0] > camera_zero["intrinsics"][0, 0, 0, 0]


def test_hydra_root_override_instantiates_both_sources(root: Path) -> None:
    """Resolved config overrides let the trainer consume both raw source callables."""
    cfg = load_config(
        "e01a_frozen_aggregator_streaming",
        overrides=[f"vkitti_root={root}", "img_size=14"],
    )
    specs = OmegaConf.to_container(cfg, resolve=True)["episode_sources"]
    train = RecurrentTrainer._configured_source(specs["train"])
    validation = RecurrentTrainer._configured_source(specs["validation"])
    assert train is not None and validation is not None
    assert next(iter(train(0)))["images"].shape == (1, cfg.sequence.total_frames, 3, 14, 14)
    assert next(iter(validation(0)))["episode_metadata"]["scene"] == "Scene20"


def test_partial_streams_cannot_cross_camera_or_variation(tmp_path: Path) -> None:
    """Complementary half streams remain invalid even when IDs join to 24."""
    write_stream(tmp_path, "Scene01", "clone", 0, list(range(12)))
    write_stream(tmp_path, "Scene01", "clone", 1, list(range(12, 24)))
    write_stream(tmp_path, "Scene01", "fog", 0, list(range(12, 24)))
    assert list(source(tmp_path, ["Scene01"], False)(0)) == []


def test_finite_camera_value_overflowing_float32_has_frame_context(tmp_path: Path) -> None:
    """A finite table value cannot become an infinite raw camera or point target."""
    write_stream(tmp_path, "Scene01", "clone", 0, list(range(24)))
    table = tmp_path / "Scene01" / "clone" / "extrinsic.txt"
    rows = np.loadtxt(table, skiprows=1, ndmin=2)
    rows[9, 5] = 1e40
    assert np.isfinite(rows).all()
    np.savetxt(table, rows, header="frame camera matrix", comments="")

    with pytest.raises(ValueError, match=r"float32 conversion Scene01/clone/Camera_0/9"):
        next(iter(source(tmp_path, ["Scene01"], False)(0)))



def test_active_config_selects_fifteen_frame_windows_and_three_prepared_segments(
    root: Path,
) -> None:
    """The configured proof-of-concept schedule reaches source and inspector."""
    cfg = load_config("e01a_frozen_aggregator_streaming")
    train = SequentialVKittiEpisodeSource(
        root, ["Scene01"], training=True, seed=17,
        total_frames=cfg.sequence.total_frames, image_size=14,
    )
    validation = SequentialVKittiEpisodeSource(
        root, ["Scene20"], training=False, seed=17,
        total_frames=cfg.sequence.total_frames, image_size=14,
    )
    report = inspect_sequential_episodes(
        train, validation, total_frames=cfg.sequence.total_frames,
        segment_frames=cfg.sequence.segment_frames,
        num_segments=cfg.sequence.num_segments,
    )
    for phase in ("train", "validation"):
        assert len(report[phase]["frame_ids"]) == 15
        assert report[phase]["segment_indices"] == [0, 1, 2]
        assert report[phase]["segment_frame_counts"] == [5, 5, 5]
