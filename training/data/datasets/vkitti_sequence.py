"""Epoch-indexed, stream-local VKITTI episode loading."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import re

import cv2
import numpy as np
import torch

try:
    from data.dataset_util import (
        crop_image_depth_and_intrinsic_by_pp,
        depth_to_world_coords_points,
        read_image_cv2,
        resize_image_depth_and_intrinsic,
        threshold_depth_map,
    )
except ModuleNotFoundError:
    from training.data.dataset_util import (
        crop_image_depth_and_intrinsic_by_pp,
        depth_to_world_coords_points,
        read_image_cv2,
        resize_image_depth_and_intrinsic,
        threshold_depth_map,
    )


RawEpisode = dict[str, Any]
_RGB_NAME = re.compile(r"rgb_([0-9]+)\.jpg$")


@dataclass(frozen=True)
class VKittiEpisodeWindow:
    """Identify a complete window by its scene, variation, camera, and frame IDs."""

    scene: str
    variation: str
    camera_id: str
    start_frame_id: int
    frame_ids: tuple[int, ...]


class SequentialVKittiEpisodeSource:
    """Provide complete raw CPU episodes in deterministic epoch order.

    The root is inspected on first use. The manifest contains only windows
    whose RGB, depth, intrinsic, and extrinsic identities all agree. Construction
    and iteration do not write to the dataset or use global random state.
    Every emitted frame has a matched, finite camera label; missing camera rows
    exclude windows before loading, so an explicit validity mask is unnecessary.
    """

    def __init__(
        self,
        dataset_root: str | Path | None,
        scenes: Iterable[str],
        *,
        training: bool,
        seed: int,
        total_frames: int,
        image_size: int = 518,
    ) -> None:
        """Store source settings; reject invalid dimensions and empty scene sets.

        The configured positive total frame count defines complete windows.
        A null root is accepted here for Hydra composition. The first call
        raises an actionable error if the root is null or absent.
        """
        if not isinstance(total_frames, int) or isinstance(total_frames, bool) or total_frames <= 0:
            raise ValueError("total_frames must be a positive integer")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.scenes = tuple(scenes)
        if not self.scenes or any(not isinstance(scene, str) for scene in self.scenes):
            raise ValueError("scenes must contain at least one scene name")
        self.training = bool(training)
        self.seed = int(seed)
        self.total_frames = total_frames
        self.image_size = image_size
        self.manifest: tuple[VKittiEpisodeWindow, ...] | None = None
        self._camera_rows: dict[tuple[str, str, str], dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def __call__(self, epoch: int) -> Iterable[RawEpisode]:
        """Return a lazy iterable of complete episodes for an integer epoch.

        Training uses a local seed and epoch permutation. Validation always
        follows sorted manifest order. Malformed metadata raises with stream
        context, and invalid incomplete windows are omitted.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        windows = list(self._manifest())
        if self.training:
            random.Random(f"{self.seed}:{epoch}").shuffle(windows)
        return self._episodes(windows)

    def _episodes(self, windows: list[VKittiEpisodeWindow]) -> Iterator[RawEpisode]:
        """Load only windows requested by the caller, such as an islice prefix."""
        for window in windows:
            yield self._load(window)

    def _manifest(self) -> tuple[VKittiEpisodeWindow, ...]:
        """Build once from sorted scene, variation, camera, and start identities."""
        if self.manifest is not None:
            return self.manifest
        root = self.dataset_root
        if root is None:
            raise ValueError("dataset root is null; set vkitti_root to a VKITTI directory")
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        windows: list[VKittiEpisodeWindow] = []
        for scene in sorted(self.scenes):
            scene_path = root / scene
            if not scene_path.is_dir():
                continue
            for variation_path in sorted(path for path in scene_path.iterdir() if path.is_dir()):
                variation = variation_path.name
                rgb_root = variation_path / "frames" / "rgb"
                for camera_path in sorted(rgb_root.glob("Camera_*")):
                    if not camera_path.is_dir():
                        continue
                    camera = camera_path.name
                    suffix = camera.removeprefix("Camera_")
                    if not suffix.isdigit():
                        continue
                    rows, invalid = self._read_camera_rows(variation_path, scene, variation, camera, int(suffix))
                    depth_root = variation_path / "frames" / "depth" / camera
                    rgb_files: dict[int, Path] = {}
                    for image_path in camera_path.glob("rgb_*.jpg"):
                        match = _RGB_NAME.fullmatch(image_path.name)
                        if match is None:
                            continue
                        frame = int(match.group(1))
                        if frame in rgb_files:
                            invalid.add(frame)
                        else:
                            rgb_files[frame] = image_path
                    valid = {
                        frame for frame, image_path in rgb_files.items()
                        if frame not in invalid
                        and image_path.is_file()
                        and (depth_root / f"depth_{frame:05d}.png").is_file()
                        and frame in rows
                    }
                    for start in sorted(valid):
                        frame_ids = tuple(range(start, start + self.total_frames))
                        if all(frame in valid for frame in frame_ids):
                            windows.append(VKittiEpisodeWindow(scene, variation, camera, start, frame_ids))
                    self._camera_rows[(scene, variation, camera)] = rows
        self.manifest = tuple(sorted(
            windows,
            key=lambda item: (item.scene, item.variation, item.camera_id, item.start_frame_id),
        ))
        return self.manifest

    @staticmethod
    def _table(path: Path, context: str, width: int) -> np.ndarray:
        """Read a numeric camera table and reject missing or malformed columns."""
        if not path.is_file():
            raise FileNotFoundError(f"missing camera table {context}: {path}")
        try:
            rows = np.loadtxt(path, skiprows=1, ndmin=2)
        except (OSError, ValueError) as error:
            raise ValueError(f"malformed camera table {context}: {path}") from error
        if rows.shape[1] != width or not rows.size:
            raise ValueError(f"malformed camera table {context}: {path}")
        return rows

    def _read_camera_rows(
        self, path: Path, scene: str, variation: str, camera: str, camera_number: int,
    ) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], set[int]]:
        """Join tables on frame and camera IDs, excluding ambiguous duplicate rows.

        Nonfinite and malformed records raise with the relevant stream and
        frame identity. Missing or duplicate rows invalidate affected windows.
        """
        context = f"{scene}/{variation}/{camera}"
        extrinsic = self._table(path / "extrinsic.txt", context, 18)
        intrinsic = self._table(path / "intrinsic.txt", context, 6)
        def index(table: np.ndarray, label: str) -> tuple[dict[int, np.ndarray], set[int]]:
            """Index one camera's finite rows and identify duplicate frame IDs."""
            selected: dict[int, np.ndarray] = {}
            duplicates: set[int] = set()
            for row in table:
                if not np.isfinite(row[:2]).all() or not float(row[0]).is_integer() or not float(row[1]).is_integer():
                    raise ValueError(f"malformed {label} identity {context}")
                if int(row[1]) != camera_number:
                    continue
                frame = int(row[0])
                if not np.isfinite(row).all():
                    raise ValueError(f"nonfinite {label} target {context}/{frame}")
                if frame in selected:
                    duplicates.add(frame)
                else:
                    selected[frame] = row
            return selected, duplicates
        ext, ext_duplicates = index(extrinsic, "extrinsic")
        intr, intr_duplicates = index(intrinsic, "intrinsic")
        invalid = ext_duplicates | intr_duplicates
        joined: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for frame in ext.keys() & intr.keys() - invalid:
            matrix = ext[frame][2:18].reshape(4, 4)[:3].copy()
            camera_matrix = np.eye(3, dtype=np.float64)
            camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2] = intr[frame][2:6]
            joined[frame] = matrix, camera_matrix
        return joined, invalid

    def _load(self, window: VKittiEpisodeWindow) -> RawEpisode:
        """Decode one validated window and return batched CPU frame targets."""
        assert self.dataset_root is not None
        variation_path = self.dataset_root / window.scene / window.variation
        rows = self._camera_rows[(window.scene, window.variation, window.camera_id)]
        processed = [
            self._frame(variation_path, window, frame, *rows[frame])
            for frame in window.frame_ids
        ]
        images, depths, extrinsics, intrinsics, world_points, cam_points, masks, sizes = zip(*processed)
        identity = f"vkitti/{window.scene}/{window.variation}/{window.camera_id}/{window.start_frame_id:05d}"
        return {
            "seq_name": [identity],
            "episode_metadata": {
                "scene": window.scene, "variation": window.variation, "camera": window.camera_id,
                "start_frame_id": window.start_frame_id, "end_frame_id": window.frame_ids[-1],
            },
            "ids": torch.tensor(window.frame_ids, dtype=torch.int64).unsqueeze(0),
            "images": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255).unsqueeze(0),
            "depths": torch.from_numpy(np.stack(depths).astype(np.float32)).unsqueeze(0),
            "extrinsics": torch.from_numpy(np.stack(extrinsics).astype(np.float32)).unsqueeze(0),
            "intrinsics": torch.from_numpy(np.stack(intrinsics).astype(np.float32)).unsqueeze(0),
            "world_points": torch.from_numpy(np.stack(world_points).astype(np.float32)).unsqueeze(0),
            "cam_points": torch.from_numpy(np.stack(cam_points).astype(np.float32)).unsqueeze(0),
            "point_masks": torch.from_numpy(np.stack(masks)).unsqueeze(0),
            "original_sizes": torch.from_numpy(np.stack(sizes).astype(np.int64)).unsqueeze(0),
        }

    def _frame(
        self, root: Path, window: VKittiEpisodeWindow, frame: int,
        extrinsic: np.ndarray, intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Read matched files, convert depth to metres, and resize without randomness.

        The existing principal-point crop and resize helpers update intrinsics.
        Required depth, camera, and point arrays are checked after float32
        conversion, with frame context if that conversion overflows. Point
        grids serve segment normalization, not point or tracking supervision.
        """
        context = f"{window.scene}/{window.variation}/{window.camera_id}/{frame}"
        rgb = root / "frames" / "rgb" / window.camera_id / f"rgb_{frame:05d}.jpg"
        depth = root / "frames" / "depth" / window.camera_id / f"depth_{frame:05d}.png"
        image = read_image_cv2(str(rgb))
        depth_map = cv2.imread(str(depth), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if image is None or depth_map is None:
            raise FileNotFoundError(f"missing RGB or depth frame {context}")
        if image.shape[:2] != depth_map.shape:
            raise ValueError(f"RGB/depth shape mismatch {context}")
        original_size = np.array(image.shape[:2], dtype=np.int64)
        depth_map = threshold_depth_map(
            depth_map / 100, max_percentile=-1, min_percentile=-1, max_depth=80,
        )
        target_size = np.array([self.image_size, self.image_size])
        image, depth_map, intrinsic, _ = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intrinsic.copy(), original_size,
        )
        current_size = np.array(image.shape[:2])
        image, depth_map, intrinsic, _ = resize_image_depth_and_intrinsic(
            image, depth_map, intrinsic, target_size, current_size, rescale_aug=False,
        )
        image, depth_map, intrinsic, _ = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intrinsic, target_size, strict=True,
        )
        world, camera, mask = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)
        if image.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(f"processed image size mismatch {context}")
        if not all(np.isfinite(value).all() for value in (depth_map, extrinsic, intrinsic, world, camera)):
            raise ValueError(f"nonfinite processed target {context}")
        targets = {
            "depth": depth_map,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "world points": world,
            "camera points": camera,
        }
        converted = {}
        with np.errstate(over="ignore", invalid="ignore"):
            for name, value in targets.items():
                converted[name] = np.asarray(value).astype(np.float32)
                if not np.isfinite(converted[name]).all():
                    raise ValueError(
                        f"nonfinite {name} after float32 conversion {context}"
                    )
        return (
            image, converted["depth"], converted["extrinsic"],
            converted["intrinsic"], converted["world points"],
            converted["camera points"], mask, original_size,
        )
