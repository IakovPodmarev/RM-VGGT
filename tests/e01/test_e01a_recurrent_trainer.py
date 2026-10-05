"""Integration contracts for the single-process recurrent episode trainer."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from launch import load_config, make_trainer
from training.recurrent_trainer import RecurrentTrainer, iter_prepared_segments, transfer_segment_to_device


def episode(scale: float = 2.0, segment_frames: int = 8) -> dict[str, object]:
    """Build three distinguishable raw CPU segments with complete geometry."""
    frames = 3 * segment_frames
    ids = torch.arange(frames, dtype=torch.int64)[None]
    images = ids.float().reshape(1, frames, 1, 1, 1).expand(1, frames, 3, 2, 2).clone()
    extrinsics = torch.zeros(1, frames, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).expand(1, frames, 3, 3).clone()
    values = torch.tensor([scale, scale * 2, scale * 4]).repeat_interleave(segment_frames)
    world_points = torch.zeros(1, frames, 2, 2, 3)
    world_points[..., 0] = values.reshape(1, frames, 1, 1)
    return {
        "seq_name": ["scene-left"],
        "ids": ids,
        "images": images,
        "depths": values.reshape(1, frames, 1, 1).expand(1, frames, 2, 2).clone(),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "cam_points": world_points.clone(),
        "world_points": world_points,
        "point_masks": torch.ones(1, frames, 2, 2, dtype=torch.bool),
        "episode_metadata": {"stream": "left"},
    }


class TinyModel(nn.Module):
    """Expose the real capability group names and observable memory resets."""

    def __init__(self) -> None:
        """Create frozen encoding and trainable memory, adaptors, and heads."""
        super().__init__()
        self.aggregator = nn.Linear(1, 1)
        self.memory_writer = nn.Linear(1, 1)
        self.memory_writer.initial_memory_bank = nn.Parameter(torch.ones(1, 1, 1))
        self.camera_read_adaptor = nn.Linear(1, 1)
        self.depth_read_adaptor = nn.Linear(1, 1)
        self.camera_head = nn.Linear(1, 1)
        self.depth_head = nn.Linear(1, 1)
        for parameter in self.aggregator.parameters():
            parameter.requires_grad_(False)
        self.aggregator.eval()
        self.initial_calls = 0
        self.forward_calls = 0
        self.observed_ids: list[tuple[int, ...]] = []
        self.first_memories: list[Tensor] = []

    def train(self, mode: bool = True) -> "TinyModel":
        """Change trainable module mode while retaining the frozen encoder in eval."""
        super().train(mode)
        self.aggregator.eval()
        return self

    def initial_memory(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return one new graph-connected learned initial memory per episode."""
        self.initial_calls += 1
        memory = self.memory_writer.initial_memory_bank.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        self.first_memories.append(memory.detach().clone())
        return memory

    def forward(self, *, images: Tensor, read_memory: Tensor):
        """Predict with incoming memory and write a connected outgoing state."""
        self.forward_calls += 1
        signal = images.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1) / 24
        camera = self.camera_head(self.camera_read_adaptor(read_memory))
        depth = self.depth_head(self.depth_read_adaptor(read_memory))
        return {"camera": camera + signal, "depth": depth + signal}, self.memory_writer(read_memory) + signal, {}


def loss_fn(prediction: dict[str, Tensor], target: dict[str, object]) -> dict[str, Tensor]:
    """Return differentiable camera, depth, and objective scalars."""
    camera = prediction["camera"].square().mean()
    depth = prediction["depth"].square().mean()
    return {"objective": camera + depth, "loss_camera": camera, "loss_reg_depth": depth}


class RecordingLogger:
    """Retain only trainer-emitted plain data for assertions."""

    def __init__(self) -> None:
        """Start with no run metadata or event records."""
        self.events: list[dict[str, object]] = []
        self.config: dict[str, object] | None = None

    def log(self, values: dict[str, object], *, step: int | None = None) -> None:
        """Record scalar and metadata fields without tensor graphs."""
        assert all(not torch.is_tensor(value) for value in values.values())
        self.events.append(dict(values))


def config(tmp_path: Path, *, epochs: int = 1, train_limit: int = 1, val_limit: int = 1) -> dict[str, object]:
    """Describe the real step, two optimizer groups, clipping, and offline logging."""
    return {
        "experiment_id": "E01a",
        "device": "cpu",
        "seed_value": 41,
        "max_epochs": epochs,
        "sequence": {"total_frames": 24, "segment_frames": 8, "num_segments": 3, "backprop_mode": "full"},
        "training": {"max_episodes_per_epoch": train_limit, "episode_batch_size": 1, "accum_steps": 1, "scheduled_updates": epochs * train_limit},
        "validation": {"max_episodes_per_epoch": val_limit},
        "optim": {
            "optimizer": {"_target_": "torch.optim.AdamW", "weight_decay": 0.05},
            "learning_rates": {"recurrent_memory": 1e-4, "prediction_heads": 5e-5},
            "scheduler": {"warmup_fraction": 0.05, "type": "cosine"},
            "gradient_clip": {"_target_": "train_utils.gradient_clip.GradientClipper", "configs": [{"module_name": ["memory_writer", "camera_read_adaptor", "depth_read_adaptor", "camera_head", "depth_head"], "max_norm": 1.0, "norm_type": 2}]},
            "amp": {"enabled": False, "amp_dtype": "bfloat16"},
        },
        "logging": {"mode": "disabled", "project": "test", "run_name": "synthetic"},
        "checkpoint": {"save_dir": str(tmp_path), "resume_checkpoint_path": None, "pretrained_checkpoint_path": None},
        "dataset_split": {"train": ["synthetic"], "val": ["synthetic"]},
        "model": {"_target_": "vggt.models.RMVGGT.RMVGGT"},
        "loss": {"_target_": "loss.MultitaskLoss"},
    }


def source(count: int, scale: float = 2.0):
    """Return a repeatable epoch-indexed source of independent CPU episodes."""
    return lambda _epoch: (episode(scale) for _ in range(count))


def test_e01a_preparation_splits_then_interleaves_normalize_transfer_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the requested segment is normalized and transferred before its model call."""
    import training.recurrent_trainer as module
    from training.data.episode import normalize_segment, split_episode

    events: list[str] = []
    raw = episode()
    untouched = deepcopy(raw)
    model = TinyModel()
    original_split, original_normalize = split_episode, normalize_segment

    def split(value, *args, **kwargs):
        """Record the initial all-CPU split."""
        events.append("split")
        return original_split(value, *args, **kwargs)

    def normalize(value):
        """Record CPU-only normalization before transfer."""
        assert value["images"].device.type == "cpu"
        events.append("normalize" + str(value["segment_index"]))
        return original_normalize(value)

    def transfer(value, device):
        """Record transfer while preserving dtypes and metadata."""
        events.append("transfer" + str(value["segment_index"]))
        return transfer_segment_to_device(value, device)

    original_forward = model.forward

    def forward(*, images, read_memory):
        """Record model work after the current transfer."""
        index = model.forward_calls
        events.append("model" + str(index))
        return original_forward(images=images, read_memory=read_memory)

    monkeypatch.setattr(module, "split_episode", split)
    monkeypatch.setattr(module, "normalize_segment", normalize)
    monkeypatch.setattr(module, "transfer_segment_to_device", transfer)
    model.forward = forward
    from training.recurrent_sequence import run_recurrent_sequence
    prepared = iter_prepared_segments(raw, device=torch.device("cpu"), total_frames=24, segment_frames=8)
    run_recurrent_sequence(model, prepared, num_segments=3, segment_frames=8)
    assert events == ["split", "normalize0", "transfer0", "model0", "normalize1", "transfer1", "model1", "normalize2", "transfer2", "model2"]
    for name in raw:
        if torch.is_tensor(raw[name]):
            assert torch.equal(raw[name], untouched[name])
    assert prepared is not raw


def test_e01a_preparation_preserves_local_geometry_and_metadata() -> None:
    """Each segment has its own unit scale, first camera, IDs, and ordered identity."""
    segments = list(iter_prepared_segments(episode(), device=torch.device("cpu"), total_frames=24, segment_frames=8))
    assert [item["ids"].tolist() for item in segments] == [[list(range(i, i + 8))] for i in (0, 8, 16)]
    assert [item["segment_index"] for item in segments] == [0, 1, 2]
    assert [item["seq_name"] for item in segments] == [["scene-left"]] * 3
    assert [item["frame_start"] for item in segments] == [0, 8, 16]
    assert [item["frame_stop"] for item in segments] == [8, 16, 24]
    for item in segments:
        assert torch.allclose(item["world_points"][..., 0], torch.ones_like(item["world_points"][..., 0]), atol=1e-3)
        assert torch.allclose(item["extrinsics"][:, 0, :3, :3], torch.eye(3)[None])
        assert item["ids"].dtype == torch.int64
        assert item["point_masks"].dtype == torch.bool
        assert item["images"].dtype == torch.float32


def test_e01a_one_episode_uses_real_step_once_and_validation_is_read_only(tmp_path: Path) -> None:
    """The actual step owns one update; validation preserves every training state."""
    model, logger = TinyModel(), RecordingLogger()
    runner = RecurrentTrainer(config(tmp_path), train_episodes=source(1), validation_episodes=source(1), model=model, loss_fn=loss_fn, logger=logger)
    frozen = [parameter.detach().clone() for parameter in model.aggregator.parameters()]
    actions = {"schedule": 0, "scaler": 0, "clip": 0}
    original_schedule = runner.optimizer.step_schedulers
    original_scaler = runner.scaler.update
    original_clipper = runner.gradient_clipper
    def schedule(progress):
        """Count the step-owned schedule action."""
        actions["schedule"] += 1
        return original_schedule(progress)
    def scaler_update(*args, **kwargs):
        """Count the step-owned scaler update."""
        actions["scaler"] += 1
        return original_scaler(*args, **kwargs)
    class CountingClipper:
        """Delegate to the configured global clipper while counting calls."""

        def __call__(self, model):
            """Count and execute one combined clip."""
            actions["clip"] += 1
            return original_clipper(model)

    runner.optimizer.step_schedulers = schedule
    runner.scaler.update = scaler_update
    runner.gradient_clipper = CountingClipper()
    train = runner.train_epoch(0)
    assert runner.completed_updates == 1
    assert actions == {"schedule": 1, "scaler": 1, "clip": 1}
    assert model.forward_calls == 3 and model.initial_calls == 1
    assert all(parameter.grad is None for parameter in model.aggregator.parameters())
    assert all(torch.equal(before, after) for before, after in zip(frozen, model.aggregator.parameters()))
    assert not model.aggregator.training
    assert "objective" in train
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optim_before = deepcopy(runner.optimizer.optimizer.state_dict())
    scaler_before = deepcopy(runner.scaler.state_dict())
    lr_before = [group["lr"] for group in runner.optimizer.optimizer.param_groups]
    validation = runner.validate_epoch(0)
    assert "objective" in validation
    assert model.forward_calls == 6 and model.initial_calls == 2
    assert runner.completed_updates == 1
    assert actions == {"schedule": 1, "scaler": 1, "clip": 1}
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert runner.optimizer.optimizer.state_dict()["state"].keys() == optim_before["state"].keys()
    assert runner.scaler.state_dict() == scaler_before
    assert [group["lr"] for group in runner.optimizer.optimizer.param_groups] == lr_before
    assert any("train/segment_0/objective" in event for event in logger.events)
    assert any("val/segment_2/objective" in event for event in logger.events)
    assert any("frame_ids" in str(event) and "batch_identities" in str(event) for event in logger.events)
    assert any("peak_memory_bytes" in str(event) and "None" in str(event) for event in logger.events)
    assert all(not torch.is_tensor(value) for event in logger.events for value in event.values())


def test_e01a_limits_are_exact_and_memory_resets_per_phase(tmp_path: Path) -> None:
    """At most two train and one validation episodes are consumed per epoch."""
    model = TinyModel()
    runner = RecurrentTrainer(config(tmp_path, train_limit=2, val_limit=1), train_episodes=source(5), validation_episodes=source(5), model=model, loss_fn=loss_fn, logger=RecordingLogger())
    runner.train_epoch(0)
    runner.validate_epoch(0)
    assert runner.completed_updates == 2
    assert model.forward_calls == 9
    assert model.initial_calls == 3
    assert len(model.first_memories) == 3
    assert all(memory.shape == (1, 1, 1) for memory in model.first_memories)


def test_e01a_launch_dispatch_and_configured_sources_are_explicit(tmp_path: Path) -> None:
    """Generic config dispatch names callable sequential source targets."""
    cfg = load_config("e01a_frozen_aggregator_streaming")
    assert cfg.trainer_target == "recurrent_trainer.RecurrentTrainer"
    assert cfg.model._target_ == "vggt.models.RMVGGT.RMVGGT"
    assert RecurrentTrainer._configured_source(OmegaConf.to_container(cfg, resolve=True)["episode_sources"]["train"]) is not None
    assert RecurrentTrainer._configured_source(OmegaConf.to_container(cfg, resolve=True)["episode_sources"]["validation"]) is not None
    class FakeTrainer:
        """Capture ordinary config dispatch without starting distributed training."""

        def __init__(self, **kwargs):
            """Retain the original composed config fields."""
            self.kwargs = kwargs

    ordinary = make_trainer(load_config("default"), trainer_factory=FakeTrainer)
    assert isinstance(ordinary, FakeTrainer)
    assert ordinary.kwargs["exp_name"] == "exp001"
    assert ordinary.kwargs["logging"].log_dir == "logs"


def test_e01a_documented_source_target_imports_without_launching_training() -> None:
    """Configured source targets resolve without constructing a model or logger."""
    cfg = load_config("e01a_frozen_aggregator_streaming")
    train = RecurrentTrainer._configured_source(OmegaConf.to_container(cfg, resolve=True)["episode_sources"]["train"])
    validation = RecurrentTrainer._configured_source(OmegaConf.to_container(cfg, resolve=True)["episode_sources"]["validation"])
    assert callable(train) and callable(validation)
    with pytest.raises(ValueError, match="dataset root"):
        train(0)


def test_e01a_skipped_scaler_attempt_reuses_schedule_and_survives_resume(tmp_path: Path) -> None:
    """A skipped AdamW call leaves state and rates intact; the next attempt uses that slot."""
    from copy import deepcopy

    logger = RecordingLogger()
    runner = RecurrentTrainer(
        config(tmp_path, train_limit=2), train_episodes=source(2),
        validation_episodes=source(1), model=TinyModel(), loss_fn=loss_fn,
        logger=logger,
    )
    underlying = runner.optimizer.optimizer
    original_step = runner.scaler.step
    original_schedule = runner.optimizer.step_schedulers
    attempted_progress = []
    attempted_rates = []
    before_weights = deepcopy(runner.model.state_dict())
    before_state = deepcopy(underlying.state_dict())
    before_rates = [group["lr"] for group in underlying.param_groups]

    def schedule(progress):
        """Verify a skipped attempt restored state before requesting the same slot."""
        if attempted_progress:
            assert runner.completed_updates == 0 and runner.skipped_attempts == 1
            assert [group["lr"] for group in underlying.param_groups] == before_rates
            assert all(torch.equal(value, before_weights[name]) for name, value in runner.model.state_dict().items())
            assert _state_equal_for_skip(underlying.state_dict(), before_state)
        attempted_progress.append(progress)
        return original_schedule(progress)

    def skip_then_step(optimizer):
        """Simulate one GradScaler overflow, then delegate the next real step."""
        attempted_rates.append([group["lr"] for group in optimizer.param_groups])
        if len(attempted_rates) == 1:
            assert runner.completed_updates == 0
            assert all(torch.equal(value, before_weights[name]) for name, value in runner.model.state_dict().items())
            assert _state_equal_for_skip(optimizer.state_dict(), before_state)
            return None
        assert [group["lr"] for group in optimizer.param_groups] != before_rates
        return original_step(optimizer)

    runner.optimizer.step_schedulers = schedule
    runner.scaler.step = skip_then_step
    runner.train_epoch(0)
    assert attempted_progress == [attempted_progress[0]] * 2
    assert attempted_rates[0] == attempted_rates[1]
    assert runner.completed_updates == 1 and runner.skipped_attempts == 1
    assert any("train/optimizer_ran" in event and event["train/optimizer_ran"] is False for event in logger.events)
    assert any("train/optimizer_ran" in event and event["train/optimizer_ran"] is True for event in logger.events)
    assert len(underlying.state) > 0
    runner.save_checkpoint(0, 1.0)
    saved = torch.load(tmp_path / "epoch_0000.pt", map_location="cpu", weights_only=False)
    assert saved["completed_updates"] == 1 and saved["skipped_attempts"] == 1
    assert saved["scheduler_progress"] == 1 / runner.scheduled_updates

    resumed = RecurrentTrainer(
        config(tmp_path / "resumed", train_limit=2), train_episodes=source(1),
        validation_episodes=source(1), model=TinyModel(), loss_fn=loss_fn,
        logger=RecordingLogger(),
    )
    resumed.resume_from_checkpoint(tmp_path / "epoch_0000.pt")
    assert resumed.completed_updates == 1 and resumed.skipped_attempts == 1
    assert [group["lr"] for group in resumed.optimizer.optimizer.param_groups] == [
        group["lr"] for group in underlying.param_groups
    ]


def _state_equal_for_skip(first: dict, second: dict) -> bool:
    """Compare optimizer state recursively before any actual AdamW update."""
    from copy import deepcopy
    if first["state"] != second["state"]:
        return False
    left, right = deepcopy(first), deepcopy(second)
    for group in left["param_groups"]:
        group.pop("lr", None)
    for group in right["param_groups"]:
        group.pop("lr", None)
    return left == right
