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
from training.data.datasets.vkitti_sequence import SequentialVKittiEpisodeSource
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



@pytest.mark.parametrize(
    ("enabled", "dtype", "mode", "autocast", "scaled"),
    [
        (False, "bfloat16", "float32", False, False),
        (True, "bfloat16", "bfloat16", True, False),
        (True, "float16", "float16", True, True),
    ],
)
def test_e01a_precision_policy_is_shared_by_training_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    enabled: bool, dtype: str, mode: str, autocast: bool, scaled: bool,
) -> None:
    """Resolve one mode and pass the same autocast policy to both paths."""
    cfg = config(tmp_path)
    cfg["optim"]["amp"] = {"enabled": enabled, "amp_dtype": dtype, "init_scale": 256.0}
    logger = RecordingLogger()
    trainer = RecurrentTrainer(
        cfg, train_episodes=source(1), validation_episodes=source(1),
        model=TinyModel(), loss_fn=loss_fn, logger=logger,
    )
    assert trainer.precision_mode == mode
    assert trainer.autocast_enabled is autocast
    assert trainer.scaler.is_enabled() is scaled
    assert trainer.scaler.get_scale() == (256.0 if scaled else 1.0)
    assert logger.events[0]["precision_mode"] == mode
    assert logger.events[0]["autocast_enabled"] is autocast
    assert logger.events[0]["scaler_enabled"] is scaled
    observed = []
    original_autocast = torch.autocast

    def record_autocast(*args, **kwargs):
        """Capture the active precision context and delegate to PyTorch."""
        observed.append((kwargs["enabled"], kwargs["dtype"]))
        return original_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", record_autocast)
    trainer.train_epoch(0)
    trainer.validate_epoch(0)
    assert observed == [(autocast, trainer.autocast_dtype)] * 2


@pytest.mark.parametrize(
    ("amp", "message"),
    [
        ({"enabled": True, "amp_dtype": "float64"}, "unsupported AMP dtype"),
        ({"enabled": True, "amp_dtype": "float16", "init_scale": 0}, "initial scale"),
    ],
)
def test_e01a_rejects_invalid_precision_configuration(
    tmp_path: Path, amp: dict[str, object], message: str,
) -> None:
    """Reject an unknown dtype or unusable FP16 scale before allocating the model."""
    cfg = config(tmp_path)
    cfg["optim"]["amp"] = amp
    with pytest.raises(ValueError, match=message):
        RecurrentTrainer(
            cfg, train_episodes=source(1), validation_episodes=source(1),
            model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
        )


def test_e01a_enabled_unsupported_autocast_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never fall back to FP32 when enabled autocast is unavailable."""
    cfg = config(tmp_path)
    cfg["optim"]["amp"] = {"enabled": True, "amp_dtype": "bfloat16"}
    monkeypatch.setattr(torch.amp.autocast_mode, "is_autocast_available", lambda _device: False)
    with pytest.raises(ValueError, match="autocast is unavailable"):
        RecurrentTrainer(
            cfg, train_episodes=source(1), validation_episodes=source(1),
            model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
        )


def test_e01a_bfloat16_support_uses_selected_cuda_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the target GPU and restore the caller's current GPU afterward."""
    if torch.cuda.device_count() < 2:
        pytest.skip("mixed-device BF16 check requires two CUDA devices")
    observed: list[int] = []

    def simulated_support(*, including_emulation: bool = False) -> bool:
        """Simulate BF16 only on GPU 1 and record the queried current device."""
        assert including_emulation is False
        current = torch.cuda.current_device()
        observed.append(current)
        return current == 1

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", simulated_support)
    cfg = config(tmp_path)
    cfg["optim"]["amp"] = {"enabled": True, "amp_dtype": "bfloat16"}
    cfg["device"] = "cuda:1"
    with torch.cuda.device(0):
        trainer = RecurrentTrainer(
            cfg, train_episodes=source(1), validation_episodes=source(1),
            model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
        )
        assert trainer.precision_mode == "bfloat16"
        assert torch.cuda.current_device() == 0
    cfg["device"] = "cuda:0"
    with torch.cuda.device(1):
        with pytest.raises(ValueError, match="native bfloat16 autocast"):
            RecurrentTrainer(
                cfg, train_episodes=source(1), validation_episodes=source(1),
                model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
            )
        assert torch.cuda.current_device() == 1
    assert observed == [1, 0]


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


def test_e01a_full_manifest_epochs_derive_horizon_and_resume(tmp_path: Path) -> None:
    """Full epochs traverse both sources and retain the derived horizon on resume."""
    class SizedSource:
        """Expose a fixed episode manifest without loading VKITTI tensors."""

        def __init__(self, count: int) -> None:
            """Store the number of episodes yielded in each epoch."""
            self.count = count

        def __len__(self) -> int:
            """Return the complete manifest size."""
            return self.count

        def __call__(self, epoch: int):
            """Yield the complete manifest for the requested epoch."""
            return source(self.count)(epoch)

    cfg = config(tmp_path, epochs=2)
    cfg["training"]["max_episodes_per_epoch"] = None
    cfg["validation"]["max_episodes_per_epoch"] = None
    cfg["training"]["scheduled_updates"] = None
    train, validation = SizedSource(3), SizedSource(2)
    logger = RecordingLogger()
    runner = RecurrentTrainer(
        cfg, train_episodes=train, validation_episodes=validation,
        model=TinyModel(), loss_fn=loss_fn, logger=logger,
    )
    assert runner.scheduled_updates == 6
    runner.train_epoch(0)
    runner.validate_epoch(0)
    assert runner.completed_updates == 3
    assert sum("train/objective" in event for event in logger.events) == 3
    assert sum("val/objective" in event for event in logger.events) == 2
    runner.save_checkpoint(0, 1.0)

    resumed_cfg = config(tmp_path / "resumed", epochs=2)
    resumed_cfg["training"]["max_episodes_per_epoch"] = None
    resumed_cfg["validation"]["max_episodes_per_epoch"] = None
    resumed_cfg["training"]["scheduled_updates"] = None
    resumed_logger = RecordingLogger()
    resumed = RecurrentTrainer(
        resumed_cfg, train_episodes=train, validation_episodes=validation,
        model=TinyModel(), loss_fn=loss_fn, logger=resumed_logger,
    )
    resumed.resume_from_checkpoint(tmp_path / "epoch_0000.pt")
    assert (resumed.next_epoch, resumed.completed_updates, resumed.scheduled_updates) == (1, 3, 6)
    resumed.run()
    assert (resumed.next_epoch, resumed.completed_updates) == (2, 6)
    assert sum("train/objective" in event for event in resumed_logger.events) == 3
    assert sum("val/objective" in event for event in resumed_logger.events) == 2


def test_e01a_explicit_empty_sources_are_not_replaced(tmp_path: Path) -> None:
    """Retain an empty supplied source and report the appropriate empty epoch."""
    empty = SequentialVKittiEpisodeSource(
        tmp_path, ["Scene01"], training=False, seed=17,
        total_frames=24, image_size=14,
    )
    runner = RecurrentTrainer(
        config(tmp_path), train_episodes=source(1), validation_episodes=empty,
        model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
    )
    assert runner.validation_episodes is empty
    with pytest.raises(ValueError, match="epoch contains no episodes"):
        runner.validate_epoch(0)

    cfg = config(tmp_path)
    cfg["training"]["max_episodes_per_epoch"] = None
    cfg["training"]["scheduled_updates"] = None
    with pytest.raises(ValueError, match="training manifest contains no eligible episodes"):
        RecurrentTrainer(
            cfg, train_episodes=empty, validation_episodes=source(1),
            model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
        )


def test_e01a_auto_horizon_requires_manifest_or_small_limit(tmp_path: Path) -> None:
    """Avoid silently planning an update horizon from an unsized source."""
    cfg = config(tmp_path)
    cfg["training"]["max_episodes_per_epoch"] = None
    cfg["training"]["scheduled_updates"] = None
    with pytest.raises(ValueError, match="sized training source"):
        RecurrentTrainer(
            cfg, train_episodes=source(2), validation_episodes=source(1),
            model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
        )


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
