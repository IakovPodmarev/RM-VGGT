"""Checkpoint, resume, and pretrained-loading contracts for recurrent training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.recurrent_trainer import RecurrentTrainer, load_pretrained_components
from test_e01a_recurrent_trainer import RecordingLogger, TinyModel, config, episode, loss_fn, source


def _run(tmp_path: Path, *, epochs: int, model: TinyModel | None = None, resume: str | None = None, episode_source=None) -> RecurrentTrainer:
    """Build and execute a short real-step run with a fixed schedule budget."""
    cfg = config(tmp_path, epochs=epochs)
    cfg["training"]["scheduled_updates"] = 2
    cfg["checkpoint"]["resume_checkpoint_path"] = resume
    runner = RecurrentTrainer(cfg, train_episodes=episode_source or source(1), validation_episodes=source(1), model=model or TinyModel(), loss_fn=loss_fn, logger=RecordingLogger())
    runner.run()
    return runner


def _state_equal(first: dict, second: dict) -> bool:
    """Compare nested model and optimizer tensors without relying on pickle identity."""
    if first.keys() != second.keys():
        return False
    for key in first:
        a, b = first[key], second[key]
        if torch.is_tensor(a):
            if not torch.equal(a, b):
                return False
        elif isinstance(a, dict):
            if not _state_equal(a, b):
                return False
        elif isinstance(a, list):
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if isinstance(x, dict):
                    if not _state_equal(x, y):
                        return False
                elif torch.is_tensor(x):
                    if not torch.equal(x, y):
                        return False
                elif x != y:
                    return False
        elif a != b:
            return False
    return True


def test_e01a_epoch_and_best_checkpoints_have_complete_state(tmp_path: Path) -> None:
    """Every epoch is stored, best follows validation, and transient states stay out."""
    runner = _run(tmp_path, epochs=2)
    first = tmp_path / "epoch_0000.pt"
    second = tmp_path / "epoch_0001.pt"
    best = tmp_path / "best.pt"
    assert first.exists() and second.exists() and best.exists()
    saved = torch.load(second, map_location="cpu", weights_only=False)
    assert set(("model", "optimizer", "scaler", "scheduler_progress", "completed_epoch", "next_epoch", "completed_updates", "best_validation_objective", "config", "run_metadata", "rng_state")) <= saved.keys()
    assert saved["completed_epoch"] == 1
    assert saved["next_epoch"] == 2
    assert saved["completed_updates"] == 2
    assert "memory_writer.initial_memory_bank" in saved["model"]
    assert not any(key.startswith(("m[", "memory_states", "transient_memory")) for key in saved)
    assert [group["name"] for group in saved["optimizer"]["param_groups"]] == ["recurrent_memory", "prediction_heads"]
    assert len(saved["optimizer"]["state"]) > 0
    assert saved["best_validation_objective"] <= torch.load(first, map_location="cpu", weights_only=False)["best_validation_objective"]
    best_state = torch.load(best, map_location="cpu", weights_only=False)
    assert best_state["best_validation_objective"] == saved["best_validation_objective"]
    assert runner.completed_updates == 2


def test_e01a_resume_restores_complete_state_and_progress(tmp_path: Path) -> None:
    """A strict epoch-boundary resume restores model, optimizer, scaler, and counters."""
    partial = _run(tmp_path / "partial", epochs=1)
    path = tmp_path / "partial" / "epoch_0000.pt"
    fresh = TinyModel()
    cfg = config(tmp_path / "resumed", epochs=2)
    cfg["training"]["scheduled_updates"] = 2
    runner = RecurrentTrainer(cfg, train_episodes=source(1), validation_episodes=source(1), model=fresh, loss_fn=loss_fn, logger=RecordingLogger())
    runner.resume_from_checkpoint(path)
    assert runner.next_epoch == 1
    assert runner.completed_updates == partial.completed_updates == 1
    assert runner.best_validation_objective == partial.best_validation_objective
    assert _state_equal(runner.model.state_dict(), partial.model.state_dict())
    assert _state_equal(runner.optimizer.optimizer.state_dict(), partial.optimizer.optimizer.state_dict())
    assert runner.scaler.state_dict() == partial.scaler.state_dict()
    assert [group["lr"] for group in runner.optimizer.optimizer.param_groups] == [group["lr"] for group in partial.optimizer.optimizer.param_groups]
    runner.run()
    assert runner.completed_updates == 2 and runner.next_epoch == 2


def test_e01a_resume_rejects_changed_schedule_before_loading_weights(tmp_path: Path) -> None:
    """A changed warmup definition cannot reinterpret saved progress on resume."""
    _run(tmp_path / "first", epochs=1)
    checkpoint = tmp_path / "first" / "epoch_0000.pt"
    cfg = config(tmp_path / "changed", epochs=2)
    cfg["training"]["scheduled_updates"] = 2
    cfg["optim"]["scheduler"]["warmup_fraction"] = 0.2
    runner = RecurrentTrainer(cfg, train_episodes=source(1), validation_episodes=source(1),
                              model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger())
    before = {name: value.clone() for name, value in runner.model.state_dict().items()}
    with pytest.raises(ValueError, match="scheduler"):
        runner.resume_from_checkpoint(checkpoint)
    assert all(torch.equal(before[name], value) for name, value in runner.model.state_dict().items())
    assert runner.completed_updates == 0 and runner.next_epoch == 0


def test_e01a_cpu_continuation_matches_uninterrupted_epoch_boundary(tmp_path: Path) -> None:
    """Restored RNG makes stochastic CPU episodes and final weights identical."""
    def random_source(_epoch: int): 
        """Draw image perturbation as each epoch requests its single episode."""
        raw = episode()
        raw["images"] = raw["images"] + torch.rand_like(raw["images"]) * 0.1
        yield raw

    torch.manual_seed(1234)
    full = _run(tmp_path / "full", epochs=2, episode_source=random_source)
    torch.manual_seed(1234)
    _run(tmp_path / "first", epochs=1, episode_source=random_source)
    continued = _run(tmp_path / "continued", epochs=2, resume=str(tmp_path / "first" / "epoch_0000.pt"), episode_source=random_source)
    assert _state_equal(full.model.state_dict(), continued.model.state_dict())
    assert _state_equal(full.optimizer.optimizer.state_dict(), continued.optimizer.optimizer.state_dict())
    assert full.completed_updates == continued.completed_updates == 2
    assert full.best_validation_objective == continued.best_validation_objective


def test_e01a_pretrained_initialization_is_distinct_from_full_resume(tmp_path: Path) -> None:
    """Required encoder and head keys load; recurrent keys stay fresh."""
    original, target = TinyModel(), TinyModel()
    recurrent_before = {key: value.clone() for key, value in target.state_dict().items() if key.startswith(("memory_writer.", "camera_read_adaptor.", "depth_read_adaptor."))}
    pretrained = {key: value.clone() for key, value in original.state_dict().items() if key.startswith(("aggregator.", "camera_head.", "depth_head."))}
    path = tmp_path / "pretrained.pt"
    torch.save({"model": pretrained}, path)
    missing, unexpected = load_pretrained_components(target, path)
    assert unexpected == []
    assert set(missing) == set(recurrent_before)
    for key, value in pretrained.items():
        assert torch.equal(target.state_dict()[key], value)
    for key, value in recurrent_before.items():
        assert torch.equal(target.state_dict()[key], value)
    cfg = config(tmp_path / "resume")
    runner = RecurrentTrainer(cfg, train_episodes=source(1), validation_episodes=source(1), model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger())
    with pytest.raises(ValueError, match="optimizer|resume|checkpoint"):
        runner.resume_from_checkpoint(path)
    broken = dict(pretrained)
    broken.pop("camera_head.weight")
    torch.save({"model": broken}, path)
    with pytest.raises(ValueError, match="camera_head"):
        load_pretrained_components(TinyModel(), path)
    broken = dict(pretrained)
    broken["unknown.weight"] = torch.ones(1)
    torch.save({"model": broken}, path)
    with pytest.raises(ValueError, match="unexpected|unknown"):
        load_pretrained_components(TinyModel(), path)


def test_e01a_best_checkpoint_keeps_lowest_validation_epoch(tmp_path: Path) -> None:
    """A worse second validation epoch cannot replace the first best model."""
    def validation_source(epoch: int):
        """Attach an epoch-specific scalar penalty to one raw validation episode."""
        raw = episode()
        raw["validation_penalty"] = float(epoch) * 1000.0
        yield raw

    def penalized_loss(prediction, target):
        """Add a validation-only penalty while keeping the model path connected."""
        values = loss_fn(prediction, target)
        values["objective"] = values["objective"] + float(target.get("validation_penalty", 0.0))
        return values

    cfg = config(tmp_path, epochs=2)
    runner = RecurrentTrainer(cfg, train_episodes=source(1), validation_episodes=validation_source, model=TinyModel(), loss_fn=penalized_loss, logger=RecordingLogger())
    runner.run()
    first = torch.load(tmp_path / "epoch_0000.pt", map_location="cpu", weights_only=False)
    second = torch.load(tmp_path / "epoch_0001.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert second["validation_objective"] > first["validation_objective"]
    assert best["completed_epoch"] == 0
    assert best["best_validation_objective"] == first["validation_objective"]


def test_e01a_lightweight_real_rmvggt_composition_trains_one_episode(tmp_path: Path) -> None:
    """The runner accepts actual writer, adaptor, and RMVGGT composition."""
    from test_e01a_rmvggt import _model

    model, aggregator, _, _ = _model()
    cfg = config(tmp_path)
    cfg["sequence"] = {"total_frames": 6, "segment_frames": 2, "num_segments": 3, "backprop_mode": "full"}

    def raw_source(_epoch: int):
        """Yield one compatible six-frame CPU episode."""
        yield episode(segment_frames=2)

    def real_loss(prediction, target):
        """Supervise the actual camera and depth outputs with finite scalars."""
        camera = prediction["pose_enc"].square().mean()
        depth = prediction["depth"].square().mean()
        return {"objective": camera + depth, "loss_camera": camera, "loss_reg_depth": depth}

    runner = RecurrentTrainer(cfg, train_episodes=raw_source, validation_episodes=raw_source, model=model, loss_fn=real_loss, logger=RecordingLogger())
    runner.run()
    assert runner.completed_updates == 1
    assert not aggregator.training
    assert all(parameter.grad is None for parameter in aggregator.parameters())
