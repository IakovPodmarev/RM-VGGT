"""Two-rank bounded train and validation integration for complete episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import socket

import numpy as np

import pytest
import torch
import torch.multiprocessing as mp

from training.distributed_recurrent_trainer import DistributedRecurrentTrainer
from training.recurrent_loss import compute_recurrent_losses
from training.recurrent_sequence import run_recurrent_sequence
from training.recurrent_trainer import iter_prepared_segments
from test_e01a_recurrent_trainer import RecordingLogger, TinyModel, config, episode, loss_fn


@dataclass(frozen=True)
class _Window:
    """Small stable episode identity without decoded frames."""

    identity: str
    scale: float


class _Source:
    """Track rank-local lazy window loads."""

    def __init__(self, windows: tuple[_Window, ...]) -> None:
        """Retain metadata and an initially empty load audit."""
        self.windows = windows
        self.loaded: list[str] = []

    def __len__(self) -> int:
        """Return the configured eligible window count."""
        return len(self.windows)

    def eligible_windows(self) -> tuple[_Window, ...]:
        """Expose metadata without loading tensor frames."""
        return self.windows

    def load_window(self, window: _Window) -> dict[str, object]:
        """Decode exactly the window assigned to this rank."""
        self.loaded.append(window.identity)
        raw = episode(window.scale)
        raw["seq_name"] = [window.identity]
        return raw

    def __call__(self, epoch: int):
        """Reject source-owned shuffling in the distributed path."""
        raise AssertionError("distributed trainer must use the episode planner")


def _reference_validation(model: TinyModel, windows: tuple[_Window, ...]) -> float:
    """Compute the unsharded episode mean using the trained model state."""
    model.eval()
    values = []
    with torch.no_grad():
        for window in windows:
            raw = episode(window.scale)
            recorded = []

            def prepared():
                """Retain targets as each segment is requested."""
                for segment in iter_prepared_segments(raw, device=torch.device("cpu"), total_frames=24, segment_frames=8):
                    recorded.append(segment)
                    yield segment

            sequence = run_recurrent_sequence(model, prepared(), num_segments=3, segment_frames=8)
            losses = compute_recurrent_losses(sequence, recorded, loss_fn, num_segments=3)
            values.append(float(losses.objective))
    return sum(values) / len(values)


def _worker(rank: int, directory: str, port: int) -> None:
    """Run a short Gloo job and save rank-local evidence for the parent."""
    import os

    torch.set_num_threads(1)
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    train_windows = tuple(_Window(f"train-{index}", float(index + 1)) for index in range(3))
    val_windows = tuple(_Window(f"val-{index}", float(index + 4)) for index in range(3))
    train, validation = _Source(train_windows), _Source(val_windows)
    cfg = config(Path(directory) / "checkpoints", train_limit=3, val_limit=3)
    cfg["training"]["scheduled_updates"] = None
    logger = RecordingLogger()
    trainer = DistributedRecurrentTrainer(
        cfg, train_episodes=train, validation_episodes=validation,
        model=TinyModel(), loss_fn=loss_fn, logger=logger,
    )
    forwards = [0]
    updates = [0]
    trainer.episode_module.module.register_forward_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
    trainer.optimizer.optimizer.register_step_post_hook(lambda *_: updates.__setitem__(0, updates[0] + 1))
    trainer.run()
    initial_calls = trainer.model.initial_calls
    reference = _reference_validation(trainer.model, val_windows)
    torch.save({
        "rank": rank, "train_loads": train.loaded, "val_loads": validation.loaded,
        "forwards": forwards[0], "optimizer_calls": updates[0],
        "completed_updates": trainer.completed_updates, "skipped_attempts": trainer.skipped_attempts,
        "scheduled_updates": trainer.scheduled_updates, "initial_calls": initial_calls,
        "reference": reference, "events": logger.events,
        "parameters": {name: parameter.detach().cpu() for name, parameter in trainer.model.named_parameters()},
    }, Path(directory) / f"rank-{rank}.pt")


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_trainer_bounded_train_and_global_validation(tmp_path: Path) -> None:
    """Two ranks train distinct episodes, pad one update, and reduce validation."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(str(tmp_path), port), nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(2)]
    assert first["scheduled_updates"] == second["scheduled_updates"] == 2
    assert first["completed_updates"] == second["completed_updates"] == 2
    assert first["skipped_attempts"] == second["skipped_attempts"] == 0
    assert first["forwards"] == second["forwards"] == 2
    assert first["optimizer_calls"] == second["optimizer_calls"] == 2
    assert first["initial_calls"] == 4
    assert second["initial_calls"] == 3
    real = [event["train/identity"] for event in first["events"] if "train/identity" in event]
    assert sorted(real) == ["train-0", "train-1", "train-2"]
    assert len(first["train_loads"]) == len(second["train_loads"]) == 2
    assert sorted(first["val_loads"] + second["val_loads"]) == ["val-0", "val-1", "val-2"]
    assert set(first["val_loads"]).isdisjoint(second["val_loads"])
    assert second["events"] == []
    for name in first["parameters"]:
        torch.testing.assert_close(first["parameters"][name], second["parameters"][name])
    summary = next(event for event in first["events"] if "val_epoch/objective" in event)
    assert summary["val_epoch/real_episodes"] == 3
    assert summary["val_epoch/objective"] == pytest.approx(first["reference"], abs=1e-4)
    assert first["reference"] == pytest.approx(second["reference"], abs=1e-7)
    assert sorted(path.name for path in (tmp_path / "checkpoints").glob("*.pt")) == ["best.pt", "epoch_0000.pt"]


def _metric_failure_worker(rank: int, directory: str, port: int) -> None:
    """Inject one post-step metric failure and save each rank's outcome."""
    import os
    from importlib import import_module

    torch.set_num_threads(1)
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    windows = tuple(_Window(f"train-{index}", float(index + 1)) for index in range(2))
    cfg = config(Path(directory) / "checkpoints", train_limit=2, val_limit=1)
    cfg["training"]["scheduled_updates"] = None
    logger = RecordingLogger()
    trainer = DistributedRecurrentTrainer(
        cfg, train_episodes=_Source(windows), validation_episodes=_Source((_Window("val-0", 4.0),)),
        model=TinyModel(), loss_fn=loss_fn, logger=logger,
    )
    calls = [0]
    trainer.optimizer.optimizer.register_step_post_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    if rank == 1:
        module = import_module("training.distributed_recurrent_trainer")

        def fail_metrics(*args, **kwargs):
            """Fail after synchronized AdamW while extracting one real episode."""
            raise RuntimeError("injected post-step metric failure")

        module._metrics = fail_metrics
    try:
        trainer.run()
    except RuntimeError as exc:
        error = str(exc)
    else:
        error = None
    torch.save({
        "error": error, "optimizer_calls": calls[0],
        "completed_updates": trainer.completed_updates, "events": logger.events,
    }, Path(directory) / f"failure-rank-{rank}.pt")


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_trainer_reports_post_step_metric_failure_to_all_ranks(tmp_path: Path) -> None:
    """Both ranks surface one rank's metric error after both AdamW calls."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_metric_failure_worker, args=(str(tmp_path), port), nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f"failure-rank-{rank}.pt", weights_only=False) for rank in range(2)]
    assert first["optimizer_calls"] == second["optimizer_calls"] == 1
    assert first["completed_updates"] == second["completed_updates"] == 1
    assert first["error"] == second["error"]
    assert "injected post-step metric failure" in first["error"]
    failures = [event["distributed/rank_failures"] for event in first["events"] if "distributed/rank_failures" in event]
    assert len(failures) == 1
    assert set(failures[0]) == {1}
    assert second["events"] == []



class _StochasticSource(_Source):
    """Consume all CPU RNG families while loading a rank-local episode."""

    def load_window(self, window: _Window) -> dict[str, object]:
        """Perturb input with Python, NumPy, and Torch draws on the owning rank."""
        raw = super().load_window(window)
        perturbation = random.random() + float(np.random.random()) + float(torch.rand(()))
        raw["images"] = raw["images"] + perturbation
        return raw


def _assert_nested_equal(first: object, second: object) -> None:
    """Compare nested checkpoint values with exact tensor and array equality."""
    if torch.is_tensor(first):
        assert torch.is_tensor(second)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray)
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert isinstance(second, type(first)) and len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def _spawn(worker, directory: Path, *args) -> None:
    """Launch two fresh Gloo processes on an available local TCP port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(str(directory), port, *args), nprocs=2, join=True)


def _continuation_worker(rank: int, directory: str, port: int, phase: str, epochs: int) -> None:
    """Run a fresh, partial, or resumed stochastic job and persist rank evidence."""
    import os
    import training.distributed_recurrent_trainer as module

    torch.set_num_threads(1)
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    root = Path(directory)
    train = _StochasticSource(tuple(_Window(f"train-{index}", float(index + 1)) for index in range(3)))
    validation = _StochasticSource(tuple(_Window(f"val-{index}", float(index + 4)) for index in range(3)))
    cfg = config(root / phase / "checkpoints", epochs=epochs, train_limit=3, val_limit=3)
    cfg["training"]["scheduled_updates"] = 12 if phase == "cadence" else 6 if phase in {"custom_cadence", "retention"} else 4
    if phase in {"custom_cadence", "retention"}:
        cfg["checkpoint"]["save_every_epochs"] = 2 if phase == "custom_cadence" else 1
    if phase == "resumed":
        cfg["checkpoint"]["resume_checkpoint_path"] = str(root / "partial" / "checkpoints" / "epoch_0000.pt")
    logger = RecordingLogger()
    writes: list[str] = []
    original_save = module.robust_torch_save

    def recording_save(checkpoint, checkpoint_path):
        """Record checkpoint writers and reject any nonzero-rank write."""
        if rank:
            raise AssertionError("nonzero rank attempted a checkpoint write")
        writes.append(Path(checkpoint_path).name)
        return original_save(checkpoint, checkpoint_path)

    module.robust_torch_save = recording_save
    torch.manual_seed(12345)
    trainer = DistributedRecurrentTrainer(
        cfg, train_episodes=train, validation_episodes=validation,
        model=TinyModel(), loss_fn=loss_fn, logger=logger,
    )
    restored_rng = None
    if phase == "resumed":
        from training.recurrent_trainer import _rng_state
        restored_rng = _rng_state()
    else:
        random.seed(1000 + rank)
        np.random.seed(2000 + rank)
        torch.manual_seed(3000 + rank)
    trainer.run()
    torch.save({
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.optimizer.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "completed_updates": trainer.completed_updates,
        "skipped_attempts": trainer.skipped_attempts,
        "scheduled_updates": trainer.scheduled_updates,
        "next_epoch": trainer.next_epoch,
        "best_validation_objective": trainer.best_validation_objective,
        "learning_rates": [group["lr"] for group in trainer.optimizer.optimizer.param_groups],
        "train_loads": train.loaded,
        "val_loads": validation.loaded,
        "validation": [event["val_epoch/objective"] for event in logger.events if "val_epoch/objective" in event],
        "writes": writes,
        "restored_rng": restored_rng,
    }, root / f"{phase}-rank-{rank}.pt")


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_fresh_process_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """Two epochs match a final-epoch checkpoint resumed in fresh processes."""
    _spawn(_continuation_worker, tmp_path, "full", 2)
    _spawn(_continuation_worker, tmp_path, "partial", 1)
    checkpoint = torch.load(tmp_path / "partial" / "checkpoints" / "epoch_0000.pt", weights_only=False)
    assert checkpoint["world_size"] == 2
    assert checkpoint["next_epoch"] == 1
    assert len(checkpoint["rng_states"]) == 2
    assert checkpoint["rng_states"][0]["python"] != checkpoint["rng_states"][1]["python"]
    assert "rng_state" not in checkpoint
    assert not any(key.startswith(("m[", "memory_states", "transient_memory")) for key in checkpoint)
    _spawn(_continuation_worker, tmp_path, "resumed", 2)
    for rank in range(2):
        full = torch.load(tmp_path / f"full-rank-{rank}.pt", weights_only=False)
        partial = torch.load(tmp_path / f"partial-rank-{rank}.pt", weights_only=False)
        resumed = torch.load(tmp_path / f"resumed-rank-{rank}.pt", weights_only=False)
        _assert_nested_equal(resumed["restored_rng"], checkpoint["rng_states"][rank])
        _assert_nested_equal(full["model"], resumed["model"])
        _assert_nested_equal(full["optimizer"], resumed["optimizer"])
        _assert_nested_equal(full["scaler"], resumed["scaler"])
        assert full["completed_updates"] == resumed["completed_updates"] == 4
        assert full["skipped_attempts"] == resumed["skipped_attempts"] == 0
        assert full["scheduled_updates"] == resumed["scheduled_updates"] == 4
        assert full["next_epoch"] == resumed["next_epoch"] == 2
        assert full["learning_rates"] == resumed["learning_rates"]
        assert full["best_validation_objective"] == resumed["best_validation_objective"]
        if rank == 0:
            assert full["validation"][-1] == resumed["validation"][-1]
        else:
            assert full["validation"] == resumed["validation"] == []
        assert full["train_loads"][:2] == partial["train_loads"]
        assert full["train_loads"][2:] == resumed["train_loads"]
        assert full["val_loads"][len(partial["val_loads"]):] == resumed["val_loads"]
        if rank == 0:
            assert "epoch_0001.pt" in full["writes"]
            assert "epoch_0000.pt" not in full["writes"]
            assert "epoch_0000.pt" in partial["writes"]
            assert "epoch_0001.pt" in resumed["writes"]
        else:
            assert full["writes"] == partial["writes"] == resumed["writes"] == []


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_epoch_files_follow_five_and_final_cadence(tmp_path: Path) -> None:
    """Only the fifth and final epochs receive numbered files in six epochs."""
    _spawn(_continuation_worker, tmp_path, "cadence", 6)
    first = torch.load(tmp_path / "cadence-rank-0.pt", weights_only=False)
    second = torch.load(tmp_path / "cadence-rank-1.pt", weights_only=False)
    assert sorted(path.name for path in (tmp_path / "cadence" / "checkpoints").glob("epoch_*.pt")) == [
        "epoch_0004.pt", "epoch_0005.pt",
    ]
    assert (tmp_path / "cadence" / "checkpoints" / "best.pt").exists()
    assert first["writes"].count("epoch_0004.pt") == first["writes"].count("epoch_0005.pt") == 1
    assert second["writes"] == []


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_custom_checkpoint_interval(tmp_path: Path) -> None:
    """A two-epoch interval writes its boundary and the third final epoch."""
    _spawn(_continuation_worker, tmp_path, "custom_cadence", 3)
    assert sorted(path.name for path in (tmp_path / "custom_cadence" / "checkpoints").glob("epoch_*.pt")) == [
        "epoch_0001.pt", "epoch_0002.pt",
    ]


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_keeps_two_latest_epoch_checkpoints(tmp_path: Path) -> None:
    """Three numbered writes retain the latest two and separate best state."""
    _spawn(_continuation_worker, tmp_path, "retention", 3)
    first = torch.load(tmp_path / "retention-rank-0.pt", weights_only=False)
    second = torch.load(tmp_path / "retention-rank-1.pt", weights_only=False)
    directory = tmp_path / "retention" / "checkpoints"
    assert sorted(path.name for path in directory.glob("epoch_*.pt")) == [
        "epoch_0001.pt", "epoch_0002.pt",
    ]
    assert (directory / "best.pt").exists()
    assert {"epoch_0000.pt", "epoch_0001.pt", "epoch_0002.pt"} <= set(first["writes"])
    assert second["writes"] == []


def _failure_worker(rank: int, directory: str, port: int, mode: str) -> None:
    """Capture each rank's checkpoint save or resume rejection without training."""
    import os
    import training.distributed_recurrent_trainer as module

    torch.set_num_threads(1)
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    root = Path(directory)
    windows = tuple(_Window(f"train-{index}", float(index + 1)) for index in range(3))
    if mode == "manifest":
        windows = tuple(reversed(windows))
    cfg = config(root / f"failure-{mode}", epochs=2, train_limit=3, val_limit=3)
    cfg["training"]["scheduled_updates"] = 4
    if mode != "save":
        checkpoint = root / ("missing.pt" if mode == "load" else "partial/checkpoints/epoch_0000.pt")
        if mode == "different_paths" and rank == 1:
            checkpoint = root / "rank-one-valid.pt"
        cfg["checkpoint"]["resume_checkpoint_path"] = str(checkpoint)
    if mode == "frame":
        cfg["sequence"] = {"total_frames": 6, "segment_frames": 2, "num_segments": 3, "backprop_mode": "full"}
    if mode == "optimizer":
        cfg["optim"]["optimizer"]["weight_decay"] = 0.1
    if mode == "horizon":
        cfg["training"]["scheduled_updates"] = 5
    if mode == "save" and rank == 0:
        def fail_save(*_args):
            """Inject a rank-zero checkpoint write failure."""
            raise OSError("injected rank-zero save failure")
        module.robust_torch_save = fail_save
    if mode == "rank_zero_load" and rank == 0:
        def fail_load(*_args, **_kwargs):
            """Inject a rank-zero checkpoint read failure."""
            raise OSError("injected rank-zero load failure")
        torch.load = fail_load
    if mode == "same_path_different_contents" and rank == 1:
        import builtins

        original_open = builtins.open

        def alternate_open(file, mode="r", *args, **kwargs):
            """Read a second valid checkpoint through the same configured path."""
            if os.fspath(file) == str(checkpoint) and mode == "rb":
                file = root / "rank-one-valid.pt"
            return original_open(file, mode, *args, **kwargs)

        builtins.open = alternate_open
    error = None
    try:
        trainer = DistributedRecurrentTrainer(
            cfg, train_episodes=_Source(windows),
            validation_episodes=_Source(tuple(_Window(f"val-{index}", float(index + 4)) for index in range(3))),
            model=TinyModel(), loss_fn=loss_fn, logger=RecordingLogger(),
        )
        trainer.run()
    except Exception as exc:
        error = str(exc)
    torch.save({"error": error}, root / f"failure-{mode}-rank-{rank}.pt")


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("mode", ["manifest", "frame", "optimizer", "horizon", "world_size", "load", "rank_zero_load", "save"])
def test_e01a_distributed_checkpoint_failures_reach_both_ranks(tmp_path: Path, mode: str) -> None:
    """Changed inputs and rank-zero I/O exceptions fail both ranks before work."""
    _spawn(_continuation_worker, tmp_path, "partial", 1)
    if mode == "world_size":
        path = tmp_path / "partial" / "checkpoints" / "epoch_0000.pt"
        state = torch.load(path, weights_only=False)
        state["world_size"] = 3
        torch.save(state, path)
    _spawn(_failure_worker, tmp_path, mode)
    errors = [torch.load(tmp_path / f"failure-{mode}-rank-{rank}.pt", weights_only=False)["error"] for rank in range(2)]
    assert errors[0] == errors[1]
    assert errors[0] is not None
    expected = {
        "manifest": "manifest or configuration",
        "frame": "manifest or configuration",
        "optimizer": "manifest or configuration",
        "horizon": "manifest or configuration",
        "world_size": "world size",
        "load": "FileNotFoundError",
        "rank_zero_load": "injected rank-zero load failure",
        "save": "injected rank-zero save failure",
    }
    assert expected[mode] in errors[0]



@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_rejects_different_valid_resume_paths_on_both_ranks(tmp_path: Path) -> None:
    """Two valid checkpoints with different model states cannot silently resume."""
    _spawn(_continuation_worker, tmp_path, "partial", 1)
    original = tmp_path / "partial" / "checkpoints" / "epoch_0000.pt"
    alternate = torch.load(original, weights_only=False)
    first_key = next(iter(alternate["model"]))
    alternate["model"][first_key] = alternate["model"][first_key] + 1
    torch.save(alternate, tmp_path / "rank-one-valid.pt")
    _spawn(_failure_worker, tmp_path, "different_paths")
    errors = [
        torch.load(tmp_path / f"failure-different_paths-rank-{rank}.pt", weights_only=False)["error"]
        for rank in range(2)
    ]
    assert errors[0] == errors[1]
    assert "resume checkpoint paths differ across ranks" in errors[0]



@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_e01a_distributed_rejects_different_contents_at_same_resume_path(tmp_path: Path) -> None:
    """A shared pathname cannot restore two different valid checkpoint files."""
    _spawn(_continuation_worker, tmp_path, "partial", 1)
    original = tmp_path / "partial" / "checkpoints" / "epoch_0000.pt"
    alternate = torch.load(original, weights_only=False)
    first_key = next(iter(alternate["model"]))
    alternate["model"][first_key] = alternate["model"][first_key] + 1
    torch.save(alternate, tmp_path / "rank-one-valid.pt")
    _spawn(_failure_worker, tmp_path, "same_path_different_contents")
    errors = [
        torch.load(tmp_path / f"failure-same_path_different_contents-rank-{rank}.pt", weights_only=False)["error"]
        for rank in range(2)
    ]
    assert errors[0] == errors[1]
    assert "resume checkpoint contents differ across ranks" in errors[0]
