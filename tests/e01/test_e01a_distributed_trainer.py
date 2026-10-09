"""Two-rank bounded train and validation integration for complete episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket

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
    assert summary["val_epoch/objective"] == pytest.approx(first["reference"], abs=1e-7)
    assert first["reference"] == pytest.approx(second["reference"], abs=1e-7)
    assert not list((tmp_path / "checkpoints").glob("*.pt"))


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
