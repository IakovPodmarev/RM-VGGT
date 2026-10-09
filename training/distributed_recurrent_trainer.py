"""Single-node, one-complete-episode-per-rank recurrent training."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
import hashlib
import math
import os
from typing import Any

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from training.distributed_update import DistributedGradientCoordinator
from training.episode_grouping import plan_episode_groups
from training.recurrent_episode import RecurrentEpisodeModule
from training.recurrent_loss import compute_recurrent_losses
from training.recurrent_sequence import run_recurrent_sequence
from training.recurrent_step import run_recurrent_train_step
from training.recurrent_trainer import RecurrentTrainer, _mean_metrics, _metrics, _restore_rng, _rng_state, _scalar
from training.train_utils.checkpoint import robust_torch_save


class _NullLogger:
    """Discard nonzero-rank events while preserving the logger protocol."""

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        """Accept one event without producing an external log."""


class DistributedRecurrentTrainer(RecurrentTrainer):
    """Reuse recurrent setup and run synchronized complete-episode updates."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        """Bind torchrun rank, initialize DDP, and retain underlying model groups.

        Training and validation sources must expose eligible_windows and
        load_window. A requested resume is applied after DDP initialization.
        """
        cfg = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        resume_path = cfg["checkpoint"].get("resume_checkpoint_path")
        if resume_path and cfg["checkpoint"].get("pretrained_checkpoint_path"):
            raise ValueError("pretrained initialization and full resume are mutually exclusive")
        cfg["checkpoint"]["resume_checkpoint_path"] = None
        checkpoint_every_epochs = cfg["checkpoint"].get("save_every_epochs", 5)
        if isinstance(checkpoint_every_epochs, bool) or not isinstance(checkpoint_every_epochs, int) or checkpoint_every_epochs <= 0:
            raise ValueError("checkpoint.save_every_epochs must be a positive integer")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("invalid torchrun rank configuration")
        device = torch.device(cfg.get("device", "cuda"))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            cfg["device"] = f"cuda:{local_rank}"
        elif device.type != "cpu":
            raise ValueError("distributed recurrent training requires CPU or CUDA")
        cfg["training"]["episodes_per_update"] = world_size
        self.rank, self.world_size, self.local_rank = rank, world_size, local_rank
        self.checkpoint_every_epochs = checkpoint_every_epochs
        if dist.is_initialized():
            raise RuntimeError("process group is already initialized")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", init_method="env://")
        try:
            if rank != 0:
                kwargs["logger"] = _NullLogger()
            super().__init__(cfg, **kwargs)
            for source in (self.train_episodes, self.validation_episodes):
                if not callable(getattr(source, "eligible_windows", None)) or not callable(getattr(source, "load_window", None)):
                    raise ValueError("distributed episode sources need eligible_windows and load_window")
            self.episode_module = DistributedDataParallel(
                RecurrentEpisodeModule(self.model, num_segments=self.num_segments, segment_frames=self.segment_frames),
                device_ids=[local_rank] if device.type == "cuda" else None,
            )
            self.gradient_coordinator = DistributedGradientCoordinator()
            if resume_path:
                self.resume_from_checkpoint(resume_path)
        except BaseException:
            dist.destroy_process_group()
            raise

    def _reports(self, report: dict[str, Any], *, epoch: int) -> list[dict[str, Any]]:
        """Gather rank evidence and fail all ranks after a local error."""
        reports: list[dict[str, Any]] = [None] * self.world_size
        dist.all_gather_object(reports, report)
        failures = {rank: item["error"] for rank, item in enumerate(reports) if item.get("error")}
        if failures:
            if self.rank == 0:
                self._log_episode("distributed", {"rank_failures": failures}, epoch)
            raise RuntimeError(f"distributed episode failed by rank: {failures}")
        return reports

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one DDP forward/backward per rank and planned episode group."""
        self.episode_module.train()
        if self.model.aggregator.training:
            raise ValueError("frozen aggregator must remain in evaluation mode")
        windows = tuple(self.train_episodes.eligible_windows())
        if self.train_limit is not None:
            if self.train_limit > len(windows):
                raise ValueError("training limit exceeds eligible episodes")
            windows = windows[:self.train_limit]
        rows: list[dict[str, Any]] = []
        for assignments in plan_episode_groups(windows, seed=self.seed, epoch=epoch, world_size=self.world_size):
            assignment = assignments[self.rank]
            raw = None
            error = None
            try:
                raw = self.train_episodes.load_window(assignment.window)
            except Exception as exc:
                error = repr(exc)
            self._reports({"error": error}, epoch=epoch)
            start = self._measure_start()
            recorded: list[Mapping[str, Any]] = []
            result = None
            error = None
            try:
                result = run_recurrent_train_step(
                    model=self.model, segments=self._prepared(raw, recorded), loss_fn=self.loss_fn,
                    optimizer=self.optimizer, scaler=self.scaler, gradient_clipper=self.gradient_clipper,
                    autocast_enabled=self.autocast_enabled, autocast_dtype=self.autocast_dtype,
                    autocast_device_type=self.device.type,
                    scheduler_progress=(self.completed_updates + 1) / self.scheduled_updates,
                    num_segments=self.num_segments, segment_frames=self.segment_frames,
                    diagnostic=self.diagnostic, episode_module=self.episode_module,
                    objective_scale=assignment.objective_scale,
                    gradient_coordinator=self.gradient_coordinator,
                )
            except Exception as exc:
                error = repr(exc)
            self._reports({"error": error}, epoch=epoch)
            metrics = None
            peak = None
            error = None
            try:
                if result.optimizer_ran:
                    self.completed_updates += 1
                else:
                    self.skipped_attempts += 1
                elapsed, peak = self._measure_end(start)
                if assignment.is_real:
                    metrics = _metrics(result.losses, recorded, weights=self.loss_weights, diagnostics=result.sequence.diagnostics)
                    metrics.update({
                        "identity": assignment.identity, "elapsed_seconds": elapsed,
                        "peak_memory_bytes": peak,
                        "gradient_norm": _scalar(result.gradient_norm) if result.gradient_norm is not None else None,
                        "scheduler_progress": self.completed_updates / self.scheduled_updates,
                        "optimizer_ran": result.optimizer_ran, "skipped_attempts": self.skipped_attempts,
                    })
                    for group in self.optimizer.optimizer.param_groups:
                        metrics["learning_rate/{}".format(group["name"])] = float(group["lr"])
            except Exception as exc:
                error = repr(exc)
            reports = self._reports({"error": error, "metrics": metrics, "peak": peak, "optimizer_ran": result.optimizer_ran}, epoch=epoch)
            if len({item["optimizer_ran"] for item in reports}) != 1:
                raise RuntimeError("optimizer decisions differ across ranks")
            for item in reports:
                if item["metrics"] is not None:
                    rows.append(item["metrics"])
                    if self.rank == 0:
                        self._log_episode("train", item["metrics"], epoch)
            if self.rank == 0:
                self._log_episode("distributed", {"peak_memory_bytes_by_rank": [item["peak"] for item in reports], "rank_failures": {}}, epoch)
            del result, recorded
        summary = _mean_metrics(rows)
        if self.rank == 0:
            self._log_episode("train_epoch", summary, epoch)
        return summary

    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Shard fixed windows and reduce the global episode-weighted objective."""
        self.model.eval()
        windows = tuple(self.validation_episodes.eligible_windows())
        windows = windows[:self.validation_limit] if self.validation_limit is not None else windows
        if not windows:
            raise ValueError("validation manifest contains no eligible episodes")
        local_rows: list[dict[str, Any]] = []
        error = None
        try:
            for window in windows[self.rank::self.world_size]:
                raw = self.validation_episodes.load_window(window)
                start = self._measure_start()
                recorded: list[Mapping[str, Any]] = []
                with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                    sequence = run_recurrent_sequence(
                        self.model, self._prepared(raw, recorded),
                        num_segments=self.num_segments, segment_frames=self.segment_frames,
                    )
                    losses = compute_recurrent_losses(sequence, recorded, self.loss_fn, num_segments=self.num_segments)
                elapsed, peak = self._measure_end(start)
                metrics = _metrics(losses, recorded, weights=self.loss_weights, diagnostics=sequence.diagnostics)
                metrics.update({"identity": window.identity, "elapsed_seconds": elapsed, "peak_memory_bytes": peak})
                local_rows.append(metrics)
        except Exception as exc:
            error = repr(exc)
        reports = self._reports({"error": error, "rows": local_rows}, epoch=epoch)
        rows = [row for item in reports for row in item["rows"]]
        local_sum = sum(row["objective"] for row in local_rows)
        totals = torch.tensor([local_sum, len(local_rows)], device=self.device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        summary = _mean_metrics(rows)
        summary["objective"] = float(totals[0].item() / totals[1].item())
        if self.rank == 0:
            for row in rows:
                self._log_episode("val", row, epoch)
            self._log_episode("val_epoch", dict(summary, real_episodes=int(totals[1].item())), epoch)
        return summary

    def _resume_signature(self) -> dict[str, Any]:
        """Describe the ordered episode manifests and settings that fix epoch work."""
        def manifest(source: Any, limit: int | None) -> tuple[Any, ...]:
            """Record every ordered eligible window, including frame IDs where present."""
            windows = tuple(source.eligible_windows())
            return tuple(
                (window.identity, asdict(window) if is_dataclass(window) else tuple(getattr(window, "frame_ids", ())))
                for window in (windows[:limit] if limit is not None else windows)
            )

        cfg = self.config
        return {
            "world_size": self.world_size,
            "train_manifest": manifest(self.train_episodes, self.train_limit),
            "validation_manifest": manifest(self.validation_episodes, self.validation_limit),
            "sequence": cfg["sequence"],
            "episode_sources": cfg.get("episode_sources"),
            "dataset_split": cfg.get("dataset_split"),
            "seed_value": self.seed,
            "training": cfg["training"],
            "validation": cfg["validation"],
            "optim": cfg["optim"],
            "model": cfg.get("model"),
            "loss": cfg.get("loss"),
            "memory": cfg.get("memory"),
        }

    def save_checkpoint(self, completed_epoch: int, validation_objective: float) -> None:
        """Gather rank RNG and write periodic, final, or improved state on rank zero.

        Epoch files are written at the configured interval and final epoch,
        retaining the two most recently written numbered files. Best state is
        separate and written on each improvement. Rank-zero write and cleanup
        failures or incompatible manifests are reported to every rank.
        """
        local = None
        error = None
        try:
            local = {"rng": _rng_state(), "signature": self._resume_signature()}
        except Exception as exc:
            error = repr(exc)
        self._reports({"error": error}, epoch=completed_epoch)
        gathered: list[dict[str, Any]] = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        error = None
        if self.rank == 0:
            try:
                signature = gathered[0]["signature"]
                if any(item["signature"] != signature for item in gathered[1:]):
                    raise ValueError("distributed episode manifests or configuration differ across ranks")
                improved = validation_objective < self.best_validation_objective
                best = float(validation_objective) if improved else self.best_validation_objective
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "scheduler_progress": self.completed_updates / self.scheduled_updates,
                    "scheduled_updates": self.scheduled_updates,
                    "completed_epoch": completed_epoch,
                    "next_epoch": completed_epoch + 1,
                    "completed_updates": self.completed_updates,
                    "skipped_attempts": self.skipped_attempts,
                    "validation_objective": float(validation_objective),
                    "best_validation_objective": best,
                    "config": self.config,
                    "run_metadata": self.run_metadata,
                    "world_size": self.world_size,
                    "resume_signature": signature,
                    "rng_states": [item["rng"] for item in gathered],
                }
                save_epoch = (completed_epoch + 1) % self.checkpoint_every_epochs == 0 or completed_epoch + 1 == self.max_epochs
                if save_epoch or improved:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                if save_epoch:
                    epoch_path = self.checkpoint_dir / f"epoch_{completed_epoch:04d}.pt"
                    robust_torch_save(checkpoint, str(epoch_path))
                if improved:
                    robust_torch_save(checkpoint, str(self.checkpoint_dir / "best.pt"))
                if save_epoch:
                    numbered = sorted(
                        (path for path in self.checkpoint_dir.glob("epoch_*.pt") if path.stem[6:].isdigit()),
                        key=lambda path: (path.stat().st_mtime_ns, path == epoch_path, path.name),
                    )
                    for stale in numbered[:-2]:
                        stale.unlink()
                self.best_validation_objective = best
            except Exception as exc:
                error = repr(exc)
        self._reports({"error": error}, epoch=completed_epoch)
        if self.rank != 0:
            self.best_validation_objective = min(self.best_validation_objective, float(validation_objective))

    def resume_from_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """Restore one verified shared checkpoint and this rank's RNG state.

        Resolve paths and compare bytes from each rank's open file before
        loading from those same handles. CUDA generators are initialized before
        restoration so lazy seeding cannot overwrite saved states. Local open,
        load, and restoration errors are reported to every rank.
        """
        paths: list[str] = [None] * self.world_size
        dist.all_gather_object(paths, os.path.realpath(os.fspath(checkpoint_path)))
        if len(set(paths)) != 1:
            self._reports({"error": "resume checkpoint paths differ across ranks"}, epoch=-1)
        with ExitStack() as checkpoint_files:
            checkpoint_file = None
            digest = None
            error = None
            try:
                checkpoint_file = checkpoint_files.enter_context(open(checkpoint_path, "rb"))
                hasher = hashlib.sha256()
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    hasher.update(chunk)
                digest = hasher.digest()
                checkpoint_file.seek(0)
            except Exception as exc:
                error = repr(exc)
            self._reports({"error": error}, epoch=-1)
            digests: list[bytes] = [None] * self.world_size
            dist.all_gather_object(digests, digest)
            if len(set(digests)) != 1:
                self._reports({"error": "resume checkpoint contents differ across ranks"}, epoch=-1)

            state = None
            error = None
            try:
                state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                required = {
                    "model", "optimizer", "scaler", "scheduler_progress", "scheduled_updates",
                    "completed_epoch", "next_epoch", "completed_updates", "skipped_attempts",
                    "validation_objective", "best_validation_objective", "world_size",
                    "resume_signature", "rng_states",
                }
                if not isinstance(state, Mapping) or not required <= state.keys():
                    raise ValueError("distributed resume checkpoint is missing training state")
                if state["world_size"] != self.world_size or len(state["rng_states"]) != self.world_size:
                    raise ValueError("world size differs from resume checkpoint")
                if state["resume_signature"] != self._resume_signature():
                    raise ValueError("manifest or configuration differs from resume checkpoint")
                if state["scheduled_updates"] != self.scheduled_updates:
                    raise ValueError("scheduled update budget differs from resume checkpoint")
                if state["next_epoch"] != state["completed_epoch"] + 1 or state["next_epoch"] < 1:
                    raise ValueError("resume checkpoint epoch counters are inconsistent")
                if state["completed_updates"] < 0 or state["skipped_attempts"] < 0:
                    raise ValueError("resume checkpoint update counters are invalid")
                if not math.isclose(state["scheduler_progress"], state["completed_updates"] / self.scheduled_updates):
                    raise ValueError("resume checkpoint schedule progress is inconsistent")
            except Exception as exc:
                error = repr(exc)
            self._reports({"error": error}, epoch=-1)
            error = None
            try:
                self.model.load_state_dict(state["model"], strict=True)
                self.model.to(self.device)
                self.optimizer.optimizer.load_state_dict(state["optimizer"])
                for item in self.optimizer.optimizer.state.values():
                    for key, value in item.items():
                        if torch.is_tensor(value):
                            item[key] = value.to(self.device)
                self.scaler.load_state_dict(state["scaler"])
                self.next_epoch = int(state["next_epoch"])
                self.completed_updates = int(state["completed_updates"])
                self.skipped_attempts = int(state["skipped_attempts"])
                self.best_validation_objective = float(state["best_validation_objective"])
                self.run_metadata = dict(state["run_metadata"])
                rank_rng = state["rng_states"][self.rank]
                if rank_rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.get_rng_state_all()
                _restore_rng(rank_rng)
            except Exception as exc:
                error = repr(exc)
            self._reports({"error": error}, epoch=-1)

    def run(self) -> None:
        """Save each completed training and validation epoch, then close DDP."""
        try:
            for epoch in range(self.next_epoch, self.max_epochs):
                self.train_epoch(epoch)
                validation = self.validate_epoch(epoch)
                self.save_checkpoint(epoch, validation["objective"])
                self.next_epoch = epoch + 1
        finally:
            if self.rank == 0 and hasattr(self.logger, "finish"):
                self.logger.finish()
            dist.destroy_process_group()
