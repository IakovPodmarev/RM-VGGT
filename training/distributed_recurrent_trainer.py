"""Single-node, one-complete-episode-per-rank recurrent training."""

from __future__ import annotations

from collections.abc import Mapping
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
from training.recurrent_trainer import RecurrentTrainer, _mean_metrics, _metrics, _scalar


class _NullLogger:
    """Discard nonzero-rank events while preserving the logger protocol."""

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        """Accept one event without producing an external log."""


class DistributedRecurrentTrainer(RecurrentTrainer):
    """Reuse recurrent setup and run synchronized complete-episode updates."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        """Bind torchrun rank, initialize DDP, and retain underlying model groups.

        Training and validation sources must expose eligible_windows and
        load_window. Full distributed checkpoint continuation is unsupported.
        """
        cfg = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        if cfg["checkpoint"].get("resume_checkpoint_path"):
            raise ValueError("distributed checkpoint resume is not implemented")
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

    def run(self) -> None:
        """Run bounded epochs and close the process group without checkpointing."""
        try:
            for epoch in range(self.next_epoch, self.max_epochs):
                self.train_epoch(epoch)
                self.validate_epoch(epoch)
                self.next_epoch = epoch + 1
        finally:
            if self.rank == 0 and hasattr(self.logger, "finish"):
                self.logger.finish()
            dist.destroy_process_group()
