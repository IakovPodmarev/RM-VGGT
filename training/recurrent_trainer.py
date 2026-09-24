"""Single-process lifecycle for recurrent episode training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any
import math
import random
import subprocess
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

try:
    from training.data.episode import normalize_segment, split_episode, validate_segment_dimensions
    from training.recurrent_loss import compute_recurrent_losses
    from training.recurrent_sequence import run_recurrent_sequence
    from training.recurrent_step import run_recurrent_train_step
    from training.train_utils.checkpoint import robust_torch_save
    from training.train_utils.optimizer import construct_optimizers
except ModuleNotFoundError:
    from data.episode import normalize_segment, split_episode, validate_segment_dimensions
    from recurrent_loss import compute_recurrent_losses
    from recurrent_sequence import run_recurrent_sequence
    from recurrent_step import run_recurrent_train_step
    from train_utils.checkpoint import robust_torch_save
    from train_utils.optimizer import construct_optimizers


Episode = Mapping[str, Any]
EpisodeSource = Callable[[int], Iterable[Episode]]
LossFunction = Callable[[Mapping[str, Any], Episode], Mapping[str, torch.Tensor]]


def transfer_segment_to_device(segment: Episode, device: torch.device) -> dict[str, Any]:
    """Copy one normalized CPU segment to a device without changing dtypes.

    Return a new mapping with tensor values on the one process-local device.
    Preserve masks, integer metadata, RGB input convention, and non-tensor
    metadata. Raise TypeError for a non-mapping or ValueError for a non-CPU
    tensor. Neither cast nor mutate input data.
    """
    if not isinstance(segment, Mapping):
        raise TypeError("segment must be a mapping")

    def transfer(value: Any) -> Any:
        """Recursively copy one value while keeping its type and scalar metadata."""
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                raise ValueError("episode transfer accepts CPU tensors only")
            return value.to(device=device)
        if isinstance(value, Mapping):
            return {key: transfer(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [transfer(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(transfer(nested) for nested in value)
        return value

    return {key: transfer(value) for key, value in segment.items()}


def iter_prepared_segments(
    raw_episode: Episode,
    *,
    device: torch.device,
    total_frames: int = 24,
    segment_frames: int = 8,
) -> Iterable[dict[str, Any]]:
    """Split CPU frames first, then normalize and transfer each requested segment.

    Yield temporal segments with frame IDs, segment metadata, and ordered
    identities preserved. Propagate episode type and geometry errors. The
    generator body executes on first request; later preparation occurs only
    after the consumer has completed the preceding model call.
    """
    raw_segments = split_episode(raw_episode, total_frames=total_frames, segment_frames=segment_frames)
    for raw_segment in raw_segments:
        normalized_segment = normalize_segment(raw_segment)
        yield transfer_segment_to_device(normalized_segment, device)


def load_pretrained_components(
    model: nn.Module, checkpoint_path: str | Path
) -> tuple[list[str], list[str]]:
    """Load required encoder and enabled-head weights, leaving recurrence fresh.

    Return missing recurrent keys and allowed extra pretrained head keys.
    Reject absent encoder or enabled-head keys, unknown extra keys, and missing
    model state with ValueError. Tensor shape mismatch raises RuntimeError.
    This path never restores optimizer state or training progress.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError("pretrained checkpoint has no model state")
    expected = model.state_dict()
    required_prefixes = tuple(
        name + "." for name in ("aggregator", "camera_head", "depth_head")
        if getattr(model, name, None) is not None
    )
    required = {key for key in expected if key.startswith(required_prefixes)}
    absent = sorted(required - state.keys())
    if absent:
        raise ValueError(f"pretrained checkpoint is missing required weights: {absent}")
    unexpected = sorted(set(state) - set(expected))
    incompatible = [key for key in unexpected if not key.startswith(("point_head.", "track_head."))]
    if incompatible:
        raise ValueError(f"pretrained checkpoint has unexpected weights: {incompatible}")
    compatible = {key: value for key, value in state.items() if key in expected and key in required}
    missing, _ = model.load_state_dict(compatible, strict=False)
    recurrent_missing = sorted(key for key in missing if key not in required)
    return recurrent_missing, unexpected


def _git_metadata() -> dict[str, Any]:
    """Read local revision and dirty status without changing repository state."""
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    """Restore all recorded generators for deterministic epoch continuation."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _scalar(value: Any) -> float:
    """Detach a scalar tensor or convert a numeric result for graph-free logs."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("logged loss terms must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


def _metrics(
    losses: Any,
    segments: list[Episode],
    *,
    weights: Mapping[str, float],
    diagnostics: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detach aggregate, segment, gate, and audit values for logging."""
    values: dict[str, Any] = {"objective": _scalar(losses.objective)}
    per_key: dict[str, list[float]] = {}
    for index, loss_map in enumerate(losses.segment_losses):
        for key, value in loss_map.items():
            number = _scalar(value)
            values[f"segment_{index}/{key}"] = number
            per_key.setdefault(key, []).append(number)
        camera = _scalar(loss_map["loss_camera"]) * weights["camera"] if "loss_camera" in loss_map else 0.0
        depth = sum(_scalar(value) for key, value in loss_map.items() if key.startswith(("loss_conf_depth", "loss_reg_depth", "loss_grad_depth"))) * weights["depth"]
        values[f"segment_{index}/camera"] = camera
        values[f"segment_{index}/depth"] = depth
        if diagnostics is not None:
            for name, diagnostic in diagnostics[index].items():
                if torch.is_tensor(diagnostic):
                    values[f"segment_{index}/gate/{name}"] = float(diagnostic.detach().float().mean().cpu().item())
        ids = segments[index].get("ids")
        if ids is not None:
            values[f"segment_{index}/frame_ids"] = ids.detach().cpu().tolist() if torch.is_tensor(ids) else ids
        if "seq_name" in segments[index]:
            values[f"segment_{index}/batch_identities"] = list(segments[index]["seq_name"])
    for key, items in per_key.items():
        values[key] = sum(items) / len(items)
    values["camera"] = sum(values[f"segment_{i}/camera"] for i in range(len(segments))) / len(segments)
    values["depth"] = sum(values[f"segment_{i}/depth"] for i in range(len(segments))) / len(segments)
    return values


def _mean_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric episode metrics without retaining tensors or audit lists."""
    if not items:
        raise ValueError("epoch contains no episodes")
    keys = [key for key, value in items[0].items() if isinstance(value, (float, int)) and not isinstance(value, bool)]
    return {key: sum(float(item[key]) for item in items) / len(items) for key in keys}


class _WandbLogger:
    """Own one configured W&B run behind the recording-logger protocol."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Start a W&B run only when the configured logging mode requests it."""
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B logging requires the wandb dependency") from error
        self.run = wandb.init(
            project=config.get("project"),
            name=config.get("run_name"),
            mode=config.get("mode", "online"),
            config=config.get("resolved_config"),
        )

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        """Send detached scalar and audit fields to the active run."""
        self.run.log(values, step=step)

    def finish(self) -> None:
        """Flush and close the active W&B run."""
        self.run.finish()


class RecurrentTrainer:
    """Own one-device recurrent epochs, validation, logging, and checkpoints.

    The sequence helper owns memory carry, the loss helper owns loss arithmetic,
    and the train step owns backward, clipping, schedule, optimizer, and scaler
    actions. Resume is supported at epoch boundaries only.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        train_episodes: EpisodeSource | None = None,
        validation_episodes: EpisodeSource | None = None,
        model: nn.Module | None = None,
        loss_fn: LossFunction | None = None,
        logger: Any | None = None,
    ) -> None:
        """Construct from resolved config and optional test collaborators.

        Reject missing episode sources, unsupported sequence or accumulation
        settings, and invalid one-device limits. Initialize model, loss,
        optimizer, scaler, clipper, logger, and optional preload or full resume.
        Construction performs no sequence update or distributed setup.
        """
        self.config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        cfg = self.config
        self.device = torch.device(cfg.get("device", "cuda"))
        if self.device.type not in {"cpu", "cuda"} or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("recurrent training requires one available CPU or CUDA device")
        sources = cfg.get("episode_sources") or {}
        self.train_episodes = train_episodes or self._configured_source(sources.get("train"))
        self.validation_episodes = validation_episodes or self._configured_source(sources.get("validation"))
        if self.train_episodes is None or self.validation_episodes is None:
            raise ValueError("recurrent training requires sequential train and validation episode sources")
        sequence = cfg["sequence"]
        validate_segment_dimensions(sequence["total_frames"], sequence["segment_frames"], sequence["num_segments"])
        if sequence.get("backprop_mode") != "full":
            raise ValueError("recurrent training supports full backpropagation only")
        training = cfg["training"]
        validation = cfg["validation"]
        if training.get("episode_batch_size") != 1 or training.get("accum_steps") != 1:
            raise ValueError("recurrent training requires episode batch size and accumulation of one")
        self.train_limit = int(training["max_episodes_per_epoch"])
        self.validation_limit = int(validation["max_episodes_per_epoch"])
        self.max_epochs = int(cfg["max_epochs"])
        self.scheduled_updates = int(training["scheduled_updates"])
        if min(self.train_limit, self.validation_limit, self.max_epochs, self.scheduled_updates) <= 0:
            raise ValueError("episode limits, epochs, and scheduled updates must be positive")
        if self.scheduled_updates < self.max_epochs * self.train_limit:
            raise ValueError("scheduled updates must cover the configured training budget")
        self.total_frames = int(sequence["total_frames"])
        self.segment_frames = int(sequence["segment_frames"])
        self.num_segments = int(sequence["num_segments"])
        self.seed = int(cfg.get("seed_value", 0))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.model = model if model is not None else hydra.utils.instantiate(OmegaConf.create(cfg["model"]))
        self.model.to(self.device)
        self.loss_fn = loss_fn if loss_fn is not None else hydra.utils.instantiate(OmegaConf.create(cfg["loss"]))
        self.optimizer = construct_optimizers(self.model, OmegaConf.create(cfg["optim"]))[0]
        self.gradient_clipper = hydra.utils.instantiate(OmegaConf.create(cfg["optim"]["gradient_clip"]))
        self.gradient_clipper.setup_clipping(self.model)
        amp = cfg["optim"]["amp"]
        self.autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp["amp_dtype"]]
        self.autocast_enabled = bool(amp["enabled"]) and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.autocast_enabled and self.autocast_dtype == torch.float16)
        self.loss_weights = {
            "camera": float((cfg.get("loss", {}).get("camera") or {}).get("weight", 1.0)),
            "depth": float((cfg.get("loss", {}).get("depth") or {}).get("weight", 1.0)),
        }
        self.next_epoch = 0
        self.completed_updates = 0
        self.best_validation_objective = math.inf
        self.checkpoint_dir = Path(cfg["checkpoint"]["save_dir"])
        self.run_metadata = {
            "experiment_id": cfg.get("experiment_id", cfg.get("exp_name")),
            "seed": self.seed,
            "dataset_split": cfg.get("dataset_split"),
            "memory_settings": cfg.get("memory", cfg.get("e01a")),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
            "optimizer_group_counts": {group["name"]: sum(p.numel() for p in group["params"]) for group in self.optimizer.optimizer.param_groups},
            **_git_metadata(),
        }
        logging_cfg = dict(cfg.get("logging", {}))
        logging_cfg["resolved_config"] = cfg
        self.logger = logger if logger is not None else _WandbLogger(logging_cfg)
        self.logger.log(dict(self.run_metadata, config=cfg), step=0)
        checkpoint_cfg = cfg["checkpoint"]
        if checkpoint_cfg.get("pretrained_checkpoint_path") and checkpoint_cfg.get("resume_checkpoint_path"):
            raise ValueError("pretrained initialization and full resume are mutually exclusive")
        if checkpoint_cfg.get("pretrained_checkpoint_path"):
            missing, unexpected = load_pretrained_components(self.model, checkpoint_cfg["pretrained_checkpoint_path"])
            self.logger.log({"pretrained_missing_keys": missing, "pretrained_unexpected_keys": unexpected}, step=0)
        if checkpoint_cfg.get("resume_checkpoint_path"):
            self.resume_from_checkpoint(checkpoint_cfg["resume_checkpoint_path"])

    @staticmethod
    def _configured_source(spec: Any) -> EpisodeSource | None:
        """Instantiate an explicitly configured epoch-indexed episode source."""
        if spec is None:
            return None
        source = hydra.utils.instantiate(OmegaConf.create(spec)) if isinstance(spec, Mapping) else spec
        if not callable(source):
            raise ValueError("configured episode source must be epoch-indexed and callable")
        return source

    def _measure_start(self) -> float:
        """Synchronize prior CUDA work and reset peak allocation for one episode."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def _measure_end(self, start: float) -> tuple[float, int | None]:
        """Synchronize current CUDA work and return elapsed seconds and peak bytes."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
        else:
            peak = None
        return time.perf_counter() - start, peak

    def _prepared(self, raw: Episode, recorded: list[Episode]) -> Iterable[dict[str, Any]]:
        """Yield prepared segments and retain consumed targets only through loss."""
        for segment in iter_prepared_segments(raw, device=self.device, total_frames=self.total_frames, segment_frames=self.segment_frames):
            recorded.append(segment)
            yield segment

    def _log_episode(self, phase: str, metrics: dict[str, Any], epoch: int) -> None:
        """Emit one graph-free phase-prefixed event with update and epoch metadata."""
        payload = {f"{phase}/{key}": value for key, value in metrics.items()}
        payload.update({"epoch": epoch, "completed_updates": self.completed_updates})
        self.logger.log(payload, step=self.completed_updates)

    def run(self) -> None:
        """Run training and validation every epoch, then save complete state."""
        for epoch in range(self.next_epoch, self.max_epochs):
            self.train_epoch(epoch)
            validation = self.validate_epoch(epoch)
            self.save_checkpoint(epoch, validation["objective"])
            self.next_epoch = epoch + 1
        if hasattr(self.logger, "finish"):
            self.logger.finish()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run bounded full-BPTT episode updates and return detached means."""
        self.model.train()
        if self.model.aggregator.training:
            raise ValueError("frozen aggregator must remain in evaluation mode")
        rows: list[dict[str, Any]] = []
        for raw in islice(self.train_episodes(epoch), self.train_limit):
            start = self._measure_start()
            recorded: list[Episode] = []
            result = run_recurrent_train_step(
                model=self.model,
                segments=self._prepared(raw, recorded),
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                scaler=self.scaler,
                gradient_clipper=self.gradient_clipper,
                autocast_enabled=self.autocast_enabled,
                autocast_dtype=self.autocast_dtype,
                autocast_device_type=self.device.type,
                scheduler_progress=(self.completed_updates + 1) / self.scheduled_updates,
                num_segments=self.num_segments,
                segment_frames=self.segment_frames,
            )
            self.completed_updates += 1
            elapsed, peak = self._measure_end(start)
            metrics = _metrics(result.losses, recorded, weights=self.loss_weights, diagnostics=result.sequence.diagnostics)
            metrics.update({
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": peak,
                "gradient_norm": _scalar(result.gradient_norm) if result.gradient_norm is not None else None,
                "scheduler_progress": self.completed_updates / self.scheduled_updates,
            })
            for group in self.optimizer.optimizer.param_groups:
                metrics[f"learning_rate/{group['name']}"] = float(group["lr"])
            self._log_episode("train", metrics, epoch)
            rows.append(metrics)
            del result, recorded
        summary = _mean_metrics(rows)
        self._log_episode("train_epoch", summary, epoch)
        return summary

    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Run bounded no-grad episode validation without changing update state."""
        self.model.eval()
        rows: list[dict[str, Any]] = []
        for raw in islice(self.validation_episodes(epoch), self.validation_limit):
            start = self._measure_start()
            recorded: list[Episode] = []
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                sequence = run_recurrent_sequence(
                    self.model, self._prepared(raw, recorded),
                    num_segments=self.num_segments, segment_frames=self.segment_frames,
                )
                losses = compute_recurrent_losses(sequence, recorded, self.loss_fn, num_segments=self.num_segments)
            elapsed, peak = self._measure_end(start)
            metrics = _metrics(losses, recorded, weights=self.loss_weights, diagnostics=sequence.diagnostics)
            metrics.update({"elapsed_seconds": elapsed, "peak_memory_bytes": peak})
            self._log_episode("val", metrics, epoch)
            rows.append(metrics)
            del sequence, losses, recorded
        summary = _mean_metrics(rows)
        self._log_episode("val_epoch", summary, epoch)
        return summary

    def save_checkpoint(self, completed_epoch: int, validation_objective: float) -> None:
        """Save epoch state and replace best state on validation improvement."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        improved = validation_objective < self.best_validation_objective
        if improved:
            self.best_validation_objective = float(validation_objective)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler_progress": self.completed_updates / self.scheduled_updates,
            "scheduled_updates": self.scheduled_updates,
            "completed_epoch": completed_epoch,
            "next_epoch": completed_epoch + 1,
            "completed_updates": self.completed_updates,
            "validation_objective": float(validation_objective),
            "best_validation_objective": self.best_validation_objective,
            "config": self.config,
            "run_metadata": self.run_metadata,
            "rng_state": _rng_state(),
        }
        robust_torch_save(checkpoint, str(self.checkpoint_dir / f"epoch_{completed_epoch:04d}.pt"))
        if improved:
            robust_torch_save(checkpoint, str(self.checkpoint_dir / "best.pt"))

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Strictly restore full epoch-boundary state to the intended device."""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "model", "optimizer", "scaler", "scheduler_progress", "scheduled_updates",
            "completed_epoch", "next_epoch", "completed_updates",
            "best_validation_objective", "config", "run_metadata", "rng_state",
        }
        if not isinstance(state, Mapping) or not required <= state.keys():
            raise ValueError("full resume checkpoint is missing optimizer or training state")
        if state["scheduled_updates"] != self.scheduled_updates:
            raise ValueError("scheduled update budget differs from resume checkpoint")
        if state["next_epoch"] != state["completed_epoch"] + 1:
            raise ValueError("resume checkpoint epoch counters are inconsistent")
        if not math.isclose(state["scheduler_progress"], state["completed_updates"] / self.scheduled_updates):
            raise ValueError("resume checkpoint schedule progress is inconsistent")
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
        self.best_validation_objective = float(state["best_validation_objective"])
        self.run_metadata = dict(state["run_metadata"])
        _restore_rng(state["rng_state"])
