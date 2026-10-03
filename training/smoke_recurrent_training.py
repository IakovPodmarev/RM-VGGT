"""Run one bounded, auditable recurrent training or resume smoke phase.

The public CLI requires a production config, real dataset, pretrained
checkpoint, seed, and CUDA device. Phase A requires a fresh output folder;
phase B reuses it. A 20-update scheduler horizon keeps the first
two cosine rates positive while exactly one episode is trained per phase.
Each invocation uses the production trainer and full sequence step. Invoke
phase B separately, in a fresh process, after inspecting phase A evidence;
its restored state is compared before the second update.
Failures write an incomplete JSON summary and exit nonzero. Diagnostic hashes
and CUDA synchronization add overhead, so timings are not throughput results.
"""

from __future__ import annotations

# ruff: noqa: E402  # Training-path compatibility is established before imports.

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
for path in (ROOT, TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from omegaconf import OmegaConf

from training.launch import load_config
from training.recurrent_trainer import RecurrentTrainer, _rng_state
from training.smoke_diagnostics import UpdateDiagnostic, trace_segment_memory
from training.inspect_sequential_episodes import inspect_sequential_episodes


def configured_phase(
    dataset_root: Path,
    checkpoint: Path,
    output: Path,
    seed: int,
    device: str,
    phase: str,
    *,
    config_name: str,
    precision: str = "float16",
):
    """Compose the production config and impose bounded smoke settings.

    Return a resolved DictConfig; reject an unsupported phase or changed
    production resolution, model/loss contract, or precision. Phase A preloads weights, phase B
    restores the epoch checkpoint. The 20-update horizon preserves positive
    warmup/cosine rates for both actual updates. The base config is not edited.
    """
    if phase not in {"a", "b"}:
        raise ValueError("phase must be a or b")
    if precision not in {"bfloat16", "float16"}:
        raise ValueError("precision must be bfloat16 or float16")
    cfg = load_config(config_name)
    if cfg.img_size != 518 or cfg.sequence.backprop_mode != "full":
        raise ValueError("production image size or backpropagation mode changed")
    if cfg.model.get("segment_frames") != cfg.sequence.segment_frames:
        raise ValueError("model frame setting differs from the sequence")
    if (
        cfg.model._target_,
        cfg.loss._target_,
        cfg.episode_sources.train._target_,
        cfg.episode_sources.validation._target_,
    ) != (
        "vggt.models.RMVGGT.RMVGGT",
        "loss.MultitaskLoss",
        "data.datasets.vkitti_sequence.SequentialVKittiEpisodeSource",
        "data.datasets.vkitti_sequence.SequentialVKittiEpisodeSource",
    ):
        raise ValueError(
            "real smoke requires the production model, loss, and episode sources"
        )
    if (
        cfg.model.enable_camera,
        cfg.model.enable_depth,
        cfg.model.enable_point,
        cfg.model.enable_track,
        cfg.loss.camera.weight,
        cfg.loss.camera.loss_type,
        cfg.loss.depth.weight,
        cfg.loss.depth.gradient_loss_fn,
        cfg.loss.depth.valid_range,
        cfg.loss.point,
        cfg.loss.track,
    ) != (
        True,
        True,
        False,
        False,
        5.0,
        "l1",
        1.0,
        "grad",
        0.98,
        None,
        None,
    ):
        raise ValueError("production heads or camera/depth loss settings changed")
    cfg.vkitti_root = str(dataset_root.resolve())
    cfg.seed_value = seed
    cfg.device = device
    cfg.max_epochs = 1 if phase == "a" else 2
    cfg.training.max_episodes_per_epoch = 1
    cfg.validation.max_episodes_per_epoch = 1
    cfg.training.scheduled_updates = 20
    cfg.logging.mode = "offline"
    cfg.logging.run_name = f"recurrent-smoke-{phase}"
    cfg.checkpoint.save_dir = str((output / "checkpoints").resolve())
    cfg.checkpoint.pretrained_checkpoint_path = (
        str(checkpoint.resolve()) if phase == "a" else None
    )
    cfg.checkpoint.resume_checkpoint_path = (
        str((output / "checkpoints" / "epoch_0000.pt").resolve())
        if phase == "b"
        else None
    )
    cfg.optim.amp.amp_dtype = precision
    if precision == "float16":
        cfg.optim.amp.init_scale = 1.0
    if (
        cfg.training.episode_batch_size,
        cfg.training.accum_steps,
        cfg.optim.optimizer.weight_decay,
        cfg.optim.learning_rates.recurrent_memory,
        cfg.optim.learning_rates.prediction_heads,
        cfg.optim.scheduler.warmup_fraction,
        cfg.optim.scheduler.type,
    ) != (1, 1, 0.05, 1e-4, 5e-5, 0.05, "cosine"):
        raise ValueError("production optimizer or schedule settings changed")
    if (
        len(cfg.optim.gradient_clip.configs) != 1
        or cfg.optim.gradient_clip.configs[0].max_norm != 1.0
    ):
        raise ValueError("real smoke requires one global gradient clip at norm 1.0")
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _file_identity(path: Path) -> dict[str, Any]:
    """Return the exact file path, byte count, and streaming SHA-256 identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def preflight(
    dataset_root: Path,
    checkpoint: Path,
    device: str,
    *,
    config_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Reject missing real prerequisites and inspect one episode from each split.

    Return dataset/checkpoint identities, GPU/software evidence, resolved AMP
    precision, and observed frame geometry. Native bfloat16 is preferred;
    float16 is explicit on CUDA hardware with suitable compute capability and
    uses the trainer's enabled GradScaler. No model or optimizer is allocated.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"VKITTI dataset root is missing: {dataset_root}")
    missing = [
        scene
        for scene in ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")
        if not (dataset_root / scene).is_dir()
    ]
    if missing:
        raise ValueError(f"VKITTI scenes are missing: {missing}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"pretrained VGGT checkpoint is missing: {checkpoint}")
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("real smoke requires one CUDA device")
    index = (
        selected.index if selected.index is not None else torch.cuda.current_device()
    )
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    with torch.cuda.device(index):
        native_bfloat16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if not native_bfloat16 and props.major < 6:
        raise ValueError(
            "device has neither native bfloat16 nor supported float16 precision"
        )
    precision = "bfloat16" if native_bfloat16 else "float16"
    cfg = configured_phase(
        dataset_root,
        checkpoint,
        Path("/tmp/unused-smoke-output"),
        seed,
        device,
        "a",
        config_name=config_name,
        precision=precision,
    )
    train = RecurrentTrainer._configured_source(cfg.episode_sources.train)
    validation = RecurrentTrainer._configured_source(cfg.episode_sources.validation)
    episodes = inspect_sequential_episodes(
        train,
        validation,
        total_frames=cfg.sequence.total_frames,
        segment_frames=cfg.sequence.segment_frames,
        num_segments=cfg.sequence.num_segments,
    )
    if episodes["train"]["scene"] == episodes["validation"]["scene"]:
        raise ValueError("training and validation scenes overlap")
    return {
        "dataset_root": str(dataset_root.resolve()),
        "checkpoint": _file_identity(checkpoint),
        "episodes": episodes,
        "seed": seed,
        "device": str(selected),
        "device_name": props.name,
        "device_total_bytes": total,
        "device_free_bytes": free,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": precision,
        "scaler_initial_scale": float(cfg.optim.amp.init_scale),
        "schedule": {
            "segment_frames": int(cfg.sequence.segment_frames),
            "num_segments": int(cfg.sequence.num_segments),
            "episode_frames": int(cfg.sequence.total_frames),
        },
        "native_bfloat16": native_bfloat16,
        "compute_capability": [props.major, props.minor],
    }


def _same(expected: Any, actual: Any, path: str = "state") -> None:
    """Recursively require exact tensor and scalar identity for restored state."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(expected.cpu(), actual.cpu()):
            raise ValueError(f"resume mismatch at {path}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"resume keys differ at {path}")
        for key in expected:
            _same(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise ValueError(f"resume sequence differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _same(left, right, f"{path}.{index}")
    elif hasattr(expected, "shape") and hasattr(actual, "shape"):
        import numpy as np

        if not np.array_equal(expected, actual):
            raise ValueError(f"resume array differs at {path}")
    elif expected != actual:
        raise ValueError(f"resume value differs at {path}")


def verify_restoration(trainer: RecurrentTrainer, path: Path) -> dict[str, Any]:
    """Compare checkpoint and fresh trainer state before update two.

    Model weights, AdamW moments/steps, groups, scaler (including disabled
    empty state), counters, schedule progress/definition, and RNG must agree.
    The scheduler is a function of normalized progress, not a stateful object.
    """
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key, actual in (
        ("model", trainer.model.state_dict()),
        ("optimizer", trainer.optimizer.optimizer.state_dict()),
        ("scaler", trainer.scaler.state_dict()),
        ("rng_state", _rng_state()),
    ):
        _same(saved[key], actual, key)
    if (trainer.next_epoch, trainer.completed_updates) != (1, 1) or (
        saved["next_epoch"],
        saved["completed_updates"],
    ) != (1, 1):
        raise ValueError("restored epoch/update counters are invalid")
    if (
        saved["scheduled_updates"] != trainer.scheduled_updates
        or saved["scheduler_progress"] != 1 / 20
    ):
        raise ValueError("restored scheduler budget/progress differs")
    _same(
        saved["config"]["optim"]["scheduler"],
        trainer.config["optim"]["scheduler"],
        "scheduler_definition",
    )
    if any("memory_states" in key or "transient_memory" in key for key in saved):
        raise ValueError("checkpoint contains transient recurrent state")
    return {
        "matched": [
            "model",
            "optimizer",
            "scaler",
            "rng_state",
            "counters",
            "scheduler_definition",
            "scheduler_progress",
        ],
        "scaler_state_empty": not bool(saved["scaler"]),
        "before_update": True,
        "checkpoint": str(path.resolve()),
    }


class EvidenceLogger:
    """Forward graph-free events to one offline W&B run and retain scalar logs."""

    def __init__(self, config: Any, output: Path) -> None:
        """Create one offline run inside the selected output directory."""
        import wandb

        self.run = wandb.init(
            project=config.logging.project,
            name=config.logging.run_name,
            mode="offline",
            dir=str(output),
            config=OmegaConf.to_container(config, resolve=True),
        )
        self.events: list[dict[str, Any]] = []

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        """Store and forward detached trainer events with their update index."""
        self.events.append({"step": step, "values": values})
        self.run.log(values, step=step)

    def finish(self) -> None:
        """Close the run after checkpoint completion."""
        self.run.finish()


def _scheduled_rates(cfg: Any, progress: float) -> dict[str, float]:
    """Evaluate the configured 5% warmup and cosine rule at update progress."""
    import math

    warmup = float(cfg.optim.scheduler.warmup_fraction)
    factor = (
        progress / warmup
        if progress <= warmup
        else 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
    )
    return {
        name: float(base) * factor for name, base in cfg.optim.learning_rates.items()
    }


def _write_json(path: Path, values: dict[str, Any]) -> None:
    """Atomically replace a detached JSON report in the selected run folder."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, default=str))
    temporary.replace(path)


def execute_phase(
    args: argparse.Namespace, preflight_result: dict[str, Any]
) -> dict[str, Any]:
    """Run one trainer lifecycle and return detached evidence for that phase.

    Phase B is a separate CLI process. The trainer restores weights, optimizer,
    scaler and RNG before any update; this function checks them against the
    saved checkpoint before calling ``run``. Failures close offline W&B.
    """
    output = Path(args.output_dir)
    cfg = configured_phase(
        Path(args.dataset_root),
        Path(args.pretrained_checkpoint),
        output,
        args.seed,
        args.device,
        args.phase,
        config_name=args.config,
        precision=preflight_result["precision"],
    )
    logger = EvidenceLogger(cfg, output)
    progress = (1 if args.phase == "a" else 2) / 20
    diagnostic = UpdateDiagnostic(
        image_shape=(1, cfg.sequence.segment_frames, 3, cfg.img_size, cfg.img_size),
        num_segments=cfg.sequence.num_segments,
        expected_rates=_scheduled_rates(cfg, progress),
        expected_decay=float(cfg.optim.optimizer.weight_decay),
    )
    started = time.perf_counter()
    stage = "trainer construction"
    try:
        trainer = RecurrentTrainer(cfg, logger=logger, diagnostic=diagnostic)
        stage = "restoration audit"
        restoration = (
            verify_restoration(trainer, Path(cfg.checkpoint.resume_checkpoint_path))
            if args.phase == "b"
            else None
        )
        stage = "training, validation, or checkpoint"
        with trace_segment_memory(
            trainer.model,
            trainer.device,
            output / f"phase_{args.phase}_segment_memory.jsonl",
        ):
            trainer.run()
    except Exception as error:
        logger.finish()
        raise RuntimeError(f"{stage}: {type(error).__name__}: {error}") from error
    expected = 1 if args.phase == "a" else 2
    if (
        trainer.completed_updates != expected
        or trainer.next_epoch != expected
        or len(diagnostic.records) != 1
    ):
        raise ValueError("phase did not complete exactly one audited update")
    path = output / "checkpoints" / f"epoch_{expected - 1:04d}.pt"
    if not path.is_file():
        raise ValueError("epoch checkpoint was not saved")
    report = {
        "phase": args.phase,
        "status": "complete",
        "segment_memory_log": str(
            (output / f"phase_{args.phase}_segment_memory.jsonl").resolve()
        ),
        "process_id": os.getpid(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "completed_updates": trainer.completed_updates,
        "next_epoch": trainer.next_epoch,
        "updates": diagnostic.records,
        "restoration": restoration,
        "base_learning_rates": dict(cfg.optim.learning_rates),
        "scaler_scale_after": float(trainer.scaler.get_scale()),
        "events": logger.events,
        "checkpoint": str(path.resolve()),
        "checkpoint_identity": _file_identity(path),
        "wandb_dir": str(logger.run.dir),
        "total_seconds": time.perf_counter() - started,
        "timing_note": "Includes detached hashing, synchronization, validation, and checkpointing; not throughput.",
    }
    _write_json(output / f"phase_{args.phase}.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run only the requested phase and persist explicit incomplete status.

    The default phase A requires a fresh output folder and stops after its
    checkpoint. Phase B requires a completed phase A summary in that folder and
    must be invoked in a new process. Zero exit means the requested phase passed;
    only phase B can mark the two-phase workflow complete. No phase is launched
    by a subprocess from here, so the operator controls the resume boundary.
    """
    parser = argparse.ArgumentParser(
        description="Audited real-data recurrent training and resume smoke"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Production Hydra configuration name without .yaml",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--phase", choices=("a", "b"), default="a")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if args.phase == "a":
        if output.exists():
            parser.error(f"output directory must be fresh: {output}")
        output.mkdir(parents=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True
    )
    summary = output / "summary.json"
    report: dict[str, Any] = (
        {
            "status": "incomplete",
            "seed": args.seed,
            "config_name": args.config,
            "git_revision": revision.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip()),
            "output_dir": str(output.resolve()),
        }
        if args.phase == "a"
        else {}
    )
    try:
        report["stage"] = "phase prerequisites"
        if args.phase == "b":
            if not summary.is_file():
                raise FileNotFoundError("phase B requires a completed phase A summary")
            report = json.loads(summary.read_text())
            if (
                report.get("status") not in {"phase_a_complete", "incomplete"}
                or report.get("a", {}).get("status") != "complete"
            ):
                raise ValueError("phase A has not completed successfully")
            if report["a"]["process_id"] == os.getpid():
                raise ValueError("phase B requires a fresh process")
            report.pop("error", None)
            if report["seed"] != args.seed:
                raise ValueError("phase B seed differs from phase A")
            if not Path(report["a"]["checkpoint"]).is_file():
                raise FileNotFoundError("phase A epoch checkpoint is missing")
            _same(
                report["a"]["checkpoint_identity"],
                _file_identity(Path(report["a"]["checkpoint"])),
                "phase_a_checkpoint",
            )
            if report["config_name"] != args.config:
                raise ValueError("phase B configuration differs from phase A")
        report["stage"] = "data, checkpoint, and device preflight"
        fresh_preflight = preflight(
            Path(args.dataset_root),
            Path(args.pretrained_checkpoint),
            args.device,
            config_name=args.config,
            seed=args.seed,
        )
        if args.phase == "b":
            previous = report["preflight"]
            for key in (
                "dataset_root",
                "checkpoint",
                "device",
                "precision",
                "episodes",
            ):
                _same(previous[key], fresh_preflight[key], f"phase_preflight.{key}")
        report["preflight"] = fresh_preflight
        report["stage"] = f"phase {args.phase} execution"
        report[args.phase] = execute_phase(args, fresh_preflight)
        report["status"] = "phase_a_complete" if args.phase == "a" else "complete"
        report.pop("stage", None)
    except Exception as error:
        trace_path = output / f"phase_{args.phase}_traceback.log"
        if output.is_dir():
            trace_path.write_text(traceback.format_exc())
            report["traceback_log"] = str(trace_path.resolve())
        memory_log = output / f"phase_{args.phase}_segment_memory.jsonl"
        if memory_log.is_file():
            report["segment_memory_log"] = str(memory_log.resolve())
            lines = memory_log.read_text().splitlines()
            if lines:
                report["last_segment_event"] = json.loads(lines[-1])
        report["status"] = "incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    if output.is_dir():
        _write_json(summary, report)
    return 0 if report["status"] in {"phase_a_complete", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
