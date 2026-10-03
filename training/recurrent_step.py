"""One complete optimizer update for an ordered recurrent sequence."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from training.recurrent_loss import RecurrentLossResult, compute_recurrent_losses
from training.recurrent_sequence import RecurrentSequenceResult, run_recurrent_sequence


@dataclass
class RecurrentTrainStepResult:
    """Return differentiable sequence outputs and the measured pre-clip norm.

    Attributes:
        sequence: Ordered predictions, diagnostics, and recurrent memory states.
        losses: Segment-local losses and their arithmetic-mean objective.
        gradient_norm: The global norm before clipping when the configured
            clipper exposes it, otherwise ``None``.

    Invariants:
        ``sequence`` and ``losses`` describe the same consumed segment order.
        The returned tensors are not detached, copied, or logged by this type.
    """

    sequence: RecurrentSequenceResult
    losses: RecurrentLossResult
    gradient_norm: float | Tensor | None


def run_recurrent_train_step(
    *,
    model: Any,
    segments: Iterable[Mapping[str, Any]],
    loss_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Tensor]],
    optimizer: Any,
    scaler: Any,
    gradient_clipper: Callable[[Any], Any],
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    scheduler_progress: float,
    autocast_device_type: str = "cuda",
    num_segments: int,
    segment_frames: int,
    diagnostic: Any | None = None,
) -> RecurrentTrainStepResult:
    """Run one full-BPTT recurrent sequence and exactly one optimizer update.

    Args:
        model: Segment model consumed by the recurrent sequence boundary.
        segments: Ordered prepared mappings; each mapping is requested only when
            its predecessor has completed model execution.
        loss_fn: Segment-local loss callable accepting predictions and targets.
        optimizer: Wrapper or optimizer owner exposing an underlying optimizer,
            zeroing, and progress-scheduler update operations.
        scaler: AMP scaler used for one scale, unscale, step, and update cycle.
        gradient_clipper: Configured combined-set global gradient clipper.
        autocast_enabled: Whether to activate the requested autocast policy.
        autocast_dtype: Reduced precision used for autocast-eligible operations.
        scheduler_progress: Normalized sequence-update progress in ``[0, 1]``.
        autocast_device_type: Device type passed to the autocast context.
        num_segments: Exact number of ordered prepared mappings to consume.
        segment_frames: Required image-frame count in every consumed mapping.
        diagnostic: Optional observer for geometry, unscaled gradients, clipping,
            scheduled rates, and the actual underlying optimizer step. It must
            retain detached evidence only and release temporary hooks on error.

    Returns:
        The sequence result, segment-local loss result, and pre-clipping global
        norm when available from the configured clipper.

    Raises:
        ValueError: If normalized scheduler progress or the sequence objective
            is invalid or nonfinite.
        TypeError: If collaborators do not provide the required callable
            optimizer, scaler, clipping, sequence, or loss capabilities.

    Side effects:
        Zeros gradients once, runs one scaled backward call, unscales once,
        clips the combined trainable set once, updates schedules once, performs
        one underlying optimizer step, and updates the scaler once.

    Invariants:
        The incoming iterable is not materialized before sequence execution.
        There is no segment-level backward, optimizer, scheduler, clipping, or
        scaler operation. Memory states remain connected across all internal
        recurrent boundaries until the single backward call finishes. The
        function neither prepares, transfers, casts, normalizes, mutates, logs,
        checkpoints, or carries caller segments across independent invocations.
    """
    if not 0.0 <= scheduler_progress <= 1.0:
        raise ValueError("scheduler progress must be in [0, 1]")
    optimizer.zero_grad(set_to_none=True)
    recorded = []
    def stream():
        for segment in segments:
            recorded.append(segment)
            yield segment
    with torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype, enabled=autocast_enabled):
        sequence = run_recurrent_sequence(model, stream(), num_segments=num_segments, segment_frames=segment_frames)
        losses = compute_recurrent_losses(sequence, recorded, loss_fn, num_segments=num_segments)
    if not torch.isfinite(losses.objective).all():
        raise ValueError("sequence objective must be finite")
    if diagnostic is not None:
        diagnostic.before_update(model, sequence, losses, recorded, optimizer)
    scaler.scale(losses.objective).backward()
    if diagnostic is not None:
        diagnostic.after_backward()
    underlying = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    scaler.unscale_(underlying)
    if diagnostic is not None:
        diagnostic.after_unscale(model)
    norm = gradient_clipper(model)
    if isinstance(norm, Mapping):
        if len(norm) != 1:
            raise ValueError("gradient clipper must report one combined norm")
        norm = next(iter(norm.values()))
    if diagnostic is not None:
        diagnostic.after_clip(norm)
    if hasattr(optimizer, "step_schedulers"):
        optimizer.step_schedulers(scheduler_progress)
    if diagnostic is not None:
        diagnostic.after_schedule(optimizer, scheduler_progress)
        # A GradScaler may silently skip its underlying step on overflow.
        # The observer counts actual optimizer calls and checks value changes.
        with diagnostic.observe_step(underlying):
            scaler.step(underlying)
    else:
        scaler.step(underlying)
    scaler.update()
    if diagnostic is not None:
        diagnostic.after_update(model, optimizer, scaler)
    return RecurrentTrainStepResult(sequence, losses, norm)
