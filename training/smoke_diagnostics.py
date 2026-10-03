"""Detached update evidence for bounded recurrent training diagnostics.

The observer uses CPU hashes rather than duplicate GPU models. Its temporary
optimizer hook is removed on both success and failure. Measurements include
hashing and therefore are diagnostic wall time, not throughput benchmarks.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch

_COMPONENTS = (
    "memory_writer",
    "camera_read_adaptor",
    "depth_read_adaptor",
    "camera_head",
    "depth_head",
)


def tensor_digest(value: torch.Tensor) -> str:
    """Hash detached tensor storage on CPU without retaining an autograd graph."""
    data = value.detach().contiguous().cpu()
    digest = hashlib.sha256(str((tuple(data.shape), str(data.dtype))).encode())
    digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_groups(model: Any, optimizer: Any) -> dict[str, Any]:
    """Require two disjoint named groups that cover every trainable parameter once.

    Return detached group names, tensor and element counts, rates, and decay.
    Reject frozen, duplicate, missing, or misplaced members.
    """
    named = dict(model.named_parameters())
    reverse = {id(value): name for name, value in named.items()}
    groups = optimizer.optimizer.param_groups
    if [group.get("name") for group in groups] != [
        "recurrent_memory",
        "prediction_heads",
    ]:
        raise ValueError("optimizer requires two ordered capability groups")
    seen: set[int] = set()
    report: dict[str, Any] = {}
    for group in groups:
        names = []
        for parameter in group["params"]:
            identifier = id(parameter)
            if (
                identifier in seen
                or identifier not in reverse
                or not parameter.requires_grad
            ):
                raise ValueError("duplicate, unknown, or frozen optimizer member")
            seen.add(identifier)
            name = reverse[identifier]
            expected = (
                "prediction_heads"
                if name.startswith(("camera_head.", "depth_head."))
                else "recurrent_memory"
            )
            if group["name"] != expected or not name.startswith(
                tuple(part + "." for part in _COMPONENTS)
            ):
                raise ValueError(f"misplaced optimizer member: {name}")
            names.append(name)
        report[group["name"]] = {
            "names": names,
            "tensors": len(names),
            "parameters": sum(named[name].numel() for name in names),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
    if seen != {id(value) for value in named.values() if value.requires_grad}:
        raise ValueError("optimizer does not cover every trainable parameter")
    return report


class UpdateDiagnostic:
    """Audit one full episode around unscale, clipping, and actual optimizer step.

    ``records`` contains JSON-safe shapes, losses, gradients, updates, and rates.
    No graph tensor survives an update. The context manager removes its step
    hook even when scaler or optimizer raises. Empty scaler state is valid.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, ...],
        num_segments: int,
        memory_shape: tuple[int, ...] = (1, 16, 512),
        expected_rates: dict[str, float] | None = None,
        expected_decay: float | None = None,
    ) -> None:
        """Create an observer with expected segment image and memory shapes.

        The caller supplies configured segment geometry. Smaller explicit
        dimensions support CPU unit tests without claiming real acceptance.
        ``expected_rates`` supplies the exact progress-scheduled rates for one
        update; the caller derives them from the configured optimizer schedule.
        ``expected_decay`` checks the unchanged AdamW decay in both groups.
        """
        self.records: list[dict[str, Any]] = []
        self.image_shape = image_shape
        self.num_segments = num_segments
        self.memory_shape = memory_shape
        self.expected_rates = expected_rates
        self.expected_decay = expected_decay
        self._before: dict[str, str] = {}
        self._steps = 0
        self._current: dict[str, Any] = {}
        self._clip_calls = 0
        self._schedule_calls = 0
        self._backward_calls = 0

    def before_update(
        self,
        model: Any,
        sequence: Any,
        losses: Any,
        segments: list[Any],
        optimizer: Any,
    ) -> None:
        """Validate stream/loss geometry and snapshot parameter hashes before backward."""
        group_report = validate_groups(model, optimizer)
        if self.expected_decay is not None and any(
            not math.isclose(group["weight_decay"], self.expected_decay)
            for group in group_report.values()
        ):
            raise ValueError("optimizer weight decay differs from configured value")
        if model.aggregator.training or any(
            p.requires_grad or p.grad is not None for p in model.aggregator.parameters()
        ):
            raise ValueError("aggregator must be frozen and in evaluation mode")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("model parameters must retain float32 storage")
        if len(sequence.memory_states) != self.num_segments + 1 or any(
            tuple(m.shape) != self.memory_shape for m in sequence.memory_states
        ):
            raise ValueError("recurrent memory geometry is invalid")
        initial_bank = model.memory_writer.initial_memory_bank
        expected_initial = (
            initial_bank.detach()
            .to(dtype=sequence.memory_states[0].dtype)
            .expand_as(sequence.memory_states[0])
        )
        if not torch.equal(sequence.memory_states[0].detach(), expected_initial):
            raise ValueError("episode did not reset to learned initial memory")
        if any(m.grad_fn is None for m in sequence.memory_states[1:-1]):
            raise ValueError("intermediate memory lost its autograd history")
        ids = torch.cat([segment["ids"].detach().cpu() for segment in segments], dim=1)
        total_frames = self.image_shape[1] * self.num_segments
        if tuple(ids.shape) != (1, total_frames) or not torch.equal(
            ids[0, 1:] - ids[0, :-1], torch.ones(total_frames - 1, dtype=ids.dtype)
        ):
            raise ValueError("episode frame IDs are not consecutive")
        if [segment["segment_index"] for segment in segments] != list(
            range(self.num_segments)
        ):
            raise ValueError("segment order is invalid")
        if any(
            tuple(segment["images"].shape) != self.image_shape for segment in segments
        ):
            raise ValueError("segment image geometry is invalid")
        identities = [tuple(segment.get("seq_name", ())) for segment in segments]
        if not identities[0] or any(
            identity != identities[0] for identity in identities
        ):
            raise ValueError("episode identity changed between segments")
        if self.image_shape[-2:] == (518, 518):
            if model.memory_writer.patch_grid_size**2 != 1369:
                raise ValueError("patch-token geometry is invalid")
            for prediction in sequence.predictions:
                for key in ("pose_enc", "depth", "depth_conf"):
                    if key not in prediction or not torch.is_tensor(prediction[key]):
                        raise ValueError(f"prediction is missing {key}")
                if tuple(prediction["pose_enc"].shape) != (*self.image_shape[:2], 9):
                    raise ValueError("camera prediction geometry is invalid")
                if tuple(prediction["depth"].shape) != (
                    *self.image_shape[:2],
                    518,
                    518,
                    1,
                ) or tuple(prediction["depth_conf"].shape) != (
                    *self.image_shape[:2],
                    518,
                    518,
                ):
                    raise ValueError("depth prediction geometry is invalid")
        terms = []
        if len(losses.segment_losses) != self.num_segments:
            raise ValueError(
                f"sequence must have exactly {self.num_segments} segment losses"
            )
        for loss in losses.segment_losses:
            row = {
                name: float(value.detach().float().cpu())
                for name, value in loss.items()
            }
            if not row or any(not math.isfinite(value) for value in row.values()):
                raise ValueError("segment loss has a nonfinite term")
            if (
                "objective" not in row
                or "loss_camera" not in row
                or not any(name.startswith("loss_") and "depth" in name for name in row)
            ):
                raise ValueError("segment loss is missing camera or depth supervision")
            terms.append(row)
        objective = float(losses.objective.detach().float().cpu())
        if not math.isfinite(objective) or not math.isclose(
            objective,
            sum(row["objective"] for row in terms) / self.num_segments,
            rel_tol=1e-4,
        ):
            raise ValueError("sequence loss is not a finite arithmetic mean")
        self._before = {name: tensor_digest(p) for name, p in model.named_parameters()}
        self._clip_calls = 0
        self._schedule_calls = 0
        self._backward_calls = 0
        self._current = {
            "frame_ids": ids[0].tolist(),
            "identity": segments[0].get("seq_name"),
            "segment_indices": list(range(self.num_segments)),
            "image_shapes": [list(s["images"].shape) for s in segments],
            "memory_shapes": [list(m.shape) for m in sequence.memory_states],
            "patch_tokens_per_frame": model.memory_writer.patch_grid_size**2
            if hasattr(model.memory_writer, "patch_grid_size")
            else None,
            "prediction_shapes": [
                {
                    key: list(value.shape)
                    for key, value in pred.items()
                    if torch.is_tensor(value)
                }
                for pred in sequence.predictions
            ],
            "losses": terms,
            "objective": objective,
            "groups_before": group_report,
        }

    def after_backward(self) -> None:
        """Count the single full-sequence backward operation before unscaling."""
        self._backward_calls += 1

    def after_unscale(self, model: Any) -> None:
        """Count finite, nonzero, zero, and missing gradients before global clipping."""
        report: dict[str, Any] = {}
        for component in _COMPONENTS:
            named = [
                (name, p)
                for name, p in model.named_parameters()
                if name.startswith(component + ".")
            ]
            present, nonzero, missing, zero, nonzero_names = 0, 0, [], [], []
            norm_sq = 0.0
            for name, parameter in named:
                gradient = parameter.grad
                if gradient is None:
                    missing.append(name)
                    continue
                if not torch.isfinite(gradient).all():
                    raise ValueError(f"nonfinite gradient: {name}")
                present += 1
                squared = float(gradient.detach().float().square().sum().cpu())
                if not math.isfinite(squared):
                    raise ValueError(f"nonfinite gradient norm: {name}")
                norm_sq += squared
                if squared > 0:
                    nonzero += 1
                    nonzero_names.append(name)
                else:
                    zero.append(name)
            if not named or not nonzero or missing:
                raise ValueError(
                    f"disconnected component: {component}; missing={missing}"
                )
            report[component] = {
                "present": present,
                "nonzero": nonzero,
                "nonzero_names": nonzero_names,
                "missing": missing,
                "zero": zero,
                "norm": math.sqrt(norm_sq),
            }
        if any(p.grad is not None for p in model.aggregator.parameters()):
            raise ValueError("frozen aggregator has gradients")
        for suffix in ("initial_memory_bank", "keep_gate"):
            selected = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith("memory_writer.") and suffix in name
            ]
            if selected and any(parameter.grad is None for _, parameter in selected):
                raise ValueError(f"memory reachability is missing {suffix}")
        self._current["gradients"] = report

    def after_clip(self, norm: Any) -> None:
        """Record exactly one finite pre-clipping global norm for this update."""
        self._clip_calls += 1
        value = (
            float(norm.detach().float().cpu()) if torch.is_tensor(norm) else float(norm)
        )
        if not math.isfinite(value):
            raise ValueError("global pre-clipping gradient norm is nonfinite")
        self._current["preclip_global_norm"] = value

    def after_schedule(self, optimizer: Any, progress: float) -> None:
        """Check scheduled group rates after one progress advancement."""
        self._schedule_calls += 1
        rates = {
            group["name"]: float(group["lr"])
            for group in optimizer.optimizer.param_groups
        }
        if self.expected_rates is not None and any(
            not math.isclose(rates[name], expected, rel_tol=1e-7, abs_tol=0.0)
            for name, expected in self.expected_rates.items()
        ):
            raise ValueError("scheduled learning rates differ from expected values")
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
            raise ValueError("scheduled learning rates must be finite and positive")
        self._current["schedule_progress"] = progress
        self._current["scheduled_rates"] = rates

    @contextmanager
    def observe_step(self, optimizer: Any) -> Iterator[None]:
        """Count underlying optimizer steps and remove the temporary hook on exit."""
        self._steps = 0
        handle = optimizer.register_step_post_hook(
            lambda *_: setattr(self, "_steps", self._steps + 1)
        )
        try:
            yield
        finally:
            handle.remove()

    def after_update(self, model: Any, optimizer: Any, scaler: Any) -> None:
        """Require one real step and supervised changes in every component."""
        if self._steps != 1:
            raise ValueError(
                f"expected one underlying optimizer step, got {self._steps}"
            )
        if self._current and (
            self._backward_calls,
            self._clip_calls,
            self._schedule_calls,
        ) != (1, 1, 1):
            raise ValueError(
                "expected one backward, clipping, and scheduler advancement"
            )
        changed = {component: [] for component in _COMPONENTS}
        for name, parameter in model.named_parameters():
            altered = tensor_digest(parameter) != self._before[name]
            if name.startswith("aggregator.") and altered:
                raise ValueError("frozen aggregator parameter changed")
            for component in _COMPONENTS:
                if name.startswith(component + ".") and altered:
                    changed[component].append(name)
        if any(not names for names in changed.values()):
            raise ValueError(f"component parameters did not change: {changed}")
        if "gradients" in self._current:
            for component, names in changed.items():
                if not set(names).intersection(
                    self._current["gradients"][component]["nonzero_names"]
                ):
                    raise ValueError(
                        f"component changed only without a nonzero supervised gradient: {component}"
                    )
        self._current.update(
            {
                "changed_parameters": changed,
                "backward_calls": self._backward_calls,
                "clipping_calls": self._clip_calls,
                "scheduler_advancements": self._schedule_calls,
                "actual_optimizer_steps": self._steps,
                "groups_after": validate_groups(model, optimizer),
                "scaler_state_empty": not bool(scaler.state_dict()),
            }
        )
        self.records.append(self._current)
        self._current = {}
        self._before = {}


@contextmanager
def trace_segment_memory(
    model: Any, device: torch.device, path: Path
) -> Iterator[None]:
    """Flush segment-entry and pre-depth-head CUDA memory evidence to JSONL.

    The two module pre-hooks hold no tensors or autograd graph. They are removed
    on success or failure, and each line is flushed before the next operation so
    an out-of-memory exception leaves its last active segment on disk.
    """
    counters = {"train": 0, "validation": 0}
    current: dict[str, Any] = {}

    def record(event: str) -> None:
        phase = "train" if model.training else "validation"
        if event == "segment_entry":
            current["phase"] = phase
            current["segment_index"] = counters[phase]
            counters[phase] += 1
        row = {
            "event": event,
            "phase": current.get("phase", phase),
            "segment_index": current.get("segment_index"),
        }
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            row.update(
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                device_free_bytes=free,
                device_total_bytes=total,
            )
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        entry_handle = model.register_forward_pre_hook(
            lambda _module, _args: record("segment_entry")
        )
        depth_handle = model.depth_head.register_forward_pre_hook(
            lambda _module, _args: record("before_depth_head")
        )
        try:
            yield
        finally:
            entry_handle.remove()
            depth_handle.remove()
