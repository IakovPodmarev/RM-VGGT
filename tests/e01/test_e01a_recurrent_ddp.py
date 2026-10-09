"""Two-rank checks for the complete recurrent episode DDP boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from training.data.datasets.vkitti_sequence import VKittiEpisodeWindow
from training.episode_grouping import plan_episode_groups
from training.distributed_update import DistributedGradientCoordinator
from training.recurrent_episode import RecurrentEpisodeModule
from training.recurrent_step import run_recurrent_train_step
from training.smoke_diagnostics import UpdateDiagnostic
from test_e01a_recurrent_step import _Model as _DiagnosticModel, _conf, _specs
from training.train_utils.gradient_clip import GradientClipper
from training.train_utils.optimizer import (
    build_recurrent_parameter_group_specs,
    construct_optimizer_for_component_groups,
)


class _SegmentModel(nn.Module):
    """Small segment model with a frozen aggregator and reachable writer."""

    def __init__(self) -> None:
        """Create a learned initial state, writer, and prediction head."""
        super().__init__()
        self.aggregator = nn.Linear(1, 1)
        self.aggregator.requires_grad_(False)
        self.memory_writer = nn.Linear(1, 1, bias=False)
        self.memory_writer.initial_memory_bank = nn.Parameter(torch.ones(1, 1, 1))
        self.camera_head = nn.Linear(1, 1, bias=False)
        self.calls = 0

    def initial_memory(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Start each episode from the same learned state."""
        self.calls = 0
        return self.memory_writer.initial_memory_bank.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def forward(self, *, images: Tensor, read_memory: Tensor):
        """Predict from incoming memory and carry an out-of-place transition."""
        self.calls += 1
        signal = images.mean(dim=(1, 2, 3, 4), keepdim=True).reshape(-1, 1, 1)
        outgoing = self.memory_writer(read_memory) + signal
        if self.calls == 1:
            outgoing.retain_grad()
        return {"value": self.camera_head(read_memory)}, outgoing, {}


class _CountingEpisode(RecurrentEpisodeModule):
    """Count calls that cross the complete episode module boundary."""

    def __init__(self, model: nn.Module) -> None:
        """Configure a three-segment episode and zero the forward count."""
        super().__init__(model, num_segments=3, segment_frames=2)
        self.forward_calls = 0

    def forward(self, segments, loss_fn):
        """Record one DDP entry before delegating the complete episode."""
        self.forward_calls += 1
        return super().forward(segments, loss_fn)


def _segments(model: _SegmentModel, rank: int):
    """Yield each rank's distinct segment only after its predecessor ran."""
    for index in range(3):
        assert model.calls == index
        yield {
            "images": torch.full((1, 2, 3, 2, 2), float(rank + index + 1)),
            "segment_index": index,
            "frame_start": 2 * index,
            "frame_stop": 2 * (index + 1),
        }
        assert model.calls == index + 1


def _last_loss(prediction, segment):
    """Supervise the last segment while keeping earlier local losses scalar."""
    value = prediction["value"].square().mean()
    return {"objective": value if segment["segment_index"] == 2 else value * 0}


def _worker(rank: int, directory: str, partial: bool = False) -> None:
    """Compare a full or padded DDP group with independent episode gradients."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/group", rank=rank, world_size=2,
    )
    try:
        if partial:
            window = VKittiEpisodeWindow("Scene01", "clone", "Camera_0", 0, tuple(range(6)))
            assignment = next(plan_episode_groups((window,), seed=17, epoch=0, world_size=2))[rank]
        else:
            assignment = None
        episode_rank = 0 if partial else rank
        torch.manual_seed(17)
        local_model = _SegmentModel()
        objective, _, _, _ = RecurrentEpisodeModule(
            local_model, num_segments=3, segment_frames=2,
        )(_segments(local_model, episode_rank), _last_loss)
        objective.backward()
        local_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in local_model.named_parameters() if parameter.requires_grad
        }

        torch.manual_seed(17)
        model = _SegmentModel()
        episode = _CountingEpisode(model)
        ddp = DistributedDataParallel(episode)
        specs = build_recurrent_parameter_group_specs(
            model, {"recurrent_memory": 1e-4, "prediction_heads": 1e-4},
        )
        optimizer = construct_optimizer_for_component_groups(
            model, {"_target_": "torch.optim.AdamW"}, specs,
        )
        clipper = GradientClipper([
            {"module_name": ["memory_writer", "camera_head"], "max_norm": 1e6},
        ])
        clipper.setup_clipping(model)
        backward_calls: list[bool] = []
        handle = model.memory_writer.weight.register_hook(
            lambda gradient: (backward_calls.append(True), gradient)[1]
        )
        result = run_recurrent_train_step(
            model=model, episode_module=ddp, segments=_segments(model, episode_rank),
            loss_fn=_last_loss, optimizer=optimizer,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            gradient_clipper=clipper, autocast_enabled=False,
            autocast_dtype=torch.float32, autocast_device_type="cpu",
            scheduler_progress=0.5, num_segments=3, segment_frames=2,
            objective_scale=assignment.objective_scale if assignment else 1.0,
        )
        handle.remove()
        first_transition = result.sequence.memory_states[1]
        torch.save({
            "local_gradients": local_gradients,
            "is_real": assignment.is_real if assignment else True,
            "identity": assignment.identity if assignment else f"episode-{rank}",
            "ddp_gradients": {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            },
            "forward_calls": episode.forward_calls,
            "backward_calls": len(backward_calls),
            "segment_calls": model.calls,
            "optimizer_ran": result.optimizer_ran,
            "first_transition_grad": first_transition.grad,
            "aggregator_grads": [parameter.grad for parameter in model.aggregator.parameters()],
            "objective": result.losses.objective.detach(),
            "segment_objectives": [item["objective"].detach() for item in result.losses.segment_losses],
        }, Path(directory) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
def test_e01a_two_rank_episode_ddp_matches_independent_gradient_mean(tmp_path: Path) -> None:
    """DDP averages complete episodes with one forward and backward per rank."""
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    rows = [
        torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for row in rows:
        assert row["forward_calls"] == 1
        assert row["backward_calls"] == 1
        assert row["segment_calls"] == 3
        assert row["optimizer_ran"] is True
        assert all(grad is None for grad in row["aggregator_grads"])
        assert row["first_transition_grad"] is not None
        assert torch.count_nonzero(row["first_transition_grad"]) > 0
        assert row["objective"] == torch.stack(row["segment_objectives"]).mean()
    for name in rows[0]["local_gradients"]:
        expected = (rows[0]["local_gradients"][name] + rows[1]["local_gradients"][name]) / 2
        for row in rows:
            torch.testing.assert_close(row["ddp_gradients"][name], expected)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
def test_e01a_partial_group_matches_its_only_real_episode(tmp_path: Path) -> None:
    """Both ranks backpropagate; only the real episode enters counts and loss."""
    mp.spawn(_worker, args=(str(tmp_path), True), nprocs=2, join=True)
    rows = [
        torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    assert [row["is_real"] for row in rows] == [True, False]
    assert rows[0]["identity"] == rows[1]["identity"]
    assert sum(row["is_real"] for row in rows) == 1
    assert sum(row["objective"] for row in rows if row["is_real"]) == rows[0]["objective"]
    for row in rows:
        assert row["forward_calls"] == row["backward_calls"] == 1
        assert row["segment_calls"] == 3
        assert row["optimizer_ran"] is True
        assert all(grad is None for grad in row["aggregator_grads"])
        for name, gradient in row["ddp_gradients"].items():
            torch.testing.assert_close(gradient, rows[0]["local_gradients"][name])
    assert torch.count_nonzero(rows[0]["first_transition_grad"]) > 0
    assert torch.count_nonzero(rows[1]["first_transition_grad"]) == 0


def _native_overflow_state() -> dict[str, float | int]:
    """Use GradScaler's own overflow path as the full-state reference."""
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
    scaler.scale(torch.ones((), requires_grad=True))
    state = scaler.state_dict()
    state["_growth_tracker"] = 1
    scaler.load_state_dict(state)
    scaler.scale(parameter.square()).backward()
    parameter.grad.fill_(float("inf"))
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    return scaler.state_dict()


def _consensus_worker(rank: int, directory: str, mode: str, overflow: bool) -> None:
    """Run one coordinated DDP update with a rank-local post-unscale fault."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/group", rank=rank, world_size=2,
    )
    try:
        torch.manual_seed(31)
        model = _SegmentModel()
        episode = _CountingEpisode(model)
        ddp = DistributedDataParallel(episode)
        rates = {"recurrent_memory": 1e-4, "prediction_heads": 5e-5}
        optimizer = construct_optimizer_for_component_groups(
            model, {"_target_": "torch.optim.AdamW"},
            build_recurrent_parameter_group_specs(model, rates),
            schedulers=[
                {"lr": lambda progress: rates["recurrent_memory"] * progress},
                {"lr": lambda progress: rates["prediction_heads"] * progress},
            ],
        )
        optimizer.step_schedulers(0.25)
        prior_rates = [group["lr"] for group in optimizer.optimizer.param_groups]
        schedule_calls: list[float] = []
        original_schedule = optimizer.step_schedulers

        def record_schedule(progress: float) -> None:
            """Record scheduler use after the shared gradient decision."""
            schedule_calls.append(progress)
            original_schedule(progress)

        optimizer.step_schedulers = record_schedule
        scaler = torch.amp.GradScaler(
            "cpu", enabled=mode == "float16", init_scale=8.0, growth_interval=2,
        )
        if scaler.is_enabled():
            scaler.scale(torch.ones((), requires_grad=True))
            state = scaler.state_dict()
            state["_growth_tracker"] = 1
            scaler.load_state_dict(state)
        if overflow:
            original_unscale = scaler.unscale_

            def inject_after_unscale(underlying: torch.optim.Optimizer) -> None:
                """Corrupt only rank zero after native local inf checks."""
                original_unscale(underlying)
                if rank == 0:
                    model.camera_head.weight.grad.fill_(float("inf"))

            scaler.unscale_ = inject_after_unscale
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer_calls: list[bool] = []
        handle = optimizer.optimizer.register_step_post_hook(
            lambda *_args: optimizer_calls.append(True)
        )
        clip_calls: list[bool] = []

        def clip(underlying_model: nn.Module) -> float:
            """Count clipping after a shared finite decision."""
            clip_calls.append(True)
            return float(torch.nn.utils.clip_grad_norm_(underlying_model.parameters(), 1e6))

        error = None
        optimizer_ran = False
        try:
            result = run_recurrent_train_step(
                model=model, episode_module=ddp, segments=_segments(model, rank),
                loss_fn=_last_loss, optimizer=optimizer, scaler=scaler,
                gradient_clipper=clip, gradient_coordinator=DistributedGradientCoordinator(),
                autocast_enabled=mode != "float32",
                autocast_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[mode],
                autocast_device_type="cpu", scheduler_progress=0.5,
                num_segments=3, segment_frames=2,
            )
            optimizer_ran = result.optimizer_ran
        except ValueError as caught:
            error = str(caught)
        finally:
            handle.remove()
        torch.save({
            "before": before,
            "after": {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            },
            "scaler_state": scaler.state_dict(),
            "native_overflow_state": _native_overflow_state() if overflow and scaler.is_enabled() else None,
            "prior_rates": prior_rates,
            "rates": [group["lr"] for group in optimizer.optimizer.param_groups],
            "optimizer_calls": len(optimizer_calls),
            "clip_calls": len(clip_calls),
            "schedule_calls": schedule_calls,
            "optimizer_ran": optimizer_ran,
            "error": error,
            "forward_calls": episode.forward_calls,
            "segment_calls": model.calls,
        }, Path(directory) / f"consensus_{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
@pytest.mark.parametrize("mode", ["float32", "bfloat16", "float16"])
def test_e01a_rank_local_nonfinite_rejects_every_optimizer(
    tmp_path: Path, mode: str,
) -> None:
    """One rank's post-unscale fault rejects or skips both complete updates."""
    mp.spawn(_consensus_worker, args=(str(tmp_path), mode, True), nprocs=2, join=True)
    rows = [
        torch.load(tmp_path / f"consensus_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for row in rows:
        assert row["forward_calls"] == 1 and row["segment_calls"] == 3
        assert row["optimizer_calls"] == 0 and row["optimizer_ran"] is False
        assert row["clip_calls"] == 0 and row["schedule_calls"] == []
        assert row["rates"] == row["prior_rates"]
        assert all(torch.equal(row["after"][name], value) for name, value in row["before"].items())
        if mode == "float16":
            assert row["error"] is None
            assert row["scaler_state"] == row["native_overflow_state"]
            assert row["scaler_state"]["_growth_tracker"] == 0
        else:
            assert "nonfinite gradient on a distributed rank" in row["error"]
            assert row["scaler_state"] == {}


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
@pytest.mark.parametrize("mode", ["float32", "bfloat16", "float16"])
def test_e01a_finite_ranks_step_once_with_identical_state(tmp_path: Path, mode: str) -> None:
    """Both ranks make one AdamW call and keep parameters, rates, and scaler aligned."""
    mp.spawn(_consensus_worker, args=(str(tmp_path), mode, False), nprocs=2, join=True)
    rows = [
        torch.load(tmp_path / f"consensus_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for row in rows:
        assert row["error"] is None
        assert row["forward_calls"] == 1 and row["segment_calls"] == 3
        assert row["optimizer_calls"] == 1 and row["optimizer_ran"] is True
        assert row["clip_calls"] == 1 and row["schedule_calls"] == [0.5]
        assert row["rates"] != row["prior_rates"]
        assert any(not torch.equal(row["after"][name], value) for name, value in row["before"].items())
    assert rows[0]["rates"] == rows[1]["rates"]
    assert rows[0]["scaler_state"] == rows[1]["scaler_state"]
    for name, value in rows[0]["after"].items():
        torch.testing.assert_close(value, rows[1]["after"][name])


def _diagnostic_segments(model: _DiagnosticModel):
    """Yield three audit-ready segments only after their predecessors ran."""
    for index in range(3):
        assert model.calls == index
        yield {
            "images": torch.full((1, 8, 3, 2, 2), float(index + 1)),
            "ids": torch.arange(index * 8, (index + 1) * 8).reshape(1, 8),
            "segment_index": index,
            "frame_start": index * 8,
            "frame_stop": (index + 1) * 8,
            "seq_name": ["scene/variation/camera/0"],
        }
        assert model.calls == index + 1


def _diagnostic_loss(prediction, _segment):
    """Supply the camera and depth terms required by UpdateDiagnostic."""
    camera = prediction["camera"].square().mean()
    depth = prediction["depth"].square().mean()
    return {"objective": camera + depth, "loss_camera": camera, "loss_depth": depth}


def _diagnostic_worker(rank: int, directory: str, mode: str) -> None:
    """Inject one post-unscale fault while auditing both rank outcomes."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/group", rank=rank, world_size=2,
    )
    try:
        torch.manual_seed(41)
        model = _DiagnosticModel()
        model.aggregator.eval()
        episode = RecurrentEpisodeModule(model, num_segments=3, segment_frames=8)
        forward_calls: list[bool] = []
        forward_hook = episode.register_forward_hook(
            lambda *_args: forward_calls.append(True)
        )
        ddp = DistributedDataParallel(episode)
        optimizer = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
        prior_rates = [group["lr"] for group in optimizer.optimizer.param_groups]
        schedule_calls: list[float] = []
        original_schedule = optimizer.step_schedulers

        def record_schedule(progress: float) -> None:
            """Record every attempted scheduler advancement."""
            schedule_calls.append(progress)
            original_schedule(progress)

        optimizer.step_schedulers = record_schedule
        scaler = torch.amp.GradScaler(
            "cpu", enabled=mode == "float16", init_scale=8.0, growth_interval=2,
        )
        if scaler.is_enabled():
            scaler.scale(torch.ones((), requires_grad=True))
            state = scaler.state_dict()
            state["_growth_tracker"] = 1
            scaler.load_state_dict(state)
        original_unscale = scaler.unscale_

        def inject_after_unscale(underlying: torch.optim.Optimizer) -> None:
            """Corrupt only rank zero after native local inf checks."""
            original_unscale(underlying)
            if rank == 0:
                model.depth_head.weight.grad.fill_(float("inf"))

        scaler.unscale_ = inject_after_unscale
        audit = UpdateDiagnostic(
            image_shape=(1, 8, 3, 2, 2), num_segments=3,
            memory_shape=(1, 1, 1), precision=mode,
        )
        inspection_calls: list[bool] = []
        original_inspection = audit.after_unscale

        def record_inspection(underlying_model, active_scaler) -> None:
            """Count diagnostic gradient inspection after the collective."""
            inspection_calls.append(True)
            original_inspection(underlying_model, active_scaler)

        audit.after_unscale = record_inspection
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer_calls: list[bool] = []
        optimizer_hook = optimizer.optimizer.register_step_post_hook(
            lambda *_args: optimizer_calls.append(True)
        )
        clip_calls: list[bool] = []

        def clip(underlying_model: nn.Module) -> float:
            """Count clipping without changing the audit's skip contract."""
            clip_calls.append(True)
            return float(torch.nn.utils.clip_grad_norm_(underlying_model.parameters(), 1.0))

        error = None
        optimizer_ran = False
        try:
            result = run_recurrent_train_step(
                model=model, episode_module=ddp, segments=_diagnostic_segments(model),
                loss_fn=_diagnostic_loss, optimizer=optimizer, scaler=scaler,
                gradient_clipper=clip, gradient_coordinator=DistributedGradientCoordinator(),
                diagnostic=audit, autocast_enabled=False,
                autocast_dtype=torch.float16 if mode == "float16" else torch.float32,
                autocast_device_type="cpu", scheduler_progress=0.5,
                num_segments=3, segment_frames=8,
            )
            optimizer_ran = result.optimizer_ran
        except ValueError as caught:
            error = str(caught)
        finally:
            optimizer_hook.remove()
            forward_hook.remove()
        torch.save({
            "error": error,
            "optimizer_ran": optimizer_ran,
            "optimizer_calls": len(optimizer_calls),
            "forward_calls": len(forward_calls),
            "segment_calls": model.calls,
            "inspection_calls": len(inspection_calls),
            "clip_calls": len(clip_calls),
            "schedule_calls": schedule_calls,
            "audit_records": audit.records,
            "scaler_state": scaler.state_dict(),
            "native_overflow_state": _native_overflow_state() if scaler.is_enabled() else None,
            "rates": [group["lr"] for group in optimizer.optimizer.param_groups],
            "prior_rates": prior_rates,
            "unchanged": all(
                torch.equal(parameter, before[name])
                for name, parameter in model.named_parameters()
            ),
        }, Path(directory) / f"diagnostic_{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
@pytest.mark.parametrize("mode", ["float32", "float16"])
def test_e01a_diagnostic_waits_for_shared_nonfinite_decision(
    tmp_path: Path, mode: str,
) -> None:
    """Audit inspection cannot interrupt consensus or a native shared skip."""
    mp.spawn(_diagnostic_worker, args=(str(tmp_path), mode), nprocs=2, join=True)
    rows = [
        torch.load(tmp_path / f"diagnostic_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for row in rows:
        assert row["forward_calls"] == 1 and row["segment_calls"] == 3
        assert row["optimizer_calls"] == 0 and row["optimizer_ran"] is False
        assert row["clip_calls"] == 0 and row["schedule_calls"] == []
        assert row["rates"] == row["prior_rates"]
        assert row["unchanged"]
        if mode == "float32":
            assert "nonfinite gradient on a distributed rank" in row["error"]
            assert row["inspection_calls"] == 0
            assert row["audit_records"] == []
            assert row["scaler_state"] == {}
        else:
            assert row["error"] is None
            assert row["inspection_calls"] == 1
            assert row["scaler_state"] == row["native_overflow_state"]
            assert len(row["audit_records"]) == 1
            record = row["audit_records"][0]
            assert record["actual_optimizer_steps"] == 0
            assert record["backward_calls"] == 1
            assert record["clipping_calls"] == 0
            assert record["scheduler_advancements"] == 0
            assert record["optimizer_outcome"] == "skipped"
    if mode == "float16":
        assert "depth_head.weight" in rows[0]["audit_records"][0]["nonfinite_gradients"]
        assert rows[1]["audit_records"][0]["nonfinite_gradients"] == {}
