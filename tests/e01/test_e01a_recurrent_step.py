"""Contracts for one full-BPTT recurrent optimizer update."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pytest
import torch
from torch import Tensor, nn

from training.recurrent_step import run_recurrent_train_step
from training.train_utils.optimizer import (
    ParameterGroupSpec,
    construct_optimizer_for_component_groups,
)


class _Model(nn.Module):
    """Small recurrent model exposing frozen and trainable capabilities."""

    def __init__(self, unexpected: bool = False) -> None:
        """Create reachable trainable paths and a frozen aggregator."""
        super().__init__()
        self.aggregator = nn.Linear(1, 1)
        self.memory_writer = nn.Linear(1, 1, bias=False)
        self.memory_writer.initial_memory_bank = nn.Parameter(torch.ones(1, 1, 1))
        self.camera_read_adaptor = nn.Linear(1, 1, bias=False)
        self.depth_read_adaptor = nn.Linear(1, 1, bias=False)
        self.camera_head = nn.Linear(1, 1, bias=False)
        self.depth_head = nn.Linear(1, 1, bias=False)
        if unexpected:
            self.unexpected = nn.Parameter(torch.ones(1))
        for parameter in self.aggregator.parameters():
            parameter.requires_grad_(False)
        self.calls = 0

    def initial_memory(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return fresh learned state for each independent sequence."""
        self.calls = 0
        return self.memory_writer.initial_memory_bank.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def forward(self, *, images: Tensor, read_memory: Tensor):
        """Predict from incoming memory and return an out-of-place transition."""
        signal = images.mean(dim=(1, 2, 3, 4), keepdim=True).reshape(-1, 1, 1)
        camera = self.camera_head(self.camera_read_adaptor(read_memory))
        depth = self.depth_head(self.depth_read_adaptor(read_memory))
        self.calls += 1
        return {"camera": camera, "depth": depth}, self.memory_writer(read_memory) + signal, {}


def _specs(model: _Model) -> list[ParameterGroupSpec]:
    """Construct the required two capability groups with explicit peak rates."""
    named = dict(model.named_parameters())
    recurrent = tuple(name for name in named if name.startswith(("memory_writer.", "camera_read_adaptor.", "depth_read_adaptor.")))
    heads = tuple(name for name in named if name.startswith(("camera_head.", "depth_head.")))
    return [
        ParameterGroupSpec("recurrent_memory", recurrent, tuple(named[name] for name in recurrent), {"lr": 1.0e-4}),
        ParameterGroupSpec("prediction_heads", heads, tuple(named[name] for name in heads), {"lr": 5.0e-5}),
    ]


def _conf() -> dict[str, object]:
    """Return the required optimizer-level defaults."""
    return {"_target_": "torch.optim.AdamW", "weight_decay": 0.05}


def test_e01a_optimizer_groups_are_exact_disjoint_complete_and_frozen_safe() -> None:
    """Groups preserve names, rates, defaults, complete coverage, and frozen exclusion."""
    model = _Model()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    groups = wrapper.optimizer.param_groups
    assert [group["name"] for group in groups] == ["recurrent_memory", "prediction_heads"]
    assert [group["lr"] for group in groups] == [1.0e-4, 5.0e-5]
    assert [group["weight_decay"] for group in groups] == [0.05, 0.05]
    selected = [parameter for group in groups for parameter in group["params"]]
    assert len(selected) == len(set(selected))
    assert set(selected) == {parameter for parameter in model.parameters() if parameter.requires_grad}
    assert all(parameter not in selected for parameter in model.aggregator.parameters())


@pytest.mark.parametrize("options", [{"name": "bad"}, {"params": []}, {"lr": 0.0}, {"lr": float("nan")}])
def test_e01a_optimizer_rejects_reserved_or_invalid_options(options: dict[str, object]) -> None:
    """Reserved PyTorch keys and invalid learning rates fail with useful errors."""
    model, specs = _Model(), _specs(_Model())
    specs = _specs(model)
    specs[0] = ParameterGroupSpec(specs[0].name, specs[0].parameter_names, specs[0].parameters, options)
    with pytest.raises(ValueError, match="recurrent_memory|name|params|lr"):
        construct_optimizer_for_component_groups(model, _conf(), specs)


def test_e01a_optimizer_rejects_uncovered_trainable_and_duplicate_names() -> None:
    """An unexpected parameter or repeated group label fails before construction."""
    with pytest.raises(ValueError, match="unexpected|uncovered"):
        model = _Model(unexpected=True)
        construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    model, specs = _Model(), _specs(_Model())
    specs = _specs(model)
    specs[1] = ParameterGroupSpec(specs[0].name, specs[1].parameter_names, specs[1].parameters, specs[1].optimizer_options)
    with pytest.raises(ValueError, match="duplicate|recurrent_memory"):
        construct_optimizer_for_component_groups(model, _conf(), specs)


def test_e01a_scheduler_has_shared_five_percent_warmup_then_cosine_shape() -> None:
    """Both explicit peaks follow 0, peak, midpoint-half, and end-zero values."""
    model, peaks = _Model(), (1.0e-4, 5.0e-5)

    def schedule(peak: float):
        return lambda progress: peak * (progress / .05 if progress <= .05 else .5 * (1 + math.cos(math.pi * (progress - .05) / .95)))

    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model), schedulers=[{"lr": schedule(peak)} for peak in peaks])
    for progress, multiplier in ((0., 0.), (.05, 1.), (.525, .5), (1., 0.)):
        wrapper.step_schedulers(progress)
        assert [group["lr"] for group in wrapper.optimizer.param_groups] == pytest.approx([peak * multiplier for peak in peaks])


class _Scaler:
    """Record one scaler lifecycle while delegating the real backward call."""

    def __init__(self) -> None:
        """Initialize ordered operation recording."""
        self.events: list[str] = []

    def is_enabled(self) -> bool:
        """Match the disabled scaler used by the unscaled test path."""
        return False

    def scale(self, objective: Tensor) -> "_Scaler":
        """Retain and record the sole objective selected for backward."""
        self.events.append("scale")
        self.objective = objective
        return self

    def backward(self) -> None:
        """Run the one backward call."""
        self.events.append("backward")
        self.objective.backward()

    def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
        """Record unscaling."""
        self.events.append("unscale")

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Record and execute one underlying step."""
        self.events.append("step")
        optimizer.step()

    def update(self) -> None:
        """Record one scaler update."""
        self.events.append("update")


class _Clipper:
    """Record one combined global clipping call."""

    def __init__(self) -> None:
        """Initialize clipping count."""
        self.calls = 0

    def __call__(self, model: nn.Module) -> float:
        """Clip every trainable parameter together and return its prior norm."""
        self.calls += 1
        return float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))


def _stream(model: _Model) -> Iterable[dict[str, Tensor]]:
    """Yield each prepared segment only after the predecessor model call."""
    for index in range(3):
        assert model.calls == index
        yield {"images": torch.full((1, 8, 3, 2, 2), float(index + 1))}
        assert model.calls == index + 1


def _loss(prediction: dict[str, Tensor], _segment: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return one differentiable local loss containing both head paths."""
    return {"objective": prediction["camera"].square().mean() + prediction["depth"].square().mean()}


def test_e01a_train_step_is_lazy_single_update_and_full_bptt() -> None:
    """One mean loss performs one ordered update and keeps both recurrent boundaries connected."""
    model, scaler, clipper = _Model(), _Scaler(), _Clipper()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    schedules: list[float] = []
    zeroes: list[bool | None] = []
    original_zero = wrapper.zero_grad
    wrapper.zero_grad = lambda *args, **kwargs: (zeroes.append(kwargs.get("set_to_none")), original_zero(*args, **kwargs))[1]
    original = wrapper.step_schedulers
    wrapper.step_schedulers = lambda progress: (schedules.append(progress), original(progress))[1]
    result = run_recurrent_train_step(model=model, segments=_stream(model), loss_fn=_loss, optimizer=wrapper, scaler=scaler, gradient_clipper=clipper, autocast_enabled=False, autocast_dtype=torch.bfloat16, autocast_device_type="cpu", scheduler_progress=.25, num_segments=3, segment_frames=8)
    assert result.losses.objective.detach().item() == pytest.approx(sum(item["objective"].item() for item in result.losses.segment_losses) / 3)
    assert scaler.events == ["scale", "backward", "unscale", "step", "update"]
    assert clipper.calls == 1 and schedules == [.25]
    assert zeroes == [True]
    assert result.sequence.memory_states[1].grad_fn is not None
    assert result.sequence.memory_states[2].grad_fn is not None
    assert all(parameter.grad is None for parameter in model.aggregator.parameters())
    for module in (model.memory_writer, model.camera_read_adaptor, model.depth_read_adaptor, model.camera_head, model.depth_head):
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_e01a_nonfinite_objective_prevents_all_mutation() -> None:
    """NaN objective prevents backward, clipping, scheduling, stepping, and updating."""
    model, scaler, clipper = _Model(), _Scaler(), _Clipper()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    wrapper.step_schedulers = lambda _progress: pytest.fail("scheduler must not run")
    with pytest.raises(ValueError, match="finite"):
        run_recurrent_train_step(model=model, segments=_stream(model), loss_fn=lambda _p, _s: {"objective": torch.tensor(float("nan"))}, optimizer=wrapper, scaler=scaler, gradient_clipper=clipper, autocast_enabled=False, autocast_dtype=torch.bfloat16, autocast_device_type="cpu", scheduler_progress=.25, num_segments=3, segment_frames=8)
    assert scaler.events == [] and clipper.calls == 0


def test_e01a_cpu_bfloat16_autocast_smoke_path() -> None:
    """Supported CPU autocast accepts float32 parameters without boundary casts."""
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("CPU bfloat16 autocast is unavailable")
    model, scaler, clipper = _Model(), _Scaler(), _Clipper()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    result = run_recurrent_train_step(model=model, segments=_stream(model), loss_fn=_loss, optimizer=wrapper, scaler=scaler, gradient_clipper=clipper, autocast_enabled=True, autocast_dtype=torch.bfloat16, autocast_device_type="cpu", scheduler_progress=.25, num_segments=3, segment_frames=8)
    assert torch.isfinite(result.losses.objective)



@pytest.mark.parametrize("mode", ["bfloat16", "float32"])
def test_e01a_unscaled_nonfinite_backward_stops_before_update(mode: str) -> None:
    """A finite loss with an infinite backward gradient cannot reach AdamW."""
    model, clipper = _Model(), _Clipper()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    underlying = wrapper.optimizer
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    underlying.step()
    underlying.zero_grad(set_to_none=True)
    weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    state = {parameter: {key: value.clone() for key, value in values.items()} for parameter, values in underlying.state.items()}
    rates = [group["lr"] for group in underlying.param_groups]
    calls = []
    handle = underlying.register_step_post_hook(lambda *_args: calls.append(True))
    wrapper.step_schedulers = lambda _progress: pytest.fail("scheduler must not run")
    hook = model.depth_head.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        with pytest.raises(ValueError, match="nonfinite gradient.*depth_head.weight"):
            run_recurrent_train_step(
                model=model, segments=_stream(model), loss_fn=_loss, optimizer=wrapper,
                scaler=torch.amp.GradScaler("cpu", enabled=False), gradient_clipper=clipper,
                autocast_enabled=mode == "bfloat16", autocast_dtype=getattr(torch, mode),
                autocast_device_type="cpu", scheduler_progress=.25, num_segments=3, segment_frames=8,
            )
    finally:
        hook.remove()
        handle.remove()
    assert not calls and clipper.calls == 0
    assert [group["lr"] for group in underlying.param_groups] == rates
    assert all(torch.equal(parameter, weights[name]) for name, parameter in model.named_parameters())
    assert underlying.state.keys() == state.keys()
    assert all(torch.equal(value, state[parameter][key]) for parameter, values in underlying.state.items() for key, value in values.items())


def test_e01a_scaled_nonfinite_backward_skips_adamw_and_schedule() -> None:
    """FP16 overflow backs off the scaler and preserves the update position."""
    model, clipper = _Model(), _Clipper()
    wrapper = construct_optimizer_for_component_groups(
        model, _conf(), _specs(model), schedulers=[{"lr": lambda progress: 1e-4 * progress}, {"lr": lambda progress: 5e-5 * progress}],
    )
    underlying = wrapper.optimizer
    weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    rates = [group["lr"] for group in underlying.param_groups]
    calls = []
    progress = []
    handle = underlying.register_step_post_hook(lambda *_args: calls.append(True))
    schedule = wrapper.step_schedulers
    wrapper.step_schedulers = lambda value: (progress.append(value), schedule(value))[1]
    hook = model.depth_head.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    try:
        result = run_recurrent_train_step(
            model=model, segments=_stream(model), loss_fn=_loss, optimizer=wrapper,
            scaler=scaler, gradient_clipper=clipper, autocast_enabled=True,
            autocast_dtype=torch.float16, autocast_device_type="cpu", scheduler_progress=.25,
            num_segments=3, segment_frames=8,
        )
    finally:
        hook.remove()
        handle.remove()
    assert torch.isfinite(result.losses.objective)
    assert result.optimizer_ran is False and not calls
    assert scaler.get_scale() == 4.0
    assert progress == [.25] and [group["lr"] for group in underlying.param_groups] == rates
    assert clipper.calls == 1 and not underlying.state
    assert all(torch.equal(parameter, weights[name]) for name, parameter in model.named_parameters())


def test_e01a_finite_backward_calls_adamw_once() -> None:
    """The finiteness guard preserves the ordinary finite-gradient update."""
    model, clipper = _Model(), _Clipper()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    calls = []
    handle = wrapper.optimizer.register_step_post_hook(lambda *_args: calls.append(True))
    try:
        result = run_recurrent_train_step(
            model=model, segments=_stream(model), loss_fn=_loss, optimizer=wrapper,
            scaler=torch.amp.GradScaler("cpu", enabled=False), gradient_clipper=clipper,
            autocast_enabled=False, autocast_dtype=torch.float32, autocast_device_type="cpu",
            scheduler_progress=.25, num_segments=3, segment_frames=8,
        )
    finally:
        handle.remove()
    assert result.optimizer_ran is True and calls == [True] and clipper.calls == 1
