"""CPU contracts for the real-data smoke harness and detached audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from training.smoke_diagnostics import (
    UpdateDiagnostic,
    tensor_digest,
    trace_segment_memory,
    validate_groups,
)
from training.smoke_recurrent_training import (
    _same,
    _scheduled_rates,
    configured_phase,
    main,
    preflight,
    verify_restoration,
)
from training.recurrent_trainer import RecurrentTrainer
from training.recurrent_step import run_recurrent_train_step
from test_e01a_recurrent_step import _Model, _conf, _specs
from training.train_utils.optimizer import construct_optimizer_for_component_groups
from test_e01a_recurrent_trainer import (
    RecordingLogger,
    TinyModel,
    config,
    loss_fn,
    source,
)


CONFIG_NAME = "e01a_frozen_aggregator_streaming"


class SmallWriter(nn.Linear):
    """Expose a learned initial bank alongside tiny trainable writer weights."""

    def __init__(self) -> None:
        """Create a two-slot memory bank used by detached audit fixtures."""
        super().__init__(1, 1)
        self.initial_memory_bank = nn.Parameter(torch.zeros(1, 2, 3))


class SmallModel(nn.Module):
    """Expose the five production capability groups with tiny parameters."""

    def __init__(self) -> None:
        """Make one frozen encoder and five trainable scalar modules."""
        super().__init__()
        self.aggregator = nn.Linear(1, 1)
        self.memory_writer = SmallWriter()
        self.camera_read_adaptor = nn.Linear(1, 1)
        self.depth_read_adaptor = nn.Linear(1, 1)
        self.camera_head = nn.Linear(1, 1)
        self.depth_head = nn.Linear(1, 1)
        for parameter in self.aggregator.parameters():
            parameter.requires_grad_(False)
        self.aggregator.eval()


class GroupWrapper:
    """Expose an underlying optimizer through the trainer's wrapper protocol."""

    def __init__(self, model: SmallModel) -> None:
        """Group each trainable member under the expected capability name."""
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "name": "recurrent_memory",
                    "params": list(model.memory_writer.parameters())
                    + list(model.camera_read_adaptor.parameters())
                    + list(model.depth_read_adaptor.parameters()),
                    "lr": 1e-4,
                },
                {
                    "name": "prediction_heads",
                    "params": list(model.camera_head.parameters())
                    + list(model.depth_head.parameters()),
                    "lr": 5e-5,
                },
            ],
            weight_decay=0.05,
        )


def test_config_is_bounded_and_base_file_unchanged(tmp_path: Path) -> None:
    """Both phases resolve fixed geometry, one episode, offline mode, and 20 steps."""
    base = Path("training/config/e01a_frozen_aggregator_streaming.yaml").read_bytes()
    first = configured_phase(
        tmp_path,
        tmp_path / "weights.pt",
        tmp_path / "run",
        7,
        "cuda:0",
        "a",
        config_name=CONFIG_NAME,
    )
    second = configured_phase(
        tmp_path,
        tmp_path / "weights.pt",
        tmp_path / "run",
        7,
        "cuda:0",
        "b",
        config_name=CONFIG_NAME,
    )
    assert (first.max_epochs, second.max_epochs) == (1, 2)
    assert first.training.scheduled_updates == second.training.scheduled_updates == 20
    assert (
        first.training.max_episodes_per_epoch
        == second.training.max_episodes_per_epoch
        == 1
    )
    assert (
        first.validation.max_episodes_per_epoch
        == second.validation.max_episodes_per_epoch
        == 1
    )
    assert first.logging.mode == second.logging.mode == "offline"
    assert first.img_size == 518 and list(first.dataset_split.validation) == ["Scene20"]
    assert (
        first.sequence.total_frames,
        first.sequence.segment_frames,
        first.sequence.num_segments,
    ) == (15, 5, 3)
    assert first.model.segment_frames == 5
    assert first.episode_sources.train.total_frames == 15
    assert (
        first.checkpoint.pretrained_checkpoint_path
        and not second.checkpoint.pretrained_checkpoint_path
    )
    assert second.checkpoint.resume_checkpoint_path.endswith("epoch_0000.pt")
    assert first.optim.amp.amp_dtype == second.optim.amp.amp_dtype == "float16"
    assert first.optim.amp.init_scale == second.optim.amp.init_scale == 1.0
    assert (
        Path("training/config/e01a_frozen_aggregator_streaming.yaml").read_bytes()
        == base
    )
    for progress in (1 / 20, 2 / 20):
        rates = _scheduled_rates(first, progress)
        factor = 0.5 * (1 + math.cos(math.pi * (progress - 0.05) / 0.95))
        assert rates["recurrent_memory"] == pytest.approx(1e-4 * factor)
        assert rates["prediction_heads"] == pytest.approx(5e-5 * factor)
        assert all(rate > 0 for rate in rates.values())


def test_changed_production_capabilities_cannot_claim_real_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tiny substitute model is rejected even when geometry remains valid."""
    from training import smoke_recurrent_training as smoke

    original = smoke.load_config(CONFIG_NAME)
    modified = OmegaConf.create(OmegaConf.to_container(original, resolve=True))
    modified.model._target_ = "tests.tiny.MockModel"
    monkeypatch.setattr(smoke, "load_config", lambda _name: modified)
    with pytest.raises(ValueError, match="production model"):
        configured_phase(
            tmp_path,
            tmp_path / "weights.pt",
            tmp_path / "run",
            7,
            "cuda:0",
            "a",
            config_name=CONFIG_NAME,
        )


def test_missing_prerequisites_fail_and_summary_remains_incomplete(
    tmp_path: Path,
) -> None:
    """Missing weights produce nonzero CLI status and machine-readable failure."""
    data = tmp_path / "dataset"
    data.mkdir()
    for scene in ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20"):
        (data / scene).mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        preflight(data, tmp_path / "absent.pt", "cuda:0", config_name=CONFIG_NAME)
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                CONFIG_NAME,
                "--dataset-root",
                str(data),
                "--pretrained-checkpoint",
                str(tmp_path / "absent.pt"),
                "--output-dir",
                str(output),
                "--seed",
                "1",
                "--device",
                "cuda:0",
            ]
        )
        == 1
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete" and "checkpoint" in summary["error"]
    assert "FileNotFoundError" in Path(summary["traceback_log"]).read_text()
    assert "a" not in summary and "b" not in summary


def test_phase_b_requires_completed_phase_a(tmp_path: Path) -> None:
    """A partial report cannot launch the resume phase or claim acceptance."""
    output = tmp_path / "run"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"status": "incomplete", "seed": 7})
    )
    assert (
        main(
            [
                "--config",
                CONFIG_NAME,
                "--dataset-root",
                str(tmp_path / "absent"),
                "--pretrained-checkpoint",
                str(tmp_path / "absent.pt"),
                "--output-dir",
                str(output),
                "--seed",
                "7",
                "--device",
                "cuda:0",
                "--phase",
                "b",
            ]
        )
        == 1
    )
    summary = json.loads((output / "summary.json").read_text())
    assert (
        summary["status"] == "incomplete" and "a" not in summary and "b" not in summary
    )


def audit_inputs(
    model: SmallModel,
) -> tuple[SimpleNamespace, SimpleNamespace, list[dict]]:
    """Build a tiny three-segment sequence with connected memory and finite losses."""
    initial = model.memory_writer.initial_memory_bank.expand(1, -1, -1).clone()
    memories = [initial, initial + 1, initial + 2, initial + 3]
    sequence = SimpleNamespace(
        memory_states=memories, predictions=[{} for _ in range(3)]
    )
    losses = SimpleNamespace(
        segment_losses=[
            {
                "objective": torch.tensor(2.0),
                "loss_camera": torch.tensor(1.0),
                "loss_depth": torch.tensor(1.0),
            }
            for _ in range(3)
        ],
        objective=torch.tensor(2.0),
    )
    segments = [
        {
            "ids": torch.arange(index * 3, index * 3 + 3).reshape(1, 3),
            "images": torch.zeros(1, 3, 3, 4, 4),
            "segment_index": index,
            "seq_name": ["scene/variation/camera/0"],
        }
        for index in range(3)
    ]
    return sequence, losses, segments


def test_group_membership_and_gradient_audit() -> None:
    """Reject wrong group membership, nonfinite and disconnected gradients."""
    model = SmallModel()
    wrapper = GroupWrapper(model)
    assert set(validate_groups(model, wrapper)) == {
        "recurrent_memory",
        "prediction_heads",
    }
    audit = UpdateDiagnostic(image_shape=(1, 3, 3, 4, 4), num_segments=3)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    audit.after_unscale(model)
    assert audit._current["gradients"]["memory_writer"]["nonzero"] == 3
    model.camera_head.weight.grad.zero_()
    audit.after_unscale(model)  # Individual zero gradients are legitimate.
    model.camera_head.bias.grad = None
    with pytest.raises(ValueError, match="disconnected"):
        audit.after_unscale(model)
    model.camera_head.bias.grad = torch.full_like(model.camera_head.bias, float("nan"))
    audit.after_unscale(model)
    assert audit._current["nonfinite_gradients"]["camera_head.bias"]["nan_elements"] == 1
    wrapper.optimizer.param_groups[0]["params"].append(model.camera_head.weight)
    with pytest.raises(ValueError, match="duplicate|misplaced"):
        validate_groups(model, wrapper)


def test_loss_geometry_rate_and_detached_summary_audits() -> None:
    """Reject invalid sequence evidence and serialize one actual audited update."""
    model = SmallModel()
    wrapper = GroupWrapper(model)
    sequence, losses, segments = audit_inputs(model)
    audit = UpdateDiagnostic(
        image_shape=(1, 3, 3, 4, 4),
        num_segments=3,
        memory_shape=(1, 2, 3),
        expected_rates={"recurrent_memory": 1e-4, "prediction_heads": 5e-5},
        expected_decay=0.05,
    )
    losses.segment_losses[0]["objective"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        audit.before_update(model, sequence, losses, segments, wrapper)
    losses.segment_losses[0]["objective"] = torch.tensor(2.0)
    segments[2]["ids"][0, 0] += 1
    with pytest.raises(ValueError, match="consecutive"):
        audit.before_update(model, sequence, losses, segments, wrapper)
    segments[2]["ids"][0, 0] -= 1
    audit.before_update(model, sequence, losses, segments, wrapper)
    audit.after_backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    audit.after_unscale(model)
    audit.after_clip(torch.tensor(4.0))
    audit.after_schedule(wrapper, 1 / 20)
    with audit.observe_step(wrapper.optimizer):
        wrapper.optimizer.step()
    audit.after_update(model, wrapper, torch.amp.GradScaler("cpu", enabled=False))
    record = audit.records[0]
    assert record["frame_ids"] == list(range(9))
    assert (
        record["actual_optimizer_steps"]
        == record["backward_calls"]
        == record["clipping_calls"]
        == record["scheduler_advancements"]
        == 1
    )
    assert record["scaler_state_empty"] is True
    assert not audit._before and not wrapper.optimizer._optimizer_step_post_hooks
    json.dumps(record)


def test_wrong_scheduled_rate_is_rejected() -> None:
    """An unchanged optimizer group rate cannot masquerade as schedule progress."""
    model = SmallModel()
    wrapper = GroupWrapper(model)
    audit = UpdateDiagnostic(
        image_shape=(1, 3, 3, 4, 4),
        num_segments=3,
        expected_rates={"recurrent_memory": 2e-4, "prediction_heads": 5e-5},
    )
    with pytest.raises(ValueError, match="scheduled learning rates"):
        audit.after_schedule(wrapper, 1 / 20)


def test_skipped_step_and_unchanged_values_fail() -> None:
    """A skipped attempt is recorded; a purported real step still needs movement."""
    model = SmallModel()
    wrapper = GroupWrapper(model)
    audit = UpdateDiagnostic(image_shape=(1, 3, 3, 4, 4), num_segments=3)
    audit._before = {name: tensor_digest(p) for name, p in model.named_parameters()}
    audit.after_update(model, wrapper, torch.amp.GradScaler("cpu", enabled=False))
    assert audit.records[0]["actual_optimizer_steps"] == 0
    assert audit.records[0]["optimizer_outcome"] == "skipped"
    audit._before = {name: tensor_digest(p) for name, p in model.named_parameters()}
    with audit.observe_step(wrapper.optimizer):
        wrapper.optimizer.step()
    with pytest.raises(ValueError, match="did not change"):
        audit.after_update(model, wrapper, torch.amp.GradScaler("cpu", enabled=False))
    assert not wrapper.optimizer._optimizer_step_post_hooks


def test_diagnostic_records_nonfinite_gradient_and_real_scaler_skip() -> None:
    """The full diagnostic path lets GradScaler skip AdamW and records the outcome."""
    model = _Model()
    model.aggregator.eval()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    audit = UpdateDiagnostic(
        image_shape=(1, 3, 3, 2, 2),
        num_segments=3,
        memory_shape=(1, 1, 1),
        precision="float16",
    )
    segments = [
        {
            "images": torch.full((1, 3, 3, 2, 2), float(index + 1)),
            "ids": torch.arange(index * 3, index * 3 + 3).reshape(1, 3),
            "segment_index": index,
            "frame_start": index * 3,
            "frame_stop": (index + 1) * 3,
            "seq_name": ["scene/variation/camera/0"],
        }
        for index in range(3)
    ]
    def loss_fn(prediction, _segment):
        camera = prediction["camera"].square().mean()
        depth = prediction["depth"].square().mean()
        return {"objective": camera + depth, "loss_camera": camera, "loss_depth": depth}

    before = {name: tensor_digest(parameter) for name, parameter in model.named_parameters()}
    rates = [group["lr"] for group in wrapper.optimizer.param_groups]
    hook = model.depth_head.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        result = run_recurrent_train_step(
            model=model,
            segments=segments,
            loss_fn=loss_fn,
            optimizer=wrapper,
            scaler=scaler,
            gradient_clipper=lambda module: torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in module.parameters() if parameter.requires_grad], 1.0
            ),
            autocast_enabled=False,
            autocast_dtype=torch.float16,
            autocast_device_type="cpu",
            scheduler_progress=1 / 20,
            num_segments=3,
            segment_frames=3,
            diagnostic=audit,
        )
    finally:
        hook.remove()
    assert result.optimizer_ran is False
    assert scaler.get_scale() == 4.0
    assert not wrapper.optimizer.state
    assert [group["lr"] for group in wrapper.optimizer.param_groups] == rates
    assert all(
        tensor_digest(parameter) == before[name]
        for name, parameter in model.named_parameters()
    )
    record = audit.records[0]
    assert record["nonfinite_gradients"]["depth_head.weight"]["positive_inf_elements"] == 1
    assert record["gradients"]["depth_head"]["nonfinite"] == ["depth_head.weight"]
    assert record["preclip_global_norm"] is None
    assert record["preclip_global_norm_nonfinite"] == "inf"
    assert record["scaler_scale"] == 8.0
    assert record["scaler_scale_after"] == 4.0
    assert record["optimizer_outcome"] == "skipped"
    assert record["optimizer_ran"] is False
    assert record["actual_optimizer_steps"] == 0
    assert record["backward_calls"] == record["clipping_calls"] == record["scheduler_advancements"] == 1
    assert not any(record["changed_parameters"].values())
    assert not wrapper.optimizer._optimizer_step_post_hooks
    json.dumps(record)


@pytest.mark.parametrize(
    ("scaler_enabled", "overflow_at"),
    [(False, "gradient"), (False, "clip_norm"), (True, "clip_norm")],
)
def test_nonfinite_diagnostic_rejects_before_adamw(
    scaler_enabled: bool, overflow_at: str
) -> None:
    """A rejected clip norm or disabled-scaler gradient leaves AdamW unchanged."""
    torch.manual_seed(17)
    model = _Model()
    model.aggregator.eval()
    wrapper = construct_optimizer_for_component_groups(model, _conf(), _specs(model))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    wrapper.optimizer.step()
    wrapper.zero_grad(set_to_none=True)
    before = {name: tensor_digest(parameter) for name, parameter in model.named_parameters()}
    state_before = {
        parameter: {key: value.detach().clone() for key, value in state.items()}
        for parameter, state in wrapper.optimizer.state.items()
    }
    rates = [group["lr"] for group in wrapper.optimizer.param_groups]
    scaler = torch.amp.GradScaler("cpu", enabled=scaler_enabled, init_scale=8.0)
    audit = UpdateDiagnostic(
        image_shape=(1, 3, 3, 2, 2),
        num_segments=3,
        memory_shape=(1, 1, 1),
    )
    segments = [
        {
            "images": torch.full((1, 3, 3, 2, 2), float(index + 1)),
            "ids": torch.arange(index * 3, index * 3 + 3).reshape(1, 3),
            "segment_index": index,
            "frame_start": index * 3,
            "frame_stop": (index + 1) * 3,
            "seq_name": ["scene/variation/camera/0"],
        }
        for index in range(3)
    ]

    def loss_fn(prediction, _segment):
        camera = prediction["camera"].square().mean()
        depth = prediction["depth"].square().mean()
        return {"objective": camera + depth, "loss_camera": camera, "loss_depth": depth}

    clip_calls = []

    def clipper(_model):
        clip_calls.append(True)
        return torch.tensor(float("inf"))

    hook = (
        model.depth_head.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )
        if overflow_at == "gradient"
        else None
    )
    try:
        message = "finite unscaled gradients" if scaler_enabled else "disabled GradScaler"
        with pytest.raises(ValueError, match=message):
            run_recurrent_train_step(
                model=model,
                segments=segments,
                loss_fn=loss_fn,
                optimizer=wrapper,
                scaler=scaler,
                gradient_clipper=clipper,
                autocast_enabled=False,
                autocast_dtype=torch.float16,
                autocast_device_type="cpu",
                scheduler_progress=1 / 20,
                num_segments=3,
                segment_frames=3,
                diagnostic=audit,
            )
    finally:
        if hook is not None:
            hook.remove()
    assert clip_calls == ([] if overflow_at == "gradient" else [True])
    assert audit._current["scaler_enabled"] is scaler_enabled
    assert scaler.get_scale() == (8.0 if scaler_enabled else 1.0)
    assert "scheduled_rates" not in audit._current
    if overflow_at == "gradient":
        assert audit._current["nonfinite_gradients"]["depth_head.weight"]["positive_inf_elements"] == 1
    else:
        assert audit._current["preclip_global_norm_nonfinite"] == "inf"
        if scaler_enabled:
            assert audit._current["nonfinite_gradients"] == {}
    assert audit.records == []
    assert [group["lr"] for group in wrapper.optimizer.param_groups] == rates
    assert all(
        tensor_digest(parameter) == before[name]
        for name, parameter in model.named_parameters()
    )
    assert wrapper.optimizer.state.keys() == state_before.keys()
    for parameter, state in wrapper.optimizer.state.items():
        assert state.keys() == state_before[parameter].keys()
        assert all(torch.equal(value, state_before[parameter][key]) for key, value in state.items())
    assert not wrapper.optimizer._optimizer_step_post_hooks


def test_resume_comparison_rejects_missing_or_changed_state() -> None:
    """Recursive restore checks compare exact tensors and reject missing keys."""
    _same(
        {"scaler": {}, "model": torch.tensor([1.0])},
        {"scaler": {}, "model": torch.tensor([1.0])},
    )
    with pytest.raises(ValueError, match="keys differ"):
        _same({"model": 1}, {})
    with pytest.raises(ValueError, match="mismatch"):
        _same(torch.tensor([1.0]), torch.tensor([2.0]))


def test_fresh_trainer_restoration_audit_precedes_second_update(tmp_path: Path) -> None:
    """Audit saved model, AdamW, empty scaler, RNG and progress before resuming."""
    first_config = config(tmp_path / "first", epochs=1)
    first_config["training"]["scheduled_updates"] = 20
    first = RecurrentTrainer(
        first_config,
        train_episodes=source(1),
        validation_episodes=source(1),
        model=TinyModel(),
        loss_fn=loss_fn,
        logger=RecordingLogger(),
    )
    first.run()
    checkpoint = tmp_path / "first" / "epoch_0000.pt"

    resumed_config = config(tmp_path / "resumed", epochs=2)
    resumed_config["training"]["scheduled_updates"] = 20
    resumed_config["checkpoint"]["resume_checkpoint_path"] = str(checkpoint)
    resumed = RecurrentTrainer(
        resumed_config,
        train_episodes=source(1),
        validation_episodes=source(1),
        model=TinyModel(),
        loss_fn=loss_fn,
        logger=RecordingLogger(),
    )
    result = verify_restoration(resumed, checkpoint)
    assert result["before_update"] is True and result["scaler_state_empty"] is True
    assert set(result["matched"]) >= {
        "model",
        "optimizer",
        "rng_state",
        "scheduler_progress",
    }
    assert resumed.completed_updates == 1 and resumed.next_epoch == 1
    resumed.run()
    assert resumed.completed_updates == 2 and resumed.next_epoch == 2


def test_alternate_frame_schedule_derives_episode_and_model_geometry() -> None:
    """Changing one configured dimension propagates to data and model inputs."""
    from training.smoke_recurrent_training import load_config

    cfg = load_config(CONFIG_NAME, overrides=["e01a.segment_frames=8"])
    assert (
        cfg.sequence.total_frames,
        cfg.sequence.segment_frames,
        cfg.sequence.num_segments,
    ) == (24, 8, 3)
    assert (
        cfg.model.segment_frames
        == cfg.episode_sources.train.total_frames / cfg.sequence.num_segments
        == 8
    )


def test_segment_memory_trace_survives_depth_failure_and_removes_hooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The last flushed row identifies the failing segment before a head raises."""

    class FailingHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("depth failed")
            return value

    class TraceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.depth_head = FailingHead()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.depth_head(value)

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (100, 200))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 30)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 40)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 50)
    model = TraceModel()
    path = tmp_path / "segments.jsonl"
    with pytest.raises(RuntimeError, match="depth failed"):
        with trace_segment_memory(model, torch.device("cuda:0"), path):
            model(torch.ones(1))
            model(torch.ones(1))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["event"], row["segment_index"]) for row in rows] == [
        ("segment_entry", 0),
        ("before_depth_head", 0),
        ("segment_entry", 1),
        ("before_depth_head", 1),
    ]
    assert rows[-1]["allocated_bytes"] == 30
    assert rows[-1]["peak_allocated_bytes"] == 50
    assert rows[-1]["device_free_bytes"] == 100
    assert not model._forward_pre_hooks and not model.depth_head._forward_pre_hooks


def test_diagnostic_uses_configured_segment_count() -> None:
    """The audit accepts a two-segment graph with three memory states."""
    model = SmallModel()
    wrapper = GroupWrapper(model)
    sequence, losses, segments = audit_inputs(model)
    sequence.memory_states = sequence.memory_states[:3]
    sequence.predictions = sequence.predictions[:2]
    losses.segment_losses = losses.segment_losses[:2]
    audit = UpdateDiagnostic(
        image_shape=(1, 3, 3, 4, 4), num_segments=2, memory_shape=(1, 2, 3)
    )
    audit.before_update(model, sequence, losses, segments[:2], wrapper)
    assert audit._current["segment_indices"] == [0, 1]
    assert audit._current["frame_ids"] == list(range(6))


def test_full_resolution_prediction_audit_uses_production_depth_layout() -> None:
    """Accept channel-last depth and scalar confidence, reject channel-first depth."""
    model = SmallModel()
    model.memory_writer.patch_grid_size = 37
    wrapper = GroupWrapper(model)
    sequence, losses, segments = audit_inputs(model)
    for segment in segments:
        segment["images"] = torch.empty((1, 3, 3, 518, 518), device="meta")
    for prediction in sequence.predictions:
        prediction.update(
            pose_enc=torch.empty((1, 3, 9), device="meta"),
            depth=torch.empty((1, 3, 518, 518, 1), device="meta"),
            depth_conf=torch.empty((1, 3, 518, 518), device="meta"),
        )
    audit = UpdateDiagnostic(
        image_shape=(1, 3, 3, 518, 518),
        num_segments=3,
        memory_shape=(1, 2, 3),
    )
    audit.before_update(model, sequence, losses, segments, wrapper)
    assert audit._current["prediction_shapes"][0]["depth"] == [1, 3, 518, 518, 1]
    sequence.predictions[0]["depth"] = torch.empty((1, 3, 1, 518, 518), device="meta")
    with pytest.raises(ValueError, match="depth prediction geometry"):
        audit.before_update(model, sequence, losses, segments, wrapper)


def test_tensor_digest_handles_scalar_and_non_scalar_parameters() -> None:
    """Raw-byte hashing preserves value, dtype, and original shape identity."""
    scalar = torch.tensor(1.25, dtype=torch.float32)
    assert tensor_digest(scalar) == tensor_digest(scalar.clone())
    assert tensor_digest(scalar) != tensor_digest(torch.tensor(1.5))
    assert tensor_digest(scalar) != tensor_digest(scalar.reshape(1))
    matrix = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert tensor_digest(matrix) == tensor_digest(matrix.clone())
    assert tensor_digest(matrix) != tensor_digest(matrix + 1)


def test_float16_smoke_initial_scale_reaches_the_trainer(tmp_path: Path) -> None:
    """A fresh scaler starts at the configured scale and remains enabled."""
    cfg = config(tmp_path)
    cfg["optim"]["amp"] = {"enabled": True, "amp_dtype": "float16", "init_scale": 1.0}
    trainer = RecurrentTrainer(
        cfg,
        train_episodes=source(1),
        validation_episodes=source(1),
        model=TinyModel(),
        loss_fn=loss_fn,
        logger=RecordingLogger(),
    )
    assert trainer.scaler.is_enabled()
    assert trainer.scaler.get_scale() == 1.0


def test_bfloat16_profile_keeps_the_default_initial_scale(tmp_path: Path) -> None:
    """The smoke override changes only the float16 precision path."""
    cfg = configured_phase(
        tmp_path,
        tmp_path / "weights.pt",
        tmp_path / "run",
        7,
        "cuda:0",
        "a",
        config_name=CONFIG_NAME,
        precision="bfloat16",
    )
    assert cfg.optim.amp.amp_dtype == "bfloat16"
    assert cfg.optim.amp.init_scale == 65536.0

def test_depth_gradient_detail_separates_current_signal_from_adamw_movement() -> None:
    """Unscaled tensor norms and zero counts identify momentum-driven changes."""
    class DepthModule(nn.Module):
        """Expose projection, block, and gate names used by the real adaptor."""

        def __init__(self):
            """Construct small parameters with the real grouping layout."""
            super().__init__()
            self.input_projection = nn.Linear(1, 1)
            self.blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
            self.output_projection = nn.Linear(1, 1)
            self.residual_gate = nn.Parameter(torch.tensor(-2.1972246))

    model = SmallModel()
    model.depth_read_adaptor = DepthModule()
    wrapper = GroupWrapper(model)
    audit = UpdateDiagnostic(image_shape=(1, 3, 3, 4, 4), num_segments=3, precision="float16")
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    audit.after_unscale(model, SimpleNamespace(get_scale=lambda: 128.0))
    detail = audit._current["depth_gradient_tensors"]
    groups = audit._current["depth_gradient_groups"]
    assert set(groups) == {"input_projection", "output_projection", "blocks.0", "blocks.1", "blocks.2", "residual_gate"}
    assert detail["depth_read_adaptor.input_projection.weight"]["norm"] == 1.0
    assert detail["depth_read_adaptor.input_projection.weight"]["zero_elements"] == 0
    assert audit._current["scaler_scale"] == 128.0
    wrapper.optimizer.step()

    before = {name: tensor_digest(parameter) for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = (
                torch.zeros_like(parameter)
                if name.startswith("depth_read_adaptor.") and name != "depth_read_adaptor.residual_gate"
                else torch.ones_like(parameter)
            )
    audit.after_unscale(model, SimpleNamespace(get_scale=lambda: 64.0))
    assert groups["blocks.0"]["zero_tensors"] == 0
    assert audit._current["depth_gradient_groups"]["blocks.0"]["zero_tensors"] == 2
    with audit.observe_step(wrapper.optimizer):
        wrapper.optimizer.step()
    moved_without_gradient = [
        name for name, parameter in model.named_parameters()
        if name.startswith("depth_read_adaptor.") and tensor_digest(parameter) != before[name]
        and detail[name]["norm"] > 0
        and audit._current["depth_gradient_tensors"][name]["norm"] == 0
    ]
    assert moved_without_gradient
