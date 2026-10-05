"""E01a contracts for equal aggregation of recurrent segment losses."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from launch import load_config
from loss import MultitaskLoss, compute_camera_loss
from training.recurrent_loss import compute_recurrent_losses as _compute_recurrent_losses
from training.recurrent_sequence import RecurrentSequenceResult



def compute_recurrent_losses(sequence, targets, loss_fn, *, num_segments=3):
    """Pass the legacy three-segment fixture count into the production loss."""
    return _compute_recurrent_losses(
        sequence, targets, loss_fn, num_segments=num_segments
    )


class _RecordingLoss:
    """Return supplied loss mappings while recording the exact call arguments."""

    def __init__(self, results: list[dict[str, Tensor]]) -> None:
        """Store result mappings and initialize an ordered invocation record."""
        self.results = results
        self.calls: list[tuple[object, object]] = []

    def __call__(self, prediction: object, target: object) -> dict[str, Tensor]:
        """Record one pair and return its pre-arranged mapping without modifying it."""
        self.calls.append((prediction, target))
        return self.results[len(self.calls) - 1]


def _sequence(predictions: list[dict[str, Tensor]]) -> RecurrentSequenceResult:
    """Create a minimal result with supplied ordered predictions and inert state data."""
    return RecurrentSequenceResult(
        predictions=predictions,
        diagnostics=[{} for _ in predictions],
        memory_states=[torch.zeros(1) for _ in range(len(predictions) + 1)],
    )


def _loss_inputs() -> tuple[
    RecurrentSequenceResult,
    list[dict[str, Tensor]],
    list[dict[str, Tensor]],
]:
    """Create three identity-distinct prediction, target, and differentiable loss mappings."""
    predictions = [{"prediction": torch.tensor(float(index))} for index in range(3)]
    targets = [{"target": torch.tensor(float(index + 3))} for index in range(3)]
    losses = [
        {"objective": torch.tensor(float(value), requires_grad=True), "component": torch.tensor(value)}
        for value in (3, 6, 12)
    ]
    return _sequence(predictions), targets, losses


def test_e01a_config_exposes_exact_camera_and_depth_loss_settings() -> None:
    """The experiment configuration selects the existing camera and depth loss contract."""
    cfg = load_config("e01a_frozen_aggregator_streaming")

    assert dict(cfg.loss) == {
        "_target_": "loss.MultitaskLoss",
        "camera": {"weight": 5.0, "loss_type": "l1"},
        "depth": {"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        "point": None,
        "track": None,
    }
    assert isinstance(MultitaskLoss(**cfg.loss), MultitaskLoss)


def test_e01a_config_disables_point_and_track_losses() -> None:
    """The recurrent configuration leaves unsupported point and track losses disabled."""
    cfg = load_config("e01a_frozen_aggregator_streaming")

    assert cfg.loss.point is None
    assert cfg.loss.track is None


def test_e01a_recurrent_loss_calls_each_segment_in_order_and_preserves_identity() -> None:
    """Every prediction/target pair is evaluated once in its supplied temporal order."""
    sequence, targets, loss_mappings = _loss_inputs()
    loss = _RecordingLoss(loss_mappings)

    result = compute_recurrent_losses(sequence, targets, loss)

    assert loss.calls == list(zip(sequence.predictions, targets))
    assert len(loss.calls) == 3
    assert all(prediction is sequence.predictions[index] for index, (prediction, _) in enumerate(loss.calls))
    assert all(target is targets[index] for index, (_, target) in enumerate(loss.calls))
    assert all(result.segment_losses[index] is loss_mappings[index] for index in range(3))


def test_e01a_recurrent_loss_uses_the_exact_arithmetic_mean_with_one_third_gradients() -> None:
    """Three scalar objectives produce their unweighted differentiable arithmetic mean."""
    sequence, targets, loss_mappings = _loss_inputs()

    result = compute_recurrent_losses(sequence, targets, _RecordingLoss(loss_mappings))
    result.objective.backward()

    torch.testing.assert_close(result.objective, torch.tensor(7.0))
    for loss_mapping in loss_mappings:
        torch.testing.assert_close(loss_mapping["objective"].grad, torch.tensor(1 / 3))


def test_e01a_recurrent_loss_keeps_objectives_as_connected_tensors() -> None:
    """Aggregation returns a tensor with autograd history rather than a Python value."""
    sequence, targets, loss_mappings = _loss_inputs()

    result = compute_recurrent_losses(sequence, targets, _RecordingLoss(loss_mappings))

    assert torch.is_tensor(result.objective)
    assert result.objective.requires_grad
    assert result.objective.grad_fn is not None


def test_e01a_recurrent_loss_preserves_cross_segment_gradient_history() -> None:
    """A mean from recurrent predictions reaches transitions before later segments."""
    initial_state = torch.tensor(2.0, requires_grad=True)
    transition_scales = torch.tensor([0.5, 0.75, 0.25], requires_grad=True)
    state = initial_state
    predictions = []
    for index in range(3):
        predictions.append({"prediction": state})
        state = state * transition_scales[index] + 1
    sequence = _sequence(predictions)
    targets = [{}, {}, {}]

    result = compute_recurrent_losses(
        sequence,
        targets,
        lambda prediction, _target: {"objective": prediction["prediction"].square()},
    )
    result.objective.backward()

    assert transition_scales.grad is not None
    assert transition_scales.grad[0].abs() > 0
    assert transition_scales.grad[1].abs() > 0


def test_e01a_recurrent_loss_keeps_segment_zero_and_introduces_no_auxiliary_losses() -> None:
    """The no-history loss mapping remains available without added result fields."""
    sequence, targets, loss_mappings = _loss_inputs()

    result = compute_recurrent_losses(sequence, targets, _RecordingLoss(loss_mappings))

    assert result.segment_losses[0] is loss_mappings[0]
    assert set(result.__dict__) == {"objective", "segment_losses"}
    assert [set(loss_mapping) for loss_mapping in result.segment_losses] == [
        {"objective", "component"}
    ] * 3


def test_e01a_recurrent_loss_does_not_mutate_inputs_or_returned_loss_mappings() -> None:
    """Predictions, targets, tensors, and returned dictionaries retain their original values."""
    sequence, targets, loss_mappings = _loss_inputs()
    prediction_values = [mapping["prediction"].clone() for mapping in sequence.predictions]
    target_values = [mapping["target"].clone() for mapping in targets]
    loss_keys = [set(mapping) for mapping in loss_mappings]
    loss_values = [mapping["component"].clone() for mapping in loss_mappings]

    result = compute_recurrent_losses(sequence, targets, _RecordingLoss(loss_mappings))

    for mapping, original in zip(sequence.predictions, prediction_values):
        torch.testing.assert_close(mapping["prediction"], original)
    for mapping, original in zip(targets, target_values):
        torch.testing.assert_close(mapping["target"], original)
    for mapping, keys, value in zip(loss_mappings, loss_keys, loss_values):
        assert set(mapping) == keys
        torch.testing.assert_close(mapping["component"], value)
    assert all(result.segment_losses[index] is loss_mappings[index] for index in range(3))


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([{}, {}], [{}, {}, {}], "predictions"),
        ([{}, {}, {}], [{}, {}], "segments"),
        ([{}, {}], [{}, {}], "predictions"),
        ([{}, object(), {}], [{}, {}, {}], "prediction"),
        ([{}, {}, {}], [{}, object(), {}], "segment"),
    ],
)
def test_e01a_recurrent_loss_rejects_invalid_counts_and_non_mapping_entries(
    predictions: list[object], targets: list[object], message: str
) -> None:
    """Count and mapping violations identify the recurrent loss input that is invalid."""
    sequence = _sequence(predictions)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError), match=message):
        compute_recurrent_losses(sequence, targets, _RecordingLoss([]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("loss_result", "error_type", "message"),
    [
        (object(), TypeError, "loss result"),
        ({}, ValueError, "objective"),
        ({"objective": 1.0}, TypeError, "objective"),
        ({"objective": torch.ones(1)}, ValueError, "scalar"),
        ({"objective": torch.tensor(float("nan"))}, ValueError, "finite"),
        ({"objective": torch.tensor(float("inf"))}, ValueError, "finite"),
    ],
)
def test_e01a_recurrent_loss_rejects_invalid_objectives(
    loss_result: object, error_type: type[Exception], message: str
) -> None:
    """Loss result mappings must expose finite scalar tensor objectives."""
    sequence, targets, _ = _loss_inputs()

    with pytest.raises(error_type, match=message):
        compute_recurrent_losses(sequence, targets, lambda _prediction, _target: loss_result)  # type: ignore[return-value]


def test_e01a_recurrent_loss_never_calls_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggregation leaves gradient execution to its caller."""
    sequence, targets, loss_mappings = _loss_inputs()
    calls = 0
    original_backward = Tensor.backward

    def record_backward(self: Tensor, *args: Any, **kwargs: Any) -> None:
        """Record unexpected tensor backward calls before delegating to PyTorch."""
        nonlocal calls
        calls += 1
        original_backward(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "backward", record_backward)
    compute_recurrent_losses(sequence, targets, _RecordingLoss(loss_mappings))

    assert calls == 0

def _camera_fixture() -> tuple[dict, dict]:
    """Return two valid camera targets and differentiable pose predictions."""
    extrinsics = torch.zeros(1, 2, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1).clone()
    target = {
        "images": torch.ones(1, 2, 3, 4, 4),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_masks": torch.zeros(1, 2, 4, 4, dtype=torch.bool),
    }
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    pose = extri_intri_to_pose_encoding(extrinsics, intrinsics, (4, 4)).detach()
    predicted = pose.clone()
    predicted[0, 0, 0] += 2
    predicted[0, 1, 0] += 4
    predicted.requires_grad_()
    return {"pose_enc_list": [predicted]}, target


def test_camera_loss_uses_valid_camera_frames_with_sparse_first_depth() -> None:
    """First-frame depth sparsity cannot disable otherwise valid camera labels."""
    prediction, target = _camera_fixture()
    target["point_masks"][:, 1] = True
    result = compute_camera_loss(prediction, target)
    torch.testing.assert_close(result["loss_T"], torch.tensor(1.0))
    result["loss_camera"].backward()
    assert prediction["pose_enc_list"][0].grad[0, 0, 0] != 0
    assert prediction["pose_enc_list"][0].grad[0, 1, 0] != 0


def test_camera_loss_explicit_mask_selects_frames_and_validates_targets() -> None:
    """Masked invalid targets are ignored; marked nonfinite or bad focal targets fail."""
    prediction, target = _camera_fixture()
    target["camera_valid_mask"] = torch.tensor([[True, False]])
    target["extrinsics"][0, 1] = float("nan")
    result = compute_camera_loss(prediction, target)
    torch.testing.assert_close(result["loss_T"], torch.tensor(2 / 3))
    result["loss_camera"].backward()
    assert prediction["pose_enc_list"][0].grad[0, 0, 0] != 0
    assert prediction["pose_enc_list"][0].grad[0, 1, 0] == 0
    target["camera_valid_mask"] = torch.tensor([[True, True]])
    with pytest.raises(ValueError, match="finite"):
        compute_camera_loss(prediction, target)
    target["extrinsics"][0, 1] = target["extrinsics"][0, 0]
    target["intrinsics"][0, 1, 0, 0] = 0
    with pytest.raises(ValueError, match="focal"):
        compute_camera_loss(prediction, target)


@pytest.mark.parametrize("mask", [
    torch.ones(2, dtype=torch.bool),
    torch.ones(1, 3, dtype=torch.bool),
    torch.ones(1, 2, dtype=torch.float32),
])
def test_camera_loss_rejects_misaligned_mask(mask: Tensor) -> None:
    """An explicit camera mask must match batch and frame axes exactly."""
    prediction, target = _camera_fixture()
    target["camera_valid_mask"] = mask
    with pytest.raises(ValueError, match="camera_valid_mask"):
        compute_camera_loss(prediction, target)


def test_camera_loss_rejects_zero_camera_eligibility_before_aggregation() -> None:
    """An empty camera segment raises instead of becoming an objective term."""
    prediction, target = _camera_fixture()
    target["camera_valid_mask"] = torch.zeros(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="no valid camera"):
        compute_camera_loss(prediction, target)

def test_zero_camera_labels_abort_before_sequence_aggregation() -> None:
    """A segment without camera labels cannot contribute a zero-valued mean term."""
    prediction, target = _camera_fixture()
    targets = [dict(target) for _ in range(3)]
    targets[1]["camera_valid_mask"] = torch.zeros(1, 2, dtype=torch.bool)
    calls = []

    def camera_only(predicted, segment):
        """Record segment evaluation and return the production camera objective."""
        calls.append(segment)
        return MultitaskLoss(camera={"weight": 5.0, "loss_type": "l1"})(predicted, segment)

    with pytest.raises(ValueError, match="no valid camera labels"):
        compute_recurrent_losses(
            _sequence([prediction] * 3), targets, camera_only,
        )
    assert calls == targets[:2]
