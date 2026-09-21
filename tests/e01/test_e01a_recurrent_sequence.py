"""E01a contracts for public segment forwarding and ordered recurrence."""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn

from training.recurrent_sequence import run_recurrent_sequence
from vggt.models.RMVGGT import RMVGGT


class _SequenceModel(nn.Module):
    """Small callable segment model that records recurrent data flow."""

    def __init__(self) -> None:
        """Initialize trainable transitions and a frozen stand-in aggregator."""
        super().__init__()
        self.writer_scales = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.prediction_scale = nn.Parameter(torch.tensor(1.0))
        self.aggregator_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.initial_memory_bank = nn.Parameter(torch.ones(1, 2, 1))
        self.initial_calls = 0
        self.module_calls = 0
        self.records: list[tuple[Tensor, Tensor, Tensor]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Record standard module invocation before dispatching to ``forward``."""
        self.module_calls += 1
        return super().__call__(*args, **kwargs)

    def initial_memory(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Expose a trainable nonzero state and reset the transition recorder."""
        self.initial_calls += 1
        self.records = []
        return self.initial_memory_bank.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def forward(self, *, images: Tensor, read_memory: Tensor) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor]]:
        """Record incoming state, predict from it, and emit an out-of-place transition."""
        segment_signal = images.mean(dim=(1, 2, 3, 4), keepdim=False).view(-1, 1, 1)
        transition_index = len(self.records)
        next_memory = read_memory * self.writer_scales[transition_index] + segment_signal
        prediction = read_memory.mean(dim=(1, 2), keepdim=True) * self.prediction_scale
        self.records.append((images, read_memory, next_memory))
        return {"prediction": prediction}, next_memory, {"state_mean": read_memory.mean()}


def _segments(
    *,
    metadata: bool = False,
    batch_size: int = 1,
    seq_names: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    """Create three independent prepared mappings with ordered frame values."""
    result = []
    for index in range(3):
        segment: dict[str, object] = {
            "images": torch.full((batch_size, 8, 3, 2, 2), float(index + 1))
        }
        if metadata:
            segment.update(segment_index=index, frame_start=index * 8, frame_stop=(index + 1) * 8)
        if seq_names is not None:
            segment["seq_name"] = list(seq_names)
        result.append(segment)
    return result


def test_e01a_forward_composes_lower_level_boundaries_once_and_preserves_outputs():
    """The public model boundary delegates exactly once without initializing memory."""
    model = object.__new__(RMVGGT)
    images, memory = torch.randn(1, 8, 3, 2, 2), torch.randn(1, 2, 1)
    cache, patch_start_idx = [None] * 24, 1
    expected = ({"prediction": torch.tensor(1.0)}, torch.tensor([2.0]), {"gate": torch.tensor(3.0)})
    model.encode_segment = Mock(return_value=(cache, patch_start_idx))
    model.forward_segment = Mock(return_value=expected)
    model.initial_memory = Mock(side_effect=AssertionError("forward must not initialize memory"))

    actual = RMVGGT.forward(model, images, memory)

    assert actual is expected
    model.encode_segment.assert_called_once_with(images)
    model.forward_segment.assert_called_once_with(cache, images, patch_start_idx, memory)
    model.initial_memory.assert_not_called()


def test_e01a_sequence_orders_calls_states_and_final_memory():
    """Three prepared mappings produce ordered predictions, diagnostics, and states."""
    model, segments = _SequenceModel(), _segments(metadata=True)

    result = run_recurrent_sequence(model, segments)

    assert model.module_calls == 3
    assert model.initial_calls == 1
    assert len(result.predictions) == len(result.diagnostics) == 3
    assert len(result.memory_states) == 4
    assert result.final_memory is result.memory_states[3]
    assert [record[0] for record in model.records] == [segment["images"] for segment in segments]
    assert model.records[0][1] is result.memory_states[0]
    assert model.records[1][1] is result.memory_states[1]
    assert model.records[2][1] is result.memory_states[2]
    assert model.records[0][2] is result.memory_states[1]
    assert model.records[1][2] is result.memory_states[2]
    assert model.records[2][2] is result.memory_states[3]


def test_e01a_sequence_requests_each_iterable_segment_after_the_previous_call():
    """The next prepared mapping is yielded only after the earlier call completes."""
    model, prepared = _SequenceModel(), _segments(metadata=True)

    def stream() -> Iterable[dict[str, object]]:
        for index, segment in enumerate(prepared):
            assert model.module_calls == index
            yield segment
            assert model.module_calls == index + 1

    result = run_recurrent_sequence(model, stream())

    assert len(result.predictions) == 3
    assert model.module_calls == 3


def test_e01a_sequence_preserves_full_bptt_and_reserves_final_state():
    """A final-segment loss crosses both transitions but cannot consume final state."""
    model = _SequenceModel()
    result = run_recurrent_sequence(model, _segments())
    result.memory_states[1].retain_grad()
    result.memory_states[2].retain_grad()
    loss = result.predictions[2]["prediction"].square().mean()
    assert torch.autograd.grad(
        loss, result.memory_states[3], allow_unused=True, retain_graph=True
    )[0] is None
    loss.backward()

    assert result.memory_states[1].grad_fn is not None
    assert result.memory_states[2].grad_fn is not None
    assert result.memory_states[1].grad is not None and result.memory_states[1].grad.abs().any()
    assert result.memory_states[2].grad is not None and result.memory_states[2].grad.abs().any()
    assert model.writer_scales.grad is not None
    assert model.writer_scales.grad[0].isfinite() and model.writer_scales.grad[0].abs().any()
    assert model.writer_scales.grad[1].isfinite() and model.writer_scales.grad[1].abs().any()
    assert model.writer_scales.grad[2] == 0
    assert model.initial_memory_bank.grad is not None and model.initial_memory_bank.grad.abs().any()
    assert not model.aggregator_scale.requires_grad and model.aggregator_scale.grad is None
    assert all(result.memory_states[3] is not record[1] for record in model.records)
    assert all(record[2] is not record[1] for record in model.records)


def test_e01a_sequence_resets_state_without_mutating_prepared_inputs():
    """Independent calls have fresh states and leave mappings and image values intact."""
    model, segments = _SequenceModel(), _segments(metadata=True)
    originals = [segment["images"].clone() for segment in segments]
    mapping_ids = [id(segment) for segment in segments]
    mapping_keys = [set(segment) for segment in segments]
    metadata = [
        (segment["segment_index"], segment["frame_start"], segment["frame_stop"])
        for segment in segments
    ]

    first = run_recurrent_sequence(model, segments)
    second = run_recurrent_sequence(model, segments)

    assert model.initial_calls == 2
    assert first.memory_states[0] is not second.memory_states[0]
    assert mapping_ids == [id(segment) for segment in segments]
    assert mapping_keys == [set(segment) for segment in segments]
    assert metadata == [
        (segment["segment_index"], segment["frame_start"], segment["frame_stop"])
        for segment in segments
    ]
    for segment, original in zip(segments, originals):
        torch.testing.assert_close(segment["images"], original)


def test_e01a_sequence_accepts_stable_ordered_batch_identities():
    """Matching per-element identities permit recurrence across every segment."""
    model = _SequenceModel()

    result = run_recurrent_sequence(
        model, _segments(batch_size=2, seq_names=("first", "second"))
    )

    assert len(result.memory_states) == 4
    assert model.module_calls == 3


@pytest.mark.parametrize(
    "identity_update",
    [
        lambda segment: segment.update(seq_name=["second", "first"]),
        lambda segment: segment.update(seq_name=["other", "second"]),
        lambda segment: segment.pop("seq_name"),
    ],
)
def test_e01a_sequence_rejects_later_changed_batch_identity_after_prior_call(
    identity_update,
):
    """A reordered, replaced, or absent later identity stops before its call."""
    model = _SequenceModel()
    segments = _segments(batch_size=2, seq_names=("first", "second"))
    identity_update(segments[1])

    with pytest.raises(ValueError, match="identity"):
        run_recurrent_sequence(model, segments)

    assert model.initial_calls == 1
    assert model.module_calls == 1


def test_e01a_sequence_rejects_identity_cardinality_before_first_call():
    """Identity metadata must provide one ordered value for every batch row."""
    model = _SequenceModel()

    with pytest.raises(ValueError, match="identity"):
        run_recurrent_sequence(model, _segments(batch_size=2, seq_names=("first",)))

    assert model.initial_calls == model.module_calls == 0


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ("not a segment iterable", "ordered iterable"),
        ([], "exactly 3"),
        (_segments()[:2], "exactly 3"),
        ([{}] * 3, "images"),
        ([{"images": torch.randn(1, 8, 3, 2)}] * 3, "rank 5"),
        ([{"images": torch.randn(1, 7, 3, 2, 2)}] * 3, "frame"),
        ([{"images": torch.randn(1, 8, 2, 2, 2)}] * 3, "channel"),
    ],
)
def test_e01a_sequence_rejects_invalid_first_segment_before_execution(segments: object, message: str):
    """Count, mapping, image, frame, and channel violations fail before calls."""
    model = _SequenceModel()

    with pytest.raises((TypeError, ValueError), match=message):
        run_recurrent_sequence(model, segments)  # type: ignore[arg-type]

    assert model.module_calls == model.initial_calls == 0


@pytest.mark.parametrize(
    ("num_segments", "segment_frames"),
    [(0, 8), (3, 0), (True, 8)],
)
def test_e01a_sequence_rejects_invalid_schedule_before_execution(
    num_segments: int, segment_frames: int
):
    """Invalid configured dimensions fail before consuming any prepared mapping."""
    model = _SequenceModel()

    with pytest.raises(ValueError, match="num_segments|segment_frames"):
        run_recurrent_sequence(
            model, _segments(), num_segments=num_segments, segment_frames=segment_frames
        )

    assert model.module_calls == model.initial_calls == 0


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda items: items.__setitem__(1, {**items[1], "images": torch.randn(2, 8, 3, 2, 2)}), "batch"),
        (lambda items: items.__setitem__(1, {**items[1], "images": torch.randn(1, 8, 3, 3, 2)}), "spatial"),
        (lambda items: items.__setitem__(1, {**items[1], "images": items[1]["images"].double()}), "dtype"),
        (lambda items: items.__setitem__(1, {**items[1], "images": torch.empty(1, 8, 3, 2, 2, device="meta")}), "device"),
    ],
)
def test_e01a_sequence_rejects_later_tensor_mismatches_after_prior_calls(transform, message: str):
    """A later lazy mapping fails after exactly the preceding model calls."""
    model, segments = _SequenceModel(), _segments()
    transform(segments)

    with pytest.raises(ValueError, match=message):
        run_recurrent_sequence(model, segments)

    assert model.initial_calls == 1
    assert model.module_calls == 1


@pytest.mark.parametrize(
    "metadata",
    [
        [{"segment_index": 0, "frame_start": 0, "frame_stop": 8}, {"segment_index": 2, "frame_start": 8, "frame_stop": 16}, {"segment_index": 1, "frame_start": 16, "frame_stop": 24}],
        [{"segment_index": 0, "frame_start": 0, "frame_stop": 8}, {"segment_index": 1, "frame_start": 7, "frame_stop": 15}, {"segment_index": 2, "frame_start": 16, "frame_stop": 24}],
        [{"segment_index": 0, "frame_start": 0, "frame_stop": 8}, {"segment_index": 1, "frame_start": 9, "frame_stop": 17}, {"segment_index": 2, "frame_start": 17, "frame_stop": 25}],
        [{"segment_index": 0, "frame_start": 0, "frame_stop": 8}, {"segment_index": 1}, {"segment_index": 2, "frame_start": 16, "frame_stop": 24}],
    ],
)
def test_e01a_sequence_rejects_later_invalid_metadata_after_prior_calls(metadata: list[dict[str, int]]):
    """Later reordered, overlapping, gapped, and incomplete ranges stop the stream."""
    model, segments = _SequenceModel(), _segments()
    for segment, fields in zip(segments, metadata):
        segment.update(fields)

    with pytest.raises(ValueError, match="metadata"):
        run_recurrent_sequence(model, segments)

    assert model.initial_calls == 1
    assert model.module_calls == 1
