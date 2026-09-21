"""Loss evaluation and aggregation for ordered recurrent segments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from training.recurrent_sequence import RecurrentSequenceResult


@dataclass
class RecurrentLossResult:
    """Hold the differentiable sequence objective and original segment losses.

    Attributes:
        objective: Arithmetic mean of one scalar objective per segment.
        segment_losses: Original loss mappings in segment order, retained by identity.
    """

    objective: Tensor
    segment_losses: list[Mapping[str, Tensor]]


def compute_recurrent_losses(
    sequence_result: RecurrentSequenceResult,
    segments: Sequence[Mapping[str, Any]],
    loss_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Tensor]],
    *,
    num_segments: int = 3,
) -> RecurrentLossResult:
    """Evaluate ordered segment losses and form one equal-weight objective.

    Args:
        sequence_result: Predictions produced once for every ordered segment.
        segments: Corresponding independently prepared target mappings.
        loss_fn: Callable accepting one prediction mapping and one target mapping.
        num_segments: Required positive count of prediction and target mappings.

    Returns:
        A differentiable mean objective and the unmodified per-segment loss mappings.

    Raises:
        TypeError: If inputs or loss results do not satisfy their mapping and tensor contracts.
        ValueError: If counts or scalar-objective validity requirements are not met.

    Invariants:
        The function does not mutate inputs or returned loss mappings, run a model,
        prepare data, or invoke gradient, optimization, logging, or checkpoint actions.
    """
    if not isinstance(num_segments, int) or isinstance(num_segments, bool) or num_segments <= 0:
        raise ValueError("num_segments must be a positive integer")
    if not isinstance(sequence_result, RecurrentSequenceResult):
        raise TypeError("sequence_result must be a recurrent sequence result")
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
        raise TypeError("segments must be a sequence of target mappings")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")

    predictions = sequence_result.predictions
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence of mappings")

    prediction_count = len(predictions)
    target_count = len(segments)
    if prediction_count != target_count:
        raise ValueError("predictions and segments must contain matching counts")
    if prediction_count != num_segments:
        raise ValueError(
            f"predictions must contain exactly {num_segments} mappings, got {prediction_count}"
        )
    if target_count != num_segments:
        raise ValueError(
            f"segments must contain exactly {num_segments} mappings, got {target_count}"
        )

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            raise TypeError(f"prediction {index} must be a mapping")
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise TypeError(f"segment {index} must be a mapping")

    segment_losses: list[Mapping[str, Tensor]] = []
    segment_objectives: list[Tensor] = []
    for index, (prediction, segment) in enumerate(zip(predictions, segments)):
        loss_dict = loss_fn(prediction, segment)
        if not isinstance(loss_dict, Mapping):
            raise TypeError(f"loss result for segment {index} must be a mapping")
        if "objective" not in loss_dict:
            raise ValueError(f"loss result for segment {index} is missing objective")
        objective = loss_dict["objective"]
        if not torch.is_tensor(objective):
            raise TypeError(f"loss objective for segment {index} must be a tensor")
        if objective.ndim != 0:
            raise ValueError(f"loss objective for segment {index} must be a scalar tensor")
        if not torch.isfinite(objective).all():
            raise ValueError(f"loss objective for segment {index} must be finite")
        segment_losses.append(loss_dict)
        segment_objectives.append(objective)

    return RecurrentLossResult(
        objective=torch.stack(segment_objectives).mean(),
        segment_losses=segment_losses,
    )
