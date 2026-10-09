"""Deterministic rank assignments for complete training episodes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class EpisodeAssignment:
    """Identify one rank's real or padded episode in a synchronized group.

    Padded ranks load a finite repeat of the group's first real window and
    backpropagate its zero-scaled objective. Only real assignments contribute
    to episode counts, losses, and metrics.
    """

    window: Any
    identity: str
    is_real: bool
    objective_scale: float


def plan_episode_groups(
    windows: Sequence[Any], *, seed: int, epoch: int, world_size: int,
) -> Iterator[tuple[EpisodeAssignment, ...]]:
    """Shuffle metadata once and yield complete rank groups without loading data.

    Every eligible identity is assigned to one real rank exactly once. A short
    final group repeats its first real window on inactive ranks; the real
    objective scale compensates for DDP's world-size gradient average.
    Invalid dimensions, empty manifests, or duplicate identities raise
    ValueError before yielding any group.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    ordered = list(windows)
    identities = [window.identity for window in ordered]
    if not ordered or any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("windows must have nonempty identities")
    if len(set(identities)) != len(identities):
        raise ValueError("eligible windows must have distinct identities")
    random.Random(f"{seed}:{epoch}").shuffle(ordered)
    for offset in range(0, len(ordered), world_size):
        real = ordered[offset:offset + world_size]
        count = len(real)
        yield tuple(
            EpisodeAssignment(
                window=window,
                identity=window.identity,
                is_real=rank < count,
                objective_scale=world_size / count if rank < count else 0.0,
            )
            for rank, window in enumerate(
                real + [real[0]] * (world_size - count)
            )
        )
