"""Deterministic metadata-only grouping of complete training windows."""

from __future__ import annotations

import pytest

from training.data.datasets.vkitti_sequence import VKittiEpisodeWindow
from training.episode_grouping import plan_episode_groups


def _windows(count: int) -> tuple[VKittiEpisodeWindow, ...]:
    """Create distinct lightweight windows without decoding frames."""
    return tuple(
        VKittiEpisodeWindow("Scene01", "clone", "Camera_0", start, (start, start + 1))
        for start in range(count)
    )


def test_e01a_groups_are_reproducible_disjoint_and_padded_only_at_end() -> None:
    """Every identity appears once as real and padding repeats a finite window."""
    windows = _windows(7)
    first = list(plan_episode_groups(windows, seed=17, epoch=3, world_size=3))
    second = list(plan_episode_groups(windows, seed=17, epoch=3, world_size=3))
    assert first == second
    assert len(first) == 3
    assert all(len(group) == 3 for group in first)
    real = [slot for group in first for slot in group if slot.is_real]
    assert {slot.identity for slot in real} == {window.identity for window in windows}
    assert len(real) == len(windows)
    assert [sum(slot.is_real for slot in group) for group in first] == [3, 3, 1]
    assert all(slot.objective_scale == 1.0 for group in first[:2] for slot in group)
    assert first[-1][0].objective_scale == 3.0
    assert all(
        not slot.is_real and slot.objective_scale == 0.0
        and slot.window is first[-1][0].window
        for slot in first[-1][1:]
    )
    changed = list(plan_episode_groups(windows, seed=17, epoch=4, world_size=3))
    assert [slot.identity for group in first for slot in group if slot.is_real] != [
        slot.identity for group in changed for slot in group if slot.is_real
    ]


@pytest.mark.parametrize(
    ("windows", "seed", "epoch", "world_size"),
    [
        ((), 17, 0, 2),
        (_windows(2), 17, -1, 2),
        (_windows(2), 17, 0, 0),
        ((_windows(1)[0], _windows(1)[0]), 17, 0, 2),
    ],
)
def test_e01a_grouping_rejects_invalid_or_duplicate_windows(
    windows: tuple[VKittiEpisodeWindow, ...], seed: int, epoch: int, world_size: int,
) -> None:
    """Bad planner inputs fail before producing a rank assignment."""
    with pytest.raises(ValueError):
        list(plan_episode_groups(windows, seed=seed, epoch=epoch, world_size=world_size))
