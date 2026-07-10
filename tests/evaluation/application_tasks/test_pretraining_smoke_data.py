from __future__ import annotations

from pathlib import Path

import pytest

from chronaris.evaluation.application_tasks.pretraining_smoke_data import (
    select_one_g1_observation_per_profile,
)


def _path(profile: int, trajectory: int, *, split: str = "train") -> Path:
    return Path(
        f"root/{split}/train_profile_{profile:03d}/"
        f"g1_state_space__train_profile_{profile:03d}__trajectory_{trajectory:03d}/"
        "clean_asynchronous/raw_dual_stream.npz"
    )


def test_pretraining_smoke_selects_one_trajectory_per_distinct_profile() -> None:
    paths = [
        _path(profile, trajectory)
        for profile in range(4)
        for trajectory in (1, 2)
    ]
    selected = select_one_g1_observation_per_profile(paths, sample_count=4)

    assert len(selected) == 4
    assert len({path.parents[2].name for path in selected}) == 4
    assert all("/train/" in path.as_posix() for path in selected)


def test_pretraining_smoke_rejects_locked_test_paths() -> None:
    with pytest.raises(ValueError, match="only from train"):
        select_one_g1_observation_per_profile(
            [_path(0, 0, split="locked_test")],
            sample_count=1,
        )
