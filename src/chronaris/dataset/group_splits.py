"""Small deterministic group split used by public-data outer folds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def split_group_train_validation(
    indices: Sequence[int],
    group_ids: Sequence[str],
    *,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    selected = tuple(int(index) for index in indices)
    if not selected or min(selected) < 0 or max(selected) >= len(group_ids):
        raise ValueError("group split indices are empty or outside group_ids")
    unique_groups = sorted({str(group_ids[index]) for index in selected})
    if len(unique_groups) < 2:
        raise ValueError("group validation requires at least two groups")
    shuffled = np.asarray(unique_groups, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = min(
        max(1, round(len(unique_groups) * validation_fraction)),
        len(unique_groups) - 1,
    )
    validation_groups = set(str(value) for value in shuffled[:validation_count])
    train = tuple(index for index in selected if str(group_ids[index]) not in validation_groups)
    validation = tuple(index for index in selected if str(group_ids[index]) in validation_groups)
    return train, validation
