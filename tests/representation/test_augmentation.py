from __future__ import annotations

import inspect

from chronaris.representation import (
    build_augmentation_realization,
    build_batch_augmentation_realizations,
)


def test_augmentation_realization_is_seeded_by_sample_epoch_not_method():
    first = build_augmentation_realization(sample_id="sample_a", epoch=3, global_seed=17)
    second = build_augmentation_realization(sample_id="sample_a", epoch=3, global_seed=17)
    changed = build_augmentation_realization(sample_id="sample_a", epoch=4, global_seed=17)

    assert first == second
    assert first.augmentation_id != changed.augmentation_id
    assert "method" not in inspect.signature(build_augmentation_realization).parameters


def test_batch_augmentation_preserves_sample_order_and_never_drops_both_modalities():
    values = build_batch_augmentation_realizations(
        ("sample_b", "sample_a"),
        epoch=1,
        global_seed=23,
    )

    assert tuple(value.sample_id for value in values) == ("sample_b", "sample_a")
    assert all(value.dropped_modality in {None, "physiology", "vehicle"} for value in values)


def test_augmentation_honors_mixed_context_lengths_and_hashes_them():
    durations = (10.0, 12.0, 30.0)
    plans = build_batch_augmentation_realizations(
        ("short", "medium", "long"), epoch=1, global_seed=17,
        context_duration_s=durations,
    )
    for plan, duration in zip(plans, durations, strict=True):
        for stream in ("physiology", "vehicle"):
            assert 0 <= getattr(plan, f"{stream}_block_start_s")
            assert getattr(plan, f"{stream}_block_start_s") + getattr(plan, f"{stream}_block_duration_s") <= duration
    changed = build_augmentation_realization(sample_id="short", epoch=1, global_seed=17, context_duration_s=30)
    assert plans[0].augmentation_id != changed.augmentation_id
