from dataclasses import replace

import torch

from chronaris.representation import (
    apply_augmentation_realizations, build_batch_augmentation_realizations,
    build_explicit_time_shift_inputs, collate_observation_samples,
)
from tests.representation.test_contracts import _sample


def test_mixed_contexts_keep_their_own_clipping_and_shift_boundaries():
    durations = (10., 12., 30.)
    batch = collate_observation_samples([
        replace(_sample(str(i)), context_duration_s=duration)
        for i, duration in enumerate(durations)
    ])
    torch.testing.assert_close(batch.context_durations_s, torch.tensor(durations, dtype=torch.float64))
    plans = build_batch_augmentation_realizations(batch.sample_ids, epoch=1, global_seed=17,
                                                  context_duration_s=batch.context_durations_s.tolist())
    augmented = apply_augmentation_realizations(batch, plans)
    shifted = build_explicit_time_shift_inputs(batch, augmented.augmentation_ids, class_indices=(4, 4, 4))
    assert not shifted.shifted_batch.vehicle_point_mask[0].any()
    assert shifted.shifted_batch.vehicle_point_mask[1:].all()
    for row, duration in enumerate(durations):
        for value in (augmented.batch, shifted.shifted_batch):
            for stream in ("physiology", "vehicle"):
                times = getattr(value, f"{stream}_timestamps_s")[row]
                valid = getattr(value, f"{stream}_point_mask")[row]
                assert (times[valid] < duration).all()
