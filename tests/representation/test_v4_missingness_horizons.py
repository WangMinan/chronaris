from collections import Counter
from dataclasses import replace

import pytest
import torch

from chronaris.representation import (AugmentationPolicy, build_augmentation_realization,
    apply_augmentation_realizations, build_common_pretext_targets, collate_observation_samples)
from tests.representation.test_augmentation_apply import _sample


def test_fixed_missingness_mixture_distribution_and_exact_clean_branch():
    policy = AugmentationPolicy(missingness_mixture=True)
    counts = Counter()
    examples = {}
    for i in range(10000):
        duration = (10., 12., 30.)[i % 3]
        plan = build_augmentation_realization(sample_id="a", epoch=i, global_seed=17,
            context_duration_s=duration, policy=policy)
        counts[plan.condition] += 1
        examples.setdefault(plan.condition, plan)
        assert plan.timestamp_jitter_sigma_s == plan.physiology_clock_offset_s == plan.vehicle_clock_offset_s == 0
        for stream in ("physiology", "vehicle"):
            length = getattr(plan, f"{stream}_block_duration_s")
            assert .1 * duration <= length <= .8 * duration if plan.condition == "block" else length == 0
            assert 0 <= getattr(plan, f"{stream}_block_start_s") <= duration - length
        assert .05 <= plan.point_dropout_probability <= .30 if plan.condition == "random" else plan.point_dropout_probability == 0
    for name, expected in (("clean", .4), ("random", .25), ("block", .25),
                           ("physiology_missing", .05), ("vehicle_missing", .05)):
        assert counts[name] / 10000 == pytest.approx(expected, abs=.015)
    batch = collate_observation_samples([_sample("a")])
    for condition in ("clean", "physiology_missing", "vehicle_missing"):
        plan = examples[condition]
        first = apply_augmentation_realizations(batch, (plan,), policy=policy)
        second = apply_augmentation_realizations(batch, (plan,), policy=policy)
        for stream in ("physiology", "vehicle"):
            assert torch.equal(getattr(first.batch, f"{stream}_values"), getattr(second.batch, f"{stream}_values"))
            if plan.dropped_modality == stream:
                assert not getattr(first.batch, f"{stream}_point_mask").any()
            else:
                for field in ("values", "timestamps_s", "feature_mask", "point_mask"):
                    assert torch.equal(getattr(first.batch, f"{stream}_{field}"), getattr(batch, f"{stream}_{field}"))
        assert all(row.condition == condition for row in first.audit_rows)


def test_prediction_targets_use_actual_seconds_and_original_window_boundary():
    base = _sample("a")
    short = replace(base, sample_id="b", group_id="b")
    batch = collate_observation_samples([base, short])
    # Different query steps in one batch: future teacher queries must not use q+1.
    batch = replace(batch, query_timestamps_s=torch.stack((batch.query_timestamps_s[0],
        batch.query_timestamps_s[1] / 3)))
    policy = AugmentationPolicy(missingness_mixture=True)
    plans = tuple(build_augmentation_realization(sample_id=s, epoch=0, global_seed=17,
        context_duration_s=d, policy=policy) for s, d in zip(batch.sample_ids, batch.context_durations_s.tolist()))
    augmented = apply_augmentation_realizations(batch, plans, policy=policy)
    horizons = (.5, 2., 5.)
    targets = build_common_pretext_targets(batch, augmented, prediction_horizons_s=horizons)
    for b in range(2):
        for q, time in enumerate(batch.query_timestamps_s[b]):
            for h, horizon in enumerate(horizons):
                for stream, offset in (("physiology", 0), ("vehicle", 2)):
                    times = getattr(batch, f"{stream}_timestamps_s")[b]
                    source = torch.nonzero(getattr(batch, f"{stream}_point_mask")[b] & (times <= time + horizon)).flatten()
                    valid = bool(len(source)) and time + horizon < batch.context_durations_s[b]
                    assert bool(targets.next_query_mask[b, q, h, offset]) == valid
                    if valid:
                        assert targets.next_query_target[b, q, h, offset] == getattr(batch, f"{stream}_values")[b, source[-1], 0]
    assert torch.equal(targets.next_query_target[~targets.next_query_mask], torch.zeros_like(targets.next_query_target[~targets.next_query_mask]))
