from dataclasses import replace

import numpy as np
import pytest
import torch

from chronaris.evaluation.application_tasks.v4_development_conditions import (
    DEVELOPMENT_CONDITIONS, apply_development_missingness, development_condition_protocol)
from chronaris.representation import collate_observation_samples
from tests.representation.test_contracts import _sample


def test_development_missingness_is_paired_deterministic_and_covers_each_context():
    original = _sample("v4_dev_mask", shift=0.)
    updates = {}
    for stream in ("physiology", "vehicle"):
        width = getattr(original, stream + "_values").shape[1]
        updates[stream + "_timestamps_s"] = np.linspace(0., 30., 301, endpoint=False)
        updates[stream + "_values"] = np.ones((301, width), dtype=np.float32)
        updates[stream + "_feature_mask"] = np.ones((301, width), dtype=bool)
    original = replace(original, **updates)
    for condition in DEVELOPMENT_CONDITIONS:
        changed = apply_development_missingness(original, condition)
        repeated = apply_development_missingness(original, condition)
        for stream in ("physiology", "vehicle"):
            values = getattr(changed, stream + "_values")
            mask = getattr(changed, stream + "_feature_mask")
            times = getattr(changed, stream + "_timestamps_s")
            positions = np.searchsorted(getattr(original, stream + "_timestamps_s"), times)
            np.testing.assert_array_equal(getattr(original, stream + "_timestamps_s")[positions], times)
            np.testing.assert_array_equal(mask, getattr(repeated, stream + "_feature_mask"))
            np.testing.assert_array_equal(values[mask], getattr(original, stream + "_values")[positions][mask])
            assert not values[~mask].any()
            if condition == "random_missing_30pct":
                assert .20 < 1 - len(times) / 301 < .40
            if condition == "contiguous_gap_15s":
                original_time = getattr(original, stream + "_timestamps_s")
                np.testing.assert_array_equal(times, original_time[(original_time < 7.5) | (original_time >= 22.5)])
            if condition == "contiguous_gap_30s" or condition == stream + "_missing":
                assert not mask.any()
        batch = collate_observation_samples([changed])
        if condition == "contiguous_gap_30s":
            assert not batch.physiology_point_mask.any() and not batch.vehicle_point_mask.any()
            assert torch.isinf(batch.physiology_observation_age_s).all()
        assert changed.sample_id == original.sample_id and changed.group_id == original.group_id
    assert original.physiology_feature_mask.all() and original.vehicle_feature_mask.all()
    protocol = development_condition_protocol()
    assert len(protocol["conditions"]) == 8 and protocol["roles"] == ["validation"]
    timing = protocol["timing_scenarios"]
    assert timing[1]["physiology_clock_offset_s"] == 1. and timing[1]["additional_physiology_lag_s"] == 0.
    assert timing[2]["physiology_clock_offset_s"] == 0. and timing[2]["additional_physiology_lag_s"] == 15.
    with pytest.raises(ValueError, match="30 seconds"):
        apply_development_missingness(replace(_sample("long"), context_duration_s=31.), "contiguous_gap_15s")
