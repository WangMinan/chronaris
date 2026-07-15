from __future__ import annotations

import numpy as np
import torch

from chronaris.evaluation.application_tasks.clean_input_contract import (
    mask_time_like_vehicle_channels,
    phase_residualize,
)
from chronaris.evaluation.application_tasks.target_reconstruction_contracts import (
    decide_target_reconstruction_allowance,
)
from chronaris.evaluation.application_tasks.target_reconstruction_maneuver import (
    _trend_class,
)
from chronaris.evaluation.application_tasks.target_reconstruction_physiology import (
    _descriptors,
    _weighted_median,
)
from chronaris.evaluation.application_tasks.target_reconstruction_training import (
    _load_sealed_outer_lineage,
)
from chronaris.representation import DualStreamObservationBatch


def test_time_like_vehicle_channels_are_fail_closed() -> None:
    batch = _batch()
    cleaned = mask_time_like_vehicle_channels(batch, (1,))
    assert torch.count_nonzero(cleaned.vehicle_values[..., 1]) == 0
    assert not cleaned.vehicle_feature_mask[..., 1].any()
    assert torch.count_nonzero(cleaned.vehicle_observation_age_s[..., 1]) == 0
    assert cleaned.vehicle_feature_mask[..., 0].all()
    assert cleaned.vehicle_point_mask.all()


def test_phase_residualization_removes_train_fitted_polynomial() -> None:
    train_phase = np.linspace(0.0, 1.0, 20)
    validation_phase = np.linspace(0.05, 0.95, 8)
    train = np.column_stack((train_phase, train_phase**2, train_phase**3))
    validation = np.column_stack(
        (validation_phase, validation_phase**2, validation_phase**3)
    )
    train_residual, validation_residual = phase_residualize(
        train,
        validation,
        train_phase=train_phase,
        validation_phase=validation_phase,
        alpha=1e-9,
    )
    assert np.max(np.abs(train_residual)) < 1e-5
    assert np.max(np.abs(validation_residual)) < 1e-5


def test_eeg_descriptors_ignore_dc_offset() -> None:
    times = np.arange(100, dtype=np.float64) / 20.0
    values = np.sin(2 * np.pi * 2.0 * times)
    baseline = _descriptors((times, values), "eeg")
    shifted = _descriptors((times, values + 50.0), "eeg")
    assert baseline.keys() == shifted.keys()
    for key in baseline:
        assert np.isclose(baseline[key], shifted[key], atol=1e-10)


def test_maneuver_trend_and_weighted_median_contracts() -> None:
    assert _trend_class(-0.5, 0.2) == 0
    assert _trend_class(0.1, 0.2) == 1
    assert _trend_class(0.5, 0.2) == 2
    assert _weighted_median([1.0, 2.0, 9.0], [0.6, 0.3, 0.1]) == 1.0


def test_allowance_requires_every_preregistered_gate() -> None:
    result = decide_target_reconstruction_allowance(
        panel={
            "completed_method_count": 6,
            "completed_main_split_count": 6,
            "missing_method_split_units": 0,
        },
        time_shortcut={"standardized_gain": 0.15},
        maneuver_score={"median_spearman": 0.65},
        maneuver_trend={"mean_macro_f1": 0.75, "worst_macro_f1": 0.60},
        physiology_residual={
            "median_rmse_ratio": 0.90,
            "mean_skill": 0.10,
            "positive_skill_split_count": 4,
        },
        high_residual={
            "mean_normalized_ap": 0.25,
            "median_normalized_ap": 0.15,
            "positive_split_count": 5,
        },
        third_pool={"completed_continuous_task_count": 2},
        protocol_valid=True,
    )
    assert result["allow_safe_fusion"] is True
    failed = decide_target_reconstruction_allowance(
        panel={
            "completed_method_count": 6,
            "completed_main_split_count": 6,
            "missing_method_split_units": 0,
        },
        time_shortcut={"standardized_gain": 0.14},
        maneuver_score={"median_spearman": 0.65},
        maneuver_trend={"mean_macro_f1": 0.75, "worst_macro_f1": 0.60},
        physiology_residual={
            "median_rmse_ratio": 0.90,
            "mean_skill": 0.10,
            "positive_skill_split_count": 4,
        },
        high_residual={
            "mean_normalized_ap": 0.25,
            "median_normalized_ap": 0.15,
            "positive_split_count": 5,
        },
        third_pool={"completed_continuous_task_count": 2},
        protocol_valid=True,
    )
    assert failed["allow_safe_fusion"] is False
    assert failed["primary_blocker"] == "time_shortcut_suppressed"


def test_sealed_outer_lineage_binds_ids_without_loading_samples(tmp_path) -> None:
    path = tmp_path / "split_manifest.json"
    path.write_text(
        '{"split_protocols":[{"fold_id":"fold01",'
        '"classification_test_context_ids":["a","b"],'
        '"response_test_context_ids":["b","c"]}]}',
        encoding="utf-8",
    )
    assert _load_sealed_outer_lineage(path) == {"fold01": ("a", "b", "c")}


def _batch() -> DualStreamObservationBatch:
    values = torch.ones((1, 3, 2), dtype=torch.float32)
    timestamps = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    query_timestamps = torch.linspace(0.0, 2.0, 96, dtype=torch.float64).unsqueeze(0)
    point_mask = torch.ones((1, 3), dtype=torch.bool)
    feature_mask = torch.ones((1, 3, 2), dtype=torch.bool)
    age = torch.ones((1, 3, 2), dtype=torch.float32)
    return DualStreamObservationBatch(
        sample_ids=("sample",),
        group_ids=("group",),
        physiology_values=values.clone(),
        physiology_timestamps_s=timestamps.clone(),
        physiology_point_mask=point_mask.clone(),
        physiology_feature_mask=feature_mask.clone(),
        physiology_observation_age_s=age.clone(),
        vehicle_values=values.clone(),
        vehicle_timestamps_s=timestamps.clone(),
        vehicle_point_mask=point_mask.clone(),
        vehicle_feature_mask=feature_mask.clone(),
        vehicle_observation_age_s=age.clone(),
        query_timestamps_s=query_timestamps,
        source_sample_hashes=("a" * 64,),
    )
