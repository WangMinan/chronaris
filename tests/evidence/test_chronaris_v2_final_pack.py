from __future__ import annotations

import pandas as pd

from chronaris.evidence.chronaris_v2_final_pack import (
    _mechanism_comparison,
    _primary_summary,
    _public_summary,
)


def test_primary_summary_applies_strict_directional_thresholds() -> None:
    frame = pd.DataFrame([
        {"metric_name": "dingxin_maneuver_macro_f1", "method": "chronaris", "direction": "higher", "value": 0.95},
        {"metric_name": "dingxin_maneuver_macro_f1", "method": "chronaris", "direction": "higher", "value": 0.96},
        {"metric_name": "dingxin_response_rmse", "method": "chronaris", "direction": "lower", "value": 0.28},
    ])

    result = _primary_summary(frame)

    assert result["threshold_passed"].all()
    assert result.loc[result["metric_name"].eq("dingxin_maneuver_macro_f1"), "value"].iloc[0] == 0.955


def test_mechanism_comparison_is_same_target_and_version_paired() -> None:
    rows = [
        {"metric": "mae_s", "target": "clock", "method": "chronaris", "value": 1.0},
        {"metric": "mae_s", "target": "lag", "method": "chronaris", "value": 2.0},
    ]
    result = _mechanism_comparison(pd.DataFrame(rows), pd.DataFrame([
        {**rows[0], "value": 1.05}, {**rows[1], "value": 2.3},
    ]))

    assert result.set_index("target").loc["clock", "within_10_percent"]
    assert not result.set_index("target").loc["lag", "within_10_percent"]


def test_public_summary_flattens_dataset_task_entries() -> None:
    result = _public_summary(pd.DataFrame([
        {"dataset_id": "nasa_csm", "candidate_id": "locked", "seed": 17, "fold_count": 17, "combined_macro_f1": 0.5},
        {"dataset_id": "uab_workload_dataset", "candidate_id": "locked", "seed": 17, "fold_count": 16, "mean_rmse": 1.0},
    ]))

    assert list(result["dataset_id"]) == ["nasa_csm", "uab_workload_dataset"]
    assert list(result["seed"]) == [17, 17]
