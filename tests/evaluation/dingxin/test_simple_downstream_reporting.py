from __future__ import annotations

import pandas as pd

from chronaris.evaluation.dingxin.simple_downstream_reporting import (
    _augment_ratios,
    _primary_result_table,
    _summary_max_difference,
)


def _row(method, task, metric, value, *, seed=17, fold="fold01"):
    return {
        "seed": seed,
        "fold_id": fold,
        "method_name": method,
        "task_name": task,
        "metric_name": metric,
        "metric_value": value,
    }


def test_augment_ratios_uses_skill_definition():
    metrics = pd.DataFrame(
        [
            _row("chronaris", "future_maneuver", "skill_vs_current_maneuver", 0.75),
            _row("chronaris", "future_physiology", "skill_vs_persistence", -3.0),
        ]
    )
    augmented = _augment_ratios(metrics)
    maneuver = augmented[augmented["metric_name"] == "rmse_ratio_vs_current"]
    physiology = augmented[augmented["metric_name"] == "rmse_ratio_vs_persistence"]
    assert maneuver["metric_value"].iloc[0] == 0.5
    assert physiology["metric_value"].iloc[0] == 2.0


def test_summary_recomputation_matches_published_shape():
    metrics = pd.DataFrame(
        [
            _row("chronaris", "future_maneuver", "macro_f1", 0.2, seed=17),
            _row("chronaris", "future_maneuver", "macro_f1", 0.4, seed=29),
        ]
    )
    published = pd.DataFrame(
        [
            {
                "method_name": "chronaris",
                "task_name": "future_maneuver",
                "metric_name": "macro_f1",
                "mean": 0.3,
                "std": 0.14142135623730953,
                "minimum": 0.2,
                "maximum": 0.4,
                "unit_count": 2,
            }
        ]
    )
    assert _summary_max_difference(metrics, published) < 1e-12


def test_primary_result_table_keeps_all_reader_methods():
    rows = []
    maneuver = (
        "macro_f1",
        "balanced_accuracy",
        "spearman",
        "normalized_mae",
        "skill_vs_current_maneuver",
        "rmse_ratio_vs_current",
    )
    physiology = (
        "standardized_rmse_macro",
        "standardized_mae_macro",
        "skill_vs_persistence",
        "rmse_ratio_vs_persistence",
        "positive_skill_field_ratio",
        "eeg_rmse_macro",
        "spo2_rmse_macro",
    )
    for method in (
        "physiology_only",
        "vehicle_only",
        "naive_time_sync",
        "mult",
        "contiformer",
        "chronaris",
    ):
        rows.extend(_row(method, "future_maneuver", metric, 1.0) for metric in maneuver)
        rows.extend(_row(method, "future_physiology", metric, 1.0) for metric in physiology)
    table = _primary_result_table(pd.DataFrame(rows))
    assert len(table) == 6
    assert set(table["method_name"]) == {
        "physiology_only",
        "vehicle_only",
        "naive_time_sync",
        "mult",
        "contiformer",
        "chronaris",
    }
