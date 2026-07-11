from __future__ import annotations

import numpy as np
import pytest

from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
    paired_trajectory_statistic,
    segmentation_metrics,
)


def test_perfect_segmentation_scores_have_fixed_boundaries_and_zero_delay() -> None:
    truth = np.asarray(
        [
            [0] * 6 + [1] * 4 + [2] * 5 + [3] * 3 + [4] * 6,
            [0] * 4 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
        ]
    )
    metrics = segmentation_metrics(truth, truth, query_step_s=0.5)

    for name, (value, direction) in metrics.items():
        if name == "boundary_detection_delay_s":
            assert value == 0.0 and direction == "lower"
        else:
            assert value == 1.0 and direction == "higher"


def test_fusion_gain_normalizes_higher_and_lower_metrics() -> None:
    common = {
        "dataset": "simulation",
        "task": "task",
        "consumer": "linear",
        "seed": 17,
        "fold": "f",
        "role": "held_out",
    }
    rows = []
    for metric, direction, values in (
        ("macro_f1", "higher", (0.5, 0.6, 0.7)),
        ("rmse", "lower", (0.5, 0.4, 0.3)),
    ):
        for method, value in zip(
            ("physiology_only", "vehicle_only", "chronaris"),
            values,
            strict=True,
        ):
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "direction": direction,
                    "method": method,
                    "value": value,
                }
            )
    gains = compute_fusion_gain_rows(rows, fusion_methods=("chronaris",))

    gain_by_metric = {row["metric"]: row["fusion_gain"] for row in gains}
    assert gain_by_metric["macro_f1"] == pytest.approx(0.1)
    assert gain_by_metric["rmse"] == pytest.approx(0.1)
    assert all(row["direction_normalized"] == "higher_is_better" for row in gains)


def test_paired_statistics_use_trajectory_units_and_are_seed_reproducible() -> None:
    first = np.asarray([0.8, 0.7, 0.9, 0.6])
    second = np.asarray([0.5, 0.6, 0.7, 0.55])
    one = paired_trajectory_statistic(first, second, seed=17, bootstrap_repetitions=200)
    two = paired_trajectory_statistic(first, second, seed=17, bootstrap_repetitions=200)

    assert one == two
    assert one.independent_unit_count == 4
    assert one.mean_difference > 0
    assert 0 <= one.permutation_p_value <= 1
