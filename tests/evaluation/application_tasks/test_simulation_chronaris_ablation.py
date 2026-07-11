from __future__ import annotations

from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_consumer_run import (
    _build_full_ablation_metric_deltas,
)


def test_full_ablation_metric_delta_normalizes_metric_direction() -> None:
    common = {
        "seed": 17,
        "task": "simulated_workload_classification",
        "consumer": "linear",
        "role": "held_out",
        "available": True,
    }
    rows = [
        {
            **common,
            "method": "chronaris",
            "metric": "macro_f1",
            "direction": "higher",
            "value": 0.8,
        },
        {
            **common,
            "method": "chronaris_no_physics",
            "metric": "macro_f1",
            "direction": "higher",
            "value": 0.7,
        },
        {
            **common,
            "method": "chronaris",
            "metric": "rmse",
            "direction": "lower",
            "value": 0.2,
        },
        {
            **common,
            "method": "chronaris_no_physics",
            "metric": "rmse",
            "direction": "lower",
            "value": 0.3,
        },
    ]

    deltas = _build_full_ablation_metric_deltas(rows)

    assert len(deltas) == 2
    assert all(row["positive_favors_full"] for row in deltas)
    assert all(abs(row["full_advantage_normalized"] - 0.1) < 1e-8 for row in deltas)
