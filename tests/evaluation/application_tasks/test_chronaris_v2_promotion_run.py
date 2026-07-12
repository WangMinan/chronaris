from __future__ import annotations

import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_promotion_run import (
    CANONICAL_CONFIRMATION_SCENARIO,
    build_class_recall_rows,
    build_locked_seed_metric_rows,
)
from chronaris.evidence.downstream_application_data import (
    DINGXIN_PRIMARY,
    SIMULATION_PRIMARY,
)


METHODS = (
    "physiology_only", "vehicle_only", "naive_time_sync",
    "mult", "contiformer", "chronaris",
)


def test_locked_primary_extraction_requires_and_ranks_all_six_methods() -> None:
    dingxin_rows = []
    simulation_rows = []
    for seed in (17, 29, 43):
        for specification in DINGXIN_PRIMARY:
            direction = "lower" if specification["metric"] == "rmse" else "higher"
            for index, method in enumerate(METHODS):
                value = 1.0 - index * 0.1 if direction == "lower" else index * 0.1
                dingxin_rows.append(
                    {"seed": seed, "method": method, "direction": direction,
                     "mean": value, **{key: specification[key] for key in ("task", "consumer", "metric")}}
                )
        for specification in SIMULATION_PRIMARY:
            direction = "lower" if specification["metric"] == "rmse" else "higher"
            for index, method in enumerate(METHODS):
                value = 1.0 - index * 0.1 if direction == "lower" else index * 0.1
                simulation_rows.append(
                    {"seed": seed, "method": method, "direction": direction,
                     "value": value, "scenario_id": CANONICAL_CONFIRMATION_SCENARIO,
                     **{key: specification[key] for key in ("task", "consumer", "metric")}}
                )

    seed_rows, panels = build_locked_seed_metric_rows(
        pd.DataFrame(dingxin_rows),
        pd.DataFrame(simulation_rows),
        canonical_scenario=CANONICAL_CONFIRMATION_SCENARIO,
    )

    assert len(seed_rows) == 18
    assert len(panels) == 108
    assert all(row["rank_first"] for row in seed_rows)


def test_class_recall_rows_use_only_held_out_minirocket_predictions(tmp_path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"seed": 17, "method": "chronaris", "consumer": "minirocket",
             "role": "held_out", "task": "maneuver_intensity_classification",
             "truth": 0, "prediction": 0},
            {"seed": 17, "method": "chronaris", "consumer": "minirocket",
             "role": "held_out", "task": "maneuver_intensity_classification",
             "truth": 0, "prediction": 1},
            {"seed": 17, "method": "chronaris", "consumer": "linear",
             "role": "held_out", "task": "maneuver_intensity_classification",
             "truth": 0, "prediction": 0},
        ]
    ).to_csv(path, index=False)

    rows = build_class_recall_rows(path)

    assert rows == [{
        "seed": 17, "method": "chronaris", "class_id": 0,
        "recall": 0.5, "sample_count": 2,
    }]
