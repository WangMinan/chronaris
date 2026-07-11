from __future__ import annotations

from chronaris.evaluation.application_tasks.simulation_stress_metrics import (
    build_stress_slope_rows,
    stress_scenario_metadata,
)


def test_stress_metadata_and_direction_normalized_slopes() -> None:
    metadata = stress_scenario_metadata()
    rows = []
    for scenario, score, error in (
        ("timestamp_jitter_000ms", 1.0, 0.1),
        ("timestamp_jitter_020ms", 0.9, 0.2),
        ("timestamp_jitter_050ms", 0.7, 0.4),
        ("timestamp_jitter_100ms", 0.4, 0.8),
    ):
        common = {
            "seed": 17,
            "method": "chronaris",
            "task": "task",
            "consumer": "linear",
            "scenario_id": scenario,
        }
        rows.extend(
            (
                {**common, "metric": "score", "direction": "higher", "value": score},
                {**common, "metric": "error", "direction": "lower", "value": error},
            )
        )

    slopes = build_stress_slope_rows(rows)

    assert len(metadata) == 35
    assert metadata["clock_offset_-3.00s"]["stress_level"] == 3.0
    assert metadata["observation_snr_05db"]["stress_level"] == 25.0
    assert len(slopes) == 2
    assert all(row["degradation_slope"] < 0 for row in slopes)
    assert all(row["direction_normalized"] == "higher_is_better" for row in slopes)
