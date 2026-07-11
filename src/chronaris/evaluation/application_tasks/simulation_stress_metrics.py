"""Factor metadata and direction-normalized degradation slopes for G2 stress."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from chronaris.simulation.aviation_dual_stream import (
    locked_stress_observation_scenarios,
)


def stress_scenario_metadata():
    result = {}
    for scenario in locked_stress_observation_scenarios():
        name = scenario.scenario_id
        if name.startswith("timestamp_jitter_"):
            factor, level = "timestamp_jitter_ms", scenario.physiology_jitter_std_ms
        elif name.startswith("clock_offset_"):
            factor, level = "absolute_clock_offset_s", abs(scenario.physiology_clock_offset_s)
        elif name.startswith("clock_drift_"):
            factor, level = "absolute_clock_drift_ppm", abs(scenario.physiology_clock_drift_ppm)
        elif name.startswith("random_missing_"):
            factor, level = "random_missing_rate", scenario.physiology_random_missing_rate
        elif name.startswith("contiguous_gap_"):
            factor, level = "contiguous_gap_s", scenario.physiology_block_gap_s
        elif name.startswith("physiology_lag_"):
            factor, level = "additional_physiology_lag_s", scenario.additional_physiology_lag_s
        elif name.startswith("observation_snr_"):
            factor, level = "snr_degradation_db", 30.0 - scenario.observation_snr_db
        elif name == "mixed_severe":
            factor, level = "mixed_severe", 1.0
        else:
            raise ValueError(f"unknown locked stress scenario: {name}")
        result[name] = {
            "stress_factor": factor,
            "stress_level": float(level),
        }
    return result


def build_stress_slope_rows(metric_rows):
    metadata = stress_scenario_metadata()
    grouped = defaultdict(list)
    for row in metric_rows:
        scenario = str(row["scenario_id"])
        factor = metadata[scenario]["stress_factor"]
        if factor == "mixed_severe" or row["value"] is None:
            continue
        key = (
            row["seed"],
            row["method"],
            row["task"],
            row["consumer"],
            row["metric"],
            factor,
            row["direction"],
        )
        normalized_score = (
            float(row["value"])
            if row["direction"] == "higher"
            else -float(row["value"])
        )
        grouped[key].append(
            (metadata[scenario]["stress_level"], normalized_score)
        )
    rows = []
    for key, values in sorted(grouped.items()):
        by_level = defaultdict(list)
        for level, score in values:
            by_level[level].append(score)
        levels = np.asarray(sorted(by_level), dtype=np.float64)
        scores = np.asarray(
            [np.mean(by_level[level]) for level in levels], dtype=np.float64
        )
        if len(levels) < 2 or levels[-1] <= levels[0]:
            continue
        normalized_levels = (levels - levels[0]) / (levels[-1] - levels[0])
        slope = float(np.polyfit(normalized_levels, scores, 1)[0])
        rows.append(
            {
                "seed": key[0],
                "method": key[1],
                "task": key[2],
                "consumer": key[3],
                "metric": key[4],
                "stress_factor": key[5],
                "source_direction": key[6],
                "direction_normalized": "higher_is_better",
                "level_count": len(levels),
                "minimum_level": float(levels[0]),
                "maximum_level": float(levels[-1]),
                "baseline_score": float(scores[0]),
                "maximum_stress_score": float(scores[-1]),
                "maximum_stress_change": float(scores[-1] - scores[0]),
                "degradation_slope": slope,
            }
        )
    return tuple(rows)
