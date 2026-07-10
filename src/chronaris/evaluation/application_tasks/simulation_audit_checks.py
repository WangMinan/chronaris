"""Acceptance checks for the aviation dual-stream simulation benchmark."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.simulation.aviation_dual_stream.generator import generate_sortie


def build_simulation_acceptance_rows(
    *,
    mode: str,
    validation: pd.DataFrame,
    paired: pd.DataFrame,
    split_identity: Mapping[str, object],
    expected_latent_count: int,
    expected_scenario_count: int,
) -> list[dict[str, object]]:
    """Evaluate the fixed smoke or formal simulation acceptance contract."""

    latent = validation.drop_duplicates(["split_id", "trajectory_id"])
    clean = validation.loc[validation["scenario_id"] == "clean_asynchronous"].copy()
    clean["lag_error_s"] = (
        clean["estimated_primary_response_lag_s"]
        - clean["true_primary_response_lag_s"]
    ).abs()
    workload_ratios = {
        "low": float(latent["workload_low_ratio"].mean()),
        "medium": float(latent["workload_medium_ratio"].mean()),
        "high": float(latent["workload_high_ratio"].mean()),
    }
    source_scan_pass = _source_independence_scan()
    rows = [
        _check(
            "latent_count",
            len(latent) == expected_latent_count,
            len(latent),
            expected_latent_count,
        ),
        _check(
            "scenario_count",
            len(validation) == expected_scenario_count,
            len(validation),
            expected_scenario_count,
        ),
        _check(
            "split_identity_disjoint",
            bool(split_identity["disjoint"]),
            split_identity["pairwise_intersection_sizes"],
            0,
        ),
        _check(
            "paired_latent_hash",
            bool(paired["latent_hash_shared"].all()),
            int(paired["latent_hash_shared"].sum()),
            len(paired),
        ),
        _check(
            "state_coverage_per_sortie",
            bool(latent["all_states_present"].all()),
            int(latent["all_states_present"].sum()),
            len(latent),
        ),
        _check(
            "minimum_complete_events",
            bool((latent["event_count"] >= 2).all()),
            int(latent["event_count"].min()),
            2,
        ),
        _check(
            "finite_vehicle_values",
            bool(validation["vehicle_values_finite"].all()),
            int(validation["vehicle_values_finite"].sum()),
            len(validation),
        ),
        _check(
            "finite_physiology_values",
            bool(validation["physiology_values_finite"].all()),
            int(validation["physiology_values_finite"].sum()),
            len(validation),
        ),
        _generator_residual_check(
            latent=latent,
            family="g1_state_space",
            metric="physical_residual_abs_median",
            check_id="g1_residual_median",
            threshold=0.10,
        ),
        _generator_residual_check(
            latent=latent,
            family="g2_event_spline",
            metric="physical_residual_abs_q95",
            check_id="g2_residual_q95",
            threshold=0.35,
        ),
        _check(
            "clean_lag_within_one_second",
            float((clean["lag_error_s"] <= 1.0).mean()) >= 0.90,
            float((clean["lag_error_s"] <= 1.0).mean()),
            0.90,
        ),
        _clock_mapping_check(validation),
        _check(
            "source_method_independence",
            source_scan_pass,
            source_scan_pass,
            True,
        ),
    ]
    if mode == "formal":
        locked_profiles_pass = _locked_profiles_have_high_load(latent)
        maneuver_types_pass = _all_types_per_split(latent)
        missing_error = _random_missing_max_error(validation)
        rows.extend(
            [
                _check(
                    "workload_low_coverage",
                    workload_ratios["low"] >= 0.15,
                    workload_ratios["low"],
                    0.15,
                ),
                _check(
                    "workload_medium_coverage",
                    workload_ratios["medium"] >= 0.15,
                    workload_ratios["medium"],
                    0.15,
                ),
                _check(
                    "workload_high_coverage",
                    workload_ratios["high"] >= 0.15,
                    workload_ratios["high"],
                    0.15,
                ),
                _check(
                    "each_locked_profile_has_high_load",
                    locked_profiles_pass,
                    locked_profiles_pass,
                    True,
                ),
                _check(
                    "all_maneuver_types_per_split",
                    maneuver_types_pass,
                    maneuver_types_pass,
                    True,
                ),
                _check(
                    "random_missing_tolerance",
                    missing_error <= 0.02 + 1e-12,
                    missing_error,
                    0.02,
                ),
            ]
        )
    return rows


def _generator_residual_check(
    *,
    latent: pd.DataFrame,
    family: str,
    metric: str,
    check_id: str,
    threshold: float,
) -> dict[str, object]:
    value = float(latent.loc[latent["generator_family"] == family, metric].max())
    return _check(check_id, value < threshold, value, threshold)


def _clock_mapping_check(validation: pd.DataFrame) -> dict[str, object]:
    columns = [
        "vehicle_clock_mapping_max_error_s",
        "physiology_clock_mapping_max_error_s",
    ]
    value = float(validation[columns].max().max())
    return _check("clock_mapping_invertible", value < 1e-9, value, 1e-9)


def _check(
    check_id: str,
    passed: bool,
    actual: object,
    expected: object,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _source_independence_scan() -> bool:
    parameters = set(inspect.signature(generate_sortie).parameters)
    if parameters & {"method", "methods", "method_name", "model_name", "checkpoint"}:
        return False
    source_root = Path("src/chronaris/simulation/aviation_dual_stream")
    content = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in source_root.glob("*.py")
    )
    prohibited_terms = ("contiformer", "candidate_id", "model_name", "method_name")
    return not any(term in content for term in prohibited_terms)


def _locked_profiles_have_high_load(latent: pd.DataFrame) -> bool:
    locked = latent.loc[latent["split_id"] == "locked_test"]
    if locked.empty:
        return False
    return bool((locked.groupby("profile_id")["workload_high_ratio"].max() > 0).all())


def _all_types_per_split(latent: pd.DataFrame) -> bool:
    for _split_id, group in latent.groupby("split_id"):
        observed = {
            int(value)
            for values in group["maneuver_type_ids"]
            for value in values
        }
        if observed != set(range(5)):
            return False
    return True


def _random_missing_max_error(validation: pd.DataFrame) -> float:
    random_only = validation.loc[
        (validation["vehicle_block_gap_s_config"] == 0)
        & (validation["physiology_block_gap_s_config"] == 0)
    ]
    errors = pd.concat(
        [
            (
                random_only["vehicle_missing_ratio"]
                - random_only["vehicle_random_missing_rate_config"]
            ).abs(),
            (
                random_only["physiology_missing_ratio"]
                - random_only["physiology_random_missing_rate_config"]
            ).abs(),
        ]
    )
    return float(errors.max()) if not errors.empty else 0.0
