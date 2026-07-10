"""Oracle, distribution and paired-observation validators."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .contracts import MANEUVER_STATE_NAMES, MANEUVER_TYPE_NAMES, SimulatedDualStreamSortie


def validate_sortie(sortie: SimulatedDualStreamSortie) -> dict[str, object]:
    """Return compact validation facts without mutating generated data."""

    state_ids = set(int(value) for value in np.unique(sortie.latent.maneuver_state))
    type_ids = set(int(value) for value in np.unique(sortie.latent.maneuver_type) if value >= 0)
    vehicle_clock_error = _clock_mapping_error(
        sortie.vehicle.true_sample_time_s,
        sortie.vehicle.observed_time_s,
        sortie.vehicle_trace.retained_jitter_s,
        offset_s=sortie.vehicle_trace.clock_offset_s,
        drift_ppm=sortie.vehicle_trace.clock_drift_ppm,
    )
    physiology_clock_error = _clock_mapping_error(
        sortie.physiology.true_sample_time_s,
        sortie.physiology.observed_time_s,
        sortie.physiology_trace.retained_jitter_s,
        offset_s=sortie.physiology_trace.clock_offset_s,
        drift_ppm=sortie.physiology_trace.clock_drift_ppm,
    )
    physical_abs = np.abs(sortie.latent.physical_residual)
    workload = sortie.latent.workload
    result = {
        "sample_id": sortie.sample_id,
        "trajectory_id": sortie.latent.trajectory_id,
        "scenario_id": sortie.observation_config.scenario_id,
        "generator_family": sortie.latent.config.generator_family,
        "latent_hash": sortie.latent_hash,
        "event_count": len(sortie.latent.events),
        "all_states_present": state_ids == set(range(len(MANEUVER_STATE_NAMES))),
        "maneuver_type_count": len(type_ids),
        "maneuver_type_ids": sorted(type_ids),
        "vehicle_values_finite": bool(np.isfinite(sortie.vehicle.values).all()),
        "physiology_values_finite": bool(np.isfinite(sortie.physiology.values).all()),
        "workload_min": float(workload.min()),
        "workload_max": float(workload.max()),
        "workload_low_ratio": float(np.mean(workload < 0.35)),
        "workload_medium_ratio": float(np.mean((workload >= 0.35) & (workload < 0.70))),
        "workload_high_ratio": float(np.mean(workload >= 0.70)),
        "physical_residual_abs_median": float(np.median(physical_abs)),
        "physical_residual_abs_q95": float(np.quantile(physical_abs, 0.95)),
        "vehicle_missing_ratio": float(1.0 - np.mean(sortie.vehicle_trace.retained_mask)),
        "physiology_missing_ratio": float(1.0 - np.mean(sortie.physiology_trace.retained_mask)),
        "vehicle_random_missing_rate_config": sortie.observation_config.vehicle_random_missing_rate,
        "physiology_random_missing_rate_config": sortie.observation_config.physiology_random_missing_rate,
        "vehicle_block_gap_s_config": sortie.observation_config.vehicle_block_gap_s,
        "physiology_block_gap_s_config": sortie.observation_config.physiology_block_gap_s,
        "vehicle_clock_offset_s_config": sortie.observation_config.vehicle_clock_offset_s,
        "physiology_clock_offset_s_config": sortie.observation_config.physiology_clock_offset_s,
        "vehicle_clock_drift_ppm_config": sortie.observation_config.vehicle_clock_drift_ppm,
        "physiology_clock_drift_ppm_config": sortie.observation_config.physiology_clock_drift_ppm,
        "vehicle_clock_mapping_max_error_s": vehicle_clock_error,
        "physiology_clock_mapping_max_error_s": physiology_clock_error,
        "estimated_primary_response_lag_s": estimate_primary_response_lag(sortie),
        "true_primary_response_lag_s": float(sortie.realized_physiology_lag_s[0]),
    }
    for state_id, state_name in enumerate(MANEUVER_STATE_NAMES):
        result[f"state_ratio_{state_name}"] = float(
            np.mean(sortie.latent.maneuver_state == state_id)
        )
    return result


def validate_paired_observations(
    sorties: Sequence[SimulatedDualStreamSortie],
) -> dict[str, object]:
    if not sorties:
        raise ValueError("at least one paired sortie is required")
    trajectory_ids = {sortie.latent.trajectory_id for sortie in sorties}
    latent_hashes = {sortie.latent_hash for sortie in sorties}
    scenario_ids = {sortie.observation_config.scenario_id for sortie in sorties}
    return {
        "trajectory_id": sorties[0].latent.trajectory_id,
        "scenario_count": len(sorties),
        "scenario_ids_unique": len(scenario_ids) == len(sorties),
        "trajectory_id_shared": len(trajectory_ids) == 1,
        "latent_hash_shared": len(latent_hashes) == 1,
        "latent_hash": sorties[0].latent_hash,
    }


def split_identity_audit(
    split_profiles_and_seeds: Mapping[str, Sequence[tuple[str, int]]],
) -> dict[str, object]:
    rows = {
        split_id: {(str(profile_id), int(seed)) for profile_id, seed in values}
        for split_id, values in split_profiles_and_seeds.items()
    }
    intersections = {}
    split_ids = sorted(rows)
    for left_index, left in enumerate(split_ids):
        for right in split_ids[left_index + 1 :]:
            intersections[f"{left}__{right}"] = len(rows[left] & rows[right])
    return {
        "split_sizes": {key: len(value) for key, value in rows.items()},
        "pairwise_intersection_sizes": intersections,
        "disjoint": all(value == 0 for value in intersections.values()),
    }


def estimate_primary_response_lag(sortie: SimulatedDualStreamSortie) -> float:
    workload = sortie.latent.workload
    physiology = sortie.latent.physiology_state[:, 0]
    dt_s = sortie.latent.config.dt_s
    max_lag_steps = min(int(round(40.0 / dt_s)), len(workload) // 2)
    workload_z = _standardize(workload)
    physiology_z = _standardize(physiology)
    correlations = []
    for lag_steps in range(max_lag_steps + 1):
        if lag_steps == 0:
            left, right = workload_z, physiology_z
        else:
            left, right = workload_z[:-lag_steps], physiology_z[lag_steps:]
        correlations.append(float(np.mean(left * right)))
    return float(np.argmax(correlations) * dt_s)


def _clock_mapping_error(
    true_time_s: np.ndarray,
    observed_time_s: np.ndarray,
    jitter_s: np.ndarray,
    *,
    offset_s: float,
    drift_ppm: float,
) -> float:
    expected = true_time_s + offset_s + drift_ppm * 1e-6 * true_time_s + jitter_s
    return float(np.max(np.abs(observed_time_s - expected))) if len(expected) else 0.0


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    return (values - float(np.mean(values))) / max(std, 1e-9)
