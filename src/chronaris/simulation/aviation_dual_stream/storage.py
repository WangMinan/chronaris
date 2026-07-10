"""Heavy scenario bundle storage and oracle-safe model input loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import (
    CONTROL_FEATURE_NAMES,
    PHYSICAL_RESIDUAL_NAMES,
    PHYSIOLOGY_FEATURE_NAMES,
    VEHICLE_FEATURE_NAMES,
    SimulatedDualStreamSortie,
)
from .deterministic_npz import write_deterministic_npz


@dataclass(frozen=True, slots=True)
class StoredScenario:
    scenario_root: str
    scenario_manifest_path: str
    raw_dual_stream_path: str
    ground_truth_path: str
    task_manifest_path: str
    raw_sha256: str
    ground_truth_sha256: str


def store_scenario(
    sortie: SimulatedDualStreamSortie,
    *,
    output_root: str | Path,
    split_id: str,
) -> StoredScenario:
    """Persist one observed scenario without exposing oracle arrays to the model archive."""

    scenario_root = (
        Path(output_root)
        / split_id
        / sortie.latent.profile_id
        / sortie.latent.trajectory_id
        / sortie.observation_config.scenario_id
    )
    scenario_root.mkdir(parents=True, exist_ok=True)
    raw_path = scenario_root / "raw_dual_stream.npz"
    truth_path = scenario_root / "ground_truth.npz"
    raw_hash = write_deterministic_npz(
        raw_path,
        {
            "vehicle_observed_time_s": sortie.vehicle.observed_time_s,
            "vehicle_values": sortie.vehicle.values,
            "vehicle_feature_names": np.asarray(VEHICLE_FEATURE_NAMES),
            "physiology_observed_time_s": sortie.physiology.observed_time_s,
            "physiology_values": sortie.physiology.values,
            "physiology_feature_names": np.asarray(PHYSIOLOGY_FEATURE_NAMES),
        },
    )
    events = sortie.latent.events
    truth_hash = write_deterministic_npz(
        truth_path,
        {
            "true_time_s": sortie.latent.true_time_s,
            "maneuver_state": sortie.latent.maneuver_state,
            "maneuver_type": sortie.latent.maneuver_type,
            "controls": sortie.latent.controls,
            "control_feature_names": np.asarray(CONTROL_FEATURE_NAMES),
            "vehicle_latent_state": sortie.latent.vehicle_state,
            "workload": sortie.latent.workload,
            "physiology_latent_state": sortie.latent.physiology_state,
            "physiology_field_lag_s": sortie.latent.physiology_field_lag_s,
            "realized_physiology_lag_s": sortie.realized_physiology_lag_s,
            "physical_residual": sortie.latent.physical_residual,
            "physical_residual_names": np.asarray(PHYSICAL_RESIDUAL_NAMES),
            "vehicle_true_sample_time_s": sortie.vehicle.true_sample_time_s,
            "physiology_true_sample_time_s": sortie.physiology.true_sample_time_s,
            "vehicle_candidate_time_s": sortie.vehicle_trace.candidate_true_time_s,
            "physiology_candidate_time_s": sortie.physiology_trace.candidate_true_time_s,
            "vehicle_retained_mask": sortie.vehicle_trace.retained_mask,
            "physiology_retained_mask": sortie.physiology_trace.retained_mask,
            "vehicle_retained_jitter_s": sortie.vehicle_trace.retained_jitter_s,
            "physiology_retained_jitter_s": sortie.physiology_trace.retained_jitter_s,
            "event_id": np.asarray([event.event_id for event in events], dtype=np.int16),
            "event_maneuver_type": np.asarray(
                [event.maneuver_type for event in events], dtype=np.int8
            ),
            "event_boundaries_s": np.asarray(
                [
                    [
                        event.entry_start_s,
                        event.sustained_start_s,
                        event.exit_start_s,
                        event.recovery_start_s,
                        event.event_end_s,
                    ]
                    for event in events
                ],
                dtype=np.float64,
            ),
        },
    )
    manifest = {
        "sample_id": sortie.sample_id,
        "split_id": split_id,
        "trajectory_id": sortie.latent.trajectory_id,
        "profile_id": sortie.latent.profile_id,
        "generator_family": sortie.latent.config.generator_family,
        "generator_version": sortie.latent.config.generator_version,
        "latent_seed": sortie.latent.latent_seed,
        "observation_seed": sortie.observation_seed,
        "observation_config": sortie.observation_config.to_dict(),
        "latent_hash": sortie.latent_hash,
        "raw_dual_stream_sha256": raw_hash,
        "ground_truth_sha256": truth_hash,
        "raw_dual_stream_path": str(raw_path),
        "ground_truth_path": str(truth_path),
        "model_input_contains_oracle": False,
    }
    manifest_path = scenario_root / "scenario_manifest.json"
    _write_json(manifest_path, manifest)
    task_manifest_path = scenario_root / "task_manifest.jsonl"
    task_manifest_path.write_text(
        json.dumps(
            {
                "sample_id": sortie.sample_id,
                "trajectory_id": sortie.latent.trajectory_id,
                "scenario_id": sortie.observation_config.scenario_id,
                "high_workload_present": bool(np.any(sortie.latent.workload >= 0.70)),
                "complete_event_count": len(sortie.latent.events),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (scenario_root / "generation.log").write_text(
        f"generated sample_id={sortie.sample_id}\n"
        f"latent_hash={sortie.latent_hash}\n"
        f"raw_sha256={raw_hash}\n"
        f"ground_truth_sha256={truth_hash}\n",
        encoding="utf-8",
    )
    return StoredScenario(
        scenario_root=str(scenario_root),
        scenario_manifest_path=str(manifest_path),
        raw_dual_stream_path=str(raw_path),
        ground_truth_path=str(truth_path),
        task_manifest_path=str(task_manifest_path),
        raw_sha256=raw_hash,
        ground_truth_sha256=truth_hash,
    )


def load_model_inputs(raw_dual_stream_path: str | Path) -> Mapping[str, np.ndarray]:
    """Load only observed timestamps/features; oracle archives are a separate API."""

    with np.load(raw_dual_stream_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    forbidden = {
        "true_time_s",
        "maneuver_state",
        "maneuver_type",
        "workload",
        "physical_residual",
        "latent_hash",
    }
    overlap = forbidden & payload.keys()
    if overlap:
        raise ValueError(f"model input archive exposes oracle fields: {sorted(overlap)}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
