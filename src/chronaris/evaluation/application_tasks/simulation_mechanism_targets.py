"""Oracle-gated labels for representation-level clock and response-lag recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


CLOCK_OFFSET_TARGET = "relative_clock_offset_magnitude_s"
RESPONSE_LAG_TARGET = "primary_physiology_response_lag_s"


@dataclass(frozen=True, slots=True)
class SimulationMechanismTargets:
    rows_by_sample_id: Mapping[str, Mapping[str, object]]
    manifest: Mapping[str, object]

    def values(self, sample_ids: Sequence[str], target_name: str) -> np.ndarray:
        if target_name not in {CLOCK_OFFSET_TARGET, RESPONSE_LAG_TARGET}:
            raise ValueError("unknown simulation mechanism target")
        return np.asarray(
            [self.rows_by_sample_id[str(value)][target_name] for value in sample_ids],
            dtype=np.float64,
        )

    def trajectory_ids(self, sample_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            str(self.rows_by_sample_id[str(value)]["trajectory_id"])
            for value in sample_ids
        )


def build_simulation_mechanism_targets(
    manifest_rows,
    *,
    scenarios: Sequence[ObservationScenarioConfig],
    representation_evidence_completed: bool,
) -> SimulationMechanismTargets:
    if not representation_evidence_completed:
        raise ValueError("mechanism targets require completed representations")
    scenario_by_id = {value.scenario_id: value for value in scenarios}
    rows_by_sample = {}
    oracle_cache = {}
    oracle_rows = []
    for raw in manifest_rows:
        sample_id = str(raw["sample_id"])
        if sample_id in rows_by_sample:
            continue
        scenario_id = str(raw["scenario_id"])
        if scenario_id not in scenario_by_id:
            raise ValueError(f"mechanism target scenario is not locked: {scenario_id}")
        observed_path = Path(str(raw["observed_path"]))
        oracle_path = observed_path.with_name("ground_truth.npz")
        if oracle_path not in oracle_cache:
            with np.load(oracle_path, allow_pickle=False) as archive:
                if "realized_physiology_lag_s" not in archive.files:
                    raise ValueError("mechanism oracle lacks realized physiology lag")
                lag = float(np.asarray(archive["realized_physiology_lag_s"])[0])
            oracle_cache[oracle_path] = lag
            oracle_rows.append(
                {
                    "oracle_path": str(oracle_path),
                    "oracle_sha256": sha256_file(oracle_path),
                    "allowed_fields_opened": ["realized_physiology_lag_s"],
                }
            )
        scenario = scenario_by_id[scenario_id]
        rows_by_sample[sample_id] = {
            "sample_id": sample_id,
            "trajectory_id": str(raw["trajectory_id"]),
            "scenario_id": scenario_id,
            CLOCK_OFFSET_TARGET: abs(
                scenario.physiology_clock_offset_s
                - scenario.vehicle_clock_offset_s
            ),
            RESPONSE_LAG_TARGET: oracle_cache[oracle_path],
        }
    return SimulationMechanismTargets(
        rows_by_sample_id=rows_by_sample,
        manifest={
            "format": "chronaris.simulation_mechanism_targets.v1",
            "sample_count": len(rows_by_sample),
            "trajectory_scenario_count": len(oracle_cache),
            "target_names": [CLOCK_OFFSET_TARGET, RESPONSE_LAG_TARGET],
            "clock_offset_semantics": (
                "absolute physiology-minus-vehicle configured clock offset"
            ),
            "response_lag_semantics": "first realized physiology field lag",
            "allowed_oracle_fields": ["realized_physiology_lag_s"],
            "oracle_opened_after_representation": True,
            "oracle_files": oracle_rows,
        },
    )
