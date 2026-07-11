from __future__ import annotations

import numpy as np
import pytest

from chronaris.evaluation.application_tasks.simulation_mechanism_consumer_run import (
    _evaluate_predictions,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_targets import (
    CLOCK_OFFSET_TARGET,
    RESPONSE_LAG_TARGET,
    build_simulation_mechanism_targets,
)
from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig


def test_mechanism_targets_are_representation_gated_and_use_locked_oracle_field(
    tmp_path,
) -> None:
    scenario_root = tmp_path / "trajectory" / "offset"
    scenario_root.mkdir(parents=True)
    observed = scenario_root / "raw_dual_stream.npz"
    np.savez(observed, placeholder=np.asarray([1]))
    np.savez(
        scenario_root / "ground_truth.npz",
        realized_physiology_lag_s=np.asarray([7.5, 8.0]),
    )
    manifest = [
        {
            "sample_id": "sample_a",
            "trajectory_id": "trajectory_a",
            "scenario_id": "offset",
            "observed_path": str(observed),
        }
    ]
    scenario = ObservationScenarioConfig(
        scenario_id="offset",
        vehicle_clock_offset_s=-0.5,
        physiology_clock_offset_s=1.0,
    )

    with pytest.raises(ValueError, match="completed representations"):
        build_simulation_mechanism_targets(
            manifest,
            scenarios=(scenario,),
            representation_evidence_completed=False,
        )
    targets = build_simulation_mechanism_targets(
        manifest,
        scenarios=(scenario,),
        representation_evidence_completed=True,
    )

    assert targets.values(("sample_a",), CLOCK_OFFSET_TARGET).tolist() == [1.5]
    assert targets.values(("sample_a",), RESPONSE_LAG_TARGET).tolist() == [7.5]
    assert targets.manifest["allowed_oracle_fields"] == [
        "realized_physiology_lag_s"
    ]


def test_mechanism_evaluation_uses_trajectory_as_metric_unit() -> None:
    rows = {
        "sample_a0": {
            "trajectory_id": "trajectory_a",
            CLOCK_OFFSET_TARGET: 1.0,
            RESPONSE_LAG_TARGET: 5.0,
        },
        "sample_a1": {
            "trajectory_id": "trajectory_a",
            CLOCK_OFFSET_TARGET: 1.0,
            RESPONSE_LAG_TARGET: 5.0,
        },
        "sample_b0": {
            "trajectory_id": "trajectory_b",
            CLOCK_OFFSET_TARGET: 2.0,
            RESPONSE_LAG_TARGET: 8.0,
        },
        "sample_b1": {
            "trajectory_id": "trajectory_b",
            CLOCK_OFFSET_TARGET: 2.0,
            RESPONSE_LAG_TARGET: 8.0,
        },
    }
    from chronaris.evaluation.application_tasks.simulation_mechanism_targets import (
        SimulationMechanismTargets,
    )

    targets = SimulationMechanismTargets(rows_by_sample_id=rows, manifest={})
    metrics, units = _evaluate_predictions(
        seed=17,
        method="chronaris",
        scenario_id="offset",
        target_name=CLOCK_OFFSET_TARGET,
        sample_ids=tuple(rows),
        truth=np.asarray([1.0, 1.0, 2.0, 2.0]),
        prediction=np.asarray([1.0, 1.2, 1.8, 2.0]),
        targets=targets,
    )

    assert {row["trajectory_count"] for row in metrics} == {2}
    assert len(units) == 4
    assert next(row for row in metrics if row["metric"] == "mae_s")["value"] == pytest.approx(0.1)
