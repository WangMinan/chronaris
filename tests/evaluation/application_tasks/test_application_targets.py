from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeData,
    build_guarded_application_consumer_targets,
)
from chronaris.representation import collate_observation_samples
from tests.representation.test_contracts import _sample


def test_stress_targets_reuse_frozen_clean_thresholds_without_train_role(
    tmp_path: Path,
) -> None:
    sample = _sample("stress_sample")
    batch = collate_observation_samples([sample])
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    observed_path = scenario_root / "raw_dual_stream.npz"
    observed_path.touch()
    np.savez(
        scenario_root / "ground_truth.npz",
        true_time_s=np.arange(0.0, 40.0, 0.05),
        workload=np.full(800, 0.5, dtype=np.float32),
        maneuver_state=np.zeros(800, dtype=np.int64),
    )
    data = ApplicationConsumerSmokeData(
        batch=batch,
        schema=sample.schema,
        role_sample_ids={"train": (), "validation": (), "held_out": batch.sample_ids},
        sample_manifest_rows=(
            {
                "sample_id": sample.sample_id,
                "role": "held_out",
                "context_start_s": 0.0,
                "context_end_s": 30.0,
                "observed_path": str(observed_path),
            },
        ),
    )
    checkpoints = []
    for method in (
        "physiology_only",
        "vehicle_only",
        "mult",
        "contiformer",
        "chronaris",
    ):
        path = tmp_path / f"{method}.pt"
        torch.save(
            {
                "format": (
                    "chronaris.common_pretraining_checkpoint.v2"
                    if method == "chronaris"
                    else "chronaris.common_pretraining_checkpoint.v1"
                ),
                "training_status": "completed",
                "label_used_for_encoder_training": False,
                "simulation_oracle_opened": False,
                "locked_test_opened": False,
                "method_name": method,
            },
            path,
        )
        checkpoints.append(path)

    targets = build_guarded_application_consumer_targets(
        data,
        completed_pretraining_checkpoints=checkpoints,
        smoke_only=False,
        workload_thresholds=(0.2, 0.8),
    )

    assert targets.workload_class.tolist() == [1]
    assert targets.manifest["workload_thresholds_train_only"] == [0.2, 0.8]
    assert targets.manifest["workload_threshold_source"] == (
        "frozen_clean_train_thresholds"
    )
    assert targets.manifest["smoke_only"] is False
