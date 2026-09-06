from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest
from dataclasses import replace

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
                "format": "chronaris.common_pretraining_checkpoint.v1",
                "training_status": "completed",
                "label_used_for_encoder_training": False,
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


def test_guided_development_guard_never_admits_confirmation_or_supervised_sources(tmp_path):
    samples = [_sample("train"), _sample("validation")]
    batch = collate_observation_samples(samples)
    path = tmp_path / "source.pt"
    payload = dict(format="chronaris.common_pretraining_checkpoint.v2", training_status="completed",
        method_name="chronaris", label_used_for_encoder_training=False, config={"max_updates": 10},
        fold={"train_sample_ids": ["train"], "validation_sample_ids": ["validation"]})
    rows = []
    for sample, role, level in zip(samples, ("train", "validation"), (.2, .8), strict=True):
        root = tmp_path / role
        root.mkdir()
        np.savez(root / "ground_truth.npz", true_time_s=np.arange(0., 40., .05),
                 workload=np.full(800, level), maneuver_state=np.zeros(800, dtype=np.int64))
        rows.append(dict(sample_id=sample.sample_id, role=role, context_start_s=0.,
                         context_end_s=30., observed_path=str(root / "raw_dual_stream.npz")))
    data = ApplicationConsumerSmokeData(batch=batch, schema=samples[0].schema,
        role_sample_ids={"train": ("train",), "validation": ("validation",), "held_out": ("sealed",)},
        sample_manifest_rows=tuple(rows))
    torch.save(payload, path)
    targets = build_guarded_application_consumer_targets(data, completed_pretraining_checkpoints=[path], task_guided_development=True)
    assert targets.manifest["oracle_opened_after_checkpoint_count"] == 1
    assert targets.manifest["workload_thresholds_train_only"] == pytest.approx([.2, .2])
    torch.save(payload | {"label_used_for_encoder_training": True}, path)
    with pytest.raises(ValueError, match="self-supervised initialization"):
        build_guarded_application_consumer_targets(data, completed_pretraining_checkpoints=[path], task_guided_development=True)
    torch.save(payload, path)
    sealed = replace(data, batch=collate_observation_samples(samples + [_sample("sealed")]))
    with pytest.raises(ValueError, match="confirmation targets"):
        build_guarded_application_consumer_targets(sealed, completed_pretraining_checkpoints=[path], task_guided_development=True)
