from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from chronaris.evaluation.application_tasks.pretraining_smoke_downstream import (
    load_guarded_pretraining_smoke_targets,
    run_fixed_linear_smoke_consumers,
)
from chronaris.representation import FoldLineage, FusionStreamBatch


def _checkpoint(path: Path, method_name: str) -> Path:
    torch.save(
        {
            "format": "chronaris.common_pretraining_checkpoint.v1",
            "training_status": "completed",
            "method_name": method_name,
            "label_used_for_encoder_training": False,
        },
        path,
    )
    return path


def test_workload_truth_guard_requires_all_five_completed_methods(tmp_path) -> None:
    with pytest.raises(ValueError, match="five completed"):
        load_guarded_pretraining_smoke_targets(
            data_manifest_rows=(),
            fold=FoldLineage("f", ("train",), (), ("test",)),
            completed_pretraining_checkpoints=(),
        )


def test_guarded_targets_open_only_workload_after_checkpoint_completion(tmp_path) -> None:
    methods = (
        "physiology_only",
        "vehicle_only",
        "mult",
        "contiformer",
        "chronaris",
    )
    checkpoints = tuple(
        _checkpoint(tmp_path / f"{method}.pt", method) for method in methods
    )
    sample_ids = ("train_a", "train_b", "validation", "held_out")
    roles = ("train", "train", "validation", "held_out")
    manifest_rows = []
    for index, sample_id in enumerate(sample_ids):
        scenario = tmp_path / sample_id / "clean_asynchronous"
        scenario.mkdir(parents=True)
        observed = scenario / "raw_dual_stream.npz"
        np.savez(observed, placeholder=np.asarray([1]))
        times = np.arange(0.0, 40.0, 0.1)
        np.savez(
            scenario / "ground_truth.npz",
            true_time_s=times,
            workload=np.full_like(times, 0.1 + index),
            forbidden_oracle=np.asarray([999]),
        )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "role": roles[index],
                "observed_path": str(observed),
            }
        )
    fold = FoldLineage(
        "f",
        ("train_a", "train_b"),
        ("validation",),
        ("held_out",),
    )
    targets, manifest = load_guarded_pretraining_smoke_targets(
        data_manifest_rows=manifest_rows,
        fold=fold,
        completed_pretraining_checkpoints=checkpoints,
    )

    assert len(targets) == 4
    assert set(targets.columns) == {
        "sample_id",
        "role",
        "future_workload_mean",
        "workload_class",
    }
    assert manifest["oracle_opened_after_checkpoint_count"] == 5
    assert all(
        item["fields_opened"] == ["true_time_s", "workload"]
        for item in manifest["oracle_files"]
    )


def test_fixed_linear_consumers_share_protocol_and_emit_smoke_metrics() -> None:
    train_ids = tuple(f"train_{index}" for index in range(6))
    validation_ids = tuple(f"validation_{index}" for index in range(3))
    held_ids = tuple(f"held_{index}" for index in range(3))
    fold = FoldLineage("f", train_ids, validation_ids, held_ids)
    all_ids = (*train_ids, *validation_ids, *held_ids)
    targets = pd.DataFrame(
        {
            "sample_id": all_ids,
            "role": (
                *("train" for _ in train_ids),
                *("validation" for _ in validation_ids),
                *("held_out" for _ in held_ids),
            ),
            "workload_class": [index % 3 for index in range(len(all_ids))],
            "future_workload_mean": np.linspace(0.1, 1.2, len(all_ids)),
        }
    )
    outputs = {
        "chronaris": {
            "train": _fusion(train_ids, 0),
            "validation": _fusion(validation_ids, 10),
            "held_out": _fusion(held_ids, 20),
        }
    }
    metrics, predictions = run_fixed_linear_smoke_consumers(
        outputs=outputs,
        targets=targets,
        fold=fold,
    )

    assert len(metrics) == 12
    assert all(row["smoke_only"] for row in metrics)
    assert {row["role"] for row in metrics} == {"validation", "held_out"}
    assert len(predictions) == 6


def _fusion(sample_ids, offset):
    generator = torch.Generator().manual_seed(17 + offset)
    sequence = torch.randn(len(sample_ids), 96, 64, generator=generator)
    return FusionStreamBatch(
        sample_ids=tuple(sample_ids),
        timestamps_s=torch.arange(96, dtype=torch.float64)
        .mul(30.0 / 96)
        .repeat(len(sample_ids), 1),
        sequence_embedding=sequence,
        valid_mask=torch.ones(len(sample_ids), 96, dtype=torch.bool),
        pooled_embedding=sequence.mean(dim=1),
        method_name="chronaris",
        fold_id="f",
        checkpoint_sha256="a" * 64,
        source_sample_hashes=tuple(
            hashlib.sha256(value.encode()).hexdigest() for value in sample_ids
        ),
    )
