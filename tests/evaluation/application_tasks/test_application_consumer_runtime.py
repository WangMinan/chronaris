from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from chronaris.evaluation.application_tasks.application_consumer_runtime import (
    ApplicationConsumerProtocol,
    run_application_method_consumers,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeTargets,
)
from chronaris.evaluation.application_tasks.application_consumers import (
    MiniRocketConsumerConfig,
    TCNConsumerConfig,
)
from chronaris.representation import FusionStreamBatch


def _output(role: str, count: int, offset: int) -> FusionStreamBatch:
    generator = torch.Generator().manual_seed(17 + offset)
    sequence = torch.randn(count, 96, 64, generator=generator)
    sample_ids = tuple(f"{role}_{index}" for index in range(count))
    return FusionStreamBatch(
        sample_ids=sample_ids,
        timestamps_s=torch.arange(96).repeat(count, 1).to(torch.float64),
        sequence_embedding=sequence,
        valid_mask=torch.ones(count, 96, dtype=torch.bool),
        pooled_embedding=sequence.mean(dim=1),
        method_name="chronaris",
        fold_id="fold_a",
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        source_sample_hashes=tuple(
            hashlib.sha256(sample_id.encode()).hexdigest() for sample_id in sample_ids
        ),
    )


def test_method_consumer_runtime_resumes_and_rebuilds_prediction(tmp_path: Path) -> None:
    outputs = {
        "train": _output("train", 15, 0),
        "validation": _output("validation", 6, 1),
        "held_out": _output("held_out", 6, 2),
    }
    sample_ids = tuple(
        sample_id
        for role in ("train", "validation", "held_out")
        for sample_id in outputs[role].sample_ids
    )
    count = len(sample_ids)
    state = np.stack(
        [np.asarray(([0] * 19 + [1] * 19 + [2] * 19 + [3] * 19 + [4] * 20)) for _ in range(count)]
    )
    targets = ApplicationConsumerSmokeTargets(
        sample_ids=sample_ids,
        roles=tuple(sample_id.split("_", 1)[0] for sample_id in sample_ids),
        future_workload_mean=torch.linspace(0.0, 1.0, count),
        workload_class=torch.tensor([index % 3 for index in range(count)]),
        maneuver_state=torch.from_numpy(state),
        boundary_mask=torch.from_numpy(
            np.column_stack((np.zeros(count, dtype=bool), state[:, 1:] != state[:, :-1]))
        ),
        manifest={"format": "test.targets.v1", "thresholds": [0.33, 0.66]},
    )
    protocol = ApplicationConsumerProtocol(
        minirocket=MiniRocketConsumerConfig(n_kernels=84),
        tcn=TCNConsumerConfig(hidden_channels=8, dropout=0.0, epochs=1),
    )
    first = run_application_method_consumers(
        method_name="chronaris",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        fold_id="fold_a",
        protocol=protocol,
        resume=True,
    )
    prediction_path = tmp_path / "chronaris" / "predictions.npz"
    original = first.model_manifest["prediction_sha256"]
    prediction_path.unlink()
    second = run_application_method_consumers(
        method_name="chronaris",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        fold_id="fold_a",
        protocol=protocol,
        resume=True,
    )
    Path(second.model_manifest["model_files"]["minirocket"]["path"]).unlink()
    rocket_recovery = run_application_method_consumers(
        method_name="chronaris",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        fold_id="fold_a",
        protocol=protocol,
        resume=True,
    )
    Path(rocket_recovery.model_manifest["model_files"]["tcn"]["path"]).unlink()
    tcn_recovery = run_application_method_consumers(
        method_name="chronaris",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        fold_id="fold_a",
        protocol=protocol,
        resume=True,
    )

    assert first.status == "completed"
    assert second.status == "resumed"
    assert second.model_manifest["prediction_sha256"] == original
    assert rocket_recovery.component_status == {
        "linear": "resumed",
        "minirocket": "completed",
        "causal_tcn": "resumed",
    }
    assert tcn_recovery.component_status == {
        "linear": "resumed",
        "minirocket": "resumed",
        "causal_tcn": "completed",
    }
    assert (
        rocket_recovery.model_manifest["model_files"]["linear"]["sha256"]
        == second.model_manifest["model_files"]["linear"]["sha256"]
    )
    assert (
        rocket_recovery.model_manifest["model_files"]["tcn"]["sha256"]
        == second.model_manifest["model_files"]["tcn"]["sha256"]
    )
    assert (
        tcn_recovery.model_manifest["model_files"]["minirocket"]["sha256"]
        == rocket_recovery.model_manifest["model_files"]["minirocket"]["sha256"]
    )
    assert tcn_recovery.model_manifest["prediction_sha256"] == original
    assert len(first.metric_rows) == 64
    assert len(first.tcn_training_rows) == 1
    assert prediction_path.is_file()
