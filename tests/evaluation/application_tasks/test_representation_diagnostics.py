from __future__ import annotations

import hashlib

import torch

from chronaris.evaluation.representation_diagnostics import (
    apply_stream_counterfactual,
    compare_representations,
    fit_feature_recovery_probe,
    fit_fidelity_probe,
    gradient_conflict_rows,
    representation_health,
)
from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    FusionStreamBatch,
)


def _fusion(method: str, sample_ids: tuple[str, ...], offset: float = 0.0):
    base = torch.arange(len(sample_ids) * 96 * 64, dtype=torch.float32).reshape(
        len(sample_ids), 96, 64
    )
    sample_offsets = torch.arange(len(sample_ids), dtype=torch.float32).reshape(-1, 1, 1)
    sequence = torch.sin(base / 97.0) + sample_offsets + offset
    valid = torch.ones(len(sample_ids), 96, dtype=torch.bool)
    return FusionStreamBatch(
        sample_ids=sample_ids,
        timestamps_s=torch.arange(96, dtype=torch.float64).repeat(len(sample_ids), 1),
        sequence_embedding=sequence,
        valid_mask=valid,
        pooled_embedding=sequence.mean(dim=1),
        method_name=method,
        fold_id="fold",
        checkpoint_sha256="a" * 64,
        source_sample_hashes=tuple(
            hashlib.sha256(sample_id.encode()).hexdigest() for sample_id in sample_ids
        ),
    )


def _observation_batch():
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)
    timestamps = torch.arange(4, dtype=torch.float64).repeat(3, 1)
    point_mask = torch.ones(3, 4, dtype=torch.bool)
    feature_mask = torch.ones(3, 4, 1, dtype=torch.bool)
    return DualStreamObservationBatch(
        sample_ids=("a", "b", "c"),
        group_ids=("a", "b", "c"),
        physiology_values=values,
        physiology_timestamps_s=timestamps,
        physiology_point_mask=point_mask,
        physiology_feature_mask=feature_mask,
        physiology_observation_age_s=torch.zeros_like(values),
        vehicle_values=values + 10,
        vehicle_timestamps_s=timestamps,
        vehicle_point_mask=point_mask,
        vehicle_feature_mask=feature_mask,
        vehicle_observation_age_s=torch.zeros_like(values),
        query_timestamps_s=torch.linspace(0, 3, 96, dtype=torch.float64).repeat(3, 1),
        source_sample_hashes=tuple(
            hashlib.sha256(value.encode()).hexdigest() for value in ("a", "b", "c")
        ),
    )


def test_representation_health_and_fidelity_are_finite() -> None:
    train_ids = tuple(f"train_{index}" for index in range(10))
    validation_ids = tuple(f"validation_{index}" for index in range(5))
    train_source = _fusion("chronaris", train_ids)
    train_target = _fusion("vehicle_only", train_ids, offset=0.25)
    validation_source = _fusion("chronaris", validation_ids, offset=0.5)
    validation_target = _fusion("vehicle_only", validation_ids, offset=0.75)

    health = representation_health(validation_source)
    fidelity = fit_fidelity_probe(
        train_source,
        train_target,
        validation_source,
        validation_target,
    )

    assert 0 < health.effective_rank <= 64
    assert health.valid_vector_count == len(validation_ids) * 96
    assert fidelity.variance_weighted_r2 > 0.99
    assert fidelity.normalized_rmse < 0.01


def test_counterfactuals_change_only_requested_stream() -> None:
    batch = _observation_batch()
    zeroed = apply_stream_counterfactual(
        batch,
        stream_name="vehicle",
        operation="zero",
    )
    shifted = apply_stream_counterfactual(
        batch,
        stream_name="physiology",
        operation="shift",
    )
    shuffled = apply_stream_counterfactual(
        batch,
        stream_name="physiology",
        operation="shuffle",
    )

    assert not zeroed.vehicle_point_mask.any()
    assert torch.equal(zeroed.physiology_values, batch.physiology_values)
    assert torch.equal(shifted.vehicle_timestamps_s, batch.vehicle_timestamps_s)
    assert torch.equal(
        shuffled.physiology_point_mask,
        shuffled.physiology_feature_mask.any(dim=-1),
    )
    assert torch.allclose(
        shifted.physiology_timestamps_s,
        batch.physiology_timestamps_s + 3.0,
    )
    comparison = compare_representations(
        torch.zeros(2, 96, 64),
        torch.ones(2, 96, 64),
        stream_name="vehicle",
        operation="zero",
    )
    assert comparison.mean_absolute_change == 1.0


def test_feature_recovery_probe_uses_task_independent_tensor_target() -> None:
    train_ids = tuple(f"train_{index}" for index in range(10))
    validation_ids = tuple(f"validation_{index}" for index in range(5))
    train = _fusion("chronaris", train_ids)
    validation = _fusion("chronaris", validation_ids)
    train_target = train.pooled_embedding[:, :3] * 2.0 + 1.0
    validation_target = validation.pooled_embedding[:, :3] * 2.0 + 1.0

    probe = fit_feature_recovery_probe(
        train,
        train_target,
        validation,
        validation_target,
        target_name="observed_semantic_groups",
    )

    assert probe.target_dimension == 3
    assert probe.variance_weighted_r2 > 0.99
    assert probe.normalized_rmse < 0.01


def test_gradient_conflict_rows_detect_opposing_losses() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    rows = gradient_conflict_rows(
        {
            "positive": parameter.sum(),
            "negative": -parameter.sum(),
        },
        (parameter,),
    )
    cosine = next(row for row in rows if row["row_type"] == "gradient_cosine")
    assert cosine["value"] < -0.999
    assert cosine["conflict"] is True
