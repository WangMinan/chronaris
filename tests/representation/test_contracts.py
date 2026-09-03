from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from chronaris.representation import (
    FUSION_OUTPUT_DIM,
    QUERY_POINT_COUNT,
    FusionStreamBatch,
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
    select_observation_batch,
    validate_fusion_method_alignment,
)
from chronaris.representation.contracts import RepresentationContractError


def _schema() -> ObservationSchema:
    return ObservationSchema(
        schema_id="test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a", "physiology.b"),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed", "observed"),
        vehicle_feature_roles=("observed",),
    )


def _sample(sample_id: str, *, shift: float = 0.0) -> ObservedDualStreamSample:
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id="group_a",
        schema=_schema(),
        physiology_values=np.asarray(
            [[1.0 + shift, 0.0], [0.0, 2.0 + shift]],
            dtype=np.float32,
        ),
        physiology_timestamps_s=np.asarray([0.0, 1.0], dtype=np.float64),
        physiology_feature_mask=np.asarray(
            [[True, False], [False, True]],
            dtype=bool,
        ),
        vehicle_values=np.asarray([[3.0 + shift], [4.0 + shift]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.5, 1.5], dtype=np.float64),
        vehicle_feature_mask=np.ones((2, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def _fusion(batch, method: str) -> FusionStreamBatch:
    sequence = torch.arange(
        len(batch.sample_ids) * QUERY_POINT_COUNT * FUSION_OUTPUT_DIM,
        dtype=torch.float32,
    ).reshape(len(batch.sample_ids), QUERY_POINT_COUNT, FUSION_OUTPUT_DIM)
    valid = torch.ones((len(batch.sample_ids), QUERY_POINT_COUNT), dtype=torch.bool)
    return FusionStreamBatch(
        sample_ids=batch.sample_ids,
        timestamps_s=batch.query_timestamps_s,
        sequence_embedding=sequence,
        valid_mask=valid,
        pooled_embedding=sequence.mean(dim=1),
        method_name=method,
        fold_id="fold_a",
        checkpoint_sha256=hashlib.sha256(method.encode()).hexdigest(),
        source_sample_hashes=batch.source_sample_hashes,
    )


def test_collator_preserves_raw_time_mask_age_and_fixed_query_contract():
    batch = collate_observation_samples([_sample("a"), _sample("b", shift=1.0)])

    assert batch.physiology_values.shape == (2, 2, 2)
    assert batch.vehicle_values.shape == (2, 2, 1)
    assert batch.query_timestamps_s.shape == (2, QUERY_POINT_COUNT)
    assert batch.query_timestamps_s[0, -1].item() < 30.0
    assert batch.physiology_observation_age_s[0, 0, 0].item() == 0.0
    assert torch.isinf(batch.physiology_observation_age_s[0, 0, 1])
    assert batch.physiology_observation_age_s[0, 1, 0].item() == 1.0

    selected = select_observation_batch(batch, ("b",))
    assert selected.sample_ids == ("b",)
    assert selected.physiology_values.shape == (1, 2, 2)


def test_collator_rejects_schema_mismatch():
    sample = _sample("a")
    bad_schema = replace(sample.schema, schema_id="different")

    with pytest.raises(RepresentationContractError, match="share one schema"):
        collate_observation_samples([sample, replace(sample, sample_id="b", schema=bad_schema)])


def test_fusion_contract_requires_valid_mask_mean_pooling():
    batch = collate_observation_samples([_sample("a")])
    output = _fusion(batch, "chronaris")

    with pytest.raises(RepresentationContractError, match="valid-mask mean"):
        replace(output, pooled_embedding=output.pooled_embedding + 1.0)


def test_method_alignment_rejects_query_or_sample_drift():
    batch = collate_observation_samples([_sample("a"), _sample("b")])
    first = _fusion(batch, "chronaris")
    second = _fusion(batch, "mult")
    assert len(validate_fusion_method_alignment([first, second])) == 64

    shifted = replace(second, timestamps_s=second.timestamps_s + 0.01)
    with pytest.raises(RepresentationContractError, match="timestamps mismatch"):
        validate_fusion_method_alignment([first, shifted])

    valid = second.valid_mask.clone()
    valid[:, -1] = False
    sequence = second.sequence_embedding.clone()
    sequence[:, -1] = 0
    masked = replace(
        second,
        sequence_embedding=sequence,
        valid_mask=valid,
        pooled_embedding=sequence[:, :-1].mean(dim=1),
    )
    with pytest.raises(RepresentationContractError, match="valid mask mismatch"):
        validate_fusion_method_alignment([first, masked])
    assert len(
        validate_fusion_method_alignment(
            [first, masked],
            require_valid_mask_match=False,
        )
    ) == 64


def test_fusion_contract_rejects_non_64_dimensional_output():
    batch = collate_observation_samples([_sample("a")])
    sequence = torch.zeros((1, QUERY_POINT_COUNT, 32), dtype=torch.float32)

    with pytest.raises(RepresentationContractError, match="shape"):
        FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s,
            sequence_embedding=sequence,
            valid_mask=torch.ones((1, QUERY_POINT_COUNT), dtype=torch.bool),
            pooled_embedding=torch.zeros((1, 32), dtype=torch.float32),
            method_name="bad",
            fold_id="fold_a",
            checkpoint_sha256="0" * 64,
            source_sample_hashes=batch.source_sample_hashes,
        )
