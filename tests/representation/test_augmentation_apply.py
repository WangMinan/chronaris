from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.fusion_encoders import causal_query_stream
from chronaris.representation import (
    AugmentationPolicy,
    AugmentationRealization,
    ObservationSchema,
    ObservedDualStreamSample,
    apply_augmentation_realizations,
    augmentation_executor_accepts_method_name,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    collate_observation_samples,
)


def _sample(sample_id: str):
    schema = ObservationSchema(
        schema_id="augmentation_apply_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a", "physiology.b"),
        vehicle_feature_names=("vehicle.a", "vehicle.b"),
        physiology_feature_roles=("observed", "observed"),
        vehicle_feature_roles=("observed", "observed"),
    )
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id=sample_id,
        schema=schema,
        physiology_values=values,
        physiology_timestamps_s=np.asarray([0.2, 2.0, 5.0, 10.0, 20.0, 29.8]),
        physiology_feature_mask=np.ones_like(values, dtype=bool),
        vehicle_values=values + 100.0,
        vehicle_timestamps_s=np.asarray([0.1, 1.0, 6.0, 11.0, 21.0, 29.9]),
        vehicle_feature_mask=np.ones_like(values, dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def _plan(sample_id: str, *, dropped_modality=None):
    return AugmentationRealization(
        augmentation_id=hashlib.sha256((sample_id + "augmentation").encode()).hexdigest(),
        sample_id=sample_id,
        epoch=0,
        global_seed=17,
        physiology_block_start_s=4.0,
        physiology_block_duration_s=3.0,
        vehicle_block_start_s=9.0,
        vehicle_block_duration_s=3.0,
        dropped_modality=dropped_modality,
        physiology_clock_offset_s=0.2,
        vehicle_clock_offset_s=-0.2,
        physiology_jitter_seed=1,
        vehicle_jitter_seed=2,
        physiology_point_dropout_seed=3,
        vehicle_point_dropout_seed=4,
    )


def test_augmentation_executor_is_method_free_deterministic_and_ordered() -> None:
    batch = collate_observation_samples([_sample("a"), _sample("b")])
    plans = (_plan("a"), _plan("b"))
    policy = AugmentationPolicy(point_dropout_probability=0.25)
    first = apply_augmentation_realizations(batch, plans, policy=policy)
    second = apply_augmentation_realizations(batch, plans, policy=policy)

    assert augmentation_executor_accepts_method_name() is False
    assert torch.equal(first.batch.physiology_values, second.batch.physiology_values)
    assert torch.equal(
        first.physiology_provenance.original_point_indices,
        second.physiology_provenance.original_point_indices,
    )
    for timestamps, mask in (
        (first.batch.physiology_timestamps_s, first.batch.physiology_point_mask),
        (first.batch.vehicle_timestamps_s, first.batch.vehicle_point_mask),
    ):
        for row in range(len(batch.sample_ids)):
            observed = timestamps[row][mask[row]]
            assert bool((observed[1:] >= observed[:-1]).all())
            assert bool(((observed >= 0) & (observed < 30)).all())
    assert all(row.removed_point_count > 0 for row in first.audit_rows)


def test_modality_dropout_never_removes_the_other_stream() -> None:
    batch = collate_observation_samples([_sample("a")])
    result = apply_augmentation_realizations(
        batch,
        (_plan("a", dropped_modality="physiology"),),
        policy=AugmentationPolicy(point_dropout_probability=0.0),
    )
    assert not result.batch.physiology_point_mask.any()
    assert result.batch.vehicle_point_mask.any()
    physiology_audit = next(
        row for row in result.audit_rows if row.stream_name == "physiology"
    )
    vehicle_audit = next(
        row for row in result.audit_rows if row.stream_name == "vehicle"
    )
    assert physiology_audit.modality_dropped is True
    assert vehicle_audit.modality_dropped is False


def test_query_sources_map_back_to_original_observation_rows() -> None:
    batch = collate_observation_samples([_sample("a")])
    result = apply_augmentation_realizations(
        batch,
        (_plan("a"),),
        policy=AugmentationPolicy(point_dropout_probability=0.0),
    )
    queried = causal_query_stream(result.batch, stream_name="physiology")
    augmented_indices = queried.source_indices.clamp_min(0)
    provenance = result.physiology_provenance.original_point_indices
    expanded = provenance.unsqueeze(1).expand(-1, queried.values.shape[1], -1)
    original_indices = torch.gather(expanded, 2, augmented_indices)
    original_indices = torch.where(
        queried.feature_mask,
        original_indices,
        torch.full_like(original_indices, -1),
    )

    assert torch.equal(original_indices >= 0, queried.feature_mask)
    assert set(original_indices[original_indices >= 0].tolist()).issubset(
        set(provenance[provenance >= 0].tolist())
    )


def test_pretext_targets_use_exact_removed_query_sources() -> None:
    batch = collate_observation_samples([_sample("a")])
    augmented = apply_augmentation_realizations(
        batch,
        (_plan("a"),),
        policy=AugmentationPolicy(point_dropout_probability=0.0),
    )
    targets = build_common_pretext_targets(batch, augmented)

    assert targets.reconstruction_mask.any()
    assert targets.reconstruction_mask.shape == targets.reconstruction_target.shape
    assert targets.next_query_mask[:, -1].sum() == 0
    assert targets.next_query_mask[:, :-1].any()
    repeated = build_common_pretext_targets(batch, augmented)
    assert torch.equal(targets.reconstruction_mask, repeated.reconstruction_mask)
    assert torch.equal(targets.reconstruction_target, repeated.reconstruction_target)


def test_lag_discrimination_shift_is_id_derived_and_keeps_stream_contract() -> None:
    batch = collate_observation_samples([_sample("a"), _sample("b")])
    ids = (_plan("a").augmentation_id, _plan("b").augmentation_id)
    lagged = build_lag_discrimination_inputs(batch, ids)

    assert set(lagged.shifts_s.tolist()).issubset({-10.0, -5.0, 5.0, 10.0})
    assert torch.equal(
        lagged.negative_batch.physiology_values,
        batch.physiology_values,
    )
    assert not torch.equal(
        lagged.negative_batch.vehicle_timestamps_s,
        batch.vehicle_timestamps_s,
    )
    for row in range(len(batch.sample_ids)):
        mask = lagged.negative_batch.vehicle_point_mask[row]
        times = lagged.negative_batch.vehicle_timestamps_s[row][mask]
        assert bool((times[1:] >= times[:-1]).all())
        assert bool(((times >= 0) & (times < 30)).all())
