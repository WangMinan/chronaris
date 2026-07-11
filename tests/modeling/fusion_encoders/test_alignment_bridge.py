from __future__ import annotations

import hashlib

import numpy as np

from chronaris.modeling.fusion_encoders.alignment_bridge import (
    build_alignment_batch_from_observations,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
)


def test_alignment_bridge_coalesces_sparse_fields_at_same_timestamp() -> None:
    schema = ObservationSchema(
        schema_id="alignment_bridge_coalescing.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a", "physiology.b"),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed", "observed"),
        vehicle_feature_roles=("observed",),
    )
    sample = ObservedDualStreamSample(
        sample_id="sample",
        group_id="group",
        schema=schema,
        physiology_values=np.asarray(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float32
        ),
        physiology_timestamps_s=np.asarray([0.0, 0.0, 1.0]),
        physiology_feature_mask=np.asarray(
            [[True, False], [False, True], [True, False]], dtype=bool
        ),
        vehicle_values=np.asarray([[4.0], [5.0]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.0, 1.0]),
        vehicle_feature_mask=np.ones((2, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(b"sample").hexdigest(),
    )
    batch = collate_observation_samples([sample])

    aligned = build_alignment_batch_from_observations(
        batch,
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
    )

    assert aligned.physiology.values.shape == (1, 2, 2)
    assert aligned.physiology.point_counts.tolist() == [2]
    assert aligned.physiology.values[0, 0].tolist() == [1.0, 2.0]
    assert aligned.physiology.feature_valid_mask[0, 0].tolist() == [True, True]
    assert aligned.physiology.offsets_s[0].tolist() == [0.0, 1.0]
    assert aligned.physiology.delta_t_s[0].tolist() == [0.0, 1.0]


def test_alignment_bridge_averages_duplicate_feature_values_at_one_timestamp() -> None:
    values = np.asarray([[1.0], [3.0]], dtype=np.float32)
    schema = ObservationSchema(
        schema_id="alignment_bridge_duplicate.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    sample = ObservedDualStreamSample(
        sample_id="duplicate",
        group_id="group",
        schema=schema,
        physiology_values=values,
        physiology_timestamps_s=np.asarray([0.0, 0.0]),
        physiology_feature_mask=np.ones((2, 1), dtype=bool),
        vehicle_values=np.asarray([[1.0]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.0]),
        vehicle_feature_mask=np.ones((1, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(b"duplicate").hexdigest(),
    )

    aligned = build_alignment_batch_from_observations(
        collate_observation_samples([sample]),
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
    )

    assert aligned.physiology.point_counts.tolist() == [1]
    assert aligned.physiology.values[0, 0, 0].item() == 2.0
