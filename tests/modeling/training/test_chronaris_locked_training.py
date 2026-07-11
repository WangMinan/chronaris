from __future__ import annotations

import hashlib

import numpy as np

from chronaris.modeling.training import (
    LockedChronarisTrainingConfig,
    load_common_pretraining_checkpoint,
    train_locked_chronaris,
)
from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)


def _sample(sample_id: str, offset: float) -> ObservedDualStreamSample:
    schema = ObservationSchema(
        schema_id="chronaris_locked_training_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.spo2",),
        vehicle_feature_names=("vehicle.speed",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id=sample_id,
        schema=schema,
        physiology_values=np.asarray(
            [[95 + offset], [96 + offset], [97 + offset], [98 + offset]],
            dtype=np.float32,
        ),
        physiology_timestamps_s=np.asarray([0.0, 5.0, 10.0, 20.0]),
        physiology_feature_mask=np.ones((4, 1), dtype=bool),
        vehicle_values=np.asarray(
            [[50 + offset], [51 + offset], [52 + offset], [53 + offset]],
            dtype=np.float32,
        ),
        vehicle_timestamps_s=np.asarray([0.0, 4.0, 12.0, 20.0]),
        vehicle_feature_mask=np.ones((4, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def test_locked_chronaris_uses_auxiliary_backward_but_public_early_stop(tmp_path) -> None:
    batch = collate_observation_samples(
        (
            _sample("train_a", 0.0),
            _sample("train_b", 1.0),
            _sample("validation", 2.0),
            _sample("held_out", 3.0),
        )
    )
    fold = FoldLineage(
        fold_id="locked_chronaris_fold",
        train_sample_ids=("train_a", "train_b"),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    config = LockedChronarisTrainingConfig(
        max_epochs=1,
        batch_size=2,
        patience=1,
        seed=17,
    )
    first = train_locked_chronaris(
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.spo2",),
        vehicle_feature_names=("vehicle.speed",),
        vehicle_field_labels=(("vehicle.speed", "true airspeed"),),
        normalizer=normalizer,
        output_root=tmp_path,
        config=config,
    )
    resumed = train_locked_chronaris(
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.spo2",),
        vehicle_feature_names=("vehicle.speed",),
        vehicle_field_labels=(("vehicle.speed", "true airspeed"),),
        normalizer=normalizer,
        output_root=tmp_path,
        config=config,
    )
    _encoder, _heads, _normalizer, payload = load_common_pretraining_checkpoint(
        first.best_checkpoint_path
    )

    assert first.status == "completed"
    assert resumed.status == "resumed"
    assert first.completed_epochs == 1
    assert len(first.auxiliary_rows) == 3
    assert {row["term_name"] for row in first.auxiliary_rows} == {
        "chronaris_continuous_alignment",
        "chronaris_physical_consistency",
        "chronaris_causal_direction",
    }
    assert all(row["weight"] == 0 for row in first.auxiliary_rows)
    assert payload["chronaris_auxiliary_enabled"] is True
    assert payload["early_stopping_uses_public_pretext_only"] is True
    assert payload["selection_uses_public_pretext_only"] is True
    assert payload["label_used_for_encoder_training"] is False
    assert payload["simulation_oracle_opened"] is False
