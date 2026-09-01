from __future__ import annotations

import hashlib

import numpy as np
import pytest

from chronaris.modeling.training import (
    CommonPretrainingConfig,
    EncoderCandidateConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
    select_observation_batch,
)
from chronaris.representation.contracts import RepresentationContractError
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _sample(sample_id: str, offset: float):
    schema = ObservationSchema(
        schema_id="common_pretraining_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id=sample_id,
        schema=schema,
        physiology_values=np.asarray(
            [[1.0 + offset], [2.0 + offset], [3.0 + offset], [4.0 + offset]],
            dtype=np.float32,
        ),
        physiology_timestamps_s=np.asarray([0.0, 5.0, 10.0, 20.0]),
        physiology_feature_mask=np.ones((4, 1), dtype=bool),
        vehicle_values=np.asarray(
            [[5.0 + offset], [6.0 + offset], [7.0 + offset], [8.0 + offset]],
            dtype=np.float32,
        ),
        vehicle_timestamps_s=np.asarray([0.0, 4.0, 12.0, 20.0]),
        vehicle_feature_mask=np.ones((4, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def test_common_pretraining_checkpoint_resume_and_export(tmp_path) -> None:
    batch = collate_observation_samples(
        [_sample("train", 0), _sample("validation", 1), _sample("held_out", 2)]
    )
    fold = FoldLineage(
        fold_id="fold_a",
        train_sample_ids=("train",),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.held_out_sample_ids,
    )
    config = CommonPretrainingConfig(epochs=1, batch_size=1, seed=17)
    first = train_common_pretext_method(
        "physiology_only",
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=config,
    )
    resumed = train_common_pretext_method(
        "physiology_only",
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=config,
    )
    encoder, _heads, loaded_normalizer, payload = load_common_pretraining_checkpoint(
        first.best_checkpoint_path
    )
    class CountingNormalizer:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.call_count = 0

        def transform(self, values):
            self.call_count += 1
            return self.wrapped.transform(values)

    counting_normalizer = CountingNormalizer(loaded_normalizer)
    adapter = TrainedFusionAdapter(
        encoder=encoder,
        normalizer=counting_normalizer,
        fold_id=fold.fold_id,
        checkpoint_sha256=sha256_file(first.best_checkpoint_path),
    )
    output = adapter(select_observation_batch(batch, fold.held_out_sample_ids))

    assert first.status == "completed"
    assert resumed.status == "resumed"
    assert first.step_count == 1
    assert all(row["status"] == "active" for row in first.training_rows)
    assert len(first.augmentation_rows) == 2
    assert payload["simulation_oracle_opened"] is False
    assert payload["format"] == "chronaris.common_pretraining_checkpoint.v2"
    assert len(payload["canonical_training_state_sha256"]) == 64
    assert counting_normalizer.call_count == 1
    assert output.sequence_embedding.shape == (1, 96, 64)


def test_common_pretraining_rejects_changed_resume_protocol(tmp_path) -> None:
    batch = collate_observation_samples(
        [_sample("train", 0), _sample("validation", 1), _sample("held_out", 2)]
    )
    fold = FoldLineage(
        fold_id="fold_a",
        train_sample_ids=("train",),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    common = dict(
        method_name="physiology_only",
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
    )
    train_common_pretext_method(
        **common,
        config=CommonPretrainingConfig(learning_rate=3e-4),
    )
    with pytest.raises(RepresentationContractError, match="protocol changed"):
        train_common_pretext_method(
            **common,
            config=CommonPretrainingConfig(learning_rate=1e-4),
        )


def test_common_pretraining_accepts_lazy_batch_provider(tmp_path) -> None:
    samples = {
        "train_a": _sample("train_a", 0),
        "train_b": _sample("train_b", 1),
        "validation": _sample("validation", 2),
        "held_out": _sample("held_out", 3),
    }
    provider = lambda sample_ids: collate_observation_samples(
        tuple(samples[sample_id] for sample_id in sample_ids)
    )
    fold = FoldLineage(
        fold_id="fold_lazy",
        train_sample_ids=("train_a", "train_b"),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
        batch_size=1,
    )

    result = train_common_pretext_method(
        "physiology_only",
        batch=None,
        batch_provider=provider,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=CommonPretrainingConfig(epochs=1, batch_size=1),
    )

    assert result.status == "completed"
    assert result.step_count == 2


def test_common_pretraining_round_trips_hidden_32_candidate(tmp_path) -> None:
    batch = collate_observation_samples(
        [_sample("train", 0), _sample("validation", 1), _sample("held_out", 2)]
    )
    fold = FoldLineage(
        fold_id="fold_candidate_c",
        train_sample_ids=("train",),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    candidate = EncoderCandidateConfig(candidate_id="C", hidden_dim=32)

    result = train_common_pretext_method(
        "physiology_only",
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=CommonPretrainingConfig(epochs=1, batch_size=1),
        candidate_config=candidate,
    )
    encoder, _heads, _normalizer, payload = load_common_pretraining_checkpoint(
        result.best_checkpoint_path
    )

    assert payload["candidate_config"]["candidate_id"] == "C"
    assert encoder.config_manifest()["backbone_config"]["hidden_dim"] == 32
