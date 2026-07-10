from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import torch

from chronaris.modeling.fusion_encoders import (
    ContinuousTimeSingleStreamEncoder,
    SingleStreamEncoderConfig,
    SingleStreamFusionAdapter,
    load_single_stream_checkpoint,
    save_single_stream_checkpoint,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from tests.representation.test_contracts import _sample


def _future_sample(sample_id: str, future_value: float) -> ObservedDualStreamSample:
    schema = ObservationSchema(
        schema_id="single_stream_future_test.v1",
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
        physiology_values=np.asarray([[1.0], [2.0], [future_value]], dtype=np.float32),
        physiology_timestamps_s=np.asarray([0.0, 5.0, 20.0], dtype=np.float64),
        physiology_feature_mask=np.ones((3, 1), dtype=bool),
        vehicle_values=np.asarray([[3.0]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.0], dtype=np.float64),
        vehicle_feature_mask=np.ones((1, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def test_two_single_stream_baselines_reuse_one_backbone_class():
    physiology = ContinuousTimeSingleStreamEncoder(
        SingleStreamEncoderConfig(
            active_stream="physiology",
            input_feature_dim=2,
            dropout=0.0,
        )
    )
    vehicle = ContinuousTimeSingleStreamEncoder(
        SingleStreamEncoderConfig(
            active_stream="vehicle",
            input_feature_dim=1,
            dropout=0.0,
        )
    )

    assert type(physiology) is type(vehicle)
    assert physiology.config.hidden_dim == vehicle.config.hidden_dim == 64
    assert physiology.config.layers == vehicle.config.layers == 2


def test_non_active_modality_change_does_not_affect_single_stream_output():
    torch.manual_seed(11)
    batch = collate_observation_samples([_sample("train"), _sample("test", shift=2.0)])
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    backbone = ContinuousTimeSingleStreamEncoder(
        SingleStreamEncoderConfig(
            active_stream="physiology",
            input_feature_dim=2,
            dropout=0.0,
        )
    ).eval()
    adapter = SingleStreamFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256="1" * 64,
    )
    perturbed = replace(batch, vehicle_values=batch.vehicle_values + 1_000_000.0)

    first = adapter(batch)
    second = adapter(perturbed)

    assert torch.equal(first.sequence_embedding, second.sequence_embedding)


def test_single_stream_backbone_and_attention_are_future_invariant():
    torch.manual_seed(13)
    train = _future_sample("train", 3.0)
    base = _future_sample("test", 9.0)
    changed = _future_sample("test", 9999.0)
    fit_batch = collate_observation_samples([train, base])
    normalizer = TrainOnlyRobustNormalizer().fit(
        fit_batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    backbone = ContinuousTimeSingleStreamEncoder(
        SingleStreamEncoderConfig(
            active_stream="physiology",
            input_feature_dim=1,
            dropout=0.0,
        )
    ).eval()
    adapter = SingleStreamFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256="2" * 64,
    )

    first = adapter(collate_observation_samples([base]))
    second = adapter(collate_observation_samples([changed]))
    past = first.timestamps_s[0] < 20.0

    assert torch.allclose(
        first.sequence_embedding[0, past],
        second.sequence_embedding[0, past],
        atol=1e-6,
        rtol=1e-6,
    )


def test_single_stream_checkpoint_round_trips_model_and_train_only_transform(tmp_path):
    torch.manual_seed(17)
    batch = collate_observation_samples([_sample("train"), _sample("test", shift=2.0)])
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    backbone = ContinuousTimeSingleStreamEncoder(
        SingleStreamEncoderConfig(
            active_stream="vehicle",
            input_feature_dim=1,
            dropout=0.0,
        )
    ).eval()
    path = save_single_stream_checkpoint(
        tmp_path / "vehicle.pt",
        backbone=backbone,
        normalizer=normalizer,
        seed=17,
    )
    loaded_backbone, loaded_normalizer, metadata = load_single_stream_checkpoint(path)
    checkpoint_hash = sha256_file(path)
    first = SingleStreamFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256=checkpoint_hash,
    )(batch)
    second = SingleStreamFusionAdapter(
        backbone=loaded_backbone,
        normalizer=loaded_normalizer,
        fold_id="fold_a",
        checkpoint_sha256=checkpoint_hash,
    )(batch)

    assert metadata["label_used_for_encoder_training"] is False
    assert loaded_normalizer.fit_sample_ids == ("train",)
    assert torch.equal(first.sequence_embedding, second.sequence_embedding)
