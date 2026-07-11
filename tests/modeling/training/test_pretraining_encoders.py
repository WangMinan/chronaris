from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    build_trainable_fusion_encoder,
    build_chronaris_auxiliary_losses,
    chronaris_auxiliary_weight_schedule,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
)


def _sample(sample_id: str):
    schema = ObservationSchema(
        schema_id="pretraining_encoder_test.v1",
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
        physiology_values=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        physiology_timestamps_s=np.asarray([0.0, 5.0, 10.0, 20.0]),
        physiology_feature_mask=np.ones((4, 1), dtype=bool),
        vehicle_values=np.asarray([[5.0], [6.0], [7.0], [8.0]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.0, 4.0, 12.0, 20.0]),
        vehicle_feature_mask=np.ones((4, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


@pytest.mark.parametrize("method_name", TRAINABLE_FUSION_METHODS)
def test_five_trainable_encoders_share_differentiable_sequence_contract(method_name):
    torch.manual_seed(17)
    batch = collate_observation_samples([_sample("a"), _sample("b")])
    encoder = build_trainable_fusion_encoder(
        method_name,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
    )
    output = encoder(batch)
    loss = output.sequence_embedding.square().mean()
    loss.backward()

    assert output.method_name == method_name
    assert output.sequence_embedding.shape == (2, 96, 64)
    assert output.modality_available_mask.shape == (2, 96)
    assert torch.isfinite(output.sequence_embedding).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    if method_name == "chronaris":
        assert output.auxiliary["physical_consistency"] != "not_applicable"
    else:
        assert output.auxiliary["physical_consistency"] == "not_applicable"


@pytest.mark.parametrize("method_name", TRAINABLE_FUSION_METHODS)
def test_hidden_32_candidate_preserves_64_dimensional_contract(method_name):
    batch = collate_observation_samples([_sample("candidate-c")])
    candidate = next(
        value for value in ENCODER_SCREEN_CANDIDATES if value.candidate_id == "C"
    )
    encoder = build_trainable_fusion_encoder(
        method_name,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        candidate_config=candidate,
    )

    output = encoder(batch)

    assert output.sequence_embedding.shape == (1, 96, 64)
    assert encoder.config_manifest()["backbone_config"]["hidden_dim"] == 32


def test_frozen_candidate_table_matches_protocol():
    assert [value.candidate_id for value in ENCODER_SCREEN_CANDIDATES] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert [value.hidden_dim for value in ENCODER_SCREEN_CANDIDATES] == [64, 64, 32, 64]
    assert [value.learning_rate for value in ENCODER_SCREEN_CANDIDATES] == [
        1e-3,
        3e-4,
        1e-3,
        1e-3,
    ]
    assert [value.dropout for value in ENCODER_SCREEN_CANDIDATES] == [0.1, 0.1, 0.1, 0.2]


def test_chronaris_auxiliary_losses_are_differentiable_and_causal_margin_active():
    torch.manual_seed(23)
    batch = collate_observation_samples([_sample("auxiliary")])
    encoder = build_trainable_fusion_encoder(
        "chronaris",
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
    )

    positive = encoder(batch, compute_chronaris_diagnostics=True)
    negative = encoder(batch, compute_chronaris_diagnostics=True)
    weights = chronaris_auxiliary_weight_schedule(20)
    losses = build_chronaris_auxiliary_losses(
        positive,
        negative,
        weights=weights,
    )
    losses.total_loss.backward()

    assert losses.alignment_count > 0
    assert losses.causal_count == losses.alignment_count
    assert losses.causal_direction >= 0.099
    assert torch.isfinite(losses.total_loss)
    assert any(parameter.grad is not None for parameter in encoder.parameters())
