from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.fusion_encoders.chronaris_v2 import (
    V2_SUBSPACE_SLICES,
    ChronarisV2EncoderConfig,
    ChronarisV2FusionEncoder,
    load_chronaris_v2_checkpoint,
    save_chronaris_v2_checkpoint,
)
from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream
from chronaris.modeling.fusion_encoders.learned_causal import (
    LearnedRelativeCausalFusion,
    LearnedRelativeCausalFusionConfig,
    LearnedRelativeCausalFusionInput,
)
from chronaris.modeling.fusion_encoders.semantic_groups import (
    build_vehicle_semantic_group_map,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
    TrainOnlyRobustNormalizer,
)


PHYSIOLOGY_NAMES = ("physiology.spo2", "physiology.eeg.af3")
VEHICLE_NAMES = (
    "vehicle.pitch_rad",
    "vehicle.roll_rate_rps",
    "vehicle.normal_load_g",
    "vehicle.speed_mps",
    "vehicle.throttle",
    "vehicle.fuel_mass",
    "vehicle.unmapped_bus_value",
)


def _sample(sample_id: str, *, future_scale: float = 1.0):
    schema = ObservationSchema(
        schema_id="chronaris_v2_test.v1",
        source_kind="unit_test",
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        physiology_feature_roles=("observed",) * len(PHYSIOLOGY_NAMES),
        vehicle_feature_roles=("observed",) * len(VEHICLE_NAMES),
    )
    physiology = np.asarray(
        [[97.0, 0.1], [96.5, 0.2], [96.0, 0.3], [95.5, 0.4]],
        dtype=np.float32,
    )
    vehicle = np.arange(28, dtype=np.float32).reshape(4, 7) / 10
    physiology[-1] *= future_scale
    vehicle[-1] *= future_scale
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id=sample_id,
        schema=schema,
        physiology_values=physiology,
        physiology_timestamps_s=np.asarray([0.0, 5.0, 10.0, 20.0]),
        physiology_feature_mask=np.ones_like(physiology, dtype=bool),
        vehicle_values=vehicle,
        vehicle_timestamps_s=np.asarray([0.0, 4.0, 12.0, 20.0]),
        vehicle_feature_mask=np.ones_like(vehicle, dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def test_semantic_group_map_assigns_every_vehicle_feature_once() -> None:
    mapping = build_vehicle_semantic_group_map(VEHICLE_NAMES)
    grouped = [index for indices in mapping.groups.values() for index in indices]

    assert sorted(grouped) == list(range(len(VEHICLE_NAMES)))
    assert mapping.groups["attitude_and_rate"] == (0, 1)
    assert mapping.groups["acceleration_and_load"] == (2,)
    assert mapping.groups["speed_and_altitude"] == (3,)
    assert mapping.groups["control_input"] == (4,)
    assert mapping.groups["status_mass_and_event"] == (5,)
    assert mapping.groups["other_numeric"] == (6,)
    assert len(mapping.mapping_sha256) == 64


def test_learned_causal_attention_excludes_future_keys() -> None:
    torch.manual_seed(17)
    model = LearnedRelativeCausalFusion(
        LearnedRelativeCausalFusionConfig(
            physiology_dim=8,
            vehicle_dim=12,
            attention_dim=16,
            output_dim=24,
            num_heads=4,
            dropout=0.0,
        )
    ).eval()
    output = model(
        LearnedRelativeCausalFusionInput(
            physiology_states=torch.randn(1, 4, 8),
            vehicle_states=torch.randn(1, 4, 12),
            physiology_valid_mask=torch.ones(1, 4, dtype=torch.bool),
            vehicle_valid_mask=torch.ones(1, 4, dtype=torch.bool),
            query_timestamps_s=torch.tensor([[0.0, 5.0, 10.0, 20.0]]),
        )
    )

    assert output.shared_embedding.shape == (1, 4, 24)
    for weights, mask in zip(output.attention_weights, output.lag_masks, strict=True):
        assert torch.count_nonzero(weights.masked_select(~mask)) == 0
    assert torch.allclose(
        output.scale_gate_weights.sum(dim=-1),
        output.scale_available_mask.any(dim=-1).to(torch.float32),
    )
    output.shared_embedding.sum().backward()
    assert model.query_projection.weight.grad is not None
    assert model.key_projection.weight.grad is not None
    assert model.value_projection.weight.grad is not None
    assert model.query_projection.weight.data_ptr() != model.key_projection.weight.data_ptr()


def test_learned_causal_missing_tokens_preserve_one_modality_and_zero_empty_queries() -> None:
    model = LearnedRelativeCausalFusion(
        LearnedRelativeCausalFusionConfig(
            physiology_dim=8,
            vehicle_dim=8,
            attention_dim=16,
            num_heads=4,
            dropout=0.0,
        )
    ).eval()
    physiology_valid = torch.tensor([[False, True, False]])
    vehicle_valid = torch.tensor([[True, False, False]])
    output = model(
        LearnedRelativeCausalFusionInput(
            physiology_states=torch.randn(1, 3, 8),
            vehicle_states=torch.randn(1, 3, 8),
            physiology_valid_mask=physiology_valid,
            vehicle_valid_mask=vehicle_valid,
            query_timestamps_s=torch.tensor([[0.0, 5.0, 10.0]]),
        )
    )

    assert output.modality_available_mask.tolist() == [[True, True, False]]
    assert torch.isfinite(output.shared_embedding).all()
    assert torch.count_nonzero(output.shared_embedding[:, 2]) == 0


def test_chronaris_v2_preserves_contract_subspaces_and_causality() -> None:
    torch.manual_seed(23)
    original = collate_observation_samples((_sample("test"),))
    changed = collate_observation_samples((_sample("test", future_scale=1000.0),))
    config = ChronarisV2EncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        internal_hidden_dim=32,
        physiology_hidden_dim=16,
        vehicle_hidden_dim=32,
        num_heads=4,
        dropout=0.0,
        physics_enabled=False,
    )
    encoder = ChronarisV2FusionEncoder(config).eval()
    first = encoder(original, compute_diagnostics=False)
    second = encoder(changed, compute_diagnostics=False)
    past = original.query_timestamps_s[0] < 20.0

    assert first.sequence_embedding.shape == (1, 96, 64)
    assert first.vehicle_private.shape[-1] == 24
    assert first.physiology_private.shape[-1] == 16
    assert first.causal_shared.shape[-1] == 24
    assert first.subspace_slices == V2_SUBSPACE_SLICES
    torch.testing.assert_close(
        first.sequence_embedding[0, past],
        second.sequence_embedding[0, past],
        rtol=1e-5,
        atol=1e-5,
    )


def test_direct_physiology_residual_preserves_causal_observed_values() -> None:
    batch = collate_observation_samples((_sample("direct"),))
    encoder = ChronarisV2FusionEncoder(
        ChronarisV2EncoderConfig(
            physiology_feature_names=PHYSIOLOGY_NAMES,
            vehicle_feature_names=VEHICLE_NAMES,
            internal_hidden_dim=32,
            physiology_hidden_dim=16,
            vehicle_hidden_dim=32,
            num_heads=4,
            dropout=0.0,
            physics_enabled=False,
            physiology_residual_mode="direct_causal_query",
        )
    ).eval()
    output = encoder(batch, compute_diagnostics=False)
    queried = causal_query_stream(batch, stream_name="physiology")

    torch.testing.assert_close(
        output.physiology_private[..., : len(PHYSIOLOGY_NAMES)][
            queried.feature_mask
        ],
        queried.values[queried.feature_mask],
    )
    assert output.physiology_private.shape[-1] == 16


def test_continuous_basis_has_positive_ordered_initial_centers() -> None:
    model = LearnedRelativeCausalFusion(
        LearnedRelativeCausalFusionConfig(
            physiology_dim=8,
            vehicle_dim=8,
            attention_dim=16,
            num_heads=4,
            lag_mode="continuous_basis",
        )
    )
    centers, bandwidths = model._lag_parameters()

    assert torch.all(centers[1:] > centers[:-1])
    assert torch.all(bandwidths > 0)


def test_v2_legacy_attention_and_mixed_output_are_executable() -> None:
    batch = collate_observation_samples((_sample("legacy"),))
    encoder = ChronarisV2FusionEncoder(
        ChronarisV2EncoderConfig(
            physiology_feature_names=PHYSIOLOGY_NAMES,
            vehicle_feature_names=VEHICLE_NAMES,
            internal_hidden_dim=32,
            physiology_hidden_dim=32,
            vehicle_hidden_dim=32,
            num_heads=4,
            dropout=0.0,
            physics_enabled=False,
            learned_causal_attention=False,
            private_shared_subspaces=False,
            corrected_physics=False,
        )
    ).eval()
    output = encoder(batch, compute_diagnostics=False)

    assert output.sequence_embedding.shape == (1, 96, 64)
    assert output.fusion_output.attention_weights
    assert encoder.effective_mechanisms()["learned_multihead_qkv"] is False
    assert encoder.effective_mechanisms()["private_shared_subspaces"] is False


def test_v2_checkpoint_round_trip_is_strictly_versioned(tmp_path) -> None:
    torch.manual_seed(31)
    batch = collate_observation_samples((_sample("train"), _sample("held-out")))
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("held-out",),
    )
    config = ChronarisV2EncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        internal_hidden_dim=32,
        physiology_hidden_dim=16,
        vehicle_hidden_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    backbone = ChronarisV2FusionEncoder(config).eval()
    path = save_chronaris_v2_checkpoint(
        tmp_path / "chronaris-v2.pt",
        backbone=backbone,
        normalizer=normalizer,
        seed=31,
    )
    loaded, _normalizer, metadata = load_chronaris_v2_checkpoint(path)

    assert metadata["format"] == "chronaris.continuous_fusion_encoder.v2"
    assert metadata["architecture_version"] == "v2"
    assert metadata["subspace_slices"] == {
        name: list(bounds) for name, bounds in V2_SUBSPACE_SLICES.items()
    }
    for actual, expected in zip(
        loaded.state_dict().values(),
        backbone.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(actual, expected)
