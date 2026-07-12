from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.fusion_encoders.chronaris_v2 import (
    ChronarisV2EncoderConfig,
    ChronarisV2FusionEncoder,
)
from chronaris.models.alignment.physics_residuals import (
    build_rigid_body_vehicle_residuals,
)
from chronaris.models.alignment.physics_state_mapping import RigidBodyStateMapping
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)


def test_strict_physics_rejects_ambiguous_axis_aggregation() -> None:
    values = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 6.0, 0.0]]]
    )
    times = torch.tensor([[0.0, 1.0, 2.0]])
    valid = torch.ones_like(values, dtype=torch.bool)
    mapping = RigidBodyStateMapping(
        speed=("speed_a", "speed_b"),
        acceleration=("acceleration",),
    )
    loose = build_rigid_body_vehicle_residuals(
        values,
        times,
        valid,
        ("speed_a", "speed_b", "acceleration"),
        mapping,
        huber_delta=1.0,
        strict_axis_pairs=False,
    )
    strict = build_rigid_body_vehicle_residuals(
        values,
        times,
        valid,
        ("speed_a", "speed_b", "acceleration"),
        mapping,
        huber_delta=1.0,
        strict_axis_pairs=True,
    )

    assert loose["vehicle_rigid_body_translation"] > 0
    assert strict["vehicle_rigid_body_translation"] == 0


def test_strict_periodic_rotation_uses_wrapped_angle_difference() -> None:
    yaw = torch.tensor([3.13, -3.13, -3.11])
    wrapped_rate = torch.tensor([0.0, 0.0231853, 0.02])
    values = torch.stack((yaw, wrapped_rate), dim=-1).unsqueeze(0)
    times = torch.tensor([[0.0, 1.0, 2.0]])
    valid = torch.ones_like(values, dtype=torch.bool)
    mapping = RigidBodyStateMapping(
        yaw=("yaw_rad",),
        yaw_rate=("yaw_rate_rps",),
    )
    strict = build_rigid_body_vehicle_residuals(
        values,
        times,
        valid,
        ("yaw_rad", "yaw_rate_rps"),
        mapping,
        huber_delta=1.0,
        strict_axis_pairs=True,
    )

    assert strict["vehicle_rigid_body_rotation"] < 1e-6


def test_v2_physics_requires_attached_inverse_normalizer() -> None:
    schema = ObservationSchema(
        schema_id="chronaris_v2_physics_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.spo2",),
        vehicle_feature_names=("vehicle.pitch_rad", "vehicle.pitch_rate_rps"),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed", "observed"),
    )

    def sample(sample_id, offset):
        return ObservedDualStreamSample(
            sample_id=sample_id,
            group_id=sample_id,
            schema=schema,
            physiology_values=np.asarray(
                [[96.0 + offset], [95.0 + offset], [94.0 + offset]],
                dtype=np.float32,
            ),
            physiology_timestamps_s=np.asarray([0.0, 1.0, 2.0]),
            physiology_feature_mask=np.ones((3, 1), dtype=bool),
            vehicle_values=np.asarray(
                [[0.0, 0.0], [0.1, 0.1], [0.2, 0.1]],
                dtype=np.float32,
            ),
            vehicle_timestamps_s=np.asarray([0.0, 1.0, 2.0]),
            vehicle_feature_mask=np.ones((3, 2), dtype=bool),
            source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
        )

    batch = collate_observation_samples((sample("train", 0.0), sample("test", 1.0)))
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    config = ChronarisV2EncoderConfig(
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
        internal_hidden_dim=32,
        physiology_hidden_dim=16,
        vehicle_hidden_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    encoder = ChronarisV2FusionEncoder(config).eval()
    unavailable = encoder(normalizer.transform(batch), compute_diagnostics=True)
    encoder.attach_normalizer(normalizer)
    active = encoder(normalizer.transform(batch), compute_diagnostics=True)

    assert all(
        component.reason == "physics_normalization_unavailable"
        for component in unavailable.physics_audit.components
    )
    assert any(component.active for component in active.physics_audit.components)
