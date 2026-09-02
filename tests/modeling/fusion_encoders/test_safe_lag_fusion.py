from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.fusion_encoders import (
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousFusionAdapter,
    ChronarisContinuousFusionEncoder,
)
from chronaris.modeling.fusion_encoders.multiscale_causal import (
    MultiScaleCausalFusionInput,
)
from chronaris.modeling.fusion_encoders.safe_lag_fusion import (
    SafeLagAwareFusion,
    SafeLagAwareFusionConfig,
    SafeLagAwareFusionOutput,
    scale_gate_entropy_regularization,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)

PHYSIOLOGY_NAMES = (
    "physiology.eeg.af3",
    "physiology.eeg.af4",
    "physiology.spo2.percent",
)
VEHICLE_NAMES = (
    "vehicle.speed_mps",
    "vehicle.altitude_m",
    "vehicle.vertical_speed_mps",
    "vehicle.roll_rad",
    "vehicle.pitch_rad",
    "vehicle.yaw_rad",
    "vehicle.roll_rate_rps",
    "vehicle.pitch_rate_rps",
    "vehicle.yaw_rate_rps",
    "vehicle.longitudinal_acc_mps2",
    "vehicle.lateral_acc_mps2",
    "vehicle.normal_load_g",
)


def _sample(sample_id: str, *, future_scale: float = 1.0):
    schema = ObservationSchema(
        schema_id="safe_lag_test.v1",
        source_kind="unit_test",
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        physiology_feature_roles=tuple("observed" for _ in PHYSIOLOGY_NAMES),
        vehicle_feature_roles=tuple("observed" for _ in VEHICLE_NAMES),
    )
    physiology = np.asarray(
        [[1.0, 1.2, 97.0], [1.1, 1.1, 96.5], [1.3, 1.0, 96.0], [1.4, 0.9, 95.5]],
        dtype=np.float32,
    )
    vehicle = np.arange(48, dtype=np.float32).reshape(4, 12) / 10.0
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


def _fusion_input(*, batch: int = 2, timepoints: int = 96, hidden: int = 64):
    torch.manual_seed(17)
    query = torch.linspace(0.0, 29.9, timepoints).unsqueeze(0).expand(batch, -1)
    return MultiScaleCausalFusionInput(
        physiology_states=torch.randn(batch, timepoints, hidden),
        vehicle_states=torch.randn(batch, timepoints, hidden),
        physiology_valid_mask=torch.ones(batch, timepoints, dtype=torch.bool),
        vehicle_valid_mask=torch.ones(batch, timepoints, dtype=torch.bool),
        query_timestamps_s=query,
    )


def test_safe_lag_forward_shape_and_gate_init() -> None:
    torch.manual_seed(17)
    fusion = SafeLagAwareFusion(SafeLagAwareFusionConfig())
    fusion.eval()
    with torch.no_grad():
        output = fusion(_fusion_input())
    assert isinstance(output, SafeLagAwareFusionOutput)
    assert output.sequence_embedding.shape == (2, 96, 64)
    assert output.cross_gate.shape == (2, 96, 1)
    assert output.scale_gate_weights.shape == (2, 96, 3)
    assert output.physiology_private.shape == (2, 96, 24)
    assert output.vehicle_private.shape == (2, 96, 24)
    assert output.cross_features.shape == (2, 96, 16)
    # Safe fallback: the cross gate starts near zero.
    assert float(output.cross_gate.mean()) < 0.05


def test_safe_lag_preserves_vehicle_info_via_bypass() -> None:
    """Perturbing the vehicle stream must move the vehicle-private output chunk.

    This is the core fix: unlike the original single-bottleneck fusion, vehicle
    information reaches the output through a dedicated private projection.
    """
    torch.manual_seed(17)
    fusion = SafeLagAwareFusion(SafeLagAwareFusionConfig()).eval()
    base_input = _fusion_input()
    with torch.no_grad():
        base = fusion(base_input).sequence_embedding
        perturbed_vehicle = _fusion_input()
        torch.manual_seed(31)
        noise = torch.randn_like(base_input.vehicle_states) * 2.0
        perturbed_vehicle = MultiScaleCausalFusionInput(
            physiology_states=base_input.physiology_states.clone(),
            vehicle_states=base_input.vehicle_states.clone() + noise,
            physiology_valid_mask=base_input.physiology_valid_mask,
            vehicle_valid_mask=base_input.vehicle_valid_mask,
            query_timestamps_s=base_input.query_timestamps_s,
        )
        veh_out = fusion(perturbed_vehicle).sequence_embedding
    vehicle_chunk_delta = (veh_out[..., 24:48] - base[..., 24:48]).abs().mean()
    cross_chunk_delta = (veh_out[..., 48:64] - base[..., 48:64]).abs().mean()
    # Vehicle-private chunk reacts strongly; cross chunk barely moves (gate ~0).
    assert float(vehicle_chunk_delta) > 1e-3
    assert float(cross_chunk_delta) < float(vehicle_chunk_delta)


def test_safe_lag_encoder_wires_and_is_causal() -> None:
    torch.manual_seed(17)
    train = _sample("train")
    test = _sample("test")
    changed = _sample("test", future_scale=1000.0)
    normalizer = TrainOnlyRobustNormalizer().fit(
        collate_observation_samples([train, test]),
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    config = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        variant="full",
        fusion_kind="safe_lag",
    )
    adapter = ChronarisContinuousFusionAdapter(
        backbone=ChronarisContinuousFusionEncoder(config).eval(),
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256="7" * 64,
    )
    adapter(collate_observation_samples([train, test]))
    first = adapter(collate_observation_samples([test]))
    second = adapter(collate_observation_samples([changed]))
    assert first.sequence_embedding.shape == (1, 96, 64)
    assert isinstance(adapter.last_encoding.fusion_output, SafeLagAwareFusionOutput)
    past = first.timestamps_s[0] < 20.0
    # Causality: future perturbation must not change past query outputs.
    assert torch.allclose(
        first.sequence_embedding[0, past],
        second.sequence_embedding[0, past],
        atol=1e-5,
        rtol=1e-5,
    )


def test_scale_gate_entropy_regularization_is_bounded_scalar() -> None:
    torch.manual_seed(17)
    weights = torch.softmax(torch.randn(2, 96, 3), dim=-1)
    available = torch.ones(2, 96, 3, dtype=torch.bool)
    penalty = scale_gate_entropy_regularization(weights, available_mask=available)
    assert penalty.ndim == 0
    assert 0.0 <= float(penalty) <= 1.0
    # A collapsed one-hot gate has penalty ~1.
    one_hot = torch.zeros(2, 96, 3)
    one_hot[..., 0] = 1.0
    collapsed = scale_gate_entropy_regularization(one_hot, available_mask=available)
    assert float(collapsed) > float(penalty)


def test_safe_lag_supports_ablation_matrix() -> None:
    from chronaris.modeling.fusion_encoders import build_chronaris_ablation_configs

    base = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        fusion_kind="safe_lag",
    )
    ablations = build_chronaris_ablation_configs(base)
    assert len(ablations) == 6
    assert all(config.fusion_kind == "safe_lag" for config in ablations)
    assert ablations[-1].variant == "no_single_stream_bypass"
