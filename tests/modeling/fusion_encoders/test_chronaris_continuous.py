from __future__ import annotations

import hashlib
import copy

import numpy as np
import torch

from chronaris.modeling.fusion_encoders import (
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousFusionAdapter,
    ChronarisContinuousFusionEncoder,
    build_chronaris_ablation_configs,
    load_chronaris_continuous_checkpoint,
    save_chronaris_continuous_checkpoint,
    validate_chronaris_ablation_diff,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


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
        schema_id="chronaris_continuous_test.v1",
        source_kind="unit_test",
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        physiology_feature_roles=tuple("observed" for _ in PHYSIOLOGY_NAMES),
        vehicle_feature_roles=tuple("observed" for _ in VEHICLE_NAMES),
    )
    physiology = np.asarray(
        [
            [1.0, 1.2, 97.0],
            [1.1, 1.1, 96.5],
            [1.3, 1.0, 96.0],
            [1.4, 0.9, 95.5],
        ],
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


def _adapter(batch, *, variant: str = "full"):
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    config = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        variant=variant,
    )
    return ChronarisContinuousFusionAdapter(
        backbone=ChronarisContinuousFusionEncoder(config).eval(),
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256="7" * 64,
    )


def test_continuous_encoder_is_causal_and_traces_real_ode_updates() -> None:
    torch.manual_seed(17)
    train = _sample("train")
    test = _sample("test")
    changed = _sample("test", future_scale=1000.0)
    adapter = _adapter(collate_observation_samples([train, test]))

    first = adapter(collate_observation_samples([test]))
    trace = adapter.last_encoding.alignment_output.physiology.path_trace
    second = adapter(collate_observation_samples([changed]))
    past = first.timestamps_s[0] < 20.0

    assert first.sequence_embedding.shape == (1, 96, 64)
    assert torch.allclose(
        first.sequence_embedding[0, past],
        second.sequence_embedding[0, past],
        atol=1e-5,
        rtol=1e-5,
    )
    assert trace is not None
    assert trace.continuous_evolution_enabled is True
    assert trace.observation_update_count == 4
    assert trace.reference_positive_evolution_count > 0
    assert trace.maximum_positive_delta_t_s > 0


def test_max_ode_step_is_checkpointed_without_changing_default() -> None:
    default = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
    )
    stepped = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        max_ode_step_s=0.5,
    )

    assert default.max_ode_step_s is None
    assert default.alignment_config().max_ode_step_s is None
    assert stepped.alignment_config().max_ode_step_s == 0.5
    assert stepped.effective_mechanisms()["max_ode_step_s"] == 0.5
    assert ChronarisContinuousEncoderConfig.from_checkpoint_dict(
        stepped.to_checkpoint_dict()
    ) == stepped


def test_learnable_semantic_queries_only_modify_safe_cross_branch_and_are_causal() -> None:
    torch.manual_seed(37)
    batch = collate_observation_samples([_sample("train"), _sample("test")])
    changed = collate_observation_samples([_sample("test", future_scale=1000.0)])
    disabled = ChronarisContinuousFusionEncoder(
        ChronarisContinuousEncoderConfig(
            physiology_feature_names=PHYSIOLOGY_NAMES,
            vehicle_feature_names=VEHICLE_NAMES,
            fusion_kind="safe_lag",
            dropout=0.0,
        )
    )
    enabled = ChronarisContinuousFusionEncoder(
        ChronarisContinuousEncoderConfig(
            physiology_feature_names=PHYSIOLOGY_NAMES,
            vehicle_feature_names=VEHICLE_NAMES,
            fusion_kind="safe_lag",
            semantic_event_enabled=True,
            learnable_semantic_queries=True,
            dropout=0.0,
        )
    )
    enabled.load_state_dict(disabled.state_dict(), strict=False)

    disabled_output = disabled(batch, compute_diagnostics=False)
    enabled_output = enabled(batch, compute_diagnostics=False)
    changed_output = enabled(changed, compute_diagnostics=False)
    past = batch.query_timestamps_s[0] < 20.0

    torch.testing.assert_close(
        enabled_output.sequence_embedding[..., :48],
        disabled_output.sequence_embedding[..., :48],
    )
    future_effect = (
        enabled_output.sequence_embedding[1, past]
        - changed_output.sequence_embedding[0, past]
    ).abs().max()
    assert float(future_effect.detach()) <= 1e-6
    assert enabled_output.semantic_event_output is not None
    assert enabled_output.semantic_event_output.query_names == (
        "flight_event",
        "physiology_response",
        "human_aircraft_coordination",
    )
    assert enabled_output.aggregated_lag_attention.shape == (2, 96, 96)

    enabled.zero_grad(set_to_none=True)
    enabled(batch, compute_diagnostics=False).sequence_embedding.square().mean().backward()
    residual = enabled.semantic_event_fusion.query_bank.query_residual
    assert residual.grad is not None
    assert torch.count_nonzero(residual.grad) > 0


def test_no_continuous_evolution_is_a_real_path_ablation() -> None:
    torch.manual_seed(19)
    batch = collate_observation_samples([_sample("train"), _sample("test")])
    held_out = collate_observation_samples([_sample("test")])
    full = _adapter(batch, variant="full")
    disabled = _adapter(batch, variant="no_continuous_evolution")
    disabled.backbone.load_state_dict(full.backbone.state_dict(), strict=False)

    full_output = full(held_out)
    disabled_output = disabled(held_out)
    trace = disabled.last_encoding.alignment_output.vehicle.path_trace

    assert trace is not None
    assert trace.continuous_evolution_enabled is False
    assert trace.observation_positive_evolution_count == 0
    assert trace.reference_positive_evolution_count == 0
    assert not torch.equal(full_output.sequence_embedding, disabled_output.sequence_embedding)


def test_physics_status_distinguishes_active_disabled_and_unavailable() -> None:
    torch.manual_seed(23)
    batch = collate_observation_samples([_sample("train"), _sample("test")])
    held_out = collate_observation_samples([_sample("test")])
    full = _adapter(batch, variant="full")
    disabled = _adapter(batch, variant="no_physics")

    full(held_out)
    disabled(held_out)
    full_components = full.last_encoding.physics_audit.components
    disabled_components = disabled.last_encoding.physics_audit.components

    assert any(component.active and component.count > 0 for component in full_components)
    assert all(
        component.status in {"active", "unavailable"}
        for component in full_components
    )
    assert any(component.status == "disabled" for component in disabled_components)
    assert all(
        component.weighted_value is None
        for component in disabled_components
    )
    physiology_components = [
        component
        for component in full_components
        if component.component_name.startswith("physiology_")
    ]
    assert physiology_components
    assert all(
        component.status == "unavailable"
        and component.count == 0
        and component.reason == "outside_motion_kinematics_contract"
        for component in physiology_components
    )


def test_task_independent_fast_path_preserves_sequence_and_gradients() -> None:
    torch.manual_seed(31)
    batch = collate_observation_samples([_sample("train"), _sample("test")])
    config = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
        dropout=0.0,
    )
    full = ChronarisContinuousFusionEncoder(config)
    fast = copy.deepcopy(full)

    full_output = full(batch, compute_diagnostics=True)
    fast_output = fast(batch, compute_diagnostics=False)

    torch.testing.assert_close(
        full_output.sequence_embedding,
        fast_output.sequence_embedding,
        rtol=1e-6,
        atol=1e-7,
    )
    assert all(
        component.reason == "task_independent_pretext_fast_path"
        for component in fast_output.physics_audit.components
    )
    full_gradients = torch.autograd.grad(
        full_output.sequence_embedding.square().mean(),
        tuple(full.parameters()),
        allow_unused=True,
    )
    fast_gradients = torch.autograd.grad(
        fast_output.sequence_embedding.square().mean(),
        tuple(fast.parameters()),
        allow_unused=True,
    )
    for actual, expected in zip(full_gradients, fast_gradients, strict=True):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fixed_ablation_matrix_changes_only_declared_mechanisms() -> None:
    full = ChronarisContinuousEncoderConfig(
        physiology_feature_names=PHYSIOLOGY_NAMES,
        vehicle_feature_names=VEHICLE_NAMES,
    )
    variants = build_chronaris_ablation_configs(full)
    assert [config.variant for config in variants] == [
        "full",
        "no_continuous_evolution",
        "no_physics",
        "no_causal_mask",
        "single_scale_lag",
    ]
    for config in variants[1:]:
        assert validate_chronaris_ablation_diff(full, config)


def test_chronaris_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(29)
    batch = collate_observation_samples([_sample("train"), _sample("test")])
    held_out = collate_observation_samples([_sample("test")])
    adapter = _adapter(batch)
    path = save_chronaris_continuous_checkpoint(
        tmp_path / "chronaris.pt",
        backbone=adapter.backbone,
        normalizer=adapter.normalizer,
        seed=29,
    )
    backbone, normalizer, metadata = load_chronaris_continuous_checkpoint(path)
    checksum = sha256_file(path)
    original = replace_adapter_hash(adapter, checksum)
    loaded = ChronarisContinuousFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256=checksum,
    )

    assert metadata["sequence_source"] == "task_head_free_continuous_fusion"
    assert torch.equal(
        original(held_out).sequence_embedding,
        loaded(held_out).sequence_embedding,
    )


def replace_adapter_hash(adapter, checksum):
    return ChronarisContinuousFusionAdapter(
        backbone=adapter.backbone,
        normalizer=adapter.normalizer,
        fold_id=adapter.fold_id,
        checkpoint_sha256=checksum,
    )
