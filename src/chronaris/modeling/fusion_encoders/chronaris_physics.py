"""Structured physical-consistency availability for Chronaris training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from chronaris.models.alignment.physics import build_rigid_body_vehicle_residuals
from chronaris.models.alignment.physics_state_mapping import (
    RigidBodyStateMapping,
    build_rigid_body_state_mapping,
)
from chronaris.models.alignment.prototype import DualStreamPrototypeOutput
from chronaris.models.alignment.torch_batch import TorchAlignmentBatch


RIGID_COMPONENTS = (
    "vehicle_rigid_body_translation",
    "vehicle_rigid_body_vertical",
    "vehicle_rigid_body_rotation",
)
PHYSIOLOGY_COMPONENTS = (
    "physiology_smoothness",
    "physiology_spo2_delta",
)


@dataclass(frozen=True, slots=True)
class PhysicsComponentStatus:
    component_name: str
    status: str
    available: bool
    active: bool
    count: int
    raw_value: torch.Tensor | None
    weighted_value: torch.Tensor | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ChronarisPhysicsAudit:
    components: tuple[PhysicsComponentStatus, ...]
    total_weighted_value: torch.Tensor
    observation_anchor: torch.Tensor | None = None
    observation_count: int = 0
    calibrated_residual: torch.Tensor | None = None
    residual_weight: float = 0.1

    @property
    def active_component_count(self) -> int:
        return sum(component.active for component in self.components)


def build_chronaris_physics_audit(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    field_labels: Mapping[str, str] | None = None,
    enabled: bool,
    weight: float,
    huber_delta: float = 1.0,
    calibration: Mapping[str, object] | None = None,
) -> ChronarisPhysicsAudit:
    """Compute available components while keeping unavailable distinct from zero."""

    if weight < 0 or huber_delta <= 0:
        raise ValueError("physics weight/delta configuration is invalid")
    if calibration is not None:
        return _calibrated_audit(output, batch, calibration, enabled=enabled, weight=weight, huber_delta=huber_delta)
    labels = field_labels or {}
    mapping = build_rigid_body_state_mapping(
        batch.vehicle.feature_names,
        field_labels=labels,
    )
    values = build_rigid_body_vehicle_residuals(
        output.vehicle.reconstructions,
        batch.vehicle.offsets_s,
        batch.vehicle.feature_valid_mask & batch.vehicle.mask.unsqueeze(-1),
        batch.vehicle.feature_names,
        mapping,
        huber_delta=huber_delta,
    )
    counts = _component_counts(batch, mapping)
    components: list[PhysicsComponentStatus] = []
    for component_name in (*RIGID_COMPONENTS, *PHYSIOLOGY_COMPONENTS):
        count = counts[component_name]
        available = count > 0
        raw_value = values[component_name] if available else None
        active = bool(enabled and available)
        if active:
            status = "active"
            reason = None
            weighted_value = raw_value * weight
        elif available:
            status = "disabled"
            reason = "ablation_no_physics" if not enabled else "zero_weight"
            weighted_value = None
        else:
            status = "unavailable"
            reason = _unavailable_reason(component_name, mapping)
            weighted_value = None
        components.append(
            PhysicsComponentStatus(
                component_name=component_name,
                status=status,
                available=available,
                active=active,
                count=count,
                raw_value=raw_value,
                weighted_value=weighted_value,
                reason=reason,
            )
        )
    active_values = [
        component.weighted_value
        for component in components
        if component.weighted_value is not None
    ]
    total = (
        torch.stack(active_values).sum()
        if active_values
        else output.vehicle.reconstructions.new_zeros(())
    )
    return ChronarisPhysicsAudit(tuple(components), total)


def _calibrated_audit(output, batch, calibration, *, enabled, weight, huber_delta):
    from chronaris.models.alignment.calibrated_physics import calibrated_kinematic_losses, observation_anchor_loss
    relations = calibrated_kinematic_losses(output.vehicle.reconstructions, batch.vehicle, calibration, huber_delta=huber_delta)
    anchor, anchor_count = observation_anchor_loss(output, batch)
    components = []
    for name in (*RIGID_COMPONENTS, *PHYSIOLOGY_COMPONENTS):
        active_rows = [(loss, count) for component, loss, count in relations if component == name and count]
        count = sum(n for _loss, n in active_rows)
        value = torch.stack([loss for loss, _n in active_rows]).mean() if count else None
        active = bool(count and enabled and weight > 0)
        components.append(PhysicsComponentStatus(
            name, "active" if active else "disabled" if count else "unavailable",
            bool(count), active, count, value, value * weight if active else None,
            None if active else "ablation_no_physics" if count else "no_calibrated_physical_relation",
        ))
    active = [loss for _component, loss, count in relations if count]
    residual = torch.stack(active).mean() if active else anchor * 0
    effective_weight = weight if enabled else 0.0
    return ChronarisPhysicsAudit(tuple(components), residual * effective_weight, anchor,
                                 anchor_count, residual, effective_weight)


def build_skipped_chronaris_physics_audit(reference: torch.Tensor) -> ChronarisPhysicsAudit:
    """Represent a deliberate task-independent fast path without claiming zero loss."""

    components = tuple(
        PhysicsComponentStatus(
            component_name=name,
            status="unavailable",
            available=False,
            active=False,
            count=0,
            raw_value=None,
            weighted_value=None,
            reason="task_independent_pretext_fast_path",
        )
        for name in (*RIGID_COMPONENTS, *PHYSIOLOGY_COMPONENTS)
    )
    return ChronarisPhysicsAudit(components, reference.new_zeros(()))


def physics_audit_to_rows(
    audit: ChronarisPhysicsAudit,
) -> tuple[dict[str, object], ...]:
    """Detach one audit into manifest/report-safe scalar rows."""

    rows = []
    for component in audit.components:
        rows.append(
            {
                "component_name": component.component_name,
                "status": component.status,
                "available": component.available,
                "active": component.active,
                "count": component.count,
                "raw_value": (
                    float(component.raw_value.detach().cpu())
                    if component.raw_value is not None
                    else None
                ),
                "weighted_value": (
                    float(component.weighted_value.detach().cpu())
                    if component.weighted_value is not None
                    else None
                ),
                "reason": component.reason,
            }
        )
    return tuple(rows)


def _component_counts(
    batch: TorchAlignmentBatch,
    mapping: RigidBodyStateMapping,
) -> dict[str, int]:
    vehicle = batch.vehicle
    counts = {
        "vehicle_rigid_body_translation": _derivative_pair_count(
            vehicle,
            mapping.speed,
            mapping.acceleration,
        ),
        "vehicle_rigid_body_vertical": _derivative_pair_count(
            vehicle,
            mapping.altitude,
            mapping.vertical_speed,
        ),
        "vehicle_rigid_body_rotation": sum(
            _derivative_pair_count(vehicle, attitude, rate)
            for attitude, rate in (
                (mapping.pitch, mapping.pitch_rate),
                (mapping.roll, mapping.roll_rate),
                (mapping.yaw, mapping.yaw_rate),
            )
        ),
    }
    counts["physiology_smoothness"] = 0
    counts["physiology_spo2_delta"] = 0
    return counts


def _derivative_pair_count(stream, source_names, target_names) -> int:
    feature_index = {name: index for index, name in enumerate(stream.feature_names)}
    source_indices = [feature_index[name] for name in source_names if name in feature_index]
    target_indices = [feature_index[name] for name in target_names if name in feature_index]
    if not source_indices or not target_indices or stream.values.shape[1] <= 1:
        return 0
    source_valid = stream.feature_valid_mask[..., source_indices].any(dim=-1)
    target_valid = stream.feature_valid_mask[..., target_indices].any(dim=-1)
    positive_delta = stream.offsets_s[:, 1:] > stream.offsets_s[:, :-1]
    valid = (
        source_valid[:, :-1]
        & source_valid[:, 1:]
        & target_valid[:, 1:]
        & positive_delta
    )
    return int(valid.sum().item())


def _unavailable_reason(component_name, mapping) -> str:
    if component_name in PHYSIOLOGY_COMPONENTS:
        return "outside_motion_kinematics_contract"
    requirements = {
        "vehicle_rigid_body_translation": bool(mapping.speed and mapping.acceleration),
        "vehicle_rigid_body_vertical": bool(mapping.altitude and mapping.vertical_speed),
        "vehicle_rigid_body_rotation": bool(
            (mapping.pitch and mapping.pitch_rate)
            or (mapping.roll and mapping.roll_rate)
            or (mapping.yaw and mapping.yaw_rate)
        ),
    }
    return (
        "no_valid_derivative_pairs"
        if requirements[component_name]
        else "required_semantic_fields_missing"
    )
