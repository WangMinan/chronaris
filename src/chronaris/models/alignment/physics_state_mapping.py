"""Rigid-body semantic mapping for Stage F physics constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_TRANSLATION_GROUP_TOKENS: Mapping[str, tuple[str, ...]] = {
    "speed": (
        "速度",
        "空速",
        "地速",
        "真空速",
        "表速",
        "speed",
        "groundspeed",
        "tas",
        "ias",
        "mach",
    ),
    "acceleration": (
        "加速度",
        "加速度模",
        "过载",
        "纵向过载",
        "法向过载",
        "acceleration",
        "accel",
        "acc",
        "nx",
        "ny",
        "nz",
    ),
    "altitude": (
        "高度",
        "海拔",
        "海拔高度",
        "气压高",
        "气压高度",
        "雷达高",
        "卫星高度",
        "altitude",
        "height",
        "alt",
    ),
    "vertical_speed": (
        "垂直速度",
        "垂向速度",
        "升降率",
        "爬升率",
        "升降速度",
        "爬升速度",
        "速度_天向",
        "天向速度",
        "vertical speed",
        "vertical_speed",
        "climb",
        "vz",
    ),
}

_ROTATION_AXIS_TOKENS: Mapping[str, tuple[str, ...]] = {
    "pitch": ("俯仰角", "pitch", "theta"),
    "pitch_rate": ("俯仰角速度", "俯仰速率", "pitch rate", "pitch_rate"),
    "roll": ("滚转角", "横滚角", "roll", "phi"),
    "roll_rate": ("滚转角速度", "横滚角速度", "滚转速率", "roll rate", "roll_rate"),
    "yaw": ("偏航角", "航向角", "真航向", "航向", "yaw", "psi", "heading"),
    "yaw_rate": ("偏航角速度", "航向角速度", "航向速率", "yaw rate", "yaw_rate"),
}

_MATCH_ORDER = (
    "pitch_rate",
    "roll_rate",
    "yaw_rate",
    "vertical_speed",
    "acceleration",
    "altitude",
    "pitch",
    "roll",
    "yaw",
    "speed",
)


@dataclass(frozen=True, slots=True)
class RigidBodyStateMapping:
    """Feature groups participating in rigid-body vehicle residuals."""

    speed: tuple[str, ...] = ()
    acceleration: tuple[str, ...] = ()
    altitude: tuple[str, ...] = ()
    vertical_speed: tuple[str, ...] = ()
    pitch: tuple[str, ...] = ()
    pitch_rate: tuple[str, ...] = ()
    roll: tuple[str, ...] = ()
    roll_rate: tuple[str, ...] = ()
    yaw: tuple[str, ...] = ()
    yaw_rate: tuple[str, ...] = ()

    def enabled_residuals(self) -> tuple[str, ...]:
        enabled: list[str] = []
        if self.speed and self.acceleration:
            enabled.append("translation")
        if self.altitude and self.vertical_speed:
            enabled.append("vertical")
        if (self.pitch and self.pitch_rate) or (self.roll and self.roll_rate) or (self.yaw and self.yaw_rate):
            enabled.append("rotation")
        return tuple(enabled)

    def missing_requirements(self) -> dict[str, tuple[str, ...]]:
        missing: dict[str, tuple[str, ...]] = {}
        if not (self.speed and self.acceleration):
            needs: list[str] = []
            if not self.speed:
                needs.append("speed")
            if not self.acceleration:
                needs.append("acceleration")
            missing["translation"] = tuple(needs)
        if not (self.altitude and self.vertical_speed):
            needs = []
            if not self.altitude:
                needs.append("altitude")
            if not self.vertical_speed:
                needs.append("vertical_speed")
            missing["vertical"] = tuple(needs)
        rotation_missing: list[str] = []
        if not (self.pitch and self.pitch_rate):
            rotation_missing.append("pitch/pitch_rate")
        if not (self.roll and self.roll_rate):
            rotation_missing.append("roll/roll_rate")
        if not (self.yaw and self.yaw_rate):
            rotation_missing.append("yaw/yaw_rate")
        if len(rotation_missing) == 3:
            missing["rotation"] = tuple(rotation_missing)
        return missing


@dataclass(frozen=True, slots=True)
class RigidBodyPhysicsDiagnostics:
    """Human-readable rigid-body residual availability summary."""

    enabled_residuals: tuple[str, ...]
    missing_requirements: Mapping[str, tuple[str, ...]]
    uses_latent_fallback: bool


@dataclass(frozen=True, slots=True)
class RigidBodyMappingDiagnostics:
    """Per-group feature/label matching diagnostics for rigid-body constraints."""

    feature_labels: Mapping[str, str]
    groups: Mapping[str, tuple[Mapping[str, str], ...]]
    unmatched_features: tuple[Mapping[str, str], ...]
    enabled_residuals: tuple[str, ...]
    missing_requirements: Mapping[str, tuple[str, ...]]


def build_rigid_body_state_mapping(
    feature_names: tuple[str, ...],
    *,
    field_labels: Mapping[str, str] | None = None,
) -> RigidBodyStateMapping:
    """Resolve feature names onto a rigid-body vehicle state mapping."""

    labels = field_labels or {}
    grouped = {name: [] for name in _TRANSLATION_GROUP_TOKENS}
    grouped.update({name: [] for name in _ROTATION_AXIS_TOKENS})
    for feature_name in feature_names:
        label = labels.get(feature_name) or labels.get(_raw_field_name(feature_name)) or feature_name
        normalized = label.lower()
        for group_name in _MATCH_ORDER:
            tokens = _TRANSLATION_GROUP_TOKENS.get(group_name) or _ROTATION_AXIS_TOKENS.get(group_name)
            assert tokens is not None
            if any(token.lower() in normalized for token in tokens):
                grouped[group_name].append(feature_name)
                break
    return RigidBodyStateMapping(
        speed=tuple(grouped["speed"]),
        acceleration=tuple(grouped["acceleration"]),
        altitude=tuple(grouped["altitude"]),
        vertical_speed=tuple(grouped["vertical_speed"]),
        pitch=tuple(grouped["pitch"]),
        pitch_rate=tuple(grouped["pitch_rate"]),
        roll=tuple(grouped["roll"]),
        roll_rate=tuple(grouped["roll_rate"]),
        yaw=tuple(grouped["yaw"]),
        yaw_rate=tuple(grouped["yaw_rate"]),
    )


def inspect_rigid_body_physics(
    feature_names: tuple[str, ...],
    *,
    field_labels: Mapping[str, str] | None = None,
    mode: str = "feature_first_with_latent_fallback",
) -> RigidBodyPhysicsDiagnostics:
    """Describe which rigid-body residuals can run on the current feature schema."""

    mapping = build_rigid_body_state_mapping(feature_names, field_labels=field_labels)
    enabled = mapping.enabled_residuals()
    return RigidBodyPhysicsDiagnostics(
        enabled_residuals=enabled,
        missing_requirements=mapping.missing_requirements(),
        uses_latent_fallback=(not enabled and mode in {"feature_first_with_latent_fallback", "latent_only"}),
    )


def build_rigid_body_mapping_diagnostics(
    feature_names: tuple[str, ...],
    *,
    field_labels: Mapping[str, str] | None = None,
    mode: str = "feature_first_with_latent_fallback",
) -> RigidBodyMappingDiagnostics:
    """Build a reader-friendly summary of rigid-body field matching."""

    labels = _resolve_feature_labels(feature_names, field_labels=field_labels)
    mapping = build_rigid_body_state_mapping(feature_names, field_labels=field_labels)
    groups = {
        "speed": _group_rows(mapping.speed, labels),
        "acceleration": _group_rows(mapping.acceleration, labels),
        "altitude": _group_rows(mapping.altitude, labels),
        "vertical_speed": _group_rows(mapping.vertical_speed, labels),
        "pitch": _group_rows(mapping.pitch, labels),
        "pitch_rate": _group_rows(mapping.pitch_rate, labels),
        "roll": _group_rows(mapping.roll, labels),
        "roll_rate": _group_rows(mapping.roll_rate, labels),
        "yaw": _group_rows(mapping.yaw, labels),
        "yaw_rate": _group_rows(mapping.yaw_rate, labels),
    }
    matched = {item["feature_name"] for rows in groups.values() for item in rows}
    unmatched = tuple(
        {"feature_name": feature_name, "label": labels[feature_name]}
        for feature_name in feature_names
        if feature_name not in matched
    )
    physics = inspect_rigid_body_physics(feature_names, field_labels=field_labels, mode=mode)
    return RigidBodyMappingDiagnostics(
        feature_labels=labels,
        groups=groups,
        unmatched_features=unmatched,
        enabled_residuals=physics.enabled_residuals,
        missing_requirements=physics.missing_requirements,
    )


def _resolve_feature_labels(
    feature_names: tuple[str, ...],
    *,
    field_labels: Mapping[str, str] | None,
) -> dict[str, str]:
    labels = field_labels or {}
    return {
        feature_name: labels.get(feature_name) or labels.get(_raw_field_name(feature_name)) or feature_name
        for feature_name in feature_names
    }


def _group_rows(
    group_features: tuple[str, ...],
    labels: Mapping[str, str],
) -> tuple[Mapping[str, str], ...]:
    return tuple(
        {"feature_name": feature_name, "label": labels[feature_name]}
        for feature_name in group_features
    )


def _raw_field_name(feature_name: str) -> str:
    return feature_name.rsplit(".", maxsplit=1)[-1]
