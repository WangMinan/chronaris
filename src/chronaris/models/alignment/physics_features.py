"""Feature grouping metadata for Stage F physics constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

PHYSICS_COMPONENT_KEYS = (
    "vehicle_semantic",
    "vehicle_smoothness",
    "vehicle_envelope",
    "vehicle_latent",
    "physiology_smoothness",
    "physiology_envelope",
    "physiology_pairwise",
    "physiology_spo2_delta",
    "physiology_latent",
)

VEHICLE_SEMANTIC_TOKENS: Mapping[str, tuple[str, ...]] = {
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
        "vel",
        "velocity",
    ),
    "acceleration": (
        "加速度",
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
    "altitude": ("高度", "海拔", "气压高", "雷达高", "altitude", "height", "alt"),
    "vertical_speed": (
        "垂直速度",
        "升降率",
        "爬升率",
        "vertical speed",
        "vertical_speed",
        "climb",
        "vz",
    ),
    "attitude": (
        "俯仰角",
        "滚转角",
        "横滚角",
        "偏航角",
        "姿态角",
        "pitch",
        "roll",
        "yaw",
        "attitude",
    ),
    "angular_rate": (
        "俯仰角速度",
        "滚转角速度",
        "横滚角速度",
        "偏航角速度",
        "角速度",
        "pitch rate",
        "roll rate",
        "yaw rate",
        "angular rate",
        "gyro",
    ),
}
VEHICLE_SEMANTIC_MATCH_ORDER = (
    "acceleration",
    "vertical_speed",
    "angular_rate",
    "speed",
    "altitude",
    "attitude",
)

EEG_PAIR_SUFFIXES = (
    ("af3", "af4"),
    ("f3", "f4"),
    ("f7", "f8"),
    ("fp1", "fp2"),
    ("o1", "o2"),
    ("p3", "p4"),
    ("t7", "t8"),
)


@dataclass(frozen=True, slots=True)
class StageFVehicleFeatureGroups:
    """Vehicle feature names grouped by available physical semantics."""

    speed: tuple[str, ...] = ()
    acceleration: tuple[str, ...] = ()
    altitude: tuple[str, ...] = ()
    vertical_speed: tuple[str, ...] = ()
    attitude: tuple[str, ...] = ()
    angular_rate: tuple[str, ...] = ()
    other_numeric: tuple[str, ...] = ()

    def enabled_semantic_pairs(self) -> tuple[str, ...]:
        enabled: list[str] = []
        if self.speed and self.acceleration:
            enabled.append("speed_acceleration")
        if self.altitude and self.vertical_speed:
            enabled.append("altitude_vertical_speed")
        if self.attitude and self.angular_rate:
            enabled.append("attitude_angular_rate")
        return tuple(enabled)


@dataclass(frozen=True, slots=True)
class StageFPhysiologyFeatureGroups:
    """Physiology feature names grouped for Stage F constraints."""

    eeg: tuple[str, ...] = ()
    spo2: tuple[str, ...] = ()
    eeg_pairs: tuple[tuple[str, str], ...] = ()
    other_numeric: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageFPhysicsContext:
    """Auxiliary statistics and semantic groups for Stage F physics losses."""

    vehicle_groups: StageFVehicleFeatureGroups = field(default_factory=StageFVehicleFeatureGroups)
    physiology_groups: StageFPhysiologyFeatureGroups = field(default_factory=StageFPhysiologyFeatureGroups)
    vehicle_envelope_lower: torch.Tensor | None = None
    vehicle_envelope_upper: torch.Tensor | None = None
    physiology_envelope_lower: torch.Tensor | None = None
    physiology_envelope_upper: torch.Tensor | None = None
    vehicle_denormalize_mean: torch.Tensor | None = None
    vehicle_denormalize_std: torch.Tensor | None = None
    physiology_denormalize_mean: torch.Tensor | None = None
    physiology_denormalize_std: torch.Tensor | None = None
    field_labels: Mapping[str, str] = field(default_factory=dict)


def build_vehicle_feature_groups(
    feature_names: tuple[str, ...],
    *,
    field_labels: Mapping[str, str] | None = None,
) -> StageFVehicleFeatureGroups:
    """Group vehicle features by semantic labels when metadata proves them."""

    labels = field_labels or {}
    grouped: dict[str, list[str]] = {key: [] for key in VEHICLE_SEMANTIC_TOKENS}
    other: list[str] = []
    for feature_name in feature_names:
        label = labels.get(feature_name) or labels.get(raw_field_name(feature_name)) or feature_name
        matched = False
        normalized_label = label.lower()
        for group_name in VEHICLE_SEMANTIC_MATCH_ORDER:
            tokens = VEHICLE_SEMANTIC_TOKENS[group_name]
            if any(token.lower() in normalized_label for token in tokens):
                grouped[group_name].append(feature_name)
                matched = True
                break
        if not matched:
            other.append(feature_name)
    return StageFVehicleFeatureGroups(
        speed=tuple(grouped["speed"]),
        acceleration=tuple(grouped["acceleration"]),
        altitude=tuple(grouped["altitude"]),
        vertical_speed=tuple(grouped["vertical_speed"]),
        attitude=tuple(grouped["attitude"]),
        angular_rate=tuple(grouped["angular_rate"]),
        other_numeric=tuple(other),
    )


def build_physiology_feature_groups(feature_names: tuple[str, ...]) -> StageFPhysiologyFeatureGroups:
    """Group physiology features into EEG/SPO2 constraints."""

    eeg = tuple(name for name in feature_names if name.lower().startswith("eeg."))
    spo2 = tuple(name for name in feature_names if "spo2" in name.lower())
    assigned = set(eeg) | set(spo2)
    suffix_index = {feature_suffix(name): name for name in eeg}
    pairs: list[tuple[str, str]] = []
    for left_suffix, right_suffix in EEG_PAIR_SUFFIXES:
        left = suffix_index.get(left_suffix)
        right = suffix_index.get(right_suffix)
        if left is not None and right is not None:
            pairs.append((left, right))
    other = tuple(name for name in feature_names if name not in assigned)
    return StageFPhysiologyFeatureGroups(eeg=eeg, spo2=spo2, eeg_pairs=tuple(pairs), other_numeric=other)


def raw_field_name(feature_name: str) -> str:
    return feature_name.rsplit(".", maxsplit=1)[-1]


def feature_suffix(feature_name: str) -> str:
    return raw_field_name(feature_name).lower()
