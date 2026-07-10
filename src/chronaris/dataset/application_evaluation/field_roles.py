"""Field-role classification for fixed-data labels and leakage exclusion."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Mapping, Sequence

import pandas as pd

from chronaris.dataset.application_evaluation.contracts import FieldRoleRecord


_OWN_AIRCRAFT_MARKERS = ("载机", "本机")
_INVALID_LABEL_MARKERS = (
    "有效性",
    "精度",
    "方差",
    "协方差",
    "置信度",
    "编号",
    "标志",
    "消息时间",
)
_CONTROL_MARKERS = ("驾驶杆", "操纵杆", "杆量", "脚蹬", "油门", "舵面", "方向舵", "升降舵", "副翼")


def build_field_role_manifest(
    records: pd.DataFrame,
    *,
    vehicle_labels_by_sortie: Mapping[str, Mapping[str, str]],
) -> tuple[FieldRoleRecord, ...]:
    """Classify every observed raw-summary field and select safe task roles."""

    vehicle_coverage = _collect_coverage(records, stats_column="raw_vehicle_stats")
    physiology_coverage = _collect_coverage(records, stats_column="raw_physiology_stats")
    roles: list[FieldRoleRecord] = []

    for (sortie_id, feature_name), coverage in sorted(vehicle_coverage.items()):
        measurement, source_field = _split_feature_name(feature_name)
        display_label = vehicle_labels_by_sortie.get(sortie_id, {}).get(feature_name)
        category, semantic_key = _classify_vehicle_field(display_label)
        metadata_status = "resolved" if display_label else "unresolved"
        roles.append(
            FieldRoleRecord(
                sortie_id=sortie_id,
                stream_kind="vehicle",
                feature_name=feature_name,
                measurement=measurement,
                source_field=source_field,
                display_label=display_label,
                unit_hint=_unit_hint(display_label),
                semantic_category=category,
                semantic_key=semantic_key,
                metadata_status=metadata_status,
                observed_window_count=coverage["observed_window_count"],
                total_window_count=coverage["total_window_count"],
                observed_point_count=coverage["observed_point_count"],
                selected_for_maneuver_label=False,
                selected_for_response_target=False,
                allowed_in_maneuver_input=True,
            )
        )

    for (sortie_id, feature_name), coverage in sorted(physiology_coverage.items()):
        measurement, source_field = _split_feature_name(feature_name)
        category = _classify_physiology_field(feature_name)
        roles.append(
            FieldRoleRecord(
                sortie_id=sortie_id,
                stream_kind="physiology",
                feature_name=feature_name,
                measurement=measurement,
                source_field=source_field,
                display_label=feature_name,
                unit_hint=None,
                semantic_category=category,
                semantic_key=feature_name.lower() if category in {"eeg", "spo2"} else None,
                metadata_status="name_derived",
                observed_window_count=coverage["observed_window_count"],
                total_window_count=coverage["total_window_count"],
                observed_point_count=coverage["observed_point_count"],
                selected_for_maneuver_label=False,
                selected_for_response_target=category in {"eeg", "spo2"},
                allowed_in_maneuver_input=True,
            )
        )

    return _select_deduplicated_maneuver_fields(roles)


def selected_maneuver_roles(roles: Sequence[FieldRoleRecord]) -> tuple[FieldRoleRecord, ...]:
    return tuple(role for role in roles if role.selected_for_maneuver_label)


def selected_response_roles(roles: Sequence[FieldRoleRecord]) -> tuple[FieldRoleRecord, ...]:
    return tuple(role for role in roles if role.selected_for_response_target)


def _collect_coverage(
    records: pd.DataFrame,
    *,
    stats_column: str,
) -> dict[tuple[str, str], dict[str, int]]:
    if stats_column not in records:
        raise ValueError(f"records are missing {stats_column}")
    totals = records.groupby("sortie_id").size().astype(int).to_dict()
    observed_windows: dict[tuple[str, str], int] = defaultdict(int)
    observed_points: dict[tuple[str, str], int] = defaultdict(int)
    known_fields: dict[str, set[str]] = defaultdict(set)
    for row in records.itertuples(index=False):
        sortie_id = str(row.sortie_id)
        stats = getattr(row, stats_column)
        feature_map = stats.get("features", {}) if isinstance(stats, Mapping) else {}
        for feature_name, payload in feature_map.items():
            key = (sortie_id, str(feature_name))
            known_fields[sortie_id].add(str(feature_name))
            if not isinstance(payload, Mapping):
                continue
            count = int(payload.get("count") or 0)
            if count > 0:
                observed_windows[key] += 1
                observed_points[key] += count
    coverage: dict[tuple[str, str], dict[str, int]] = {}
    for sortie_id, fields in known_fields.items():
        for feature_name in fields:
            key = (sortie_id, feature_name)
            coverage[key] = {
                "observed_window_count": int(observed_windows.get(key, 0)),
                "total_window_count": int(totals.get(sortie_id, 0)),
                "observed_point_count": int(observed_points.get(key, 0)),
            }
    return coverage


def _classify_vehicle_field(display_label: str | None) -> tuple[str, str | None]:
    if not display_label:
        return "unresolved", None
    compact = _compact(display_label)
    if any(marker in compact for marker in _INVALID_LABEL_MARKERS):
        return "metadata_or_quality", None
    is_control = any(marker in compact for marker in _CONTROL_MARKERS)
    is_own_aircraft = any(marker in compact for marker in _OWN_AIRCRAFT_MARKERS)
    if "目标" in compact and not is_own_aircraft:
        return "other_aircraft_or_target", None
    if not is_own_aircraft and not is_control:
        return "non_own_aircraft", None

    category: str | None = None
    if "过载" in compact:
        category = "normal_load"
    elif "加速度" in compact:
        category = "acceleration"
    elif "角速度" in compact or "角速率" in compact:
        if "俯仰" in compact:
            category = "pitch_rate"
        elif "横滚" in compact or "滚转" in compact:
            category = "roll_rate"
        elif "偏航" in compact or "航向" in compact:
            category = "yaw_rate"
    elif "俯仰" in compact:
        category = "pitch"
    elif "横滚" in compact or "滚转" in compact:
        category = "roll"
    elif "偏航" in compact or "真航向" in compact or "航向" in compact:
        category = "yaw_or_heading"
    elif "速度" in compact:
        category = "speed"
    elif is_control:
        category = "control_input"
    if category is None:
        return "other_own_aircraft", None
    return category, _semantic_key(category, compact)


def _semantic_key(category: str, compact_label: str) -> str:
    axis = ""
    for marker, value in (
        ("北向", "north"),
        ("西向", "west"),
        ("天向", "up"),
        ("纵向", "longitudinal"),
        ("侧向", "lateral"),
        ("法向", "normal"),
    ):
        if marker in compact_label:
            axis = value
            break
    if category == "control_input":
        for marker, value in (
            ("驾驶杆", "stick"),
            ("操纵杆", "stick"),
            ("脚蹬", "pedal"),
            ("油门", "throttle"),
            ("方向舵", "rudder"),
            ("升降舵", "elevator"),
            ("副翼", "aileron"),
        ):
            if marker in compact_label:
                axis = value
                break
    return category if not axis else f"{category}_{axis}"


def _select_deduplicated_maneuver_fields(
    roles: Sequence[FieldRoleRecord],
) -> tuple[FieldRoleRecord, ...]:
    candidates: dict[tuple[str, str], list[FieldRoleRecord]] = defaultdict(list)
    for role in roles:
        if role.stream_kind == "vehicle" and role.semantic_key:
            candidates[(role.sortie_id, role.semantic_key)].append(role)
    selected_names: set[tuple[str, str]] = set()
    for key, key_roles in candidates.items():
        selected = sorted(
            key_roles,
            key=lambda role: (
                -role.observed_window_count,
                -role.observed_point_count,
                role.feature_name,
            ),
        )[0]
        selected_names.add((key[0], selected.feature_name))

    resolved: list[FieldRoleRecord] = []
    for role in roles:
        selected = (role.sortie_id, role.feature_name) in selected_names
        if selected:
            resolved.append(
                replace(
                    role,
                    selected_for_maneuver_label=True,
                    allowed_in_maneuver_input=False,
                    exclusion_reason="maneuver_label_source",
                )
            )
        else:
            resolved.append(role)
    return tuple(resolved)


def _classify_physiology_field(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "eeg" in lowered:
        return "eeg"
    if "spo2" in lowered or "血氧" in feature_name:
        return "spo2"
    if "heart" in lowered or "hr" in lowered or "心率" in feature_name:
        return "heart_rate"
    return "other_physiology"


def _split_feature_name(feature_name: str) -> tuple[str, str]:
    if "." not in feature_name:
        return feature_name, feature_name
    measurement, source_field = feature_name.split(".", 1)
    return measurement, source_field


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _unit_hint(display_label: str | None) -> str | None:
    if not display_label:
        return None
    tokens = re.findall(r"\[([^\[\]]+)\]", display_label)
    for token in reversed(tokens):
        if token.startswith("_") or any(marker in token for marker in ("速度", "加速度", "角度", "过载")):
            return token.lstrip("_")
    return None
