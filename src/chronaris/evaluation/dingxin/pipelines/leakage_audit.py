"""Leakage checks for task evaluation private proxy tasks."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


IDENTITY_FEATURE_TOKENS = (
    "sample_id",
    "raw_sample_id",
    "sortie_id",
    "flight_sortie_id",
    "pilot_id",
    "view_id",
)
TEMPORAL_FEATURE_TOKENS = (
    "timestamp",
    "raw_timestamp",
    "start_offset",
    "end_offset",
    "window_index",
    "window_position",
    "window_fraction",
)
DEFAULT_DERIVED_FEATURE_TOKENS = (
    "vehicle_proxy_score",
    "physiology_proxy_score",
    "label_quantile",
    "label_threshold",
    "class_code",
    "class_label",
    "y_label",
    "target_window",
    "next_window_raw",
    "future_",
)


@dataclass(frozen=True, slots=True)
class LabelFeatureOverlapAudit:
    task_name: str
    label_source_fields: list[str]
    input_feature_fields: list[str]
    derived_input_features: list[str]
    forbidden_feature_families: list[str]
    direct_overlap_fields: list[str]
    deterministic_overlap_features: list[str]
    temporal_identity_features: list[str]
    leakage_safe: bool
    audit_status: str
    audit_message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LabelFeatureLeakageError(ValueError):
    """Raised when leakage-safe mode sees a forbidden input feature."""


def audit_label_feature_overlap(
    *,
    task_name: str,
    label_source_fields: Sequence[str],
    input_feature_fields: Sequence[str],
    derived_input_features: Sequence[str] = (),
    forbidden_feature_families: Sequence[str] = (),
    leakage_safe: bool = True,
    fail_on_leakage: bool = True,
) -> LabelFeatureOverlapAudit:
    """Audit whether a task input feature set leaks label-source information."""

    labels = sorted({str(field) for field in label_source_fields if str(field)})
    features = sorted({str(field) for field in input_feature_fields if str(field)})
    derived = sorted({str(field) for field in derived_input_features if str(field)})
    forbidden = sorted({str(field) for field in forbidden_feature_families if str(field)})
    label_tokens = [_normalize_token(field) for field in labels]
    direct = [
        feature
        for feature in features
        if any(token and token in _normalize_token(feature) for token in label_tokens)
    ]
    deterministic_tokens = tuple(_normalize_token(token) for token in (*DEFAULT_DERIVED_FEATURE_TOKENS, *derived, *forbidden))
    deterministic = [
        feature
        for feature in features
        if any(token and token in _normalize_token(feature) for token in deterministic_tokens)
    ]
    temporal_tokens = tuple(_normalize_token(token) for token in (*IDENTITY_FEATURE_TOKENS, *TEMPORAL_FEATURE_TOKENS))
    temporal_identity = [
        feature
        for feature in features
        if any(token and token in _normalize_token(feature) for token in temporal_tokens)
    ]
    issues: list[str] = []
    if direct:
        issues.append(f"direct label-source overlap: {', '.join(direct[:8])}")
    if deterministic:
        issues.append(f"deterministic label-derived features: {', '.join(deterministic[:8])}")
    if temporal_identity:
        issues.append(f"temporal or identity features: {', '.join(temporal_identity[:8])}")
    audit_status = "pass" if not issues else "failed"
    audit_message = "leakage-safe audit passed" if not issues else "; ".join(issues)
    result = LabelFeatureOverlapAudit(
        task_name=task_name,
        label_source_fields=labels,
        input_feature_fields=features,
        derived_input_features=derived,
        forbidden_feature_families=forbidden,
        direct_overlap_fields=direct,
        deterministic_overlap_features=deterministic,
        temporal_identity_features=temporal_identity,
        leakage_safe=bool(leakage_safe and not issues),
        audit_status=audit_status,
        audit_message=audit_message,
    )
    if leakage_safe and fail_on_leakage and issues:
        raise LabelFeatureLeakageError(f"{task_name} leakage audit failed: {audit_message}")
    return result


def write_label_feature_overlap_audit(
    audits: Sequence[LabelFeatureOverlapAudit],
    *,
    output_root: str | Path,
) -> tuple[str, str]:
    """Write JSON and CSV audit tables."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [audit.to_dict() for audit in audits]
    json_path = root / "label_feature_overlap_audit.json"
    csv_path = root / "label_feature_overlap_audit.csv"
    json_path.write_text(json.dumps({"audits": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = (
        "task_name",
        "label_source_fields",
        "input_feature_fields",
        "derived_input_features",
        "forbidden_feature_families",
        "direct_overlap_fields",
        "deterministic_overlap_features",
        "temporal_identity_features",
        "leakage_safe",
        "audit_status",
        "audit_message",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return str(json_path), str(csv_path)


def _normalize_token(value: object) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value
