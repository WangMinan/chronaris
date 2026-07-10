"""Stable contracts for fixed-data application task construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ApplicationContextRecord:
    """One 30-second context and its optional future response window."""

    context_id: str
    sortie_id: str
    view_id: str
    pilot_id: int
    start_window_index: int
    end_window_index: int
    source_sample_ids: tuple[str, ...]
    end_sample_id: str
    target_sample_id: str | None
    start_offset_ms: int
    end_offset_ms: int
    classification_eligible: bool
    response_eligible: bool
    classification_exclusion_reason: str | None = None
    response_exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "source_sample_ids": list(self.source_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class FieldRoleRecord:
    """One source feature and its task/evidence role."""

    sortie_id: str
    stream_kind: str
    feature_name: str
    measurement: str
    source_field: str
    display_label: str | None
    unit_hint: str | None
    semantic_category: str
    semantic_key: str | None
    metadata_status: str
    observed_window_count: int
    total_window_count: int
    observed_point_count: int
    selected_for_maneuver_label: bool
    selected_for_response_target: bool
    allowed_in_maneuver_input: bool
    exclusion_reason: str | None = None

    @property
    def valid_window_ratio(self) -> float:
        if self.total_window_count <= 0:
            return 0.0
        return float(self.observed_window_count / self.total_window_count)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "valid_window_ratio": self.valid_window_ratio,
        }


@dataclass(frozen=True, slots=True)
class OuterFoldDefinition:
    """One leakage-safe outer evaluation fold."""

    fold_id: str
    split_strategy: str
    held_out_group: str
    train_group_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]
    classification_train_context_ids: tuple[str, ...]
    classification_test_context_ids: tuple[str, ...]
    response_train_context_ids: tuple[str, ...]
    response_test_context_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class FoldTaskLabelResult:
    """Fold-specific labels and fit metadata for both real-data tasks."""

    fold_id: str
    split_strategy: str
    status: str
    label_rows: tuple[Mapping[str, object], ...]
    threshold_rows: tuple[Mapping[str, object], ...]
    selected_response_fields: tuple[str, ...]
    fit_sample_hashes: Mapping[str, str]
    warnings: tuple[str, ...] = ()


def stable_sample_hash(sample_ids: Sequence[str]) -> str:
    """Hash a set-like sample-id collection in stable sorted order."""

    payload = json.dumps(sorted(str(value) for value in sample_ids), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
