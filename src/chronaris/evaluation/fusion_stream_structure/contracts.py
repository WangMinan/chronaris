"""Input contracts for E3 fusion stream structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ALLOWED_METHODS = {
    "chronaris",
    "mult",
    "contiformer",
    "naive_time_sync",
}
METHOD_ALIASES = {
    "chronaris_full": "chronaris",
    "naive_sync": "naive_time_sync",
}
REQUIRED_BASE_COLUMNS = ("method_name", "sortie_id", "view_id")
TIME_FALLBACK_COLUMNS = ("time", "start_offset_ms", "window_index")
WINDOW_ID_FALLBACK_COLUMNS = ("window_id", "sample_id", "raw_sample_id")
FUSION_FEATURE_PATTERN = re.compile(r"^fusion_feature_(\d+)$")
EVIDENCE_QUADRANT = "fusion_stream_structure"
CLAIM_BOUNDARY = "unsupervised_structure_diagnostic_not_confirmed_metric"

POSTHOC_COLUMNS = {
    "maneuver_proxy_label",
    "maneuver_event_interval",
    "physio_fluctuation_interval",
    "weak_event_boundary",
    "pilot_id",
    "sample_partition",
}
META_COLUMNS = {
    "method_name",
    "sortie_id",
    "view_id",
    "window_id",
    "sample_id",
    "raw_sample_id",
    "time",
    "window_index",
    "start_offset_ms",
    "end_offset_ms",
    "status",
    *POSTHOC_COLUMNS,
}
FORBIDDEN_SIDECAR_COLUMNS = {
    "alignment_score",
    "physics_residual",
    "partial_data",
    "diag_attention_entropy",
    "diag_top_event_concentration",
    "diag_event_mask_interference",
    "diag__attention_entropy",
    "diag__top_event_concentration",
    "diag__event_mask_interference",
    "diag__causal_residual_gate",
}
FORBIDDEN_SIDECAR_PREFIXES = (
    "phys_latent_",
    "av_latent_",
    "residual__",
    "ctx__",
    "diag_",
    "diag__",
    "partial_data",
)


class ContractError(ValueError):
    """Raised when an E3 input contract is violated."""

    def __init__(self, message: str, *, status: str = "contract_violation", details: Mapping[str, object] | None = None):
        super().__init__(message)
        self.status = status
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class FusionStreamRunConfig:
    """Runtime configuration shared by the E3 preprocessing and evaluators."""

    min_T: int = 30
    low_variance_threshold: float = 1e-8
    pca_explained_variance: float = 0.95
    tol: str | int = "auto"
    m_grid: str | Sequence[int] = "auto"
    methods: tuple[str, ...] = ("chronaris", "naive_time_sync", "mult", "contiformer")
    composite_weights: Mapping[str, Mapping[str, float]] = field(default_factory=lambda: {
        "structure_stability_score": {
            "cross_view_segment_stability": 0.5,
            "nn_segment_cross_view_consistency": 0.5,
        },
        "replay_consistency_score": {
            "clap_state_replay_consistency": 0.5,
            "motif_event_consistency": 0.5,
        },
        "event_alignment_score": {
            "cp_tolerance_hit_rate": 0.25,
            "segment_event_purity": 0.25,
            "motif_event_consistency": 0.25,
            "discord_maneuver_overlap": 0.25,
        },
        "fragment_replay_recall": {
            "motif_event_consistency": 0.4,
            "nn_segment_cross_view_consistency": 0.4,
            "discord_maneuver_overlap": 0.2,
        },
    })


@dataclass(frozen=True, slots=True)
class FusionStreamContract:
    """Contract metadata for an E3 long table."""

    required_columns: tuple[str, ...] = REQUIRED_BASE_COLUMNS
    allowed_methods: frozenset[str] = frozenset(ALLOWED_METHODS)
    time_fallback_columns: tuple[str, ...] = TIME_FALLBACK_COLUMNS
    window_id_fallback_columns: tuple[str, ...] = WINDOW_ID_FALLBACK_COLUMNS
    evidence_quadrant: str = EVIDENCE_QUADRANT
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True, slots=True)
class FusionStreamRecord:
    """One method/sortie/view stream after contract validation."""

    method_name: str
    sortie_id: str
    view_id: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    window_ids: tuple[object, ...]
    times: tuple[float, ...]
    feature_name_map: Mapping[str, str] = field(default_factory=dict)
    posthoc_columns: tuple[str, ...] = ()
    status: str = "available"
    unavailable_reason: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.method_name, self.sortie_id, self.view_id)

    @property
    def T(self) -> int:
        return len(self.frame)

    @property
    def d(self) -> int:
        return len(self.feature_columns)

    def feature_matrix(self) -> np.ndarray:
        return self.frame.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    method_feature_dimensions: Mapping[str, int]


def normalize_method_name(method_name: object) -> str:
    normalized = str(method_name).strip().lower()
    normalized = METHOD_ALIASES.get(normalized, normalized)
    if normalized not in ALLOWED_METHODS:
        raise ContractError(
            f"unsupported E3 method_name: {method_name!r}",
            details={"method_name": method_name, "allowed_methods": sorted(ALLOWED_METHODS)},
        )
    return normalized


def detect_forbidden_sidecar_columns(columns: Sequence[object]) -> tuple[str, ...]:
    forbidden = []
    for column in columns:
        name = str(column)
        if name in FORBIDDEN_SIDECAR_COLUMNS or any(name.startswith(prefix) for prefix in FORBIDDEN_SIDECAR_PREFIXES):
            forbidden.append(name)
    return tuple(sorted(dict.fromkeys(forbidden)))


def detect_feature_columns(
    frame: pd.DataFrame,
    *,
    explicit_feature_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return main feature columns, preferring canonical fusion_feature_* names."""

    if explicit_feature_columns is not None:
        columns = tuple(str(column) for column in explicit_feature_columns)
    else:
        fusion_columns = []
        for column in frame.columns:
            matched = FUSION_FEATURE_PATTERN.match(str(column))
            if matched:
                fusion_columns.append((int(matched.group(1)), str(column)))
        if fusion_columns:
            columns = tuple(column for _index, column in sorted(fusion_columns))
        else:
            columns = tuple(
                str(column)
                for column in frame.columns
                if str(column) not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
            )
    forbidden = detect_forbidden_sidecar_columns(columns)
    if forbidden:
        raise ContractError(
            "forbidden sidecar columns were selected as E3 main features",
            details={"forbidden_columns": list(forbidden)},
        )
    return columns


def validate_input_frame(
    frame: pd.DataFrame,
    *,
    explicit_feature_columns: Sequence[str] | None = None,
    strict_monotonic: bool = False,
) -> ValidationResult:
    """Validate and normalize a fusion stream long table."""

    if frame.empty:
        raise ContractError("E3 input frame is empty", details={"row_count": 0})
    missing_base = [column for column in REQUIRED_BASE_COLUMNS if column not in frame.columns]
    if missing_base:
        raise ContractError(
            "E3 input frame is missing required columns",
            details={"missing_columns": missing_base},
        )

    normalized = frame.copy()
    window_source = _first_present_column(normalized, WINDOW_ID_FALLBACK_COLUMNS)
    if window_source is None:
        raise ContractError(
            "E3 input frame is missing window_id or a supported fallback",
            details={"required_any": list(WINDOW_ID_FALLBACK_COLUMNS)},
        )
    if "window_id" not in normalized.columns:
        normalized["window_id"] = normalized[window_source]

    time_source = _first_present_column(normalized, TIME_FALLBACK_COLUMNS)
    if time_source is None:
        raise ContractError(
            "E3 input frame is missing time or a supported fallback sorting column",
            details={"required_any": list(TIME_FALLBACK_COLUMNS)},
        )
    if "time" not in normalized.columns:
        normalized["time"] = normalized[time_source]

    normalized["method_name"] = [normalize_method_name(value) for value in normalized["method_name"]]
    normalized["time"] = pd.to_numeric(normalized["time"], errors="coerce")
    if normalized["time"].isna().any():
        raise ContractError("E3 time column contains non-numeric or missing values")

    feature_columns = detect_feature_columns(
        normalized,
        explicit_feature_columns=explicit_feature_columns,
    )
    if not feature_columns:
        raise ContractError("E3 input frame does not contain any fusion feature columns")
    missing_features = [column for column in feature_columns if column not in normalized.columns]
    if missing_features:
        raise ContractError(
            "explicit E3 feature columns are missing from the input frame",
            details={"missing_feature_columns": missing_features},
        )
    for column in feature_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized.loc[:, list(feature_columns)].isna().all(axis=None):
        raise ContractError("E3 fusion feature columns are all empty")

    if strict_monotonic:
        _raise_on_non_monotonic_input(normalized)

    normalized = normalized.sort_values(
        ["method_name", "sortie_id", "view_id", "time", "window_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    dimensions = {
        str(method): int(
            normalized.loc[normalized["method_name"] == method, list(feature_columns)]
            .dropna(axis=1, how="all")
            .shape[1]
        )
        for method in sorted(normalized["method_name"].unique())
    }
    return ValidationResult(
        frame=normalized,
        feature_columns=tuple(feature_columns),
        method_feature_dimensions=dimensions,
    )


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _raise_on_non_monotonic_input(frame: pd.DataFrame) -> None:
    for key, group in frame.groupby(["method_name", "sortie_id", "view_id"], sort=False):
        values = group["time"].to_numpy(dtype=float)
        if values.size > 1 and np.any(np.diff(values) < 0):
            raise ContractError(
                "E3 input time order is not monotonic within a method/sortie/view group",
                details={"group": tuple(str(item) for item in key)},
            )
