"""Minimal E0 experiment-input adaptation from windowed sortie samples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from chronaris.schema.models import AlignedPoint, DatasetBuildResult, SampleWindow, ScalarValue, StreamKind


@dataclass(frozen=True, slots=True)
class E0InputConfig:
    """Controls how E0 experiment samples are generated."""

    physiology_measurements: tuple[str, ...] | None = None
    vehicle_measurements: tuple[str, ...] | None = None
    include_windows_without_both_streams: bool = False


@dataclass(frozen=True, slots=True)
class NumericStreamMatrix:
    """A numeric, model-ready representation for one stream within one window."""

    stream_kind: StreamKind
    point_count: int
    feature_names: tuple[str, ...]
    point_offsets_ms: tuple[int, ...]
    point_measurements: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    dropped_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class E0ExperimentSample:
    """A minimal model-ready sample produced from one SampleWindow."""

    sample_id: str
    sortie_id: str
    start_offset_ms: int
    end_offset_ms: int
    physiology: NumericStreamMatrix
    vehicle: NumericStreamMatrix
    notes: tuple[str, ...] = field(default_factory=tuple)


def build_e0_experiment_samples(
    result: DatasetBuildResult,
    *,
    config: E0InputConfig | None = None,
) -> tuple[E0ExperimentSample, ...]:
    """Convert windowed sortie data into minimal numeric experiment samples."""

    active_config = config or E0InputConfig()
    samples: list[E0ExperimentSample] = []
    filtered_points_by_window = [
        (
            _filter_points(window.physiology_points, active_config.physiology_measurements),
            _filter_points(window.vehicle_points, active_config.vehicle_measurements),
        )
        for window in result.windows
    ]
    physiology_feature_names, physiology_dropped_fields = _discover_numeric_features(
        tuple(point for physiology_points, _vehicle_points in filtered_points_by_window for point in physiology_points)
    )
    vehicle_feature_names, vehicle_dropped_fields = _discover_numeric_features(
        tuple(point for _physiology_points, vehicle_points in filtered_points_by_window for point in vehicle_points)
    )

    for window, (physiology_points, vehicle_points) in zip(result.windows, filtered_points_by_window):

        if (
            not active_config.include_windows_without_both_streams
            and (not physiology_points or not vehicle_points)
        ):
            continue

        physiology_matrix = build_numeric_stream_matrix(
            StreamKind.PHYSIOLOGY,
            physiology_points,
            feature_names=physiology_feature_names,
            dropped_fields=physiology_dropped_fields,
        )
        vehicle_matrix = build_numeric_stream_matrix(
            StreamKind.VEHICLE,
            vehicle_points,
            feature_names=vehicle_feature_names,
            dropped_fields=vehicle_dropped_fields,
        )
        notes = []
        if not physiology_matrix.feature_names:
            notes.append("physiology stream has no numeric features after filtering")
        if not vehicle_matrix.feature_names:
            notes.append("vehicle stream has no numeric features after filtering")

        samples.append(
            E0ExperimentSample(
                sample_id=window.sample_id,
                sortie_id=window.sortie_id,
                start_offset_ms=window.start_offset_ms,
                end_offset_ms=window.end_offset_ms,
                physiology=physiology_matrix,
                vehicle=vehicle_matrix,
                notes=tuple(notes),
            )
        )

    return tuple(samples)


def build_numeric_stream_matrix(
    stream_kind: StreamKind,
    points: tuple[AlignedPoint, ...],
    *,
    feature_names: tuple[str, ...] | None = None,
    dropped_fields: tuple[str, ...] | None = None,
) -> NumericStreamMatrix:
    """Convert one stream's aligned points into a numeric feature matrix."""

    if not points:
        return NumericStreamMatrix(
            stream_kind=stream_kind,
            point_count=0,
            feature_names=feature_names or (),
            point_offsets_ms=(),
            point_measurements=(),
            values=(),
            dropped_fields=dropped_fields or (),
        )

    if feature_names is None or dropped_fields is None:
        feature_names, dropped_fields = _discover_numeric_features(points)
    rows = tuple(_build_value_row(point, feature_names) for point in points)

    return NumericStreamMatrix(
        stream_kind=stream_kind,
        point_count=len(points),
        feature_names=feature_names,
        point_offsets_ms=tuple(point.offset_ms for point in points),
        point_measurements=tuple(point.point.measurement for point in points),
        values=rows,
        dropped_fields=dropped_fields,
    )


def summarize_e0_samples(samples: tuple[E0ExperimentSample, ...]) -> dict[str, int]:
    """Return a compact numeric summary for one E0 sample collection."""

    return {
        "sample_count": len(samples),
        "physiology_feature_count_max": max((len(sample.physiology.feature_names) for sample in samples), default=0),
        "vehicle_feature_count_max": max((len(sample.vehicle.feature_names) for sample in samples), default=0),
    }


def _filter_points(
    points: tuple[AlignedPoint, ...],
    measurement_whitelist: tuple[str, ...] | None,
) -> tuple[AlignedPoint, ...]:
    if measurement_whitelist is None:
        return points
    allowed = set(measurement_whitelist)
    return tuple(point for point in points if point.point.measurement in allowed)


def _discover_numeric_features(points: tuple[AlignedPoint, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    feature_names: set[str] = set()
    dropped_fields: set[str] = set()

    for aligned_point in points:
        measurement = aligned_point.point.measurement
        for field_name, value in aligned_point.point.values.items():
            qualified_name = _qualify_feature_name(measurement, field_name)
            coerced = _coerce_numeric(value)
            if coerced is None:
                dropped_fields.add(qualified_name)
            else:
                feature_names.add(qualified_name)

    return tuple(sorted(feature_names)), tuple(sorted(dropped_fields - feature_names))


def _build_value_row(
    point: AlignedPoint,
    feature_names: tuple[str, ...],
) -> tuple[float, ...]:
    measurement = point.point.measurement
    qualified_values = {
        _qualify_feature_name(measurement, field_name): _coerce_numeric(value)
        for field_name, value in point.point.values.items()
    }
    return tuple(
        qualified_values.get(feature_name, math.nan)
        if qualified_values.get(feature_name) is not None
        else math.nan
        for feature_name in feature_names
    )


def _qualify_feature_name(measurement: str, field_name: str) -> str:
    return f"{measurement}.{field_name}"


def _coerce_numeric(value: ScalarValue) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or ";" in stripped or ":" in stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None
