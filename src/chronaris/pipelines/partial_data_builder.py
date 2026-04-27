"""Vehicle-only partial-data builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from chronaris.features.experiment_input import NumericStreamMatrix, build_numeric_stream_matrix
from chronaris.pipelines.partial_data_contracts import (
    PartialDataBuildResult,
    PartialDataConfig,
    PartialDataEntry,
    PartialDataManifest,
    PartialMeasurementMetadata,
    PartialPointChunk,
    PartialPointChunkProvider,
    PartialPointProvider,
    PartialStreamSample,
    PartialVehicleMetadataProvider,
    concrete_measurements,
    parse_required_utc,
    write_jsonl,
)
from chronaris.schema.models import AlignedPoint, RawPoint, StreamKind


@dataclass(slots=True)
class PartialDataBuilder:
    """Build repo-local single-stream assets from standardized partial entries."""

    config: PartialDataConfig = field(default_factory=PartialDataConfig)
    point_provider: PartialPointProvider | None = None
    chunk_provider: PartialPointChunkProvider | None = None
    metadata_provider: PartialVehicleMetadataProvider | None = None

    def run(
        self,
        entries: Sequence[PartialDataEntry],
        *,
        output_root: str | Path,
    ) -> PartialDataBuildResult:
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "partial_data_manifest.jsonl"
        window_manifest_path = output_path / "vehicle_only_window_manifest.jsonl"
        feature_bundle_path = output_path / "vehicle_only_feature_bundle.npz"

        manifest_rows: list[dict[str, object]] = []
        window_rows: list[dict[str, object]] = []
        stream_matrices: list[NumericStreamMatrix] = []
        built_samples: list[PartialStreamSample] = []

        for entry in entries:
            row = entry.to_dict()
            status = "manifest_only"
            reason = None
            output_sample_count = 0
            input_point_count = 0
            retained_point_count = 0
            measurement_metadata = _default_measurement_metadata(entry)

            if entry.stream_kind == StreamKind.VEHICLE.value:
                try:
                    measurement_metadata = self._resolve_measurement_metadata(entry)
                    (
                        status,
                        reason,
                        output_sample_count,
                        input_point_count,
                        retained_point_count,
                    ) = self._build_vehicle_entry(
                        entry,
                        metadata_by_measurement=measurement_metadata,
                        stream_matrices=stream_matrices,
                        built_samples=built_samples,
                        window_rows=window_rows,
                    )
                except Exception as exc:
                    status = "provider_error"
                    reason = str(exc)

            row.update(
                {
                    "builder_status": status,
                    "builder_reason": reason,
                    "output_sample_count": output_sample_count,
                    "input_point_count": input_point_count,
                    "retained_point_count": retained_point_count,
                    "measurement_statuses": [
                        metadata.to_dict() for metadata in measurement_metadata.values()
                    ],
                    "max_points_per_field_per_window": (
                        self.config.max_points_per_field_per_window
                    ),
                }
            )
            manifest_rows.append(row)

        write_jsonl(manifest_path, manifest_rows)
        write_jsonl(window_manifest_path, window_rows)

        final_feature_bundle_path: str | None = None
        if stream_matrices:
            _write_vehicle_feature_bundle(feature_bundle_path, stream_matrices, built_samples)
            final_feature_bundle_path = str(feature_bundle_path)

        manifest = PartialDataManifest(
            export_version=self.config.export_version,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            entry_count=len(entries),
            built_entry_count=sum(1 for row in manifest_rows if row["builder_status"] == "built"),
            skipped_entry_count=sum(1 for row in manifest_rows if row["builder_status"] != "built"),
        )
        return PartialDataBuildResult(
            manifest=manifest,
            manifest_path=str(manifest_path),
            window_manifest_path=str(window_manifest_path),
            feature_bundle_path=final_feature_bundle_path,
            built_samples=tuple(built_samples),
        )

    def _resolve_measurement_metadata(
        self,
        entry: PartialDataEntry,
    ) -> Mapping[str, PartialMeasurementMetadata]:
        if self.metadata_provider is None:
            return _default_measurement_metadata(entry)
        metadata = dict(self.metadata_provider(entry))
        for measurement in concrete_measurements(entry):
            metadata.setdefault(
                measurement,
                PartialMeasurementMetadata(
                    measurement=measurement,
                    status="skipped",
                    reason="metadata_missing",
                ),
            )
        return metadata

    def _build_vehicle_entry(
        self,
        entry: PartialDataEntry,
        *,
        metadata_by_measurement: Mapping[str, PartialMeasurementMetadata],
        stream_matrices: list[NumericStreamMatrix],
        built_samples: list[PartialStreamSample],
        window_rows: list[dict[str, object]],
    ) -> tuple[str, str | None, int, int, int]:
        if self.config.window_config.duration_ms != self.config.window_config.stride_ms:
            raise ValueError(
                "vehicle-only streaming partial build requires non-overlapping windows."
            )

        available_measurements = tuple(
            measurement
            for measurement, metadata in metadata_by_measurement.items()
            if metadata.status == "available"
        )
        if not available_measurements:
            return "no_supported_measurements", "all measurements skipped by metadata", 0, 0, 0

        start_utc = parse_required_utc(entry.time_range.get("start_utc"), "time_range.start_utc")
        stop_utc = parse_required_utc(entry.time_range.get("stop_utc"), "time_range.stop_utc")
        if stop_utc <= start_utc:
            raise ValueError("time_range.stop_utc must be greater than start_utc.")

        input_point_count = 0
        retained_point_count = 0
        active_windows: dict[int, _VehicleWindowAccumulator] = {}
        original_sample_count = len(built_samples)
        chunk_iterable = self._iter_vehicle_chunks(entry, start_utc=start_utc, stop_utc=stop_utc)

        for chunk in chunk_iterable:
            chunk_points = sorted(chunk.points, key=lambda point: (point.timestamp, point.measurement))
            for point in chunk_points:
                input_point_count += 1
                offset_ms = int((point.timestamp - start_utc).total_seconds() * 1000)
                if offset_ms < 0 or point.timestamp >= stop_utc:
                    continue
                _flush_ready_vehicle_windows(
                    active_windows,
                    next_offset_ms=offset_ms,
                    entry=entry,
                    stream_matrices=stream_matrices,
                    built_samples=built_samples,
                    window_rows=window_rows,
                    config=self.config,
                )
                filtered = _filter_point_by_metadata(point, metadata_by_measurement)
                if filtered is None:
                    continue
                window_index = offset_ms // self.config.window_config.stride_ms
                window_start = window_index * self.config.window_config.stride_ms
                window_end = window_start + self.config.window_config.duration_ms
                if not (window_start <= offset_ms < window_end):
                    continue
                accumulator = active_windows.setdefault(
                    window_index,
                    _VehicleWindowAccumulator(
                        window_index=window_index,
                        start_offset_ms=window_start,
                        end_offset_ms=window_end,
                    ),
                )
                if accumulator.add(
                    filtered,
                    offset_ms=offset_ms,
                    max_points_per_field=self.config.max_points_per_field_per_window,
                ):
                    retained_point_count += 1

        _flush_all_vehicle_windows(
            active_windows,
            entry=entry,
            stream_matrices=stream_matrices,
            built_samples=built_samples,
            window_rows=window_rows,
            config=self.config,
        )
        output_sample_count = len(built_samples) - original_sample_count
        if output_sample_count > 0:
            return "built", None, output_sample_count, input_point_count, retained_point_count
        if input_point_count == 0:
            return "no_points", None, 0, input_point_count, retained_point_count
        if retained_point_count == 0:
            return "no_retained_metadata_fields", None, 0, input_point_count, retained_point_count
        return "no_windows", None, 0, input_point_count, retained_point_count

    def _iter_vehicle_chunks(
        self,
        entry: PartialDataEntry,
        *,
        start_utc: datetime,
        stop_utc: datetime,
    ) -> Iterable[PartialPointChunk]:
        del start_utc, stop_utc
        if self.chunk_provider is not None:
            return self.chunk_provider(entry)
        if self.point_provider is not None:
            return (
                PartialPointChunk(
                    start_utc=parse_required_utc(
                        entry.time_range.get("start_utc"),
                        "time_range.start_utc",
                    ),
                    stop_utc=parse_required_utc(
                        entry.time_range.get("stop_utc"),
                        "time_range.stop_utc",
                    ),
                    points=tuple(self.point_provider(entry)),
                ),
            )
        return ()


@dataclass(slots=True)
class _VehicleWindowAccumulator:
    window_index: int
    start_offset_ms: int
    end_offset_ms: int
    aligned_points: list[AlignedPoint] = field(default_factory=list)
    feature_point_counts: dict[str, int] = field(default_factory=dict)
    measurement_point_counts: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        point: RawPoint,
        *,
        offset_ms: int,
        max_points_per_field: int,
    ) -> bool:
        retained_values = {}
        for field_name, value in point.values.items():
            feature_name = f"{point.measurement}.{field_name}"
            current_count = self.feature_point_counts.get(feature_name, 0)
            if current_count >= max_points_per_field:
                continue
            self.feature_point_counts[feature_name] = current_count + 1
            retained_values[field_name] = value
        if not retained_values:
            return False
        retained_point = RawPoint(
            stream_kind=point.stream_kind,
            measurement=point.measurement,
            timestamp=point.timestamp,
            values=retained_values,
            clock_time=point.clock_time,
            timestamp_precision_digits=point.timestamp_precision_digits,
            tags=point.tags,
            source=point.source,
        )
        self.aligned_points.append(AlignedPoint(point=retained_point, offset_ms=offset_ms))
        self.measurement_point_counts[point.measurement] = (
            self.measurement_point_counts.get(point.measurement, 0) + 1
        )
        return True


def _default_measurement_metadata(
    entry: PartialDataEntry,
) -> Mapping[str, PartialMeasurementMetadata]:
    return {
        measurement: PartialMeasurementMetadata(
            measurement=measurement,
            status="available",
            allow_all_fields=True,
            reason="metadata_provider_not_configured",
        )
        for measurement in concrete_measurements(entry)
    }


def _filter_point_by_metadata(
    point: RawPoint,
    metadata_by_measurement: Mapping[str, PartialMeasurementMetadata],
) -> RawPoint | None:
    metadata = metadata_by_measurement.get(point.measurement)
    if metadata is None or metadata.status != "available":
        return None
    if metadata.allow_all_fields:
        values = dict(point.values)
    else:
        allowed = set(metadata.field_names)
        values = {
            field_name: value
            for field_name, value in point.values.items()
            if field_name in allowed
        }
    if not values:
        return None
    return RawPoint(
        stream_kind=point.stream_kind,
        measurement=point.measurement,
        timestamp=point.timestamp,
        values=values,
        clock_time=point.clock_time,
        timestamp_precision_digits=point.timestamp_precision_digits,
        tags=point.tags,
        source=point.source,
    )


def _flush_ready_vehicle_windows(
    active_windows: dict[int, _VehicleWindowAccumulator],
    *,
    next_offset_ms: int,
    entry: PartialDataEntry,
    stream_matrices: list[NumericStreamMatrix],
    built_samples: list[PartialStreamSample],
    window_rows: list[dict[str, object]],
    config: PartialDataConfig,
) -> None:
    ready_indexes = [
        window_index
        for window_index, accumulator in active_windows.items()
        if accumulator.end_offset_ms <= next_offset_ms
    ]
    for window_index in sorted(ready_indexes):
        accumulator = active_windows.pop(window_index)
        _flush_vehicle_window(
            accumulator,
            entry=entry,
            stream_matrices=stream_matrices,
            built_samples=built_samples,
            window_rows=window_rows,
            config=config,
        )


def _flush_all_vehicle_windows(
    active_windows: dict[int, _VehicleWindowAccumulator],
    *,
    entry: PartialDataEntry,
    stream_matrices: list[NumericStreamMatrix],
    built_samples: list[PartialStreamSample],
    window_rows: list[dict[str, object]],
    config: PartialDataConfig,
) -> None:
    for window_index in sorted(active_windows):
        _flush_vehicle_window(
            active_windows[window_index],
            entry=entry,
            stream_matrices=stream_matrices,
            built_samples=built_samples,
            window_rows=window_rows,
            config=config,
        )
    active_windows.clear()


def _flush_vehicle_window(
    accumulator: _VehicleWindowAccumulator,
    *,
    entry: PartialDataEntry,
    stream_matrices: list[NumericStreamMatrix],
    built_samples: list[PartialStreamSample],
    window_rows: list[dict[str, object]],
    config: PartialDataConfig,
) -> None:
    if len(accumulator.aligned_points) < config.window_config.min_vehicle_points:
        return
    matrix = build_numeric_stream_matrix(StreamKind.VEHICLE, tuple(accumulator.aligned_points))
    if not matrix.feature_names:
        return
    sample = PartialStreamSample(
        sample_id=f"{entry.sortie_id}:{accumulator.window_index:04d}",
        sortie_id=entry.sortie_id,
        stream_kind=entry.stream_kind,
        window_index=accumulator.window_index,
        start_offset_ms=accumulator.start_offset_ms,
        end_offset_ms=accumulator.end_offset_ms,
        point_count=matrix.point_count,
        feature_names=matrix.feature_names,
    )
    stream_matrices.append(matrix)
    built_samples.append(sample)
    field_point_count_max = max(accumulator.feature_point_counts.values(), default=0)
    window_rows.append(
        {
            "sortie_id": entry.sortie_id,
            "sample_id": sample.sample_id,
            "window_index": accumulator.window_index,
            "start_offset_ms": accumulator.start_offset_ms,
            "end_offset_ms": accumulator.end_offset_ms,
            "vehicle_point_count": matrix.point_count,
            "feature_count": len(matrix.feature_names),
            "feature_names": list(matrix.feature_names),
            "measurement_point_counts": dict(sorted(accumulator.measurement_point_counts.items())),
            "field_point_count_max": field_point_count_max,
            "capped_field_count": sum(
                1
                for count in accumulator.feature_point_counts.values()
                if count >= config.max_points_per_field_per_window
            ),
            "max_points_per_field": config.max_points_per_field_per_window,
        }
    )


def _write_vehicle_feature_bundle(
    path: Path,
    matrices: Sequence[NumericStreamMatrix],
    built_samples: Sequence[PartialStreamSample],
) -> None:
    feature_names = _merge_feature_names(matrix.feature_names for matrix in matrices)
    values, offsets_s, valid_mask, point_counts = _pad_stream_matrices(matrices, feature_names=feature_names)
    np.savez(
        path,
        feature_names=np.asarray(feature_names, dtype="<U128"),
        values=values,
        point_offsets_s=offsets_s,
        valid_mask=valid_mask,
        point_counts=np.asarray(point_counts, dtype=np.int64),
        sample_ids=np.asarray([sample.sample_id for sample in built_samples], dtype="<U160"),
        sortie_ids=np.asarray([sample.sortie_id for sample in built_samples], dtype="<U128"),
        window_indices=np.asarray([sample.window_index for sample in built_samples], dtype=np.int64),
        window_start_offsets_s=np.asarray(
            [sample.start_offset_ms / 1000.0 for sample in built_samples],
            dtype=np.float32,
        ),
        window_end_offsets_s=np.asarray(
            [sample.end_offset_ms / 1000.0 for sample in built_samples],
            dtype=np.float32,
        ),
    )


def _merge_feature_names(feature_name_groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in feature_name_groups:
        for feature_name in group:
            if feature_name in seen:
                continue
            seen.add(feature_name)
            merged.append(feature_name)
    return tuple(merged)


def _pad_stream_matrices(
    matrices: Sequence[NumericStreamMatrix],
    *,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    max_points = max((matrix.point_count for matrix in matrices), default=0)
    value_array = np.full(
        (len(matrices), max_points, len(feature_names)),
        np.nan,
        dtype=np.float32,
    )
    offset_array = np.full((len(matrices), max_points), np.nan, dtype=np.float32)
    valid_mask = np.zeros((len(matrices), max_points), dtype=bool)
    point_counts: list[int] = []

    for matrix_index, matrix in enumerate(matrices):
        point_counts.append(matrix.point_count)
        local_indexes = {
            feature_index[name]: local_index
            for local_index, name in enumerate(matrix.feature_names)
            if name in feature_index
        }
        for point_index in range(matrix.point_count):
            offset_array[matrix_index, point_index] = matrix.point_offsets_ms[point_index] / 1000.0
            valid_mask[matrix_index, point_index] = True
            for target_index, local_index in local_indexes.items():
                value_array[matrix_index, point_index, target_index] = matrix.values[point_index][local_index]
    return value_array, offset_array, valid_mask, tuple(point_counts)
