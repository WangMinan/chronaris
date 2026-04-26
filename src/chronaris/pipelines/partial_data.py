"""Partial-data manifesting and single-stream sample building."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.features.experiment_input import NumericStreamMatrix, build_numeric_stream_matrix
from chronaris.schema.models import (
    RawPoint,
    SortieBundle,
    SortieLocator,
    SortieMetadata,
    StreamKind,
    WindowConfig,
)


@dataclass(frozen=True, slots=True)
class PartialDataEntry:
    """Standardized repo-consumable record for non-dual-stream assets."""

    sortie_id: str
    source_type: str
    stream_kind: str
    data_tier: str
    time_range: Mapping[str, str | None]
    measurement_family: tuple[str, ...]
    usable_for_pretraining: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    raw_manifest_path: str | None = None
    inventory_reference: str | None = None
    bucket: str | None = None
    tag_filters: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sortie_id": self.sortie_id,
            "source_type": self.source_type,
            "stream_kind": self.stream_kind,
            "data_tier": self.data_tier,
            "raw_manifest_path": self.raw_manifest_path,
            "inventory_reference": self.inventory_reference,
            "time_range": dict(self.time_range),
            "measurement_family": list(self.measurement_family),
            "usable_for_pretraining": self.usable_for_pretraining,
            "notes": list(self.notes),
            "bucket": self.bucket,
            "tag_filters": dict(self.tag_filters),
        }


@dataclass(frozen=True, slots=True)
class PartialStreamSample:
    """One windowed single-stream sample produced from a partial-data entry."""

    sample_id: str
    sortie_id: str
    stream_kind: str
    window_index: int
    start_offset_ms: int
    end_offset_ms: int
    point_count: int
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartialDataManifest:
    """Output summary for one partial-data builder run."""

    export_version: str
    generated_at_utc: str
    entry_count: int
    built_entry_count: int
    skipped_entry_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "export_version": self.export_version,
            "generated_at_utc": self.generated_at_utc,
            "entry_count": self.entry_count,
            "built_entry_count": self.built_entry_count,
            "skipped_entry_count": self.skipped_entry_count,
        }


@dataclass(frozen=True, slots=True)
class PartialDataConfig:
    """Controls partial-data manifesting and vehicle-only sample export."""

    export_version: str = "partial-data-v1"
    window_config: WindowConfig = field(
        default_factory=lambda: WindowConfig(
            duration_ms=5_000,
            stride_ms=5_000,
            min_physiology_points=0,
            min_vehicle_points=1,
        )
    )


@dataclass(frozen=True, slots=True)
class PartialDataBuildResult:
    """Concrete artifact paths from one partial-data builder run."""

    manifest: PartialDataManifest
    manifest_path: str
    window_manifest_path: str
    feature_bundle_path: str | None
    built_samples: tuple[PartialStreamSample, ...]


PartialPointProvider = Callable[[PartialDataEntry], Sequence[RawPoint]]


@dataclass(slots=True)
class PartialDataBuilder:
    """Build repo-local single-stream assets from standardized partial entries."""

    config: PartialDataConfig = field(default_factory=PartialDataConfig)
    point_provider: PartialPointProvider | None = None

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
            if (
                entry.stream_kind == StreamKind.VEHICLE.value
                and self.point_provider is not None
            ):
                try:
                    points = tuple(self.point_provider(entry))
                except Exception as exc:
                    reason = str(exc)
                else:
                    if points:
                        dataset_result = SortieDatasetBuilder(
                            window_config=self.config.window_config,
                        ).build(
                            SortieBundle(
                                locator=SortieLocator(sortie_id=entry.sortie_id),
                                metadata=SortieMetadata(sortie_id=entry.sortie_id),
                                vehicle_points=points,
                            )
                        )
                        for window in dataset_result.windows:
                            matrix = build_numeric_stream_matrix(StreamKind.VEHICLE, window.vehicle_points)
                            if not matrix.feature_names:
                                continue
                            stream_matrices.append(matrix)
                            built_samples.append(
                                PartialStreamSample(
                                    sample_id=window.sample_id,
                                    sortie_id=entry.sortie_id,
                                    stream_kind=entry.stream_kind,
                                    window_index=window.window_index,
                                    start_offset_ms=window.start_offset_ms,
                                    end_offset_ms=window.end_offset_ms,
                                    point_count=matrix.point_count,
                                    feature_names=matrix.feature_names,
                                )
                            )
                            window_rows.append(
                                {
                                    "sortie_id": entry.sortie_id,
                                    "sample_id": window.sample_id,
                                    "window_index": window.window_index,
                                    "start_offset_ms": window.start_offset_ms,
                                    "end_offset_ms": window.end_offset_ms,
                                    "vehicle_point_count": len(window.vehicle_points),
                                    "feature_names": list(matrix.feature_names),
                                }
                            )
                        output_sample_count = len(
                            [sample for sample in built_samples if sample.sortie_id == entry.sortie_id]
                        )
                        status = "built" if output_sample_count > 0 else "no_windows"
                    else:
                        status = "no_points"
            row.update(
                {
                    "builder_status": status,
                    "builder_reason": reason,
                    "output_sample_count": output_sample_count,
                }
            )
            manifest_rows.append(row)

        _write_jsonl(manifest_path, manifest_rows)
        _write_jsonl(window_manifest_path, window_rows)

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


def load_partial_data_entries(path: str | Path) -> tuple[PartialDataEntry, ...]:
    """Load standardized partial-data entries from JSONL."""

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(
        PartialDataEntry(
            sortie_id=row["sortie_id"],
            source_type=row["source_type"],
            stream_kind=row["stream_kind"],
            data_tier=row["data_tier"],
            raw_manifest_path=row.get("raw_manifest_path"),
            inventory_reference=row.get("inventory_reference"),
            time_range=dict(row.get("time_range", {})),
            measurement_family=tuple(row.get("measurement_family", ())),
            usable_for_pretraining=bool(row.get("usable_for_pretraining", False)),
            notes=tuple(row.get("notes", ())),
            bucket=row.get("bucket"),
            tag_filters=dict(row.get("tag_filters", {})),
        )
        for row in rows
    )


def dump_partial_data_entries(
    entries: Iterable[PartialDataEntry],
    *,
    path: str | Path,
) -> None:
    """Write standardized partial-data entries to JSONL."""

    _write_jsonl(Path(path), [entry.to_dict() for entry in entries])


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
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
        sample_ids=np.asarray([sample.sample_id for sample in built_samples], dtype="<U128"),
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
