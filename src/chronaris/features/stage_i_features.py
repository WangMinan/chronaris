"""Feature extraction for Stage I public-dataset baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chronaris.dataset.stage_i_contracts import StageITaskEntry

UAB_DATASET_ID = "uab_workload_dataset"
NASA_DATASET_ID = "nasa_csm"


@dataclass(frozen=True, slots=True)
class StageIFeatureTableResult:
    """Prepared feature table plus schema metadata."""

    feature_table: pd.DataFrame
    feature_group_columns: Mapping[str, tuple[str, ...]]
    missing_group_sample_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for columns in self.feature_group_columns.values():
            for column in columns:
                if column not in ordered:
                    ordered.append(column)
        return tuple(ordered)

    @property
    def eeg_feature_columns(self) -> tuple[str, ...]:
        return tuple(self.feature_group_columns.get("eeg", ()))

    @property
    def ecg_feature_columns(self) -> tuple[str, ...]:
        return tuple(self.feature_group_columns.get("ecg", ()))

    @property
    def peripheral_feature_columns(self) -> tuple[str, ...]:
        return tuple(self.feature_group_columns.get("peripheral", ()))

    @property
    def missing_ecg_session_counts(self) -> Mapping[str, int]:
        return dict(self.missing_group_sample_counts.get("ecg", {}))

    def feature_schema(self) -> dict[str, object]:
        return {
            "feature_count": len(self.feature_columns),
            "feature_group_counts": {
                group_name: len(columns)
                for group_name, columns in self.feature_group_columns.items()
            },
            "eeg_feature_count": len(self.eeg_feature_columns),
            "ecg_feature_count": len(self.ecg_feature_columns),
            "peripheral_feature_count": len(self.peripheral_feature_columns),
            "feature_group_columns": {
                group_name: list(columns)
                for group_name, columns in self.feature_group_columns.items()
            },
            "feature_columns": list(self.feature_columns),
            "metadata_columns": [
                column
                for column in self.feature_table.columns
                if column not in self.feature_columns
            ],
            "missing_group_sample_counts": {
                key: dict(value) for key, value in self.missing_group_sample_counts.items()
            },
        }


@dataclass(frozen=True, slots=True)
class _WindowSpec:
    sample_id: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _FixedWindowLayout:
    start: float
    duration: float
    sample_ids: tuple[str, ...]


@dataclass(slots=True)
class _WindowStreamState:
    window_specs: tuple[_WindowSpec, ...]
    prefix: str
    value_columns: tuple[str, ...]
    stats: tuple[str, ...]
    invalid_nonpositive_columns: set[str]
    current_index: int = 0
    accumulator: "_RunningStatAccumulator | None" = None

    def consume_frame(self, frame: pd.DataFrame, *, time_column: str) -> list[dict[str, object]]:
        if frame.empty or self.current_index >= len(self.window_specs):
            return []

        frame = frame.reset_index(drop=True)
        times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
        rows: list[dict[str, object]] = []
        cursor = 0

        while self.current_index < len(self.window_specs):
            if cursor >= len(frame):
                break
            spec = self.window_specs[self.current_index]
            current_time = float(times[cursor])
            if current_time >= spec.end:
                rows.extend(self._flush_current())
                continue
            if float(times[-1]) < spec.start:
                break

            start_index = int(np.searchsorted(times, spec.start, side="left"))
            if start_index >= len(frame):
                break
            if float(times[start_index]) >= spec.end:
                rows.extend(self._flush_current())
                cursor = start_index
                continue

            end_index = int(np.searchsorted(times, spec.end, side="left"))
            if start_index < end_index:
                self._update_accumulator(
                    frame.iloc[start_index:end_index][list(self.value_columns)].reset_index(drop=True)
                )

            if float(times[-1]) < spec.end:
                break

            rows.extend(self._flush_current())
            cursor = max(end_index, start_index + 1)

        return rows

    def _flush_current(self) -> list[dict[str, object]]:
        if self.current_index >= len(self.window_specs):
            return []
        spec = self.window_specs[self.current_index]
        self.current_index += 1
        if self.accumulator is None:
            return []
        row = self.accumulator.to_row(sample_id=spec.sample_id, prefix=self.prefix, stats=self.stats)
        self.accumulator = None
        return [row]

    def _update_accumulator(self, frame: pd.DataFrame) -> None:
        if self.accumulator is None:
            self.accumulator = _RunningStatAccumulator(columns=self.value_columns)
        self.accumulator.update(frame, invalid_nonpositive_columns=self.invalid_nonpositive_columns)


@dataclass(slots=True)
class _RunningStatAccumulator:
    columns: Sequence[str]
    counts: dict[str, int] = field(default_factory=dict)
    sums: dict[str, float] = field(default_factory=dict)
    sums_sq: dict[str, float] = field(default_factory=dict)
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)

    def update(self, frame: pd.DataFrame, *, invalid_nonpositive_columns: set[str]) -> None:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        for column in invalid_nonpositive_columns:
            if column in numeric.columns:
                numeric[column] = numeric[column].where(numeric[column] > 0)
        for column in self.columns:
            if column not in numeric.columns:
                continue
            series = numeric[column].dropna()
            if series.empty:
                continue
            values = series.to_numpy(dtype=float)
            self.counts[column] = self.counts.get(column, 0) + int(len(values))
            self.sums[column] = self.sums.get(column, 0.0) + float(values.sum())
            self.sums_sq[column] = self.sums_sq.get(column, 0.0) + float(np.square(values).sum())
            current_min = float(values.min())
            current_max = float(values.max())
            self.mins[column] = current_min if column not in self.mins else min(self.mins[column], current_min)
            self.maxs[column] = current_max if column not in self.maxs else max(self.maxs[column], current_max)

    def to_row(self, *, sample_id: str, prefix: str, stats: tuple[str, ...]) -> dict[str, object]:
        row: dict[str, object] = {"sample_id": sample_id}
        for column in self.columns:
            safe_name = column.replace(".", "_")
            count = self.counts.get(column, 0)
            if count == 0:
                mean = std = minimum = maximum = float("nan")
            else:
                mean = self.sums[column] / count
                variance = max((self.sums_sq[column] / count) - mean * mean, 0.0)
                std = float(np.sqrt(variance))
                minimum = self.mins[column]
                maximum = self.maxs[column]
            if "mean" in stats:
                row[f"{prefix}__{safe_name}__mean"] = float(mean)
            if "std" in stats:
                row[f"{prefix}__{safe_name}__std"] = float(std)
            if "min" in stats:
                row[f"{prefix}__{safe_name}__min"] = float(minimum)
            if "max" in stats:
                row[f"{prefix}__{safe_name}__max"] = float(maximum)
        return row


def build_uab_feature_table(
    dataset_root: str | Path,
    entries: Sequence[StageITaskEntry],
) -> StageIFeatureTableResult:
    """Build the Stage I feature table for the local UAB dataset."""

    active_entries = tuple(entry for entry in entries if entry.dataset_id == UAB_DATASET_ID)
    if not active_entries:
        raise ValueError("no UAB Stage I task entries were provided.")

    granularities = {entry.sample_granularity for entry in active_entries}
    if granularities == {"session"}:
        return _build_uab_session_feature_table(dataset_root, active_entries)
    if granularities == {"window"}:
        return _build_uab_window_feature_table(dataset_root, active_entries)
    raise ValueError(f"mixed or unsupported UAB sample granularities: {sorted(granularities)}")


def build_nasa_csm_feature_table(
    dataset_root: str | Path,
    entries: Sequence[StageITaskEntry],
) -> StageIFeatureTableResult:
    """Build the Stage I feature table for the local NASA CSM dataset."""

    active_entries = tuple(entry for entry in entries if entry.dataset_id == NASA_DATASET_ID)
    if not active_entries:
        raise ValueError("no NASA CSM Stage I task entries were provided.")
    if {entry.sample_granularity for entry in active_entries} != {"window"}:
        raise ValueError("NASA CSM Stage I currently supports window samples only.")

    manifest_frame = _build_manifest_frame(active_entries)
    root = Path(dataset_root)
    csv_window_groups = _group_entries_by_csv_path(active_entries)
    eeg_frames: list[pd.DataFrame] = []
    peripheral_frames: list[pd.DataFrame] = []

    for relative_csv_path, group_entries in sorted(csv_window_groups.items()):
        csv_path = root / relative_csv_path
        header = pd.read_csv(csv_path, nrows=0)
        eeg_columns = tuple(column for column in header.columns if column.startswith("EEG_"))
        peripheral_columns = tuple(column for column in ("ECG", "R", "GSR") if column in header.columns)
        window_specs = {
            group_entries[0].recording_id or group_entries[0].session_id: _build_window_specs(
                group_entries,
                time_range_builder=lambda entry: (
                    float(entry.context_payload["window_start_s"]),
                    float(entry.context_payload["window_end_s"]),
                ),
            )[group_entries[0].recording_id or group_entries[0].session_id]
        }
        eeg_frames.append(
            _stream_window_feature_frame_from_csv(
                csv_path=csv_path,
                recording_id=group_entries[0].recording_id or group_entries[0].session_id,
                window_specs=window_specs,
                value_columns=eeg_columns,
                prefix="eeg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=(),
            )
        )
        peripheral_frames.append(
            _stream_window_feature_frame_from_csv(
                csv_path=csv_path,
                recording_id=group_entries[0].recording_id or group_entries[0].session_id,
                window_specs=window_specs,
                value_columns=peripheral_columns,
                prefix="peripheral",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("ECG", "R", "GSR"),
            )
        )

    eeg_features = _concat_feature_frames(eeg_frames, key_column="sample_id")
    peripheral_features = _concat_feature_frames(peripheral_frames, key_column="sample_id")
    feature_table = manifest_frame.merge(eeg_features, on="sample_id", how="left")
    feature_table = feature_table.merge(peripheral_features, on="sample_id", how="left")
    feature_table = feature_table.sort_values(["subset_id", "subject_id", "recording_id", "window_index"]).reset_index(drop=True)

    feature_group_columns = {
        "eeg": tuple(column for column in eeg_features.columns if column != "sample_id"),
        "peripheral": tuple(column for column in peripheral_features.columns if column != "sample_id"),
    }
    return StageIFeatureTableResult(
        feature_table=feature_table,
        feature_group_columns=feature_group_columns,
        missing_group_sample_counts={
            "peripheral": _count_missing_group_samples(feature_table, feature_group_columns["peripheral"]),
        },
    )


def _build_uab_session_feature_table(
    dataset_root: str | Path,
    entries: Sequence[StageITaskEntry],
) -> StageIFeatureTableResult:
    root = Path(dataset_root) / UAB_DATASET_ID
    manifest_frame = _build_manifest_frame(entries)

    n_back_eeg = _build_session_feature_frame_from_parquet(
        parquet_path=root / "data_n_back_test" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(
            column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")
        ),
        prefix="eeg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=(),
    )
    heat_eeg = _build_session_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(
            column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")
        ),
        prefix="eeg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=(),
    )
    flight_eeg = _build_session_feature_frame_from_parquet(
        parquet_path=root / "data_flight_simulator" / "eeg" / "eeg.parquet",
        key_columns=("subject", "flight"),
        id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(
            column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")
        ),
        prefix="eeg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=(),
    )

    n_back_ecg = _merge_feature_frames(
        (
            _build_session_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "test"),
                id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            _build_session_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_ibi.parquet",
                key_columns=("subject", "test"),
                id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("rr_int",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("rr_int",),
            ),
            _build_session_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_br.parquet",
                key_columns=("subject", "test"),
                id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("br",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("br",),
            ),
        ),
        key_column="sample_id",
    )
    heat_ecg = _build_session_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "ecg" / "ecg.parquet",
        key_columns=("subject", "test"),
        id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(
            column for column in ("hr", "rr_int") if column in schema_names
        ),
        prefix="ecg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=("hr", "rr_int"),
    )
    flight_ecg = _merge_feature_frames(
        (
            _build_session_feature_frame_from_parquet(
                parquet_path=root / "data_flight_simulator" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "flight"),
                id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            _build_session_feature_frame_from_parquet(
                parquet_path=root / "data_flight_simulator" / "ecg" / "ecg_ibi.parquet",
                key_columns=("subject", "flight"),
                id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("rr_int",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("rr_int",),
            ),
        ),
        key_column="sample_id",
    )

    eeg_features = _concat_feature_frames((n_back_eeg, heat_eeg, flight_eeg), key_column="sample_id")
    ecg_features = _concat_feature_frames((n_back_ecg, heat_ecg, flight_ecg), key_column="sample_id")
    feature_table = manifest_frame.merge(eeg_features, on="sample_id", how="left")
    feature_table = feature_table.merge(ecg_features, on="sample_id", how="left")
    feature_table = feature_table.sort_values(["subset_id", "subject_id", "session_id"]).reset_index(drop=True)

    feature_group_columns = {
        "eeg": tuple(column for column in eeg_features.columns if column != "sample_id"),
        "ecg": tuple(column for column in ecg_features.columns if column != "sample_id"),
    }
    return StageIFeatureTableResult(
        feature_table=feature_table,
        feature_group_columns=feature_group_columns,
        missing_group_sample_counts={
            "ecg": _count_missing_group_samples(feature_table, feature_group_columns["ecg"]),
        },
    )


def _build_uab_window_feature_table(
    dataset_root: str | Path,
    entries: Sequence[StageITaskEntry],
) -> StageIFeatureTableResult:
    root = Path(dataset_root) / UAB_DATASET_ID
    manifest_frame = _build_manifest_frame(entries)
    window_specs = _build_window_specs(entries, time_range_builder=_entry_iso_window_bounds)

    n_back_eeg = _build_window_feature_frame_from_parquet(
        parquet_path=root / "data_n_back_test" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(
            column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")
        ),
        prefix="eeg",
        stats=("mean", "std", "p25", "p75"),
        invalid_nonpositive_columns=(),
    )
    heat_eeg = _build_window_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        recording_id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(
            column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")
        ),
        prefix="eeg",
        stats=("mean", "std", "p25", "p75"),
        invalid_nonpositive_columns=(),
    )
    flight_eeg = _build_window_feature_frame_from_parquet(
        parquet_path=root / "data_flight_simulator" / "eeg" / "eeg.parquet",
        key_columns=("subject", "flight"),
        recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(
            column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")
        ),
        prefix="eeg",
        stats=("mean", "std", "p25", "p75"),
        invalid_nonpositive_columns=(),
    )

    n_back_ecg = _merge_feature_frames(
        (
            _build_window_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "test"),
                recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            _build_window_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_ibi.parquet",
                key_columns=("subject", "test"),
                recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("rr_int",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("rr_int",),
            ),
            _build_window_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_br.parquet",
                key_columns=("subject", "test"),
                recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("br",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("br",),
            ),
        ),
        key_column="sample_id",
    )
    heat_ecg = _build_window_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "ecg" / "ecg.parquet",
        key_columns=("subject", "test"),
        recording_id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(
            column for column in ("hr", "rr_int") if column in schema_names
        ),
        prefix="ecg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=("hr", "rr_int"),
    )
    flight_ecg = _merge_feature_frames(
        (
            _build_window_feature_frame_from_parquet(
                parquet_path=root / "data_flight_simulator" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "flight"),
                recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            _build_window_feature_frame_from_parquet(
                parquet_path=root / "data_flight_simulator" / "ecg" / "ecg_ibi.parquet",
                key_columns=("subject", "flight"),
                recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("rr_int",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("rr_int",),
            ),
        ),
        key_column="sample_id",
    )

    eeg_features = _concat_feature_frames((n_back_eeg, heat_eeg, flight_eeg), key_column="sample_id")
    ecg_features = _concat_feature_frames((n_back_ecg, heat_ecg, flight_ecg), key_column="sample_id")
    feature_table = manifest_frame.merge(eeg_features, on="sample_id", how="left")
    feature_table = feature_table.merge(ecg_features, on="sample_id", how="left")
    feature_table = feature_table.sort_values(["subset_id", "subject_id", "recording_id", "window_index"]).reset_index(drop=True)

    feature_group_columns = {
        "eeg": tuple(column for column in eeg_features.columns if column != "sample_id"),
        "ecg": tuple(column for column in ecg_features.columns if column != "sample_id"),
    }
    return StageIFeatureTableResult(
        feature_table=feature_table,
        feature_group_columns=feature_group_columns,
        missing_group_sample_counts={
            "ecg": _count_missing_group_samples(feature_table, feature_group_columns["ecg"]),
        },
    )


def _build_manifest_frame(entries: Sequence[StageITaskEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": entry.sample_id,
                "dataset_id": entry.dataset_id,
                "subset_id": entry.subset_id,
                "subject_id": entry.subject_id,
                "session_id": entry.session_id,
                "recording_id": entry.recording_id or entry.session_id,
                "split_group": entry.split_group,
                "training_role": entry.training_role,
                "sample_granularity": entry.sample_granularity,
                "window_index": entry.window_index,
                "window_duration_s": entry.window_duration_s,
                "task_family": entry.task_family,
                "label_namespace": entry.label_namespace,
                "window_start_utc": entry.window_start_utc,
                "window_end_utc": entry.window_end_utc,
                "objective_label_name": entry.objective_label_name,
                "objective_label_value": entry.objective_label_value,
                "subjective_target_name": entry.subjective_target_name,
                "subjective_target_value": entry.subjective_target_value,
                "source_partition": entry.context_payload.get("source_partition"),
                "recording_type": entry.context_payload.get("recording_type"),
                "objective_label_text": entry.context_payload.get("objective_label_text"),
            }
            for entry in entries
        ]
    )


def _group_entries_by_csv_path(entries: Sequence[StageITaskEntry]) -> dict[str, tuple[StageITaskEntry, ...]]:
    grouped: dict[str, list[StageITaskEntry]] = {}
    for entry in entries:
        relative_path = str(entry.source_refs["csv_path"])
        grouped.setdefault(relative_path, []).append(entry)
    return {key: tuple(value) for key, value in grouped.items()}


def _build_window_specs(
    entries: Sequence[StageITaskEntry],
    *,
    time_range_builder: Callable[[StageITaskEntry], tuple[float, float]],
) -> dict[str, tuple[_WindowSpec, ...]]:
    grouped: dict[str, list[_WindowSpec]] = {}
    for entry in entries:
        recording_id = entry.recording_id or entry.session_id
        start, end = time_range_builder(entry)
        grouped.setdefault(recording_id, []).append(
            _WindowSpec(
                sample_id=entry.sample_id,
                start=float(start),
                end=float(end),
            )
        )
    return {
        key: tuple(sorted(value, key=lambda spec: (spec.start, spec.sample_id)))
        for key, value in grouped.items()
    }


def _entry_iso_window_bounds(entry: StageITaskEntry) -> tuple[float, float]:
    start = pd.Timestamp(entry.window_start_utc)
    end = pd.Timestamp(entry.window_end_utc)
    return start.timestamp(), end.timestamp()


def _build_session_feature_frame_from_parquet(
    *,
    parquet_path: Path,
    key_columns: tuple[str, ...],
    id_builder: Callable[[tuple[object, ...]], str],
    value_columns_selector: Callable[[tuple[str, ...]], tuple[str, ...]],
    prefix: str,
    stats: tuple[str, ...],
    invalid_nonpositive_columns: Iterable[str],
) -> pd.DataFrame:
    schema_names = tuple(pq.ParquetFile(parquet_path).schema.names)
    value_columns = value_columns_selector(schema_names)
    return _stream_group_feature_frame_from_parquet(
        parquet_path=parquet_path,
        key_columns=key_columns,
        id_builder=id_builder,
        value_columns=value_columns,
        prefix=prefix,
        stats=stats,
        invalid_nonpositive_columns=invalid_nonpositive_columns,
    )


def _build_window_feature_frame_from_parquet(
    *,
    parquet_path: Path,
    key_columns: tuple[str, ...],
    recording_id_builder: Callable[[tuple[object, ...]], str],
    window_specs: Mapping[str, tuple[_WindowSpec, ...]],
    value_columns_selector: Callable[[tuple[str, ...]], tuple[str, ...]],
    prefix: str,
    stats: tuple[str, ...],
    invalid_nonpositive_columns: Iterable[str],
) -> pd.DataFrame:
    schema_names = tuple(pq.ParquetFile(parquet_path).schema.names)
    value_columns = value_columns_selector(schema_names)
    if not value_columns:
        return pd.DataFrame({"sample_id": []})

    reader = pq.ParquetFile(parquet_path)
    layouts = {
        recording_id: _FixedWindowLayout(
            start=specs[0].start,
            duration=specs[0].end - specs[0].start,
            sample_ids=tuple(spec.sample_id for spec in specs),
        )
        for recording_id, specs in window_specs.items()
        if specs
    }
    accumulators: dict[str, dict[int, _RunningStatAccumulator]] = {}
    for batch in reader.iter_batches(batch_size=65_536, columns=[*key_columns, "datetime", *value_columns]):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["__time_seconds"] = frame["datetime"].astype("datetime64[ns]").astype("int64") / 1_000_000_000.0
        for key, group in frame.groupby(list(key_columns), sort=False):
            normalized_key = key if isinstance(key, tuple) else (key,)
            recording_id = recording_id_builder(_normalize_key(normalized_key))
            layout = layouts.get(recording_id)
            if layout is None:
                continue
            times = group["__time_seconds"].to_numpy(dtype=float)
            window_indices = np.floor((times - layout.start) / layout.duration).astype(int)
            valid = (window_indices >= 0) & (window_indices < len(layout.sample_ids))
            if not np.any(valid):
                continue
            valid &= times < (layout.start + (window_indices + 1) * layout.duration)
            if not np.any(valid):
                continue
            numeric = group.loc[valid, list(value_columns)].apply(pd.to_numeric, errors="coerce")
            for column in invalid_nonpositive_columns:
                if column in numeric.columns:
                    numeric[column] = numeric[column].where(numeric[column] > 0)
            numeric["__window_index"] = window_indices[valid]
            by_window = accumulators.setdefault(recording_id, {})
            for window_index, window_frame in numeric.groupby("__window_index", sort=False):
                accumulator = by_window.setdefault(
                    int(window_index),
                    _RunningStatAccumulator(columns=value_columns),
                )
                accumulator.update(
                    window_frame.drop(columns="__window_index"),
                    invalid_nonpositive_columns=set(),
                )

    rows: list[dict[str, object]] = []
    for recording_id, by_window in accumulators.items():
        layout = layouts[recording_id]
        for window_index, accumulator in by_window.items():
            rows.append(
                accumulator.to_row(
                    sample_id=layout.sample_ids[window_index],
                    prefix=prefix,
                    stats=stats,
                )
            )
    return pd.DataFrame(rows)


def _stream_window_feature_frame_from_csv(
    *,
    csv_path: Path,
    recording_id: str,
    window_specs: Mapping[str, tuple[_WindowSpec, ...]],
    value_columns: tuple[str, ...],
    prefix: str,
    stats: tuple[str, ...],
    invalid_nonpositive_columns: Iterable[str],
) -> pd.DataFrame:
    if not value_columns:
        return pd.DataFrame({"sample_id": []})
    specs = window_specs.get(recording_id)
    if not specs:
        return pd.DataFrame({"sample_id": []})

    state = _WindowStreamState(
        window_specs=specs,
        prefix=prefix,
        value_columns=value_columns,
        stats=stats,
        invalid_nonpositive_columns=set(invalid_nonpositive_columns),
    )
    rows: list[dict[str, object]] = []
    usecols = ["TimeSecs", *value_columns]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=200_000):
        if chunk.empty:
            continue
        rows.extend(state.consume_frame(chunk.rename(columns={"TimeSecs": "time_value"}), time_column="time_value"))
    return pd.DataFrame(rows)


def _stream_group_feature_frame_from_parquet(
    *,
    parquet_path: Path,
    key_columns: tuple[str, ...],
    id_builder: Callable[[tuple[object, ...]], str],
    value_columns: tuple[str, ...],
    prefix: str,
    stats: tuple[str, ...],
    invalid_nonpositive_columns: Iterable[str],
) -> pd.DataFrame:
    if not value_columns:
        return pd.DataFrame({"sample_id": []})

    file_reader = pq.ParquetFile(parquet_path)
    rows: list[dict[str, object]] = []
    current_key: tuple[object, ...] | None = None
    current_frames: list[pd.DataFrame] = []

    for batch in file_reader.iter_batches(batch_size=65_536, columns=[*key_columns, *value_columns]):
        frame = batch.to_pandas()
        for batch_key, batch_values in _iter_contiguous_segments(frame, key_columns, value_columns):
            if current_key is None:
                current_key = batch_key
                current_frames = [batch_values]
                continue
            if batch_key == current_key:
                current_frames.append(batch_values)
                continue
            rows.append(
                _summarize_sample_frame(
                    sample_id=id_builder(current_key),
                    frame_parts=current_frames,
                    prefix=prefix,
                    stats=stats,
                    invalid_nonpositive_columns=set(invalid_nonpositive_columns),
                )
            )
            current_key = batch_key
            current_frames = [batch_values]

    if current_key is not None:
        rows.append(
            _summarize_sample_frame(
                sample_id=id_builder(current_key),
                frame_parts=current_frames,
                prefix=prefix,
                stats=stats,
                invalid_nonpositive_columns=set(invalid_nonpositive_columns),
            )
        )

    return pd.DataFrame(rows)


def _iter_contiguous_segments(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> Iterable[tuple[tuple[object, ...], pd.DataFrame]]:
    if frame.empty:
        return ()
    keys = list(map(_normalize_key, frame[list(key_columns)].itertuples(index=False, name=None)))
    start = 0
    segments: list[tuple[tuple[object, ...], pd.DataFrame]] = []
    for index in range(1, len(keys) + 1):
        if index == len(keys) or keys[index] != keys[start]:
            segments.append((keys[start], frame.iloc[start:index][list(value_columns)].reset_index(drop=True)))
            start = index
    return tuple(segments)


def _summarize_sample_frame(
    *,
    sample_id: str,
    frame_parts: Sequence[pd.DataFrame],
    prefix: str,
    stats: tuple[str, ...],
    invalid_nonpositive_columns: set[str],
) -> dict[str, object]:
    data = frame_parts[0] if len(frame_parts) == 1 else pd.concat(frame_parts, axis=0, ignore_index=True)
    numeric = data.apply(pd.to_numeric, errors="coerce")
    for column in invalid_nonpositive_columns:
        if column in numeric.columns:
            numeric[column] = numeric[column].where(numeric[column] > 0)

    row: dict[str, object] = {"sample_id": sample_id}
    means = numeric.mean()
    stds = numeric.std(ddof=0)
    mins = numeric.min()
    maxs = numeric.max()
    p25 = numeric.quantile(0.25)
    p75 = numeric.quantile(0.75)
    for column in numeric.columns:
        safe_name = column.replace(".", "_")
        if "mean" in stats:
            row[f"{prefix}__{safe_name}__mean"] = float(means.get(column, float("nan")))
        if "std" in stats:
            row[f"{prefix}__{safe_name}__std"] = float(stds.get(column, float("nan")))
        if "min" in stats:
            row[f"{prefix}__{safe_name}__min"] = float(mins.get(column, float("nan")))
        if "max" in stats:
            row[f"{prefix}__{safe_name}__max"] = float(maxs.get(column, float("nan")))
        if "p25" in stats:
            row[f"{prefix}__{safe_name}__p25"] = float(p25.get(column, float("nan")))
        if "p75" in stats:
            row[f"{prefix}__{safe_name}__p75"] = float(p75.get(column, float("nan")))
    return row


def _merge_feature_frames(frames: Sequence[pd.DataFrame], *, key_column: str) -> pd.DataFrame:
    active_frames = [frame for frame in frames if not frame.empty]
    if not active_frames:
        return pd.DataFrame({key_column: []})
    merged = active_frames[0]
    for frame in active_frames[1:]:
        overlapping = set(merged.columns) & set(frame.columns) - {key_column}
        if overlapping:
            raise ValueError(f"feature column collision detected: {sorted(overlapping)}")
        merged = merged.merge(frame, on=key_column, how="outer")
    return merged


def _concat_feature_frames(frames: Sequence[pd.DataFrame], *, key_column: str) -> pd.DataFrame:
    active_frames = [frame for frame in frames if not frame.empty]
    if not active_frames:
        return pd.DataFrame({key_column: []})
    return pd.concat(active_frames, axis=0, ignore_index=True, sort=False)


def _count_missing_group_samples(feature_table: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, int]:
    if not feature_columns:
        return {}
    return {
        subset_id: int(
            feature_table.loc[feature_table["subset_id"] == subset_id, list(feature_columns)].isna().all(axis=1).sum()
        )
        for subset_id in sorted(feature_table["subset_id"].dropna().astype(str).unique())
    }


def _normalize_key(values: Sequence[object]) -> tuple[object, ...]:
    normalized: list[object] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(value)
        elif isinstance(value, float) and value.is_integer():
            normalized.append(int(value))
        else:
            normalized.append(value)
    return tuple(normalized)
