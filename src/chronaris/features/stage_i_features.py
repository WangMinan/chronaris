"""Feature extraction for Stage I public-dataset baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.dataset.stage_i_contracts import StageITaskEntry
from chronaris.features.stage_i_feature_helpers import (
    build_manifest_frame,
    build_session_feature_frame_from_parquet,
    build_window_feature_frame_from_parquet,
    build_window_specs,
    concat_feature_frames,
    count_missing_group_samples,
    entry_iso_window_bounds,
    group_entries_by_csv_path,
    merge_feature_frames,
    stream_window_feature_frame_from_csv,
)

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
            "feature_group_counts": {group_name: len(columns) for group_name, columns in self.feature_group_columns.items()},
            "eeg_feature_count": len(self.eeg_feature_columns),
            "ecg_feature_count": len(self.ecg_feature_columns),
            "peripheral_feature_count": len(self.peripheral_feature_columns),
            "feature_group_columns": {group_name: list(columns) for group_name, columns in self.feature_group_columns.items()},
            "feature_columns": list(self.feature_columns),
            "metadata_columns": [column for column in self.feature_table.columns if column not in self.feature_columns],
            "missing_group_sample_counts": {key: dict(value) for key, value in self.missing_group_sample_counts.items()},
        }


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

    manifest_frame = build_manifest_frame(active_entries)
    root = Path(dataset_root)
    csv_window_groups = group_entries_by_csv_path(active_entries)
    eeg_frames: list[pd.DataFrame] = []
    peripheral_frames: list[pd.DataFrame] = []

    for relative_csv_path, group_entries in sorted(csv_window_groups.items()):
        csv_path = root / relative_csv_path
        header = pd.read_csv(csv_path, nrows=0)
        eeg_columns = tuple(column for column in header.columns if column.startswith("EEG_"))
        peripheral_columns = tuple(column for column in ("ECG", "R", "GSR") if column in header.columns)
        recording_id = group_entries[0].recording_id or group_entries[0].session_id
        window_specs = {
            recording_id: build_window_specs(
                group_entries,
                time_range_builder=lambda entry: (
                    float(entry.context_payload["window_start_s"]),
                    float(entry.context_payload["window_end_s"]),
                ),
            )[recording_id]
        }
        eeg_frames.append(
            stream_window_feature_frame_from_csv(
                csv_path=csv_path,
                recording_id=recording_id,
                window_specs=window_specs,
                value_columns=eeg_columns,
                prefix="eeg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=(),
            )
        )
        peripheral_frames.append(
            stream_window_feature_frame_from_csv(
                csv_path=csv_path,
                recording_id=recording_id,
                window_specs=window_specs,
                value_columns=peripheral_columns,
                prefix="peripheral",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("ECG", "R", "GSR"),
            )
        )

    eeg_features = concat_feature_frames(eeg_frames, key_column="sample_id")
    peripheral_features = concat_feature_frames(peripheral_frames, key_column="sample_id")
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
        missing_group_sample_counts={"peripheral": count_missing_group_samples(feature_table, feature_group_columns["peripheral"])},
    )


def _build_uab_session_feature_table(
    dataset_root: str | Path,
    entries: Sequence[StageITaskEntry],
) -> StageIFeatureTableResult:
    root = Path(dataset_root) / UAB_DATASET_ID
    manifest_frame = build_manifest_frame(entries)

    n_back_eeg = build_session_feature_frame_from_parquet(
        parquet_path=root / "data_n_back_test" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")),
        prefix="eeg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=(),
    )
    heat_eeg = build_session_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")),
        prefix="eeg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=(),
    )
    flight_eeg = build_session_feature_frame_from_parquet(
        parquet_path=root / "data_flight_simulator" / "eeg" / "eeg.parquet",
        key_columns=("subject", "flight"),
        id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")),
        prefix="eeg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=(),
    )

    n_back_ecg = merge_feature_frames(
        (
            build_session_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "test"),
                id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            build_session_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_ibi.parquet",
                key_columns=("subject", "test"),
                id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("rr_int",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("rr_int",),
            ),
            build_session_feature_frame_from_parquet(
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
    heat_ecg = build_session_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "ecg" / "ecg.parquet",
        key_columns=("subject", "test"),
        id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        value_columns_selector=lambda schema_names: tuple(column for column in ("hr", "rr_int") if column in schema_names),
        prefix="ecg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=("hr", "rr_int"),
    )
    flight_ecg = merge_feature_frames(
        (
            build_session_feature_frame_from_parquet(
                parquet_path=root / "data_flight_simulator" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "flight"),
                id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            build_session_feature_frame_from_parquet(
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

    eeg_features = concat_feature_frames((n_back_eeg, heat_eeg, flight_eeg), key_column="sample_id")
    ecg_features = concat_feature_frames((n_back_ecg, heat_ecg, flight_ecg), key_column="sample_id")
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
        missing_group_sample_counts={"ecg": count_missing_group_samples(feature_table, feature_group_columns["ecg"])},
    )


def _build_uab_window_feature_table(
    dataset_root: str | Path,
    entries: Sequence[StageITaskEntry],
) -> StageIFeatureTableResult:
    root = Path(dataset_root) / UAB_DATASET_ID
    manifest_frame = build_manifest_frame(entries)
    window_specs = build_window_specs(entries, time_range_builder=entry_iso_window_bounds)

    n_back_eeg = build_window_feature_frame_from_parquet(
        parquet_path=root / "data_n_back_test" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")),
        prefix="eeg",
        stats=("mean", "std", "p25", "p75"),
        invalid_nonpositive_columns=(),
    )
    heat_eeg = build_window_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        recording_id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")),
        prefix="eeg",
        stats=("mean", "std", "p25", "p75"),
        invalid_nonpositive_columns=(),
    )
    flight_eeg = build_window_feature_frame_from_parquet(
        parquet_path=root / "data_flight_simulator" / "eeg" / "eeg.parquet",
        key_columns=("subject", "flight"),
        recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(column for column in schema_names if column.startswith("POW.") or column.startswith("PM.")),
        prefix="eeg",
        stats=("mean", "std", "p25", "p75"),
        invalid_nonpositive_columns=(),
    )

    n_back_ecg = merge_feature_frames(
        (
            build_window_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "test"),
                recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            build_window_feature_frame_from_parquet(
                parquet_path=root / "data_n_back_test" / "ecg" / "ecg_ibi.parquet",
                key_columns=("subject", "test"),
                recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("rr_int",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("rr_int",),
            ),
            build_window_feature_frame_from_parquet(
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
    heat_ecg = build_window_feature_frame_from_parquet(
        parquet_path=root / "data_heat_the_chair" / "ecg" / "ecg.parquet",
        key_columns=("subject", "test"),
        recording_id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
        window_specs=window_specs,
        value_columns_selector=lambda schema_names: tuple(column for column in ("hr", "rr_int") if column in schema_names),
        prefix="ecg",
        stats=("mean", "std", "min", "max"),
        invalid_nonpositive_columns=("hr", "rr_int"),
    )
    flight_ecg = merge_feature_frames(
        (
            build_window_feature_frame_from_parquet(
                parquet_path=root / "data_flight_simulator" / "ecg" / "ecg_hr.parquet",
                key_columns=("subject", "flight"),
                recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
                window_specs=window_specs,
                value_columns_selector=lambda schema_names: ("hr",),
                prefix="ecg",
                stats=("mean", "std", "min", "max"),
                invalid_nonpositive_columns=("hr",),
            ),
            build_window_feature_frame_from_parquet(
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

    eeg_features = concat_feature_frames((n_back_eeg, heat_eeg, flight_eeg), key_column="sample_id")
    ecg_features = concat_feature_frames((n_back_ecg, heat_ecg, flight_ecg), key_column="sample_id")
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
        missing_group_sample_counts={"ecg": count_missing_group_samples(feature_table, feature_group_columns["ecg"])},
    )
