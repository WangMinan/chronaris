"""Native-time CLARE central/peripheral samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.lazy_observed import (
    LazyObservedDataset,
    NativeSampleRecord,
    merge_native_feature_series,
    source_window_hash,
)
from chronaris.representation import ObservationSchema, ObservedDualStreamSample


CENTRAL_COLUMNS = ("TP9", "AF7", "AF8", "TP10")
CENTRAL_NAMES = tuple(f"central.eeg_{name.lower()}" for name in CENTRAL_COLUMNS)
PERIPH_NAMES = ("peripheral.eda", "peripheral.ecg")
CLARE_SCHEMA = ObservationSchema(
    schema_id="clare_native.v4",
    source_kind="clare_public",
    physiology_feature_names=CENTRAL_NAMES,
    vehicle_feature_names=PERIPH_NAMES,
    physiology_feature_roles=tuple("observed" for _ in CENTRAL_NAMES),
    vehicle_feature_roles=tuple("observed" for _ in PERIPH_NAMES),
)
_PREPROCESSING_VERSION = "native_physio_integrity_v3.2.2"


@dataclass(frozen=True, slots=True)
class ClareNativeRecord(NativeSampleRecord):
    eeg_path: Path
    eda_path: Path
    ecg_path: Path
    window_start_s: float


def build_clare_native_dataset(
    root: str | Path,
    *,
    subject_limit: int = 16,
    context_duration_s: float = 10.0,
    window_stride: int = 2,
    cache_root: str | Path | None = None,
    max_memory_cache_bytes: int = 512 * 1024**2,
) -> LazyObservedDataset[ClareNativeRecord]:
    root = Path(root)
    records: list[ClareNativeRecord] = []
    for subject in sorted(root.glob("EEG/[0-9]*"))[:subject_limit]:
        subject_id = subject.name
        labels_path = root / "Labels" / f"{subject_id}.csv"
        if not labels_path.exists():
            continue
        labels = pd.read_csv(labels_path)
        for experiment in range(4):
            eeg_path = root / "EEG" / subject_id / f"eeg_data_exp_{experiment}.csv"
            eda_path = root / "EDA" / subject_id / f"eda_data_experiment_{experiment}.csv"
            ecg_path = root / "ECG" / subject_id / f"ecg_data_experiment_{experiment}.csv"
            label_column = f"level_{experiment}"
            if not all(path.exists() for path in (eeg_path, eda_path, ecg_path)) or label_column not in labels:
                continue
            recording_start = float(
                pd.read_csv(eeg_path, usecols=["Timestamp"], nrows=1).iloc[0, 0]
            )
            for window_index, label in enumerate(labels[label_column]):
                if window_index % window_stride or not np.isfinite(label):
                    continue
                start = recording_start + window_index * context_duration_s
                sample_id = f"{subject_id}__experiment{experiment}_window{window_index}"
                source_hash = source_window_hash(
                    (eeg_path, eda_path, ecg_path),
                    sample_id,
                    start,
                    context_duration_s,
                    int(label),
                    _PREPROCESSING_VERSION,
                )
                records.append(
                    ClareNativeRecord(
                        sample_id=sample_id,
                        group_id=subject_id,
                        label=int(label),
                        context_duration_s=context_duration_s,
                        source_sample_hash=source_hash,
                        eeg_path=eeg_path,
                        eda_path=eda_path,
                        ecg_path=ecg_path,
                        window_start_s=start,
                    )
                )
    if not records:
        raise ValueError("no usable CLARE native-time samples found")
    return LazyObservedDataset(
        records,
        schema=CLARE_SCHEMA,
        loader=_load_native_sample,
        cache_root=cache_root,
        max_memory_cache_bytes=max_memory_cache_bytes,
    )


def _load_native_sample(record: ClareNativeRecord) -> ObservedDualStreamSample:
    central_timestamps, central_values = _read_window(
        record.eeg_path,
        CENTRAL_COLUMNS,
        record,
    )
    central_mask = np.isfinite(central_values)
    central_values = np.where(central_mask, central_values, 0.0).astype(np.float32)
    central_keep = central_mask.any(axis=1)

    eda_timestamps, eda_values = _read_window(
        record.eda_path,
        ("GSR Conductance CAL",),
        record,
    )
    ecg_timestamps, ecg_values = _read_window(
        record.ecg_path,
        ("ECG LL-RA CAL",),
        record,
    )
    peripheral_timestamps, peripheral_values, peripheral_mask = merge_native_feature_series(
        (
            (eda_timestamps, eda_values, (0,)),
            (ecg_timestamps, ecg_values, (1,)),
        ),
        feature_count=len(PERIPH_NAMES),
    )
    return ObservedDualStreamSample(
        sample_id=record.sample_id,
        group_id=record.group_id,
        schema=CLARE_SCHEMA,
        physiology_values=central_values[central_keep],
        physiology_timestamps_s=central_timestamps[central_keep],
        physiology_feature_mask=central_mask[central_keep],
        vehicle_values=peripheral_values,
        vehicle_timestamps_s=peripheral_timestamps,
        vehicle_feature_mask=peripheral_mask,
        source_sample_hash=record.source_sample_hash,
        context_duration_s=record.context_duration_s,
    )


def _read_window(
    path: Path,
    columns: tuple[str, ...],
    record: ClareNativeRecord,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=("Timestamp", *columns))
    timestamps = frame["Timestamp"].to_numpy(np.float64) - record.window_start_s
    keep = (timestamps >= 0.0) & (timestamps < record.context_duration_s)
    return timestamps[keep], frame.loc[keep, list(columns)].to_numpy(np.float64)
