"""Native-time CogPilot samples for difficulty and event-response tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from chronaris.dataset.lazy_observed import (
    LazyObservedDataset,
    NativeSampleRecord,
    merge_native_feature_series,
    source_window_hash,
)
from chronaris.representation import ObservationSchema, ObservedDualStreamSample
from chronaris.dataset.native_table_cache import read_native_table, select_native_subjects


PHYS_FILES = {
    "lslshimmereda": ("ppg_finger_mV", "eda_hand_l_kOhms"),
    "lslshimmerresp": ("respiration_trace_mV",),
}
ECG_FILE = "lslshimmerecg"
ECG_COL = "ecg_projection_ll_ra_mV"
VEH_FILE = "lslxp11xpcac"
VEH_COLS = (
    "aircraft_indicated_airspeed_kias",
    "aircraft_pitch_deg",
    "aircraft_roll_deg",
    "aircraft_agl_altitude_m",
    "aircraft_climb_rate_mps",
    "aircraft_ils_deflection_gs",
    "aircraft_ils_deflection_h",
    "aircraft_velocity_u_mps",
)
PHYS_NAMES = (
    "physiology.ppg",
    "physiology.eda_conductance_us",
    "physiology.resp",
    "physiology.ecg",
)
VEH_NAMES = tuple(f"vehicle.{name.removeprefix('aircraft_')}" for name in VEH_COLS)
COGPILOT_SCHEMA = ObservationSchema(
    schema_id="cogpilot_native.v4",
    source_kind="cogpilot_public",
    physiology_feature_names=PHYS_NAMES,
    vehicle_feature_names=VEH_NAMES,
    physiology_feature_roles=tuple("observed" for _ in PHYS_NAMES),
    vehicle_feature_roles=tuple("observed" for _ in VEH_NAMES),
)
_DAY_TO_SECONDS = 86400.0
_PREPROCESSING_VERSION = "native_window_contract_v4"


@dataclass(frozen=True, slots=True)
class CogPilotNativeRecord(NativeSampleRecord):
    physiology_paths: tuple[Path, ...]
    ecg_path: Path
    vehicle_path: Path
    window_start_native: float
    time_scale: float = _DAY_TO_SECONDS


def build_cogpilot_difficulty_dataset(
    root: str | Path,
    *,
    subject_limit: int | None = 20,
    subject_ids: Sequence[str] | None = None,
    all_legal_windows: bool = False,
    window_start_s: float = 60.0,
    context_duration_s: float = 30.0,
    cache_root: str | Path | None = None,
    max_memory_cache_bytes: int = 512 * 1024**2,
) -> LazyObservedDataset[CogPilotNativeRecord]:
    if context_duration_s <= 0 or window_start_s < 0:
        raise ValueError("CogPilot context bounds are invalid")
    records: list[CogPilotNativeRecord] = []
    for subject in select_native_subjects(Path(root).glob("sub-cp*"), subject_limit=subject_limit, subject_ids=subject_ids):
        for run in sorted(subject.glob("ses-*/level-*_run-*")):
            paths = _required_paths(run)
            if paths is None:
                continue
            physiology_paths, ecg_path, vehicle_path = paths
            origin = max(_first_timestamp(path) for path in (*physiology_paths, ecg_path, vehicle_path))
            start = origin + window_start_s / _DAY_TO_SECONDS
            level = int(run.name.split("_")[0].split("-")[1][:2]) - 1
            sample_id = f"{subject.name}::{run.name}"
            starts = (start,)
            if all_legal_windows:
                bounds = [_recording_bounds(path) for path in (*physiology_paths, ecg_path, vehicle_path)]
                origin = max(value[0] for value in bounds)
                stop = min(value[1] for value in bounds)
                count = max(0, int(np.ceil(((stop - origin) * _DAY_TO_SECONDS - window_start_s) / context_duration_s)))
                candidates = (origin + (window_start_s + i * context_duration_s) / _DAY_TO_SECONDS for i in range(count))
                starts = tuple(value for value in candidates if value + context_duration_s / _DAY_TO_SECONDS <= stop)
            for index, start in enumerate(starts):
                current_id = (f"{subject.name}::{run.parent.name}::{run.name}::window{index:04d}"
                              if all_legal_windows else sample_id)
                records.append(_difficulty_record(current_id, subject.name, level, start, context_duration_s,
                                                   physiology_paths, ecg_path, vehicle_path))
    return _dataset(records, cache_root, max_memory_cache_bytes)


def _difficulty_record(sample_id, group_id, level, start, context_duration_s, physiology_paths, ecg_path, vehicle_path):
    source_hash = source_window_hash(
        (*physiology_paths, ecg_path, vehicle_path), sample_id, start, context_duration_s, _PREPROCESSING_VERSION)
    return CogPilotNativeRecord(
        sample_id=sample_id, group_id=group_id, label=level, context_duration_s=context_duration_s,
        source_sample_hash=source_hash, physiology_paths=physiology_paths, ecg_path=ecg_path,
        vehicle_path=vehicle_path, window_start_native=start)


def build_cogpilot_event_response_dataset(
    root: str | Path,
    *,
    subject_limit: int | None = 20,
    subject_ids: Sequence[str] | None = None,
    context_duration_s: float = 12.0,
    response_pre_s: float = 2.0,
    response_post_s: float = 8.0,
    minimum_event_gap_s: float = 15.0,
    max_events_per_run: int | None = 3,
    cache_root: str | Path | None = None,
    max_memory_cache_bytes: int = 512 * 1024**2,
) -> LazyObservedDataset[CogPilotNativeRecord]:
    if min(context_duration_s, response_pre_s, response_post_s, minimum_event_gap_s) <= 0:
        raise ValueError("CogPilot event durations must be positive")
    if max_events_per_run is not None and max_events_per_run <= 0:
        raise ValueError("event limit must be positive or None")
    records: list[CogPilotNativeRecord] = []
    for subject in select_native_subjects(Path(root).glob("sub-cp*"), subject_limit=subject_limit, subject_ids=subject_ids):
        for run in sorted(subject.glob("ses-*/level-*_run-*")):
            paths = _required_paths(run)
            if paths is None:
                continue
            physiology_paths, ecg_path, vehicle_path = paths
            events = _event_times_and_responses(
                vehicle_path,
                physiology_paths[0],
                context_duration_s=context_duration_s,
                response_pre_s=response_pre_s,
                response_post_s=response_post_s,
                minimum_event_gap_s=minimum_event_gap_s,
                max_events=max_events_per_run,
            )
            bounds = [_recording_bounds(path) for path in (*physiology_paths, ecg_path, vehicle_path)] if max_events_per_run is None else None
            for event_index, (event_time, response) in enumerate(events):
                sample_id = f"{subject.name}::{run.name}::event{event_index}"
                start = event_time - context_duration_s / _DAY_TO_SECONDS
                if bounds is not None:
                    if start < max(value[0] for value in bounds) or event_time > min(value[1] for value in bounds):
                        continue
                    sample_id = f"{subject.name}::{run.parent.name}::{run.name}::event{event_index:04d}"
                source_hash = source_window_hash(
                    (*physiology_paths, ecg_path, vehicle_path),
                    sample_id,
                    start,
                    context_duration_s,
                    response,
                    _PREPROCESSING_VERSION,
                )
                records.append(
                    CogPilotNativeRecord(
                        sample_id=sample_id,
                        group_id=subject.name,
                        label=response,
                        context_duration_s=context_duration_s,
                        source_sample_hash=source_hash,
                        physiology_paths=physiology_paths,
                        ecg_path=ecg_path,
                        vehicle_path=vehicle_path,
                        window_start_native=start,
                    )
                )
    return _dataset(records, cache_root, max_memory_cache_bytes)


def _dataset(records, cache_root, max_memory_cache_bytes):
    if not records:
        raise ValueError("no usable CogPilot native-time samples found")
    return LazyObservedDataset(
        records,
        schema=COGPILOT_SCHEMA,
        loader=_load_native_sample,
        cache_root=cache_root,
        max_memory_cache_bytes=max_memory_cache_bytes,
    )


def _load_native_sample(record: CogPilotNativeRecord) -> ObservedDualStreamSample:
    physiology_series = []
    next_feature = 0
    for path, columns in zip(record.physiology_paths, PHYS_FILES.values(), strict=True):
        timestamps, values = _read_window(path, columns, record)
        if "eda_hand_l_kOhms" in columns:
            column = columns.index("eda_hand_l_kOhms")
            values[:, column] = _resistance_to_conductance(values[:, column])
        indices = tuple(range(next_feature, next_feature + len(columns)))
        physiology_series.append((timestamps, values, indices))
        next_feature += len(columns)
    ecg_timestamps, ecg_values = _read_window(record.ecg_path, (ECG_COL,), record)
    physiology_series.append((ecg_timestamps, ecg_values, (len(PHYS_NAMES) - 1,)))
    physiology_timestamps, physiology_values, physiology_mask = merge_native_feature_series(
        physiology_series,
        feature_count=len(PHYS_NAMES),
    )
    vehicle_timestamps, vehicle_values = _read_window(record.vehicle_path, VEH_COLS, record)
    vehicle_mask = np.isfinite(vehicle_values)
    vehicle_values = np.where(vehicle_mask, vehicle_values, 0.0).astype(np.float32)
    keep = vehicle_mask.any(axis=1)
    return ObservedDualStreamSample(
        sample_id=record.sample_id,
        group_id=record.group_id,
        schema=COGPILOT_SCHEMA,
        physiology_values=physiology_values,
        physiology_timestamps_s=physiology_timestamps,
        physiology_feature_mask=physiology_mask,
        vehicle_values=vehicle_values[keep],
        vehicle_timestamps_s=vehicle_timestamps[keep],
        vehicle_feature_mask=vehicle_mask[keep],
        source_sample_hash=record.source_sample_hash,
        context_duration_s=record.context_duration_s,
    )


def _required_paths(run: Path) -> tuple[tuple[Path, ...], Path, Path] | None:
    physiology = tuple(
        next(run.glob(f"*stream-{token}*_dat.csv"), None) for token in PHYS_FILES
    )
    ecg = next(run.glob(f"*stream-{ECG_FILE}*_dat.csv"), None)
    vehicle = next(run.glob(f"*stream-{VEH_FILE}*_dat.csv"), None)
    if any(path is None for path in physiology) or ecg is None or vehicle is None:
        return None
    return physiology, ecg, vehicle


def _first_timestamp(path: Path) -> float:
    return float(pd.read_csv(path, usecols=["time_dn"], nrows=1).iloc[0, 0])


def _read_window(
    path: Path,
    columns: tuple[str, ...],
    record: CogPilotNativeRecord,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _recording_table(path, columns)
    native_time = frame["time_dn"].to_numpy(np.float64)
    relative = (native_time - record.window_start_native) * record.time_scale
    keep = (relative >= 0.0) & (relative < record.context_duration_s)
    return relative[keep], frame.loc[keep, list(columns)].to_numpy(np.float64, copy=True)


def _recording_table(path, fallback_columns=None):
    for token, columns in (*PHYS_FILES.items(), (ECG_FILE, (ECG_COL,)), (VEH_FILE, VEH_COLS)):
        if f"stream-{token}" in path.name:
            return read_native_table(path, ("time_dn", *columns))
    if fallback_columns is None:
        raise ValueError(f"unrecognized CogPilot recording: {path.name}")
    return read_native_table(path, ("time_dn", *fallback_columns))


def _recording_bounds(path):
    times = _recording_table(path)["time_dn"].to_numpy(np.float64)
    times = times[np.isfinite(times)]
    if not len(times):
        raise ValueError(f"CogPilot recording has no finite timestamps: {path}")
    return float(times.min()), float(times.max())


def _event_times_and_responses(
    vehicle_path: Path,
    eda_path: Path,
    *,
    context_duration_s: float,
    response_pre_s: float,
    response_post_s: float,
    minimum_event_gap_s: float,
    max_events: int | None = 3,
) -> tuple[tuple[float, float], ...]:
    vehicle = _recording_table(vehicle_path, ("aircraft_roll_deg",))
    times = vehicle["time_dn"].to_numpy(np.float64)
    roll = vehicle["aircraft_roll_deg"].to_numpy(np.float64)
    valid = np.isfinite(times) & np.isfinite(roll)
    times, roll = times[valid], roll[valid]
    if len(times) < 3:
        return ()
    seconds = (times - times[0]) * _DAY_TO_SECONDS
    roll_rate = np.abs(np.gradient(roll, seconds))
    distance = max(1, int(minimum_event_gap_s / np.median(np.diff(seconds))))
    peaks, _ = find_peaks(roll_rate, height=np.nanpercentile(roll_rate, 85), distance=distance)
    if max_events is not None and len(peaks) > max_events:
        peaks = np.sort(peaks[np.argsort(roll_rate[peaks])[-max_events:]])
    eda = _recording_table(eda_path, ("eda_hand_l_kOhms",))
    eda_time = eda["time_dn"].to_numpy(np.float64)
    eda_value = _resistance_to_conductance(
        eda["eda_hand_l_kOhms"].to_numpy(np.float64)
    )
    rows = []
    for peak in peaks:
        event_time = times[peak]
        if max_events is None and event_time + response_post_s / _DAY_TO_SECONDS > np.nanmax(eda_time):
            continue
        pre = (
            np.isfinite(eda_value)
            & (eda_time >= event_time - response_pre_s / _DAY_TO_SECONDS)
            & (eda_time < event_time)
        )
        post = (
            np.isfinite(eda_value)
            & (eda_time >= event_time)
            & (eda_time < event_time + response_post_s / _DAY_TO_SECONDS)
        )
        if event_time - context_duration_s / _DAY_TO_SECONDS < max(times[0], eda_time[0]) or not pre.any() or not post.any():
            continue
        rows.append(
            (
                float(event_time),
                float(np.median(eda_value[post]) - np.median(eda_value[pre])),
            )
        )
    return tuple(rows)


def _resistance_to_conductance(values: np.ndarray) -> np.ndarray:
    resistance = np.asarray(values, dtype=np.float64)
    return np.divide(
        1000.0,
        resistance,
        out=np.full_like(resistance, np.nan),
        where=np.isfinite(resistance) & (resistance > 0),
    )
