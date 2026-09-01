"""Native-time CogPilot samples for difficulty and event-response tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
PHYS_NAMES = ("physiology.ppg", "physiology.eda", "physiology.resp", "physiology.hr")
VEH_NAMES = tuple(f"vehicle.{name.removeprefix('aircraft_')}" for name in VEH_COLS)
COGPILOT_SCHEMA = ObservationSchema(
    schema_id="cogpilot_native.v3",
    source_kind="cogpilot_public",
    physiology_feature_names=PHYS_NAMES,
    vehicle_feature_names=VEH_NAMES,
    physiology_feature_roles=tuple("observed" for _ in PHYS_NAMES),
    vehicle_feature_roles=tuple("observed" for _ in VEH_NAMES),
)
_DAY_TO_SECONDS = 86400.0


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
    subject_limit: int = 20,
    window_start_s: float = 60.0,
    context_duration_s: float = 30.0,
    cache_root: str | Path | None = None,
    max_memory_cache_bytes: int = 512 * 1024**2,
) -> LazyObservedDataset[CogPilotNativeRecord]:
    records: list[CogPilotNativeRecord] = []
    for subject in sorted(Path(root).glob("sub-cp*"))[:subject_limit]:
        for run in sorted(subject.glob("ses-*/level-*_run-*")):
            paths = _required_paths(run)
            if paths is None:
                continue
            physiology_paths, ecg_path, vehicle_path = paths
            origin = max(_first_timestamp(path) for path in (*physiology_paths, ecg_path, vehicle_path))
            start = origin + window_start_s / _DAY_TO_SECONDS
            level = int(run.name.split("_")[0].split("-")[1][:2]) - 1
            sample_id = f"{subject.name}::{run.name}"
            source_hash = source_window_hash(
                (*physiology_paths, ecg_path, vehicle_path),
                sample_id,
                start,
                context_duration_s,
            )
            records.append(
                CogPilotNativeRecord(
                    sample_id=sample_id,
                    group_id=subject.name,
                    label=level,
                    context_duration_s=context_duration_s,
                    source_sample_hash=source_hash,
                    physiology_paths=physiology_paths,
                    ecg_path=ecg_path,
                    vehicle_path=vehicle_path,
                    window_start_native=start,
                )
            )
    return _dataset(records, cache_root, max_memory_cache_bytes)


def build_cogpilot_event_response_dataset(
    root: str | Path,
    *,
    subject_limit: int = 20,
    context_duration_s: float = 12.0,
    response_pre_s: float = 2.0,
    response_post_s: float = 8.0,
    minimum_event_gap_s: float = 15.0,
    cache_root: str | Path | None = None,
    max_memory_cache_bytes: int = 512 * 1024**2,
) -> LazyObservedDataset[CogPilotNativeRecord]:
    records: list[CogPilotNativeRecord] = []
    for subject in sorted(Path(root).glob("sub-cp*"))[:subject_limit]:
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
            )
            for event_index, (event_time, response) in enumerate(events):
                sample_id = f"{subject.name}::{run.name}::event{event_index}"
                start = event_time - context_duration_s / _DAY_TO_SECONDS
                source_hash = source_window_hash(
                    (*physiology_paths, ecg_path, vehicle_path),
                    sample_id,
                    start,
                    context_duration_s,
                    response,
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
        indices = tuple(range(next_feature, next_feature + len(columns)))
        physiology_series.append((timestamps, values, indices))
        next_feature += len(columns)
    hr_timestamps, hr_values = _heart_rate_window(record.ecg_path, record)
    physiology_series.append((hr_timestamps, hr_values[:, None], (len(PHYS_NAMES) - 1,)))
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
    try:
        physiology = tuple(next(run.glob(f"*stream-{token}*_dat.csv")) for token in PHYS_FILES)
        return physiology, next(run.glob(f"*stream-{ECG_FILE}*_dat.csv")), next(run.glob(f"*stream-{VEH_FILE}*_dat.csv"))
    except StopIteration:
        return None


def _first_timestamp(path: Path) -> float:
    return float(pd.read_csv(path, usecols=["time_dn"], nrows=1).iloc[0, 0])


def _read_window(
    path: Path,
    columns: tuple[str, ...],
    record: CogPilotNativeRecord,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=("time_dn", *columns))
    native_time = frame["time_dn"].to_numpy(np.float64)
    relative = (native_time - record.window_start_native) * record.time_scale
    keep = (relative >= 0.0) & (relative < record.context_duration_s)
    return relative[keep], frame.loc[keep, list(columns)].to_numpy(np.float64)


def _heart_rate_window(
    path: Path,
    record: CogPilotNativeRecord,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=("time_dn", ECG_COL))
    relative = (frame["time_dn"].to_numpy(np.float64) - record.window_start_native) * record.time_scale
    ecg = frame[ECG_COL].to_numpy(np.float64)
    support = np.isfinite(ecg) & (relative >= -2.0) & (relative < record.context_duration_s + 2.0)
    relative, ecg = relative[support], ecg[support]
    if len(relative) < 3:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    threshold = np.nanpercentile(ecg, 90)
    spacing = max(1, int(0.4 / max(float(np.median(np.diff(relative))), 1e-6)))
    peaks, _ = find_peaks(ecg, height=threshold, distance=spacing)
    if len(peaks) < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    timestamps = 0.5 * (relative[peaks][1:] + relative[peaks][:-1])
    values = np.clip(60.0 / np.maximum(np.diff(relative[peaks]), 1e-3), 30.0, 200.0)
    keep = (timestamps >= 0.0) & (timestamps < record.context_duration_s)
    return timestamps[keep], values[keep]


def _event_times_and_responses(
    vehicle_path: Path,
    eda_path: Path,
    *,
    context_duration_s: float,
    response_pre_s: float,
    response_post_s: float,
    minimum_event_gap_s: float,
) -> tuple[tuple[float, float], ...]:
    vehicle = pd.read_csv(vehicle_path, usecols=("time_dn", "aircraft_roll_deg"))
    times = vehicle["time_dn"].to_numpy(np.float64)
    roll = vehicle["aircraft_roll_deg"].to_numpy(np.float64)
    valid = np.isfinite(times) & np.isfinite(roll)
    times, roll = times[valid], roll[valid]
    seconds = (times - times[0]) * _DAY_TO_SECONDS
    roll_rate = np.abs(np.gradient(roll, seconds))
    distance = max(1, int(minimum_event_gap_s / np.median(np.diff(seconds))))
    peaks, _ = find_peaks(roll_rate, height=np.nanpercentile(roll_rate, 85), distance=distance)
    if len(peaks) > 3:
        peaks = np.sort(peaks[np.argsort(roll_rate[peaks])[-3:]])
    eda = pd.read_csv(eda_path, usecols=("time_dn", "eda_hand_l_kOhms"))
    eda_time = eda["time_dn"].to_numpy(np.float64)
    eda_value = eda["eda_hand_l_kOhms"].to_numpy(np.float64)
    rows = []
    for peak in peaks:
        event_time = times[peak]
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
        rows.append((float(event_time), float(eda_value[post].mean() - eda_value[pre].mean())))
    return tuple(rows)
