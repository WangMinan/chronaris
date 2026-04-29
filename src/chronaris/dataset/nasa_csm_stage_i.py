"""NASA CSM adapter for Stage I attention-state windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from chronaris.dataset.stage_i_contracts import StageITaskEntry, isoformat_utc

DATASET_ID = "nasa_csm"
WINDOW_DURATION_SECONDS = 5.0
WINDOW_STEP_SECONDS = 5.0
BACKGROUND_ROLE = "inventory_only"
PRIMARY_ROLE = "primary"

EVENT_LABELS = {
    0: "background",
    1: "SS",
    2: "CA",
    5: "DA",
}

BENCHMARK_TYPES = {"CA", "DA", "SS"}


@dataclass(frozen=True, slots=True)
class NASACSMPreparedTaskSet:
    """Prepared NASA CSM Stage I entries."""

    entries: tuple[StageITaskEntry, ...]
    subset_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _EventSegment:
    event_code: int
    start_s: float
    end_s: float


def build_nasa_csm_task_entries(dataset_root: str | Path) -> NASACSMPreparedTaskSet:
    """Build the NASA CSM Stage I manifest from extracted CSV files."""

    extracted_root = Path(dataset_root) / DATASET_ID / "extracted"
    entries: list[StageITaskEntry] = []
    for path in sorted(extracted_root.glob("*/*.csv")):
        subject_id = f"subject_{path.parent.name}"
        recording_type = _parse_recording_type(path)
        subset_id = "benchmark" if recording_type in BENCHMARK_TYPES else "loft"
        recording_id = f"nasa_csm__{subject_id}__{recording_type.lower()}"
        anchor = _recording_anchor(subject_id, recording_type)
        window_index = 0
        for segment in _iter_event_segments(path):
            for window_start_s, window_end_s in _iter_segment_windows(segment):
                event_label = EVENT_LABELS[segment.event_code]
                training_role = PRIMARY_ROLE if segment.event_code != 0 else BACKGROUND_ROLE
                context_payload = {
                    "objective_label_text": event_label,
                    "event_code": segment.event_code,
                    "recording_type": recording_type,
                    "source_partition": subset_id,
                    "window_start_s": window_start_s,
                    "window_end_s": window_end_s,
                    "time_reference": "synthetic_utc_from_TimeSecs",
                    "window_strategy": "fixed_5s_nonzero_single_event",
                }
                window_start = anchor + timedelta(seconds=window_start_s)
                window_end = anchor + timedelta(seconds=window_end_s)
                entries.append(
                    StageITaskEntry(
                        sample_id=f"{recording_id}__window_{window_index:05d}",
                        dataset_id=DATASET_ID,
                        subset_id=subset_id,
                        subject_id=subject_id,
                        session_id=recording_id,
                        split_group=subject_id,
                        training_role=training_role,
                        sample_granularity="window",
                        recording_id=recording_id,
                        window_index=window_index,
                        window_duration_s=WINDOW_DURATION_SECONDS,
                        task_family="attention_state",
                        label_namespace="attention_state",
                        window_start_utc=isoformat_utc(window_start),
                        window_end_utc=isoformat_utc(window_end),
                        source_refs={
                            "csv_path": str(path.relative_to(Path(dataset_root))),
                        },
                        objective_label_name="attention_state",
                        objective_label_value=segment.event_code,
                        context_payload=context_payload,
                    )
                )
                window_index += 1
    subset_counts = {
        subset_id: sum(1 for entry in entries if entry.subset_id == subset_id)
        for subset_id in ("benchmark", "loft")
    }
    return NASACSMPreparedTaskSet(entries=tuple(entries), subset_counts=subset_counts)


def _parse_recording_type(path: Path) -> str:
    return path.stem.split("_", maxsplit=1)[1].upper()


def _recording_anchor(subject_id: str, recording_type: str) -> datetime:
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    subject_offset = int(subject_id.split("_")[-1])
    type_offset = {"CA": 0, "DA": 6, "SS": 12, "LOFT": 18}[recording_type]
    return base + timedelta(days=subject_offset, hours=type_offset)


def _iter_event_segments(path: Path) -> Iterator[_EventSegment]:
    current_event: int | None = None
    current_start: float | None = None
    last_time: float | None = None
    last_interval = 0.0

    for chunk in pd.read_csv(path, usecols=["TimeSecs", "Event"], chunksize=200_000):
        if chunk.empty:
            continue
        time_values = pd.to_numeric(chunk["TimeSecs"], errors="coerce").to_numpy(dtype=float)
        event_values = pd.to_numeric(chunk["Event"], errors="coerce").fillna(0).to_numpy(dtype=float).astype(int)
        valid = np.isfinite(time_values)
        time_values = time_values[valid]
        event_values = event_values[valid]
        if len(time_values) == 0:
            continue

        diffs = np.diff(time_values)
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs):
            last_interval = float(np.median(positive_diffs))

        change_indices = np.flatnonzero(event_values[1:] != event_values[:-1]) + 1
        boundaries = np.concatenate(([0], change_indices, [len(event_values)]))
        for start_index, end_index in zip(boundaries[:-1], boundaries[1:], strict=True):
            event_code = int(event_values[start_index])
            segment_start = float(time_values[start_index])
            if current_event is None:
                current_event = event_code
                current_start = segment_start
            elif start_index == 0 and event_code == current_event:
                pass
            else:
                yield _EventSegment(
                    event_code=current_event,
                    start_s=float(current_start),
                    end_s=segment_start,
                )
                current_event = event_code
                current_start = segment_start
            last_time = float(time_values[end_index - 1])

    if current_event is not None and current_start is not None and last_time is not None:
        yield _EventSegment(
            event_code=current_event,
            start_s=float(current_start),
            end_s=float(last_time + max(last_interval, 0.0)),
        )


def _iter_segment_windows(segment: _EventSegment) -> Iterable[tuple[float, float]]:
    cursor = segment.start_s
    while cursor + WINDOW_DURATION_SECONDS <= segment.end_s + 1e-9:
        yield cursor, cursor + WINDOW_DURATION_SECONDS
        cursor += WINDOW_STEP_SECONDS
