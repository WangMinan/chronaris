"""Streaming helpers for Stage I fixed-grid sequence export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chronaris.features.stage_i_feature_helpers import WindowSpec


@dataclass(slots=True)
class RunningSequenceAccumulator:
    """Accumulate per-bin means for one fixed window."""

    value_columns: tuple[str, ...]
    target_steps: int
    window_start: float
    window_end: float
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None

    def update(
        self,
        frame: pd.DataFrame,
        *,
        time_column: str,
        invalid_nonpositive_columns: set[str],
    ) -> None:
        if frame.empty:
            return
        if self.sums is None or self.counts is None:
            shape = (self.target_steps, len(self.value_columns))
            self.sums = np.zeros(shape, dtype=np.float64)
            self.counts = np.zeros(shape, dtype=np.float64)
        numeric = frame.loc[:, list(self.value_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        for column in invalid_nonpositive_columns:
            if column in numeric.columns:
                numeric[column] = numeric[column].where(numeric[column] > 0)
        times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(
            dtype=float,
        )
        valid_time = np.isfinite(times)
        if not np.any(valid_time):
            return
        times = times[valid_time]
        numeric = numeric.loc[valid_time].reset_index(drop=True)
        duration = max(self.window_end - self.window_start, 1e-6)
        relative = (times - self.window_start) / duration
        step_indices = np.floor(relative * self.target_steps).astype(int)
        step_indices = np.clip(step_indices, 0, self.target_steps - 1)
        for column_index, column_name in enumerate(self.value_columns):
            values = pd.to_numeric(
                numeric[column_name],
                errors="coerce",
            ).to_numpy(dtype=float)
            valid_value = np.isfinite(values)
            if not np.any(valid_value):
                continue
            np.add.at(
                self.sums[:, column_index],
                step_indices[valid_value],
                values[valid_value],
            )
            np.add.at(
                self.counts[:, column_index],
                step_indices[valid_value],
                1.0,
            )

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sums is None or self.counts is None:
            values = np.zeros(
                (self.target_steps, len(self.value_columns)),
                dtype=np.float32,
            )
            mask = np.zeros((self.target_steps,), dtype=np.uint8)
            return values, mask
        values = np.divide(
            self.sums,
            self.counts,
            out=np.zeros_like(self.sums, dtype=np.float64),
            where=self.counts > 0,
        ).astype(np.float32)
        mask = np.any(self.counts > 0, axis=1).astype(np.uint8)
        return values, mask


@dataclass(slots=True)
class WindowSequenceStreamState:
    """Consume one sorted recording stream into fixed-grid window arrays."""

    window_specs: tuple[WindowSpec, ...]
    value_columns: tuple[str, ...]
    target_steps: int
    invalid_nonpositive_columns: set[str]
    current_index: int = 0
    accumulator: RunningSequenceAccumulator | None = None

    def consume_frame(
        self,
        frame: pd.DataFrame,
        *,
        time_column: str,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        if frame.empty or self.current_index >= len(self.window_specs):
            return []
        frame = frame.reset_index(drop=True)
        times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(
            dtype=float,
        )
        if len(times) == 0:
            return []
        rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        while self.current_index < len(self.window_specs):
            spec = self.window_specs[self.current_index]
            if float(times[-1]) < spec.start:
                break
            if self.accumulator is None:
                self.accumulator = RunningSequenceAccumulator(
                    value_columns=self.value_columns,
                    target_steps=self.target_steps,
                    window_start=spec.start,
                    window_end=spec.end,
                )
            start_index = int(np.searchsorted(times, spec.start, side="left"))
            if start_index >= len(times):
                break
            end_index = int(np.searchsorted(times, spec.end, side="left"))
            if start_index < end_index:
                self.accumulator.update(
                    frame.iloc[start_index:end_index].reset_index(drop=True),
                    time_column=time_column,
                    invalid_nonpositive_columns=self.invalid_nonpositive_columns,
                )
            if float(times[-1]) < spec.end:
                break
            rows.append(self._flush_current())
        return rows

    def finish(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        while self.current_index < len(self.window_specs):
            rows.append(self._flush_current())
        return rows

    def _flush_current(self) -> tuple[str, np.ndarray, np.ndarray]:
        if self.current_index >= len(self.window_specs):
            raise RuntimeError("window sequence stream state is already exhausted.")
        spec = self.window_specs[self.current_index]
        if self.accumulator is None:
            self.accumulator = RunningSequenceAccumulator(
                value_columns=self.value_columns,
                target_steps=self.target_steps,
                window_start=spec.start,
                window_end=spec.end,
            )
        values, mask = self.accumulator.to_arrays()
        self.accumulator = None
        self.current_index += 1
        return spec.sample_id, values, mask


def stream_window_sequence_arrays_from_csv(
    *,
    csv_path: Path,
    recording_id: str,
    window_specs: Mapping[str, tuple[WindowSpec, ...]],
    value_columns: tuple[str, ...],
    target_steps: int,
    invalid_nonpositive_columns: Iterable[str],
    time_column: str = "TimeSecs",
    chunksize: int = 200_000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    specs = window_specs.get(recording_id)
    if not specs:
        return {}
    state = WindowSequenceStreamState(
        window_specs=specs,
        value_columns=value_columns,
        target_steps=target_steps,
        invalid_nonpositive_columns=set(invalid_nonpositive_columns),
    )
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    usecols = [time_column, *value_columns]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        rows.extend(state.consume_frame(chunk, time_column=time_column))
    rows.extend(state.finish())
    return {sample_id: (values, mask) for sample_id, values, mask in rows}


def stream_window_sequence_arrays_from_parquet(
    *,
    parquet_path: Path,
    key_columns: tuple[str, ...],
    recording_id_builder: Callable[[tuple[object, ...]], str],
    window_specs: Mapping[str, tuple[WindowSpec, ...]],
    value_columns: tuple[str, ...],
    target_steps: int,
    invalid_nonpositive_columns: Iterable[str],
    time_column: str = "datetime",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not value_columns:
        return {}
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    states = {
        recording_id: WindowSequenceStreamState(
            window_specs=specs,
            value_columns=value_columns,
            target_steps=target_steps,
            invalid_nonpositive_columns=set(invalid_nonpositive_columns),
        )
        for recording_id, specs in window_specs.items()
        if specs
    }
    reader = pq.ParquetFile(parquet_path)
    for batch in reader.iter_batches(
        batch_size=65_536,
        columns=[*key_columns, time_column, *value_columns],
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["__time_seconds"] = (
            frame[time_column].astype("datetime64[ns]").astype("int64")
            / 1_000_000_000.0
        )
        for key, group in frame.groupby(list(key_columns), sort=False):
            normalized_key = key if isinstance(key, tuple) else (key,)
            recording_id = recording_id_builder(_normalize_key(normalized_key))
            state = states.get(recording_id)
            if state is None:
                continue
            group = group.sort_values("__time_seconds").reset_index(drop=True)
            rows.extend(state.consume_frame(group, time_column="__time_seconds"))
    for state in states.values():
        rows.extend(state.finish())
    return {sample_id: (values, mask) for sample_id, values, mask in rows}


def build_common_time_axis(
    *,
    entry_count: int,
    target_steps: int,
    window_duration_s: float,
) -> np.ndarray:
    centers = (
        (np.arange(target_steps, dtype=np.float32) + 0.5)
        * float(window_duration_s)
        / float(target_steps)
    )
    return np.repeat(centers.reshape(1, target_steps), entry_count, axis=0)


def infer_parquet_value_columns(
    parquet_path: Path,
    *,
    selector: Callable[[tuple[str, ...]], tuple[str, ...]],
) -> tuple[str, ...]:
    return selector(tuple(pq.ParquetFile(parquet_path).schema.names))


def infer_csv_value_columns(
    csv_path: Path,
    *,
    selector: Callable[[tuple[str, ...]], tuple[str, ...]],
) -> tuple[str, ...]:
    return selector(tuple(pd.read_csv(csv_path, nrows=0).columns))


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
