"""Bounded lazy loading for native-time dual-stream samples."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
)


RecordT = TypeVar("RecordT", bound="NativeSampleRecord")


@dataclass(frozen=True, slots=True)
class NativeSampleRecord:
    sample_id: str
    group_id: str
    label: float | int
    context_duration_s: float
    source_sample_hash: str


class LazyObservedDataset(Generic[RecordT]):
    """Load only requested samples and keep bounded memory/disk caches."""

    def __init__(
        self,
        records: Sequence[RecordT],
        *,
        schema: ObservationSchema,
        loader: Callable[[RecordT], ObservedDualStreamSample],
        cache_root: str | Path | None = None,
        max_memory_cache_bytes: int = 512 * 1024**2,
    ) -> None:
        if not records or max_memory_cache_bytes < 0:
            raise ValueError("native dataset records must be non-empty and cache size non-negative")
        self.records = tuple(records)
        self.schema = schema
        self._loader = loader
        self._by_id = {record.sample_id: record for record in self.records}
        if len(self._by_id) != len(self.records):
            raise ValueError("native dataset sample IDs must be unique")
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.max_memory_cache_bytes = int(max_memory_cache_bytes)
        self._memory: OrderedDict[str, ObservedDualStreamSample] = OrderedDict()
        self._memory_bytes = 0

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(record.group_id for record in self.records)

    @property
    def labels(self) -> tuple[float | int, ...]:
        return tuple(record.label for record in self.records)

    def load_sample(self, sample_id: str) -> ObservedDualStreamSample:
        record = self._by_id.get(str(sample_id))
        if record is None:
            raise KeyError(f"unknown native sample: {sample_id}")
        cached = self._memory.pop(record.sample_id, None)
        if cached is not None:
            self._memory[record.sample_id] = cached
            return cached
        cache_path = self._cache_path(record)
        sample = (
            self._load_disk_cache(cache_path, record)
            if cache_path is not None and cache_path.exists()
            else self._loader(record)
        )
        if sample.sample_id != record.sample_id or sample.source_sample_hash != record.source_sample_hash:
            raise ValueError("native sample loader changed record lineage")
        if cache_path is not None and not cache_path.exists():
            self._write_disk_cache(cache_path, sample)
        self._remember(sample)
        return sample

    def batch_provider(self, sample_ids: Sequence[str]):
        return collate_observation_samples([self.load_sample(value) for value in sample_ids])

    def batch_ids(
        self,
        batch_size: int,
        sample_ids: Iterable[str] | None = None,
    ) -> tuple[tuple[str, ...], ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = tuple(sample_ids) if sample_ids is not None else self.sample_ids
        return tuple(values[index : index + batch_size] for index in range(0, len(values), batch_size))

    def _cache_path(self, record: RecordT) -> Path | None:
        if self.cache_root is None:
            return None
        name = hashlib.sha256(record.sample_id.encode()).hexdigest()[:20]
        return self.cache_root / f"{name}_{record.source_sample_hash[:12]}.npz"

    def _load_disk_cache(self, path: Path, record: RecordT) -> ObservedDualStreamSample:
        with np.load(path, allow_pickle=False) as payload:
            return ObservedDualStreamSample(
                sample_id=record.sample_id,
                group_id=record.group_id,
                schema=self.schema,
                physiology_values=payload["physiology_values"],
                physiology_timestamps_s=payload["physiology_timestamps_s"],
                physiology_feature_mask=payload["physiology_feature_mask"],
                vehicle_values=payload["vehicle_values"],
                vehicle_timestamps_s=payload["vehicle_timestamps_s"],
                vehicle_feature_mask=payload["vehicle_feature_mask"],
                source_sample_hash=record.source_sample_hash,
                context_duration_s=record.context_duration_s,
            )

    @staticmethod
    def _write_disk_cache(path: Path, sample: ObservedDualStreamSample) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    physiology_values=sample.physiology_values,
                    physiology_timestamps_s=sample.physiology_timestamps_s,
                    physiology_feature_mask=sample.physiology_feature_mask,
                    vehicle_values=sample.vehicle_values,
                    vehicle_timestamps_s=sample.vehicle_timestamps_s,
                    vehicle_feature_mask=sample.vehicle_feature_mask,
                )
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _remember(self, sample: ObservedDualStreamSample) -> None:
        size = _sample_nbytes(sample)
        if size > self.max_memory_cache_bytes:
            return
        while self._memory and self._memory_bytes + size > self.max_memory_cache_bytes:
            _, removed = self._memory.popitem(last=False)
            self._memory_bytes -= _sample_nbytes(removed)
        self._memory[sample.sample_id] = sample
        self._memory_bytes += size


def merge_native_feature_series(
    series: Sequence[tuple[np.ndarray, np.ndarray, Sequence[int]]],
    *,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a timestamp union with sparse per-feature masks and no interpolation."""

    nonempty = [item for item in series if len(item[0])]
    if not nonempty:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, feature_count), dtype=np.float32),
            np.empty((0, feature_count), dtype=bool),
        )
    union = np.unique(np.concatenate([item[0] for item in nonempty]).astype(np.float64))
    values = np.zeros((len(union), feature_count), dtype=np.float32)
    mask = np.zeros((len(union), feature_count), dtype=bool)
    for timestamps, source_values, feature_indices in nonempty:
        rows = np.searchsorted(union, timestamps)
        for source_column, target_column in enumerate(feature_indices):
            valid = np.isfinite(source_values[:, source_column])
            values[rows[valid], target_column] = source_values[valid, source_column]
            mask[rows[valid], target_column] = True
    keep = mask.any(axis=1)
    return union[keep], values[keep], mask[keep]


def source_window_hash(paths: Sequence[Path], *parts: object) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    for part in parts:
        digest.update(repr(part).encode())
    return digest.hexdigest()


def _sample_nbytes(sample: ObservedDualStreamSample) -> int:
    arrays = (
        sample.physiology_values,
        sample.physiology_timestamps_s,
        sample.physiology_feature_mask,
        sample.vehicle_values,
        sample.vehicle_timestamps_s,
        sample.vehicle_feature_mask,
    )
    return sum(value.nbytes for value in arrays)
