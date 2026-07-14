"""Leakage-safe raw-context features for Dingxin feasibility screening."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_selection import f_classif, f_regression

from chronaris.evaluation.application_tasks.dingxin_context_data import (
    DingxinLazyContextIndex,
)
from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream


@dataclass(frozen=True, slots=True)
class DingxinFeatureCache:
    sample_ids: tuple[str, ...]
    timestamps_s: np.ndarray
    physiology_values: np.ndarray
    physiology_mask: np.ndarray
    physiology_age_s: np.ndarray
    vehicle_values: np.ndarray
    vehicle_mask: np.ndarray
    vehicle_age_s: np.ndarray

    def positions(self, sample_ids: Sequence[str]) -> np.ndarray:
        lookup = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        missing = sorted(set(str(value) for value in sample_ids) - set(lookup))
        if missing:
            raise KeyError(f"feature cache lacks samples: {missing[:3]}")
        return np.asarray([lookup[str(value)] for value in sample_ids], dtype=np.int64)


def load_or_build_feature_cache(
    *,
    index: DingxinLazyContextIndex,
    sample_ids: Sequence[str],
    output_path: str | Path,
) -> DingxinFeatureCache:
    path = Path(output_path)
    identifiers = tuple(sorted(set(str(value) for value in sample_ids)))
    expected_hash = _sample_hash(identifiers)
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            actual_ids = tuple(str(value) for value in archive["sample_ids"])
            actual_hash = str(archive["sample_hash"].item())
        if actual_ids == identifiers and actual_hash == expected_hash:
            return load_feature_cache(path)
        raise ValueError("existing raw feature cache has different sample lineage")

    physiology_values = []
    physiology_mask = []
    physiology_age = []
    vehicle_values = []
    vehicle_mask = []
    vehicle_age = []
    timestamps = []
    for sample_id in identifiers:
        batch = index.load_batch((sample_id,))
        physiology = causal_query_stream(batch, stream_name="physiology")
        vehicle = causal_query_stream(batch, stream_name="vehicle")
        timestamps.append(batch.query_timestamps_s[0].detach().cpu().numpy())
        physiology_values.append(physiology.values[0].detach().cpu().numpy())
        physiology_mask.append(physiology.feature_mask[0].detach().cpu().numpy())
        physiology_age.append(
            physiology.observation_age_s[0].detach().cpu().numpy()
        )
        vehicle_values.append(vehicle.values[0].detach().cpu().numpy())
        vehicle_mask.append(vehicle.feature_mask[0].detach().cpu().numpy())
        vehicle_age.append(vehicle.observation_age_s[0].detach().cpu().numpy())

    payload = {
        "sample_ids": np.asarray(identifiers),
        "sample_hash": np.asarray(expected_hash),
        "timestamps_s": np.asarray(timestamps, dtype=np.float64),
        "physiology_values": np.asarray(physiology_values, dtype=np.float32),
        "physiology_mask": np.asarray(physiology_mask, dtype=np.uint8),
        "physiology_age_s": np.asarray(physiology_age, dtype=np.float32),
        "vehicle_values": np.asarray(vehicle_values, dtype=np.float32),
        "vehicle_mask": np.asarray(vehicle_mask, dtype=np.uint8),
        "vehicle_age_s": np.asarray(vehicle_age, dtype=np.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    return load_feature_cache(path)


def load_feature_cache(path: str | Path) -> DingxinFeatureCache:
    with np.load(path, allow_pickle=False) as archive:
        return DingxinFeatureCache(
            sample_ids=tuple(str(value) for value in archive["sample_ids"]),
            timestamps_s=archive["timestamps_s"].astype(np.float64),
            physiology_values=archive["physiology_values"].astype(np.float32),
            physiology_mask=archive["physiology_mask"].astype(bool),
            physiology_age_s=archive["physiology_age_s"].astype(np.float32),
            vehicle_values=archive["vehicle_values"].astype(np.float32),
            vehicle_mask=archive["vehicle_mask"].astype(bool),
            vehicle_age_s=archive["vehicle_age_s"].astype(np.float32),
        )


def summary_features(
    cache: DingxinFeatureCache,
    *,
    sample_ids: Sequence[str],
    modality: str,
    history_s: float,
) -> np.ndarray:
    positions = cache.positions(sample_ids)
    matrices = []
    if modality in {"physiology", "dual"}:
        matrices.append(
            _summary(
                cache.physiology_values[positions],
                cache.physiology_mask[positions],
                cache.physiology_age_s[positions],
                cache.timestamps_s[positions],
                history_s=history_s,
            )
        )
    if modality in {"vehicle", "dual"}:
        matrices.append(
            _summary(
                cache.vehicle_values[positions],
                cache.vehicle_mask[positions],
                cache.vehicle_age_s[positions],
                cache.timestamps_s[positions],
                history_s=history_s,
            )
        )
    if not matrices:
        raise ValueError(f"unsupported modality: {modality}")
    return np.concatenate(matrices, axis=1).astype(np.float32)


def sequence_features(
    cache: DingxinFeatureCache,
    *,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    modality: str,
    target: np.ndarray,
    target_kind: str,
    history_s: float = 30.0,
    channel_budget: int = 32,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build train-normalized [N,C,T] inputs with train-only channel selection."""

    train_positions = cache.positions(train_sample_ids)
    validation_positions = cache.positions(validation_sample_ids)
    streams = []
    if modality in {"physiology", "dual"}:
        streams.append(
            (
                cache.physiology_values,
                cache.physiology_mask,
                cache.physiology_age_s,
            )
        )
    if modality in {"vehicle", "dual"}:
        streams.append(
            (cache.vehicle_values, cache.vehicle_mask, cache.vehicle_age_s)
        )
    if not streams:
        raise ValueError(f"unsupported modality: {modality}")

    train_parts = []
    validation_parts = []
    total_selected = 0
    per_stream_budget = max(1, channel_budget // len(streams))
    for values, mask, age in streams:
        train_values = _history_slice(
            values[train_positions], cache.timestamps_s[train_positions], history_s
        )
        validation_values = _history_slice(
            values[validation_positions],
            cache.timestamps_s[validation_positions],
            history_s,
        )
        train_mask = _history_slice(
            mask[train_positions], cache.timestamps_s[train_positions], history_s
        )
        validation_mask = _history_slice(
            mask[validation_positions],
            cache.timestamps_s[validation_positions],
            history_s,
        )
        train_age = _history_slice(
            age[train_positions], cache.timestamps_s[train_positions], history_s
        )
        validation_age = _history_slice(
            age[validation_positions],
            cache.timestamps_s[validation_positions],
            history_s,
        )
        selected = select_train_channels(
            train_values,
            train_mask,
            target=target,
            target_kind=target_kind,
            channel_budget=min(per_stream_budget, train_values.shape[-1]),
        )
        total_selected += len(selected)
        normalized_train, normalized_validation = _robust_normalize_sequence(
            train_values[:, :, selected],
            validation_values[:, :, selected],
            train_mask[:, :, selected],
            validation_mask[:, :, selected],
        )
        train_parts.extend(
            (
                normalized_train.transpose(0, 2, 1),
                train_mask[:, :, selected].astype(np.float32).transpose(0, 2, 1),
                _normalized_age(train_age[:, :, selected]).transpose(0, 2, 1),
            )
        )
        validation_parts.extend(
            (
                normalized_validation.transpose(0, 2, 1),
                validation_mask[:, :, selected]
                .astype(np.float32)
                .transpose(0, 2, 1),
                _normalized_age(validation_age[:, :, selected]).transpose(0, 2, 1),
            )
        )
    return (
        np.ascontiguousarray(np.concatenate(train_parts, axis=1), dtype=np.float32),
        np.ascontiguousarray(
            np.concatenate(validation_parts, axis=1), dtype=np.float32
        ),
        total_selected,
    )


def select_train_channels(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    target: np.ndarray,
    target_kind: str,
    channel_budget: int,
) -> np.ndarray:
    mean, std, _minimum, _maximum, first, last, _count = _masked_statistics(
        values,
        mask,
    )
    summaries = np.nan_to_num(np.stack((mean, std, last - first), axis=2))
    scores = np.zeros(values.shape[-1], dtype=np.float64)
    for statistic in range(summaries.shape[2]):
        matrix = summaries[:, :, statistic]
        variable = np.flatnonzero(np.ptp(matrix, axis=0) > 1e-10)
        if not len(variable):
            continue
        try:
            if target_kind == "classification":
                statistic_scores, _ = f_classif(matrix[:, variable], target)
            else:
                statistic_scores, _ = f_regression(matrix[:, variable], target)
        except ValueError:
            statistic_scores = np.zeros(len(variable), dtype=np.float64)
        scores[variable] = np.maximum(
            scores[variable],
            np.nan_to_num(statistic_scores, nan=0.0),
        )
    observed = np.flatnonzero(mask.any(axis=(0, 1)))
    if not len(observed):
        raise ValueError("stream has no observed train channel")
    ranked = observed[np.argsort(scores[observed], kind="mergesort")[::-1]]
    return np.sort(ranked[:channel_budget]).astype(np.int64)


def _summary(values, mask, age, timestamps, *, history_s):
    values = _history_slice(values, timestamps, history_s)
    mask = _history_slice(mask, timestamps, history_s)
    age = _history_slice(age, timestamps, history_s)
    mean, std, minimum, maximum, first, last, count = _masked_statistics(
        values,
        mask,
    )
    last_index = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    valid_fraction = mask.mean(axis=1)
    last_age = np.take_along_axis(age, last_index[:, None, :], axis=1)[:, 0]
    last_age = np.where(count > 0, last_age, 0.0)
    mean_age = np.divide(
        np.where(mask, age, 0.0).sum(axis=1),
        count,
        out=np.zeros_like(mean, dtype=np.float64),
        where=count > 0,
    )
    features = np.concatenate(
        (
            last,
            mean,
            std,
            minimum,
            maximum,
            last - first,
            valid_fraction,
            _normalized_age(last_age),
            _normalized_age(mean_age),
        ),
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _masked_statistics(values, mask):
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1, dtype=np.float64)
    mean = np.divide(
        total,
        count,
        out=np.zeros_like(total, dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(mask, values - mean[:, None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1, dtype=np.float64),
        count,
        out=np.zeros_like(total, dtype=np.float64),
        where=count > 0,
    )
    minimum = np.where(mask, values, np.inf).min(axis=1)
    maximum = np.where(mask, values, -np.inf).max(axis=1)
    minimum = np.where(count > 0, minimum, 0.0)
    maximum = np.where(count > 0, maximum, 0.0)
    first_index = np.argmax(mask, axis=1)
    last_index = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    first = np.take_along_axis(values, first_index[:, None, :], axis=1)[:, 0]
    last = np.take_along_axis(values, last_index[:, None, :], axis=1)[:, 0]
    first = np.where(count > 0, first, 0.0)
    last = np.where(count > 0, last, 0.0)
    return mean, np.sqrt(variance), minimum, maximum, first, last, count


def _history_slice(values, timestamps, history_s):
    if history_s >= 29.999:
        return values
    final = timestamps.max(axis=1, keepdims=True)
    selected = timestamps >= (final - float(history_s) - 1e-9)
    counts = selected.sum(axis=1)
    if len(set(int(value) for value in counts)) != 1:
        raise ValueError("query grids do not share a common history slice")
    count = int(counts[0])
    return values[:, -count:]


def _robust_normalize_sequence(train, validation, train_mask, validation_mask):
    masked = np.where(train_mask, train, np.nan)
    median = np.nanmedian(masked, axis=(0, 1))
    q25 = np.nanquantile(masked, 0.25, axis=(0, 1))
    q75 = np.nanquantile(masked, 0.75, axis=(0, 1))
    scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    normalized_train = np.where(train_mask, (train - median) / scale, 0.0)
    normalized_validation = np.where(
        validation_mask, (validation - median) / scale, 0.0
    )
    return (
        np.clip(normalized_train, -20.0, 20.0).astype(np.float32),
        np.clip(normalized_validation, -20.0, 20.0).astype(np.float32),
    )


def _normalized_age(age):
    return np.log1p(np.clip(np.nan_to_num(age, nan=30.0), 0.0, 30.0)) / np.log1p(
        30.0
    )


def _sample_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
