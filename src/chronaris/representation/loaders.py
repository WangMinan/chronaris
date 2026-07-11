"""Observed-only loaders for the simulation benchmark and fixed Dingxin snapshot."""

from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation.snapshot_io import iter_raw_point_snapshot
from chronaris.representation.collation import (
    ObservedDualStreamSample,
    stable_observed_sample_hash,
)
from chronaris.representation.contracts import (
    ObservationSchema,
    RepresentationContractError,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


SIMULATION_OBSERVED_KEYS = frozenset(
    {
        "vehicle_observed_time_s",
        "vehicle_values",
        "vehicle_feature_names",
        "physiology_observed_time_s",
        "physiology_values",
        "physiology_feature_names",
    }
)


@dataclass(frozen=True, slots=True)
class DingxinObservationSchemaPlan:
    """One common feature order plus sortie-specific raw field mappings."""

    schema: ObservationSchema
    physiology_raw_to_index: Mapping[str, Mapping[str, int]]
    vehicle_raw_to_index: Mapping[str, Mapping[str, int]]
    snapshot_manifest: Mapping[str, object]
    snapshot_root: str


@dataclass(frozen=True, slots=True)
class CachedSparseSnapshotStream:
    timestamps_epoch_s: np.ndarray
    indptr: np.ndarray
    feature_indices: np.ndarray
    feature_values: np.ndarray
    feature_count: int


@dataclass(frozen=True, slots=True)
class DingxinSnapshotPointCache:
    """Verified allowed-field CSR cache; contexts remain sliced on demand."""

    streams_by_path: Mapping[str, CachedSparseSnapshotStream]
    file_sha256: Mapping[str, str]


def build_dingxin_snapshot_point_cache(
    plan: DingxinObservationSchemaPlan,
) -> DingxinSnapshotPointCache:
    root = Path(plan.snapshot_root)
    streams_by_path = {}
    file_hashes = {}
    for item in plan.snapshot_manifest["files"]:
        path = root / str(item["relative_path"])
        actual_hash = sha256_file(path)
        if actual_hash != str(item["sha256"]):
            raise RepresentationContractError(
                f"Dingxin snapshot cache hash mismatch: {path}"
            )
        sortie_id = str(item["sortie_id"])
        raw_to_index = (
            plan.physiology_raw_to_index[sortie_id]
            if item["stream_kind"] == "physiology"
            else plan.vehicle_raw_to_index[sortie_id]
        )
        timestamps = array("d")
        indices = array("i")
        values = array("f")
        indptr = array("q", [0])
        source_point_count = 0
        for point in iter_raw_point_snapshot(path):
            source_point_count += 1
            start_index = len(indices)
            for field, raw_value in sorted(point.values.items()):
                index = raw_to_index.get(f"{point.measurement}.{field}")
                if index is None:
                    continue
                value = _finite_float(raw_value)
                if value is None:
                    continue
                indices.append(index)
                values.append(value)
            if len(indices) == start_index:
                continue
            timestamps.append(point.timestamp.timestamp())
            indptr.append(len(indices))
        if source_point_count != int(item["point_count"]):
            raise RepresentationContractError(
                f"Dingxin snapshot cache point count mismatch: {path}"
            )
        streams_by_path[str(path.resolve())] = CachedSparseSnapshotStream(
            timestamps_epoch_s=np.frombuffer(timestamps, dtype=np.float64),
            indptr=np.frombuffer(indptr, dtype=np.int64),
            feature_indices=np.frombuffer(indices, dtype=np.int32),
            feature_values=np.frombuffer(values, dtype=np.float32),
            feature_count=len(raw_to_index),
        )
        file_hashes[str(path.resolve())] = actual_hash
    return DingxinSnapshotPointCache(
        streams_by_path=streams_by_path,
        file_sha256=file_hashes,
    )


def load_simulation_observed_context(
    raw_dual_stream_path: str | Path,
    *,
    context_start_s: float,
    context_duration_s: float = 30.0,
    sample_id: str | None = None,
    group_id: str | None = None,
) -> ObservedDualStreamSample:
    """Load one simulation context without opening its separate truth archive."""

    path = Path(raw_dual_stream_path)
    if path.name != "raw_dual_stream.npz":
        raise RepresentationContractError(
            "simulation model loader only accepts raw_dual_stream.npz"
        )
    if context_start_s < 0 or context_duration_s <= 0:
        raise RepresentationContractError("simulation context bounds are invalid")
    with np.load(path, allow_pickle=False) as archive:
        keys = frozenset(archive.files)
        if keys != SIMULATION_OBSERVED_KEYS:
            extra = sorted(keys - SIMULATION_OBSERVED_KEYS)
            missing = sorted(SIMULATION_OBSERVED_KEYS - keys)
            raise RepresentationContractError(
                f"simulation observed archive schema mismatch; extra={extra}, missing={missing}"
            )
        payload = {name: archive[name] for name in archive.files}
    physiology_names = tuple(
        f"physiology.{value}" for value in payload["physiology_feature_names"].astype(str)
    )
    vehicle_names = tuple(
        f"vehicle.{value}" for value in payload["vehicle_feature_names"].astype(str)
    )
    raw_hash = sha256_file(path)
    schema = ObservationSchema(
        schema_id="aviation_simulation_observed.v1",
        source_kind="method_independent_simulation",
        physiology_feature_names=physiology_names,
        vehicle_feature_names=vehicle_names,
        physiology_feature_roles=tuple("observed" for _ in physiology_names),
        vehicle_feature_roles=tuple("observed" for _ in vehicle_names),
        source_manifest_sha256=raw_hash,
    )
    manifest = _read_optional_json(path.with_name("scenario_manifest.json"))
    resolved_sample_id = sample_id or str(manifest.get("sample_id") or path.parent.name)
    resolved_group_id = group_id or str(
        manifest.get("profile_id") or path.parent.parent.parent.name
    )
    end_s = context_start_s + context_duration_s
    physiology = _slice_dense_observed_stream(
        timestamps=np.asarray(payload["physiology_observed_time_s"], dtype=np.float64),
        values=np.asarray(payload["physiology_values"], dtype=np.float32),
        start_s=context_start_s,
        end_s=end_s,
    )
    vehicle = _slice_dense_observed_stream(
        timestamps=np.asarray(payload["vehicle_observed_time_s"], dtype=np.float64),
        values=np.asarray(payload["vehicle_values"], dtype=np.float32),
        start_s=context_start_s,
        end_s=end_s,
    )
    return ObservedDualStreamSample(
        sample_id=f"{resolved_sample_id}::context_{context_start_s:07.3f}",
        group_id=resolved_group_id,
        schema=schema,
        physiology_values=physiology["values"],
        physiology_timestamps_s=physiology["timestamps"],
        physiology_feature_mask=physiology["feature_mask"],
        vehicle_values=vehicle["values"],
        vehicle_timestamps_s=vehicle["timestamps"],
        vehicle_feature_mask=vehicle["feature_mask"],
        source_sample_hash=stable_observed_sample_hash(
            raw_hash,
            f"{context_start_s:.9f}",
            f"{context_duration_s:.9f}",
            schema.schema_sha256,
        ),
        context_duration_s=context_duration_s,
    )


def build_dingxin_observation_schema_plan(
    *,
    snapshot_root: str | Path,
    field_role_manifest_path: str | Path,
) -> DingxinObservationSchemaPlan:
    """Build one leakage-safe feature order shared by every fixed Dingxin sortie."""

    root = Path(snapshot_root)
    manifest_path = root / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles_path = Path(field_role_manifest_path)
    roles = pd.read_csv(roles_path)
    required = {
        "sortie_id",
        "stream_kind",
        "measurement",
        "source_field",
        "semantic_category",
        "allowed_in_maneuver_input",
    }
    missing = sorted(required - set(roles.columns))
    if missing:
        raise RepresentationContractError(
            f"field role manifest is missing columns: {missing}"
        )
    plans_by_sortie = _plans_by_sortie(manifest)
    physiology_by_sortie: dict[str, dict[str, tuple[str, str]]] = {}
    vehicle_by_sortie: dict[str, dict[str, tuple[str, str]]] = {}
    excluded: list[str] = []
    for sortie_id, frame in roles.groupby("sortie_id", sort=True):
        sortie = str(sortie_id)
        if sortie not in plans_by_sortie:
            raise RepresentationContractError(
                f"field role sortie is absent from snapshot manifest: {sortie}"
            )
        vehicle_channels = {
            measurement: index
            for index, measurement in enumerate(
                plans_by_sortie[sortie]["vehicle_measurements"]
            )
        }
        physiology_rows: dict[str, tuple[str, str]] = {}
        vehicle_rows: dict[str, tuple[str, str]] = {}
        for row in frame.itertuples(index=False):
            raw_name = f"{row.measurement}.{row.source_field}"
            if not bool(row.allowed_in_maneuver_input):
                excluded.append(f"{sortie}:{raw_name}")
                continue
            role = f"observed:{row.semantic_category}"
            if str(row.stream_kind) == "physiology":
                canonical = f"physiology.{raw_name}"
                physiology_rows[canonical] = (raw_name, role)
            elif str(row.stream_kind) == "vehicle":
                if str(row.measurement) not in vehicle_channels:
                    continue
                canonical = (
                    f"vehicle.channel_{vehicle_channels[str(row.measurement)]:02d}."
                    f"{row.source_field}"
                )
                vehicle_rows[canonical] = (raw_name, role)
        physiology_by_sortie[sortie] = physiology_rows
        vehicle_by_sortie[sortie] = vehicle_rows
    physiology_names = _common_feature_names(physiology_by_sortie, "physiology")
    vehicle_names = _common_feature_names(vehicle_by_sortie, "vehicle")
    physiology_roles = tuple(
        physiology_by_sortie[next(iter(sorted(physiology_by_sortie)))][name][1]
        for name in physiology_names
    )
    vehicle_roles = tuple(
        vehicle_by_sortie[next(iter(sorted(vehicle_by_sortie)))][name][1]
        for name in vehicle_names
    )
    source_manifest_hash = _combined_file_hash((manifest_path, roles_path))
    schema = ObservationSchema(
        schema_id="dingxin_fixed_common_observed.v1",
        source_kind="dingxin_fixed_snapshot",
        physiology_feature_names=physiology_names,
        vehicle_feature_names=vehicle_names,
        physiology_feature_roles=physiology_roles,
        vehicle_feature_roles=vehicle_roles,
        excluded_feature_names=tuple(sorted(set(excluded))),
        source_manifest_sha256=source_manifest_hash,
    )
    return DingxinObservationSchemaPlan(
        schema=schema,
        physiology_raw_to_index={
            sortie: {
                feature_map[name][0]: index
                for index, name in enumerate(physiology_names)
            }
            for sortie, feature_map in physiology_by_sortie.items()
        },
        vehicle_raw_to_index={
            sortie: {
                feature_map[name][0]: index
                for index, name in enumerate(vehicle_names)
            }
            for sortie, feature_map in vehicle_by_sortie.items()
        },
        snapshot_manifest=manifest,
        snapshot_root=str(root),
    )


def load_dingxin_observed_context(
    plan: DingxinObservationSchemaPlan,
    context: Mapping[str, object],
    *,
    point_cache: DingxinSnapshotPointCache | None = None,
) -> ObservedDualStreamSample:
    """Read one 30-second Dingxin context from the frozen raw-point snapshot."""

    required = {
        "context_id",
        "sortie_id",
        "view_id",
        "start_offset_ms",
        "end_offset_ms",
    }
    missing = sorted(required - set(context))
    if missing:
        raise RepresentationContractError(f"Dingxin context is missing fields: {missing}")
    sortie_id = str(context["sortie_id"])
    view_id = str(context["view_id"])
    start_offset_ms = int(context["start_offset_ms"])
    end_offset_ms = int(context["end_offset_ms"])
    duration_s = (end_offset_ms - start_offset_ms) / 1000.0
    if abs(duration_s - 30.0) > 1e-6:
        raise RepresentationContractError(
            f"Dingxin context duration must be 30 seconds, got {duration_s}"
        )
    source_plan = _find_view_plan(plan.snapshot_manifest, sortie_id, view_id)
    start_utc = _parse_time(str(source_plan["start_utc"])) + timedelta(
        milliseconds=start_offset_ms
    )
    end_utc = start_utc + timedelta(seconds=duration_s)
    files = _snapshot_files(plan.snapshot_manifest, sortie_id, view_id)
    root = Path(plan.snapshot_root)
    physiology_path = root / str(files["physiology"]["relative_path"])
    vehicle_path = root / str(files["vehicle"]["relative_path"])
    physiology = _load_sparse_snapshot_stream(
        physiology_path,
        raw_to_index=plan.physiology_raw_to_index[sortie_id],
        feature_count=len(plan.schema.physiology_feature_names),
        start_utc=start_utc,
        end_utc=end_utc,
        cached_stream=(
            None
            if point_cache is None
            else point_cache.streams_by_path.get(str(physiology_path.resolve()))
        ),
    )
    vehicle = _load_sparse_snapshot_stream(
        vehicle_path,
        raw_to_index=plan.vehicle_raw_to_index[sortie_id],
        feature_count=len(plan.schema.vehicle_feature_names),
        start_utc=start_utc,
        end_utc=end_utc,
        cached_stream=(
            None
            if point_cache is None
            else point_cache.streams_by_path.get(str(vehicle_path.resolve()))
        ),
    )
    return ObservedDualStreamSample(
        sample_id=str(context["context_id"]),
        group_id=view_id,
        schema=plan.schema,
        physiology_values=physiology["values"],
        physiology_timestamps_s=physiology["timestamps"],
        physiology_feature_mask=physiology["feature_mask"],
        vehicle_values=vehicle["values"],
        vehicle_timestamps_s=vehicle["timestamps"],
        vehicle_feature_mask=vehicle["feature_mask"],
        source_sample_hash=stable_observed_sample_hash(
            str(files["physiology"]["sha256"]),
            str(files["vehicle"]["sha256"]),
            str(context["context_id"]),
            plan.schema.schema_sha256,
        ),
        context_duration_s=duration_s,
    )


def _slice_dense_observed_stream(
    *,
    timestamps: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> dict[str, np.ndarray]:
    keep = (timestamps >= start_s) & (timestamps < end_s)
    selected_values = values[keep]
    selected_timestamps = timestamps[keep] - start_s
    feature_mask = np.isfinite(selected_values)
    selected_values = np.nan_to_num(
        selected_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    row_keep = feature_mask.any(axis=1)
    return {
        "values": selected_values[row_keep],
        "timestamps": selected_timestamps[row_keep].astype(np.float64),
        "feature_mask": feature_mask[row_keep].astype(bool),
    }


def _load_sparse_snapshot_stream(
    path: Path,
    *,
    raw_to_index: Mapping[str, int],
    feature_count: int,
    start_utc: datetime,
    end_utc: datetime,
    cached_stream: CachedSparseSnapshotStream | None = None,
) -> dict[str, np.ndarray]:
    if cached_stream is not None:
        if cached_stream.feature_count != feature_count:
            raise RepresentationContractError(
                f"Dingxin cached feature count mismatch: {path}"
            )
        return _slice_cached_sparse_stream(
            cached_stream,
            start_utc=start_utc,
            end_utc=end_utc,
        )
    point_iterator = (
        iter_raw_point_snapshot(path)
    )
    selected_points = []
    for point in point_iterator:
        if point.timestamp < start_utc:
            continue
        if point.timestamp >= end_utc:
            break
        selected_points.append(point)
    if not selected_points:
        return {
            "values": np.empty((0, feature_count), dtype=np.float32),
            "timestamps": np.empty((0,), dtype=np.float64),
            "feature_mask": np.empty((0, feature_count), dtype=bool),
        }
    values = np.zeros((len(selected_points), feature_count), dtype=np.float32)
    feature_mask = np.zeros((len(selected_points), feature_count), dtype=bool)
    timestamps = np.empty((len(selected_points),), dtype=np.float64)
    output_index = 0
    for point in selected_points:
        for field, raw_value in point.values.items():
            index = raw_to_index.get(f"{point.measurement}.{field}")
            if index is None:
                continue
            value = _finite_float(raw_value)
            if value is None:
                continue
            values[output_index, index] = value
            feature_mask[output_index, index] = True
        if not feature_mask[output_index].any():
            continue
        timestamps[output_index] = (point.timestamp - start_utc).total_seconds()
        output_index += 1
    if output_index == 0:
        return {
            "values": np.empty((0, feature_count), dtype=np.float32),
            "timestamps": np.empty((0,), dtype=np.float64),
            "feature_mask": np.empty((0, feature_count), dtype=bool),
        }
    return {
        "values": values[:output_index],
        "timestamps": timestamps[:output_index],
        "feature_mask": feature_mask[:output_index],
    }


def _slice_cached_sparse_stream(
    stream: CachedSparseSnapshotStream,
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, np.ndarray]:
    start_epoch = start_utc.timestamp()
    end_epoch = end_utc.timestamp()
    left = int(np.searchsorted(stream.timestamps_epoch_s, start_epoch, side="left"))
    right = int(np.searchsorted(stream.timestamps_epoch_s, end_epoch, side="left"))
    point_count = right - left
    if point_count <= 0:
        return {
            "values": np.empty((0, stream.feature_count), dtype=np.float32),
            "timestamps": np.empty((0,), dtype=np.float64),
            "feature_mask": np.empty((0, stream.feature_count), dtype=bool),
        }
    values = np.zeros((point_count, stream.feature_count), dtype=np.float32)
    feature_mask = np.zeros((point_count, stream.feature_count), dtype=bool)
    for output_index, source_index in enumerate(range(left, right)):
        item_start = int(stream.indptr[source_index])
        item_end = int(stream.indptr[source_index + 1])
        indices = stream.feature_indices[item_start:item_end]
        values[output_index, indices] = stream.feature_values[item_start:item_end]
        feature_mask[output_index, indices] = True
    return {
        "values": values,
        "timestamps": stream.timestamps_epoch_s[left:right] - start_epoch,
        "feature_mask": feature_mask,
    }


def _plans_by_sortie(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for item in manifest.get("plans", []):
        plan = dict(item)
        sortie_id = str(plan["sortie_id"])
        existing = result.get(sortie_id)
        if existing is not None and tuple(existing["vehicle_measurements"]) != tuple(
            plan["vehicle_measurements"]
        ):
            raise RepresentationContractError(
                f"inconsistent vehicle measurement order for sortie {sortie_id}"
            )
        result[sortie_id] = plan
    if not result:
        raise RepresentationContractError("snapshot manifest has no view plans")
    return result


def _common_feature_names(
    mappings: Mapping[str, Mapping[str, tuple[str, str]]],
    stream_name: str,
) -> tuple[str, ...]:
    if not mappings:
        raise RepresentationContractError(f"no {stream_name} schema mappings were built")
    common = set.intersection(*(set(value) for value in mappings.values()))
    if not common:
        raise RepresentationContractError(
            f"no common {stream_name} features exist across fixed sorties"
        )
    return tuple(sorted(common))


def _find_view_plan(
    manifest: Mapping[str, object],
    sortie_id: str,
    view_id: str,
) -> Mapping[str, object]:
    matches = [
        dict(item)
        for item in manifest.get("plans", [])
        if str(item["sortie_id"]) == sortie_id and str(item["view_id"]) == view_id
    ]
    if len(matches) != 1:
        raise RepresentationContractError(
            f"expected one snapshot view plan for {view_id}, found {len(matches)}"
        )
    return matches[0]


def _snapshot_files(
    manifest: Mapping[str, object],
    sortie_id: str,
    view_id: str,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for item in manifest.get("files", []):
        record = dict(item)
        if str(record["sortie_id"]) != sortie_id:
            continue
        stream_kind = str(record["stream_kind"])
        if stream_kind == "vehicle" and record.get("view_id") is None:
            result["vehicle"] = record
        elif stream_kind == "physiology" and str(record.get("view_id")) == view_id:
            result["physiology"] = record
    if set(result) != {"physiology", "vehicle"}:
        raise RepresentationContractError(
            f"snapshot files are incomplete for view {view_id}: {sorted(result)}"
        )
    return result


def _finite_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _combined_file_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _read_optional_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
