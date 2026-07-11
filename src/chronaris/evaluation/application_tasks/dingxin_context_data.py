"""Lazy fixed-snapshot contexts and fold-task bindings for Dingxin evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.representation import (
    DingxinObservationSchemaPlan,
    DingxinSnapshotPointCache,
    build_dingxin_observation_schema_plan,
    build_dingxin_snapshot_point_cache,
    collate_observation_samples,
    load_dingxin_observed_context,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class DingxinLazyContextIndex:
    plan: DingxinObservationSchemaPlan
    point_cache: DingxinSnapshotPointCache
    contexts: pd.DataFrame
    context_by_id: Mapping[str, Mapping[str, object]]

    def load_sample(self, context_id: str):
        try:
            context = self.context_by_id[str(context_id)]
        except KeyError as exc:
            raise KeyError(f"unknown Dingxin application context: {context_id}") from exc
        if not bool(context["input_fully_observed"]):
            raise ValueError(f"Dingxin context input is incomplete: {context_id}")
        return load_dingxin_observed_context(
            self.plan,
            context,
            point_cache=self.point_cache,
        )

    def load_batch(self, context_ids: Sequence[str]):
        return collate_observation_samples(
            tuple(self.load_sample(context_id) for context_id in context_ids)
        )


def build_dingxin_lazy_context_index(
    *,
    snapshot_root: str | Path,
    field_role_manifest_path: str | Path,
    context_manifest_path: str | Path,
) -> DingxinLazyContextIndex:
    plan = build_dingxin_observation_schema_plan(
        snapshot_root=snapshot_root,
        field_role_manifest_path=field_role_manifest_path,
    )
    contexts = pd.read_json(context_manifest_path, lines=True)
    view_plans = {
        item["view_id"]: item for item in plan.snapshot_manifest["plans"]
    }
    rows = []
    for row in contexts.to_dict("records"):
        view_id = str(row["view_id"])
        source_plan = view_plans[view_id]
        start = datetime.fromisoformat(str(source_plan["start_utc"]))
        stop = datetime.fromisoformat(str(source_plan["stop_utc"]))
        stop_offset_ms = int(round((stop - start).total_seconds() * 1000))
        start_offset_ms = int(row["start_offset_ms"])
        end_offset_ms = int(row["end_offset_ms"])
        duration_ms = end_offset_ms - start_offset_ms
        fully_observed = duration_ms == 30_000 and end_offset_ms <= stop_offset_ms
        if duration_ms != 30_000:
            unavailable_reason = "input_context_duration_not_30_seconds"
        elif end_offset_ms > stop_offset_ms:
            unavailable_reason = "input_interval_not_fully_observed"
        else:
            unavailable_reason = None
        rows.append(
            {
                **row,
                "context_duration_ms": duration_ms,
                "snapshot_stop_offset_ms": stop_offset_ms,
                "input_fully_observed": fully_observed,
                "input_unavailable_reason": unavailable_reason,
                "schema_sha256": plan.schema.schema_sha256,
            }
        )
    catalog = pd.DataFrame(rows).sort_values(
        ["sortie_id", "view_id", "end_window_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return DingxinLazyContextIndex(
        plan=plan,
        point_cache=build_dingxin_snapshot_point_cache(plan),
        contexts=catalog,
        context_by_id={row["context_id"]: row for row in rows},
    )


def audit_lazy_dingxin_contexts(index: DingxinLazyContextIndex):
    rows = []
    for context in index.contexts.itertuples(index=False):
        if not context.input_fully_observed:
            rows.append(
                {
                    "context_id": context.context_id,
                    "status": "unavailable",
                    "reason": context.input_unavailable_reason,
                    "physiology_point_count": 0,
                    "vehicle_point_count": 0,
                    "physiology_observed_feature_count": 0,
                    "vehicle_observed_feature_count": 0,
                    "maximum_relative_timestamp_s": None,
                    "source_sample_hash": None,
                }
            )
            continue
        sample = index.load_sample(context.context_id)
        maximum = max(
            float(sample.physiology_timestamps_s.max()),
            float(sample.vehicle_timestamps_s.max()),
        )
        rows.append(
            {
                "context_id": context.context_id,
                "status": "completed",
                "reason": None,
                "physiology_point_count": int(sample.physiology_values.shape[0]),
                "vehicle_point_count": int(sample.vehicle_values.shape[0]),
                "physiology_observed_feature_count": int(
                    sample.physiology_feature_mask.any(axis=0).sum()
                ),
                "vehicle_observed_feature_count": int(
                    sample.vehicle_feature_mask.any(axis=0).sum()
                ),
                "maximum_relative_timestamp_s": maximum,
                "source_sample_hash": sample.source_sample_hash,
            }
        )
    return pd.DataFrame(rows)


def build_fold_task_context_bindings(
    *,
    index: DingxinLazyContextIndex,
    target_archive_manifest_path: str | Path,
):
    manifest = pd.read_csv(target_archive_manifest_path)
    context_lookup = index.contexts.set_index("context_id")
    rows = []
    archive_verification = []
    for item in manifest.itertuples(index=False):
        archive_path = Path(item.archive_path)
        actual = sha256_file(archive_path)
        if actual != item.archive_sha256:
            raise ValueError(f"Dingxin target archive hash mismatch: {archive_path}")
        threshold_path = Path(item.threshold_path)
        threshold_actual = sha256_file(threshold_path)
        if threshold_actual != item.threshold_sha256:
            raise ValueError(f"Dingxin threshold hash mismatch: {threshold_path}")
        archive_verification.append(
            {
                "fold_id": item.fold_id,
                "task_slug": item.task_slug,
                "archive_path": str(archive_path),
                "archive_sha256": actual,
                "threshold_path": str(threshold_path),
                "threshold_sha256": threshold_actual,
            }
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            context_ids = tuple(str(value) for value in archive["context_ids"])
            split_roles = tuple(str(value) for value in archive["split_roles"])
            statuses = tuple(str(value) for value in archive["statuses"])
            fit_hashes = tuple(str(value) for value in archive["fit_sample_hashes"])
            input_starts = archive["input_start_offset_ms"].astype(np.int64)
            input_ends = archive["input_end_exclusive_ms"].astype(np.int64)
            target_starts = archive["target_start_offset_ms"].astype(np.int64)
            target_ends = archive["target_end_exclusive_ms"].astype(np.int64)
            class_targets = archive["class_target"].astype(np.int64)
            continuous_targets = archive["continuous_target"].astype(np.float32)
            binary_targets = archive["binary_target"].astype(np.int64)
        if len(set(context_ids)) != len(context_ids):
            raise ValueError(f"duplicate context IDs in target archive: {archive_path}")
        missing = sorted(set(context_ids) - set(context_lookup.index))
        if missing:
            raise ValueError(f"target contexts absent from raw catalog: {missing[:5]}")
        for position, context_id in enumerate(context_ids):
            context = context_lookup.loc[context_id]
            input_available = bool(context.input_fully_observed)
            target_available = statuses[position] == "completed"
            if not input_available:
                binding_status = str(context.input_unavailable_reason)
            elif not target_available:
                binding_status = statuses[position]
            else:
                binding_status = "available"
            rows.append(
                {
                    "fold_id": item.fold_id,
                    "task_slug": item.task_slug,
                    "split_role": split_roles[position],
                    "context_id": context_id,
                    "sortie_id": context.sortie_id,
                    "view_id": context.view_id,
                    "input_start_offset_ms": int(input_starts[position]),
                    "input_end_exclusive_ms": int(input_ends[position]),
                    "target_start_offset_ms": int(target_starts[position]),
                    "target_end_exclusive_ms": int(target_ends[position]),
                    "target_status": statuses[position],
                    "class_target": int(class_targets[position]),
                    "continuous_target": float(continuous_targets[position]),
                    "binary_target": int(binary_targets[position]),
                    "input_fully_observed": input_available,
                    "binding_status": binding_status,
                    "fit_sample_hash": fit_hashes[position],
                    "schema_sha256": index.plan.schema.schema_sha256,
                    "archive_sha256": actual,
                }
            )
    return pd.DataFrame(rows), archive_verification


def schema_manifest(index: DingxinLazyContextIndex):
    schema = index.plan.schema
    cached_streams = tuple(index.point_cache.streams_by_path.values())
    return {
        "format": "chronaris.dingxin_lazy_context_schema.v1",
        "schema_id": schema.schema_id,
        "schema_sha256": schema.schema_sha256,
        "physiology_feature_count": len(schema.physiology_feature_names),
        "vehicle_feature_count": len(schema.vehicle_feature_names),
        "excluded_feature_count": len(schema.excluded_feature_names),
        "physiology_feature_names": list(schema.physiology_feature_names),
        "vehicle_feature_names": list(schema.vehicle_feature_names),
        "excluded_feature_names": list(schema.excluded_feature_names),
        "snapshot_root": index.plan.snapshot_root,
        "materialization_policy": "lazy_small_batch_only",
        "cached_snapshot_file_count": len(index.point_cache.streams_by_path),
        "cached_observed_point_count": sum(
            len(stream.timestamps_epoch_s) for stream in cached_streams
        ),
        "cached_observed_value_count": sum(
            len(stream.feature_values) for stream in cached_streams
        ),
        "cached_sparse_array_bytes": sum(
            stream.timestamps_epoch_s.nbytes
            + stream.indptr.nbytes
            + stream.feature_indices.nbytes
            + stream.feature_values.nbytes
            for stream in cached_streams
        ),
        "precomputed_dense_context_bundle": False,
    }


def dingxin_vehicle_field_labels(
    index: DingxinLazyContextIndex,
    *,
    field_role_manifest_path: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Map canonical vehicle inputs to metadata labels used by physics rules."""
    roles = pd.read_csv(field_role_manifest_path)
    sortie_id = sorted(index.plan.vehicle_raw_to_index)[0]
    frame = roles[
        (roles["sortie_id"].astype(str) == sortie_id)
        & (roles["stream_kind"].astype(str) == "vehicle")
    ]
    raw_labels = {
        f"{row.measurement}.{row.source_field}": str(row.display_label)
        for row in frame.itertuples(index=False)
    }
    raw_by_index = {
        position: raw_name
        for raw_name, position in index.plan.vehicle_raw_to_index[sortie_id].items()
    }
    return tuple(
        (
            canonical,
            raw_labels.get(raw_by_index.get(position, ""), canonical),
        )
        for position, canonical in enumerate(
            index.plan.schema.vehicle_feature_names
        )
    )
