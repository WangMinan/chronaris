"""Dataset loading helpers for E3 fusion stream structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.evaluation.fusion_stream_structure.contracts import (
    FusionStreamRecord,
    detect_forbidden_sidecar_columns,
    normalize_method_name,
    validate_input_frame,
)


@dataclass(frozen=True, slots=True)
class FusionStreamDataset:
    records: Mapping[tuple[str, str, str], FusionStreamRecord]
    manifest: Mapping[str, object]
    unavailable_methods: tuple[Mapping[str, object], ...] = field(default_factory=tuple)


def load_fusion_stream_long_table(path: str | Path) -> FusionStreamDataset:
    """Read a CSV/JSONL/Parquet E3 long table and return grouped stream records."""

    source_path = Path(path)
    frame = _read_table(source_path)
    validation = validate_input_frame(frame)
    records = records_from_validated_frame(validation.frame, validation.feature_columns)
    manifest = {
        "source_type": "fusion_stream_long_table",
        "source_path": str(source_path),
        "row_count": int(len(validation.frame)),
        "stream_count": int(len(records)),
        "method_feature_dimensions": dict(validation.method_feature_dimensions),
    }
    return FusionStreamDataset(records=records, manifest=manifest)


def records_from_validated_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    feature_name_maps: Mapping[tuple[str, str, str], Mapping[str, str]] | None = None,
) -> dict[tuple[str, str, str], FusionStreamRecord]:
    records: dict[tuple[str, str, str], FusionStreamRecord] = {}
    posthoc_columns = tuple(
        column
        for column in (
            "maneuver_proxy_label",
            "maneuver_event_interval",
            "physio_fluctuation_interval",
            "weak_event_boundary",
            "pilot_id",
            "sample_partition",
        )
        if column in frame.columns
    )
    for (method_name, sortie_id, view_id), group in frame.groupby(
        ["method_name", "sortie_id", "view_id"],
        sort=True,
    ):
        ordered = group.sort_values(["time", "window_id"], kind="mergesort").reset_index(drop=True)
        key = (str(method_name), str(sortie_id), str(view_id))
        records[key] = FusionStreamRecord(
            method_name=str(method_name),
            sortie_id=str(sortie_id),
            view_id=str(view_id),
            frame=ordered,
            feature_columns=tuple(str(column) for column in feature_columns),
            window_ids=tuple(ordered["window_id"].to_list()),
            times=tuple(float(value) for value in ordered["time"].to_list()),
            feature_name_map=dict((feature_name_maps or {}).get(key, {column: column for column in feature_columns})),
            posthoc_columns=posthoc_columns,
        )
    return records


def build_fusion_streams_from_dingxin_artifacts(
    *,
    e_run_manifest_path: str | Path,
    f_run_manifest_path: str | Path,
    methods: Sequence[str] = ("chronaris", "naive_time_sync", "mult", "contiformer"),
    enable_optimized_chronaris: bool = True,
    chronaris_variant_name: str = "chronaris_opt",
    include_proxy_columns: bool = True,
    max_groups: int | None = None,
) -> FusionStreamDataset:
    """Build E3 streams from existing Dingxin feature-export artifacts.

    This function reuses the established feature-frame builders. It does not
    train deep comparison models and does not rerun feature export.
    """

    from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
        TASK_MANEUVER,
        TASK_RESPONSE,
        build_variant_feature_frames,
        derive_private_proxy_task_entries,
        load_aligned_private_records,
    )
    feature_model_sources, feature_model_sources_import_error = _load_feature_model_sources()

    requested_methods = tuple(normalize_method_name(method) for method in methods)
    records = load_aligned_private_records(
        e_run_manifest_path=str(e_run_manifest_path),
        f_run_manifest_path=str(f_run_manifest_path),
    )
    selected_groups = _select_record_groups(records, max_groups=max_groups)
    if selected_groups is not None:
        records = selected_groups

    task_payload = derive_private_proxy_task_entries(records) if include_proxy_columns else None
    proxy_columns = _build_posthoc_columns(records, task_payload, TASK_MANEUVER, TASK_RESPONSE)
    variant_frames, diagnostics = build_variant_feature_frames(
        records,
        enable_optimized_chronaris=enable_optimized_chronaris,
        target_variant_name=chronaris_variant_name,
    )

    rows: list[dict[str, object]] = []
    feature_name_maps: dict[tuple[str, str, str], Mapping[str, str]] = {}
    method_sources: dict[str, object] = {}
    unavailable: list[Mapping[str, object]] = []
    removed_sidecars: dict[str, Mapping[str, object]] = {}
    for method_name in requested_methods:
        variant_name = _resolve_variant_for_method(
            method_name,
            variant_frames=variant_frames,
            chronaris_variant_name=chronaris_variant_name,
            feature_model_sources=feature_model_sources,
        )
        if variant_name is None:
            unavailable.append(method_unavailable_entry(
                method_name,
                "no_reusable_fusion_feature_frame",
            ))
            continue
        variant_frame = variant_frames.get(variant_name, pd.DataFrame())
        if variant_frame.empty or "feature_values" not in variant_frame.columns:
            unavailable.append(method_unavailable_entry(
                method_name,
                "empty_or_missing_feature_values",
                variant_name=variant_name,
            ))
            continue
        converted = _variant_frame_to_e3_rows(
            variant_frame,
            method_name=method_name,
            variant_name=variant_name,
            proxy_columns=proxy_columns,
        )
        if not converted["rows"]:
            unavailable.append(method_unavailable_entry(
                method_name,
                "no_safe_fusion_features_after_sidecar_filter",
                variant_name=variant_name,
            ))
            removed_sidecars[method_name] = converted["removed_sidecars"]
            continue
        rows.extend(converted["rows"])
        feature_name_maps.update(converted["feature_name_maps"])
        method_sources[method_name] = {
            "variant_name": variant_name,
            "feature_count": converted["feature_count"],
        }
        removed_sidecars[method_name] = converted["removed_sidecars"]

    if not rows:
        return FusionStreamDataset(
            records={},
            manifest={
                "source_type": "dingxin_feature_export",
                "status": "no_available_methods",
                "e_run_manifest_path": str(e_run_manifest_path),
                "f_run_manifest_path": str(f_run_manifest_path),
                "requested_methods": list(requested_methods),
                "unavailable_methods": list(unavailable),
            },
            unavailable_methods=tuple(unavailable),
        )

    long_frame = pd.DataFrame(rows)
    long_frame = _derive_weak_event_boundaries(long_frame)
    validation = validate_input_frame(long_frame)
    records_by_key = records_from_validated_frame(
        validation.frame,
        validation.feature_columns,
        feature_name_maps=feature_name_maps,
    )
    manifest = {
        "source_type": "dingxin_feature_export",
        "e_run_manifest_path": str(e_run_manifest_path),
        "f_run_manifest_path": str(f_run_manifest_path),
        "requested_methods": list(requested_methods),
        "available_methods": sorted({key[0] for key in records_by_key}),
        "unavailable_methods": list(unavailable),
        "method_sources": method_sources,
        "method_feature_dimensions": dict(validation.method_feature_dimensions),
        "removed_sidecar_features": removed_sidecars,
        "diagnostics": diagnostics,
        "feature_model_sources_import_error": feature_model_sources_import_error,
        "record_group_limit": max_groups,
        "row_count": int(len(validation.frame)),
        "stream_count": int(len(records_by_key)),
    }
    return FusionStreamDataset(
        records=records_by_key,
        manifest=manifest,
        unavailable_methods=tuple(unavailable),
    )


def method_unavailable_entry(
    method_name: str,
    reason: str,
    *,
    variant_name: str | None = None,
) -> dict[str, object]:
    entry = {
        "method_name": normalize_method_name(method_name),
        "status": "method_unavailable",
        "reason": reason,
    }
    if variant_name is not None:
        entry["variant_name"] = variant_name
    return entry


def _load_feature_model_sources() -> tuple[Mapping[str, tuple[str | None, str | None]], str | None]:
    try:
        from chronaris.evaluation.dingxin.pipelines.thirdparty_comparison import FEATURE_MODEL_SOURCES
    except Exception as exc:
        return {
            "chronaris_full": ("chronaris_opt", "full_safe"),
            "naive_time_sync": ("naive_sync", "dual_projection"),
            "classical_baseline": ("f_full", "dual_projection"),
        }, repr(exc)
    return FEATURE_MODEL_SOURCES, None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported E3 table format: {path}")


def _select_record_groups(records: pd.DataFrame, *, max_groups: int | None) -> pd.DataFrame | None:
    if max_groups is None or max_groups <= 0:
        return None
    group_keys = list(records.groupby(["sortie_id", "view_id"], sort=True).groups.keys())
    selected = set(group_keys[:max_groups])
    mask = [
        (row.sortie_id, row.view_id) in selected
        for row in records.itertuples(index=False)
    ]
    return records.loc[mask].reset_index(drop=True)


def _build_posthoc_columns(
    records: pd.DataFrame,
    task_payload: Mapping[str, object] | None,
    task_maneuver: str,
    task_response: str,
) -> Mapping[str, Mapping[str, object]]:
    if not task_payload:
        return {}
    maneuver_by_sample = {
        entry.sample_id: entry.label_value
        for entry in task_payload["by_task"].get(task_maneuver, ())
    }
    response_by_sample = {
        entry.sample_id: entry.label_value
        for entry in task_payload["by_task"].get(task_response, ())
        if entry.label_value is not None
    }
    response_values = np.asarray([float(value) for value in response_by_sample.values()], dtype=float)
    response_threshold = float(np.nanquantile(response_values, 0.75)) if response_values.size else None
    payload = {}
    record_by_sample = records.set_index("sample_id", drop=False)
    for sample_id, record in record_by_sample.iterrows():
        response_value = response_by_sample.get(sample_id)
        payload[str(sample_id)] = {
            "maneuver_proxy_label": maneuver_by_sample.get(sample_id),
            "physio_fluctuation_interval": (
                bool(response_threshold is not None and response_value is not None and float(response_value) >= response_threshold)
            ),
            "pilot_id": int(record["pilot_id"]),
            "sample_partition": record.get("sample_partition"),
        }
    return payload


def _resolve_variant_for_method(
    method_name: str,
    *,
    variant_frames: Mapping[str, pd.DataFrame],
    chronaris_variant_name: str,
    feature_model_sources: Mapping[str, tuple[str, str]],
) -> str | None:
    if method_name == "chronaris":
        for candidate in (
            chronaris_variant_name,
            feature_model_sources.get("chronaris_full", (None, None))[0],
            "chronaris_full",
        ):
            if candidate in variant_frames:
                return str(candidate)
        return None
    if method_name == "naive_time_sync":
        for candidate in (
            feature_model_sources.get("naive_time_sync", (None, None))[0],
            "naive_sync",
        ):
            if candidate in variant_frames:
                return str(candidate)
        return None
    if method_name in variant_frames:
        return method_name
    return None


def _variant_frame_to_e3_rows(
    variant_frame: pd.DataFrame,
    *,
    method_name: str,
    variant_name: str,
    proxy_columns: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    feature_names = _safe_feature_names(variant_frame)
    forbidden = detect_forbidden_sidecar_columns(feature_names)
    safe_feature_names = tuple(name for name in feature_names if name not in forbidden)
    feature_name_map = {
        f"fusion_feature_{index}": name
        for index, name in enumerate(safe_feature_names, start=1)
    }
    rows = []
    feature_name_maps: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for source_row in variant_frame.itertuples(index=False):
        values = getattr(source_row, "feature_values")
        if not isinstance(values, Mapping):
            continue
        row = {
            "method_name": method_name,
            "sortie_id": str(source_row.sortie_id),
            "view_id": str(source_row.view_id),
            "window_id": str(source_row.sample_id),
            "time": int(source_row.window_index),
            "source_variant": variant_name,
        }
        row.update(proxy_columns.get(str(source_row.sample_id), {}))
        for feature_column, source_name in feature_name_map.items():
            value = values.get(source_name)
            row[feature_column] = float(value) if value is not None else np.nan
        rows.append(row)
        key = (method_name, str(source_row.sortie_id), str(source_row.view_id))
        feature_name_maps.setdefault(key, feature_name_map)
    return {
        "rows": rows,
        "feature_count": len(feature_name_map),
        "feature_name_maps": feature_name_maps,
        "removed_sidecars": {
            "count": len(forbidden),
            "feature_names": list(forbidden),
        },
    }


def _safe_feature_names(frame: pd.DataFrame) -> tuple[str, ...]:
    names: set[str] = set()
    for values in frame["feature_values"]:
        if isinstance(values, Mapping):
            for key, value in values.items():
                if value is None:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue
                names.add(str(key))
    return tuple(sorted(names))


def _derive_weak_event_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    if "maneuver_proxy_label" not in frame.columns:
        frame["weak_event_boundary"] = False
        return frame
    rows = []
    for _key, group in frame.groupby(["method_name", "sortie_id", "view_id"], sort=False):
        ordered = group.sort_values(["time", "window_id"], kind="mergesort").copy()
        labels = ordered["maneuver_proxy_label"].astype(str).to_list()
        boundaries = [False]
        for previous, current in zip(labels[:-1], labels[1:]):
            boundaries.append(previous != current and previous != "None" and current != "None")
        ordered["weak_event_boundary"] = boundaries
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)
