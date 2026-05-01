"""Artifact-writing and report helpers for Stage H export."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from chronaris.pipelines.causal_fusion import StageGCausalFusionTensorExport
    from chronaris.pipelines.stage_h_export import (
        StageHExportConfig,
        StageHRunManifest,
        StageHSortieManifest,
        StageHViewExecutionResult,
        StageHViewManifest,
    )
    from chronaris.access.stage_h_profile import StageHSortieProfile


def render_stage_h_report(
    *,
    config: StageHExportConfig,
    run_manifest: StageHRunManifest,
    sortie_manifests: Sequence[StageHSortieManifest],
    view_manifests: Sequence[StageHViewManifest],
) -> str:
    """Render the human-readable Stage H report that links back to artifacts."""

    lines = [
        f"# Stage H Export v1 - {config.run_id}",
        "",
        f"- export version: `{config.export_version}`",
        f"- export profile: `{config.export_profile}`",
        f"- generated at UTC: `{run_manifest.generated_at_utc}`",
        f"- artifact root: `{run_manifest.output_root}`",
        f"- run manifest: `{run_manifest.output_root}/run_manifest.json`",
        f"- report path: `{run_manifest.report_path}`",
        f"- sortie count: `{len(sortie_manifests)}`",
        f"- generated view count: `{run_manifest.generated_view_count}`",
        f"- generated view ids: `{', '.join(run_manifest.generated_view_ids) if run_manifest.generated_view_ids else '(none)'}`",
        "",
        "## Frozen Config",
        "",
        f"- input normalization: `{config.preview_config.input_normalization_mode}`",
        f"- physics constraint family: `{config.preview_config.physics_constraint_family}`",
        f"- causal fusion enabled: `{config.causal_fusion_enabled}`",
        f"- causal fusion state source: `{config.causal_fusion_config.state_source}`",
        f"- intermediate partition: `{config.preview_config.intermediate_partition}`",
        f"- window duration ms: `{config.window_config.duration_ms}`",
        f"- window stride ms: `{config.window_config.stride_ms}`",
        f"- physiology point limit per measurement: `{config.resolved_physiology_point_limit_per_measurement}`",
        f"- vehicle point limit per measurement: `{config.resolved_vehicle_point_limit_per_measurement}`",
        f"- point limit note: `{config.point_limit_note}`",
        "",
        "## Sortie Summary",
        "",
        "| sortie | pilots | physiology availability | vehicle family | views |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sortie_manifest in sortie_manifests:
        lines.append(
            "| "
            f"`{sortie_manifest.sortie_id}` | "
            f"`{', '.join(str(pilot_id) for pilot_id in sortie_manifest.pilot_ids)}` | "
            f"`{', '.join(sortie_manifest.physiology_measurements)}` | "
            f"`{', '.join(sortie_manifest.vehicle_measurements)}` | "
            f"`{', '.join(sortie_manifest.exported_view_ids)}` |"
        )
    lines.extend(
        [
            "",
            "## View Packages",
            "",
            "| view | windows | model samples | diagnostics | stage G | feature bundle |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for view_manifest in view_manifests:
        failed_checks = summarize_failed_diagnostic_checks(view_manifest)
        lines.append(
            "| "
            f"`{view_manifest.view_id}` | "
            f"{view_manifest.window_count} | "
            f"{view_manifest.model_sample_count} | "
            f"`{view_manifest.projection_diagnostics_verdict}`"
            f"{(' (' + failed_checks + ')') if failed_checks else ''} | "
            f"`{'enabled' if view_manifest.stage_g_available else 'disabled'}` | "
            f"`{view_manifest.artifact_paths['feature_bundle_npz']}` |"
        )

    warn_details = [render_warn_detail(view_manifest) for view_manifest in view_manifests if view_manifest.projection_diagnostics_verdict != "PASS"]
    if warn_details:
        lines.extend(["", "## Diagnostics Warnings", ""])
        lines.extend(warn_details)

    if run_manifest.partial_data is not None:
        lines.extend(
            [
                "",
                "## Partial Data",
                "",
                f"- manifest path: `{run_manifest.partial_data.get('manifest_path')}`",
                f"- window manifest path: `{run_manifest.partial_data.get('window_manifest_path')}`",
                f"- feature bundle path: `{run_manifest.partial_data.get('feature_bundle_path')}`",
                f"- entry count: `{run_manifest.partial_data.get('entry_count')}`",
                f"- built entry count: `{run_manifest.partial_data.get('built_entry_count')}`",
                f"- skipped entry count: `{run_manifest.partial_data.get('skipped_entry_count')}`",
            ]
        )

    if run_manifest.failures:
        lines.extend(["", "## Failures", ""])
        for failure in run_manifest.failures:
            lines.append(
                f"- [{failure.get('scope')}] `{failure.get('sortie_id', '-')}` "
                f"`{failure.get('view_id', '-')}`: `{failure.get('message')}`"
            )

    if run_manifest.skipped:
        lines.extend(["", "## Skipped", ""])
        for item in run_manifest.skipped:
            lines.append(
                f"- [{item.get('scope')}] `{item.get('sortie_id', '-')}` "
                f"`{item.get('view_id', '-')}`: `{item.get('reason')}`"
            )

    lines.append("")
    return "\n".join(lines)


def summarize_failed_diagnostic_checks(view_manifest: StageHViewManifest) -> str:
    failed = load_failed_diagnostic_checks(view_manifest)
    if not failed:
        return ""
    return ", ".join(check["name"] for check in failed)


def render_warn_detail(view_manifest: StageHViewManifest) -> str:
    failed = load_failed_diagnostic_checks(view_manifest)
    if not failed:
        return f"- `{view_manifest.view_id}`: diagnostics verdict `{view_manifest.projection_diagnostics_verdict}`"
    details = "; ".join(
        f"{check['name']}={check['actual']:.6f} {check['operator']} {check['expected']:.6f}"
        for check in failed
    )
    return f"- `{view_manifest.view_id}`: {details}"


def load_failed_diagnostic_checks(view_manifest: StageHViewManifest) -> tuple[dict[str, object], ...]:
    path = view_manifest.artifact_paths.get("projection_diagnostics_summary_json")
    if not path:
        return ()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return ()
    checks = payload.get("threshold_evaluation", {}).get("checks", ())
    failed = []
    for check in checks:
        if check.get("passed"):
            continue
        failed.append(
            {
                "name": str(check.get("name", "unknown")),
                "actual": float(check.get("actual", 0.0)),
                "operator": str(check.get("operator", "?")),
                "expected": float(check.get("expected", 0.0)),
            }
        )
    return tuple(failed)


def build_run_config_summary(config: StageHExportConfig) -> dict[str, object]:
    return {
        "export_version": config.export_version,
        "export_profile": config.export_profile,
        "sortie_ids": list(config.sortie_ids),
        "input_normalization_mode": config.preview_config.input_normalization_mode,
        "physics_constraint_family": config.preview_config.physics_constraint_family,
        "physics_constraint_mode": config.preview_config.physics_constraint_mode,
        "causal_fusion_enabled": config.causal_fusion_enabled,
        "causal_fusion_state_source": config.causal_fusion_config.state_source,
        "intermediate_partition": config.preview_config.intermediate_partition,
        "window_duration_ms": config.window_config.duration_ms,
        "window_stride_ms": config.window_config.stride_ms,
        "preview_point_limit_per_measurement": config.preview_point_limit_per_measurement,
        "physiology_point_limit_per_measurement": config.resolved_physiology_point_limit_per_measurement,
        "vehicle_point_limit_per_measurement": config.resolved_vehicle_point_limit_per_measurement,
        "point_limit_note": config.point_limit_note,
        "export_scope_overrides_utc": {
            sortie_id: [bounds[0].isoformat(), bounds[1].isoformat()]
            for sortie_id, bounds in config.export_scope_overrides_utc.items()
        },
    }


def resolve_export_scope(
    profile: StageHSortieProfile,
    *,
    overrides: Mapping[str, tuple[object, object]],
):
    override = overrides.get(profile.sortie_id)
    if override is not None:
        return override
    return profile.clip_start_utc, profile.clip_stop_utc


def write_view_artifacts(
    *,
    view_dir: Path,
    execution: StageHViewExecutionResult,
) -> dict[str, str]:
    feature_bundle_path = view_dir / "feature_bundle.npz"
    intermediate_summary_path = view_dir / "intermediate_summary.json"
    projection_summary_path = view_dir / "projection_diagnostics_summary.json"
    causal_fusion_summary_path = view_dir / "causal_fusion_summary.json"
    window_manifest_path = view_dir / "window_manifest.jsonl"

    write_feature_bundle(
        feature_bundle_path,
        intermediate_export=execution.intermediate_export,
        stage_g_tensor_export=execution.stage_g_tensor_export,
    )
    write_json(intermediate_summary_path, build_intermediate_summary(execution.intermediate_export))
    write_json(
        projection_summary_path,
        {
            "summary": asdict(execution.diagnostics_summary),
            "threshold_evaluation": asdict(execution.threshold_evaluation),
        },
    )
    if execution.stage_g_result is not None:
        write_json(causal_fusion_summary_path, execution.stage_g_result.to_dict())
    write_window_manifest(
        window_manifest_path,
        dataset_result=execution.dataset_result,
        selected_sample_ids=set(execution.sample_ids),
    )
    return {
        "feature_bundle_npz": str(feature_bundle_path),
        "intermediate_summary_json": str(intermediate_summary_path),
        "projection_diagnostics_summary_json": str(projection_summary_path),
        "causal_fusion_summary_json": str(causal_fusion_summary_path) if execution.stage_g_result is not None else "",
        "window_manifest_jsonl": str(window_manifest_path),
    }


def write_feature_bundle(
    path: Path,
    *,
    intermediate_export,
    stage_g_tensor_export: StageGCausalFusionTensorExport | None,
) -> None:
    physiology_reference_projection = np.asarray([sample.physiology.reference_projected_states for sample in intermediate_export.samples], dtype=np.float32)
    vehicle_reference_projection = np.asarray([sample.vehicle.reference_projected_states for sample in intermediate_export.samples], dtype=np.float32)
    reference_offsets_s = np.asarray([sample.physiology.reference_offsets_s for sample in intermediate_export.samples], dtype=np.float32)
    fused_representation = np.zeros((0,), dtype=np.float32)
    attention_weights = np.zeros((0,), dtype=np.float32)
    vehicle_event_scores = np.zeros((0,), dtype=np.float32)
    if stage_g_tensor_export is not None:
        fused_representation = np.asarray(stage_g_tensor_export.fused_states, dtype=np.float32)
        attention_weights = np.asarray(stage_g_tensor_export.attention_weights, dtype=np.float32)
        vehicle_event_scores = np.asarray(stage_g_tensor_export.vehicle_event_scores, dtype=np.float32)
    np.savez(
        path,
        physiology_reference_projection=physiology_reference_projection,
        vehicle_reference_projection=vehicle_reference_projection,
        fused_representation=fused_representation,
        reference_offsets_s=reference_offsets_s,
        attention_weights=attention_weights,
        vehicle_event_scores=vehicle_event_scores,
    )


def build_intermediate_summary(intermediate_export) -> dict[str, object]:
    sample_ids = [sample.sample_id for sample in intermediate_export.samples]
    physiology_l2 = [sample.physiology.mean_reference_projection_l2 for sample in intermediate_export.samples]
    vehicle_l2 = [sample.vehicle.mean_reference_projection_l2 for sample in intermediate_export.samples]
    cosines = [sample.mean_reference_projection_cosine for sample in intermediate_export.samples]
    return {
        "partition": intermediate_export.partition,
        "sample_count": intermediate_export.sample_count,
        "reference_point_count": intermediate_export.reference_point_count,
        "sample_ids": sample_ids,
        "mean_physiology_reference_projection_l2": mean(physiology_l2),
        "mean_vehicle_reference_projection_l2": mean(vehicle_l2),
        "mean_reference_projection_cosine": mean(cosines),
    }


def write_window_manifest(
    path: Path,
    *,
    dataset_result,
    selected_sample_ids: set[str],
) -> None:
    rows = []
    for window in dataset_result.windows:
        rows.append(
            {
                "sample_id": window.sample_id,
                "sortie_id": window.sortie_id,
                "window_index": window.window_index,
                "start_offset_ms": window.start_offset_ms,
                "end_offset_ms": window.end_offset_ms,
                "physiology_point_count": len(window.physiology_points),
                "vehicle_point_count": len(window.vehicle_points),
                "selected_for_model": window.sample_id in selected_sample_ids,
            }
        )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
