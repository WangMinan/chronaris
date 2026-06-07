"""Minimal thesis-facing runtime/demo entrypoints."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from chronaris.features import load_stage_i_case_study_run

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIRuntimeDemoConfig:
    """Config for the minimal Stage I runtime/demo entry."""

    run_id: str
    source_path: str
    artifact_root: str = "docs/artifacts/assets/stage_i_runtime_demo"
    report_root: str = "docs/artifacts/stage_i"
    source_type: str = "auto"
    export_window_csv: bool = True


@dataclass(frozen=True, slots=True)
class StageIRuntimeDemoRunResult:
    """Machine outputs for one runtime/demo invocation."""

    run_id: str
    source_type: str
    artifact_root: str
    summary_path: str
    report_path: str
    window_csv_path: str | None
    summary: Mapping[str, object]


def run_stage_i_runtime_demo(
    config: StageIRuntimeDemoConfig,
) -> StageIRuntimeDemoRunResult:
    """Build a thesis-facing summary over frozen Stage H or private package assets."""

    resolved_source_type = _resolve_source_type(
        Path(config.source_path),
        source_type=config.source_type,
    )
    LOGGER.info(
        "stage_i_runtime_demo start run_id=%s source_type=%s source_path=%s",
        config.run_id,
        resolved_source_type,
        config.source_path,
    )
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "runtime_demo_summary.json"
    report_path = report_root / f"stage-i-runtime-demo-{config.run_id}.md"

    if resolved_source_type == "stage_h_run_manifest":
        summary, window_rows = _build_stage_h_demo_summary(Path(config.source_path))
    else:
        summary, window_rows = _build_private_package_demo_summary(Path(config.source_path))

    summary_payload = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "source_type": resolved_source_type,
        "source_path": str(Path(config.source_path)),
        "artifact_root": str(run_root),
        **summary,
    }
    window_csv_path: str | None = None
    if config.export_window_csv and window_rows:
        csv_path = run_root / "runtime_demo_windows.csv"
        pd.DataFrame(window_rows).to_csv(csv_path, index=False)
        window_csv_path = str(csv_path)
        summary_payload["window_csv_path"] = window_csv_path

    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path.write_text(
        render_stage_i_runtime_demo_report(summary_payload) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "stage_i_runtime_demo finished run_id=%s summary_path=%s report_path=%s window_csv=%s",
        config.run_id,
        summary_path,
        report_path,
        window_csv_path,
    )
    return StageIRuntimeDemoRunResult(
        run_id=config.run_id,
        source_type=resolved_source_type,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        window_csv_path=window_csv_path,
        summary=summary_payload,
    )


def render_stage_i_runtime_demo_report(summary: Mapping[str, object]) -> str:
    """Render the minimal runtime/demo report."""

    source_type = str(summary["source_type"])
    lines = [
        f"# Stage I Runtime Demo - {summary['run_id']}",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- source_type: `{source_type}`",
        f"- source_path: `{summary['source_path']}`",
    ]
    if summary.get("window_csv_path"):
        lines.append(f"- window_csv_path: `{summary['window_csv_path']}`")
    lines.append("")

    if source_type == "stage_h_run_manifest":
        stage_h = summary["stage_h"]
        lines.extend(
            [
                "## Stage H Runtime Overview",
                "",
                f"- run_manifest_path: `{stage_h['run_manifest_path']}`",
                f"- sortie_count: `{stage_h['sortie_count']}`",
                f"- view_count: `{stage_h['view_count']}`",
                f"- total_window_count: `{stage_h['total_window_count']}`",
                f"- case_window_count: `{stage_h['case_window_count']}`",
                f"- view_verdict_counts: `{stage_h['view_verdict_counts']}`",
                "- task_prediction_status: `not_available_for_frozen_stage_h_views`",
                "",
                "## View Summary",
                "",
                "| view | verdict | windows | case windows | mean projection cosine | mean attention entropy | mean top event | mean top contribution | top sample |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for view in stage_h["views"]:
            lines.append(
                f"| `{view['view_id']}` | `{view['projection_diagnostics_verdict']}` | "
                f"{view['window_count']} | {view['case_partition_sample_count']} | "
                f"{view['mean_projection_cosine']:.6f} | {view['mean_attention_entropy']:.6f} | "
                f"{view['mean_top_event_score']:.6f} | {view['mean_top_contribution_score']:.6f} | "
                f"`{view['top_sample_id']}` |"
            )
        if summary.get("window_csv_path"):
            lines.extend(
                [
                    "",
                    "## Case Windows",
                    "",
                    "- 详细窗口级导出已写入 `runtime_demo_windows.csv`，可直接用于论文配图或人工复核。",
                ]
            )
    else:
        package = summary["optimized_package"]
        lines.extend(
            [
                "## Optimized Package Overview",
                "",
                f"- package_path: `{package['package_path']}`",
                f"- target_variant_name: `{package['target_variant_name']}`",
                f"- source_run_id: `{package['source_run_id']}`",
                f"- record_sample_count: `{package['record_sample_count']}`",
                f"- record_view_count: `{package['record_view_count']}`",
                f"- selected_vehicle_field_count: `{package['selected_vehicle_field_count']}`",
                f"- selected_physiology_field_count: `{package['selected_physiology_field_count']}`",
                "",
                "## Dependency Contract",
                "",
            ]
        )
        for key, value in package["dependency_contracts"].items():
            lines.append(f"- {key}: `{value}`")
        lines.extend(
            [
                "",
                "## Task Export Summary",
                "",
                "| task | task_type | head_family | recommended_head | prediction_contract_available | metric snapshot |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for task in package["tasks"]:
            lines.append(
                f"| `{task['task_id']}` | `{task['task_type']}` | `{task['head_family']}` | "
                f"`{task['recommended_head']}` | `{task['prediction_contract_available']}` | "
                f"`{task['metric_snapshot']}` |"
            )
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
            ]
        )
        for key, value in package["diagnostics"].items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def _resolve_source_type(path: Path, *, source_type: str) -> str:
    if source_type != "auto":
        if source_type not in {"stage_h_run_manifest", "optimized_candidate_package"}:
            raise ValueError(f"unsupported source_type: {source_type}")
        return source_type
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "package_version" in payload and "dependency_contracts" in payload:
        return "optimized_candidate_package"
    if "sortie_manifest_paths" in payload:
        return "stage_h_run_manifest"
    raise ValueError(
        "failed to infer runtime source type; expected Stage H run manifest or optimized package."
    )


def _build_stage_h_demo_summary(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    run_input = load_stage_i_case_study_run(path)
    verdict_counts = Counter(
        view.projection_diagnostics_verdict
        for view in run_input.views
    )
    total_window_count = sum(view.window_count for view in run_input.views)
    case_window_count = sum(view.case_partition_sample_count for view in run_input.views)
    view_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for view in run_input.views:
        sample_rows = {
            str(sample.get("sample_id")): sample
            for sample in view.causal_summary.get("samples", [])
            if isinstance(sample, Mapping) and sample.get("sample_id")
        }
        top_sample_id = ""
        top_sample = None
        if sample_rows:
            top_sample_id, top_sample = max(
                sample_rows.items(),
                key=lambda item: float(item[1].get("top_contribution_score", 0.0)),
            )
        projection_summary = view.projection_summary.get("summary", {})
        view_rows.append(
            {
                "view_id": view.view_id,
                "sortie_id": view.sortie_id,
                "pilot_id": view.pilot_id,
                "projection_diagnostics_verdict": view.projection_diagnostics_verdict,
                "window_count": view.window_count,
                "case_partition_sample_count": view.case_partition_sample_count,
                "mean_projection_cosine": float(
                    projection_summary.get("mean_projection_cosine", 0.0)
                ),
                "mean_projection_l2_gap": float(
                    projection_summary.get("mean_projection_l2_gap", 0.0)
                ),
                "mean_attention_entropy": _safe_float(
                    view.causal_summary.get("mean_attention_entropy"),
                    fallback=_mean_attention_entropy(view.stage_h_view.attention_weights),
                ),
                "mean_top_event_score": _safe_float(
                    view.causal_summary.get("mean_top_event_score"),
                    fallback=float(np.max(view.stage_h_view.vehicle_event_scores, axis=-1).mean()),
                ),
                "mean_top_contribution_score": _safe_float(
                    view.causal_summary.get("mean_top_contribution_score"),
                ),
                "top_sample_id": top_sample_id,
                "top_sample_top_event_offset_s": _safe_nested_float(
                    top_sample,
                    "top_event_offset_s",
                ),
                "top_sample_top_event_score": _safe_nested_float(
                    top_sample,
                    "top_event_score",
                ),
                "top_sample_top_contribution_offset_s": _safe_nested_float(
                    top_sample,
                    "top_contribution_offset_s",
                ),
                "top_sample_top_contribution_score": _safe_nested_float(
                    top_sample,
                    "top_contribution_score",
                ),
            }
        )
        for row in view.case_window_rows:
            sample_payload = sample_rows.get(row.sample_id, {})
            window_rows.append(
                {
                    "view_id": view.view_id,
                    "sortie_id": row.sortie_id,
                    "pilot_id": view.pilot_id,
                    "projection_diagnostics_verdict": view.projection_diagnostics_verdict,
                    "sample_id": row.sample_id,
                    "window_index": row.window_index,
                    "start_offset_ms": row.start_offset_ms,
                    "end_offset_ms": row.end_offset_ms,
                    "physiology_point_count": row.physiology_point_count,
                    "vehicle_point_count": row.vehicle_point_count,
                    "selected_for_model": row.selected_for_model,
                    "top_event_offset_s": _safe_nested_float(
                        sample_payload,
                        "top_event_offset_s",
                    ),
                    "top_event_score": _safe_nested_float(
                        sample_payload,
                        "top_event_score",
                    ),
                    "top_contribution_offset_s": _safe_nested_float(
                        sample_payload,
                        "top_contribution_offset_s",
                    ),
                    "top_contribution_score": _safe_nested_float(
                        sample_payload,
                        "top_contribution_score",
                    ),
                }
            )
    return (
        {
            "stage_h": {
                "run_manifest_path": str(path),
                "sortie_count": len(run_input.stage_h_run.run_manifest.get("sortie_manifest_paths", {})),
                "view_count": len(run_input.views),
                "total_window_count": total_window_count,
                "case_window_count": case_window_count,
                "view_verdict_counts": dict(verdict_counts),
                "views": view_rows,
            }
        },
        window_rows,
    )


def _build_private_package_demo_summary(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    package = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[dict[str, object]] = []
    for task_id, payload in package.get("tasks", {}).items():
        task_type = str(payload.get("task_type", "unknown"))
        head_family = str(payload.get("head_family", "n/a"))
        recommended_head = str(payload.get("recommended_head", head_family))
        prediction_contract_available = bool(
            payload.get("thresholds")
            or payload.get("available_heads")
            or payload.get("feature_columns")
        )
        metric_payload = (
            payload.get("cross_validated_best_metrics")
            or payload.get("cross_validated_metrics")
            or {}
        )
        tasks.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "head_family": head_family,
                "recommended_head": recommended_head,
                "prediction_contract_available": prediction_contract_available,
                "metric_snapshot": _format_metric_snapshot(metric_payload),
            }
        )
    return (
        {
            "optimized_package": {
                "package_path": str(path),
                "source_run_id": str(package.get("run_id", "")),
                "target_variant_name": str(package.get("target_variant_name", "")),
                "dependency_contracts": dict(package.get("dependency_contracts", {})),
                "record_sample_count": int(
                    package.get("records_summary", {}).get("sample_count", 0)
                ),
                "record_view_count": int(
                    package.get("records_summary", {}).get("view_count", 0)
                ),
                "selected_vehicle_field_count": len(
                    package.get("selected_vehicle_fields", [])
                ),
                "selected_physiology_field_count": len(
                    package.get("selected_physiology_fields", [])
                ),
                "diagnostics": dict(package.get("diagnostics", {})),
                "tasks": tasks,
            }
        },
        [],
    )


def _safe_float(value: object, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    return float(value)


def _safe_nested_float(mapping: object, key: str) -> float:
    if not isinstance(mapping, Mapping):
        return 0.0
    value = mapping.get(key)
    return 0.0 if value is None else float(value)


def _mean_attention_entropy(attention_weights: np.ndarray) -> float:
    weights = np.asarray(attention_weights, dtype=np.float32)
    if weights.size == 0:
        return 0.0
    clipped = np.clip(weights, 1e-8, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1)
    return float(entropy.mean())


def _format_metric_snapshot(metrics: Mapping[str, object]) -> str:
    if not metrics:
        return "no_metrics"
    for ordered_keys in (
        ("macro_f1", "balanced_accuracy"),
        ("rmse", "mae"),
        ("top1_accuracy", "mrr"),
    ):
        if all(key in metrics for key in ordered_keys):
            return ", ".join(
                f"{key}={_format_number(metrics[key])}"
                for key in ordered_keys
            )
    return ", ".join(
        f"{key}={_format_number(value)}"
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    )


def _format_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return "nan"
    return str(value)
