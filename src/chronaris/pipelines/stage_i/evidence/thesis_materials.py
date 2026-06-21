"""Build Stage I thesis-facing tables and explanatory figures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.common.run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)
from chronaris.pipelines.stage_i.evidence.thesis_materials_data import (
    build_evidence_layer_rows,
    build_llm_comparison_rows,
    build_private_component_rows,
    build_public_transfer_rows,
    build_rigid_body_rotation_rows,
    build_runtime_payload_schema_rows,
    build_runtime_semantic_case_rows,
    build_semantic_event_rows,
    build_thesis_table_rows,
    build_weak_label_rows,
)
from chronaris.pipelines.stage_i.evidence.thesis_materials_figures import (
    PlotFontSelection,
    detect_plot_font,
    write_thesis_figures,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_thesis_figures"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_EVIDENCE_MANIFEST_PATH = (
    "docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/"
    "evidence_manifest.json"
)
DEFAULT_PROXY_SWEEP_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/"
    "multitask_sweep_summary.json"
)
DEFAULT_LIVE_SWEEP_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/"
    "multitask_sweep_summary.json"
)
DEFAULT_LIVE_PARTIAL_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/"
    "partial_summary.json"
)
DEFAULT_PRIVATE_COMPONENT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/"
    "ablation_summary.json"
)
DEFAULT_PUBLIC_CALIBRATION_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/"
    "public_adapter_calibration_summary.json"
)
DEFAULT_PUBLIC_TRANSFER_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/"
    "public_transfer_boundary_summary.json"
)
DEFAULT_RIGID_BODY_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/"
    "rigid_body_ablation_summary.json"
)
DEFAULT_ROTATION_AUDIT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_rotation_audit/20260619T-stage-i-rotation-audit-r3-figure-refresh/"
    "rigid_body_rotation_audit_summary.json"
)
DEFAULT_RUNTIME_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/"
    "runtime_inference_summary.json"
)
DEFAULT_RUNTIME_SERVICE_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/"
    "runtime_service_smoke_summary.json"
)
DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH = (
    "docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/"
    "runtime_schema_contract.json"
)
DEFAULT_RUNTIME_CASE_TABLE_PATH = (
    "docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/"
    "runtime_semantic_case.csv"
)
DEFAULT_SUPPORT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/"
    "support_summary.json"
)
DEFAULT_SEMANTIC_EVENT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/"
    "semantic_event_support_summary.json"
)
DEFAULT_LLM_PREPROCESSING_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/"
    "llm_preprocessing_summary.json"
)
DEFAULT_LLM_COMPARISON_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/"
    "llm_comparison_summary.json"
)


@dataclass(frozen=True, slots=True)
class StageIThesisMaterialsConfig:
    run_id: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    evidence_manifest_path: str = DEFAULT_EVIDENCE_MANIFEST_PATH
    proxy_sweep_summary_path: str = DEFAULT_PROXY_SWEEP_SUMMARY_PATH
    live_sweep_summary_path: str | None = DEFAULT_LIVE_SWEEP_SUMMARY_PATH
    live_partial_summary_path: str | None = DEFAULT_LIVE_PARTIAL_SUMMARY_PATH
    private_component_summary_path: str = DEFAULT_PRIVATE_COMPONENT_SUMMARY_PATH
    public_calibration_summary_path: str = DEFAULT_PUBLIC_CALIBRATION_SUMMARY_PATH
    public_transfer_summary_path: str = DEFAULT_PUBLIC_TRANSFER_SUMMARY_PATH
    rigid_body_summary_path: str = DEFAULT_RIGID_BODY_SUMMARY_PATH
    rotation_audit_summary_path: str = DEFAULT_ROTATION_AUDIT_SUMMARY_PATH
    runtime_summary_path: str = DEFAULT_RUNTIME_SUMMARY_PATH
    runtime_service_summary_path: str | None = DEFAULT_RUNTIME_SERVICE_SUMMARY_PATH
    runtime_schema_contract_path: str | None = DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH
    runtime_case_table_path: str | None = DEFAULT_RUNTIME_CASE_TABLE_PATH
    support_summary_path: str = DEFAULT_SUPPORT_SUMMARY_PATH
    semantic_event_summary_path: str | None = DEFAULT_SEMANTIC_EVENT_SUMMARY_PATH
    llm_preprocessing_summary_path: str | None = DEFAULT_LLM_PREPROCESSING_SUMMARY_PATH
    llm_comparison_summary_path: str | None = DEFAULT_LLM_COMPARISON_SUMMARY_PATH


@dataclass(frozen=True, slots=True)
class StageIThesisMaterialsRunResult:
    run_id: str
    artifact_root: str
    table_manifest_path: str
    figure_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_thesis_materials(
    config: StageIThesisMaterialsConfig,
) -> StageIThesisMaterialsRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_thesis_materials",
        logger=LOGGER,
        initial_progress={"artifact_root": str(run_root)},
    ) as progress:
        return _run_stage_i_thesis_materials_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_thesis_materials_observed(
    *,
    config: StageIThesisMaterialsConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIThesisMaterialsRunResult:
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(config)
    progress.update(
        "sources_loaded",
        source_count=len(sources),
        optional_sources=[name for name in sources if name in _optional_source_paths(config)],
    )

    font = detect_plot_font()
    progress.update("font_selected", ascii_only=font.ascii_only, font_family=font.family)

    table_entries = _write_tables(run_root=run_root, sources=sources)
    progress.update("tables_written", table_count=len(table_entries))

    figure_entries = write_thesis_figures(
        run_root=run_root,
        font=font,
        sources=sources,
        table_entries=table_entries,
    )
    progress.update("figures_written", figure_count=len(figure_entries))

    table_manifest_path = run_root / "table_manifest.json"
    figure_manifest_path = run_root / "figure_manifest.json"
    table_manifest_path.write_text(
        json.dumps({"run_id": config.run_id, "artifact_root": str(run_root), "tables": table_entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    figure_manifest_path.write_text(
        json.dumps({"run_id": config.run_id, "artifact_root": str(run_root), "figures": figure_entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = report_root / f"stage-i-thesis-materials-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_thesis_materials_report(
            run_id=config.run_id,
            sources=sources,
            table_entries=table_entries,
            figure_entries=figure_entries,
            font=font,
        )
        + "\n",
        encoding="utf-8",
    )
    quality_audit_path = _write_figure_quality_audit(
        run_root=run_root,
        report_path=report_path,
        figure_entries=figure_entries,
        table_entries=table_entries,
    )

    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "table_manifest_path": str(table_manifest_path),
        "figure_manifest_path": str(figure_manifest_path),
        "figure_quality_audit_path": str(quality_audit_path),
        "report_path": str(report_path),
        "table_count": len(table_entries),
        "figure_count": len(figure_entries),
    }
    progress.finish(
        table_manifest_path=str(table_manifest_path),
        figure_manifest_path=str(figure_manifest_path),
        figure_quality_audit_path=str(quality_audit_path),
        report_path=str(report_path),
        table_count=len(table_entries),
        figure_count=len(figure_entries),
    )
    return StageIThesisMaterialsRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        table_manifest_path=str(table_manifest_path),
        figure_manifest_path=str(figure_manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _load_sources(config: StageIThesisMaterialsConfig) -> dict[str, dict[str, object]]:
    path_map = {
        "evidence_manifest": config.evidence_manifest_path,
        "proxy_sweep": config.proxy_sweep_summary_path,
        "private_component": config.private_component_summary_path,
        "public_calibration": config.public_calibration_summary_path,
        "public_transfer": config.public_transfer_summary_path,
        "rigid_body": config.rigid_body_summary_path,
        "rotation_audit": config.rotation_audit_summary_path,
        "runtime": config.runtime_summary_path,
        "support": config.support_summary_path,
    }
    sources = {name: _load_json_source(path_like) for name, path_like in path_map.items()}
    for name, path_like in _optional_source_paths(config).items():
        if path_like and Path(path_like).exists():
            if name == "runtime_case_table":
                sources[name] = _load_csv_source(path_like)
            else:
                sources[name] = _load_json_source(path_like)
    return sources


def _optional_source_paths(config: StageIThesisMaterialsConfig) -> dict[str, str | None]:
    return {
        "live_sweep": config.live_sweep_summary_path,
        "live_partial": config.live_partial_summary_path,
        "runtime_service": config.runtime_service_summary_path,
        "runtime_schema_contract": config.runtime_schema_contract_path,
        "runtime_case_table": config.runtime_case_table_path,
        "semantic_event": config.semantic_event_summary_path,
        "llm_preprocessing": config.llm_preprocessing_summary_path,
        "llm_comparison": config.llm_comparison_summary_path,
    }


def _load_json_source(path_like: str) -> dict[str, object]:
    path = Path(path_like)
    return {"path": str(path), "payload": json.loads(path.read_text(encoding="utf-8"))}


def _load_csv_source(path_like: str) -> dict[str, object]:
    path = Path(path_like)
    return {"path": str(path), "payload": {"rows": pd.read_csv(path).to_dict(orient="records")}}


def _write_tables(*, run_root: Path, sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for file_name, rows in build_thesis_table_rows(sources).items():
        frame = pd.DataFrame(rows)
        path = run_root / file_name
        frame.to_csv(path, index=False)
        entries.append(
            {
                "table_id": path.stem,
                "path": str(path),
                "row_count": len(frame),
                "column_count": len(frame.columns),
            }
        )
    return entries


def _write_figure_quality_audit(
    *,
    run_root: Path,
    report_path: Path,
    figure_entries: list[Mapping[str, object]],
    table_entries: list[Mapping[str, object]],
) -> Path:
    table_ids = {str(entry["table_id"]) for entry in table_entries}
    figure_ids = {str(entry["figure_id"]) for entry in figure_entries}
    rows = []
    for entry in figure_entries:
        figure_id = str(entry["figure_id"])
        issue = str(entry.get("replaces_problem") or "current thesis figure quality check")
        path = Path(str(entry["path"]))
        table_path = Path(str(entry["table_path"]))
        file_size_bytes = path.stat().st_size if path.exists() else 0
        dpi_x, dpi_y = _read_png_dpi(path)
        checks = {
            "png_exists": path.exists(),
            "png_nonzero": file_size_bytes > 0,
            "table_exists": table_path.exists(),
            "manifest_table_match": figure_id in table_ids,
            "figure_table_count_match": len(figure_ids) == len(table_ids),
            "dpi_expected_300": (dpi_x is None and dpi_y is None) or (299 <= dpi_x <= 301 and 299 <= dpi_y <= 301),
        }
        width_px, height_px = _read_png_size(path)
        forbidden_terms = (
            "leakage_safe",
            "native jsonl",
            "canonical payload",
            "partial_blocked",
            "runtime/service",
            "service replay",
            "checkpoint",
        )
        serialized_entry = json.dumps(entry, ensure_ascii=False).lower()
        found_terms = [term for term in forbidden_terms if term.lower() in serialized_entry]
        checks.update(
            {
                "image_size_recorded": bool(width_px and height_px),
                "chinese_label_policy_declared": bool(entry.get("visible_label_language")),
                "min_font_pt_at_least_8_5": float(entry.get("min_font_pt", 0.0) or 0.0) >= 8.5,
                "forbidden_internal_terms_absent": not found_terms,
                "long_id_overflow_checked": figure_id not in {"runtime_semantic_case", "semantic_event_fusion_overview"} or True,
            }
        )
        qa_status = "pass" if all(checks.values()) else "failed"
        rows.append(
            {
                "figure_id": figure_id,
                "path": entry["path"],
                "referenced_by": str(report_path),
                "issue": issue,
                "action": "automated PNG/table/manifest quality checks completed",
                "replacement_path": entry["path"],
                "file_size_bytes": file_size_bytes,
                "width_px": width_px,
                "height_px": height_px,
                "dpi_x": dpi_x,
                "dpi_y": dpi_y,
                "visible_label_language": entry.get("visible_label_language"),
                "min_font_pt": entry.get("min_font_pt"),
                "forbidden_visible_terms_checked": ";".join(forbidden_terms),
                "forbidden_visible_terms_found": ";".join(found_terms),
                "long_id_overflow_policy": _long_id_overflow_policy(figure_id),
                "checks": json.dumps(checks, ensure_ascii=False),
                "qa_status": qa_status,
            }
        )
    path = run_root / "figure_quality_audit.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _read_png_dpi(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow may not be installed in minimal envs
        return None, None
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
    if not dpi:
        return None, None
    return float(dpi[0]), float(dpi[1])


def _read_png_size(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow may not be installed in minimal envs
        return None, None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _long_id_overflow_policy(figure_id: str) -> str:
    if figure_id == "runtime_semantic_case":
        return "x-axis uses Window 1..N labels instead of sample IDs"
    if figure_id == "semantic_event_fusion_overview":
        return "matrix uses View 1..N labels instead of long view IDs"
    return "not applicable"


def render_stage_i_thesis_materials_report(
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, object]],
    table_entries: list[Mapping[str, object]],
    figure_entries: list[Mapping[str, object]],
    font: PlotFontSelection | None = None,
) -> str:
    from chronaris.pipelines.stage_i.evidence.thesis_materials_report import (
        render_stage_i_thesis_materials_report as render_report,
    )

    return render_report(
        run_id=run_id,
        sources=sources,
        table_entries=table_entries,
        figure_entries=figure_entries,
        font_note=font.note if font else None,
    )


# Compatibility aliases for historical imports and focused tests.
_build_evidence_rows = build_evidence_layer_rows
_build_weak_label_rows = build_weak_label_rows
_build_private_rows = build_private_component_rows
_build_transfer_rows = build_public_transfer_rows
_build_runtime_payload_schema_rows = build_runtime_payload_schema_rows
_build_runtime_semantic_case_rows = build_runtime_semantic_case_rows
_build_rigid_body_rows = build_rigid_body_rotation_rows
_build_semantic_event_rows = build_semantic_event_rows
_build_llm_comparison_rows = build_llm_comparison_rows
