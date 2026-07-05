"""Support and ablation report aggregation for task evaluation thesis evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.evidence.support_builders import (
    _build_alignment_support_summary,
    _build_causal_support_summary,
    _build_main_ablation_matrix,
    _build_support_matrix,
    _load_json,
    _write_support_overview_plot,
)
from chronaris.evidence.support_reporting import (
    render_task_eval_ablation_support_report,
    render_task_eval_alignment_support_report,
    render_task_eval_causal_support_report,
)

DEFAULT_ALIGNMENT_E_SUMMARY_PATH = (
    "docs/artifacts/assets/alignment-preview-stage-f-closure-2026-04-22-e-baseline/"
    "projection_diagnostics_summary.json"
)
DEFAULT_ALIGNMENT_F_SUMMARY_PATH = (
    "docs/artifacts/assets/alignment-preview-stage-f-closure-2026-04-22-stage-f-full/"
    "projection_diagnostics_summary.json"
)
DEFAULT_CAUSAL_G_SUMMARY_PATH = (
    "docs/artifacts/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/"
    "causal_fusion_summary.json"
)
DEFAULT_FEATURE_EXPORT_RUN_MANIFEST_PATH = (
    "docs/artifacts/runs/2026-04-27_feature-export-closure/run_manifest.json"
)
DEFAULT_CASE_STUDY_SUMMARY_PATH = (
    "docs/artifacts/assets/task_eval/20260429T000000Z-task-eval-phase2-case-study/"
    "case_study_summary.json"
)
DEFAULT_CASE_STUDY_ABLATION_CSV_PATH = (
    "docs/artifacts/assets/task_eval/20260429T000000Z-task-eval-phase2-case-study/"
    "ablation_summary.csv"
)
DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH = (
    "docs/artifacts/runs/2026-06-07_dingxin-opt-package/"
    "private_benchmark_summary.json"
)
DEFAULT_DEEP_COMPARISON_SUMMARY_PATH = (
    "docs/artifacts/runs/2026-05-01_full-loso-deep-comparison/comparison_summary.json"
)
DEFAULT_REPORT_ROOT = "docs/artifacts"
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"


@dataclass(frozen=True, slots=True)
class StageISupportConfig:
    run_id: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    alignment_e_summary_path: str = DEFAULT_ALIGNMENT_E_SUMMARY_PATH
    alignment_f_summary_path: str = DEFAULT_ALIGNMENT_F_SUMMARY_PATH
    causal_g_summary_path: str = DEFAULT_CAUSAL_G_SUMMARY_PATH
    feature_export_run_manifest_path: str = DEFAULT_FEATURE_EXPORT_RUN_MANIFEST_PATH
    case_study_summary_path: str = DEFAULT_CASE_STUDY_SUMMARY_PATH
    case_study_ablation_csv_path: str = DEFAULT_CASE_STUDY_ABLATION_CSV_PATH
    private_benchmark_summary_path: str = DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH
    deep_comparison_summary_path: str | None = DEFAULT_DEEP_COMPARISON_SUMMARY_PATH


@dataclass(frozen=True, slots=True)
class StageISupportRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    matrix_path: str
    main_matrix_path: str
    alignment_report_path: str
    causal_report_path: str
    ablation_report_path: str
    overview_plot_path: str | None
    summary: Mapping[str, object]


def run_task_eval_support(
    config: StageISupportConfig,
) -> StageISupportRunResult:
    artifact_root = Path(config.artifact_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    alignment_report_path = report_root / f"task-eval-alignment-support-{config.run_id}.md"
    causal_report_path = report_root / f"task-eval-causal-support-{config.run_id}.md"
    ablation_report_path = report_root / f"task-eval-ablation-support-{config.run_id}.md"
    summary_path = artifact_root / "support_summary.json"
    matrix_path = artifact_root / "support_matrix.csv"
    main_matrix_path = artifact_root / "ablation_matrix.csv"
    overview_plot_target = artifact_root / "support_overview.png"

    e_projection = _load_json(config.alignment_e_summary_path)
    f_projection = _load_json(config.alignment_f_summary_path)
    g_causal = _load_json(config.causal_g_summary_path)
    feature_export_run_manifest = _load_json(config.feature_export_run_manifest_path)
    case_study_summary = _load_json(config.case_study_summary_path)
    case_study_ablation = pd.read_csv(config.case_study_ablation_csv_path)
    private_summary = _load_json(config.private_benchmark_summary_path)
    deep_summary = (
        _load_json(config.deep_comparison_summary_path)
        if config.deep_comparison_summary_path
        else {}
    )

    alignment_support = _build_alignment_support_summary(
        e_projection=e_projection,
        f_projection=f_projection,
        feature_export_run_manifest=feature_export_run_manifest,
        case_study_summary=case_study_summary,
    )
    causal_support = _build_causal_support_summary(
        g_causal=g_causal,
        case_study_summary=case_study_summary,
        case_study_ablation=case_study_ablation,
        private_summary=private_summary,
        deep_summary=deep_summary,
    )
    support_matrix = _build_support_matrix(
        alignment_support=alignment_support,
        causal_support=causal_support,
    )
    support_matrix.to_csv(matrix_path, index=False)
    main_matrix = _build_main_ablation_matrix(
        alignment_support=alignment_support,
        causal_support=causal_support,
    )
    main_matrix.to_csv(main_matrix_path, index=False)
    overview_plot_path = _write_support_overview_plot(
        main_matrix,
        path=overview_plot_target,
    )

    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "artifact_root": str(artifact_root),
        "sources": {
            "alignment_e_summary_path": str(Path(config.alignment_e_summary_path)),
            "alignment_f_summary_path": str(Path(config.alignment_f_summary_path)),
            "causal_g_summary_path": str(Path(config.causal_g_summary_path)),
            "feature_export_run_manifest_path": str(Path(config.feature_export_run_manifest_path)),
            "case_study_summary_path": str(Path(config.case_study_summary_path)),
            "case_study_ablation_csv_path": str(Path(config.case_study_ablation_csv_path)),
            "private_benchmark_summary_path": str(Path(config.private_benchmark_summary_path)),
            "deep_comparison_summary_path": (
                str(Path(config.deep_comparison_summary_path))
                if config.deep_comparison_summary_path
                else None
            ),
        },
        "alignment_support": alignment_support,
        "causal_support": causal_support,
        "support_matrix_path": str(matrix_path),
        "main_ablation_matrix_path": str(main_matrix_path),
        "main_ablation_rows": json.loads(main_matrix.to_json(orient="records")),
        "overview_plot_path": overview_plot_path,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    alignment_report_path.write_text(
        render_task_eval_alignment_support_report(summary) + "\n",
        encoding="utf-8",
    )
    causal_report_path.write_text(
        render_task_eval_causal_support_report(summary) + "\n",
        encoding="utf-8",
    )
    ablation_report_path.write_text(
        render_task_eval_ablation_support_report(summary) + "\n",
        encoding="utf-8",
    )
    return StageISupportRunResult(
        run_id=config.run_id,
        artifact_root=str(artifact_root),
        summary_path=str(summary_path),
        matrix_path=str(matrix_path),
        main_matrix_path=str(main_matrix_path),
        alignment_report_path=str(alignment_report_path),
        causal_report_path=str(causal_report_path),
        ablation_report_path=str(ablation_report_path),
        overview_plot_path=overview_plot_path,
        summary=summary,
    )
