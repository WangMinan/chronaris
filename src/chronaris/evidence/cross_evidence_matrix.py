"""P32 task evaluation private/public cross-evidence matrix builder."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
DEFAULT_PUBLIC_FUSION_REFRESH_ROOT = "docs/artifacts/runs/2026-07-01_public-fusion-calibration"


@dataclass(frozen=True, slots=True)
class StageICrossEvidenceMatrixConfig:
    run_id: str
    private_thirdparty_root: str
    public_ablation_root: str
    private_ablation_root: str = "docs/artifacts/runs/2026-06-19_dingxin-leakage-safe-ablation"
    public_model_comparison_root: str = "docs/artifacts/runs/2026-07-01_public-model-comparison"
    public_fusion_refresh_root: str = DEFAULT_PUBLIC_FUSION_REFRESH_ROOT
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT


@dataclass(frozen=True, slots=True)
class StageICrossEvidenceMatrixResult:
    run_id: str
    artifact_root: str
    matrix_csv_path: str
    matrix_json_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def build_task_eval_cross_evidence_matrix(
    config: StageICrossEvidenceMatrixConfig,
) -> StageICrossEvidenceMatrixResult:
    run_root = _resolve_path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(_private_thirdparty_rows(_resolve_path(config.private_thirdparty_root)))
    rows.extend(_private_ablation_rows(_resolve_path(config.private_ablation_root)))
    rows.extend(_public_model_rows(_resolve_path(config.public_model_comparison_root)))
    rows.extend(_public_ablation_rows(_resolve_path(config.public_ablation_root)))
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        raise ValueError("cross evidence matrix has no rows.")
    matrix_path = run_root / "cross_evidence_matrix.csv"
    matrix_json_path = run_root / "cross_evidence_matrix.json"
    summary_md_path = run_root / "cross_evidence_summary.md"
    matrix.to_csv(matrix_path, index=False)
    matrix_json_path.write_text(
        json.dumps({"run_id": config.run_id, "rows": rows}, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_render_summary_md(config.run_id, matrix) + "\n", encoding="utf-8")
    figure_paths = _render_figures(run_root, matrix)
    manifest_path = run_root / "evidence_manifest.json"
    report_path = _resolve_path(config.report_root) / f"task-eval-cross-evidence-matrix-{config.run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": config.run_id,
        "status": "completed",
        "generated_at_utc": _utc_now(),
        "artifact_root": str(run_root),
        "cross_evidence_matrix_csv": str(matrix_path),
        "cross_evidence_matrix_json": str(matrix_json_path),
        "cross_evidence_summary_md": str(summary_md_path),
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "figure_paths": figure_paths,
        "quadrant_counts": matrix["evidence_quadrant"].value_counts().to_dict(),
        "source_roots": {
            "private_thirdparty_root": str(_resolve_path(config.private_thirdparty_root)),
            "private_ablation_root": str(_resolve_path(config.private_ablation_root)),
            "public_model_comparison_root": str(_resolve_path(config.public_model_comparison_root)),
            "public_fusion_refresh_root": str(_resolve_path(config.public_fusion_refresh_root)),
            "public_ablation_root": str(_resolve_path(config.public_ablation_root)),
        },
    }
    manifest_path.write_text(
        json.dumps({**summary, "summary_path": str(summary_md_path)}, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary, matrix) + "\n", encoding="utf-8")
    return StageICrossEvidenceMatrixResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        matrix_csv_path=str(matrix_path),
        matrix_json_path=str(matrix_json_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _private_thirdparty_rows(root: Path) -> list[dict[str, object]]:
    long_path = root / "model_comparison_long.csv"
    improvement_path = root / "improvement_summary.csv"
    rows = []
    if not long_path.exists():
        return rows
    long_frame = pd.read_csv(long_path)
    improvement = pd.read_csv(improvement_path) if improvement_path.exists() else pd.DataFrame()
    delta_lookup = {
        (row["task_name"], row["split_strategy"], row["metric"], row["baseline_model"]): row
        for row in improvement.to_dict(orient="records")
    }
    for row in long_frame.to_dict(orient="records"):
        baseline = None
        delta_abs = np.nan
        delta_rel = np.nan
        if row["model_name"] != "chronaris_full":
            delta_row = delta_lookup.get((row["task_name"], row["split_strategy"], row["metric"], row["model_name"]))
            if delta_row:
                baseline = "chronaris_full"
                delta_abs = delta_row.get("delta_abs")
                delta_rel = delta_row.get("delta_rel_pct")
        rows.append(
            _matrix_row(
                evidence_quadrant="private_thirdparty_comparison",
                dataset_role="private_real_dual_stream",
                dataset_id="private_feature_export",
                task_group=row["task_name"],
                model_or_component=row["model_name"],
                metric=row["metric"],
                value=row["value_mean"],
                baseline=baseline,
                delta_abs=delta_abs,
                delta_rel_pct=delta_rel,
                protocol=f"leakage_safe_v1/{row['split_strategy']}",
                artifact_path=str(long_path),
                figure_path=str(root / "fig_private_thirdparty_delta_heatmap.png"),
                midterm_use="private model comparison",
                wording_boundary="T1/T2/T3 proxy tasks on private real dual-stream feature export data",
            )
        )
    return rows


def _private_ablation_rows(root: Path) -> list[dict[str, object]]:
    path = root / "ablation_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("rows", []):
        rows.append(
            _matrix_row(
                evidence_quadrant="private_component_ablation",
                dataset_role="private_real_dual_stream",
                dataset_id="private_feature_export",
                task_group=row.get("task_name"),
                model_or_component=row.get("variant_name"),
                metric=row.get("primary_metric_name"),
                value=row.get("primary_metric_value"),
                baseline="full_model",
                delta_abs=row.get("delta_vs_full"),
                delta_rel_pct=row.get("relative_delta_percent"),
                protocol=row.get("protocol", "leakage_safe_v1"),
                artifact_path=str(path),
                figure_path=str(root / "model_backbone_ablation.png"),
                midterm_use="private component ablation",
                wording_boundary="leakage-safe private proxy component diagnosis",
            )
        )
    return rows


def _public_model_rows(root: Path) -> list[dict[str, object]]:
    long_path = root / "model_comparison_long.csv"
    improvement_path = root / "improvement_summary.csv"
    rows = []
    if not long_path.exists():
        return rows
    long_frame = pd.read_csv(long_path)
    improvement = pd.read_csv(improvement_path) if improvement_path.exists() else pd.DataFrame()
    delta_lookup = {
        (row["dataset_id"], row["task_group"], row["metric_name"], row["baseline_model"]): row
        for row in improvement.to_dict(orient="records")
    }
    for row in long_frame.to_dict(orient="records"):
        baseline = None
        delta_abs = np.nan
        delta_rel = np.nan
        delta_row = delta_lookup.get((row["dataset_id"], row["task_group"], row["metric_name"], row["model_name"]))
        if delta_row:
            baseline = delta_row.get("chronaris_model")
            delta_abs = delta_row.get("delta_abs")
            delta_rel = delta_row.get("delta_rel_pct")
        rows.append(
            _matrix_row(
                evidence_quadrant="public_model_comparison",
                dataset_role="public_context_proxy",
                dataset_id=row["dataset_id"],
                task_group=row["task_group"],
                model_or_component=row["model_name"],
                metric=row["metric_name"],
                value=row["value"],
                baseline=baseline,
                delta_abs=delta_abs,
                delta_rel_pct=delta_rel,
                protocol=row.get("protocol"),
                artifact_path=str(long_path),
                figure_path=str(root / "fig_public_model_delta_heatmap.png"),
                midterm_use="public model comparison",
                wording_boundary="NASA/UAB public adapter context-proxy evidence",
            )
        )
    return rows


def _public_ablation_rows(root: Path) -> list[dict[str, object]]:
    path = root / "ablation_summary.csv"
    rows = []
    if not path.exists():
        return rows
    frame = pd.read_csv(path)
    for row in frame.to_dict(orient="records"):
        rows.append(
            _matrix_row(
                evidence_quadrant="public_component_ablation",
                dataset_role="public_context_proxy",
                dataset_id=row.get("dataset_id"),
                task_group=row.get("task_group"),
                model_or_component=row.get("variant_id"),
                metric=row.get("metric"),
                value=row.get("value_mean"),
                baseline="full",
                delta_abs=row.get("delta_abs_mean"),
                delta_rel_pct=row.get("delta_rel_pct_mean"),
                protocol="p31_public_ablation_full_loso_or_bounded",
                artifact_path=str(path),
                figure_path=str(root / "fig_public_ablation_delta_heatmap.png"),
                midterm_use="public component ablation",
                wording_boundary="public adapter context-proxy mechanism check",
            )
        )
    return rows


def _matrix_row(**fields: object) -> dict[str, object]:
    row = {
        "evidence_quadrant": "",
        "dataset_role": "",
        "dataset_id": "",
        "task_group": "",
        "model_or_component": "",
        "metric": "",
        "value": np.nan,
        "baseline": "",
        "delta_abs": np.nan,
        "delta_rel_pct": np.nan,
        "protocol": "",
        "artifact_path": "",
        "figure_path": "",
        "midterm_use": "",
        "wording_boundary": "",
    }
    row.update(fields)
    return row


def _render_figures(root: Path, matrix: pd.DataFrame) -> dict[str, str]:
    paths = {
        "fig_cross_evidence_matrix": str(root / "fig_cross_evidence_matrix.png"),
        "fig_cross_evidence_metric_overview": str(root / "fig_cross_evidence_metric_overview.png"),
        "fig_private_public_evidence_roles": str(root / "fig_private_public_evidence_roles.png"),
        "fig_private_public_result_summary": str(root / "fig_private_public_result_summary.png"),
        "fig_method_claim_support_map": str(root / "fig_method_claim_support_map.png"),
    }
    _plot_quadrants(matrix, paths["fig_cross_evidence_matrix"])
    _plot_metric_overview(matrix, paths["fig_cross_evidence_metric_overview"])
    _plot_roles(matrix, paths["fig_private_public_evidence_roles"])
    _plot_result_summary(matrix, paths["fig_private_public_result_summary"])
    _plot_claim_support_map(matrix, paths["fig_method_claim_support_map"])
    return paths


def _plot_quadrants(matrix: pd.DataFrame, path: str) -> None:
    counts = matrix.groupby(["dataset_role", "evidence_quadrant"]).size().unstack(fill_value=0)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    image = axis.imshow(counts.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    axis.set_xticks(np.arange(len(counts.columns)))
    axis.set_xticklabels(counts.columns, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(counts.index)))
    axis.set_yticklabels(counts.index)
    axis.set_title("task evaluation cross-evidence matrix row coverage")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            axis.text(j, i, str(int(counts.iloc[i, j])), ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_metric_overview(matrix: pd.DataFrame, path: str) -> None:
    frame = matrix.copy()
    frame["abs_delta"] = pd.to_numeric(frame["delta_abs"], errors="coerce").abs()
    top = frame.dropna(subset=["abs_delta"]).sort_values("abs_delta", ascending=False).head(20)
    fig, axis = plt.subplots(figsize=(11, 6))
    if top.empty:
        axis.text(0.5, 0.5, "no delta rows", ha="center", va="center")
        axis.axis("off")
    else:
        labels = top["evidence_quadrant"] + "\n" + top["model_or_component"].astype(str) + " / " + top["metric"].astype(str)
        axis.barh(np.arange(len(top)), top["delta_abs"].astype(float), color="#4f81bd")
        axis.set_yticks(np.arange(len(top)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_title("Largest cross-evidence deltas")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_roles(matrix: pd.DataFrame, path: str) -> None:
    counts = matrix["dataset_role"].value_counts()
    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90)
    axis.set_title("Private real-dual-stream vs public context-proxy rows")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_result_summary(matrix: pd.DataFrame, path: str) -> None:
    frame = matrix.copy()
    frame["has_delta"] = pd.to_numeric(frame["delta_abs"], errors="coerce").notna()
    grouped = (
        frame.groupby("evidence_quadrant", sort=False)
        .agg(row_count=("metric", "count"), delta_row_count=("has_delta", "sum"))
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(grouped))
    width = 0.38
    axis.bar(x - width / 2, grouped["row_count"].astype(float), width, label="metric rows", color="#2f6f9f")
    axis.bar(x + width / 2, grouped["delta_row_count"].astype(float), width, label="delta rows", color="#7a9c42")
    axis.set_xticks(x)
    axis.set_xticklabels(grouped["evidence_quadrant"], rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("row count")
    axis.set_title("Private/public result coverage by evidence quadrant")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_claim_support_map(matrix: pd.DataFrame, path: str) -> None:
    table = pd.crosstab(matrix["midterm_use"], matrix["evidence_quadrant"])
    fig, axis = plt.subplots(figsize=(10, max(4.5, len(table.index) * 0.55)))
    if table.empty:
        axis.text(0.5, 0.5, "no support rows", ha="center", va="center")
        axis.axis("off")
    else:
        image = axis.imshow(table.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
        axis.set_xticks(np.arange(len(table.columns)))
        axis.set_xticklabels(table.columns, rotation=25, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(table.index)))
        axis.set_yticklabels(table.index, fontsize=8)
        axis.set_title("Method claim support map")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                axis.text(j, i, str(int(table.iloc[i, j])), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _render_summary_md(run_id: str, matrix: pd.DataFrame) -> str:
    counts = matrix["evidence_quadrant"].value_counts().to_dict()
    return "\n".join(
        [
            f"# task evaluation Cross-evidence Summary - {run_id}",
            "",
            "task evaluation now contains a cross-evidence matrix covering private real dual-stream comparison, private leakage-safe component ablation, public model comparison and public fusion ablation.",
            "",
            "Quadrant row counts:",
            "",
            *[f"- `{key}`: {value}" for key, value in counts.items()],
        ]
    )


def _render_report(summary: Mapping[str, object], matrix: pd.DataFrame) -> str:
    lines = [
        f"# task evaluation Cross-evidence Matrix - {summary['run_id']}",
        "",
        "## Executive Summary",
        "",
        "task evaluation now contains a cross-evidence matrix covering private real dual-stream comparison, private leakage-safe component ablation, public model comparison and public fusion ablation. "
        "The private Dingxin/feature export branch evaluates Chronaris against MulT and ContiFormer under the same leakage-safe task protocol; the public NASA/UAB branch decomposes the P28 public fusion refresh result into lag, event-bias, context-stream and fusion-head contributions.",
        "",
        "## Matrix",
        "",
        f"- csv: `{summary['cross_evidence_matrix_csv']}`",
        f"- json: `{summary['cross_evidence_matrix_json']}`",
        f"- summary: `{summary['cross_evidence_summary_md']}`",
        "",
        "## Quadrant Counts",
        "",
        _markdown_table(pd.DataFrame([summary["quadrant_counts"]]).T.reset_index().rename(columns={"index": "quadrant", 0: "row_count"})),
        "",
        "## Figure index",
        "",
    ]
    for name, path in (summary.get("figure_paths") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Preview rows",
            "",
            _markdown_table(matrix.head(30)),
            "",
            "## Reproducibility",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- evidence_manifest: `{summary['evidence_manifest_path']}`",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if np.isfinite(value) else "")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
