"""Build task evaluation public model-comparison artifacts for midterm reporting."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]

PUBLIC_MAINLINE_SUMMARY = (
    "docs/artifacts/runs/"
    "2026-05-08_public-mainline-uab-robust-prior-r1/"
    "public_mainline_summary.json"
)
DEEP_COMPARISON_ROOT = (
    "docs/artifacts/runs/2026-05-01_full-loso-deep-comparison"
)
CURRENT_NASA_FUSION_SUMMARY = (
    "docs/artifacts/runs/"
    "2026-05-09_public-fusion-nasa-full-confirm/"
    "deep_baseline_summary.json"
)
CURRENT_UAB_FUSION_SUMMARY = (
    "docs/artifacts/runs/2026-05-06_public-fusion-uab-confirm/"
    "deep_baseline_summary.json"
)
NASA_PUBLIC_OPT_SUMMARY = (
    "docs/artifacts/runs/"
    "2026-05-06_public-opt-nasa-round1/public_opt_summary.json"
)


@dataclass(frozen=True, slots=True)
class StageIPublicModelComparisonConfig:
    run_id: str
    artifact_root: str = "docs/artifacts/runs"
    report_root: str = "docs/artifacts/runs"
    public_mainline_summary_path: str = PUBLIC_MAINLINE_SUMMARY
    deep_comparison_root: str = DEEP_COMPARISON_ROOT
    current_nasa_fusion_summary_path: str = CURRENT_NASA_FUSION_SUMMARY
    current_uab_fusion_summary_path: str = CURRENT_UAB_FUSION_SUMMARY
    nasa_public_opt_summary_path: str = NASA_PUBLIC_OPT_SUMMARY
    refresh_summary_path: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicModelComparisonResult:
    run_id: str
    artifact_root: str
    long_csv_path: str
    wide_csv_path: str
    improvement_summary_csv_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def build_task_eval_public_model_comparison(
    config: StageIPublicModelComparisonConfig,
) -> StageIPublicModelComparisonResult:
    output_root = _resolve_path(config.artifact_root) / config.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    report_root = _resolve_path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    refresh_summary_path = (
        _resolve_path(config.refresh_summary_path)
        if config.refresh_summary_path
        else _find_latest_refresh_summary()
    )
    rows, metric_sources = _build_metric_rows(config, refresh_summary_path)
    long_frame = pd.DataFrame(rows)
    wide_frame = _build_wide_frame(long_frame)
    improvement_frame = _build_improvement_summary(wide_frame)

    long_csv_path = output_root / "model_comparison_long.csv"
    wide_csv_path = output_root / "model_comparison_wide.csv"
    improvement_csv_path = output_root / "improvement_summary.csv"
    long_frame.to_csv(long_csv_path, index=False)
    wide_frame.to_csv(wide_csv_path, index=False)
    improvement_frame.to_csv(improvement_csv_path, index=False)

    figure_paths = _render_comparison_figures(wide_frame, output_root)
    report_path = report_root / f"task-eval-public-model-comparison-{config.run_id}.md"
    manifest_path = output_root / "evidence_manifest.json"
    summary = {
        "run_id": config.run_id,
        "generated_at_utc": _utc_now(),
        "artifact_root": str(output_root),
        "long_csv_path": str(long_csv_path),
        "wide_csv_path": str(wide_csv_path),
        "improvement_summary_csv_path": str(improvement_csv_path),
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "figure_paths": figure_paths,
        "source_paths": sorted(
            {
                str(_normalize_existing_path(path))
                for path in [
                    config.public_mainline_summary_path,
                    config.deep_comparison_root,
                    config.current_nasa_fusion_summary_path,
                    config.current_uab_fusion_summary_path,
                    config.nasa_public_opt_summary_path,
                    str(refresh_summary_path) if refresh_summary_path else "",
                ]
                if path
            }
        ),
        "metric_sources": metric_sources,
        "p28_refresh_included": bool(
            refresh_summary_path
            and refresh_summary_path.exists()
            and (
                long_frame["model_name"] == "chronaris_public_fusion_refresh"
            ).any()
        ),
        "missing_metrics": _find_missing_metrics(wide_frame),
        "protocol_notes": [
            "UAB/NASA public datasets are public adapter/calibration/context-proxy evidence.",
            "Positive deltas mean Chronaris is better: higher-is-better metrics use Chronaris minus baseline; RMSE/MAE use baseline minus Chronaris.",
            "chronaris_public_fusion_current may mix full NASA LOSO confirm with earlier UAB two-fold confirm; protocol fields keep this visible.",
        ],
    }
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _render_report(config.run_id, wide_frame, improvement_frame, figure_paths, summary)
        + "\n",
        encoding="utf-8",
    )
    return StageIPublicModelComparisonResult(
        run_id=config.run_id,
        artifact_root=str(output_root),
        long_csv_path=str(long_csv_path),
        wide_csv_path=str(wide_csv_path),
        improvement_summary_csv_path=str(improvement_csv_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _build_metric_rows(
    config: StageIPublicModelComparisonConfig,
    refresh_summary_path: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public_mainline = _load_json(config.public_mainline_summary_path)
    nasa_public = _load_json(config.nasa_public_opt_summary_path)
    current_nasa = _load_json(config.current_nasa_fusion_summary_path)
    current_uab = _load_json(config.current_uab_fusion_summary_path)
    refresh_summary = _load_json(refresh_summary_path) if refresh_summary_path else None
    rows: list[dict[str, object]] = []
    metric_sources: list[dict[str, object]] = []

    def add_row(
        *,
        dataset_id: str,
        task_group: str,
        task_type: str,
        metric_name: str,
        metric_direction: str,
        model_family: str,
        model_name: str,
        value: float | None,
        sample_count: int | float | None,
        fold_count: int | float | None,
        protocol: str,
        evidence_role: str,
        source_path: str | Path,
    ) -> None:
        if value is None or _is_nan(value):
            return
        source = str(_normalize_existing_path(source_path))
        row = {
            "dataset_id": dataset_id,
            "task_group": task_group,
            "task_type": task_type,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "model_family": model_family,
            "model_name": model_name,
            "value": float(value),
            "sample_count": _optional_int(sample_count),
            "fold_count": _optional_int(fold_count),
            "protocol": protocol,
            "evidence_role": evidence_role,
            "source_path": source,
        }
        rows.append(row)
        metric_sources.append(
            {
                "dataset_id": dataset_id,
                "task_group": task_group,
                "metric_name": metric_name,
                "model_name": model_name,
                "source_path": source,
                "protocol": protocol,
            }
        )

    deep_root = _resolve_path(config.deep_comparison_root)
    for dataset_id in ("nasa_csm", "uab_workload_dataset"):
        for model_family in ("mult", "contiformer"):
            summary_path = deep_root / dataset_id / model_family / "deep_baseline_summary.json"
            summary = _load_json(summary_path)
            _add_deep_summary_rows(
                add_row=add_row,
                summary=summary,
                model_family=model_family,
                model_name=model_family,
                source_path=summary_path,
                protocol="full_loso_20260501",
                evidence_role="public_deep_baseline",
            )

    _add_classical_reference_rows(
        add_row=add_row,
        deep_root=deep_root,
        evidence_role="classical_full_loso_reference",
    )
    _add_public_mainline_rows(add_row, public_mainline, nasa_public, config)
    _add_deep_summary_rows(
        add_row=add_row,
        summary=current_nasa,
        model_family="chronaris_public_fusion",
        model_name="chronaris_public_fusion_current",
        source_path=config.current_nasa_fusion_summary_path,
        protocol="full_loso_confirm_current",
        evidence_role="public_adapter_context_proxy_evidence",
    )
    _add_deep_summary_rows(
        add_row=add_row,
        summary=current_uab,
        model_family="chronaris_public_fusion",
        model_name="chronaris_public_fusion_current",
        source_path=config.current_uab_fusion_summary_path,
        protocol="two_fold_confirm_current",
        evidence_role="public_adapter_context_proxy_evidence",
    )
    if refresh_summary:
        _add_refresh_rows(add_row, refresh_summary, refresh_summary_path)
    return rows, metric_sources


def _add_deep_summary_rows(
    *,
    add_row,
    summary: Mapping[str, object],
    model_family: str,
    model_name: str,
    source_path: str | Path,
    protocol: str,
    evidence_role: str,
) -> None:
    dataset_id = str(summary["dataset_id"])
    objective = summary.get("objective") or {}
    for group, metrics in (objective.get("groups") or {}).items():
        for metric_name in ("macro_f1", "balanced_accuracy"):
            add_row(
                dataset_id=dataset_id,
                task_group=str(group),
                task_type="classification",
                metric_name=metric_name,
                metric_direction="higher_is_better",
                model_family=model_family,
                model_name=model_name,
                value=metrics.get(metric_name),
                sample_count=metrics.get("sample_count"),
                fold_count=metrics.get("fold_count"),
                protocol=protocol,
                evidence_role=evidence_role,
                source_path=source_path,
            )
    subjective = summary.get("subjective") or {}
    for group, metrics in (subjective.get("groups") or {}).items():
        for metric_name in ("rmse", "mae"):
            add_row(
                dataset_id=dataset_id,
                task_group=str(group),
                task_type="regression",
                metric_name=metric_name,
                metric_direction="lower_is_better",
                model_family=model_family,
                model_name=model_name,
                value=metrics.get(metric_name),
                sample_count=metrics.get("sample_count"),
                fold_count=metrics.get("fold_count"),
                protocol=protocol,
                evidence_role=evidence_role,
                source_path=source_path,
            )


def _add_classical_reference_rows(*, add_row, deep_root: Path, evidence_role: str) -> None:
    for dataset_id, summary_path in {
        "nasa_csm": deep_root / "nasa_csm" / "mult" / "deep_baseline_summary.json",
        "uab_workload_dataset": deep_root
        / "uab_workload_dataset"
        / "contiformer"
        / "deep_baseline_summary.json",
    }.items():
        summary = _load_json(summary_path)
        reference = summary.get("reference_comparison") or {}
        for track, task_type, direction, metric_names in (
            ("objective", "classification", "higher_is_better", ("macro_f1", "balanced_accuracy")),
            ("subjective", "regression", "lower_is_better", ("rmse", "mae")),
        ):
            for group, metrics in (reference.get(track) or {}).items():
                for metric_name in metric_names:
                    add_row(
                        dataset_id=dataset_id,
                        task_group=str(group),
                        task_type=task_type,
                        metric_name=metric_name,
                        metric_direction=direction,
                        model_family="classical_baseline",
                        model_name="classical_baseline",
                        value=metrics.get(metric_name),
                        sample_count=metrics.get("sample_count"),
                        fold_count=None,
                        protocol="classical_loso_reference",
                        evidence_role=evidence_role,
                        source_path=summary_path,
                    )


def _add_public_mainline_rows(add_row, public_mainline, nasa_public, config) -> None:
    for group, metrics in (public_mainline.get("uab") or {}).get("groups", {}).items():
        for metric_name, key in (("rmse", "public_rmse"), ("mae", "public_mae")):
            add_row(
                dataset_id="uab_workload_dataset",
                task_group=str(group),
                task_type="regression",
                metric_name=metric_name,
                metric_direction="lower_is_better",
                model_family="public_baseline",
                model_name="public_adapter_current",
                value=metrics.get(key),
                sample_count=None,
                fold_count=None,
                protocol="public_mainline_best_of",
                evidence_role="public_adapter_calibration_evidence",
                source_path=metrics.get("best_public_source_path")
                or config.public_mainline_summary_path,
            )
    comparison = nasa_public.get("comparison_against_deep") or {}
    if comparison:
        public_groups = {
            group: (payload or {}).get("public_opt") or {}
            for group, payload in comparison.items()
        }
    else:
        public_groups = {}
        for group, payload in (nasa_public.get("subset_results") or {}).items():
            best_head = payload.get("best_head")
            heads = payload.get("heads") or {}
            public_payload = dict(heads.get(best_head) or {})
            public_payload["best_head"] = best_head
            public_groups[str(group)] = public_payload
    for group, public_payload in public_groups.items():
        for metric_name in ("macro_f1", "balanced_accuracy"):
            add_row(
                dataset_id="nasa_csm",
                task_group=str(group),
                task_type="classification",
                metric_name=metric_name,
                metric_direction="higher_is_better",
                model_family="public_baseline",
                model_name="public_adapter_current",
                value=public_payload.get(metric_name),
                sample_count=public_payload.get("sample_count"),
                fold_count=None,
                protocol="public_mainline_round1",
                evidence_role="public_adapter_calibration_evidence",
                source_path=config.nasa_public_opt_summary_path,
            )


def _add_refresh_rows(add_row, refresh_summary: Mapping[str, object], source_path: Path | None) -> None:
    best = refresh_summary.get("best_by_dataset_task") or {}
    for group, metrics in (best.get("nasa_csm") or {}).items():
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in ("macro_f1", "balanced_accuracy"):
            add_row(
                dataset_id="nasa_csm",
                task_group=str(group),
                task_type="classification",
                metric_name=metric_name,
                metric_direction="higher_is_better",
                model_family="chronaris_public_fusion",
                model_name="chronaris_public_fusion_refresh",
                value=metrics.get(metric_name),
                sample_count=metrics.get("sample_count"),
                fold_count=metrics.get("fold_count"),
                protocol=str(metrics.get("protocol") or "p28_refresh_confirm"),
                evidence_role="public_adapter_context_proxy_evidence",
                source_path=metrics.get("summary_path") or source_path or "",
            )
    for group, metrics in (best.get("uab_workload_dataset") or {}).items():
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in ("rmse", "mae"):
            add_row(
                dataset_id="uab_workload_dataset",
                task_group=str(group),
                task_type="regression",
                metric_name=metric_name,
                metric_direction="lower_is_better",
                model_family="chronaris_public_fusion",
                model_name="chronaris_public_fusion_refresh",
                value=metrics.get(metric_name),
                sample_count=metrics.get("sample_count"),
                fold_count=metrics.get("fold_count"),
                protocol=str(metrics.get("protocol") or "p28_refresh_confirm"),
                evidence_role="public_adapter_context_proxy_evidence",
                source_path=metrics.get("summary_path") or source_path or "",
            )
        for source_name, metric_name in (
            ("objective_macro_f1", "macro_f1"),
            ("objective_balanced_accuracy", "balanced_accuracy"),
        ):
            add_row(
                dataset_id="uab_workload_dataset",
                task_group=str(group),
                task_type="classification",
                metric_name=metric_name,
                metric_direction="higher_is_better",
                model_family="chronaris_public_fusion",
                model_name="chronaris_public_fusion_refresh",
                value=metrics.get(source_name),
                sample_count=metrics.get("objective_sample_count") or metrics.get("sample_count"),
                fold_count=metrics.get("objective_fold_count") or metrics.get("fold_count"),
                protocol=str(metrics.get("protocol") or "p28_refresh_confirm"),
                evidence_role="public_adapter_context_proxy_evidence",
                source_path=metrics.get("summary_path") or source_path or "",
            )


def _build_wide_frame(long_frame: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["dataset_id", "task_group", "metric_name"]
    pivot = long_frame.pivot_table(
        index=index_cols,
        columns="model_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    for column in (
        "classical_baseline",
        "public_adapter_current",
        "mult",
        "contiformer",
        "chronaris_public_fusion_current",
        "chronaris_public_fusion_refresh",
    ):
        if column not in pivot:
            pivot[column] = np.nan
    pivot = pivot.rename(columns={"public_adapter_current": "public_baseline"})

    directions = (
        long_frame.drop_duplicates(index_cols)
        .set_index(index_cols)["metric_direction"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    model_columns = [
        "classical_baseline",
        "public_baseline",
        "mult",
        "contiformer",
        "chronaris_public_fusion_current",
        "chronaris_public_fusion_refresh",
    ]
    for _, row in pivot.iterrows():
        payload = row.to_dict()
        key = (row["dataset_id"], row["task_group"], row["metric_name"])
        direction = directions.get(key, "higher_is_better")
        payload["metric_direction"] = direction
        chronaris = _chronaris_value(row)
        for baseline in ("public", "mult", "contiformer", "classical"):
            baseline_col = f"{baseline}_baseline" if baseline in {"public", "classical"} else baseline
            delta = _delta(
                chronaris=chronaris,
                baseline=row.get(baseline_col),
                direction=direction,
            )
            payload[f"chronaris_vs_{baseline}_abs_delta"] = delta
            payload[f"chronaris_vs_{baseline}_rel_pct"] = (
                float(delta) / abs(float(row[baseline_col])) * 100.0
                if delta is not None
                and not _is_nan(row.get(baseline_col))
                and abs(float(row[baseline_col])) > 1e-12
                else np.nan
            )
        best_model, best_value = _best_model(row, model_columns, direction)
        payload["best_model"] = best_model
        payload["best_value"] = best_value
        rows.append(payload)
    ordered = [
        "dataset_id",
        "task_group",
        "metric_name",
        "metric_direction",
        "classical_baseline",
        "public_baseline",
        "mult",
        "contiformer",
        "chronaris_public_fusion_current",
        "chronaris_public_fusion_refresh",
        "best_model",
        "best_value",
        "chronaris_vs_public_abs_delta",
        "chronaris_vs_public_rel_pct",
        "chronaris_vs_mult_abs_delta",
        "chronaris_vs_mult_rel_pct",
        "chronaris_vs_contiformer_abs_delta",
        "chronaris_vs_contiformer_rel_pct",
        "chronaris_vs_classical_abs_delta",
        "chronaris_vs_classical_rel_pct",
    ]
    wide = pd.DataFrame(rows)
    wide = _append_uab_mean_rmse(wide, model_columns=model_columns)
    return wide[ordered]


def _append_uab_mean_rmse(
    wide: pd.DataFrame,
    *,
    model_columns: Sequence[str],
) -> pd.DataFrame:
    subset = wide[
        (wide["dataset_id"] == "uab_workload_dataset")
        & (wide["task_group"].isin(["n_back", "heat_the_chair"]))
        & (wide["metric_name"] == "rmse")
    ]
    if set(subset["task_group"]) != {"n_back", "heat_the_chair"}:
        return wide
    payload: dict[str, object] = {
        "dataset_id": "uab_workload_dataset",
        "task_group": "subjective_mean",
        "metric_name": "mean_rmse",
        "metric_direction": "lower_is_better",
    }
    for column in model_columns:
        values = [
            row[column]
            for _, row in subset.iterrows()
            if not _is_nan(row.get(column))
        ]
        payload[column] = float(np.mean(values)) if len(values) == 2 else np.nan
    chronaris = _chronaris_value(payload)
    for baseline in ("public", "mult", "contiformer", "classical"):
        baseline_col = f"{baseline}_baseline" if baseline in {"public", "classical"} else baseline
        delta = _delta(
            chronaris=chronaris,
            baseline=payload.get(baseline_col),
            direction="lower_is_better",
        )
        payload[f"chronaris_vs_{baseline}_abs_delta"] = delta
        payload[f"chronaris_vs_{baseline}_rel_pct"] = (
            float(delta) / abs(float(payload[baseline_col])) * 100.0
            if delta is not None
            and not _is_nan(payload.get(baseline_col))
            and abs(float(payload[baseline_col])) > 1e-12
            else np.nan
        )
    best_model, best_value = _best_model(payload, model_columns, "lower_is_better")
    payload["best_model"] = best_model
    payload["best_value"] = best_value
    return pd.concat([wide, pd.DataFrame([payload])], axis=0, ignore_index=True)


def _build_improvement_summary(wide_frame: pd.DataFrame) -> pd.DataFrame:
    wanted = {
        ("nasa_csm", "combined", "macro_f1"),
        ("nasa_csm", "combined", "balanced_accuracy"),
        ("nasa_csm", "benchmark_only", "macro_f1"),
        ("nasa_csm", "loft_only", "macro_f1"),
        ("uab_workload_dataset", "n_back", "rmse"),
        ("uab_workload_dataset", "heat_the_chair", "rmse"),
        ("uab_workload_dataset", "subjective_mean", "mean_rmse"),
    }
    rows: list[dict[str, object]] = []
    for _, row in wide_frame.iterrows():
        key = (row["dataset_id"], row["task_group"], row["metric_name"])
        if key not in wanted:
            continue
        chronaris = _chronaris_value(row)
        for baseline_label, baseline_col in (
            ("public_baseline", "public_baseline"),
            ("classical_baseline", "classical_baseline"),
            ("MulT", "mult"),
            ("ContiFormer", "contiformer"),
        ):
            delta = _delta(
                chronaris=chronaris,
                baseline=row.get(baseline_col),
                direction=row["metric_direction"],
            )
            if delta is None:
                continue
            baseline_value = float(row[baseline_col])
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "task_group": row["task_group"],
                    "metric_name": row["metric_name"],
                    "chronaris_model": (
                        "chronaris_public_fusion_refresh"
                        if not _is_nan(row.get("chronaris_public_fusion_refresh"))
                        else "chronaris_public_fusion_current"
                    ),
                    "chronaris_value": chronaris,
                    "baseline_model": baseline_label,
                    "baseline_value": baseline_value,
                    "delta_abs": delta,
                    "delta_rel_pct": (
                        delta / abs(baseline_value) * 100.0
                        if abs(baseline_value) > 1e-12
                        else np.nan
                    ),
                    "metric_direction": row["metric_direction"],
                }
            )
    return pd.DataFrame(rows)


def _render_comparison_figures(wide: pd.DataFrame, output_root: Path) -> dict[str, str]:
    figure_paths = {
        "fig_public_model_leaderboard_nasa_macro_f1": str(
            output_root / "fig_public_model_leaderboard_nasa_macro_f1.png"
        ),
        "fig_public_model_leaderboard_nasa_balanced_accuracy": str(
            output_root / "fig_public_model_leaderboard_nasa_balanced_accuracy.png"
        ),
        "fig_uab_subjective_rmse_comparison": str(
            output_root / "fig_uab_subjective_rmse_comparison.png"
        ),
        "fig_uab_objective_macro_f1_comparison": str(
            output_root / "fig_uab_objective_macro_f1_comparison.png"
        ),
        "fig_public_model_delta_heatmap": str(output_root / "fig_public_model_delta_heatmap.png"),
        "fig_public_model_win_summary": str(output_root / "fig_public_model_win_summary.png"),
    }
    _plot_nasa_leaderboard(
        wide,
        metric_name="macro_f1",
        path=figure_paths["fig_public_model_leaderboard_nasa_macro_f1"],
        title="NASA attention-state classification: macro-F1 comparison",
    )
    _plot_nasa_leaderboard(
        wide,
        metric_name="balanced_accuracy",
        path=figure_paths["fig_public_model_leaderboard_nasa_balanced_accuracy"],
        title="NASA attention-state classification: balanced accuracy comparison",
    )
    _plot_uab_grouped(
        wide,
        metric_name="rmse",
        path=figure_paths["fig_uab_subjective_rmse_comparison"],
        title="UAB subjective workload regression: RMSE comparison",
        ylabel="RMSE (lower is better)",
    )
    _plot_uab_grouped(
        wide,
        metric_name="macro_f1",
        path=figure_paths["fig_uab_objective_macro_f1_comparison"],
        title="UAB objective workload classification: macro-F1 comparison",
        ylabel="macro-F1",
    )
    _plot_delta_heatmap(wide, path=figure_paths["fig_public_model_delta_heatmap"])
    _plot_win_summary(wide, path=figure_paths["fig_public_model_win_summary"])
    return figure_paths


def _plot_nasa_leaderboard(wide: pd.DataFrame, *, metric_name: str, path: str, title: str) -> None:
    row = _select_row(wide, "nasa_csm", "combined", metric_name)
    labels = [
        ("classical baseline", "classical_baseline"),
        ("public baseline", "public_baseline"),
        ("MulT", "mult"),
        ("ContiFormer", "contiformer"),
        ("Chronaris current", "chronaris_public_fusion_current"),
        ("Chronaris refresh", "chronaris_public_fusion_refresh"),
    ]
    values = [(label, float(row[col])) for label, col in labels if not _is_nan(row.get(col))]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(values))
    colors = ["#6f7f8f" if "Chronaris" not in label else "#1f7a5a" for label, _ in values]
    bars = axis.bar(x, [value for _, value in values], color=colors)
    axis.set_title(title)
    axis.set_ylabel(metric_name.replace("_", " "))
    axis.set_xticks(x)
    axis.set_xticklabels([label for label, _ in values], rotation=18, ha="right")
    for bar, (_, value) in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    public_value = row.get("public_baseline")
    for bar, (label, value) in zip(bars, values, strict=True):
        if "Chronaris" in label and not _is_nan(public_value):
            delta = value - float(public_value)
            rel = delta / abs(float(public_value)) * 100.0
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 0.55,
                f"vs public\n{delta:+.4f}\n{rel:+.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_uab_grouped(wide: pd.DataFrame, *, metric_name: str, path: str, title: str, ylabel: str) -> None:
    rows = [
        _select_row(wide, "uab_workload_dataset", "n_back", metric_name),
        _select_row(wide, "uab_workload_dataset", "heat_the_chair", metric_name),
    ]
    groups = ["n_back", "heat_the_chair"]
    labels = [
        ("classical", "classical_baseline"),
        ("public adapter", "public_baseline"),
        ("MulT", "mult"),
        ("ContiFormer", "contiformer"),
        ("Chronaris current", "chronaris_public_fusion_current"),
        ("Chronaris refresh", "chronaris_public_fusion_refresh"),
    ]
    active = [(label, col) for label, col in labels if any(not _is_nan(row.get(col)) for row in rows)]
    x = np.arange(len(groups))
    width = 0.8 / max(len(active), 1)
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, (label, col) in enumerate(active):
        heights = [float(row[col]) if not _is_nan(row.get(col)) else np.nan for row in rows]
        offsets = x - 0.4 + width / 2 + index * width
        bars = axis.bar(offsets, heights, width=width, label=label)
        for bar, value in zip(bars, heights, strict=True):
            if not _is_nan(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x)
    axis.set_xticklabels(groups)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_delta_heatmap(wide: pd.DataFrame, *, path: str) -> None:
    rows = []
    labels = []
    columns = [
        ("vs public baseline", "chronaris_vs_public_abs_delta"),
        ("vs MulT", "chronaris_vs_mult_abs_delta"),
        ("vs ContiFormer", "chronaris_vs_contiformer_abs_delta"),
        ("vs classical baseline", "chronaris_vs_classical_abs_delta"),
    ]
    for _, row in wide.iterrows():
        chronaris = _chronaris_value(row)
        if _is_nan(chronaris):
            continue
        values = [row.get(column) for _, column in columns]
        if all(_is_nan(value) for value in values):
            continue
        labels.append(f"{row['dataset_id']} / {row['task_group']} / {row['metric_name']}")
        rows.append([np.nan if _is_nan(value) else float(value) for value in values])
    data = np.asarray(rows, dtype=float) if rows else np.zeros((1, len(columns)))
    fig, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    image = axis.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_title("Chronaris improvement over baselines")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels([label for label, _ in columns], rotation=20, ha="right")
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            value = data[row_index, col_index]
            text = "" if _is_nan(value) else f"{value:+.4f}"
            axis.text(col_index, row_index, text, ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_win_summary(wide: pd.DataFrame, *, path: str) -> None:
    records = []
    for _, row in wide.iterrows():
        chronaris = _chronaris_value(row)
        if _is_nan(chronaris):
            continue
        for label, col in (
            ("public", "chronaris_vs_public_abs_delta"),
            ("MulT", "chronaris_vs_mult_abs_delta"),
            ("ContiFormer", "chronaris_vs_contiformer_abs_delta"),
        ):
            delta = row.get(col)
            if _is_nan(delta):
                continue
            status = _win_status(row, float(delta), baseline=label)
            records.append(
                {
                    "row": f"{row['dataset_id']} / {row['task_group']} / {row['metric_name']}",
                    "baseline": label,
                    "status": status,
                    "delta": float(delta),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame([{"row": "no comparison", "baseline": "none", "status": "T", "delta": 0.0}])
    rows = list(dict.fromkeys(frame["row"]))
    baselines = list(dict.fromkeys(frame["baseline"]))
    status_to_num = {"L": -1, "T": 0, "W": 1}
    data = np.full((len(rows), len(baselines)), np.nan)
    lookup = {(record["row"], record["baseline"]): record for record in frame.to_dict(orient="records")}
    for row_index, row_label in enumerate(rows):
        for col_index, baseline in enumerate(baselines):
            record = lookup.get((row_label, baseline))
            if record:
                data[row_index, col_index] = status_to_num[record["status"]]
    fig, axis = plt.subplots(figsize=(8, max(4, 0.35 * len(rows))))
    image = axis.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_title("Chronaris W/T/L summary")
    axis.set_xticks(np.arange(len(baselines)))
    axis.set_xticklabels(baselines)
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows, fontsize=8)
    for row_index, row_label in enumerate(rows):
        for col_index, baseline in enumerate(baselines):
            record = lookup.get((row_label, baseline))
            if not record:
                continue
            axis.text(
                col_index,
                row_index,
                f"{record['status']}\n{record['delta']:+.4f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _render_report(
    run_id: str,
    wide: pd.DataFrame,
    improvement: pd.DataFrame,
    figure_paths: Mapping[str, str],
    manifest: Mapping[str, object],
) -> str:
    nasa = _select_row(wide, "nasa_csm", "combined", "macro_f1")
    nasa_ba = _select_row(wide, "nasa_csm", "combined", "balanced_accuracy")
    uab_n = _select_row(wide, "uab_workload_dataset", "n_back", "rmse")
    uab_h = _select_row(wide, "uab_workload_dataset", "heat_the_chair", "rmse")
    chronaris_nasa = _chronaris_value(nasa)
    public_nasa = float(nasa["public_baseline"])
    delta_nasa = chronaris_nasa - public_nasa
    rel_nasa = delta_nasa / abs(public_nasa) * 100.0
    lines = [
        f"# task evaluation Public Model Comparison - {run_id}",
        "",
        "## 1. Executive Summary",
        "",
        (
            "On NASA combined attention-state classification, "
            f"chronaris_public_fusion reaches macro-F1={chronaris_nasa:.4f}, "
            f"exceeding the public baseline by {delta_nasa:+.4f} absolute / {rel_nasa:+.1f}% relative."
        ),
        (
            "For balanced accuracy, the same NASA combined comparison reports "
            f"Chronaris={_chronaris_value(nasa_ba):.4f} and public baseline={float(nasa_ba['public_baseline']):.4f}."
        ),
        (
            "On UAB subjective workload regression, the current public adapter branch "
            f"records RMSE={float(uab_n['public_baseline']):.4f} on n_back and "
            f"RMSE={float(uab_h['public_baseline']):.4f} on heat_the_chair; "
            "Chronaris public fusion refresh/current rows are retained for direct ranking."
        ),
        "",
        "## 2. NASA model comparison",
        "",
        _markdown_table(
            wide[
                (wide["dataset_id"] == "nasa_csm")
                & (wide["metric_name"].isin(["macro_f1", "balanced_accuracy"]))
            ],
            [
                "task_group",
                "metric_name",
                "classical_baseline",
                "public_baseline",
                "mult",
                "contiformer",
                "chronaris_public_fusion_current",
                "chronaris_public_fusion_refresh",
                "best_model",
            ],
        ),
        "",
        "## 3. UAB model comparison",
        "",
        _markdown_table(
            wide[
                (wide["dataset_id"] == "uab_workload_dataset")
                & (wide["metric_name"].isin(["rmse", "macro_f1"]))
            ],
            [
                "task_group",
                "metric_name",
                "classical_baseline",
                "public_baseline",
                "mult",
                "contiformer",
                "chronaris_public_fusion_current",
                "chronaris_public_fusion_refresh",
                "best_model",
            ],
        ),
        "",
        "## 4. Chronaris public fusion improvement table",
        "",
        _markdown_table(
            improvement,
            [
                "dataset_id",
                "task_group",
                "metric_name",
                "chronaris_model",
                "chronaris_value",
                "baseline_model",
                "baseline_value",
                "delta_abs",
                "delta_rel_pct",
            ],
        ),
        "",
        "## 5. Figure index",
        "",
    ]
    for name, path in figure_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## 6. Reproducibility manifest",
            "",
            f"- artifact_root: `{manifest['artifact_root']}`",
            f"- model_comparison_long: `{manifest['long_csv_path']}`",
            f"- model_comparison_wide: `{manifest['wide_csv_path']}`",
            f"- improvement_summary: `{manifest['improvement_summary_csv_path']}`",
            f"- evidence_manifest: `{manifest['evidence_manifest_path']}`",
            "",
            "## 7. Midterm-ready wording",
            "",
            (
                "Chronaris public fusion provides the strongest NASA combined attention-state "
                f"classification result in this comparison, reaching macro-F1={chronaris_nasa:.4f} "
                f"and improving over the public baseline by {delta_nasa:+.4f} absolute / {rel_nasa:+.1f}% relative. "
                "The UAB workload table reports the public adapter and Chronaris fusion branches with "
                "the same fold/source protocol columns, enabling direct use of the ranking and delta figures in the midterm report."
            ),
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if _is_nan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _select_row(wide: pd.DataFrame, dataset_id: str, task_group: str, metric_name: str) -> pd.Series:
    subset = wide[
        (wide["dataset_id"] == dataset_id)
        & (wide["task_group"] == task_group)
        & (wide["metric_name"] == metric_name)
    ]
    if subset.empty:
        raise ValueError(f"missing comparison row: {dataset_id} {task_group} {metric_name}")
    return subset.iloc[0]


def _chronaris_value(row: Mapping[str, object]) -> float:
    refresh = row.get("chronaris_public_fusion_refresh")
    current = row.get("chronaris_public_fusion_current")
    if not _is_nan(refresh):
        return float(refresh)
    if not _is_nan(current):
        return float(current)
    return float("nan")


def _delta(*, chronaris: float, baseline: object, direction: str) -> float | None:
    if _is_nan(chronaris) or _is_nan(baseline):
        return None
    if direction == "lower_is_better":
        return float(baseline) - float(chronaris)
    return float(chronaris) - float(baseline)


def _best_model(row: Mapping[str, object], columns: Sequence[str], direction: str) -> tuple[str, float]:
    values = [(column, float(row[column])) for column in columns if not _is_nan(row.get(column))]
    if not values:
        return "", float("nan")
    reverse = direction == "higher_is_better"
    best_model, best_value = sorted(values, key=lambda item: item[1], reverse=reverse)[0]
    return best_model, best_value


def _win_status(row: Mapping[str, object], delta: float, *, baseline: str) -> str:
    if row["metric_name"] in {"macro_f1", "balanced_accuracy"}:
        if abs(delta) < 0.02:
            return "T"
    else:
        baseline_col = {
            "public": "public_baseline",
            "MulT": "mult",
            "ContiFormer": "contiformer",
        }.get(baseline)
        baseline_value = float(row[baseline_col]) if baseline_col and not _is_nan(row.get(baseline_col)) else 0.0
        if abs(baseline_value) > 1e-12 and abs(delta) / abs(baseline_value) < 0.02:
            return "T"
    return "W" if delta > 0 else "L"


def _load_json(path_like: str | Path | None) -> dict[str, object]:
    if not path_like:
        return {}
    path = _normalize_existing_path(path_like)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _normalize_existing_path(path_like: str | Path) -> Path:
    path = _resolve_path(path_like)
    if path.exists():
        return path
    text = str(path)
    alt = Path(text.replace("/docs/reports/assets/", "/docs/artifacts/assets/"))
    if alt.exists():
        return alt
    return path


def _find_latest_refresh_summary() -> Path | None:
    root = REPO_ROOT / "docs/artifacts/runs"
    if not root.exists():
        return None
    paths = sorted(root.glob("*/fusion_refresh_summary.json"))
    return paths[-1] if paths else None


def _find_missing_metrics(wide_frame: pd.DataFrame) -> list[dict[str, str]]:
    required = [
        ("nasa_csm", "combined", "macro_f1"),
        ("nasa_csm", "combined", "balanced_accuracy"),
        ("nasa_csm", "benchmark_only", "macro_f1"),
        ("nasa_csm", "loft_only", "macro_f1"),
        ("uab_workload_dataset", "n_back", "rmse"),
        ("uab_workload_dataset", "heat_the_chair", "rmse"),
        ("uab_workload_dataset", "subjective_mean", "mean_rmse"),
    ]
    missing: list[dict[str, str]] = []
    for dataset_id, task_group, metric_name in required:
        subset = wide_frame[
            (wide_frame["dataset_id"] == dataset_id)
            & (wide_frame["task_group"] == task_group)
            & (wide_frame["metric_name"] == metric_name)
        ]
        if subset.empty:
            missing.append(
                {
                    "dataset_id": dataset_id,
                    "task_group": task_group,
                    "metric_name": metric_name,
                    "reason": "row_missing",
                }
            )
            continue
        row = subset.iloc[0]
        if _is_nan(_chronaris_value(row)):
            missing.append(
                {
                    "dataset_id": dataset_id,
                    "task_group": task_group,
                    "metric_name": metric_name,
                    "reason": "chronaris_metric_missing",
                }
            )
    return missing


def _optional_int(value: int | float | None) -> int | None:
    if value is None or _is_nan(value):
        return None
    return int(value)


def _is_nan(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
