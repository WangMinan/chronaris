"""P36 optimized Chronaris re-evaluation aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.modeling.common.plot_labels import label_vertical_bars


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P30_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-private-thirdparty-comparison-gpuopt-r1"
)
DEFAULT_P31_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-public-fusion-ablation-gpuopt-r1"
)
DEFAULT_P32_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-cross-evidence-matrix-gpuopt-r1"
)
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"


@dataclass(frozen=True, slots=True)
class StageIOptimizedReevaluationConfig:
    run_id: str
    p30_root: str = str(DEFAULT_P30_ROOT)
    p31_root: str = str(DEFAULT_P31_ROOT)
    p32_root: str = str(DEFAULT_P32_ROOT)
    p34_root: str | None = None
    p35_root: str | None = None
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT


@dataclass(frozen=True, slots=True)
class StageIOptimizedReevaluationResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_optimized_reevaluation(
    config: StageIOptimizedReevaluationConfig,
) -> StageIOptimizedReevaluationResult:
    run_root = _resolve_path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    p30_root = _resolve_path(config.p30_root)
    p31_root = _resolve_path(config.p31_root)
    p32_root = _resolve_path(config.p32_root)
    p34_root = _resolve_path(config.p34_root) if config.p34_root else _latest_root("task_eval_task_heads_optimization")
    p35_root = _resolve_path(config.p35_root) if config.p35_root else _latest_root("task_eval_stream_role_fusion")
    runtime_device = _aggregate_runtime_device(p34_root, p35_root)
    _write_json(
        run_root / "optimized_reevaluation_config.json",
        {
            "run_id": config.run_id,
            "runtime_device": runtime_device,
            "p30_root": str(p30_root),
            "p31_root": str(p31_root),
            "p32_root": str(p32_root),
            "p34_root": str(p34_root) if p34_root else None,
            "p35_root": str(p35_root) if p35_root else None,
        },
    )
    status = _optimized_status(p34_root, p35_root)
    private_path = _write_private_comparison(p30_root, p34_root, p35_root, run_root / "optimized_private_comparison.csv")
    public_path = _copy_or_empty(
        p35_root / "public_metrics.csv" if p35_root else None,
        run_root / "optimized_public_comparison.csv",
    )
    delta_p30_path = _copy_or_empty(
        p34_root / "improvement_vs_p30.csv" if p34_root else None,
        run_root / "optimized_delta_vs_p30.csv",
    )
    delta_p31_path = _copy_or_empty(
        p35_root / "comparison_vs_p31.csv" if p35_root else None,
        run_root / "optimized_delta_vs_p31.csv",
    )
    delta_p27_path = _write_delta_vs_public_prior(p31_root, run_root / "optimized_delta_vs_p27_p28.csv")
    cross_path = _write_cross_evidence(p32_root, private_path, public_path, run_root / "optimized_cross_evidence_matrix.csv")
    model_selection_path = run_root / "model_selection_summary.json"
    model_summary = _model_selection_summary(private_path, public_path, delta_p30_path, delta_p31_path, status)
    model_summary["runtime_device"] = runtime_device
    _write_json(model_selection_path, model_summary)
    figure_paths = _render_figures(run_root, private_path, public_path, delta_p30_path, delta_p31_path, cross_path)
    report_path = _resolve_path(config.report_root) / f"task-eval-optimized-chronaris-reevaluation-{config.run_id}.md"
    manifest_path = run_root / "evidence_manifest.json"
    summary = {
        "run_id": config.run_id,
        "status": status,
        "runtime_device": runtime_device,
        "generated_at_utc": _utc_now(),
        "artifact_root": str(run_root),
        "p30_root": str(p30_root),
        "p31_root": str(p31_root),
        "p32_root": str(p32_root),
        "p34_root": str(p34_root) if p34_root else None,
        "p35_root": str(p35_root) if p35_root else None,
        "optimized_reevaluation_config_json": str(run_root / "optimized_reevaluation_config.json"),
        "optimized_private_comparison_csv": str(private_path),
        "optimized_public_comparison_csv": str(public_path),
        "optimized_delta_vs_p30_csv": str(delta_p30_path),
        "optimized_delta_vs_p31_csv": str(delta_p31_path),
        "optimized_delta_vs_p27_p28_csv": str(delta_p27_path),
        "optimized_cross_evidence_matrix_csv": str(cross_path),
        "model_selection_summary_json": str(model_selection_path),
        "figure_paths": figure_paths,
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "protocol_boundary": _protocol_boundary(status),
    }
    summary_path = run_root / "optimized_reevaluation_summary.json"
    _write_json(summary_path, summary)
    _write_json(manifest_path, {**summary, "summary_path": str(summary_path), "stage": "P36"})
    if status != "completed":
        _write_json(run_root / "partial_summary.json", summary)
    _write_json(
        run_root / "progress.json",
        {
            "run_id": config.run_id,
            "stage": "P36",
            "status": status,
            "runtime_device": runtime_device,
            "completed": True,
            "artifact_root": str(run_root),
            "generated_at_utc": summary["generated_at_utc"],
        },
    )
    (run_root / "resume_command.txt").write_text(_resume_command(config, p30_root, p31_root, p32_root, p34_root, p35_root) + "\n", encoding="utf-8")
    (run_root / "run.log").write_text(
        "\n".join(
            [
                f"{summary['generated_at_utc']} INFO stage=P36 run_id={config.run_id} status={status} runtime_device={runtime_device}",
                "P36 aggregation used fixed P30/P31/P32 references and current P34/P35 optimized artifacts.",
                "No historical baseline artifacts were overwritten.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(summary, model_summary) + "\n", encoding="utf-8")
    return StageIOptimizedReevaluationResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _write_private_comparison(
    p30_root: Path,
    p34_root: Path | None,
    p35_root: Path | None,
    output_path: Path,
) -> Path:
    p30 = pd.read_csv(p30_root / "model_comparison_wide.csv")
    rows = []
    for row in p30.to_dict(orient="records"):
        for model in ("chronaris_full", "mult", "contiformer", "classical_baseline", "naive_time_sync"):
            if model in row and not pd.isna(row[model]):
                rows.append({**_metric_keys(row), "model_name": model, "value": float(row[model]), "source": "P30"})
    if p34_root:
        p34_path = p34_root / "task_head_metrics_wide.csv"
        if not p34_path.exists():
            p34_path = p34_root / "model_comparison_wide.csv"
        if p34_path.exists():
            p34 = pd.read_csv(p34_path)
            for row in p34.to_dict(orient="records"):
                for model in ("chronaris_v2_task_heads", "v2_no_vehicle_aux_head", "v2_no_residual_t2_head", "v2_no_contrastive_t3_loss"):
                    if model in row and not pd.isna(row[model]):
                        rows.append({**_metric_keys(row), "model_name": model, "value": float(row[model]), "source": "P34"})
    if p35_root and (p35_root / "private_metrics.csv").exists():
        p35 = _safe_read_csv(p35_root / "private_metrics.csv")
        if not p35.empty:
            if "source_stage" in p35:
                p35 = p35[p35["source_stage"].astype(str).eq("P35_v3_confirm")]
            if "value_mean" in p35:
                for row in p35.to_dict(orient="records"):
                    rows.append(
                        {
                            **_metric_keys(row),
                            "model_name": row.get("model_name"),
                            "value": float(row["value_mean"]),
                            "source": "P35_v3_confirm",
                        }
                    )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _write_delta_vs_public_prior(p31_root: Path, output_path: Path) -> Path:
    source = p31_root / "ablation_summary.csv"
    if not source.exists():
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path
    frame = pd.read_csv(source)
    keep = frame[frame["variant_id"].isin(["full", "no_lag_window", "context_only"])].copy()
    keep["comparison_boundary"] = "P31/P27/P28 fixed public context-proxy reference"
    keep.to_csv(output_path, index=False)
    return output_path


def _write_cross_evidence(
    p32_root: Path,
    private_path: Path,
    public_path: Path,
    output_path: Path,
) -> Path:
    frames = []
    p32 = p32_root / "cross_evidence_matrix.csv"
    if p32.exists():
        frames.append(pd.read_csv(p32))
    private = pd.read_csv(private_path)
    if not private.empty:
        frames.append(
            pd.DataFrame(
                {
                    "evidence_quadrant": "optimized_private_reevaluation",
                    "dataset_role": "private_real_dual_stream",
                    "dataset_id": "private_feature_export",
                    "task_group": private["task_name"],
                    "model_or_component": private["model_name"],
                    "metric": private["metric"],
                    "value": private["value"],
                    "protocol": private["source"],
                    "midterm_use": "optimized Chronaris comparison",
                    "wording_boundary": "T1/T2/T3 remain private proxy tasks",
                }
            )
        )
    public = _safe_read_csv(public_path)
    if not public.empty and "dataset_id" in public:
        frames.append(
            pd.DataFrame(
                {
                    "evidence_quadrant": "optimized_public_route_reference",
                    "dataset_role": "public_context_proxy",
                    "dataset_id": public["dataset_id"],
                    "task_group": public.get("task_group", ""),
                    "model_or_component": public.get("variant_id", ""),
                    "metric": public.get("metric", ""),
                    "value": public.get("value_mean", np.nan),
                    "protocol": "P35/P31 reference",
                    "midterm_use": "stream-role routing boundary",
                    "wording_boundary": "public second stream is context proxy, not real vehicle",
                }
            )
        )
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output_path, index=False)
    return output_path


def _model_selection_summary(
    private_path: Path,
    public_path: Path,
    delta_p30_path: Path,
    delta_p31_path: Path,
    status: str,
) -> dict[str, object]:
    private = _safe_read_csv(private_path)
    public = _safe_read_csv(public_path)
    delta_p30 = _safe_read_csv(delta_p30_path) if delta_p30_path.exists() else pd.DataFrame()
    delta_p31 = _safe_read_csv(delta_p31_path) if delta_p31_path.exists() else pd.DataFrame()
    return {
        "private_rows": int(private.shape[0]),
        "public_rows": int(public.shape[0]),
        "delta_vs_p30_rows": int(delta_p30.shape[0]),
        "delta_vs_p31_rows": int(delta_p31.shape[0]),
        "selection_status": "completed_requested_p35_v3_confirm" if status == "completed" else "partial_until_full_p35_v3_confirm",
        "positive_delta_convention": "classification/retrieval optimized-baseline; regression baseline-optimized",
    }


def _render_figures(
    run_root: Path,
    private_path: Path,
    public_path: Path,
    delta_p30_path: Path,
    delta_p31_path: Path,
    cross_path: Path,
) -> dict[str, str]:
    paths = {
        "fig_p36_private_before_after": str(run_root / "fig_p36_private_before_after.png"),
        "fig_p36_private_vs_thirdparty_delta": str(run_root / "fig_p36_private_vs_thirdparty_delta.png"),
        "fig_p36_public_before_after": str(run_root / "fig_p36_public_before_after.png"),
        "fig_p36_taskwise_win_summary": str(run_root / "fig_p36_taskwise_win_summary.png"),
        "fig_p36_cross_evidence_matrix_v2": str(run_root / "fig_p36_cross_evidence_matrix_v2.png"),
        "fig_p36_method_claim_support_map": str(run_root / "fig_p36_method_claim_support_map.png"),
    }
    _metric_plot(_safe_read_csv(private_path), paths["fig_p36_private_before_after"], "value")
    _metric_plot(_safe_read_csv(delta_p30_path), paths["fig_p36_private_vs_thirdparty_delta"], "delta_abs_positive_is_better")
    public = _safe_read_csv(public_path)
    _metric_plot(public, paths["fig_p36_public_before_after"], "value_mean")
    _win_plot(_safe_read_csv(delta_p30_path), paths["fig_p36_taskwise_win_summary"])
    _count_plot(_safe_read_csv(cross_path), "evidence_quadrant", paths["fig_p36_cross_evidence_matrix_v2"])
    _count_plot(_safe_read_csv(cross_path), "midterm_use", paths["fig_p36_method_claim_support_map"])
    return paths


def _metric_plot(frame: pd.DataFrame, path: str, value_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if frame.empty or value_col not in frame:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        label_col = "model_name" if "model_name" in frame else "variant_id" if "variant_id" in frame else "metric"
        labels = frame[label_col].astype(str).str.slice(0, 28) + "\n" + frame.get("metric", "").astype(str)
        values = frame[value_col].astype(float)
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _win_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if frame.empty or "delta_abs_positive_is_better" not in frame:
        counts = {"win": 0, "tie": 0, "loss": 0}
    else:
        values = frame["delta_abs_positive_is_better"].astype(float)
        counts = {"win": int((values > 1e-9).sum()), "tie": int((values.abs() <= 1e-9).sum()), "loss": int((values < -1e-9).sum())}
    values = list(counts.values())
    bars = ax.bar(list(counts), values)
    label_vertical_bars(ax, bars, values)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _count_plot(frame: pd.DataFrame, column: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if frame.empty or column not in frame:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        counts = frame[column].astype(str).value_counts()
        bars = ax.bar(counts.index.str.slice(0, 28), counts.values)
        label_vertical_bars(ax, bars, counts.values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_report(summary: Mapping[str, object], model_summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            f"# task evaluation Optimized Chronaris Re-evaluation - {summary['run_id']}",
            "",
            f"- status: `{summary['status']}`",
            f"- runtime_device: `{summary['runtime_device']}`",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- boundary: {summary['protocol_boundary']}",
            "",
            "## Selection Summary",
            f"- private rows: `{model_summary['private_rows']}`",
            f"- public rows: `{model_summary['public_rows']}`",
            f"- selection_status: `{model_summary['selection_status']}`",
            "",
            "## Outputs",
            f"- private comparison: `{summary['optimized_private_comparison_csv']}`",
            f"- public comparison: `{summary['optimized_public_comparison_csv']}`",
            f"- delta vs P30: `{summary['optimized_delta_vs_p30_csv']}`",
            f"- delta vs P31: `{summary['optimized_delta_vs_p31_csv']}`",
            f"- cross evidence: `{summary['optimized_cross_evidence_matrix_csv']}`",
        ]
    )


def _resume_command(
    config: StageIOptimizedReevaluationConfig,
    p30_root: Path,
    p31_root: Path,
    p32_root: Path,
    p34_root: Path | None,
    p35_root: Path | None,
) -> str:
    parts = [
        str(REPO_ROOT / "scripts/task_eval/evidence/build_optimized_chronaris_reevaluation.py"),
        "--run-id",
        config.run_id,
        "--p30-root",
        str(p30_root),
        "--p31-root",
        str(p31_root),
        "--p32-root",
        str(p32_root),
    ]
    if p34_root:
        parts.extend(["--p34-root", str(p34_root)])
    if p35_root:
        parts.extend(["--p35-root", str(p35_root)])
    parts.extend(
        [
            "--artifact-root",
            str(_resolve_path(config.artifact_root)),
            "--report-root",
            str(_resolve_path(config.report_root)),
        ]
    )
    return "/home/wangminan/env/anaconda3/envs/chronaris/bin/python " + " ".join(parts)


def _optimized_status(p34_root: Path | None, p35_root: Path | None) -> str:
    required = (
        (p34_root, "task_head_optimization_summary.json"),
        (p35_root, "stream_role_fusion_summary.json"),
    )
    statuses = []
    for root, filename in required:
        if root is None:
            return "partial"
        summary_path = root / filename
        if not summary_path.exists():
            return "partial"
        try:
            status = str(json.loads(summary_path.read_text(encoding="utf-8")).get("status", "unknown"))
        except json.JSONDecodeError:
            return "partial"
        statuses.append(status)
    return "completed" if statuses and all(status == "completed" for status in statuses) else "partial"


def _protocol_boundary(status: str) -> str:
    if status == "completed":
        return (
            "P36 aggregates completed P34 task-head confirm and completed requested P35 private/public v3 confirm "
            "with fixed P30/P31/P32 references; it does not overwrite historical results."
        )
    return (
        "P36 aggregates available P34/P35 optimized artifacts with fixed P30/P31/P32 references; incomplete "
        "or missing v3 confirm rows remain partial and must not be written as completed."
    )


def _copy_or_empty(source: Path | None, output_path: Path) -> Path:
    if source and source.exists():
        _safe_read_csv(source).to_csv(output_path, index=False)
    else:
        pd.DataFrame().to_csv(output_path, index=False)
    return output_path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metric_keys(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "task_name": row["task_name"],
        "split_strategy": row["split_strategy"],
        "metric": row["metric"],
    }


def _latest_root(name: str) -> Path | None:
    root = REPO_ROOT / "docs/artifacts/assets" / name
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    return sorted(candidates)[-1] if candidates else None


def _aggregate_runtime_device(p34_root: Path | None, p35_root: Path | None) -> str:
    devices = []
    for root, filename in (
        (p34_root, "task_head_optimization_summary.json"),
        (p35_root, "stream_role_fusion_summary.json"),
    ):
        if not root:
            continue
        summary_path = root / filename
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        device = data.get("runtime_device")
        if isinstance(device, str) and device:
            devices.append(device)
    if "cuda" in devices:
        return "cuda"
    return devices[0] if devices else "unknown"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _resolve_path(path_like: str | Path | None) -> Path:
    if path_like is None:
        raise ValueError("path must not be None")
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value)!r}")
