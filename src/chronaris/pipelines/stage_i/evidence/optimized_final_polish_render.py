"""Rendering helpers for P37 optimized final-polish artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from chronaris.pipelines.stage_i.common.plot_labels import label_vertical_bars


def render_figures(run_root: Path, tables: Mapping[str, Path]) -> dict[str, str]:
    paths = {
        "fig_p37_t3_retrieval_leaderboard": str(run_root / "fig_p37_t3_retrieval_leaderboard.png"),
        "fig_p37_t3_delta_vs_p34": str(run_root / "fig_p37_t3_delta_vs_p34.png"),
        "fig_p37_t1_macro_f1_leaderboard": str(run_root / "fig_p37_t1_macro_f1_leaderboard.png"),
        "fig_p37_t1_gate_sweep": str(run_root / "fig_p37_t1_gate_sweep.png"),
        "fig_p37_t1_confusion_matrix_best": str(run_root / "fig_p37_t1_confusion_matrix_best.png"),
        "fig_p37_public_route_nasa_macro_f1": str(run_root / "fig_p37_public_route_nasa_macro_f1.png"),
        "fig_p37_public_route_uab_rmse": str(run_root / "fig_p37_public_route_uab_rmse.png"),
        "fig_p37_route_gate_profile": str(run_root / "fig_p37_route_gate_profile.png"),
        "fig_p37_public_route_delta_heatmap": str(run_root / "fig_p37_public_route_delta_heatmap.png"),
        "fig_p37_overall_acceptance_summary": str(run_root / "fig_p37_overall_acceptance_summary.png"),
        "fig_p37_gpu_runtime": str(run_root / "fig_p37_gpu_runtime.png"),
        "fig_p37_t3_hard_negative_margin": str(run_root / "fig_p37_t3_hard_negative_margin.png"),
        "fig_p37_t3_similarity_distribution": str(run_root / "fig_p37_t3_similarity_distribution.png"),
    }
    _metric_bar(_safe_read(tables["t3_final_polish_metrics_csv"]), "mrr", paths["fig_p37_t3_retrieval_leaderboard"])
    _delta_bar(_safe_read(run_root / "t3_delta_vs_p34.csv"), paths["fig_p37_t3_delta_vs_p34"])
    _metric_bar(_safe_read(tables["t1_calibration_metrics_csv"]), "macro_f1", paths["fig_p37_t1_macro_f1_leaderboard"])
    _gate_bar(_safe_read(tables["route_gate_calibration_csv"]), paths["fig_p37_route_gate_profile"])
    public = _safe_read(tables["public_route_calibration_metrics_csv"])
    _public_bar(public[public.get("dataset_id", "") == "nasa_csm"] if not public.empty else public, paths["fig_p37_public_route_nasa_macro_f1"])
    _public_bar(public[public.get("dataset_id", "") == "uab_workload_dataset"] if not public.empty else public, paths["fig_p37_public_route_uab_rmse"])
    _delta_bar(_safe_read(tables["p37_delta_vs_p35_csv"]), paths["fig_p37_public_route_delta_heatmap"])
    _t1_gate_sweep(_safe_read(run_root / "t1_gate_sweep.csv"), paths["fig_p37_t1_gate_sweep"])
    _confusion_matrix(run_root / "t1_confusion_matrices", paths["fig_p37_t1_confusion_matrix_best"])
    _acceptance_summary(run_root, paths["fig_p37_overall_acceptance_summary"])
    _gpu_runtime(Path(tables["gpu_perf_summary_json"]), paths["fig_p37_gpu_runtime"])
    _diagnostic_loss(_safe_read(run_root / "t3_hard_negative_diagnostics.csv"), paths["fig_p37_t3_hard_negative_margin"])
    _similarity_summary(_safe_read(run_root / "t3_similarity_distribution.csv"), paths["fig_p37_t3_similarity_distribution"])
    return paths


def render_report(
    summary: Mapping[str, object],
    accepted: Mapping[str, object],
    rejected: Mapping[str, object],
) -> str:
    t3_delta = _safe_read(Path(str(summary["p37_delta_vs_p34_csv"])))
    t1_delta = _safe_read(Path(str(summary["t1_calibration_metrics_csv"])))
    t1_vs_p34 = _safe_read(Path(str(summary["artifact_root"])) / "t1_delta_vs_p34.csv")
    public = _safe_read(Path(str(summary["public_route_calibration_metrics_csv"])))
    p35_delta = _safe_read(Path(str(summary["p37_delta_vs_p35_csv"])))
    p30_p31_delta = _safe_read(Path(str(summary["p37_delta_vs_p30_p31_csv"])))
    return "\n".join(
        [
            f"# Stage I Optimized Final Polish - {summary['run_id']}",
            "",
            "## Executive Summary",
            f"- status: `{summary['status']}`",
            f"- runtime_device: `{summary['runtime_device']}`",
            "- boundary: P30/P31/P34/P35/P36 are fixed references; public rows are context-proxy evidence.",
            "- decision: accept P37 T1 calibration and public route calibration where they improve confirmed references; keep P34 T3 retrieval as the confirmed result.",
            "",
            "## Fixed references and protocol boundary",
            f"- P34 reference: `{summary['p34_root']}`",
            f"- P35 reference: `{summary['p35_root']}`",
            "",
            "## T3 final polish",
            f"- metrics: `{summary['t3_final_polish_metrics_csv']}`",
            f"- delta vs P34: `{summary['p37_delta_vs_p34_csv']}`",
            _markdown_table(
                t3_delta,
                ["metric", "p34_value", "p37_model", "p37_value", "delta_positive_is_better"],
                max_rows=8,
            ),
            "",
            "## T1 calibration",
            f"- metrics: `{summary['t1_calibration_metrics_csv']}`",
            _markdown_table(
                t1_vs_p34,
                ["split_strategy", "metric", "p34_value", "p37_model", "p37_value", "delta_positive_is_better"],
                max_rows=8,
            ),
            _markdown_table(
                t1_delta,
                ["task_name", "split_strategy", "model_name", "metric", "value_mean"],
                max_rows=8,
            ),
            "",
            "## Public route calibration",
            f"- metrics: `{summary['public_route_calibration_metrics_csv']}`",
            f"- gate calibration: `{summary['route_gate_calibration_csv']}`",
            _markdown_table(
                public,
                ["dataset_id", "variant_id", "primary_metric", "selection_score", "combined_macro_f1", "mean_rmse"],
                max_rows=8,
            ),
            _markdown_table(
                p35_delta,
                ["dataset_id", "metric", "p37_variant", "p37_value", "reference_value", "delta_positive_is_better"],
                max_rows=8,
            ),
            _markdown_table(
                p30_p31_delta,
                ["scope", "dataset_id", "metric", "reference", "p37_value", "reference_value", "delta_positive_is_better"],
                max_rows=12,
            ),
            "",
            "## Accepted / rejected candidates",
            f"- accepted: `{accepted}`",
            f"- rejected: `{rejected}`",
            "",
            "## GPU runtime summary",
            f"- GPU summary: `{summary['gpu_perf_summary_json']}`",
            "",
            "## Figure index",
            *[f"- {key}: `{value}`" for key, value in dict(summary["figure_paths"]).items()],
            "",
            "## Reproducibility manifest",
            f"- manifest: `{summary['evidence_manifest_path']}`",
            f"- resume: `{summary['resume_command_txt']}`",
            "",
            "## Midterm / thesis-ready wording",
            "P37 final polish focuses on the two remaining diagnostic gaps after P34/P35: retrieval ranking and public context-proxy routing. The accepted P37 candidate is used only where it improves the P34/P35 confirmed metric under the same split and leakage boundary; otherwise P34/P35 remain the confirmed optimized result.",
        ]
    )


def _metric_bar(frame: pd.DataFrame, metric: str, path: str) -> None:
    subset = frame[frame.get("metric", "").astype(str).eq(metric)] if not frame.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 4))
    if subset.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        values = subset["value_mean"].astype(float)
        labels = subset["model_name"].astype(str) + "\n" + subset["split_strategy"].astype(str)
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _public_bar(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if frame.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        values = frame["selection_score"].astype(float)
        bars = ax.bar(frame["variant_id"].astype(str), values)
        label_vertical_bars(ax, bars, values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _delta_bar(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if frame.empty or "delta_positive_is_better" not in frame:
        ax.text(0.5, 0.5, "no delta rows", ha="center", va="center")
    else:
        values = frame["delta_positive_is_better"].astype(float)
        labels = frame.get("metric", pd.Series(range(len(frame)))).astype(str)
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _gate_bar(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    columns = ["lag_gate_mean", "context_gate_mean", "vehicle_gate_mean", "causal_gate_mean"]
    if frame.empty:
        ax.text(0.5, 0.5, "no gate rows", ha="center", va="center")
    else:
        values = frame[columns].astype(float).mean()
        bars = ax.bar(columns, values)
        label_vertical_bars(ax, bars, values)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _t1_gate_sweep(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if frame.empty or "p37_gate_mean" not in frame:
        ax.text(0.5, 0.5, "no gate rows", ha="center", va="center")
    else:
        grouped = frame.groupby("model_name", as_index=False)["p37_gate_mean"].mean()
        values = grouped["p37_gate_mean"].astype(float)
        bars = ax.bar(grouped["model_name"].astype(str), values)
        label_vertical_bars(ax, bars, values)
        ax.set_ylabel("mean gate")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _confusion_matrix(matrix_dir: Path, path: str) -> None:
    files = sorted(matrix_dir.glob("*_confusion.csv"))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if not files:
        ax.text(0.5, 0.5, "no confusion rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        scored = []
        for file in files:
            candidate = pd.read_csv(file, index_col=0)
            diagonal = sum(candidate.iloc[idx, idx] for idx in range(min(candidate.shape)))
            scored.append((float(diagonal), file, candidate))
        _, best_file, matrix = max(scored, key=lambda item: item[0])
        image = ax.imshow(matrix.astype(float), cmap="Blues")
        ax.set_title(best_file.stem.replace("_confusion", ""), fontsize=8)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        for row_idx, (_, row) in enumerate(matrix.iterrows()):
            for col_idx, value in enumerate(row):
                ax.text(col_idx, row_idx, int(value), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _acceptance_summary(run_root: Path, path: str) -> None:
    accepted = _safe_json(run_root / "accepted_candidate_summary.json").get("accepted", [])
    rejected = _safe_json(run_root / "rejected_candidate_summary.json").get("rejected", [])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    values = [len(accepted), len(rejected)]
    bars = ax.bar(["accepted", "rejected"], values, color=["#2ca25f", "#de2d26"])
    label_vertical_bars(ax, bars, values)
    ax.set_ylim(0, max(values + [1]) + 0.5)
    ax.set_ylabel("candidate scopes")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _gpu_runtime(path: Path, output_path: str) -> None:
    sources = _safe_json(path).get("sources", [])
    labels = [str(source.get("source", f"source_{idx + 1}")) for idx, source in enumerate(sources)]
    values = [float(source.get("best_batch_size", 0) or 0) for source in sources]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if not labels:
        ax.text(0.5, 0.5, "no gpu rows", ha="center", va="center")
    else:
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.set_ylabel("best batch size")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _diagnostic_loss(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if frame.empty or not {"epoch", "train_loss", "model_name"}.issubset(frame.columns):
        ax.text(0.5, 0.5, "no diagnostic loss rows", ha="center", va="center")
    else:
        for model_name, subset in frame.groupby("model_name"):
            grouped = subset.groupby("epoch", as_index=False)["train_loss"].mean()
            ax.plot(grouped["epoch"], grouped["train_loss"], marker="o", label=str(model_name))
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _similarity_summary(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if frame.empty or "rank" not in frame.columns:
        ax.text(0.5, 0.5, "no rank summary", ha="center", va="center")
    else:
        subset = frame[frame.iloc[:, 0].astype(str).isin(["mean", "25%", "50%", "75%"])]
        values = subset["rank"].astype(float)
        bars = ax.bar(subset.iloc[:, 0].astype(str), values)
        label_vertical_bars(ax, bars, values)
        ax.set_ylabel("retrieval rank")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _placeholder(path: str, text: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _safe_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int) -> str:
    if frame.empty:
        return "_No rows._"
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return "_No matching columns._"
    subset = frame.loc[:, existing].head(max_rows).copy()
    for column in subset.columns:
        if pd.api.types.is_numeric_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: f"{value:.6g}" if pd.notna(value) else "")
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in subset.to_numpy()]
    header = "| " + " | ".join(existing) + " |"
    separator = "| " + " | ".join("---" for _ in existing) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])
