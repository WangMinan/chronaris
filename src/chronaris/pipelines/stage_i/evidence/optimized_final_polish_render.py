"""Rendering helpers for optimized final-polish artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from chronaris.pipelines.stage_i.common.plot_labels import label_vertical_bars


_MODEL_DISPLAY_NAMES = {
    "p37_t3_info_nce_temp0p05_hardw2": "检索任务：InfoNCE低温候选",
    "p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10": "分类任务：焦点损失+门控候选",
    "p37_t1_focal_gamma1_gate0p65_ls0p05": "分类任务：轻门控候选",
    "p37_public_force_adaptive_context_gate": "公开数据路线：自适应上下文门控",
    "p37_public_context_adapter_only_cap2x_do0p2": "公开数据路线：上下文适配增强",
}

_SOURCE_DISPLAY_NAMES = {
    "t3_screen_root": "检索任务筛选",
    "t1_screen_root": "分类任务筛选",
    "private_confirm_root": "鼎新真实数据确认",
    "public_confirm_root": "公开数据确认",
}


def render_figures(run_root: Path, tables: Mapping[str, Path]) -> dict[str, str]:
    _configure_plot_font()
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
    t3_delta_display = _with_display_columns(t3_delta)
    t1_delta_display = _with_display_columns(t1_delta)
    t1_vs_reference_display = _with_display_columns(t1_vs_p34)
    public_display = _with_display_columns(public)
    p35_delta_display = _with_display_columns(p35_delta)
    p30_p31_delta_display = _with_display_columns(p30_p31_delta)
    return "\n".join(
        [
            f"# Stage I Optimized Final Polish - {summary['run_id']}",
            "",
            "## Executive Summary",
            f"- status: `{summary['status']}`",
            f"- runtime_device: `{summary['runtime_device']}`",
            "- boundary: earlier comparison and ablation runs are fixed references; public rows use a context-derived second input stream.",
            "- decision: accept the classification-task calibration and public-data route calibration where they improve confirmed references; keep the confirmed retrieval-task result where the new candidate does not improve it.",
            "",
            "## Fixed references and protocol boundary",
            f"- task-head confirmed reference: `{summary['p34_root']}`",
            f"- stream-role confirmed reference: `{summary['p35_root']}`",
            "",
            "## Retrieval Task Final Polish",
            f"- metrics: `{summary['t3_final_polish_metrics_csv']}`",
            f"- delta vs confirmed retrieval reference: `{summary['p37_delta_vs_p34_csv']}`",
            _markdown_table(
                t3_delta_display,
                ["metric_display", "confirmed_reference_value", "candidate_model", "candidate_value", "delta_positive_is_better"],
                max_rows=8,
            ),
            "",
            "## Classification Task Calibration",
            f"- metrics: `{summary['t1_calibration_metrics_csv']}`",
            _markdown_table(
                t1_vs_reference_display,
                ["split_strategy", "metric_display", "confirmed_reference_value", "candidate_model", "candidate_value", "delta_positive_is_better"],
                max_rows=8,
            ),
            _markdown_table(
                t1_delta_display,
                ["task", "split_strategy", "model", "metric_display", "value_mean"],
                max_rows=8,
            ),
            "",
            "## Public route calibration",
            f"- metrics: `{summary['public_route_calibration_metrics_csv']}`",
            f"- gate calibration: `{summary['route_gate_calibration_csv']}`",
            _markdown_table(
                public_display,
                ["dataset_id", "variant", "primary_metric", "selection_score", "combined_macro_f1", "mean_rmse"],
                max_rows=8,
            ),
            _markdown_table(
                p35_delta_display,
                ["dataset_id", "metric_display", "candidate_variant", "candidate_value", "reference_value", "delta_positive_is_better"],
                max_rows=8,
            ),
            _markdown_table(
                p30_p31_delta_display,
                ["scope_display", "dataset_id", "metric_display", "reference_display", "candidate_value", "reference_value", "delta_positive_is_better"],
                max_rows=12,
            ),
            "",
            "## Accepted / rejected candidates",
            *_candidate_summary_lines("accepted", accepted.get("accepted", []) if isinstance(accepted, Mapping) else []),
            *_candidate_summary_lines("rejected", rejected.get("rejected", []) if isinstance(rejected, Mapping) else []),
            "",
            "## GPU runtime summary",
            f"- GPU summary: `{summary['gpu_perf_summary_json']}`",
            "",
            "## Figure index",
            *[f"- {_display_figure_key(key)}: `{value}`" for key, value in dict(summary["figure_paths"]).items()],
            "",
            "## Reproducibility manifest",
            f"- manifest: `{summary['evidence_manifest_path']}`",
            f"- resume: `{summary['resume_command_txt']}`",
            "",
            "## Midterm / thesis-ready wording",
            "The final polish focuses on the two remaining diagnostic gaps after the confirmed references: retrieval ranking and public-data routing with a context-derived second input stream. A new candidate is used only where it improves the confirmed metric under the same split and leakage boundary; otherwise the existing confirmed result remains the thesis-facing result.",
        ]
    )


def _metric_bar(frame: pd.DataFrame, metric: str, path: str) -> None:
    subset = frame[frame.get("metric", "").astype(str).eq(metric)] if not frame.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 4))
    if subset.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        values = subset["value_mean"].astype(float)
        labels = subset["model_name"].map(_display_model_label).astype(str) + "\n" + subset["split_strategy"].astype(str)
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
        bars = ax.bar(frame["variant_id"].map(_display_model_label).astype(str), values)
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
        bars = ax.bar(grouped["model_name"].map(_display_model_label).astype(str), values)
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
        ax.set_title(_display_model_label(best_file.stem.replace("_confusion", "")), fontsize=8)
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
    labels = [_display_source_label(source.get("source", f"source_{idx + 1}")) for idx, source in enumerate(sources)]
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
            ax.plot(grouped["epoch"], grouped["train_loss"], marker="o", label=_display_model_label(model_name))
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


def _display_model_label(value: object) -> str:
    text = str(value)
    return _MODEL_DISPLAY_NAMES.get(text, text.replace("_", " "))


def _display_source_label(value: object) -> str:
    text = str(value)
    return _SOURCE_DISPLAY_NAMES.get(text, text.replace("_", " "))


def _display_task_label(value: object) -> str:
    text = str(value)
    if text.startswith("T1"):
        return "分类任务"
    if text.startswith("T2"):
        return "回归任务"
    if text.startswith("T3"):
        return "检索任务"
    return text.replace("_", " ")


def _display_metric_label(value: object) -> str:
    text = str(value)
    return {
        "macro_f1": "macro-F1",
        "balanced_accuracy": "balanced accuracy",
        "rmse": "RMSE",
        "mae": "MAE",
        "top1": "Top-1",
        "top3": "Top-3",
        "top5": "Top-5",
        "mrr": "MRR",
    }.get(text, text)


def _with_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "task_name" in display:
        display["task"] = display["task_name"].map(_display_task_label)
    if "metric" in display:
        display["metric_display"] = display["metric"].map(_display_metric_label)
    if "model_name" in display:
        display["model"] = display["model_name"].map(_display_model_label)
    if "variant_id" in display:
        display["variant"] = display["variant_id"].map(_display_model_label)
    if "p37_model" in display:
        display["candidate_model"] = display["p37_model"].map(_display_model_label)
    if "p37_variant" in display:
        display["candidate_variant"] = display["p37_variant"].map(_display_model_label)
    if "p34_value" in display:
        display["confirmed_reference_value"] = display["p34_value"]
    if "p37_value" in display:
        display["candidate_value"] = display["p37_value"]
    if "scope" in display:
        display["scope_display"] = display["scope"].map(_display_scope_label)
    if "reference" in display:
        display["reference_display"] = display["reference"].map(_display_reference_label)
    return display


def _display_figure_key(value: object) -> str:
    text = str(value).replace("fig_p37_", "")
    replacements = {
        "t1": "classification_task",
        "t3": "retrieval_task",
        "public_route": "public_data_route",
        "gpu": "gpu",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("vs_p34", "vs confirmed retrieval reference")
    text = text.replace("vs_p35", "vs confirmed stream-role reference")
    text = text.replace("vs_p30_p31", "vs confirmed comparison references")
    return text.replace("_", " ")


def _display_scope_label(value: object) -> str:
    text = str(value)
    return {
        "private": "鼎新真实数据",
        "public": "公开数据",
        "T1": "分类任务",
        "T2": "回归任务",
        "T3": "检索任务",
        "public_route": "公开数据路线",
    }.get(text, text.replace("_", " "))


def _display_reference_label(value: object) -> str:
    text = str(value)
    replacements = {
        "P30 chronaris_full": "已确认 Chronaris 鼎新基线",
        "P31 no_lag_window": "已确认公开数据无滞后窗口基线",
        "P31 context_only": "已确认公开数据上下文输入基线",
    }
    return replacements.get(text, text.replace("_", " "))


def _candidate_summary_lines(label: str, rows: object) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return [f"- {label}: none"]
    lines = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scope = _display_scope_label(row.get("scope", "candidate"))
        reason = _display_decision_text(row.get("reason", ""))
        lines.append(f"- {label}: {scope}；{reason}")
    return lines or [f"- {label}: none"]


def _display_decision_text(value: object) -> str:
    text = str(value)
    replacements = {
        "P37": "current candidate",
        "P35": "confirmed stream-role reference",
        "P34": "confirmed retrieval reference",
        "T1": "classification task",
        "T3": "retrieval task",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _configure_plot_font() -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return
    for family in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.sans-serif"] = [family, *plt.rcParams.get("font.sans-serif", [])]
        plt.rcParams["axes.unicode_minus"] = False
        return


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
