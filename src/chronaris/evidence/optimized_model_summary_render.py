"""Rendering helpers for task evaluation optimized model summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.modeling.common.plot_labels import (
    label_vertical_bars,
)


def render_figures(
    run_root: Path,
    model_table: pd.DataFrame,
    key_metric_table: pd.DataFrame,
    gate_table: pd.DataFrame,
    gpu_table: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "fig_model_summary_status": str(run_root / "fig_model_summary_status.png"),
        "fig_model_summary_metric_delta": str(run_root / "fig_model_summary_metric_delta.png"),
        "fig_model_summary_gate_profile": str(run_root / "fig_model_summary_gate_profile.png"),
        "fig_model_summary_gpu_runtime": str(run_root / "fig_model_summary_gpu_runtime.png"),
    }
    _status_plot(model_table, paths["fig_model_summary_status"])
    _delta_plot(key_metric_table, paths["fig_model_summary_metric_delta"])
    _gate_plot(gate_table, paths["fig_model_summary_gate_profile"])
    _gpu_plot(gpu_table, paths["fig_model_summary_gpu_runtime"])
    return paths


def render_report(
    summary: Mapping[str, object],
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    gpu: pd.DataFrame,
) -> str:
    p34_lines = _metric_lines(metrics[metrics["stage"] == "P34"] if "stage" in metrics else pd.DataFrame())
    gate_lines = _gate_lines(gates)
    gpu_lines = _gpu_lines(gpu)
    return "\n".join(
        [
            f"# task evaluation Optimized Model Summary - {summary['run_id']}",
            "",
            f"- status: `{summary['status']}`",
            f"- runtime_device: `{summary['runtime_device']}`",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- boundary: {summary['protocol_boundary']}",
            "",
            "## P34 Key Deltas",
            *p34_lines,
            "",
            "## P35 Stream-role Gates",
            *gate_lines,
            "",
            "## GPU Runtime",
            *gpu_lines,
            "",
            "## Outputs",
            f"- summary: `{summary['optimized_model_summary_csv']}`",
            f"- key metrics: `{summary['key_metric_summary_csv']}`",
            f"- gates: `{summary['stream_role_gate_summary_csv']}`",
            f"- gpu: `{summary['gpu_runtime_summary_csv']}`",
            f"- claim boundary: `{summary['claim_boundary_summary_csv']}`",
            f"- resume commands: `{summary['resume_commands_txt']}`",
        ]
    )


def _status_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = frame["status"].astype(str).value_counts() if "status" in frame else pd.Series(dtype=int)
    if counts.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        bars = ax.bar(counts.index, counts.values, color=["#3a6ea5", "#d17a22", "#6a994e"][: len(counts)])
        label_vertical_bars(ax, bars, counts.values)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _delta_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    data = frame[(frame["stage"] == "P34") & frame["delta_positive_is_better"].notna()] if "stage" in frame else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "no P34 deltas", ha="center", va="center")
    else:
        labels = (
            data["dataset_or_task"]
            .astype(str)
            .str.replace("T1_", "T1\n", regex=False)
            .str.replace("T2_", "T2\n", regex=False)
            .str.replace("T3_", "T3\n", regex=False)
            + "\n"
            + data["metric"].astype(str)
        )
        colors = ["#2a9d8f" if value >= 0 else "#c44536" for value in data["delta_positive_is_better"].astype(float)]
        values = data["delta_positive_is_better"].astype(float)
        bars = ax.bar(labels, values, color=colors)
        label_vertical_bars(ax, bars, values)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("positive = optimized better")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _gate_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    gate_cols = ["lag_gate_mean", "context_gate_mean", "vehicle_gate_mean", "causal_gate_mean"]
    if frame.empty or not set(gate_cols).issubset(frame.columns):
        ax.text(0.5, 0.5, "no gate rows", ha="center", va="center")
    else:
        labels = frame["dataset_id"].astype(str).str.slice(0, 22)
        x = np.arange(len(frame))
        width = 0.18
        for offset, col in enumerate(gate_cols):
            values = frame[col].astype(float)
            bars = ax.bar(x + (offset - 1.5) * width, values, width=width, label=col.replace("_gate_mean", ""))
            label_vertical_bars(ax, bars, values)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _gpu_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if frame.empty or "max_gpu_memory_gb" not in frame:
        ax.text(0.5, 0.5, "no gpu rows", ha="center", va="center")
    else:
        values = pd.to_numeric(frame["max_gpu_memory_gb"], errors="coerce").fillna(0.0)
        bars = ax.bar(frame["stage"].astype(str), values, color="#4c78a8")
        label_vertical_bars(ax, bars, values)
        ax.set_ylabel("max GPU memory GB")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _metric_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["- no P34 metric rows"]
    lines = []
    for row in frame.to_dict(orient="records"):
        lines.append(
            "- `{}` `{}`: P30=`{}` P34=`{}` delta=`{}` status=`{}`".format(
                row.get("dataset_or_task"),
                row.get("metric"),
                _fmt(row.get("reference_value")),
                _fmt(row.get("optimized_value")),
                _fmt(row.get("delta_positive_is_better")),
                row.get("status"),
            )
        )
    return lines


def _gate_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["- no gate rows"]
    lines = []
    for row in frame.to_dict(orient="records"):
        lines.append(
            "- `{}` role=`{}` route=`{}` lag=`{}` context=`{}` vehicle=`{}` causal=`{}`".format(
                row.get("dataset_id"),
                row.get("second_stream_role"),
                row.get("fusion_route"),
                _fmt(row.get("lag_gate_mean")),
                _fmt(row.get("context_gate_mean")),
                _fmt(row.get("vehicle_gate_mean")),
                _fmt(row.get("causal_gate_mean")),
            )
        )
    return lines


def _gpu_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["- no GPU rows"]
    lines = []
    for row in frame.to_dict(orient="records"):
        lines.append(
            "- `{}` device=`{}` gpu=`{}` cache=`{}` batch=`{}` amp=`{}` compile=`{}` max_mem_gb=`{}` samples_per_sec=`{}` util_pct=`{}`".format(
                row.get("stage"),
                row.get("runtime_device"),
                row.get("gpu_name"),
                row.get("tensor_cache_mode"),
                _fmt(_first_present(row.get("best_batch_size"), row.get("batch_size"))),
                row.get("amp_mode"),
                row.get("torch_compile_mode"),
                _fmt(row.get("max_gpu_memory_gb")),
                _fmt(row.get("samples_per_sec")),
                _fmt(row.get("gpu_util_pct")),
            )
        )
    return lines


def _first_present(*values: object) -> object:
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            if value not in (None, ""):
                return value
            continue
        if not np.isnan(numeric):
            return value
    return None


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "NA"
    return f"{numeric:.6g}"
