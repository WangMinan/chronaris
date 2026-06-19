"""Plotting helpers for Stage I thesis-facing materials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten, wrap
from typing import Mapping

import pandas as pd
import numpy as np

from chronaris.pipelines.stage_i.evidence.thesis_materials_data import (
    build_evidence_layer_rows,
    build_llm_comparison_rows,
    build_model_backbone_ablation_rows,
    build_private_component_rows,
    build_public_transfer_rows,
    build_rigid_body_rotation_rows,
    build_runtime_payload_schema_rows,
    build_runtime_semantic_case_rows,
    build_semantic_event_rows,
    build_task_adapter_ablation_rows,
    build_weak_label_rows,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

DEFAULT_CJK_FONT_CANDIDATES = (
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
)

LAYER_COLORS = {
    "thesis_weak_label": "#3f6f5a",
    "private_proxy": "#9a5b43",
    "public_adapter_calibration": "#385f86",
    "runtime_schema": "#7861a6",
    "semantic_support": "#6f7d3d",
    "rigid_body": "#9a7a35",
    "llm_preprocessing": "#8a5276",
}

INK = "#1f2937"
MUTED = "#64748b"
GRID = "#e2e8f0"
PALE = "#f8fafc"
GOLD = "#d7a84f"
BLUE = "#436a92"
GREEN = "#4f7c5c"
ORANGE = "#c97855"
PURPLE = "#7a5fa2"


@dataclass(frozen=True, slots=True)
class PlotFontSelection:
    family: str | None
    ascii_only: bool
    note: str


def detect_plot_font() -> PlotFontSelection:
    try:
        from matplotlib import font_manager
    except Exception:
        return PlotFontSelection(None, True, "matplotlib unavailable; used ASCII-safe labels")
    available_names = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in DEFAULT_CJK_FONT_CANDIDATES:
        if candidate in available_names:
            return PlotFontSelection(candidate, False, f"using CJK font {candidate}")
    return PlotFontSelection(None, True, "CJK font missing; used ASCII-safe labels")


def write_thesis_figures(
    *,
    run_root: Path,
    font: PlotFontSelection,
    sources: Mapping[str, Mapping[str, object]],
    table_entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    table_paths = {Path(str(entry["path"])).stem: str(entry["path"]) for entry in table_entries}
    return [
        _plot_evidence_layer_overview(run_root / "evidence_layer_overview.png", sources, font, table_paths),
        _plot_runtime_payload_schema(run_root / "runtime_payload_schema.png", sources, font, table_paths),
        _plot_runtime_semantic_case(run_root / "runtime_semantic_case.png", sources, font, table_paths),
        _plot_rigid_body_rotation(run_root / "rigid_body_rotation_audit.png", sources, font, table_paths),
        _plot_weak_label_sweep(run_root / "weak_label_sweep_ablation.png", sources, font, table_paths),
        _plot_private_component(run_root / "chronaris_opt_component_ablation.png", sources, font, table_paths),
        _plot_model_backbone_ablation(run_root / "model_backbone_ablation.png", sources, font, table_paths),
        _plot_task_adapter_ablation(run_root / "task_adapter_ablation.png", sources, font, table_paths),
        _plot_public_transfer(run_root / "public_transfer_boundary.png", sources, font, table_paths),
        _plot_semantic_event_fusion(run_root / "semantic_event_fusion_overview.png", sources, font, table_paths),
        _plot_llm_comparison(run_root / "llm_comparison_a0_a4.png", sources, font, table_paths),
    ]


def _plot_evidence_layer_overview(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_evidence_layer_rows(sources)).sort_values("display_order")
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(13.8, 8.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, len(frame) + 1.1)
    headers = [
        _label(font, "证据层级", "Evidence layer"),
        _label(font, "主计数", "Primary count"),
        _label(font, "辅助计数", "Secondary count"),
        _label(font, "当前成果与后续作用", "Current result and next role"),
    ]
    x_positions = [0.2, 3.1, 5.2, 7.28]
    widths = [2.65, 1.78, 1.78, 4.45]
    for x, width, header in zip(x_positions, widths, headers, strict=True):
        ax.text(x + 0.05, len(frame) + 0.55, header, fontsize=11, weight="bold", color=INK, va="center")
        ax.plot([x, x + width], [len(frame) + 0.25, len(frame) + 0.25], color=GRID, lw=1.2)
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        y = len(frame) - row_index - 0.5
        color = LAYER_COLORS.get(str(row["evidence_layer"]), "#7a8793")
        ax.add_patch(
            patches.Rectangle((0.15, y - 0.36), 11.45, 0.72, facecolor=PALE, edgecolor=GRID, lw=0.8)
        )
        ax.add_patch(
            patches.Rectangle((0.15, y - 0.36), 0.12, 0.72, facecolor=color, edgecolor=color, lw=0)
        )
        ax.text(
            0.38,
            y,
            _label(font, str(row["layer_title_cn"]), str(row["layer_title"])),
            fontsize=10.2,
            weight="bold",
            color=INK,
            va="center",
        )
        _metric_cell(ax, 3.1, y, _metric_name_cn(row["primary_metric_name"]), row["primary_metric_value"], color)
        _metric_cell(ax, 5.2, y, _metric_name_cn(row["secondary_metric_name"]), row["secondary_metric_value"], color)
        ax.text(7.34, y + 0.12, _short(row["key_status"], 62), fontsize=8.8, color=INK, va="center")
        ax.text(
            7.34,
            y - 0.17,
            _short(_label(font, str(row["boundary_cn"]), str(row["boundary_cn"])), 76),
            fontsize=8.0,
            color=MUTED,
            va="center",
        )
    ax.set_title(_label(font, "当前阶段研究成果与验证材料矩阵", "Current Research Results and Validation Materials Matrix"), fontsize=16, weight="bold", color=INK, pad=16)
    ax.text(
        0.15,
        0.02,
        _label(font, "所有数值来自现有结构化产物；不同证据层不合并为同一结论。", "All values are parsed from existing artifacts; evidence layers remain separate."),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="evidence_layer_overview",
        path=path,
        source_paths=_source_paths(sources, "evidence_manifest", "live_sweep", "private_component", "public_calibration", "runtime_schema_contract", "support", "rotation_audit", "llm_comparison"),
        evidence_layer="cross_layer_index",
        table_path=table_paths["evidence_layer_overview"],
        metric_definition="Evidence-layer matrix with source-derived counts/statuses; replaces artifact-present bars.",
        recommended_placement="appendix_or_supporting",
        replaces_problem="evidence_layer_overview no longer uses all-one artifact-present bars.",
    )


def _plot_runtime_payload_schema(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_runtime_payload_schema_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(13.4, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    boxes = [(0.35, 0.82, 3.95, 3.15), (7.7, 0.82, 3.95, 3.15)]
    colors = [ORANGE, GREEN]
    for idx, row in enumerate(frame.to_dict(orient="records")):
        x, y, w, h = boxes[idx]
        color = colors[idx]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.08",
                facecolor="#fffaf3" if idx == 0 else "#f3faf5",
                edgecolor=color,
                lw=1.6,
            )
        )
        ax.add_patch(
            patches.Rectangle((x, y + h - 0.62), w, 0.62, facecolor="#fff5e7" if idx == 0 else "#e9f6ed", edgecolor="none")
        )
        ax.text(
            x + 0.2,
            y + h - 0.38,
            _label(font, row["payload_name_cn"], row["payload_name"]),
            fontsize=12.3,
            weight="bold",
            color=INK,
            va="center",
        )
        if idx == 0:
            metric_rows = [
                ("生理字段", row["physiology_feature_count"]),
                ("已对齐飞机状态字段", row["vehicle_feature_count"]),
                ("契约状态", _schema_status_cn(row["schema_status"])),
                ("显式缺失字段", row["missing_vehicle_feature_count"]),
            ]
        else:
            metric_rows = [
                ("生理字段", row["physiology_feature_count"]),
                ("训练字段契约", row["vehicle_feature_count"]),
                ("契约状态", _schema_status_cn(row["schema_status"])),
                ("显式缺失字段", row["missing_vehicle_feature_count"]),
            ]
        for offset, (name, value) in enumerate(metric_rows):
            yy = y + h - 1.12 - offset * 0.46
            ax.plot([x + 0.18, x + w - 0.18], [yy - 0.22, yy - 0.22], color=GRID, lw=0.8)
            ax.text(x + 0.24, yy, name, fontsize=9.3, color=MUTED, va="center")
            ax.text(x + w - 0.24, yy, str(value), fontsize=12.0 if offset == 1 else 10.6, color=color if name == "契约状态" else INK, weight="bold", ha="right", va="center")
        ax.text(
            x + 0.24,
            y + 0.24,
            _wrap_text(_label(font, row["contract_note_cn"], row["contract_note"]), 30),
            fontsize=8.2,
            color=MUTED,
            va="bottom",
        )
    _draw_arrow(ax, patches, (4.52, 2.42), (7.48, 2.42), lw=2.3, mutation_scale=18)
    ax.text(
        6.0,
        2.86,
        _label(font, "字段规范映射与显式缺失标记", "schema mapping and explicit missing masks"),
        ha="center",
        fontsize=10.8,
        weight="bold",
        color=INK,
    )
    ax.text(
        6.0,
        1.72,
        _label(font, "字段排列、契约映射、缺失掩码和模型输入检查", "field ordering, contract mapping, missing masks, model input validation"),
        ha="center",
        fontsize=8.8,
        color=MUTED,
    )
    ax.set_title(_label(font, "运行输入字段规范与模型契约检查", "Runtime Input Field Schema and Model Contract Check"), fontsize=15.5, weight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_payload_schema",
        path=path,
        source_paths=_source_paths(sources, "runtime_service", "runtime_schema_contract"),
        evidence_layer="runtime_schema",
        table_path=table_paths["runtime_payload_schema"],
        metric_definition="Native replay payload records field ordering and explicit missing masks; canonical payload validates the 1930-dimension training-field contract.",
        replaces_problem="runtime_payload_schema is rendered as a contract comparison instead of a generic schema plot.",
    )


def _plot_runtime_semantic_case(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_runtime_semantic_case_rows(sources))
    if frame.empty:
        raise ValueError("runtime_semantic_case requires a non-empty runtime_case_table source")
    for column in ("window_order", "semantic_top_event_attribution", "risk_proxy_confidence", "workload_proxy_prediction", "event_replay_tag_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("window_order").reset_index(drop=True)
    case = frame.iloc[0].to_dict()
    query_names = frame["semantic_top_query_name"].fillna("unknown").astype(str)
    query_names_cn = query_names.map(_query_name_cn)
    point_colors = [_query_color(name) for name in query_names]
    plt, patches = _import_matplotlib(font)
    fig = plt.figure(figsize=(14.6, 7.6))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 2.6], hspace=0.48, wspace=0.36)

    ax_cards = fig.add_subplot(grid[0, :])
    ax_cards.set_axis_off()
    ax_cards.set_xlim(0, 12)
    ax_cards.set_ylim(0, 2)
    cards = [
        (_label(font, "代表窗口", "selected windows"), str(len(frame)), GREEN),
        (
            _label(font, "查询类型", "query types"),
            "、".join(dict.fromkeys(query_names_cn.tolist())),
            PURPLE,
        ),
        (
            _label(font, "选择规则", "selection rule"),
            "查询变化 / 归因峰值 / 输出变化",
            ORANGE,
        ),
        (
            _label(font, "数据视图", "data view"),
            "人机双流窗口",
            GOLD,
        ),
    ]
    for idx, (title, value, color) in enumerate(cards):
        x = 0.22 + idx * 2.88
        ax_cards.add_patch(
            patches.FancyBboxPatch(
                (x, 0.16),
                2.65,
                0.62,
                boxstyle="round,pad=0.05,rounding_size=0.06",
                facecolor="#fbfdff",
                edgecolor=color,
                lw=1.35,
            )
        )
        ax_cards.text(x + 0.16, 0.57, title, fontsize=8.2, color=MUTED, va="center")
        ax_cards.text(x + 0.16, 0.32, _short(value, 26), fontsize=10.2, weight="bold", color=INK, va="center")

    ax_attr = fig.add_subplot(grid[1, :2])
    x_values = list(range(len(frame)))
    y_values = frame["semantic_top_event_attribution"].fillna(0.0).astype(float).tolist()
    for x_value, y_value, color in zip(x_values, y_values, point_colors, strict=True):
        ax_attr.vlines(x_value, 0, y_value, color=color, alpha=0.35, lw=2.2)
    ax_attr.scatter(x_values, y_values, s=74, c=point_colors, edgecolor=INK, linewidth=0.6, zorder=4)
    ax_attr.set_xticks(x_values, frame["window_label"].astype(str).tolist(), rotation=0)
    ax_attr.set_xlim(-0.45, max(len(frame) - 0.55, 0.55))
    if y_values:
        ax_attr.set_ylim(0, max(max(y_values) * 1.18, 1.0))
    ax_attr.set_ylabel(_label(font, "事件归因得分", "event attribution"))
    ax_attr.set_xlabel(_label(font, "代表窗口", "selected window"))
    ax_attr.set_title(
        _label(font, "窗口、主导查询与事件归因", "Window, Dominant Query, and Event Attribution"),
        fontsize=11,
        weight="bold",
    )
    ax_attr.grid(axis="y", color=GRID, lw=0.8)
    ax_attr.set_axisbelow(True)
    legend_names = list(dict.fromkeys(query_names.tolist()))
    if len(legend_names) > 1:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w", label=_query_name_cn(name), markerfacecolor=_query_color(name), markeredgecolor=INK, markersize=7)
            for name in legend_names
        ]
        ax_attr.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    annotation_indices = _runtime_annotation_indices(frame)
    for idx in annotation_indices:
        row = frame.iloc[idx]
        ax_attr.annotate(
            f"{row['window_label']}\n{_query_name_cn(row['semantic_top_query_name'])}",
            xy=(idx, float(row["semantic_top_event_attribution"])),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )

    ax_dist = fig.add_subplot(grid[1, 2])
    counts = query_names_cn.value_counts().sort_values()
    color_by_cn = {_query_name_cn(name): _query_color(name) for name in legend_names}
    ax_dist.barh(counts.index.tolist(), counts.values.tolist(), color=[color_by_cn.get(name, MUTED) for name in counts.index], edgecolor=INK, linewidth=0.4)
    ax_dist.set_title(_label(font, "查询类型分布", "Query distribution"), fontsize=11, weight="bold")
    ax_dist.set_xlabel(_label(font, "窗口数", "windows"))
    ax_dist.grid(axis="x", color=GRID, lw=0.8)
    ax_dist.set_axisbelow(True)
    for y, value in enumerate(counts.values.tolist()):
        ax_dist.text(value, y, f" {value}", va="center", fontsize=8.2, color=INK)

    ax_ranges = fig.add_subplot(grid[1, 3])
    ax_ranges.set_axis_off()
    ax_ranges.set_xlim(0, 4.2)
    ax_ranges.set_ylim(-0.12, 2.2)
    range_cards = [
        (_label(font, "风险置信度", "risk confidence"), _range_text(frame["risk_proxy_confidence"]), BLUE),
        (_label(font, "工作负荷输出", "workload output"), _range_text(frame["workload_proxy_prediction"]), GREEN),
        (_label(font, "复盘检索得分", "event score"), _range_text(frame["event_replay_tag_score"]), PURPLE),
    ]
    for idx, (name, value, color) in enumerate(range_cards):
        y = 1.72 - idx * 0.62
        ax_ranges.add_patch(
            patches.FancyBboxPatch(
                (0.08, y - 0.25),
                3.8,
                0.43,
                boxstyle="round,pad=0.035,rounding_size=0.045",
                facecolor=PALE,
                edgecolor=GRID,
                lw=0.9,
            )
        )
        ax_ranges.text(0.25, y, name, fontsize=8.4, color=MUTED, va="center")
        ax_ranges.text(3.68, y, value, fontsize=10.0, color=color, weight="bold", ha="right", va="center")
    ax_ranges.set_title(_label(font, "任务输出范围", "Task Output Ranges"), fontsize=10.8, weight="bold", color=INK)

    fig.suptitle(_label(font, "运行语义案例：代表窗口归因与任务输出", "Runtime Semantic Case: Representative Windows"), fontsize=15.2, weight="bold", color=INK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_semantic_case",
        path=path,
        source_paths=_source_paths(sources, "runtime_case_table", "support", "runtime_service", "runtime_schema_contract"),
        evidence_layer="runtime_semantic_support",
        table_path=table_paths["runtime_semantic_case"],
        metric_definition="Runtime case values are selected from runtime_semantic_case.csv and shown as dominant query, semantic attribution, and task output ranges.",
        case_definition=str(case.get("case_definition") or "runtime semantic support case"),
        replaces_problem="uses a compact selected-window axis and replaces near-flat/repeated attribution bars with case-level semantic attribution and task-output ranges.",
    )


def _plot_rigid_body_rotation(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_rigid_body_rotation_rows(sources))
    family_rows = frame.loc[frame["row_type"] == "family_metric"].copy()
    matrix_rows = frame.loc[frame["row_type"] == "rotation_field_matrix"].copy()
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 5.4), gridspec_kw={"width_ratios": [0.9, 0.9, 0.9, 1.1]})
    pivot = family_rows.pivot(index="family", columns="metric_name", values="metric_value")
    families = [name for name in ("minimal", "full", "rigid_body") if name in pivot.index]
    metrics = [
        ("test_total", "联合损失", BLUE),
        ("test_alignment", "对齐损失", GOLD),
        ("test_physics_total", "物理残差", PURPLE),
    ]
    family_labels = {"minimal": "基础配置", "full": "弱物理约束", "rigid_body": "刚体约束"}
    for axis, (metric, title, color) in zip(axes[:3], metrics, strict=True):
        values = [float(pivot.loc[family, metric]) for family in families]
        axis.bar(range(len(families)), values, color=color, edgecolor=INK, linewidth=0.4)
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(_ascii_tick_formatter())
        axis.yaxis.set_minor_formatter(_null_tick_formatter())
        axis.set_xticks(range(len(families)), [family_labels.get(family, family) for family in families], rotation=18, ha="right")
        axis.set_title(f"{title}\n越低越好", fontsize=10.8, weight="bold")
        axis.grid(axis="y", color=GRID, lw=0.8)
        axis.set_axisbelow(True)
        for i, value in enumerate(values):
            axis.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8.0, color=INK)
    axes[0].set_ylabel(_label(font, "log尺度", "log scale"))
    axes[3].set_title(_label(font, "旋转字段可用性", "Rotation field availability"), fontsize=11, weight="bold")
    axis_labels = ["pitch", "roll", "yaw"]
    field_labels = ["angle", "rate"]
    matrix = []
    for axis in axis_labels:
        row_values = []
        for field_type in field_labels:
            selected = matrix_rows.loc[(matrix_rows["axis"] == axis) & (matrix_rows["field_type"] == field_type)]
            row_values.append(1 if not selected.empty and bool(selected["available"].iloc[0]) else 0)
        matrix.append(row_values)
    axes[3].imshow(matrix, cmap=_availability_cmap(), vmin=0, vmax=1, aspect="auto")
    axes[3].set_xticks(range(len(field_labels)), ["角度", "角速度"])
    axes[3].set_yticks(range(len(axis_labels)), ["俯仰", "横滚", "航向"])
    for y, axis in enumerate(axis_labels):
        for x, field_type in enumerate(field_labels):
            label = "已识别" if matrix[y][x] else "待接入"
            axes[3].text(x, y, label, ha="center", va="center", color=INK if matrix[y][x] else "#7f1d1d", fontsize=10, weight="bold")
    axes[3].text(
        0.5,
        -0.18,
        _label(font, "平移与垂向残差已进入训练，旋转残差已完成字段基础诊断。", "Translation and vertical residuals are trained; rotation fields have been diagnosed."),
        ha="center",
        va="top",
        transform=axes[3].transAxes,
        fontsize=8.4,
        color=MUTED,
    )
    fig.suptitle(_label(font, "刚体约束与旋转字段诊断", "Rigid-Body Constraint and Rotation Field Diagnostics"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="rigid_body_rotation_audit",
        path=path,
        source_paths=_source_paths(sources, "rigid_body", "rotation_audit"),
        evidence_layer="rigid_body_rotation_diagnostics",
        table_path=table_paths["rigid_body_rotation_audit"],
        metric_definition="三类损失分开显示并使用对数尺度；旋转矩阵标记角度和角速度字段可用性，当前数据缺少成对角速度字段。",
        replaces_problem="rotation rate absence is shown as a matrix, not empty bars.",
    )


def _plot_weak_label_sweep(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_weak_label_rows(sources))
    completed = frame.loc[frame["row_type"] == "completed_run"].copy()
    completed["test_total_numeric"] = pd.to_numeric(completed["test_total"], errors="coerce")
    best = completed.sort_values("test_total_numeric").groupby("sample_source", as_index=False).first()
    plt, _ = _import_matplotlib(font)
    fig = plt.figure(figsize=(14.2, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])
    metrics = [
        ("best_test_total", "联合损失"),
        ("test_task_total", "任务损失"),
        ("test_causal_total", "因果约束损失"),
    ]
    for idx, (metric, title) in enumerate(metrics):
        ax = fig.add_subplot(grid[0, idx])
        values = []
        labels = []
        for row in best.to_dict(orient="records"):
            labels.append(_sample_source_label(row["sample_source"]))
            if metric == "best_test_total":
                values.append(float(row["test_total"]))
            else:
                values.append(float(row[metric]))
        ax.bar(labels, values, color=[GREEN if "live" in label else BLUE for label in labels], edgecolor=INK, linewidth=0.4)
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(8)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}" if value < 10 else f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax_heat = fig.add_subplot(grid[1, :])
    heat = completed.pivot_table(index="sample_source", columns="lag_label", values="test_total", aggfunc="min")
    heat = heat.reindex(sorted(heat.index), axis=0)
    heat = heat.reindex(sorted(heat.columns, key=lambda item: str(item)), axis=1)
    im = ax_heat.imshow(heat.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax_heat.set_xticks(range(len(heat.columns)), [str(col) for col in heat.columns])
    ax_heat.set_yticks(range(len(heat.index)), [_sample_source_label(idx) for idx in heat.index], fontsize=8)
    ax_heat.set_title(_label(font, "滞后窗口参数 × 数据读取方式", "Lag-window parameter by data input route"), fontsize=10.5, weight="bold")
    heat_values = [float(value) for value in heat.to_numpy().ravel() if pd.notna(value)]
    heat_threshold = (min(heat_values) + max(heat_values)) / 2 if heat_values else 0
    for y in range(len(heat.index)):
        for x in range(len(heat.columns)):
            value = heat.iloc[y, x]
            if pd.notna(value):
                text_color = "white" if float(value) >= heat_threshold else INK
                ax_heat.text(x, y, f"{float(value):.1f}", ha="center", va="center", fontsize=8.2, color=text_color, weight="bold")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02, label="联合损失")
    cbar.ax.yaxis.set_major_formatter(_ascii_tick_formatter())
    cbar.ax.yaxis.set_minor_formatter(_null_tick_formatter())
    fig.suptitle(_label(font, "弱监督多任务参数与数据读取方式对比", "Weak-label Multitask Parameters and Data Input Routes"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="weak_label_sweep_ablation",
        path=path,
        source_paths=_source_paths(sources, "proxy_sweep", "live_sweep", "live_partial"),
        evidence_layer="thesis_weak_label",
        table_path=table_paths["weak_label_sweep_ablation"],
        metric_definition="小规模弱监督参数组合按数据读取方式和滞后窗口参数比较；部分执行与续跑状态仅保留在结构化表和清单中。",
        replaces_problem="weak-label sweep no longer uses two near-identical bars.",
    )


def _plot_private_component(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_private_component_rows(sources))
    if "ablation_group" in frame and (frame["ablation_group"] == "model_backbone").any():
        frame = frame.loc[frame["ablation_group"] == "model_backbone"].copy()
        preferred = ["naive_time_sync", "continuous_dual_state", "remove_physics_constraint", "remove_causal_mask", "remove_semantic_event_fusion", "full_model"]
    else:
        preferred = [
            "naive_sync",
            "e_baseline",
            "f_full",
            "g_min",
            "chronaris_opt",
            "chronaris_opt_no_causal_mask",
            "chronaris_opt_no_time_residual",
            "chronaris_opt_no_task_head",
        ]
        frame = frame.loc[frame["variant_name"].isin(preferred)].copy()
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.7))
    tasks = [
        ("T1_maneuver_intensity_class", _label(font, "T1 机动强度：宏平均F1值越高越好", "T1 macro-F1 higher is better"), False),
        ("T2_next_window_physiology_response", _label(font, "T2 生理响应：RMSE越低越好", "T2 RMSE lower is better"), True),
        ("T3_paired_pilot_window_retrieval", _label(font, "T3 配对检索：Top-1准确率越高越好", "T3 Top-1 higher is better"), False),
    ]
    for ax, (task, title, log_x) in zip(axes.flat[:3], tasks, strict=True):
        task_rows = frame.loc[frame["task_name"] == task].copy()
        task_rows["order"] = task_rows["variant_name"].map({name: idx for idx, name in enumerate(preferred)})
        task_rows = task_rows.sort_values("order", ascending=False)
        values = task_rows["primary_metric_value"].astype(float)
        if not task_rows.empty and bool(np.isclose(values.to_numpy(dtype=float), 0.0).all()):
            _draw_zero_metric_panel(ax, task_rows, title, font)
            continue
        colors = [_variant_color(role) for role in task_rows["variant_role"]]
        ax.barh(task_rows["display_variant_cn"], values, color=colors, edgecolor=INK, linewidth=0.4)
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(_ascii_tick_formatter())
            ax.xaxis.set_minor_formatter(_null_tick_formatter())
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for y, value in enumerate(task_rows["primary_metric_value"].astype(float).tolist()):
            ax.text(value, y, f" {value:.3g}", va="center", fontsize=8.0, color=INK)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(8)
    ax = axes.flat[3]
    _draw_degradation_heatmap(ax, frame, font)
    fig.suptitle(_label(font, "组件消融总览：分任务指标与退化热力图", "Component Ablation Overview"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="chronaris_opt_component_ablation",
        path=path,
        source_paths=_source_paths(sources, "private_component"),
        evidence_layer="private_proxy",
        table_path=table_paths["chronaris_opt_component_ablation"],
        metric_definition="Task metrics are shown on native scales and component-by-task degradation is shown as raw and relative delta.",
        recommended_placement="supporting_overview",
        replaces_problem="mixed-unit delta bars are replaced by task facets plus normalized contribution; all-zero task panels are rendered as diagnostics instead of blank bars.",
    )


def _plot_model_backbone_ablation(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_model_backbone_ablation_rows(sources))
    if frame.empty:
        frame = pd.DataFrame(build_private_component_rows(sources))
    plt, _ = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    _draw_degradation_heatmap(ax, frame, font)
    fig.suptitle(_label(font, "模型骨干结构消融", "Model Backbone Ablation"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="model_backbone_ablation",
        path=path,
        source_paths=_source_paths(sources, "private_component"),
        evidence_layer="private_proxy_leakage_safe",
        table_path=table_paths["model_backbone_ablation"],
        metric_definition="Leakage-safe model-backbone ablation with raw delta and relative percent degradation by task.",
        recommended_placement="main_experiment_component_analysis",
    )


def _plot_task_adapter_ablation(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_task_adapter_ablation_rows(sources))
    if frame.empty:
        frame = pd.DataFrame(build_private_component_rows(sources)).head(0)
    plt, _ = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    _draw_degradation_heatmap(ax, frame, font)
    fig.suptitle(_label(font, "任务适配层消融", "Task Adapter Ablation"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="task_adapter_ablation",
        path=path,
        source_paths=_source_paths(sources, "private_component"),
        evidence_layer="private_proxy_leakage_safe",
        table_path=table_paths["task_adapter_ablation"],
        metric_definition="Leakage-safe task-adapter ablation with seed mean/std and component-by-task degradation.",
        recommended_placement="main_experiment_component_analysis",
    )


def _plot_public_transfer(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_public_transfer_rows(sources)).sort_values("segment_order")
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(13.6, 5.1))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.7)
    x_positions = [0.35, 4.35, 8.35]
    box_w = 3.2
    box_h = 2.15
    colors = ["#eaf3fb", "#eef7ef", "#fff2e7"]
    edges = [BLUE, GREEN, ORANGE]
    for idx, row in enumerate(frame.to_dict(orient="records")):
        x = x_positions[idx]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 1.35),
                box_w,
                box_h,
                boxstyle="round,pad=0.08,rounding_size=0.06",
                facecolor=colors[idx],
                edgecolor=edges[idx],
                lw=1.6,
            )
        )
        ax.text(x + box_w / 2, 3.16, _label(font, row["segment_title_cn"], row["segment_title"]), ha="center", fontsize=12.2, weight="bold", color=INK)
        ax.text(x + box_w / 2, 2.66, _wrap_text(_label(font, row["data_scope_cn"], row["data_scope_cn"]), 20), ha="center", va="center", fontsize=9.1, color=INK)
        ax.text(x + box_w / 2, 2.12, _wrap_text(_label(font, row["evidence_role_cn"], row["evidence_role_cn"]), 20), ha="center", va="center", fontsize=9.0, color=MUTED)
        ax.text(x + box_w / 2, 1.62, _short(row["main_output_cn"], 34), ha="center", fontsize=8.4, color=edges[idx], weight="bold")
        if idx < 2:
            _draw_arrow(ax, patches, (x + box_w + 0.18, 2.42), (x_positions[idx + 1] - 0.18, 2.42), lw=2.2, mutation_scale=17)
            ax.text((x + box_w + x_positions[idx + 1]) / 2, 2.72, _label(font, "支撑", "supports"), ha="center", fontsize=8.4, color=MUTED)
    ax.text(
        6.0,
        0.72,
        _label(font, "公开数据用于适配校准，真实航空双流用于验证闭环，组件代理任务用于结构敏感性测试。", "Public data supports adaptation, private dual-stream data supports validation, proxy tasks support structural sensitivity tests."),
        ha="center",
        fontsize=10.2,
        color=INK,
        weight="bold",
    )
    ax.set_title(_label(font, "公开数据适配、真实航空验证与组件分析的作用分工", "Role Split: Public Adaptation, Aviation Validation, and Component Analysis"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="public_transfer_boundary",
        path=path,
        source_paths=_source_paths(sources, "public_transfer", "public_calibration", "private_component"),
        evidence_layer="transfer_boundary",
        table_path=table_paths["public_transfer_boundary"],
        metric_definition="Chinese positive role diagram separating public adapter, private weak-label mainline, and private proxy diagnostics.",
        replaces_problem="defensive boundary copy is replaced by positive Chinese role wording.",
    )


def _plot_semantic_event_fusion(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_semantic_event_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig = plt.figure(figsize=(14.2, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.82, 1.15])
    ax_flow = fig.add_subplot(grid[0])
    ax_flow.set_axis_off()
    ax_flow.set_xlim(0, 12)
    ax_flow.set_ylim(0, 2.7)
    steps_cn = ["人机双流输入", "因果掩码", "事件表示", "语义查询", "归因对齐", "任务输出"]
    steps_en = ["dual-stream input", "causal mask", "event tokens", "semantic query", "attribution", "risk/workload/event outputs"]
    x_positions = [0.25, 2.16, 4.04, 5.92, 7.83, 9.72]
    widths = [1.42, 1.36, 1.36, 1.42, 1.42, 1.88]
    for idx, (x, width) in enumerate(zip(x_positions, widths, strict=True)):
        ax_flow.add_patch(
            patches.FancyBboxPatch(
                (x, 1.04),
                width,
                0.74,
                boxstyle="round,pad=0.06,rounding_size=0.05",
                facecolor="#f5f7fb",
                edgecolor=[BLUE, BLUE, GREEN, GOLD, PURPLE, ORANGE][idx],
                lw=1.4,
            )
        )
        ax_flow.text(x + width / 2, 1.41, _wrap_text(_label(font, steps_cn[idx], steps_en[idx]), 12), ha="center", va="center", fontsize=9.4, color=INK, weight="bold" if idx in {0, 5} else "normal")
        if idx < len(x_positions) - 1:
            _draw_arrow(ax_flow, patches, (x + width + 0.1, 1.41), (x_positions[idx + 1] - 0.1, 1.41), lw=1.9, mutation_scale=15)
    ax_flow.set_title(_label(font, "双流语义事件融合链路", "Dual-Stream Semantic Event Fusion Path"), fontsize=13, weight="bold", color=INK)
    ax = fig.add_subplot(grid[1])
    query_rows = frame.loc[frame["row_type"] == "view_query_attribution"].copy()
    if not query_rows.empty:
        query_rows["mean_query_attribution"] = pd.to_numeric(query_rows["mean_query_attribution"], errors="coerce")
        heat = query_rows.pivot_table(index="view_id", columns="query_type_cn", values="mean_query_attribution", aggfunc="mean")
        im = ax.imshow(heat.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(heat.columns)), list(heat.columns), fontsize=9)
        ax.set_yticks(range(len(heat.index)), [_identifier_label(view, width=28) for view in heat.index], fontsize=8)
        ax.set_title(_label(font, "数据视图 × 查询类型平均归因得分", "Data view by query type mean attribution"), fontsize=11, weight="bold")
        for y in range(len(heat.index)):
            for x in range(len(heat.columns)):
                value = heat.iloc[y, x]
                if pd.notna(value):
                    ax.text(x, y, f"{float(value):.2f}", ha="center", va="center", fontsize=8.2, color=INK)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=_label(font, "平均归因得分", "mean attribution"))
        cbar.ax.yaxis.set_major_formatter(_ascii_tick_formatter())
        cbar.ax.yaxis.set_minor_formatter(_null_tick_formatter())
    else:
        ax.set_axis_off()
        requirement = ""
        missing = frame.loc[frame["row_type"] == "source_requirement"]
        if not missing.empty:
            requirement = str(missing.iloc[0].get("missing_source_data_requirement") or "")
        ax.text(0.5, 0.62, _label(font, "当前结构化产物未保存完整“数据视图 × 查询类型”归因矩阵", "Current source artifact lacks a complete view-query attribution matrix"), ha="center", va="center", fontsize=12, weight="bold", color=INK)
        ax.text(0.5, 0.42, _wrap_text(requirement, 58), ha="center", va="center", fontsize=9.6, color=MUTED)
    fig.suptitle(_label(font, "语义事件融合：事件表示与语义查询归因", "Semantic Event Fusion: Event Representation and Query Attribution"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="semantic_event_fusion_overview",
        path=path,
        source_paths=_source_paths(sources, "support", "semantic_event"),
        evidence_layer="semantic_support",
        table_path=table_paths["semantic_event_fusion_overview"],
        case_definition="Pipeline diagram plus view-query heatmap when complete source scores exist; otherwise records the source-data requirement.",
        replaces_problem="replaces the old Stage G minimal causal-mask heatmap as the main semantic fusion figure.",
    )


def _plot_llm_comparison(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_llm_comparison_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(15.2, 6.35))
    ax.set_axis_off()
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 5.45)
    card_rows = frame.loc[frame["condition"].str.startswith("A")].copy()
    x_positions = [0.5, 3.35, 6.2, 9.05, 11.9]
    card_w = 2.0
    card_h = 2.64
    card_y = 1.45
    mid_y = card_y + 1.33
    colors = [BLUE, GREEN, GOLD, PURPLE, ORANGE]
    condition_labels = {
        "A0_baseline": "A0\n基础查询集合",
        "A1_llm_context": "A1\n字段与窗口语义上下文",
        "A2_llm_semantic_hints": "A2\n白名单语义查询建议",
        "A3_llm_runtime_explanation": "A3\n运行案例解释",
        "A4_human_review_packet": "A4\n人工复核材料",
    }
    for idx, row in enumerate(card_rows.to_dict(orient="records")):
        x = x_positions[idx]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, card_y),
                card_w,
                card_h,
                boxstyle="round,pad=0.06,rounding_size=0.07",
                facecolor="#fbfdff",
                edgecolor=colors[idx],
                lw=1.6,
            )
        )
        ax.text(
            x + card_w / 2,
            card_y + card_h - 0.28,
            condition_labels.get(str(row["condition"]), str(row["condition"])),
            ha="center",
            va="top",
            fontsize=8.6,
            weight="bold",
            color=colors[idx],
            linespacing=1.12,
        )
        ax.text(
            x + card_w / 2,
            card_y + card_h - 0.98,
            _wrap_text(_llm_stage_cn(row), 18),
            ha="center",
            va="center",
            fontsize=8.2,
            color=INK,
            linespacing=1.15,
        )
        ax.text(x + card_w / 2, card_y + 1.12, _llm_metric_name_cn(row["metric_name"]), ha="center", fontsize=7.7, color=MUTED)
        ax.text(x + card_w / 2, card_y + 0.78, _short(row["metric_value"], 16), ha="center", fontsize=14.6, weight="bold", color=INK)
        ax.text(
            x + card_w / 2,
            card_y + 0.31,
            _wrap_text(_llm_card_note(font, row), 22),
            ha="center",
            va="center",
            fontsize=7.2,
            color=MUTED,
            linespacing=1.12,
        )
        if idx < len(x_positions) - 1:
            _draw_arrow(ax, patches, (x + card_w + 0.18, mid_y), (x_positions[idx + 1] - 0.18, mid_y), lw=1.8, mutation_scale=14)
    pre_row = frame.loc[frame["condition"].str.endswith("preprocessing_run")].iloc[0].to_dict()
    ax.text(
        0.5,
        0.76,
        _llm_preprocessing_note_cn(pre_row),
        fontsize=9.5,
        color=INK,
        weight="bold",
    )
    ax.set_title(_label(font, "大语言模型辅助数据预处理与复核流程", "LLM-Assisted Data Preprocessing and Review Flow"), fontsize=15, weight="bold", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="llm_comparison_a0_a4",
        path=path,
        source_paths=_source_paths(sources, "llm_preprocessing", "llm_comparison"),
        evidence_layer="llm_preprocessing_comparison",
        table_path=table_paths["llm_comparison_a0_a4"],
        metric_definition="A0-A4 comparison from the LLM preprocessing comparison summary; LLM hints show coverage only, not attribution improvement.",
        replaces_problem="adds LLM preprocessing/comparison evidence without treating LLM output as truth.",
    )


def _draw_arrow(ax, patches, start: tuple[float, float], end: tuple[float, float], *, lw: float, mutation_scale: float) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=lw,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def _figure_entry(
    *,
    figure_id: str,
    path: Path,
    source_paths: list[str],
    evidence_layer: str,
    table_path: str,
    metric_definition: str | None = None,
    case_definition: str | None = None,
    recommended_placement: str | None = None,
    replaces_problem: str | None = None,
) -> dict[str, object]:
    payload = {
        "figure_id": figure_id,
        "path": str(path),
        "exists": path.exists(),
        "source_path": [path for path in source_paths if path],
        "evidence_layer": evidence_layer,
        "table_path": table_path,
    }
    if metric_definition is not None:
        payload["metric_definition"] = metric_definition
    if case_definition is not None:
        payload["case_definition"] = case_definition
    if recommended_placement is not None:
        payload["recommended_placement"] = recommended_placement
    if replaces_problem is not None:
        payload["replaces_problem"] = replaces_problem
    return payload


def _import_matplotlib(font: PlotFontSelection):
    from matplotlib import colors
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from matplotlib import rcParams

    if font.family:
        rcParams["font.family"] = [font.family]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    return plt, patches


def _ascii_tick_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _position: f"{value:g}".replace("−", "-"))


def _null_tick_formatter():
    from matplotlib.ticker import NullFormatter

    return NullFormatter()


def _availability_cmap():
    from matplotlib import colors

    return colors.ListedColormap(["#fde8e2", "#e6f4ea"])


def _draw_zero_metric_panel(ax, task_rows: pd.DataFrame, title: str, font: PlotFontSelection) -> None:
    from matplotlib.patches import Rectangle

    metric_name = str(task_rows["primary_metric_name"].iloc[0]).replace("_", " ")
    metric_name_cn = _metric_title_cn(metric_name)
    protocol = str(task_rows.get("protocol", pd.Series([""])).iloc[0] or "")
    split_strategy = str(task_rows.get("split_strategy", pd.Series([""])).iloc[0] or "")
    seed_count = int(task_rows.get("seed_count", pd.Series([0])).max() or 0)
    variant_count = int(task_rows["variant_name"].nunique())
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10.5, weight="bold", pad=9)
    ax.add_patch(
        Rectangle(
            (0.04, 0.14),
            0.92,
            0.72,
            facecolor=PALE,
            edgecolor=GRID,
            linewidth=1.0,
        )
    )
    ax.text(
        0.5,
        0.63,
        _label(font, f"所有变体 {metric_name_cn} = 0", f"All variants {metric_name} = 0"),
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.5,
        0.48,
        _label(font, "严格候选池下未命中；不是缺失数据", "Strict candidate pool: no hits; data is present"),
        ha="center",
        va="center",
        fontsize=9.8,
        color=MUTED,
    )
    detail = _label(
        font,
        f"变体数={variant_count} | 种子数={seed_count} | 协议={protocol or 'unknown'}",
        f"variants={variant_count} | seeds={seed_count} | protocol={protocol or 'unknown'}",
    )
    ax.text(0.5, 0.33, detail, ha="center", va="center", fontsize=8.4, color=MUTED)
    if split_strategy:
        ax.text(
            0.5,
            0.23,
            _label(font, f"评价方式：{split_strategy} / {metric_name}", f"evaluation: {split_strategy} / {metric_name}"),
            ha="center",
            va="center",
            fontsize=8.0,
            color=MUTED,
        )


def _label(font: PlotFontSelection, cn_label: str, ascii_label: str) -> str:
    return ascii_label if font.ascii_only else cn_label


def _metric_title_cn(value: str) -> str:
    mapping = {
        "macro f1": "宏平均F1",
        "rmse": "RMSE",
        "top1 accuracy": "Top-1",
    }
    return mapping.get(value, value)


def _metric_cell(ax, x: float, y: float, name: object, value: object, color: str) -> None:
    ax.text(x + 0.05, y + 0.11, str(name), fontsize=7.8, color=MUTED, va="center")
    ax.text(x + 0.05, y - 0.16, _short(value, 22), fontsize=11.5, color=color, weight="bold", va="center")


def _draw_degradation_heatmap(ax, frame: pd.DataFrame, font: PlotFontSelection) -> None:
    ax.set_axis_off()
    if frame.empty:
        ax.text(0.5, 0.5, _label(font, "暂无防泄漏消融行", "No leakage-safe ablation rows"), ha="center", va="center", fontsize=11, color=MUTED)
        return
    working = frame.copy()
    if "relative_delta_percent" not in working or working["relative_delta_percent"].isna().all():
        working["relative_delta_percent"] = working["normalized_delta_vs_full"].fillna(0.0).astype(float) * 100.0
    tasks = [task for task in ("T1_maneuver_intensity_class", "T2_next_window_physiology_response", "T3_paired_pilot_window_retrieval") if task in set(working["task_name"])]
    variants = list(dict.fromkeys(working["display_variant_cn"].fillna(working["display_variant"]).astype(str).tolist()))
    heat = np.full((len(variants), len(tasks)), np.nan, dtype=float)
    labels = [["" for _ in tasks] for _ in variants]
    for row in working.to_dict(orient="records"):
        if row["task_name"] not in tasks:
            continue
        y = variants.index(str(row.get("display_variant_cn") or row.get("display_variant")))
        x = tasks.index(row["task_name"])
        relative = float(row.get("relative_delta_percent") or 0.0)
        delta = float(row.get("delta_vs_full") or 0.0)
        heat[y, x] = relative
        labels[y][x] = f"{delta:.3g}\n{relative:.1f}%"
    ax.set_axis_on()
    im = ax.imshow(np.nan_to_num(heat, nan=0.0), cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(tasks)), [_task_label_cn(task) for task in tasks], fontsize=8.5)
    ax.set_yticks(range(len(variants)), variants, fontsize=8.2)
    for y in range(len(variants)):
        for x in range(len(tasks)):
            ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=7.6, color=INK)
    ax.set_title(_label(font, "组件 × 任务性能退化（原始变化量 / 相对变化）", "Component by task degradation"), fontsize=10.5, weight="bold")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=_label(font, "相对变化百分比", "relative percent"))
    cbar.ax.yaxis.set_major_formatter(_ascii_tick_formatter())
    cbar.ax.yaxis.set_minor_formatter(_null_tick_formatter())


def _metric_name_cn(value: object) -> str:
    mapping = {
        "sample_count": "样本数",
        "task_entry_count": "任务条目数",
        "component_rows": "组件行数",
        "task_count": "任务数",
        "calibration_rows": "校准行数",
        "baseline_categories": "基线类别数",
        "replay_window_count": "回放窗口数",
        "vehicle_feature_gap": "飞机字段缺口",
        "view_count": "数据视图数",
        "query_count": "查询类型数",
        "enabled_residual_count": "已启用残差数",
        "rotation_status": "旋转状态",
        "request_count": "请求数",
        "semantic_query_coverage": "语义查询覆盖",
    }
    return mapping.get(str(value), str(value))


def _schema_status_cn(value: object) -> str:
    mapping = {"aligned": "已对齐", "exact": "已通过", "missing": "缺失"}
    return mapping.get(str(value), str(value))


def _query_name_cn(value: object) -> str:
    mapping = {
        "risk_proxy": "风险",
        "workload_proxy": "工作负荷",
        "event_replay_tag": "事件复盘",
        "unknown": "未知",
    }
    return mapping.get(str(value), str(value))


def _task_label_cn(value: object) -> str:
    mapping = {
        "T1_maneuver_intensity_class": "T1\n机动强度",
        "T2_next_window_physiology_response": "T2\n生理响应",
        "T3_paired_pilot_window_retrieval": "T3\n配对检索",
    }
    return mapping.get(str(value), str(value))


def _sample_source_label(value: object) -> str:
    text = str(value)
    if text == "live_influx":
        return "真实时序\n数据读取"
    if text == "stage_h_window_stats_proxy":
        return "标准化窗口\n统计输入"
    return text.replace("_", "\n")


def _llm_card_note(font: PlotFontSelection, row: Mapping[str, object]) -> str:
    condition = str(row.get("condition", ""))
    metric_note = str(row.get("metric_note", ""))
    if condition.startswith("A0"):
        return _label(font, "内置查询集合", metric_note.replace("_", " "))
    if condition.startswith("A1"):
        return _label(font, "标签未改写；变化数为0", metric_note.replace("_", " "))
    if condition.startswith("A2"):
        return _label(font, "3->7；新增 4", metric_note.replace("_", " "))
    if condition.startswith("A3"):
        return _label(font, "4/12；完整性=1.0", metric_note.replace("_", " "))
    if condition.startswith("A4"):
        return _label(font, "人工复核未完成", metric_note.replace("_", " "))
    return metric_note.replace("_", " ")


def _llm_stage_cn(row: Mapping[str, object]) -> str:
    condition = str(row.get("condition", ""))
    mapping = {
        "A0_baseline": "内置查询集合",
        "A1_llm_context": "接入语义上下文",
        "A2_llm_semantic_hints": "白名单语义建议",
        "A3_llm_runtime_explanation": "运行解释子集",
        "A4_human_review_packet": "人工复核材料",
    }
    return mapping.get(condition, str(row.get("stage_cn") or row.get("stage") or ""))


def _llm_metric_name_cn(value: object) -> str:
    mapping = {
        "baseline_query_count": "基础查询数",
        "attached_entry_count": "接入条目数",
        "query_count": "查询总数",
        "explained_case_count": "解释案例数",
        "review_item_count": "复核条目数",
        "request_count": "请求数",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _llm_preprocessing_note_cn(row: Mapping[str, object]) -> str:
    note = str(row.get("metric_note") or "")
    parts: dict[str, str] = {}
    for item in note.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip()] = value.strip()
    return (
        f"预处理请求 {row.get('metric_value')} 次；"
        f"错误 {parts.get('errors', '0')} 次；"
        f"语义建议 {parts.get('hints', '0')} 条；"
        f"运行解释 {parts.get('runtime_explanations', '0')} 条"
    )


def _runtime_annotation_indices(frame: pd.DataFrame) -> list[int]:
    selected: list[int] = []
    previous_query = None
    previous_value = None
    for idx, row in frame.iterrows():
        query = str(row.get("semantic_top_query_name") or "")
        value = float(row.get("semantic_top_event_attribution") or 0.0)
        changed_query = previous_query is not None and query != previous_query
        changed_value = previous_value is not None and abs(value - previous_value) > 1e-6
        if idx == 0 or changed_query or changed_value:
            selected.append(int(idx))
        previous_query = query
        previous_value = value
    top_idx = int(frame["semantic_top_event_attribution"].fillna(0.0).idxmax())
    result: list[int] = []
    for idx in selected[:2]:
        if idx not in result:
            result.append(idx)
    if top_idx not in result:
        result.append(top_idx)
    for idx in selected[2:]:
        if len(result) >= 3:
            break
        if idx not in result:
            result.append(idx)
    return result[:3]


def _query_color(name: object) -> str:
    value = str(name)
    if value == "risk_proxy":
        return BLUE
    if value == "workload_proxy":
        return GREEN
    if value == "event_replay_tag":
        return GOLD
    return PURPLE


def _fmt_metric(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "NA"
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _range_text(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    low = float(values.min())
    high = float(values.max())
    if abs(high - low) < 1e-9:
        return _fmt_metric(low)
    return f"{_fmt_metric(low)} - {_fmt_metric(high)}"


def _identifier_label(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    text = text.replace("__pilot_", "\npilot_")
    lines: list[str] = []
    for part in text.splitlines():
        lines.extend(_split_identifier_part(part, width))
    return "\n".join(lines)


def _runtime_case_view_label(value: object) -> str:
    text = "" if value is None else str(value)
    if "__pilot_" in text:
        sortie_id, pilot_id = text.split("__pilot_", 1)
        return f"飞行架次: {sortie_id}\n飞行员: {pilot_id}"
    return _identifier_label(text, width=70)


def _split_identifier_part(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    lines: list[str] = []
    remaining = text
    separators = ("_", ":", "#", "-")
    while len(remaining) > width:
        split_at = max(remaining.rfind(separator, 0, width + 1) for separator in separators)
        if split_at < max(8, width // 2):
            split_at = width
        else:
            split_at += 1
        lines.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        lines.append(remaining)
    return lines


def _short(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    return shorten(text, width=width, placeholder="...")


def _wrap_text(text: str, width: int) -> str:
    value = str(text)
    if len(value) <= width:
        return value
    if " " in value:
        return "\n".join(
            wrap(value, width=width, break_long_words=False, break_on_hyphens=False)[:3]
        )
    return "\n".join(value[index : index + width] for index in range(0, min(len(value), width * 3), width))


def _variant_color(role: str) -> str:
    if role == "full_candidate":
        return GREEN
    if role == "component_removed":
        return ORANGE
    return "#cbd5e1"


def _status_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    completed = frame.loc[frame["row_type"] == "completed_run"].copy()
    for sample_source, group in completed.groupby("sample_source"):
        rows.append(
            {
                "label": _sample_source_label(sample_source),
                "completed": int(group["child_run_id"].nunique()),
                "remaining": 0,
            }
        )
    partial = frame.loc[frame["row_type"] == "partial_status"]
    for row in partial.to_dict(orient="records"):
        completed_count = int(row.get("completed_combination_count") or 0)
        target_count = int(row.get("target_combination_count") or completed_count)
        rows.append(
            {
                "label": f"{_sample_source_label(row['sample_source'])}\npartial",
                "completed": completed_count,
                "remaining": max(target_count - completed_count, 0),
            }
        )
    return rows


def _source_paths(sources: Mapping[str, Mapping[str, object]], *names: str) -> list[str]:
    paths = []
    for name in names:
        source = sources.get(name)
        if source and source.get("path"):
            paths.append(str(source["path"]))
    return paths
