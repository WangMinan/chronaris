"""Plotting helpers for task evaluation thesis-facing materials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten, wrap
from typing import Mapping

import pandas as pd
import numpy as np

from chronaris.evidence.thesis_materials_data import (
    build_evidence_layer_rows,
    build_llm_comparison_rows,
    build_model_backbone_ablation_rows,
    build_private_component_rows,
    build_public_transfer_rows,
    build_rigid_body_rotation_rows,
    build_runtime_payload_schema_rows,
    build_runtime_service_flow_rows,
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
        _plot_runtime_service_flow(run_root / "runtime_service_flow.png", sources, font, table_paths),
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
    fig, ax = plt.subplots(figsize=(14.6, 8.8))
    ax.set_axis_off()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.35)
    card_w = 2.82
    card_h = 2.58
    x_positions = [0.35, 3.45, 6.55, 9.65]
    y_positions = [4.25, 1.35]

    def draw_card(
        *,
        x: float,
        y: float,
        headline: str,
        quantity: str,
        role: str,
        color: str,
        facecolor: str = "#ffffff",
        linestyle: str = "-",
    ) -> None:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                card_w,
                card_h,
                boxstyle="round,pad=0.055,rounding_size=0.06",
                facecolor=facecolor,
                edgecolor=GRID,
                lw=1.15,
                linestyle=linestyle,
            )
        )
        ax.add_patch(
            patches.Rectangle((x, y + card_h - 0.2), card_w, 0.2, facecolor=color, edgecolor="none")
        )
        ax.text(
            x + 0.18,
            y + card_h - 0.34,
            _wrap_text_limited(headline, 12, 2),
            fontsize=11.2,
            weight="bold",
            color=INK,
            va="top",
            linespacing=1.1,
        )
        ax.text(
            x + 0.18,
            y + card_h - 1.16,
            _wrap_text_limited(quantity, 14, 2),
            fontsize=10.0,
            weight="bold",
            color=color,
            va="top",
            linespacing=1.08,
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + 0.14, y + 0.18),
                card_w - 0.28,
                0.78,
                boxstyle="round,pad=0.035,rounding_size=0.035",
                facecolor="#f8fafc",
                edgecolor="none",
            )
        )
        ax.text(
            x + 0.2,
            y + 0.82,
            _wrap_text_limited(role, 14, 3),
            fontsize=8.8,
            color=MUTED,
            va="top",
            linespacing=1.12,
        )

    rows = frame.to_dict(orient="records")
    for index, row in enumerate(rows):
        row_idx, col_idx = divmod(index, 4)
        x = x_positions[col_idx]
        y = y_positions[row_idx]
        color = LAYER_COLORS.get(str(row["evidence_layer"]), "#7a8793")
        headline, quantity, role = _evidence_card_text(row, font)
        draw_card(x=x, y=y, headline=headline, quantity=quantity, role=role, color=color)

    draw_card(
        x=x_positions[3],
        y=y_positions[1],
        headline=_label(font, "证据边界", "Evidence boundary"),
        quantity=_label(font, "分层解读\n不合并结论", "Separate layers\nNo merged conclusion"),
        role=_label(
            font,
            "弱监督、组件诊断、适配、契约、语义与预处理输出分别解读。",
            "Weak labels, proxies, adapters, contracts, semantics, and preprocessing stay separate.",
        ),
        color="#94a3b8",
        facecolor=PALE,
        linestyle="--",
    )
    ax.set_title(_label(font, "中期报告证据层总览", "Midterm Evidence Layer Overview"), fontsize=16, weight="bold", color=INK, pad=14)
    ax.text(
        0.35,
        0.55,
        _label(font, "所有数值来自现有结构化产物；不同证据层不合并为同一结论。", "All values are parsed from existing artifacts; evidence layers remain separate."),
        fontsize=9.2,
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
        metric_definition="Evidence-layer card overview with source-derived counts and report roles.",
        recommended_placement="appendix_or_supporting",
        replaces_problem="evidence_layer_overview is redrawn as a 2x4 report-readable card overview.",
        min_font_pt=8.8,
    )


def _plot_runtime_payload_schema(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_runtime_payload_schema_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(13.4, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.9)
    native = frame.loc[frame["payload_side"] == "left"].iloc[0].to_dict()
    canonical = frame.loc[frame["payload_side"] == "right"].iloc[0].to_dict()
    missing_groups = int(native.get("missing_vehicle_measurement_group_count") or 0)
    boxes = [
        (0.35, 1.05, 3.05, 2.82, "原始回放输入", ORANGE, "#fff7ed"),
        (4.45, 1.05, 3.05, 2.82, "字段规范检查", BLUE, "#f1f7ff"),
        (8.55, 1.05, 3.05, 2.82, "统一服务输入", GREEN, "#f2fbf4"),
    ]
    detail_rows = [
        [
            f"已读取飞机状态字段{native.get('vehicle_feature_count')}",
            f"生理字段{native.get('physiology_feature_count')}",
            f"回放窗口{native.get('sample_count')}",
        ],
        [
            f"待补齐范围：{missing_groups}个测量组",
            "字段顺序与名称已对齐",
            "缺失范围保留来源记录",
        ],
        [
            f"统一格式按{canonical.get('vehicle_feature_count')}维保留缺失标记",
            f"生理字段{canonical.get('physiology_feature_count')}",
            "模型输入检查通过",
        ],
    ]
    for idx, (x, y, w, h, title, color, fill) in enumerate(boxes):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.08",
                facecolor=fill,
                edgecolor=color,
                lw=1.6,
            )
        )
        ax.text(
            x + 0.2,
            y + h - 0.42,
            _label(font, title, title),
            fontsize=12.6,
            weight="bold",
            color=INK,
            va="center",
        )
        for offset, value in enumerate(detail_rows[idx]):
            yy = y + h - 1.02 - offset * 0.52
            ax.text(x + 0.26, yy, _wrap_text(value, 18), fontsize=10.7, color=INK, va="center", linespacing=1.1)
        ax.text(x + 0.26, y + 0.34, "状态：" + ["已读取", "已检查", "已通过"][idx], fontsize=10.0, color=color, weight="bold")
        if idx < len(boxes) - 1:
            _draw_arrow(ax, patches, (x + w + 0.3, y + h / 2), (boxes[idx + 1][0] - 0.3, y + h / 2), lw=2.2, mutation_scale=17)
    ax.text(
        6.0,
        0.48,
        _label(font, "该图仅说明运行字段格式与模型输入检查；缺源范围按测量组记录，不把缺失项写成实验失败。", "The figure reports input-format checks and missing-source scope."),
        ha="center",
        fontsize=9.8,
        color=MUTED,
    )
    ax.set_title(_label(font, "运行输入字段规范对照", "Runtime Input Field Format Check"), fontsize=15.5, weight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_payload_schema",
        path=path,
        source_paths=_source_paths(sources, "runtime_service", "runtime_schema_contract"),
        evidence_layer="runtime_schema",
        table_path=table_paths["runtime_payload_schema"],
        metric_definition="Runtime input format comparison showing observed vehicle fields, missing measurement groups, and unified input dimensions.",
        replaces_problem="runtime_payload_schema is rendered as a contract comparison instead of a generic schema plot.",
        min_font_pt=9.8,
    )


def _plot_runtime_service_flow(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_runtime_service_flow_rows(sources)).sort_values("step_order")
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.0)
    positions = [(0.7, 3.85), (5.55, 3.85), (0.7, 1.1), (5.55, 1.1)]
    colors = [BLUE, ORANGE, PURPLE, GREEN]
    for idx, row in enumerate(frame.to_dict(orient="records")):
        x, y = positions[idx]
        color = colors[idx]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                3.75,
                1.72,
                boxstyle="round,pad=0.07,rounding_size=0.08",
                facecolor="#ffffff",
                edgecolor=color,
                lw=1.7,
            )
        )
        ax.text(x + 0.22, y + 1.36, row["step_title_cn"], fontsize=13.0, weight="bold", color=INK, va="center")
        ax.text(x + 0.22, y + 0.95, _wrap_text(str(row["core_quantity_cn"]), 24), fontsize=10.8, weight="bold", color=color, va="center")
        ax.text(x + 0.22, y + 0.47, _wrap_text(str(row["detail_cn"]), 31), fontsize=9.8, color=MUTED, va="center", linespacing=1.15)
    _draw_arrow(ax, patches, (4.68, 4.72), (5.32, 4.72), lw=2.1, mutation_scale=16)
    _draw_arrow(ax, patches, (7.43, 3.72), (7.43, 2.98), lw=2.1, mutation_scale=16)
    _draw_arrow(ax, patches, (5.32, 1.97), (4.68, 1.97), lw=2.1, mutation_scale=16)
    output = next((row for row in frame.to_dict(orient="records") if int(row["step_order"]) == 4), {})
    output_items = [item for item in str(output.get("output_items_cn") or "").split(";") if item]
    ax.text(5.55, 0.55, "输出清单：" + "、".join(output_items), fontsize=10.6, color=INK, weight="bold")
    ax.set_title(_label(font, "运行服务流程", "Runtime Service Flow"), fontsize=15.5, weight="bold", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_service_flow",
        path=path,
        source_paths=_source_paths(sources, "runtime_service", "runtime_schema_contract"),
        evidence_layer="runtime_schema",
        table_path=table_paths["runtime_service_flow"],
        metric_definition="Report-facing 2x2 runtime flow drawn from runtime smoke summary and schema-contract summary.",
        replaces_problem="runtime_service_flow is redrawn as a 2x2 Chinese process diagram without internal runtime labels.",
        min_font_pt=9.8,
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
        (_label(font, "代表性窗口数", "selected windows"), str(len(frame)), GREEN),
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
    display_labels = [f"窗口{index + 1}" for index in range(len(frame))]
    ax_attr.set_xticks(x_values, display_labels, rotation=0)
    ax_attr.set_xlim(-0.45, max(len(frame) - 0.55, 0.55))
    if y_values:
        ax_attr.set_ylim(0, max(max(y_values) * 1.18, 1.0))
    ax_attr.set_ylabel(_label(font, "事件贡献强度（相对值）", "event contribution strength"))
    ax_attr.set_xlabel(_label(font, "代表性窗口", "selected window"))
    ax_attr.set_title(
        _label(font, "窗口、主导查询与事件贡献", "Window, dominant query, and event contribution"),
        fontsize=12,
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
            f"窗口{idx + 1}\n{_query_name_cn(row['semantic_top_query_name'])}",
            xy=(idx, float(row["semantic_top_event_attribution"])),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.6,
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
        (_label(font, "风险输出范围", "risk output range"), _range_text(frame["risk_proxy_confidence"], digits=3), BLUE),
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

    fig.suptitle(_label(font, "代表性窗口的任务输出与事件贡献", "Representative-window task outputs and event contributions"), fontsize=15.2, weight="bold", color=INK)
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
        min_font_pt=8.6,
    )


def _plot_rigid_body_rotation(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_rigid_body_rotation_rows(sources))
    family_rows = frame.loc[frame["row_type"] == "family_metric"].copy()
    matrix_rows = frame.loc[frame["row_type"] == "rotation_field_matrix"].copy()
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6))
    pivot = family_rows.pivot(index="family", columns="metric_name", values="metric_value")
    families = [name for name in ("minimal", "full", "rigid_body") if name in pivot.index]
    metrics = [
        ("test_total", "联合损失（对数刻度）", BLUE),
        ("test_physics_total", "物理残差损失（对数刻度）", PURPLE),
        ("test_alignment", "对齐损失（对数刻度）", GOLD),
    ]
    family_labels = {"minimal": "基础配置", "full": "完整物理配置", "rigid_body": "平移与垂向配置"}
    for axis, (metric, title, color) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], metrics, strict=True):
        values = [max(float(pivot.loc[family, metric]), 1e-12) for family in families]
        axis.bar(range(len(families)), values, color=color, edgecolor=INK, linewidth=0.4)
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(_sci_tick_formatter())
        axis.yaxis.set_minor_formatter(_null_tick_formatter())
        axis.set_xticks(range(len(families)), [family_labels.get(family, family) for family in families], rotation=0)
        axis.set_title(f"{title}\n越低越好", fontsize=12.0, weight="bold")
        axis.grid(axis="y", color=GRID, lw=0.8)
        axis.set_axisbelow(True)
        for i, value in enumerate(values):
            axis.text(i, value, _sci_label(value), ha="center", va="bottom", fontsize=9.2, color=INK)
    axes[0, 0].set_ylabel(_label(font, "对数刻度", "log scale"), fontsize=10.5)
    ax_matrix = axes[1, 1]
    ax_matrix.set_title(_label(font, "姿态角与角速度字段边界", "Rotation field availability"), fontsize=12.0, weight="bold")
    axis_labels = ["pitch", "roll", "yaw"]
    field_labels = ["angle", "rate"]
    matrix = []
    for axis in axis_labels:
        row_values = []
        for field_type in field_labels:
            selected = matrix_rows.loc[(matrix_rows["axis"] == axis) & (matrix_rows["field_type"] == field_type)]
            row_values.append(1 if not selected.empty and bool(selected["available"].iloc[0]) else 0)
        matrix.append(row_values)
    ax_matrix.imshow(matrix, cmap=_availability_cmap(), vmin=0, vmax=1, aspect="auto")
    ax_matrix.set_xticks(range(len(field_labels)), ["角度", "角速度"], fontsize=11)
    ax_matrix.set_yticks(range(len(axis_labels)), ["俯仰", "横滚", "航向"], fontsize=11)
    for y, axis in enumerate(axis_labels):
        for x, field_type in enumerate(field_labels):
            label = "可用" if matrix[y][x] else "字段边界"
            ax_matrix.text(x, y, label, ha="center", va="center", color=INK if matrix[y][x] else "#7f1d1d", fontsize=11.2, weight="bold")
    ax_matrix.text(
        0.5,
        -0.18,
        _label(font, "角速度字段未成对出现，因此旋转残差未启用；这是字段边界，不是实验失败。", "Rotation rates are missing as paired fields; rotation residual is not enabled."),
        ha="center",
        va="top",
        transform=ax_matrix.transAxes,
        fontsize=9.6,
        color=MUTED,
    )
    fig.suptitle(_label(font, "刚体配置（旋转残差未启用）", "Rigid-Body Configuration with Rotation Residual Disabled"), fontsize=15.5, weight="bold", color=INK)
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
        min_font_pt=9.2,
    )


def _plot_weak_label_sweep(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_weak_label_rows(sources))
    completed = frame.loc[frame["row_type"] == "completed_run"].copy()
    completed["test_total_numeric"] = pd.to_numeric(completed["test_total"], errors="coerce")
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    lag_order = ["none"] + sorted([value for value in completed["lag_label"].dropna().unique().tolist() if str(value) != "none"], key=lambda value: str(value))
    x = np.arange(len(lag_order))
    for sample_source, group in completed.groupby("sample_source"):
        values = []
        for lag in lag_order:
            selected = group.loc[group["lag_label"].astype(str) == str(lag)]
            values.append(float(selected["test_total_numeric"].min()) if not selected.empty else np.nan)
        color = GREEN if sample_source == "live_influx" else BLUE
        ax.plot(x, values, marker="o", color=color, lw=2.0, label=_sample_source_label(sample_source).replace("\n", ""))
        for x_idx, value in enumerate(values):
            if pd.notna(value):
                ax.text(x_idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9.2, color=INK)
    ax.set_xticks(x, ["无" if str(item) == "none" else str(item) for item in lag_order])
    ax.set_xlabel(_label(font, "时间滞后窗口", "Lag window"), fontsize=10.5)
    ax.set_ylabel(_label(font, "联合训练损失", "Joint training loss"), fontsize=10.5)
    ax.set_title(_label(font, "滞后窗口趋势", "Lag-window trend"), fontsize=12.0, weight="bold")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.6)

    ax_cov = axes[1]
    status_rows = _status_rows(frame)
    labels = [row["label"].replace("\n", "") for row in status_rows]
    completed_counts = [row["completed"] for row in status_rows]
    remaining_counts = [row["remaining"] for row in status_rows]
    totals = [max(c + r, 1) for c, r in zip(completed_counts, remaining_counts, strict=True)]
    coverage = [c / total for c, total in zip(completed_counts, totals, strict=True)]
    bars = ax_cov.bar(labels, coverage, color=[GREEN if "真实" in label else BLUE for label in labels], edgecolor=INK, linewidth=0.5)
    ax_cov.set_ylim(0, 1.08)
    ax_cov.yaxis.set_major_formatter(_percent_tick_formatter())
    ax_cov.set_ylabel(_label(font, "任务记录覆盖率", "Task-record coverage"), fontsize=10.5)
    ax_cov.set_title(_label(font, "已完成参数组合覆盖", "Completed sweep coverage"), fontsize=12.0, weight="bold")
    ax_cov.grid(axis="y", color=GRID, lw=0.8)
    ax_cov.set_axisbelow(True)
    for bar, count, total in zip(bars, completed_counts, totals, strict=True):
        ax_cov.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{count}/{total}", ha="center", fontsize=9.4, color=INK)
    ax_cov.text(
        0.5,
        -0.25,
        _label(font, "当前结构化网格未保存风险阈值、负荷阈值变化；本图仅展示已落盘的滞后窗口和续跑边界。", "Risk/workload thresholds are not stored in this grid."),
        transform=ax_cov.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
        linespacing=1.15,
    )
    fig.suptitle(_label(font, "弱监督参数 sweep：滞后窗口与记录覆盖", "Weak-label sweep: lag window and coverage"), fontsize=15, weight="bold", color=INK)
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
        replaces_problem="weak-label sweep uses a lag-window trend plus task-record coverage instead of a repeated heatmap.",
        min_font_pt=9.2,
    )


def _plot_private_component(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_private_component_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig = plt.figure(figsize=(14.2, 8.3))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.48, 1.2], hspace=0.26)
    tasks = [
        ("T1_maneuver_intensity_class", "macro_f1", "宏平均F1", "越高越好", BLUE),
        ("T2_next_window_physiology_response", "rmse", "RMSE", "越低越好", ORANGE),
        ("T3_paired_pilot_window_retrieval", "top1_accuracy", "Top-1准确率", "越高越好", GREEN),
    ]
    ax_cards = fig.add_subplot(grid[0])
    ax_cards.set_axis_off()
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    full_candidate_note = False
    card_rows = []
    for task, metric_name, metric_cn, direction, color in tasks:
        task_rows = frame.loc[(frame["task_name"] == task) & (frame["primary_metric_name"] == metric_name)].copy()
        full_rows = task_rows.loc[task_rows["variant_role"] == "full_candidate"].copy()
        if full_rows.empty:
            full_rows = task_rows.head(1)
        full_candidate_note = full_candidate_note or (
            len(full_rows) > 1 and full_rows["primary_metric_value"].nunique(dropna=False) == 1
        )
        preferred_rows = full_rows.loc[full_rows["variant_name"] == "full_model"]
        baseline_row = preferred_rows.iloc[0] if not preferred_rows.empty else full_rows.iloc[0]
        card_rows.append(
            {
                "task": _task_label_cn(task),
                "metric": f"{metric_cn}（{direction}）",
                "value": _fmt_metric(float(baseline_row["primary_metric_value"]), digits=3),
                "protocol": str(baseline_row.get("report_protocol_cn") or "严格评价协议"),
                "seed_count": baseline_row.get("seed_count"),
                "color": color,
            }
        )
    for idx, card in enumerate(card_rows):
        x = 0.02 + idx * 0.326
        y = 0.18
        w = 0.294
        h = 0.74
        color = str(card["color"])
        ax_cards.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor="#ffffff",
                edgecolor=GRID,
                lw=1.0,
            )
        )
        ax_cards.add_patch(patches.Rectangle((x, y + h - 0.065), w, 0.065, facecolor=color, edgecolor="none"))
        task_title = str(card["task"]).replace("\n", "：")
        ax_cards.text(x + 0.026, y + h - 0.14, task_title, fontsize=11.0, weight="bold", color=INK, va="top")
        ax_cards.text(x + 0.026, y + h - 0.29, str(card["metric"]), fontsize=9.4, color=MUTED, va="top")
        ax_cards.text(x + 0.026, y + 0.21, str(card["value"]), fontsize=17.0, weight="bold", color=color, va="bottom")
        seed_count = card.get("seed_count")
        seed_text = f"{int(seed_count)} seeds" if pd.notna(seed_count) else "seed count NA"
        ax_cards.text(
            x + w - 0.026,
            y + 0.22,
            _label(font, f"{card['protocol']} / {seed_text}", f"{card['protocol']} / {seed_text}"),
            fontsize=8.5,
            color=MUTED,
            ha="right",
            va="bottom",
        )
    if full_candidate_note:
        ax_cards.text(
            0.02,
            0.03,
            _label(
                font,
                "注：CSV 中“完整任务输入”和“完整方案”同属 chronaris_opt / full_safe 基线，本图只保留一次基线读数；组件差异看下方相对变化热图。",
                "Note: full task input and full model are the same chronaris_opt / full_safe baseline; only one baseline readout is kept here.",
            ),
            fontsize=8.8,
            color=MUTED,
            va="bottom",
        )
    ax_heat = fig.add_subplot(grid[1])
    _draw_component_overview_heatmap(ax_heat, frame, font)
    fig.suptitle(_label(font, "组件消融总览：完整基线与相对变化", "Component ablation overview"), fontsize=15, weight="bold", color=INK)
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.08, right=0.94, hspace=0.26)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="chronaris_opt_component_ablation",
        path=path,
        source_paths=_source_paths(sources, "private_component"),
        evidence_layer="private_proxy",
        table_path=table_paths["chronaris_opt_component_ablation"],
        metric_definition="Task absolute metrics are shown on native scales; relative component changes are centered at zero with positive values meaning worse performance after removal.",
        recommended_placement="supporting_overview",
        replaces_problem="redundant full-candidate bars are replaced by baseline metric cards and a compact signed relative-change overview.",
        min_font_pt=8.8,
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
        evidence_layer="private_proxy_strict_protocol",
        table_path=table_paths["model_backbone_ablation"],
        metric_definition="Strict-protocol model-backbone ablation with signed relative performance change by task.",
        recommended_placement="main_experiment_component_analysis",
        replaces_problem="model_backbone_ablation uses full task names, full model first, and a zero-centered diverging heatmap.",
        min_font_pt=8.5,
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
        evidence_layer="private_proxy_strict_protocol",
        table_path=table_paths["task_adapter_ablation"],
        metric_definition="Strict-protocol task-adapter ablation with signed relative performance change by task.",
        recommended_placement="main_experiment_component_analysis",
        replaces_problem="task_adapter_ablation uses readable task-input labels and a zero-centered diverging heatmap.",
        min_font_pt=8.5,
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
        _label(font, "公开数据用于适配校准，鼎新真实航空双流用于验证闭环，组件诊断任务用于结构敏感性测试。", "Public data supports adaptation, Dingxin dual-stream data supports validation, and component-diagnostic tasks support structural sensitivity tests."),
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
        metric_definition="Chinese positive role diagram separating public adapters, Dingxin weak-label mainline evidence, and Dingxin component diagnostics.",
        replaces_problem="defensive boundary copy is replaced by positive Chinese role wording.",
    )


def _plot_semantic_event_fusion(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_semantic_event_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig = plt.figure(figsize=(13.8, 7.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.86, 1.25])
    ax_flow = fig.add_subplot(grid[0])
    ax_flow.set_axis_off()
    ax_flow.set_xlim(0, 12)
    ax_flow.set_ylim(0, 2.7)
    steps_cn = ["人机双流输入", "因果掩码", "事件表示", "语义查询", "贡献对齐", "任务输出"]
    steps_en = ["dual-stream input", "causal mask", "event representation", "semantic queries", "contribution alignment", "task outputs"]
    x_positions = [0.25, 2.15, 4.04, 5.93, 7.82, 9.72]
    widths = [1.45, 1.38, 1.38, 1.42, 1.42, 1.88]
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
        ax_flow.text(x + width / 2, 1.41, _wrap_text(_label(font, steps_cn[idx], steps_en[idx]), 12), ha="center", va="center", fontsize=10.0, color=INK, weight="bold" if idx in {0, 5} else "normal")
        if idx < len(x_positions) - 1:
            _draw_arrow(ax_flow, patches, (x + width + 0.1, 1.41), (x_positions[idx + 1] - 0.1, 1.41), lw=1.9, mutation_scale=15)
    ax_flow.set_title(_label(font, "语义查询与事件贡献关系", "Semantic query and event-contribution relation"), fontsize=13, weight="bold", color=INK)
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
        view_rows = frame.loc[frame["row_type"] == "view_attribution"].copy()
        ax.set_axis_on()
        query_names = []
        for value in frame["query_names"].dropna().astype(str).tolist():
            query_names.extend([part for part in value.split(";") if part])
        query_order = [name for name in ("risk_proxy", "workload_proxy", "event_replay_tag") if name in set(query_names)]
        if not query_order:
            query_order = ["risk_proxy", "workload_proxy", "event_replay_tag"]
        view_labels = [f"视图{idx + 1}" for idx in range(len(view_rows))]
        support = np.ones((len(view_labels), len(query_order)), dtype=float)
        ax.imshow(support, cmap=_coverage_cmap(), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(query_order)), [_query_name_cn(name) for name in query_order], fontsize=11)
        ax.set_yticks(range(len(view_labels)), view_labels, fontsize=11)
        for y, row in enumerate(view_rows.to_dict(orient="records")):
            for x, query in enumerate(query_order):
                ax.text(x, y, "支撑", ha="center", va="center", fontsize=11.0, weight="bold", color=INK)
            ax.text(
                len(query_order) - 0.02,
                y,
                f"  样本{int(row.get('sample_count') or 0)}",
                ha="left",
                va="center",
                fontsize=9.6,
                color=MUTED,
            )
        ax.set_xlim(-0.5, len(query_order) + 0.9)
        ax.set_title(_label(font, "3个人机双流数据视图 × 3类查询覆盖状态", "3 dual-stream views by 3 query categories"), fontsize=12, weight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        requirement = ""
        missing = frame.loc[frame["row_type"] == "source_requirement"]
        if not missing.empty:
            requirement = str(missing.iloc[0].get("missing_source_data_requirement") or "")
        ax.text(
            0.5,
            -0.19,
            _label(font, "未保存完整归因数值矩阵，因此只展示覆盖/支撑状态：" + requirement, "Complete attribution scores are unavailable, so only coverage status is shown."),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.4,
            color=MUTED,
            linespacing=1.14,
        )
    fig.suptitle(_label(font, "语义事件融合总览", "Semantic Event Fusion Overview"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="semantic_event_fusion_overview",
        path=path,
        source_paths=_source_paths(sources, "support", "semantic_event"),
        evidence_layer="semantic_support",
        table_path=table_paths["semantic_event_fusion_overview"],
        case_definition="Pipeline diagram plus view-query heatmap when complete source scores exist; otherwise coverage/support status is drawn without fabricated attribution values.",
        replaces_problem="semantic fusion overview is redrawn as a report-readable flow and coverage/status matrix.",
        min_font_pt=9.4,
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
        "A0_baseline": "基础查询",
        "A1_llm_context": "语义上下文接入",
        "A2_llm_semantic_hints": "查询建议",
        "A3_llm_runtime_explanation": "案例解释",
        "A4_human_review_packet": "复核材料",
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
            fontsize=9.8,
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
            fontsize=8.9,
            color=INK,
            linespacing=1.15,
        )
        ax.text(x + card_w / 2, card_y + 1.12, _llm_metric_name_cn(row["metric_name"]), ha="center", fontsize=8.5, color=MUTED)
        ax.text(x + card_w / 2, card_y + 0.78, _short(row["metric_value"], 16), ha="center", fontsize=14.6, weight="bold", color=INK)
        ax.text(
            x + card_w / 2,
            card_y + 0.31,
            _wrap_text(_llm_card_note(font, row), 22),
            ha="center",
            va="center",
            fontsize=8.6,
            color=MUTED,
            linespacing=1.12,
        )
        if idx < len(x_positions) - 1:
            _draw_arrow(ax, patches, (x + card_w + 0.18, mid_y), (x_positions[idx + 1] - 0.18, mid_y), lw=1.8, mutation_scale=14)
    pre_row = frame.loc[frame["condition"].str.endswith("preprocessing_run")].iloc[0].to_dict()
    review_row = frame.loc[frame["condition"] == "A4_human_review_packet"].iloc[0].to_dict()
    ax.text(
        0.5,
        0.76,
        _llm_preprocessing_note_cn(pre_row, review_item_count=review_row.get("metric_value")),
        fontsize=9.6,
        color=INK,
        weight="bold",
    )
    ax.set_title(_label(font, "大语言模型辅助数据预处理与复核流程", "LLM-assisted data preprocessing and review flow"), fontsize=15, weight="bold", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="llm_comparison_a0_a4",
        path=path,
        source_paths=_source_paths(sources, "llm_preprocessing", "llm_comparison"),
        evidence_layer="llm_preprocessing_comparison",
        table_path=table_paths["llm_comparison_a0_a4"],
        metric_definition="LLM preprocessing comparison from existing summary; hints show rule-checked coverage only, not attribution improvement.",
        replaces_problem="adds LLM preprocessing/comparison evidence without treating LLM output as truth.",
        min_font_pt=8.6,
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
    min_font_pt: float = 8.5,
) -> dict[str, object]:
    payload = {
        "figure_id": figure_id,
        "path": str(path),
        "exists": path.exists(),
        "source_reference": [_source_reference_label(path) for path in source_paths if path],
        "evidence_layer": evidence_layer,
        "table_path": table_path,
        "visible_label_language": "zh_cn_or_report_abbrev",
        "min_font_pt": min_font_pt,
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


def _source_reference_label(path_like: str) -> str:
    path = str(path_like)
    mapping = {
        "task_eval_evidence": "evidence manifest",
        "task_eval_multitask_sweep": "weak-label sweep summary",
        "task_eval_private_leakage_safe_ablation": "private component ablation summary",
        "task_eval_public_adapter_calibration": "public adapter calibration summary",
        "task_eval_public_transfer_boundary": "public transfer boundary summary",
        "task_eval_runtime_service": "runtime service summary and field contract",
        "task_eval_runtime_inference": "runtime replay summary",
        "task_eval_support": "semantic support summary",
        "task_eval_semantic_event_support": "semantic event support summary",
        "task_eval_rotation_audit": "rotation audit summary",
        "task_eval_rigid_body": "rigid-body ablation summary",
        "task_eval_llm_preprocessing": "LLM preprocessing summary",
        "task_eval_llm_comparison": "LLM comparison summary",
        "task_eval_thesis_figures": "runtime semantic case table",
    }
    for key, label in mapping.items():
        if key in path:
            return label
    return Path(path).name or "source artifact"


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


def _sci_tick_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _position: _sci_label(value))


def _percent_tick_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _position: f"{value:.0%}")


def _signed_percent_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _position: f"{value:+.0f}%")


def _null_tick_formatter():
    from matplotlib.ticker import NullFormatter

    return NullFormatter()


def _availability_cmap():
    from matplotlib import colors

    return colors.ListedColormap(["#fde8e2", "#e6f4ea"])


def _coverage_cmap():
    from matplotlib import colors

    return colors.ListedColormap(["#f8fafc", "#e7f3ee"])


def _sci_label(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(number))))
    mantissa = number / (10 ** exponent)
    if -2 <= exponent <= 2:
        return f"{number:.3g}"
    return f"{mantissa:.2f}×10^{exponent}"


def _heatmap_text_color(value: float, limit: float) -> str:
    if limit <= 0:
        return INK
    return "white" if abs(value) / limit >= 0.55 else INK


def _variant_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    if text in {"full_model", "chronaris_opt", "full_leakage_safe_task_input"}:
        return (0, text)
    if text.startswith("remove_") or text.startswith("chronaris_opt_no_"):
        return (2, text)
    if text.startswith("only_") or text == "single_modality_only":
        return (3, text)
    return (1, text)


def _group_label_cn(value: object) -> str:
    mapping = {
        "model_backbone": "模型骨干",
        "task_adapter": "任务输入",
    }
    return mapping.get(str(value), str(value))


def _evidence_card_text(row: Mapping[str, object], font: PlotFontSelection) -> tuple[str, str, str]:
    layer = str(row.get("evidence_layer") or "")
    headline = _label(font, str(row.get("layer_title_cn")), str(row.get("layer_title")))
    if layer == "thesis_weak_label":
        quantity = f"{row.get('primary_metric_value')}个窗口\n{row.get('secondary_metric_value')}条任务记录"
    elif layer == "runtime_schema":
        quantity = f"{row.get('primary_metric_value')}个回放窗口\n{row.get('secondary_metric_value')}个字段差异已校验"
    elif layer == "llm_preprocessing":
        review_count = _extract_first_int(row.get("key_status"))
        quantity = f"{review_count}条人工复核材料\nLLM仅作预处理" if review_count is not None else str(row.get("key_status"))
    else:
        quantity = (
            f"{_metric_name_cn(row.get('primary_metric_name'))}：{row.get('primary_metric_value')}\n"
            f"{_metric_name_cn(row.get('secondary_metric_name'))}：{row.get('secondary_metric_value')}"
        )
    role = _evidence_card_role(row, font)
    return headline, quantity, role


def _evidence_card_role(row: Mapping[str, object], font: PlotFontSelection) -> str:
    layer = str(row.get("evidence_layer") or "")
    fallback = str(row.get("boundary_cn") or row.get("key_status") or "")
    compact_cn = {
        "thesis_weak_label": "论文任务原型与参数比较；后续接专家复核。",
        "private_proxy": "定位模型骨干、任务适配层和严格评价协议影响。",
        "public_adapter_calibration": "支撑公开数据接口、校准基线和外部任务对照。",
        "runtime_schema": "原始回放已对齐；统一契约输入通过校验。",
        "semantic_support": "展示事件表示、语义查询和归因对齐支撑。",
        "rigid_body": "平移/垂向已入训；旋转保留字段诊断。",
        "llm_preprocessing": "字段语义、查询建议、案例解释与复核材料。",
    }
    compact_ascii = {
        "thesis_weak_label": "Prototype and parameter evidence; expert review remains separate.",
        "private_proxy": "Locates Dingxin component diagnostics, backbone, task input, and strict protocol effects.",
        "public_adapter_calibration": "Supports public adapters, calibration baselines, and external task checks.",
        "runtime_schema": "Replay inputs are aligned; canonical contract passes validation.",
        "semantic_support": "Shows event representation, semantic query, and attribution support.",
        "rigid_body": "Translation and vertical residuals are active; rotation remains diagnostic.",
        "llm_preprocessing": "Field semantics, query advice, case explanation, and review packets.",
    }
    return _label(font, compact_cn.get(layer, fallback), compact_ascii.get(layer, fallback))


def _extract_first_int(value: object) -> int | None:
    import re

    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


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
        ax.text(0.5, 0.5, _label(font, "暂无严格评价协议消融行", "No strict-protocol ablation rows"), ha="center", va="center", fontsize=11, color=MUTED)
        return
    from matplotlib import colors

    working = frame.copy()
    if "relative_delta_percent" not in working or working["relative_delta_percent"].isna().all():
        working["relative_delta_percent"] = working["normalized_delta_vs_full"].fillna(0.0).astype(float) * 100.0
    tasks = [task for task in ("T1_maneuver_intensity_class", "T2_next_window_physiology_response", "T3_paired_pilot_window_retrieval") if task in set(working["task_name"])]
    working["variant_label"] = working["display_variant_cn"].fillna(working["display_variant"]).astype(str)
    ordered_rows = sorted(
        working[["variant_name", "variant_label"]].drop_duplicates().to_dict(orient="records"),
        key=lambda row: _variant_sort_key(row["variant_name"]),
    )
    variants = [str(row["variant_label"]) for row in ordered_rows]
    heat = np.full((len(variants), len(tasks)), np.nan, dtype=float)
    labels = [["" for _ in tasks] for _ in variants]
    for row in working.to_dict(orient="records"):
        if row["task_name"] not in tasks:
            continue
        y = variants.index(str(row.get("variant_label")))
        x = tasks.index(row["task_name"])
        relative = float(row.get("relative_delta_percent") or 0.0)
        delta = float(row.get("delta_vs_full") or 0.0)
        heat[y, x] = relative
        labels[y][x] = f"{delta:+.3g}\n{relative:+.1f}%"
    ax.set_axis_on()
    finite_values = [abs(float(value)) for value in heat.ravel() if pd.notna(value)]
    limit = max(max(finite_values), 1.0) if finite_values else 1.0
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im = ax.imshow(np.nan_to_num(heat, nan=0.0), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(tasks)), [_task_label_cn(task) for task in tasks], fontsize=9.4)
    ax.set_yticks(range(len(variants)), variants, fontsize=8.8)
    for y in range(len(variants)):
        for x in range(len(tasks)):
            value = heat[y, x]
            text_color = _heatmap_text_color(float(value), limit) if pd.notna(value) else INK
            ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=8.5, color=text_color, weight="bold")
    ax.set_title(_label(font, "移除组件后的性能变化（正值表示变差）", "Performance change after component removal"), fontsize=11.3, weight="bold")
    zero_tasks = [
        _task_label_cn(task)
        for x, task in enumerate(tasks)
        if np.nanmax(np.abs(heat[:, x])) <= 1e-9
    ]
    if zero_tasks:
        ax.text(
            0.5,
            -0.16,
            "；".join(zero_tasks) + "：当前配置下无区分度",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            color=MUTED,
        )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.035, label=_label(font, "性能变化百分比", "performance change percent"))
    cbar.ax.yaxis.set_major_formatter(_signed_percent_formatter())
    cbar.ax.yaxis.set_minor_formatter(_null_tick_formatter())


def _draw_component_overview_heatmap(ax, frame: pd.DataFrame, font: PlotFontSelection) -> None:
    if frame.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, _label(font, "暂无组件消融行", "No ablation rows"), ha="center", va="center", fontsize=11, color=MUTED)
        return
    from matplotlib import colors

    working = frame.copy()
    working["relative_delta_percent"] = pd.to_numeric(working.get("relative_delta_percent"), errors="coerce").fillna(0.0)
    working = working.loc[working["variant_role"] != "full_candidate"].copy()
    tasks = [task for task in ("T1_maneuver_intensity_class", "T2_next_window_physiology_response", "T3_paired_pilot_window_retrieval") if task in set(frame["task_name"])]
    groups = [group for group in ("model_backbone", "task_adapter") if group in set(working["ablation_group"])]
    group_labels = [_group_label_cn(group) for group in groups]
    heat = np.zeros((len(groups), len(tasks)), dtype=float)
    labels = [["" for _ in tasks] for _ in groups]
    for y, group in enumerate(groups):
        group_rows = working.loc[working["ablation_group"] == group]
        for x, task in enumerate(tasks):
            values = group_rows.loc[group_rows["task_name"] == task, "relative_delta_percent"].astype(float)
            value = float(values.mean()) if not values.empty else 0.0
            heat[y, x] = value
            labels[y][x] = f"{value:+.1f}%"
    limit = max([abs(float(value)) for value in heat.ravel()] + [1.0])
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im = ax.imshow(heat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(tasks)), [_task_label_cn(task) for task in tasks], fontsize=10.0)
    ax.set_yticks(range(len(group_labels)), group_labels, fontsize=10.0)
    for y in range(len(groups)):
        for x in range(len(tasks)):
            ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=10.0, color=_heatmap_text_color(heat[y, x], limit), weight="bold")
    ax.set_title(_label(font, "组件类别平均性能变化（正值表示移除后变差）", "Average component-group performance change"), fontsize=11.8, weight="bold")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label=_label(font, "平均变化百分比", "average change percent"))
    cbar.ax.yaxis.set_major_formatter(_signed_percent_formatter())
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
        "T1_maneuver_intensity_class": "分类任务\n机动强度",
        "T2_next_window_physiology_response": "回归任务\n生理响应",
        "T3_paired_pilot_window_retrieval": "检索任务\n配对窗口",
    }
    return mapping.get(str(value), str(value))


def _sample_source_label(value: object) -> str:
    text = str(value)
    if text == "live_influx":
        return "真实时序\n数据读取"
    if text == "feature_export_window_stats_proxy":
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
        return _label(font, "经规则校验：3→7", metric_note.replace("_", " "))
    if condition.startswith("A3"):
        return _label(font, "4个案例完整解释", metric_note.replace("_", " "))
    if condition.startswith("A4"):
        return _label(font, "待人工复核", metric_note.replace("_", " "))
    return metric_note.replace("_", " ")


def _llm_stage_cn(row: Mapping[str, object]) -> str:
    condition = str(row.get("condition", ""))
    mapping = {
        "A0_baseline": "基础查询集合",
        "A1_llm_context": "字段与窗口语义上下文",
        "A2_llm_semantic_hints": "经规则校验的查询建议",
        "A3_llm_runtime_explanation": "选取案例解释",
        "A4_human_review_packet": "结构化复核材料",
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


def _llm_preprocessing_note_cn(row: Mapping[str, object], *, review_item_count: object = None) -> str:
    note = str(row.get("metric_note") or "")
    parts: dict[str, str] = {}
    for item in note.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip()] = value.strip()
    return (
        f"{parts.get('field_semantics', 'NA')}条字段语义说明；"
        f"{parts.get('weak_label_reviews', 'NA')}条弱监督规则复核意见；"
        f"{parts.get('runtime_explanations', 'NA')}个选取案例已形成完整解释；"
        f"{review_item_count if review_item_count is not None else 'NA'}条复核材料待人工复核"
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


def _range_text(series: pd.Series, *, digits: int = 3) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    low = float(values.min())
    high = float(values.max())
    if abs(high - low) < 1e-9:
        return _fmt_metric(low, digits=digits)
    return f"{_fmt_metric(low, digits=digits)} - {_fmt_metric(high, digits=digits)}"


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


def _wrap_text_limited(text: str, width: int, max_lines: int) -> str:
    value = str(text or "")
    lines: list[str] = []
    for part in value.splitlines() or [""]:
        if not part:
            lines.append("")
        elif " " in part:
            lines.extend(wrap(part, width=width, break_long_words=False, break_on_hyphens=False))
        else:
            lines.extend(part[index : index + width] for index in range(0, len(part), width))
    if len(lines) <= max_lines:
        return "\n".join(lines)
    clipped = lines[:max_lines]
    clipped[-1] = _truncate_text_line(clipped[-1], width)
    return "\n".join(clipped)


def _truncate_text_line(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return "." * width
    if " " in text:
        return shorten(text, width=width, placeholder="...")
    return f"{text[: width - 3]}..."


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
