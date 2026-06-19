"""Plotting helpers for Stage I thesis-facing materials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten, wrap
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.evidence.thesis_materials_data import (
    build_evidence_layer_rows,
    build_llm_comparison_rows,
    build_private_component_rows,
    build_public_transfer_rows,
    build_rigid_body_rotation_rows,
    build_runtime_payload_schema_rows,
    build_runtime_semantic_case_rows,
    build_semantic_event_rows,
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
        _plot_public_transfer(run_root / "public_transfer_boundary.png", sources, font, table_paths),
        _plot_semantic_event_fusion(run_root / "semantic_event_fusion_overview.png", sources, font, table_paths),
        _plot_llm_comparison(run_root / "llm_comparison_a0_a4.png", sources, font, table_paths),
    ]


def _plot_evidence_layer_overview(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_evidence_layer_rows(sources)).sort_values("display_order")
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, len(frame) + 1.1)
    headers = [
        _label(font, "证据层级", "Evidence layer"),
        _label(font, "主计数", "Primary count"),
        _label(font, "辅助计数", "Secondary count"),
        _label(font, "边界与状态", "Boundary and status"),
    ]
    x_positions = [0.2, 3.25, 5.35, 7.45]
    widths = [2.8, 1.85, 1.85, 4.2]
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
            y + 0.12,
            _label(font, str(row["layer_title_cn"]), str(row["layer_title"])),
            fontsize=10.2,
            weight="bold",
            color=INK,
            va="center",
        )
        ax.text(0.38, y - 0.16, _short(row.get("source_run_id"), 42), fontsize=7.5, color=MUTED, va="center")
        _metric_cell(ax, 3.25, y, row["primary_metric_name"], row["primary_metric_value"], color)
        _metric_cell(ax, 5.35, y, row["secondary_metric_name"], row["secondary_metric_value"], color)
        ax.text(7.5, y + 0.12, _short(row["key_status"], 68), fontsize=8.4, color=INK, va="center")
        ax.text(
            7.5,
            y - 0.17,
            _short(_label(font, str(row["boundary_cn"]), str(row["boundary_cn"])), 72),
            fontsize=7.4,
            color=MUTED,
            va="center",
        )
    ax.set_title(_label(font, "Stage I 中期证据层级矩阵", "Stage I Evidence Layer Matrix"), fontsize=16, weight="bold", color=INK, pad=16)
    ax.text(
        0.15,
        0.02,
        _label(font, "所有数值来自现有 artifact JSON/CSV；不同证据层不合并为同一结论。", "All values are parsed from existing artifact JSON/CSV; evidence layers remain separate."),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="evidence_layer_overview",
        path=path,
        source_paths=_source_paths(sources, "evidence_manifest", "live_sweep", "private_component", "public_calibration", "runtime_schema_contract", "support", "rotation_audit", "llm_comparison"),
        evidence_layer="cross_layer_index",
        table_path=table_paths["evidence_layer_overview"],
        metric_definition="Evidence-layer matrix with source-derived counts/statuses; replaces artifact-present bars.",
        replaces_problem="evidence_layer_overview no longer uses all-one artifact-present bars.",
    )


def _plot_runtime_payload_schema(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_runtime_payload_schema_rows(sources))
    plt, patches = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
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
        metric_rows = [
            ("physiology", row["physiology_feature_count"]),
            ("vehicle", row["vehicle_feature_count"]),
            ("schema status", row["schema_status"]),
            ("missing vehicle", row["missing_vehicle_feature_count"]),
        ]
        for offset, (name, value) in enumerate(metric_rows):
            yy = y + h - 1.12 - offset * 0.46
            ax.plot([x + 0.18, x + w - 0.18], [yy - 0.22, yy - 0.22], color=GRID, lw=0.8)
            ax.text(x + 0.24, yy, name, fontsize=9.3, color=MUTED, va="center")
            ax.text(x + w - 0.24, yy, str(value), fontsize=11.2, color=color if name == "schema status" else INK, weight="bold", ha="right", va="center")
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
        _label(font, "服务层契约化", "service contract"),
        ha="center",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )
    ax.text(
        6.0,
        1.72,
        _label(font, "canonical exact 已闭合服务契约\nnative exact 仍需补齐 vehicle groups", "canonical exact closes service schema\nnative exact still needs vehicle groups"),
        ha="center",
        fontsize=8.4,
        color=MUTED,
    )
    ax.set_title(_label(font, "运行时 payload 字段契约对照", "Runtime Payload Schema Contract"), fontsize=15, weight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_payload_schema",
        path=path,
        source_paths=_source_paths(sources, "runtime_service", "runtime_schema_contract"),
        evidence_layer="runtime_schema",
        table_path=table_paths["runtime_payload_schema"],
        metric_definition="Native replay payload remains aligned; canonical service payload reaches exact schema.",
        replaces_problem="runtime_payload_schema is rendered as a contract comparison instead of a generic schema plot.",
    )


def _plot_runtime_semantic_case(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_runtime_semantic_case_rows(sources))
    if frame.empty:
        raise ValueError("runtime_semantic_case requires a non-empty runtime_case_table source")
    numeric_columns = [
        "window_order",
        "semantic_top_event_attribution",
        "top_contribution_score",
        "risk_proxy_confidence",
        "workload_proxy_prediction",
        "event_replay_tag_score",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("window_order").reset_index(drop=True)
    case = frame.iloc[0].to_dict()
    query_names = frame["semantic_top_query_name"].fillna("unknown").astype(str)
    point_colors = [_query_color(name) for name in query_names]
    plt, patches = _import_matplotlib(font)
    fig = plt.figure(figsize=(14.8, 8.8))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.25, 2.45, 1.25], hspace=0.52, wspace=0.36)

    ax_cards = fig.add_subplot(grid[0, :])
    ax_cards.set_axis_off()
    ax_cards.set_xlim(0, 12)
    ax_cards.set_ylim(0, 2)
    ax_cards.add_patch(
        patches.FancyBboxPatch(
            (0.22, 1.05),
            11.45,
            0.72,
            boxstyle="round,pad=0.05,rounding_size=0.06",
            facecolor="#fbfdff",
            edgecolor=BLUE,
            lw=1.35,
        )
    )
    ax_cards.text(0.42, 1.58, _label(font, "view_id / 架次", "view_id / sortie"), fontsize=8.2, color=MUTED, va="center")
    ax_cards.text(
        0.42,
        1.29,
        _runtime_case_view_label(case.get("view_id")),
        fontsize=9.8,
        weight="bold",
        color=INK,
        va="center",
        linespacing=1.0,
    )
    cards = [
        (_label(font, "展示窗口", "windows"), str(len(frame)), GREEN),
        (
            _label(font, "native / canonical", "native / canonical"),
            f"{case.get('native_feature_schema_status')} / {case.get('canonical_feature_schema_status')}",
            PURPLE,
        ),
        (
            _label(font, "vehicle 字段", "vehicle fields"),
            f"{_fmt_metric(case.get('input_vehicle_feature_count'), 0)} -> {_fmt_metric(case.get('expected_vehicle_feature_count'), 0)}",
            ORANGE,
        ),
        (
            _label(font, "缺失 groups", "missing groups"),
            _fmt_metric(case.get("native_missing_measurement_group_count"), 0),
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
        ax_cards.text(x + 0.16, 0.32, value, fontsize=10.8, weight="bold", color=INK, va="center")

    ax_attr = fig.add_subplot(grid[1, :3])
    x_values = frame["window_order"].fillna(0).astype(int).tolist()
    y_values = frame["semantic_top_event_attribution"].fillna(0.0).astype(float).tolist()
    for x_value, y_value, color in zip(x_values, y_values, point_colors, strict=True):
        ax_attr.vlines(x_value, 0, y_value, color=color, alpha=0.35, lw=2.2)
    ax_attr.scatter(x_values, y_values, s=74, c=point_colors, edgecolor=INK, linewidth=0.6, zorder=4)
    ax_attr.set_xticks(x_values, frame["window_label"].astype(str).tolist(), rotation=0)
    ax_attr.set_ylabel("semantic_top_event_attribution")
    ax_attr.set_xlabel(_label(font, "runtime replay 窗口", "runtime replay window"))
    ax_attr.set_title(
        _label(font, "窗口级语义归因：只标注变化点与最高值", "Window-level semantic attribution: changes and peak only"),
        fontsize=11,
        weight="bold",
    )
    ax_attr.grid(axis="y", color=GRID, lw=0.8)
    ax_attr.set_axisbelow(True)
    legend_names = list(dict.fromkeys(query_names.tolist()))
    if len(legend_names) > 1:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=_query_color(name), markeredgecolor=INK, markersize=7)
            for name in legend_names
        ]
        ax_attr.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    annotation_indices = _runtime_annotation_indices(frame)
    for idx in annotation_indices:
        row = frame.iloc[idx]
        ax_attr.annotate(
            f"{row['window_label']}\n{row['semantic_top_query_name']}",
            xy=(int(row["window_order"]), float(row["semantic_top_event_attribution"])),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )

    ax_dist = fig.add_subplot(grid[1, 3])
    counts = query_names.value_counts().sort_values()
    ax_dist.barh(counts.index.tolist(), counts.values.tolist(), color=[_query_color(name) for name in counts.index], edgecolor=INK, linewidth=0.4)
    ax_dist.set_title(_label(font, "query 类型分布", "Query distribution"), fontsize=11, weight="bold")
    ax_dist.set_xlabel(_label(font, "窗口数", "windows"))
    ax_dist.grid(axis="x", color=GRID, lw=0.8)
    ax_dist.set_axisbelow(True)
    for y, value in enumerate(counts.values.tolist()):
        ax_dist.text(value, y, f" {value}", va="center", fontsize=8.2, color=INK)

    ax_ranges = fig.add_subplot(grid[2, :2])
    ax_ranges.set_axis_off()
    ax_ranges.set_xlim(0, 6)
    ax_ranges.set_ylim(-0.12, 1.8)
    range_cards = [
        ("risk_proxy_confidence", _range_text(frame["risk_proxy_confidence"]), BLUE),
        ("workload_proxy_prediction", _range_text(frame["workload_proxy_prediction"]), GREEN),
        ("event_replay_tag_score", _range_text(frame["event_replay_tag_score"]), PURPLE),
    ]
    for idx, (name, value, color) in enumerate(range_cards):
        y = 1.24 - idx * 0.52
        ax_ranges.add_patch(
            patches.FancyBboxPatch(
                (0.08, y - 0.23),
                5.55,
                0.39,
                boxstyle="round,pad=0.035,rounding_size=0.045",
                facecolor=PALE,
                edgecolor=GRID,
                lw=0.9,
            )
        )
        ax_ranges.text(0.25, y, name, fontsize=8.4, color=MUTED, va="center")
        ax_ranges.text(5.35, y, value, fontsize=10.5, color=color, weight="bold", ha="right", va="center")
    ax_ranges.set_title(_label(font, "近乎平稳序列改用范围值", "Near-flat series shown as ranges"), fontsize=10.8, weight="bold", color=INK)

    ax_schema = fig.add_subplot(grid[2, 2:])
    ax_schema.set_axis_off()
    ax_schema.set_xlim(0, 6)
    ax_schema.set_ylim(-0.12, 1.8)
    schema_lines = [
        (
            _label(font, "原生 replay payload", "native replay payload"),
            f"{case.get('native_feature_schema_status')} | vehicle={_fmt_metric(case.get('input_vehicle_feature_count'), 0)}",
            ORANGE,
        ),
        (
            _label(font, "契约化 service payload", "canonical service payload"),
            f"{case.get('canonical_feature_schema_status')} | expected={_fmt_metric(case.get('expected_vehicle_feature_count'), 0)}",
            GREEN,
        ),
        (
            _label(font, "schema gap 摘要", "schema gap summary"),
            f"missing vehicle={_fmt_metric(case.get('missing_vehicle_feature_count'), 0)} | groups={_fmt_metric(case.get('native_missing_measurement_group_count'), 0)}",
            PURPLE,
        ),
    ]
    for idx, (label, value, color) in enumerate(schema_lines):
        y = 1.24 - idx * 0.52
        ax_schema.add_patch(
            patches.FancyBboxPatch(
                (0.1, y - 0.23),
                5.6,
                0.39,
                boxstyle="round,pad=0.035,rounding_size=0.045",
                facecolor=PALE,
                edgecolor=GRID,
                lw=0.9,
            )
        )
        ax_schema.text(0.27, y, label, fontsize=8.4, color=MUTED, va="center")
        ax_schema.text(5.48, y, value, fontsize=10.1, color=color, weight="bold", ha="right", va="center")
    ax_schema.set_title(_label(font, "schema 状态摘要", "Schema status summary"), fontsize=10.8, weight="bold", color=INK)

    fig.suptitle(_label(font, "runtime 与语义事件案例复盘", "Runtime and Semantic Event Case Review"), fontsize=15.2, weight="bold", color=INK)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_semantic_case",
        path=path,
        source_paths=_source_paths(sources, "runtime_case_table", "support", "runtime_service", "runtime_schema_contract"),
        evidence_layer="runtime_semantic_support",
        table_path=table_paths["runtime_semantic_case"],
        metric_definition="Runtime case values are copied from runtime_semantic_case.csv and shown as semantic attribution, query distribution, KPI ranges, and schema status.",
        case_definition=str(case.get("case_definition") or "runtime semantic support case"),
        replaces_problem="replaces near-flat line and repeated attribution bars with case-level semantic attribution and schema-status summary.",
    )


def _plot_rigid_body_rotation(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_rigid_body_rotation_rows(sources))
    family_rows = frame.loc[frame["row_type"] == "family_metric"].copy()
    matrix_rows = frame.loc[frame["row_type"] == "rotation_field_matrix"].copy()
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0]})
    pivot = family_rows.pivot(index="family", columns="metric_name", values="metric_value")
    families = [name for name in ("minimal", "full", "rigid_body") if name in pivot.index]
    metrics = ["test_total", "test_alignment", "test_physics_total"]
    colors = [BLUE, GOLD, PURPLE]
    width = 0.22
    for offset, metric in enumerate(metrics):
        values = [float(pivot.loc[family, metric]) for family in families]
        positions = [i + (offset - 1) * width for i in range(len(families))]
        axes[0].bar(positions, values, width=width, color=colors[offset], label=metric)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(families)), families)
    axes[0].set_ylabel(_label(font, "log 尺度损失", "loss on log scale"))
    axes[0].set_title(_label(font, "A. minimal/full/rigid_body 对比", "A. minimal/full/rigid_body comparison"), fontsize=11, weight="bold")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_axisbelow(True)
    axes[1].set_title(_label(font, "B. rotation 字段可用性矩阵", "B. rotation field availability"), fontsize=11, weight="bold")
    axis_labels = ["pitch", "roll", "yaw"]
    field_labels = ["angle", "rate"]
    matrix = []
    for axis in axis_labels:
        row_values = []
        for field_type in field_labels:
            selected = matrix_rows.loc[(matrix_rows["axis"] == axis) & (matrix_rows["field_type"] == field_type)]
            row_values.append(1 if not selected.empty and bool(selected["available"].iloc[0]) else 0)
        matrix.append(row_values)
    axes[1].imshow(matrix, cmap=_availability_cmap(), vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(len(field_labels)), field_labels)
    axes[1].set_yticks(range(len(axis_labels)), axis_labels)
    for y, axis in enumerate(axis_labels):
        for x, field_type in enumerate(field_labels):
            label = "available" if matrix[y][x] else "missing"
            axes[1].text(x, y, label, ha="center", va="center", color=INK if matrix[y][x] else "#7f1d1d", fontsize=10, weight="bold")
    reason = str(matrix_rows["rotation_disabled_reason"].dropna().iloc[0]) if not matrix_rows["rotation_disabled_reason"].dropna().empty else ""
    axes[1].text(
        0.5,
        -0.18,
        _label(font, "translation + vertical 已启用；rotation disabled 因缺少 pitch/roll/yaw rate。", "translation + vertical enabled; rotation disabled because pitch/roll/yaw rate is missing."),
        ha="center",
        va="top",
        transform=axes[1].transAxes,
        fontsize=8.4,
        color=MUTED,
    )
    fig.suptitle(_label(font, "刚体约束与 rotation 字段诊断", "Rigid-Body Constraint and Rotation Diagnostics"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="rigid_body_rotation_audit",
        path=path,
        source_paths=_source_paths(sources, "rigid_body", "rotation_audit"),
        evidence_layer="rigid_body_rotation_diagnostics",
        table_path=table_paths["rigid_body_rotation_audit"],
        metric_definition=f"Family losses use log scale; rotation matrix marks rate fields as missing. {reason}",
        replaces_problem="rotation rate absence is shown as a matrix, not empty bars.",
    )


def _plot_weak_label_sweep(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_weak_label_rows(sources))
    completed = frame.loc[frame["row_type"] == "completed_run"].copy()
    completed["test_total_numeric"] = pd.to_numeric(completed["test_total"], errors="coerce")
    best = completed.sort_values("test_total_numeric").groupby("sample_source", as_index=False).first()
    plt, _ = _import_matplotlib(font)
    fig = plt.figure(figsize=(14.2, 8.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
    metrics = [
        ("best_test_total", "test_total"),
        ("test_task_total", "task_total"),
        ("test_causal_total", "causal_total"),
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
    ax_status = fig.add_subplot(grid[1, 0])
    status_rows = _status_rows(frame)
    y_labels = [row["label"] for row in status_rows]
    completed_counts = [row["completed"] for row in status_rows]
    remaining_counts = [row["remaining"] for row in status_rows]
    y_pos = range(len(status_rows))
    ax_status.barh(y_pos, completed_counts, color=GREEN, label="completed")
    ax_status.barh(y_pos, remaining_counts, left=completed_counts, color=ORANGE, label="partial/blocked boundary")
    ax_status.set_yticks(list(y_pos), y_labels, fontsize=8)
    ax_status.set_xlabel("runs")
    ax_status.set_title(_label(font, "运行状态", "Run status"), fontsize=10.5, weight="bold")
    ax_status.legend(fontsize=8, frameon=False)
    ax_heat = fig.add_subplot(grid[1, 1:])
    heat = completed.pivot_table(index="sample_source", columns="lag_label", values="test_total", aggfunc="min")
    heat = heat.reindex(sorted(heat.index), axis=0)
    heat = heat.reindex(sorted(heat.columns, key=lambda item: str(item)), axis=1)
    im = ax_heat.imshow(heat.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax_heat.set_xticks(range(len(heat.columns)), [str(col) for col in heat.columns])
    ax_heat.set_yticks(range(len(heat.index)), [_sample_source_label(idx) for idx in heat.index], fontsize=8)
    ax_heat.set_title(_label(font, "小网格 lag_window_points × sample_source", "Small-grid lag_window_points by sample_source"), fontsize=10.5, weight="bold")
    heat_values = [float(value) for value in heat.to_numpy().ravel() if pd.notna(value)]
    heat_threshold = (min(heat_values) + max(heat_values)) / 2 if heat_values else 0
    for y in range(len(heat.index)):
        for x in range(len(heat.columns)):
            value = heat.iloc[y, x]
            if pd.notna(value):
                text_color = "white" if float(value) >= heat_threshold else INK
                ax_heat.text(x, y, f"{float(value):.1f}", ha="center", va="center", fontsize=8.2, color=text_color, weight="bold")
    fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02, label="test_total")
    fig.suptitle(_label(font, "weak-label 小网格与 resume 边界", "Weak-Label Small Grid and Resume Boundary"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="weak_label_sweep_ablation",
        path=path,
        source_paths=_source_paths(sources, "proxy_sweep", "live_sweep", "live_partial"),
        evidence_layer="thesis_weak_label",
        table_path=table_paths["weak_label_sweep_ablation"],
        metric_definition="Compares proxy/live best metrics, completed versus partial runs, and small-grid lag settings.",
        replaces_problem="weak-label sweep no longer uses two near-identical bars.",
    )


def _plot_private_component(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(build_private_component_rows(sources))
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    tasks = [
        ("T1_maneuver_intensity_class", _label(font, "T1 classification: macro_f1 越高越好", "T1 classification: macro_f1 higher is better"), False),
        ("T2_next_window_physiology_response", _label(font, "T2 regression: RMSE 越低越好", "T2 regression: RMSE lower is better"), True),
        ("T3_paired_pilot_window_retrieval", _label(font, "T3 retrieval: top1_accuracy 越高越好", "T3 retrieval: top1_accuracy higher is better"), False),
    ]
    for ax, (task, title, log_x) in zip(axes.flat[:3], tasks, strict=True):
        task_rows = frame.loc[frame["task_name"] == task].copy()
        task_rows["order"] = task_rows["variant_name"].map({name: idx for idx, name in enumerate(preferred)})
        task_rows = task_rows.sort_values("order", ascending=False)
        colors = [_variant_color(role) for role in task_rows["variant_role"]]
        ax.barh(task_rows["display_variant"], task_rows["primary_metric_value"], color=colors, edgecolor=INK, linewidth=0.4)
        if log_x:
            ax.set_xscale("log")
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(8)
    ax = axes.flat[3]
    contribution = frame.loc[frame["variant_role"] == "component_removed"].copy()
    contribution["label"] = contribution["display_variant"] + "\n" + contribution["task_name"].str.extract(r"^(T\d)")[0]
    contribution = contribution.sort_values(["display_variant", "task_name"], ascending=[True, False])
    ax.barh(contribution["label"], contribution["normalized_delta_vs_full"], color=PURPLE, edgecolor=INK, linewidth=0.4)
    ax.set_xlim(0, 1.05)
    ax.set_title(_label(font, "组件移除相对贡献（按任务归一）", "Normalized relative contribution of component removal"), fontsize=10.5, weight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(_label(font, "chronaris_opt 组件消融：分任务尺度", "chronaris_opt Component Ablation by Task Scale"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="chronaris_opt_component_ablation",
        path=path,
        source_paths=_source_paths(sources, "private_component"),
        evidence_layer="private_proxy",
        table_path=table_paths["chronaris_opt_component_ablation"],
        metric_definition="T1/T2/T3 are separated by metric scale; normalized delta avoids RMSE dominating other tasks.",
        replaces_problem="mixed-unit delta bars are replaced by task facets plus normalized contribution.",
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
        _label(font, "公开适配支撑外部基线，私有双流支撑论文主线，代理基准支撑组件分析。", "Public adapter supports external baselines; private dual-stream supports the thesis mainline; proxy benchmark supports component analysis."),
        ha="center",
        fontsize=10.2,
        color=INK,
        weight="bold",
    )
    ax.set_title(_label(font, "公开适配、私有主线与代理消融的正向分工", "Positive Roles Across Public Adapter, Private Mainline, and Proxy Ablation"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
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
    steps_cn = ["双流输入", "因果掩码", "事件 token", "语义 query", "归因对齐", "风险/负荷/事件输出"]
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
    frame = frame.sort_values("mean_top_event_attribution", ascending=True)
    labels = [_identifier_label(view, width=32) for view in frame["view_id"]]
    y_positions = list(range(len(frame)))
    ax.barh(y_positions, frame["mean_top_event_attribution"], color=PURPLE, edgecolor=INK, linewidth=0.4)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("mean_top_event_attribution")
    ax.set_title(_label(font, "三视图 query-to-event attribution 支撑", "View-Level Query-to-Event Attribution Support"), fontsize=11, weight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for y, row in enumerate(frame.to_dict(orient="records")):
        ax.text(
            row["mean_top_event_attribution"],
            y,
            f"  {row['dominant_query']} | tokens={float(row['mean_event_token_count']):.2f}",
            va="center",
            fontsize=8.2,
            color=INK,
        )
    fig.suptitle(_label(font, "语义事件融合：event token 与 query attribution", "Semantic Event Fusion: Event Tokens and Query Attribution"), fontsize=15, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="semantic_event_fusion_overview",
        path=path,
        source_paths=_source_paths(sources, "support", "semantic_event"),
        evidence_layer="semantic_support",
        table_path=table_paths["semantic_event_fusion_overview"],
        case_definition="Pipeline diagram plus view-level event-token/query attribution rows from semantic support summary.",
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
        "A0_baseline": "A0\nbaseline",
        "A1_llm_context": "A1\ncontext",
        "A2_llm_semantic_hints": "A2\nsemantic hints",
        "A3_llm_runtime_explanation": "A3\nruntime explanation",
        "A4_human_review_packet": "A4\nreview packet",
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
            _wrap_text(_label(font, row["stage_cn"], row["stage"]), 18),
            ha="center",
            va="center",
            fontsize=8.2,
            color=INK,
            linespacing=1.15,
        )
        ax.text(x + card_w / 2, card_y + 1.12, f"{row['metric_name']}", ha="center", fontsize=7.7, color=MUTED)
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
        _label(font, "DeepSeek 预处理：", "DeepSeek preprocessing: ")
        + f"{pre_row['metric_name']}={pre_row['metric_value']} | {pre_row['metric_note']}",
        fontsize=9.5,
        color=INK,
        weight="bold",
    )
    ax.text(
        0.5,
        0.33,
        _label(font, "边界：LLM 输出只作为 preprocessing context、白名单 semantic hints、runtime explanation 和待人工复核材料。", "Boundary: LLM output is preprocessing context, whitelisted semantic hints, runtime explanation, and pending human-review material."),
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_title(_label(font, "LLM 预处理 A0-A4 对比流", "LLM Preprocessing A0-A4 Comparison Flow"), fontsize=15, weight="bold", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
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


def _availability_cmap():
    from matplotlib import colors

    return colors.ListedColormap(["#fde8e2", "#e6f4ea"])


def _label(font: PlotFontSelection, cn_label: str, ascii_label: str) -> str:
    return ascii_label if font.ascii_only else cn_label


def _metric_cell(ax, x: float, y: float, name: object, value: object, color: str) -> None:
    ax.text(x + 0.05, y + 0.11, str(name), fontsize=7.8, color=MUTED, va="center")
    ax.text(x + 0.05, y - 0.16, _short(value, 22), fontsize=11.5, color=color, weight="bold", va="center")


def _sample_source_label(value: object) -> str:
    text = str(value)
    if text == "live_influx":
        return "live\ninflux"
    if text == "stage_h_window_stats_proxy":
        return "Stage-H\nproxy"
    return text.replace("_", "\n")


def _llm_card_note(font: PlotFontSelection, row: Mapping[str, object]) -> str:
    condition = str(row.get("condition", ""))
    metric_note = str(row.get("metric_note", ""))
    if condition.startswith("A0"):
        return _label(font, "内置 query bank", metric_note.replace("_", " "))
    if condition.startswith("A1"):
        return _label(font, "标签未改写；changed=0", metric_note.replace("_", " "))
    if condition.startswith("A2"):
        return _label(font, "3->7；新增 4", metric_note.replace("_", " "))
    if condition.startswith("A3"):
        return _label(font, "4/12；完整性=1.0", metric_note.replace("_", " "))
    if condition.startswith("A4"):
        return _label(font, "人工复核未完成", metric_note.replace("_", " "))
    return metric_note.replace("_", " ")


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
        return f"架次: {sortie_id}\npilot: {pilot_id}"
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
