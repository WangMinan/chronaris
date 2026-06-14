"""Figure builders for Stage I midterm evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.pipelines.stage_i.evidence.midterm_metrics import (
    _extract_nasa_fusion_metrics,
    _extract_uab_fairness_metrics,
    _infer_timestamp,
    _primary_task_metric,
)

DEFAULT_CJK_FONT_CANDIDATES = (
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
)


@dataclass(frozen=True, slots=True)
class PlotFontSelection:
    family: str | None
    ascii_only: bool
    note: str


def _detect_plot_font() -> PlotFontSelection:
    try:
        from matplotlib import font_manager
    except Exception:  # pragma: no cover - import guard
        return PlotFontSelection(None, True, "matplotlib unavailable; used ASCII-safe labels")
    available_names = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in DEFAULT_CJK_FONT_CANDIDATES:
        if candidate in available_names:
            return PlotFontSelection(candidate, False, f"using CJK font {candidate}")
    return PlotFontSelection(None, True, "CJK font missing; used ASCII-safe labels")


def _write_figures(
    *,
    plots_root: Path,
    sources: Mapping[str, Mapping[str, object]],
    font_selection: PlotFontSelection,
    metrics_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    del metrics_frame
    return [
        _plot_chain_status(plots_root / "thesis_chain_status.png", sources, font_selection),
        _plot_private_task_metrics(plots_root / "private_opt_task_metrics.png", sources, font_selection),
        _plot_public_mainline_metrics(plots_root / "public_mainline_metrics.png", sources, font_selection),
        _plot_support_ablation(plots_root / "support_ablation_metrics.png", sources, font_selection),
        _plot_anchor_windows(plots_root / "anchor_window_scores.png", sources, font_selection),
        _plot_runtime_timeline(plots_root / "runtime_progress_timeline.png", sources, font_selection),
    ]


def _plot_chain_status(path: Path, sources: Mapping[str, Mapping[str, object]], font_selection: PlotFontSelection) -> dict[str, object]:
    support = sources["support"]["payload"]
    private = sources["private"]["payload"]
    public_mainline = sources["public_mainline"]["payload"]
    labels = [("E", "Align"), ("F", "Physics"), ("G", "Causal"), ("H", "Export"), ("I-private", "Private"), ("I-public", "Public")]
    subtitles = [
        f"cos={support['alignment_support']['alignment_chain']['e_baseline']['mean_projection_cosine']:.3f}",
        f"cos={support['alignment_support']['alignment_chain']['f_full']['mean_projection_cosine']:.3f}",
        f"top={support['causal_support']['g_min']['mean_top_contribution_score']:.3f}",
        f"views={support['alignment_support']['stage_h_export']['generated_view_count']}",
        f"opt={private['private_optimality_supported']}",
        str(public_mainline["public_mainline_status"]),
    ]
    return _save_stage_boxes_plot(path, labels, subtitles, font_selection, "Thesis Chain Status", "论文链路状态图")


def _save_stage_boxes_plot(path: Path, labels: Sequence[tuple[str, str]], subtitles: Sequence[str], font_selection: PlotFontSelection, ascii_title: str, cn_title: str) -> dict[str, object]:
    plt, _ = _import_matplotlib(font_selection)
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 2.8))
    for index, ((primary, secondary), subtitle) in enumerate(zip(labels, subtitles, strict=True)):
        x0 = index * 1.7
        ax.add_patch(Rectangle((x0, 0.25), 1.4, 0.9, facecolor="#e8f0ea", edgecolor="#355b3e", linewidth=1.5))
        ax.text(x0 + 0.7, 0.92, primary, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x0 + 0.7, 0.68, secondary, ha="center", va="center", fontsize=9)
        ax.text(x0 + 0.7, 0.42, subtitle, ha="center", va="center", fontsize=8)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x0 + 1.55, 0.7), xytext=(x0 + 1.42, 0.7), arrowprops={"arrowstyle": "->", "color": "#355b3e", "lw": 1.4})
    ax.set_xlim(-0.1, len(labels) * 1.7 - 0.1)
    ax.set_ylim(0.15, 1.25)
    ax.set_axis_off()
    ax.set_title(_pick_label(font_selection, cn_title, ascii_title), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_row("thesis_chain_status", path, "论文链路状态图", "Thesis Chain Status", font_selection.note)


def _plot_private_task_metrics(path: Path, sources: Mapping[str, Mapping[str, object]], font_selection: PlotFontSelection) -> dict[str, object]:
    plt, _ = _import_matplotlib(font_selection)
    private_payload = sources["private"]["payload"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for axis, task_id in zip(axes, private_payload["tasks"], strict=True):
        task_payload = private_payload["tasks"][task_id]
        variant_names = [private_payload["target_variant_name"], private_payload["conclusion"]["no_mask_variant_name"], "naive_sync"]
        values = []
        for variant_name in variant_names:
            variant_payload = task_payload["variants"][variant_name]
            _, value = _primary_task_metric(task_id, variant_payload)
            values.append(value)
        axis.bar(variant_names, values, color=["#355b3e", "#7a9e7e", "#c5c9b8"])
        axis.set_title(task_id.replace("_", "\n"), fontsize=9)
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle(_pick_label(font_selection, "private opt 三任务指标图", "Private Opt Task Metrics"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_row("private_opt_task_metrics", path, "private opt 三任务指标图", "Private Opt Task Metrics", "")


def _plot_public_mainline_metrics(path: Path, sources: Mapping[str, Mapping[str, object]], font_selection: PlotFontSelection) -> dict[str, object]:
    plt, _ = _import_matplotlib(font_selection)
    public_mainline = sources["public_mainline"]["payload"]
    fairness = _extract_uab_fairness_metrics(sources)
    confirm = _extract_nasa_fusion_metrics(sources)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    uab_groups = list(public_mainline["uab"]["groups"])
    public_rmse = [public_mainline["uab"]["groups"][group]["public_rmse"] for group in uab_groups]
    deep_rmse = [fairness[group]["rmse"] for group in uab_groups]
    x = np.arange(len(uab_groups))
    axes[0].bar(x - 0.18, public_rmse, width=0.36, label="public_mainline", color="#355b3e")
    axes[0].bar(x + 0.18, deep_rmse, width=0.36, label="contiformer", color="#a2b29f")
    axes[0].set_xticks(x, uab_groups)
    axes[0].set_title("UAB RMSE", fontsize=10)
    axes[0].legend(fontsize=8)
    nasa_values = [
        public_mainline["nasa"]["combined_macro_f1"],
        float(confirm["macro_f1"]),
        0.40,
    ]
    axes[1].bar(["public_opt", "fusion_confirm", "gate"], nasa_values, color=["#355b3e", "#6b8f71", "#d9ae61"])
    axes[1].set_title("NASA combined macro-F1", fontsize=10)
    fig.suptitle(_pick_label(font_selection, "public mainline 指标图", "Public Mainline Metrics"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_row("public_mainline_metrics", path, "public mainline 指标图", "Public Mainline Metrics", "")


def _plot_support_ablation(path: Path, sources: Mapping[str, Mapping[str, object]], font_selection: PlotFontSelection) -> dict[str, object]:
    plt, _ = _import_matplotlib(font_selection)
    rows = pd.DataFrame(sources["support"]["payload"]["main_ablation_rows"])
    selected = rows.loc[
        rows["variant"].isin(("g_min", "vehicle_delta_suppressed", "no_event_bias")),
        ["variant", "delta_mean_top_contribution_score"],
    ].fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.barh(selected["variant"], selected["delta_mean_top_contribution_score"], color=["#355b3e", "#d37c5c", "#d9ae61"])
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_title(_pick_label(font_selection, "support ablation 图", "Support Ablation"), fontsize=12)
    ax.set_xlabel("delta_mean_top_contribution_score")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_row("support_ablation", path, "support ablation 图", "Support Ablation", "")


def _plot_anchor_windows(path: Path, sources: Mapping[str, Mapping[str, object]], font_selection: PlotFontSelection) -> dict[str, object]:
    plt, _ = _import_matplotlib(font_selection)
    anchors = pd.DataFrame(sources["anchor"]["payload"]["anchors"]).head(9)
    labels = [f"#{int(rank)}" for rank in anchors["anchor_rank"]]
    colors = ["#d37c5c" if verdict == "WARN" else "#355b3e" for verdict in anchors["view_verdict"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(labels, anchors["anchor_score"], color=colors)
    ax.set_title(_pick_label(font_selection, "anchor window 图", "Anchor Window Scores"), fontsize=12)
    ax.set_xlabel("anchor_rank")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_row("anchor_window_scores", path, "anchor window 图", "Anchor Window Scores", "")


def _plot_runtime_timeline(path: Path, sources: Mapping[str, Mapping[str, object]], font_selection: PlotFontSelection) -> dict[str, object]:
    plt, _ = _import_matplotlib(font_selection)
    timeline_items = [
        ("private_pkg", _infer_timestamp(sources["private"]["payload"], sources["private"]["path"])),
        ("support", _infer_timestamp(sources["support"]["payload"], sources["support"]["path"])),
        ("anchor", _infer_timestamp(sources["anchor"]["payload"], sources["anchor"]["path"])),
        ("public_mainline", _infer_timestamp(sources["public_mainline"]["payload"], sources["public_mainline"]["path"])),
        ("fusion_round2", _infer_timestamp(sources["public_fusion_screen"]["payload"], sources["public_fusion_screen"]["path"])),
    ]
    if "nasa_fusion_confirm" in sources:
        timeline_items.append(("nasa_confirm", _infer_timestamp(sources["nasa_fusion_confirm"]["payload"], sources["nasa_fusion_confirm"]["path"])))
    if "uab_fairness" in sources:
        timeline_items.append(("uab_fairness", _infer_timestamp(sources["uab_fairness"]["payload"], sources["uab_fairness"]["path"])))
    ordered_items = [(name, ts) for name, ts in timeline_items if ts is not None]
    ordered_items.sort(key=lambda item: item[1])
    x = np.arange(len(ordered_items))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(x, np.zeros_like(x), color="#6b8f71", linewidth=1.2)
    ax.scatter(x, np.zeros_like(x), s=80, color="#355b3e")
    ax.set_yticks([])
    ax.set_xticks(x, [name for name, _ in ordered_items], rotation=20)
    for idx, (_, timestamp) in enumerate(ordered_items):
        ax.text(idx, 0.02, timestamp.strftime("%m-%d"), ha="center", va="bottom", fontsize=8)
    ax.set_title(_pick_label(font_selection, "runtime progress timeline 图", "Runtime Progress Timeline"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_row("runtime_progress_timeline", path, "runtime progress timeline 图", "Runtime Progress Timeline", "")


def _figure_row(
    figure_id: str,
    path: Path,
    title_cn: str,
    title_ascii: str,
    note: str,
) -> dict[str, object]:
    return {
        "figure_id": figure_id,
        "title_cn": title_cn,
        "title_ascii": title_ascii,
        "path": str(path),
        "exists": path.exists(),
        "note": note,
    }


def _import_matplotlib(font_selection: PlotFontSelection):
    from matplotlib import pyplot as plt
    from matplotlib import rcParams

    if font_selection.family:
        rcParams["font.family"] = [font_selection.family]
    rcParams["axes.unicode_minus"] = False
    return plt, rcParams


def _pick_label(font_selection: PlotFontSelection, cn_label: str, ascii_label: str) -> str:
    return ascii_label if font_selection.ascii_only else cn_label
