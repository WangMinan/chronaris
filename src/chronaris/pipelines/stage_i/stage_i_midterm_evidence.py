"""Stage I midterm evidence-pack builder."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_midterm"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_PRIVATE_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/"
    "private_benchmark_summary.json"
)
DEFAULT_SUPPORT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/"
    "support_summary.json"
)
DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_mainline/"
    "20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/"
    "public_mainline_summary.json"
)
DEFAULT_ANCHOR_MANIFEST_PATH = (
    "docs/artifacts/assets/stage_i_anchor/20260506T165435Z-stage-i-anchor/"
    "anchor_manifest.json"
)
DEFAULT_DEEP_COMPARISON_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i/20260501T-full-loso-deep-comparison/"
    "comparison_summary.json"
)
DEFAULT_PUBLIC_FUSION_SCREEN_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_fusion_screen/"
    "20260506T-stage-i-public-fusion-screen-round2/fusion_screen_summary.json"
)
DEFAULT_NASA_PUBLIC_OPT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/"
    "public_opt_summary.json"
)
DEFAULT_UAB_PUBLIC_OPT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/"
    "public_opt_summary.json"
)
KEEP_STAGE_I_REPORTS = {
    "README.md",
    "stage-i-ablation-support-20260506T120000Z-stage-i-support.md",
    "stage-i-alignment-support-20260506T120000Z-stage-i-support.md",
    "stage-i-anchor-20260506T165435Z-stage-i-anchor.md",
    "stage-i-case-study-phase2-2026-04-29.md",
    "stage-i-causal-support-20260506T120000Z-stage-i-support.md",
    "stage-i-closure-2026-04-30.md",
    "stage-i-deep-comparison-full-loso-2026-05-01.md",
    "stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round2.md",
    "stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md",
    "stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md",
    "stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md",
    "stage-i-public-opt-20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1.md",
    "stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md",
    "stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md",
}
TEXT_SCAN_ROOTS = ("AGENTS.md", "docs", "scripts", "src", "tests")
TEXT_SCAN_SUFFIXES = {".md", ".py"}
DEFAULT_CJK_FONT_CANDIDATES = (
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
)


@dataclass(frozen=True, slots=True)
class StageIMidtermEvidenceConfig:
    run_id: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    workspace_root: str = "."
    private_summary_path: str = DEFAULT_PRIVATE_SUMMARY_PATH
    support_summary_path: str = DEFAULT_SUPPORT_SUMMARY_PATH
    public_mainline_summary_path: str = DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH
    anchor_manifest_path: str = DEFAULT_ANCHOR_MANIFEST_PATH
    deep_comparison_summary_path: str = DEFAULT_DEEP_COMPARISON_SUMMARY_PATH
    public_fusion_screen_summary_path: str = DEFAULT_PUBLIC_FUSION_SCREEN_SUMMARY_PATH
    nasa_public_opt_summary_path: str = DEFAULT_NASA_PUBLIC_OPT_SUMMARY_PATH
    uab_public_opt_summary_path: str = DEFAULT_UAB_PUBLIC_OPT_SUMMARY_PATH
    uab_fairness_summary_path: str | None = None
    nasa_fusion_confirm_summary_path: str | None = None


@dataclass(frozen=True, slots=True)
class StageIMidtermEvidenceRunResult:
    run_id: str
    artifact_root: str
    manifest_path: str
    metrics_path: str
    figure_index_path: str
    cleanup_audit_path: str
    report_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PlotFontSelection:
    family: str | None
    ascii_only: bool
    note: str


def run_stage_i_midterm_evidence(
    config: StageIMidtermEvidenceConfig,
) -> StageIMidtermEvidenceRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_midterm_evidence",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "report_root": config.report_root,
        },
    ) as progress:
        return _run_stage_i_midterm_evidence_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_midterm_evidence_observed(
    *,
    config: StageIMidtermEvidenceConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIMidtermEvidenceRunResult:
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    plots_root = run_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(config)
    progress.update("sources_loaded", source_count=len(sources))

    font_selection = _detect_plot_font()
    progress.update(
        "font_selected",
        font_family=font_selection.family,
        ascii_only=font_selection.ascii_only,
    )

    metrics_rows = _build_metrics_rows(config=config, sources=sources)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_path = run_root / "midterm_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    progress.update("metrics_written", metric_row_count=len(metrics_frame))

    cleanup_audit = _build_cleanup_audit(
        workspace_root=Path(config.workspace_root),
        report_root=report_root,
        keep_names=KEEP_STAGE_I_REPORTS | {f"stage-i-midterm-{config.run_id}.md"},
    )
    cleanup_audit_path = run_root / "cleanup_audit.json"
    cleanup_audit_path.write_text(
        json.dumps(cleanup_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.update(
        "cleanup_audit_written",
        outdated_doc_count=cleanup_audit["summary"]["outdated_doc_candidate_count"],
        unreferenced_code_count=cleanup_audit["summary"]["unreferenced_code_candidate_count"],
    )

    figures = _write_figures(
        plots_root=plots_root,
        sources=sources,
        font_selection=font_selection,
        metrics_frame=metrics_frame,
    )
    figure_index_path = run_root / "midterm_figure_index.csv"
    pd.DataFrame(figures).to_csv(figure_index_path, index=False)
    progress.update("figures_written", figure_count=len(figures))

    report_path = report_root / f"stage-i-midterm-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_midterm_report(
            run_id=config.run_id,
            sources=sources,
            font_selection=font_selection,
            metrics_frame=metrics_frame,
            cleanup_audit=cleanup_audit,
            figures=figures,
            artifact_root=run_root,
        )
        + "\n",
        encoding="utf-8",
    )
    progress.update("report_written", report_path=str(report_path))

    summary = _build_manifest_summary(
        config=config,
        run_root=run_root,
        metrics_frame=metrics_frame,
        figures=figures,
        cleanup_audit=cleanup_audit,
        font_selection=font_selection,
        sources=sources,
        report_path=report_path,
        metrics_path=metrics_path,
        figure_index_path=figure_index_path,
        cleanup_audit_path=cleanup_audit_path,
    )
    manifest_path = run_root / "midterm_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.finish(
        manifest_path=str(manifest_path),
        report_path=str(report_path),
        figure_index_path=str(figure_index_path),
    )
    return StageIMidtermEvidenceRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        manifest_path=str(manifest_path),
        metrics_path=str(metrics_path),
        figure_index_path=str(figure_index_path),
        cleanup_audit_path=str(cleanup_audit_path),
        report_path=str(report_path),
        summary=summary,
    )


def _load_sources(config: StageIMidtermEvidenceConfig) -> dict[str, dict[str, object]]:
    path_map = {
        "private": config.private_summary_path,
        "support": config.support_summary_path,
        "public_mainline": config.public_mainline_summary_path,
        "anchor": config.anchor_manifest_path,
        "deep_comparison": config.deep_comparison_summary_path,
        "public_fusion_screen": config.public_fusion_screen_summary_path,
        "nasa_public_opt": config.nasa_public_opt_summary_path,
        "uab_public_opt": config.uab_public_opt_summary_path,
    }
    optional_path_map = {
        "uab_fairness": config.uab_fairness_summary_path,
        "nasa_fusion_confirm": config.nasa_fusion_confirm_summary_path,
    }
    loaded = {
        name: {"path": path_like, "payload": _load_json(path_like)}
        for name, path_like in path_map.items()
    }
    for name, path_like in optional_path_map.items():
        if path_like:
            loaded[name] = {"path": path_like, "payload": _load_json(path_like)}
    return loaded


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def _build_metrics_rows(
    *,
    config: StageIMidtermEvidenceConfig,
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    del config
    rows: list[dict[str, object]] = []
    private_payload = sources["private"]["payload"]
    for task_id, task_payload in private_payload["tasks"].items():
        target = task_payload["variants"][private_payload["target_variant_name"]]
        no_mask = task_payload["variants"][private_payload["conclusion"]["no_mask_variant_name"]]
        target_metric_name, target_value = _primary_task_metric(task_id, target)
        _, no_mask_value = _primary_task_metric(task_id, no_mask)
        rows.extend(
            [
                _metric_row("private", task_id, target_metric_name, "chronaris_opt", target_value, None, sources["private"]["path"], ""),
                _metric_row("private", task_id, target_metric_name, "chronaris_opt_no_causal_mask", no_mask_value, target_value - no_mask_value if task_id != "T2_next_window_physiology_response" else no_mask_value - target_value, sources["private"]["path"], "vs target"),
            ]
        )
    support_payload = sources["support"]["payload"]
    rows.extend(
        [
            _metric_row("support", "alignment", "mean_projection_cosine", "e_baseline", support_payload["alignment_support"]["alignment_chain"]["e_baseline"]["mean_projection_cosine"], None, sources["support"]["path"], ""),
            _metric_row("support", "alignment", "mean_projection_cosine", "f_full", support_payload["alignment_support"]["alignment_chain"]["f_full"]["mean_projection_cosine"], support_payload["alignment_support"]["alignment_chain"]["delta_f_minus_e"]["mean_projection_cosine"], sources["support"]["path"], "delta_f_minus_e"),
            _metric_row("support", "causal", "mean_top_contribution_score", "g_min", support_payload["causal_support"]["g_min"]["mean_top_contribution_score"], None, sources["support"]["path"], ""),
            _metric_row("support", "causal", "delta_mean_top_contribution_score", support_payload["causal_support"]["case_study"]["strongest_ablation"]["name"], support_payload["causal_support"]["case_study"]["strongest_ablation"]["delta_mean_top_contribution_score"], None, sources["support"]["path"], "strongest ablation"),
        ]
    )
    public_mainline = sources["public_mainline"]["payload"]
    for subset_id, payload in public_mainline["uab"]["groups"].items():
        rows.append(
            _metric_row(
                "public",
                f"uab/{subset_id}",
                "rmse",
                payload["best_public_head"],
                payload["public_rmse"],
                payload["rmse_margin_vs_best_deep"],
                sources["public_mainline"]["path"],
                f"best_deep={payload['best_deep_model']}",
            )
        )
    rows.append(
        _metric_row(
            "public",
            "nasa/combined",
            "macro_f1",
            public_mainline["nasa"]["combined_best_head"],
            public_mainline["nasa"]["combined_macro_f1"],
            public_mainline["nasa"]["combined_macro_f1"] - 0.40,
            sources["public_mainline"]["path"],
            "public opt closed gate",
        )
    )
    confirm_metrics = _extract_nasa_fusion_metrics(sources)
    rows.append(
        _metric_row(
            "public",
            "nasa/fusion_confirm",
            "macro_f1",
            confirm_metrics["candidate_id"],
            confirm_metrics["macro_f1"],
            confirm_metrics["macro_f1"] - 0.40,
            confirm_metrics["source_path"],
            "balanced_class epochs10 if provided",
        )
    )
    anchor_payload = sources["anchor"]["payload"]
    rows.extend(
        [
            _metric_row("anchor", "overview", "selected_view_count", "anchor", anchor_payload["overview"]["selected_view_count"], None, sources["anchor"]["path"], ""),
            _metric_row("anchor", "overview", "selected_anchor_count", "anchor", anchor_payload["overview"]["selected_anchor_count"], None, sources["anchor"]["path"], ""),
            _metric_row("anchor", "overview", "top_anchor_score", "anchor_rank_1", float(anchor_payload["anchors"][0]["anchor_score"]), None, sources["anchor"]["path"], ""),
        ]
    )
    fairness_metrics = _extract_uab_fairness_metrics(sources)
    for subset_id, payload in fairness_metrics.items():
        rows.append(
            _metric_row(
                "fairness",
                f"uab/{subset_id}",
                "rmse",
                "contiformer_confirm",
                payload["rmse"],
                payload["rmse"] - payload["historical_rmse"],
                payload["source_path"],
                "delta_vs_historical",
            )
        )
    return rows


def _metric_row(
    section: str,
    scope: str,
    metric_name: str,
    variant: str,
    value: float | int | None,
    delta: float | int | None,
    source_path: str,
    note: str,
) -> dict[str, object]:
    return {
        "section": section,
        "scope": scope,
        "metric_name": metric_name,
        "variant": variant,
        "value": value,
        "delta": delta,
        "source_path": source_path,
        "note": note,
    }


def _primary_task_metric(task_id: str, payload: Mapping[str, object]) -> tuple[str, float]:
    if task_id == "T1_maneuver_intensity_class":
        best_metrics = payload["best_metrics"]
        return "macro_f1", float(best_metrics["macro_f1"])
    if task_id == "T2_next_window_physiology_response":
        best_metrics = payload["best_metrics"]
        return "rmse", float(best_metrics["rmse"])
    return "top1_accuracy", float(payload["top1_accuracy"])


def _extract_uab_fairness_metrics(
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float | str]]:
    historical_groups = (
        sources["deep_comparison"]["payload"]["datasets"]["uab_workload_dataset"]["models"]["contiformer"]["summary"]["subjective"]["groups"]
    )
    if "uab_fairness" not in sources:
        return {
            subset_id: {
                "rmse": float(metrics["rmse"]),
                "historical_rmse": float(metrics["rmse"]),
                "source_path": sources["deep_comparison"]["path"],
            }
            for subset_id, metrics in historical_groups.items()
        }
    fairness_summary = sources["uab_fairness"]["payload"]
    confirm_groups = fairness_summary["subjective"]["groups"]
    return {
        subset_id: {
            "rmse": float(confirm_groups[subset_id]["rmse"]),
            "historical_rmse": float(historical_groups[subset_id]["rmse"]),
            "source_path": sources["uab_fairness"]["path"],
        }
        for subset_id in historical_groups
    }


def _extract_nasa_fusion_metrics(
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, float | str]:
    key = "nasa_fusion_confirm" if "nasa_fusion_confirm" in sources else "public_fusion_screen"
    payload = sources[key]["payload"]
    if "per_dataset_rankings" in payload:
        best_row = payload["per_dataset_rankings"]["nasa_csm"][0]
        return {
            "candidate_id": best_row["candidate_id"],
            "macro_f1": float(best_row["selection_score"]),
            "balanced_accuracy": float(best_row["secondary_score"]),
            "source_path": sources[key]["path"],
            "runtime_device": str(payload.get("runtime_device", "unknown")),
            "source_type": "fusion_screen",
        }
    combined = payload["objective"]["groups"]["combined"]
    return {
        "candidate_id": str(Path(str(payload["artifact_root"])).name),
        "macro_f1": float(combined["macro_f1"]),
        "balanced_accuracy": float(combined["balanced_accuracy"]),
        "source_path": sources[key]["path"],
        "runtime_device": str(payload.get("runtime_device", "unknown")),
        "source_type": "full_confirm",
    }


def _build_cleanup_audit(
    *,
    workspace_root: Path,
    report_root: Path,
    keep_names: set[str],
) -> dict[str, object]:
    corpus = list(_iter_repo_texts(workspace_root))
    report_candidates = []
    stage_i_report_paths = sorted(report_root.glob("*.md"))
    for path in stage_i_report_paths:
        if path.name in keep_names:
            continue
        hits = _find_reference_hits(
            corpus=corpus,
            token=path.name,
            ignore_paths={str(path.relative_to(workspace_root))},
        )
        blocking_hits = [
            hit
            for hit in hits
            if hit["path"]
            not in {
                "docs/README.md",
                "docs/artifacts/ARTIFACTS.md",
                "docs/artifacts/stage_i/README.md",
            }
            and not str(hit["path"]).startswith("tests/")
        ]
        report_candidates.append(
            {
                "path": str(path.relative_to(workspace_root)),
                "suggested_archive_path": str(
                    Path("docs/artifacts/stage_i/archive/public_history") / path.name
                ),
                "reference_hits": hits,
                "safe_to_move": not blocking_hits,
            }
        )
    unreferenced_code_candidates = []
    coverage_rows = []
    for path in sorted((workspace_root / "src/chronaris/pipelines/stage_i").glob("*.py")):
        module_token = f"chronaris.pipelines.stage_i.{path.stem}"
        stem_token = path.stem
        hits = _find_reference_hits(
            corpus=corpus,
            token=module_token,
            ignore_paths={str(path.relative_to(workspace_root))},
        )
        basename_hits = _find_reference_hits(
            corpus=corpus,
            token=path.name,
            ignore_paths={str(path.relative_to(workspace_root))},
        )
        stem_hits = _find_reference_hits(
            corpus=corpus,
            token=stem_token,
            ignore_paths={str(path.relative_to(workspace_root))},
        )
        test_hits = [
            hit["path"]
            for hit in hits + basename_hits + stem_hits
            if hit["path"].startswith("tests/")
        ]
        coverage_rows.append(
            {
                "path": str(path.relative_to(workspace_root)),
                "test_hit_count": len(set(test_hits)),
                "test_hits": sorted(set(test_hits)),
            }
        )
        if path.name == "__init__.py":
            continue
        external_hits = {
            hit["path"]
            for hit in hits + basename_hits + stem_hits
            if hit["path"] != str(path.relative_to(workspace_root))
        }
        if not external_hits:
            unreferenced_code_candidates.append(
                {
                    "path": str(path.relative_to(workspace_root)),
                    "safe_to_delete": True,
                    "reference_hits": [],
                    "test_hits": [],
                }
            )
    return {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "outdated_report_candidates": report_candidates,
        "unreferenced_code_candidates": unreferenced_code_candidates,
        "test_coverage_status": coverage_rows,
        "summary": {
            "outdated_doc_candidate_count": len(report_candidates),
            "unreferenced_code_candidate_count": len(unreferenced_code_candidates),
        },
    }


def _iter_repo_texts(workspace_root: Path) -> Iterable[tuple[str, list[str]]]:
    for root_name in TEXT_SCAN_ROOTS:
        root_path = workspace_root / root_name
        if root_name == "AGENTS.md":
            yield (root_name, root_path.read_text(encoding="utf-8").splitlines())
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SCAN_SUFFIXES:
                continue
            relative_path = str(path.relative_to(workspace_root))
            if relative_path.startswith("docs/artifacts/assets/") or "__pycache__" in relative_path:
                continue
            yield (relative_path, path.read_text(encoding="utf-8").splitlines())


def _find_reference_hits(
    *,
    corpus: Sequence[tuple[str, list[str]]],
    token: str,
    ignore_paths: set[str],
) -> list[dict[str, object]]:
    hits = []
    for relative_path, lines in corpus:
        if relative_path in ignore_paths:
            continue
        matched_lines = [
            line_number
            for line_number, line in enumerate(lines, start=1)
            if token in line
        ]
        if matched_lines:
            hits.append({"path": relative_path, "line_numbers": matched_lines[:10]})
    return hits


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


def _infer_timestamp(payload: Mapping[str, object], source_path: str):
    candidates = [
        payload.get("generated_at_utc"),
        payload.get("generated_at"),
        payload.get("run_id"),
        Path(source_path).parent.name,
        Path(source_path).name,
    ]
    for candidate in candidates:
        timestamp = _parse_timestamp_candidate(candidate)
        if timestamp is not None:
            return timestamp
    return pd.Timestamp(Path(source_path).stat().st_mtime, unit="s", tz="UTC")


def _parse_timestamp_candidate(candidate: object):
    if not candidate or not isinstance(candidate, str):
        return None
    if candidate.endswith("Z") and "T" in candidate:
        try:
            return pd.Timestamp(candidate)
        except Exception:
            pass
    match = re.search(r"(20\d{6}T\d{6}Z)", candidate)
    if not match:
        return None
    try:
        return pd.Timestamp(match.group(1))
    except Exception:
        return None


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


def render_stage_i_midterm_report(
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, object]],
    font_selection: PlotFontSelection,
    metrics_frame: pd.DataFrame,
    cleanup_audit: Mapping[str, object],
    figures: Sequence[Mapping[str, object]],
    artifact_root: Path,
) -> str:
    public_mainline = sources["public_mainline"]["payload"]
    private_payload = sources["private"]["payload"]
    confirm = _extract_nasa_fusion_metrics(sources)
    fairness = _extract_uab_fairness_metrics(sources)
    fairness_text = ", ".join(
        f"{subset}={payload['rmse']:.4f} (hist={payload['historical_rmse']:.4f})"
        for subset, payload in fairness.items()
    )
    figure_lines = [f"- `{item['figure_id']}`: `{item['path']}`" for item in figures]
    metric_preview = _render_markdown_table(metrics_frame.head(12))
    if float(confirm["macro_f1"]) <= 0.40:
        gate_text = "negative evidence"
    elif confirm.get("source_type") == "full_confirm":
        gate_text = "full confirm passed"
    else:
        gate_text = "needs full confirm"
    return "\n".join(
        [
            f"# Stage I Midterm Evidence - {run_id}",
            "",
            "## 1. 总判断",
            "",
            f"- 当前私有主线：`chronaris_opt`，`private_optimality_supported={private_payload['private_optimality_supported']}`。",
            f"- 当前公开主线：`{public_mainline['public_mainline_status']}`。",
            f"- NASA public fusion confirm：`macro-F1={float(confirm['macro_f1']):.4f}`，相对 `0.40` 门槛结论为 `{gate_text}`。",
            f"- UAB fairness confirm：`{fairness_text}`。",
            "",
            "## 2. 字体与图件",
            "",
            f"- 字体策略：`{font_selection.note}`",
            f"- 图件根目录：`{artifact_root / 'plots'}`",
            *figure_lines,
            "",
            "## 3. 指标摘录",
            "",
            metric_preview,
            "",
            "## 4. Cleanup Audit",
            "",
            f"- outdated_report_candidates：`{cleanup_audit['summary']['outdated_doc_candidate_count']}`",
            f"- unreferenced_code_candidates：`{cleanup_audit['summary']['unreferenced_code_candidate_count']}`",
            f"- cleanup_audit_path：`{artifact_root / 'cleanup_audit.json'}`",
            "",
            "## 5. 说明",
            "",
            "- 本报告只整编当前仓库已验证事实与本轮 confirm，不改写 Stage E/F/G/H contract。",
            "- UAB `target_prior_median` 仍只写成 `uab_public_adapter` / calibration baseline，不包装成融合模块本体胜利。",
            "- 若后续环境补强了 `fonts-wqy-zenhei` 或 `fonts-noto-cjk`，可直接重跑本入口重导中文标题图件。",
        ]
    )


def _render_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "- no metrics"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_manifest_summary(
    *,
    config: StageIMidtermEvidenceConfig,
    run_root: Path,
    metrics_frame: pd.DataFrame,
    figures: Sequence[Mapping[str, object]],
    cleanup_audit: Mapping[str, object],
    font_selection: PlotFontSelection,
    sources: Mapping[str, Mapping[str, object]],
    report_path: Path,
    metrics_path: Path,
    figure_index_path: Path,
    cleanup_audit_path: Path,
) -> dict[str, object]:
    public_mainline = sources["public_mainline"]["payload"]
    private_payload = sources["private"]["payload"]
    confirm = _extract_nasa_fusion_metrics(sources)
    return {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "report_path": str(report_path),
        "midterm_metrics_path": str(metrics_path),
        "midterm_figure_index_path": str(figure_index_path),
        "cleanup_audit_path": str(cleanup_audit_path),
        "font_selection": {
            "family": font_selection.family,
            "ascii_only": font_selection.ascii_only,
            "note": font_selection.note,
        },
        "sources": {
            name: {"path": payload["path"]}
            for name, payload in sources.items()
        },
        "status": {
            "private_optimality_supported": private_payload["private_optimality_supported"],
            "public_mainline_status": public_mainline["public_mainline_status"],
            "nasa_fusion_macro_f1": float(confirm["macro_f1"]),
            "nasa_fusion_gate_passed": float(confirm["macro_f1"]) > 0.40,
        },
        "counts": {
            "metric_row_count": len(metrics_frame),
            "figure_count": len(figures),
            "outdated_doc_candidate_count": cleanup_audit["summary"]["outdated_doc_candidate_count"],
            "unreferenced_code_candidate_count": cleanup_audit["summary"]["unreferenced_code_candidate_count"],
        },
    }
