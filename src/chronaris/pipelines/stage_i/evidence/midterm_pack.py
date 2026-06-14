"""Stage I midterm evidence-pack builder."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from chronaris.pipelines.stage_i.common.run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)
from chronaris.pipelines.stage_i.evidence.midterm_figures import (
    PlotFontSelection,
    _detect_plot_font,
    _write_figures,
)
from chronaris.pipelines.stage_i.evidence.midterm_metrics import (
    _extract_nasa_fusion_metrics,
    _extract_uab_fairness_metrics,
    _infer_timestamp,
    _primary_task_metric,
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
