"""Key-condition anchor selection over frozen Stage H assets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.features import load_stage_i_case_study_run
from chronaris.pipelines.stage_i.stage_i_case_study import (
    StageICaseStudyViewResult,
    build_stage_i_case_study_results,
)

DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH = (
    "docs/reports/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/"
    "private_benchmark_summary.json"
)
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIAnchorConfig:
    """Config for thesis-facing key-condition anchor export."""

    run_id: str
    stage_h_run_manifest_path: str
    output_root: str = "docs/reports/assets/stage_i_anchor"
    report_root: str = "docs/reports/stage_i"
    private_benchmark_summary_path: str | None = DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH
    top_k_windows: int = 5
    view_verdict_filter: str = "all"
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class StageIAnchorRunResult:
    """Artifacts produced by one anchor-selection pass."""

    run_id: str
    artifact_root: str
    anchor_manifest_path: str
    anchor_windows_csv_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_anchor(
    config: StageIAnchorConfig,
) -> StageIAnchorRunResult:
    """Select stable key-condition anchors from frozen Stage H views."""

    LOGGER.info(
        "stage_i_anchor start run_id=%s stage_h_run_manifest=%s private_summary=%s",
        config.run_id,
        config.stage_h_run_manifest_path,
        config.private_benchmark_summary_path,
    )
    run_input = load_stage_i_case_study_run(config.stage_h_run_manifest_path)
    view_results, pilot_comparisons = build_stage_i_case_study_results(
        run_input,
        top_k_windows=config.top_k_windows,
        device=config.device,
    )
    filtered_results = _filter_view_results(
        view_results,
        verdict_filter=config.view_verdict_filter,
    )
    anchor_rows = _build_anchor_rows(
        filtered_results,
        pilot_comparisons=pilot_comparisons,
    )
    LOGGER.info(
        "stage_i_anchor selected views=%d anchors=%d verdict_filter=%s",
        len(filtered_results),
        len(anchor_rows),
        config.view_verdict_filter,
    )
    private_no_mask = _load_private_no_mask_summary(config.private_benchmark_summary_path)

    artifact_root = Path(config.output_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "anchor_manifest.json"
    windows_csv_path = artifact_root / "anchor_windows.csv"
    report_path = report_root / f"stage-i-anchor-{config.run_id}.md"

    pd.DataFrame(anchor_rows).to_csv(windows_csv_path, index=False)
    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "artifact_root": str(artifact_root),
        "source_paths": {
            "stage_h_run_manifest_path": config.stage_h_run_manifest_path,
            "private_benchmark_summary_path": config.private_benchmark_summary_path,
        },
        "selection_policy": {
            "top_k_windows_per_view": config.top_k_windows,
            "view_verdict_filter": config.view_verdict_filter,
            "anchor_score_formula": (
                "top_contribution + 0.5*top_event + 0.25*abs(pair_delta_top_contribution) "
                "+ 0.1*abs(pair_delta_projection_cosine) + warn_bonus"
            ),
            "warn_bonus": 0.25,
        },
        "overview": {
            "selected_view_count": len(filtered_results),
            "selected_anchor_count": len(anchor_rows),
            "view_verdict_counts": dict(
                Counter(
                    result.view_summary.verdict
                    for result in filtered_results
                )
            ),
        },
        "view_summaries": [
            _build_view_summary_row(
                result,
                pilot_comparisons=pilot_comparisons,
            )
            for result in filtered_results
        ],
        "pilot_comparisons": [comparison.to_dict() for comparison in pilot_comparisons],
        "private_no_mask_summary": private_no_mask,
        "anchors": anchor_rows,
        "anchor_windows_csv_path": str(windows_csv_path),
    }
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_stage_i_anchor_report(summary) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "stage_i_anchor finished run_id=%s manifest_path=%s report_path=%s",
        config.run_id,
        manifest_path,
        report_path,
    )
    return StageIAnchorRunResult(
        run_id=config.run_id,
        artifact_root=str(artifact_root),
        anchor_manifest_path=str(manifest_path),
        anchor_windows_csv_path=str(windows_csv_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_stage_i_anchor_report(summary: Mapping[str, object]) -> str:
    """Render the markdown anchor report."""

    lines = [
        f"# Stage I Anchor Report - {summary['run_id']}",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- stage_h_run_manifest_path: `{summary['source_paths']['stage_h_run_manifest_path']}`",
        f"- anchor_windows_csv_path: `{summary['anchor_windows_csv_path']}`",
        "",
        "## Overview",
        "",
        f"- selected_view_count: `{summary['overview']['selected_view_count']}`",
        f"- selected_anchor_count: `{summary['overview']['selected_anchor_count']}`",
        f"- view_verdict_counts: `{summary['overview']['view_verdict_counts']}`",
        "",
        "## Selection Policy",
        "",
        f"- top_k_windows_per_view: `{summary['selection_policy']['top_k_windows_per_view']}`",
        f"- view_verdict_filter: `{summary['selection_policy']['view_verdict_filter']}`",
        f"- anchor_score_formula: `{summary['selection_policy']['anchor_score_formula']}`",
        "",
        "## View Summary",
        "",
        "| view | verdict | strongest ablation | strongest delta top contribution | strongest delta top event | paired abs delta cosine | paired abs delta contribution |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["view_summaries"]:
        lines.append(
            f"| `{row['view_id']}` | `{row['verdict']}` | `{row['strongest_ablation_name']}` | "
            f"{row['strongest_ablation_delta_top_contribution']:.6f} | "
            f"{row['strongest_ablation_delta_top_event']:.6f} | "
            f"{row['paired_abs_delta_mean_projection_cosine']:.6f} | "
            f"{row['paired_abs_delta_mean_top_contribution_score']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Anchor Windows",
            "",
            "| rank | view | sample | window index | verdict | anchor score | top event | top contribution | strongest ablation |",
            "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["anchors"]:
        lines.append(
            f"| {row['anchor_rank']} | `{row['view_id']}` | `{row['sample_id']}` | "
            f"{row['window_index']} | `{row['view_verdict']}` | {row['anchor_score']:.6f} | "
            f"{row['top_event_score']:.6f} | {row['top_contribution_score']:.6f} | "
            f"`{row['strongest_ablation_name']}` |"
        )

    lines.extend(["", "## Paired-Pilot Comparison", ""])
    if summary["pilot_comparisons"]:
        lines.extend(
            [
                "| sortie | reference view | comparison view | delta mean cosine | delta top contribution |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in summary["pilot_comparisons"]:
            lines.append(
                f"| `{row['sortie_id']}` | `{row['reference_view_id']}` | `{row['comparison_view_id']}` | "
                f"{row['delta_mean_projection_cosine']:+.6f} | "
                f"{row['delta_mean_top_contribution_score']:+.6f} |"
            )
    else:
        lines.append("- no_paired_pilot_comparison")

    lines.extend(["", "## Private No-Mask Comparison", ""])
    private_summary = summary["private_no_mask_summary"]
    if private_summary["available"]:
        lines.extend(
            [
                "| task | target metrics | no-mask metrics | target beats no-mask |",
                "| --- | --- | --- | --- |",
            ]
        )
        for task_id, row in private_summary["tasks"].items():
            lines.append(
                f"| `{task_id}` | `{row['target_metrics']}` | `{row['no_mask_metrics']}` | "
                f"`{row['target_beats_no_mask']}` |"
            )
    else:
        lines.append("- private_no_mask_summary_unavailable")
    return "\n".join(lines)


def _filter_view_results(
    view_results: tuple[StageICaseStudyViewResult, ...],
    *,
    verdict_filter: str,
) -> tuple[StageICaseStudyViewResult, ...]:
    if verdict_filter == "all":
        return view_results
    if verdict_filter == "warn_only":
        return tuple(
            result
            for result in view_results
            if result.view_summary.verdict == "WARN"
        )
    if verdict_filter == "pass_only":
        return tuple(
            result
            for result in view_results
            if result.view_summary.verdict == "PASS"
        )
    raise ValueError(f"unsupported view_verdict_filter: {verdict_filter}")


def _build_anchor_rows(
    view_results: tuple[StageICaseStudyViewResult, ...],
    *,
    pilot_comparisons,
) -> list[dict[str, object]]:
    paired_gap_by_view = _build_paired_gap_by_view(pilot_comparisons)
    rows: list[dict[str, object]] = []
    for result in view_results:
        strongest_ablation = _select_strongest_ablation(result)
        paired_gap = paired_gap_by_view.get(
            result.view_summary.view_id,
            {
                "paired_abs_delta_mean_projection_cosine": 0.0,
                "paired_abs_delta_mean_top_contribution_score": 0.0,
            },
        )
        warn_bonus = 0.25 if result.view_summary.verdict == "WARN" else 0.0
        for rank_in_view, window in enumerate(result.top_windows, start=1):
            anchor_score = (
                float(window.top_contribution_score)
                + 0.5 * float(window.top_event_score)
                + 0.25 * float(
                    paired_gap["paired_abs_delta_mean_top_contribution_score"]
                )
                + 0.1 * float(
                    paired_gap["paired_abs_delta_mean_projection_cosine"]
                )
                + warn_bonus
            )
            rows.append(
                {
                    "anchor_id": f"{result.view_summary.view_id}:{window.sample_id}",
                    "view_id": result.view_summary.view_id,
                    "sortie_id": result.view_summary.sortie_id,
                    "pilot_id": result.view_summary.pilot_id,
                    "view_verdict": result.view_summary.verdict,
                    "sample_id": window.sample_id,
                    "window_index": window.window_index,
                    "start_offset_ms": window.start_offset_ms,
                    "end_offset_ms": window.end_offset_ms,
                    "rank_in_view": rank_in_view,
                    "anchor_score": anchor_score,
                    "top_event_offset_s": float(window.top_event_offset_s),
                    "top_event_score": float(window.top_event_score),
                    "top_contribution_offset_s": float(
                        window.top_contribution_offset_s
                    ),
                    "top_contribution_score": float(
                        window.top_contribution_score
                    ),
                    "strongest_ablation_name": strongest_ablation["name"],
                    "strongest_ablation_delta_top_event": strongest_ablation[
                        "delta_mean_top_event_score"
                    ],
                    "strongest_ablation_delta_top_contribution": strongest_ablation[
                        "delta_mean_top_contribution_score"
                    ],
                    **paired_gap,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["anchor_score"]),
            row["view_verdict"] != "WARN",
            -float(row["top_contribution_score"]),
            str(row["sample_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["anchor_rank"] = index
    return rows


def _build_view_summary_row(
    result: StageICaseStudyViewResult,
    *,
    pilot_comparisons,
) -> dict[str, object]:
    strongest_ablation = _select_strongest_ablation(result)
    paired_gap = _build_paired_gap_by_view(pilot_comparisons).get(
        result.view_summary.view_id,
        {
            "paired_abs_delta_mean_projection_cosine": 0.0,
            "paired_abs_delta_mean_top_contribution_score": 0.0,
        },
    )
    return {
        "view_id": result.view_summary.view_id,
        "sortie_id": result.view_summary.sortie_id,
        "pilot_id": result.view_summary.pilot_id,
        "verdict": result.view_summary.verdict,
        "strongest_ablation_name": strongest_ablation["name"],
        "strongest_ablation_delta_top_event": strongest_ablation[
            "delta_mean_top_event_score"
        ],
        "strongest_ablation_delta_top_contribution": strongest_ablation[
            "delta_mean_top_contribution_score"
        ],
        **paired_gap,
    }


def _select_strongest_ablation(
    result: StageICaseStudyViewResult,
) -> dict[str, float | str]:
    non_baseline = tuple(
        ablation
        for ablation in result.ablations
        if ablation.name != "projection_refusion_baseline"
    )
    strongest = min(
        non_baseline,
        key=lambda item: item.delta_mean_top_contribution_score,
    )
    return {
        "name": strongest.name,
        "delta_mean_top_event_score": float(
            strongest.delta_mean_top_event_score
        ),
        "delta_mean_top_contribution_score": float(
            strongest.delta_mean_top_contribution_score
        ),
    }


def _build_paired_gap_by_view(
    pilot_comparisons,
) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {}
    for comparison in pilot_comparisons:
        payload = {
            "paired_abs_delta_mean_projection_cosine": abs(
                float(comparison.delta_mean_projection_cosine)
            ),
            "paired_abs_delta_mean_top_contribution_score": abs(
                float(comparison.delta_mean_top_contribution_score)
            ),
        }
        mapping[comparison.reference_view_id] = payload
        mapping[comparison.comparison_view_id] = payload
    return mapping


def _load_private_no_mask_summary(path_like: str | None) -> dict[str, object]:
    if not path_like:
        return {"available": False, "tasks": {}}
    path = Path(path_like)
    if not path.exists():
        return {"available": False, "tasks": {}}
    summary = json.loads(path.read_text(encoding="utf-8"))
    target_variant_name = str(
        summary.get("conclusion", {}).get(
            "target_variant_name",
            summary.get("target_variant_name", "chronaris_opt"),
        )
    )
    no_mask_variant_name = str(
        summary.get("conclusion", {}).get(
            "no_mask_variant_name",
            f"{target_variant_name}_no_causal_mask",
        )
    )
    tasks: dict[str, dict[str, object]] = {}
    for task_id, payload in summary.get("tasks", {}).items():
        variants = payload.get("variants", {})
        target_payload = variants.get(target_variant_name, {})
        no_mask_payload = variants.get(no_mask_variant_name, {})
        tasks[task_id] = {
            "target_metrics": _format_variant_metrics(target_payload),
            "no_mask_metrics": _format_variant_metrics(no_mask_payload),
            "target_beats_no_mask": _target_beats_no_mask(
                task_type=str(payload.get("task_type", "")),
                target_payload=target_payload,
                no_mask_payload=no_mask_payload,
            ),
        }
    return {
        "available": bool(tasks),
        "target_variant_name": target_variant_name,
        "no_mask_variant_name": no_mask_variant_name,
        "tasks": tasks,
    }


def _format_variant_metrics(payload: Mapping[str, object]) -> str:
    if not payload:
        return "not_available"
    if "best_metrics" in payload:
        metrics = payload["best_metrics"]
        if "macro_f1" in metrics and "balanced_accuracy" in metrics:
            return (
                f"macro_f1={float(metrics['macro_f1']):.6f}, "
                f"balanced_accuracy={float(metrics['balanced_accuracy']):.6f}"
            )
        if "rmse" in metrics and "mae" in metrics:
            return (
                f"rmse={float(metrics['rmse']):.6f}, "
                f"mae={float(metrics['mae']):.6f}"
            )
    if "top1_accuracy" in payload and "mrr" in payload:
        return (
            f"top1_accuracy={float(payload['top1_accuracy']):.6f}, "
            f"mrr={float(payload['mrr']):.6f}"
        )
    return "not_available"


def _target_beats_no_mask(
    *,
    task_type: str,
    target_payload: Mapping[str, object],
    no_mask_payload: Mapping[str, object],
) -> bool:
    if task_type == "classification":
        return float(target_payload.get("best_metrics", {}).get("macro_f1", 0.0)) > float(
            no_mask_payload.get("best_metrics", {}).get("macro_f1", 0.0)
        )
    if task_type == "regression":
        return float(target_payload.get("best_metrics", {}).get("rmse", float("inf"))) < float(
            no_mask_payload.get("best_metrics", {}).get("rmse", float("inf"))
        )
    if task_type == "retrieval":
        return float(target_payload.get("top1_accuracy", 0.0)) > float(
            no_mask_payload.get("top1_accuracy", 0.0)
        )
    return False
