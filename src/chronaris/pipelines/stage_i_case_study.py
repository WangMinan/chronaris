"""Phase 2 case-study pipeline over frozen Stage H assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from chronaris.evaluation import (
    StageICaseStudyAblationMetrics,
    StageICaseStudyPilotComparison,
    StageICaseStudyViewSummary,
    StageICaseStudyWindowRanking,
    build_pilot_comparisons,
    build_view_summary,
    build_window_rankings,
    compute_case_study_ablations,
    explain_warn_view,
)
from chronaris.features import StageICaseStudyViewInput, load_stage_i_case_study_run


@dataclass(frozen=True, slots=True)
class StageICaseStudyConfig:
    """Frozen configuration for the Stage I Phase 2 run."""

    run_id: str
    stage_h_run_manifest_path: str
    output_root: str
    report_path: str
    top_k_windows: int = 5

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "stage_h_run_manifest_path": self.stage_h_run_manifest_path,
            "output_root": self.output_root,
            "report_path": self.report_path,
            "top_k_windows": self.top_k_windows,
            "ablation_paths": [
                "projection_refusion_baseline",
                "no_event_bias",
                "no_state_normalization",
                "vehicle_delta_suppressed",
            ],
        }


@dataclass(frozen=True, slots=True)
class StageICaseStudyViewResult:
    """One view-level Phase 2 result bundle."""

    view_summary: StageICaseStudyViewSummary
    ablations: tuple[StageICaseStudyAblationMetrics, ...]
    top_windows: tuple[StageICaseStudyWindowRanking, ...]
    warn_explanation: str | None


@dataclass(frozen=True, slots=True)
class StageICaseStudyRunResult:
    """Machine outputs and report paths from one Phase 2 run."""

    config: StageICaseStudyConfig
    generated_at_utc: str
    artifact_root: str
    report_path: str
    view_results: tuple[StageICaseStudyViewResult, ...]
    pilot_comparisons: tuple[StageICaseStudyPilotComparison, ...]
    summary_path: str
    view_summary_csv_path: str
    ablation_summary_csv_path: str
    window_rankings_csv_path: str


def run_stage_i_case_study(
    config: StageICaseStudyConfig,
) -> StageICaseStudyRunResult:
    """Run the fixed Phase 2 case-study family over a Stage H run."""

    run_input = load_stage_i_case_study_run(config.stage_h_run_manifest_path)
    provisional_results: list[tuple[StageICaseStudyViewInput, tuple[StageICaseStudyAblationMetrics, ...], np.ndarray]] = []
    for view in run_input.views:
        ablations, baseline_fused_states = compute_case_study_ablations(view)
        provisional_results.append((view, ablations, baseline_fused_states))

    view_summaries = tuple(
        build_view_summary(view, baseline=ablations[0], baseline_fused_states=baseline_fused_states)
        for view, ablations, baseline_fused_states in provisional_results
    )
    pilot_comparisons = build_pilot_comparisons(view_summaries)
    view_results = tuple(
        StageICaseStudyViewResult(
            view_summary=summary,
            ablations=ablations,
            top_windows=build_window_rankings(view, baseline=ablations[0], top_k=config.top_k_windows),
            warn_explanation=(
                explain_warn_view(
                    view,
                    view_summary=summary,
                    ablations=ablations,
                    pilot_comparisons=pilot_comparisons,
                )
                if summary.verdict != "PASS"
                else None
            ),
        )
        for summary, (view, ablations, _baseline_fused_states) in zip(view_summaries, provisional_results, strict=True)
    )

    artifact_root = Path(config.output_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    generated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary_path = artifact_root / "case_study_summary.json"
    view_summary_csv_path = artifact_root / "view_summary.csv"
    ablation_summary_csv_path = artifact_root / "ablation_summary.csv"
    window_rankings_csv_path = artifact_root / "window_rankings.csv"
    _write_case_study_artifacts(
        summary_path=summary_path,
        view_summary_csv_path=view_summary_csv_path,
        ablation_summary_csv_path=ablation_summary_csv_path,
        window_rankings_csv_path=window_rankings_csv_path,
        config=config,
        generated_at_utc=generated_at_utc,
        view_results=view_results,
        pilot_comparisons=pilot_comparisons,
    )
    return StageICaseStudyRunResult(
        config=config,
        generated_at_utc=generated_at_utc,
        artifact_root=str(artifact_root),
        report_path=config.report_path,
        view_results=view_results,
        pilot_comparisons=pilot_comparisons,
        summary_path=str(summary_path),
        view_summary_csv_path=str(view_summary_csv_path),
        ablation_summary_csv_path=str(ablation_summary_csv_path),
        window_rankings_csv_path=str(window_rankings_csv_path),
    )


def write_stage_i_case_study_report(
    result: StageICaseStudyRunResult,
) -> str:
    """Write the Phase 2 markdown report and return its path."""

    report_path = Path(result.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage_i_case_study_report(result) + "\n", encoding="utf-8")
    return str(report_path)


def render_stage_i_case_study_report(result: StageICaseStudyRunResult) -> str:
    """Render the Chinese Phase 2 report."""

    lines = [
        f"# Stage I Phase 2 Case Study - {result.config.run_id}",
        "",
        f"- generated at UTC: `{result.generated_at_utc}`",
        f"- Stage H run manifest: `{result.config.stage_h_run_manifest_path}`",
        f"- artifact root: `{result.artifact_root}`",
        f"- report path: `{result.report_path}`",
        f"- top-k windows per view: `{result.config.top_k_windows}`",
        "",
        "## Fixed Ablation Family",
        "",
        "1. `projection_refusion_baseline`",
        "2. `no_event_bias`",
        "3. `no_state_normalization`",
        "4. `vehicle_delta_suppressed`",
        "",
        "## View Summary",
        "",
        "- `hidden-vs-projection cosine` 使用 fused L2 norm profile 的 cosine；原因是当前 Stage H 导出的 hidden fused 为 `96` 维，而 projection rerun baseline 为 `48` 维，不能直接做逐向量余弦。",
        "",
        "| view | verdict | windows | case samples | mean cosine | cosine cv | l2 gap | l2 gap cv | mean attention entropy | mean top contribution | hidden-vs-projection cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view_result in result.view_results:
        summary = view_result.view_summary
        lines.append(
            f"| `{summary.view_id}` | `{summary.verdict}` | {summary.window_count} | "
            f"{summary.case_partition_sample_count} | {summary.mean_projection_cosine:.6f} | "
            f"{summary.projection_cosine_cv:.6f} | {summary.mean_projection_l2_gap:.6f} | "
            f"{summary.projection_l2_gap_cv:.6f} | {summary.mean_attention_entropy:.6f} | "
            f"{summary.mean_top_contribution_score:.6f} | "
            f"{summary.exported_hidden_fused_vs_projection_baseline_cosine:.6f} |"
        )
    lines.extend(["", "## Same-Sortie Pilot Comparison", ""])
    if result.pilot_comparisons:
        lines.extend(
            [
                "| sortie | reference view | comparison view | delta mean cosine | delta cosine cv | delta l2 gap | delta l2 gap cv | delta attention entropy | delta top contribution | delta hidden/projection cosine |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in result.pilot_comparisons:
            lines.append(
                f"| `{comparison.sortie_id}` | `{comparison.reference_view_id}` | `{comparison.comparison_view_id}` | "
                f"{comparison.delta_mean_projection_cosine:+.6f} | {comparison.delta_projection_cosine_cv:+.6f} | "
                f"{comparison.delta_mean_projection_l2_gap:+.6f} | {comparison.delta_projection_l2_gap_cv:+.6f} | "
                f"{comparison.delta_mean_attention_entropy:+.6f} | {comparison.delta_mean_top_contribution_score:+.6f} | "
                f"{comparison.delta_exported_hidden_fused_vs_projection_baseline_cosine:+.6f} |"
            )
    else:
        lines.append("- 本轮没有可比较的同 sortie 多 pilot view。")

    lines.extend(["", "## Ablation Summary", ""])
    for view_result in result.view_results:
        lines.extend(
            [
                f"### {view_result.view_summary.view_id}",
                "",
                "| ablation | mean attention entropy | delta entropy | mean max attention | delta max attention | mean top event | delta top event | mean top contribution | delta top contribution | mean fused L2 | delta fused L2 | cosine to projection baseline | delta cosine |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for ablation in view_result.ablations:
            lines.append(
                f"| `{ablation.name}` | {ablation.mean_attention_entropy:.6f} | "
                f"{ablation.delta_mean_attention_entropy:+.6f} | {ablation.mean_max_attention:.6f} | "
                f"{ablation.delta_mean_max_attention:+.6f} | {ablation.mean_top_event_score:.6f} | "
                f"{ablation.delta_mean_top_event_score:+.6f} | {ablation.mean_top_contribution_score:.6f} | "
                f"{ablation.delta_mean_top_contribution_score:+.6f} | {ablation.mean_fused_l2_norm:.6f} | "
                f"{ablation.delta_fused_l2_norm:+.6f} | {ablation.mean_fused_cosine_to_projection_baseline:.6f} | "
                f"{ablation.delta_fused_cosine_to_projection_baseline:+.6f} |"
            )
        lines.append("")

    lines.extend(["## Top Windows", ""])
    for view_result in result.view_results:
        lines.extend(
            [
                f"### {view_result.view_summary.view_id}",
                "",
                "| sample | window index | start ms | end ms | top event offset s | top event score | top contribution offset s | top contribution score |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for ranking in view_result.top_windows:
            lines.append(
                f"| `{ranking.sample_id}` | {ranking.window_index} | {ranking.start_offset_ms} | "
                f"{ranking.end_offset_ms} | {ranking.top_event_offset_s:.6f} | {ranking.top_event_score:.6f} | "
                f"{ranking.top_contribution_offset_s:.6f} | {ranking.top_contribution_score:.6f} |"
            )
        lines.append("")

    warn_results = tuple(result_item for result_item in result.view_results if result_item.warn_explanation)
    lines.extend(["## WARN View Interpretation", ""])
    if warn_results:
        for view_result in warn_results:
            lines.append(f"### {view_result.view_summary.view_id}")
            lines.append("")
            lines.append(view_result.warn_explanation or "")
            lines.append("")
    else:
        lines.append("- 本轮主线 view 无 `WARN`。")
        lines.append("")

    lines.extend(
        [
            "## Machine Artifacts",
            "",
            f"- summary json: `{result.summary_path}`",
            f"- view summary csv: `{result.view_summary_csv_path}`",
            f"- ablation summary csv: `{result.ablation_summary_csv_path}`",
            f"- window rankings csv: `{result.window_rankings_csv_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_case_study_artifacts(
    *,
    summary_path: Path,
    view_summary_csv_path: Path,
    ablation_summary_csv_path: Path,
    window_rankings_csv_path: Path,
    config: StageICaseStudyConfig,
    generated_at_utc: str,
    view_results: tuple[StageICaseStudyViewResult, ...],
    pilot_comparisons: tuple[StageICaseStudyPilotComparison, ...],
) -> None:
    view_summary_rows = [result.view_summary.to_dict() for result in view_results]
    ablation_rows = []
    window_rows = []
    for result in view_results:
        for ablation in result.ablations:
            payload = ablation.to_dict()
            payload["view_id"] = result.view_summary.view_id
            payload["sortie_id"] = result.view_summary.sortie_id
            payload["pilot_id"] = result.view_summary.pilot_id
            payload.pop("sample_metrics", None)
            ablation_rows.append(payload)
        window_rows.extend(ranking.to_dict() for ranking in result.top_windows)
    summary_payload = {
        "config": config.to_dict(),
        "generated_at_utc": generated_at_utc,
        "view_results": [
            {
                "view_summary": result.view_summary.to_dict(),
                "ablations": [ablation.to_dict() for ablation in result.ablations],
                "top_windows": [ranking.to_dict() for ranking in result.top_windows],
                "warn_explanation": result.warn_explanation,
            }
            for result in view_results
        ],
        "pilot_comparisons": [comparison.to_dict() for comparison in pilot_comparisons],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(view_summary_rows).to_csv(view_summary_csv_path, index=False)
    pd.DataFrame(ablation_rows).to_csv(ablation_summary_csv_path, index=False)
    pd.DataFrame(window_rows).to_csv(window_rankings_csv_path, index=False)
