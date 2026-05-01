"""Stage I Phase 3 orchestration and closure helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chronaris.dataset import build_nasa_csm_task_entries, build_uab_task_entries
from chronaris.features import build_nasa_csm_feature_table, build_uab_feature_table
from chronaris.pipelines.stage_i_baseline import run_stage_i_baselines, write_baseline_artifacts
from chronaris.pipelines.stage_i_phase3_assets import (
    build_stage_i_dataset_summary,
    build_uab_session_window_comparison,
    load_stage_i_baseline_artifacts,
    read_stage_i_dataset_summary,
    summarize_artifact_bundle,
    write_local_baseline_report,
    write_prepared_dataset,
)
from chronaris.pipelines.stage_i_phase3_reporting import (
    render_stage_i_phase3_report,
    write_phase3_summary_plots,
)


@dataclass(frozen=True, slots=True)
class StageIPhase3Config:
    dataset_root: str
    output_root: str
    run_id: str
    prior_uab_session_artifact_root: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPhase3RunResult:
    artifact_root: str
    uab_artifact_root: str
    nasa_artifact_root: str
    closure_summary_path: str
    closure_report_path: str
    closure_summary: Mapping[str, object]


def run_stage_i_phase3(config: StageIPhase3Config) -> StageIPhase3RunResult:
    """Prepare UAB/NASA Phase 3 assets and write a closure summary."""

    artifact_root = Path(config.output_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    uab_artifact_root = artifact_root / "uab_window"
    nasa_artifact_root = artifact_root / "nasa_attention"
    uab_artifact_root.mkdir(parents=True, exist_ok=True)
    nasa_artifact_root.mkdir(parents=True, exist_ok=True)

    uab_entries = build_uab_task_entries(config.dataset_root, profile="window_v2").entries
    uab_feature_result = build_uab_feature_table(config.dataset_root, uab_entries)
    uab_summary = build_stage_i_dataset_summary(uab_entries, uab_feature_result)
    write_prepared_dataset(
        artifact_root=uab_artifact_root,
        entries=uab_entries,
        feature_result=uab_feature_result,
        summary=uab_summary,
    )
    uab_baselines = run_stage_i_baselines(
        uab_feature_result.feature_table,
        dataset_id="uab_workload_dataset",
        profile="window_v2",
        task_family="workload",
        artifact_root=uab_artifact_root,
    )
    write_baseline_artifacts(uab_baselines, artifact_root=uab_artifact_root)
    uab_report_path = write_local_baseline_report(
        artifact_root=uab_artifact_root,
        dataset_summary=uab_summary,
        baselines=uab_baselines,
    )

    nasa_entries = build_nasa_csm_task_entries(config.dataset_root).entries
    nasa_feature_result = build_nasa_csm_feature_table(config.dataset_root, nasa_entries)
    nasa_summary = build_stage_i_dataset_summary(nasa_entries, nasa_feature_result)
    write_prepared_dataset(
        artifact_root=nasa_artifact_root,
        entries=nasa_entries,
        feature_result=nasa_feature_result,
        summary=nasa_summary,
    )
    nasa_baselines = run_stage_i_baselines(
        nasa_feature_result.feature_table,
        dataset_id="nasa_csm",
        profile="window_v2",
        task_family="attention_state",
        artifact_root=nasa_artifact_root,
    )
    write_baseline_artifacts(nasa_baselines, artifact_root=nasa_artifact_root)
    nasa_report_path = write_local_baseline_report(
        artifact_root=nasa_artifact_root,
        dataset_summary=nasa_summary,
        baselines=nasa_baselines,
    )

    comparison_summary = build_uab_session_window_comparison(
        config.prior_uab_session_artifact_root,
        uab_baselines,
    )
    closure_summary = _build_closure_summary(
        artifact_root=artifact_root,
        run_id=config.run_id,
        uab_bundle=summarize_artifact_bundle(uab_artifact_root, uab_summary, uab_baselines),
        nasa_bundle=summarize_artifact_bundle(nasa_artifact_root, nasa_summary, nasa_baselines),
        comparison_summary=comparison_summary,
        uab_report_path=uab_report_path,
        nasa_report_path=nasa_report_path,
    )
    return _write_closure_outputs(artifact_root, closure_summary)


def compose_stage_i_phase3_closure(
    *,
    artifact_root: str | Path,
    prior_uab_session_artifact_root: str | None = None,
) -> StageIPhase3RunResult:
    """Compose the Phase 3 closure summary from existing artifact folders."""

    artifact_root = Path(artifact_root)
    uab_artifact_root = artifact_root / "uab_window"
    nasa_artifact_root = artifact_root / "nasa_attention"
    uab_summary = read_stage_i_dataset_summary(uab_artifact_root)
    nasa_summary = read_stage_i_dataset_summary(nasa_artifact_root)
    uab_baselines = load_stage_i_baseline_artifacts(uab_artifact_root, require_subjective=True)
    nasa_baselines = load_stage_i_baseline_artifacts(nasa_artifact_root)
    uab_report_path = write_local_baseline_report(
        artifact_root=uab_artifact_root,
        dataset_summary=uab_summary,
        baselines=uab_baselines,
    )
    nasa_report_path = write_local_baseline_report(
        artifact_root=nasa_artifact_root,
        dataset_summary=nasa_summary,
        baselines=nasa_baselines,
    )

    closure_summary = _build_closure_summary(
        artifact_root=artifact_root,
        run_id=artifact_root.name,
        uab_bundle=summarize_artifact_bundle(uab_artifact_root, uab_summary, uab_baselines),
        nasa_bundle=summarize_artifact_bundle(nasa_artifact_root, nasa_summary, nasa_baselines),
        comparison_summary=build_uab_session_window_comparison(
            prior_uab_session_artifact_root,
            uab_baselines,
        ),
        uab_report_path=uab_report_path,
        nasa_report_path=nasa_report_path,
    )
    return _write_closure_outputs(artifact_root, closure_summary)


def _build_closure_summary(
    *,
    artifact_root: Path,
    run_id: str,
    uab_bundle: Mapping[str, object],
    nasa_bundle: Mapping[str, object],
    comparison_summary: Mapping[str, object],
    uab_report_path: Path,
    nasa_report_path: Path,
) -> dict[str, object]:
    closure_summary = {
        "phase": "stage_i_phase3",
        "run_id": run_id,
        "generated_at_utc": pd_timestamp_utc(),
        "artifact_root": str(artifact_root),
        "uab_window": uab_bundle,
        "nasa_attention": nasa_bundle,
        "uab_session_comparison": comparison_summary,
        "reports": {
            "uab_baseline_report": str(uab_report_path),
            "nasa_baseline_report": str(nasa_report_path),
        },
    }
    closure_summary["plot_paths"] = write_phase3_summary_plots(artifact_root, closure_summary)
    return closure_summary


def _write_closure_outputs(
    artifact_root: Path,
    closure_summary: Mapping[str, object],
) -> StageIPhase3RunResult:
    closure_summary_path = artifact_root / "closure_summary.json"
    closure_summary_path.write_text(json.dumps(closure_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    closure_report_path = artifact_root / "stage_i_phase3_closure_report.md"
    closure_report_path.write_text(render_stage_i_phase3_report(closure_summary) + "\n", encoding="utf-8")
    return StageIPhase3RunResult(
        artifact_root=str(artifact_root),
        uab_artifact_root=str(artifact_root / "uab_window"),
        nasa_artifact_root=str(artifact_root / "nasa_attention"),
        closure_summary_path=str(closure_summary_path),
        closure_report_path=str(closure_report_path),
        closure_summary=closure_summary,
    )


def pd_timestamp_utc() -> str:
    import pandas as pd

    return pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z")
