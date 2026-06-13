"""Transfer-boundary report builder for public adapter evidence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_public_transfer_boundary"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_CALIBRATION_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_adapter_calibration/"
    "20260607T-stage-i-public-adapter-calibration-r1/public_adapter_calibration_summary.json"
)
DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_mainline/"
    "20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/public_mainline_summary.json"
)
DEFAULT_MULTITASK_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/"
    "multitask_summary.json"
)
DEFAULT_PRIVATE_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/"
    "private_benchmark_summary.json"
)
TRANSFER_EVIDENCE_LAYER = "transfer_boundary"


@dataclass(frozen=True, slots=True)
class StageIPublicTransferBoundaryConfig:
    """Configuration for one public transfer-boundary summary build."""

    run_id: str
    calibration_summary_path: str = DEFAULT_CALIBRATION_SUMMARY_PATH
    public_mainline_summary_path: str = DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH
    multitask_summary_path: str = DEFAULT_MULTITASK_SUMMARY_PATH
    private_summary_path: str = DEFAULT_PRIVATE_SUMMARY_PATH
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT


@dataclass(frozen=True, slots=True)
class StageIPublicTransferBoundaryRunResult:
    """Artifacts written by one transfer-boundary summary build."""

    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_public_transfer_boundary(
    config: StageIPublicTransferBoundaryConfig,
) -> StageIPublicTransferBoundaryRunResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_public_transfer_boundary",
        logger=LOGGER,
        initial_progress={"artifact_root": str(run_root)},
    ) as progress:
        return _run_stage_i_public_transfer_boundary_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_public_transfer_boundary_observed(
    *,
    config: StageIPublicTransferBoundaryConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIPublicTransferBoundaryRunResult:
    calibration_summary = _load_json(config.calibration_summary_path)
    public_mainline_summary = _load_json(config.public_mainline_summary_path)
    multitask_summary = _load_json(config.multitask_summary_path)
    private_summary = _load_json(config.private_summary_path)
    progress.update("sources_loaded", source_count=4)

    data_boundary_rows = [
        {
            "corpus": "private_stage_h",
            "modality_pair": "real_physiology + real_vehicle_timeseries",
            "labels": "risk_proxy/workload_proxy/event_replay_tag or T1/T2/T3",
            "granularity": "window/view",
            "time_reference": "Stage H unified timeline with sortie/pilot/view ids",
            "evidence_role": "thesis_weak_label + private_proxy",
        },
        {
            "corpus": "uab_workload_dataset",
            "modality_pair": "physiology + task_context_proxy",
            "labels": "subjective workload / public adapter target",
            "granularity": "window",
            "time_reference": "public prepared sequence timeline",
            "evidence_role": "public_adapter/calibration",
        },
        {
            "corpus": "nasa_csm",
            "modality_pair": "physiology + scenario_context_proxy",
            "labels": "attention_state / public adapter target",
            "granularity": "window/sequence",
            "time_reference": "public prepared sequence timeline",
            "evidence_role": "public_adapter/calibration",
        },
    ]
    task_boundary_rows = [
        {
            "layer": "thesis_weak_label",
            "task_scope": "risk_proxy / workload_proxy / event_replay_tag",
            "note": "weak labels built from private Stage H aligned windows",
        },
        {
            "layer": "private_proxy",
            "task_scope": "T1/T2/T3",
            "note": "private proxy benchmark diagnostics only",
        },
        {
            "layer": "public_adapter/calibration",
            "task_scope": "UAB/NASA public tasks",
            "note": "context proxy only, not real vehicle stream",
        },
    ]
    performance_rows = _build_performance_rows(
        calibration_summary=calibration_summary,
        public_mainline_summary=public_mainline_summary,
    )
    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "evidence_layer": TRANSFER_EVIDENCE_LAYER,
        "source_paths": {
            "calibration_summary_path": config.calibration_summary_path,
            "public_mainline_summary_path": config.public_mainline_summary_path,
            "multitask_summary_path": config.multitask_summary_path,
            "private_summary_path": config.private_summary_path,
        },
        "private_mainline": {
            "multitask_run_id": multitask_summary.get("run_id"),
            "private_proxy_run_id": private_summary.get("run_id"),
            "thesis_boundary": (
                private_summary.get("evidence_layers", {})
                .get("thesis_task_evidence", {})
                .get("summary", {})
                .get("thesis_task_boundary")
            ),
        },
        "public_mainline_status": public_mainline_summary.get("public_mainline_status"),
        "data_boundary_rows": data_boundary_rows,
        "task_boundary_rows": task_boundary_rows,
        "performance_rows": performance_rows,
        "boundary_note": (
            "公开 UAB/NASA 结果只用于 adapter/calibration 与 transfer-boundary 说明，"
            "不能改写为论文私有双流本体 fully closed。"
        ),
    }
    summary_path = run_root / "public_transfer_boundary_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"stage-i-public-transfer-boundary-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_public_transfer_boundary_report(summary) + "\n",
        encoding="utf-8",
    )
    progress.finish(summary_path=str(summary_path), report_path=str(report_path))
    return StageIPublicTransferBoundaryRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_stage_i_public_transfer_boundary_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Public Transfer Boundary - {summary['run_id']}",
        "",
        f"- evidence_layer: `{summary['evidence_layer']}`",
        f"- public_mainline_status: `{summary['public_mainline_status']}`",
        f"- boundary_note: `{summary['boundary_note']}`",
        "",
        "## Data Boundary",
        "",
        "| corpus | modality_pair | labels | granularity | time_reference | evidence_role |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("data_boundary_rows", []):
        lines.append(
            "| "
            f"`{row['corpus']}` | "
            f"`{row['modality_pair']}` | "
            f"`{row['labels']}` | "
            f"`{row['granularity']}` | "
            f"`{row['time_reference']}` | "
            f"`{row['evidence_role']}` |"
        )
    lines.extend(
        [
            "",
            "## Task Boundary",
            "",
            "| layer | task_scope | note |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary.get("task_boundary_rows", []):
        lines.append(
            "| "
            f"`{row['layer']}` | "
            f"`{row['task_scope']}` | "
            f"`{row['note']}` |"
        )
    lines.extend(
        [
            "",
            "## Performance References",
            "",
            "| source_type | dataset | subset | metric | value | source_path |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in summary.get("performance_rows", []):
        lines.append(
            "| "
            f"`{row['source_type']}` | "
            f"`{row['dataset_id']}` | "
            f"`{row['subset_id']}` | "
            f"`{row['primary_metric_name']}` | "
            f"{row['primary_metric_value']:.6f} | "
            f"`{row['source_path']}` |"
        )
    return "\n".join(lines)


def _build_performance_rows(
    *,
    calibration_summary: Mapping[str, object],
    public_mainline_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [
        dict(row)
        for row in calibration_summary.get("best_by_category", {}).values()
    ]
    rows.append(
        {
            "source_type": "public_mainline_status",
            "dataset_id": "public_mainline",
            "subset_id": "summary",
            "primary_metric_name": "status",
            "primary_metric_value": 1.0 if public_mainline_summary.get("public_mainline_status") else 0.0,
            "source_path": public_mainline_summary.get("source_paths", {}).get("uab_summary_path"),
        }
    )
    return rows


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))
