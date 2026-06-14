"""Public adapter and calibration summary builder for Stage I."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from chronaris.pipelines.stage_i.common.run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_public_adapter_calibration"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_UAB_SKLEARN_SUMMARY_PATHS = (
    "docs/artifacts/assets/stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab/public_opt_summary.json",
    "docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/public_opt_summary.json",
)
DEFAULT_NASA_SUMMARY_PATHS = (
    "docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/public_opt_summary.json",
)
DEFAULT_UAB_TORCH_SUMMARY_PATHS = (
    "docs/artifacts/assets/stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch/public_opt_torch_summary.json",
    "docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/public_opt_torch_summary.json",
)
PUBLIC_ADAPTER_EVIDENCE_LAYER = "public_adapter_calibration"


@dataclass(frozen=True, slots=True)
class StageIPublicAdapterCalibrationConfig:
    """Configuration for one public adapter calibration summary build."""

    run_id: str
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    uab_sklearn_summary_paths: tuple[str, ...] = DEFAULT_UAB_SKLEARN_SUMMARY_PATHS
    nasa_summary_paths: tuple[str, ...] = DEFAULT_NASA_SUMMARY_PATHS
    uab_torch_summary_paths: tuple[str, ...] = DEFAULT_UAB_TORCH_SUMMARY_PATHS
    git_commit: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicAdapterCalibrationRunResult:
    """Artifacts written by one public adapter calibration summary build."""

    run_id: str
    artifact_root: str
    summary_path: str
    table_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_public_adapter_calibration(
    config: StageIPublicAdapterCalibrationConfig,
) -> StageIPublicAdapterCalibrationRunResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_public_adapter_calibration",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "uab_sklearn_source_count": len(config.uab_sklearn_summary_paths),
            "nasa_source_count": len(config.nasa_summary_paths),
            "uab_torch_source_count": len(config.uab_torch_summary_paths),
        },
    ) as progress:
        return _run_stage_i_public_adapter_calibration_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_public_adapter_calibration_observed(
    *,
    config: StageIPublicAdapterCalibrationConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIPublicAdapterCalibrationRunResult:
    rows: list[dict[str, object]] = []
    source_index: dict[str, dict[str, object]] = {}

    for path_like in config.uab_sklearn_summary_paths:
        payload = _load_json(path_like)
        source_index[path_like] = {
            "dataset_id": payload.get("dataset_id"),
            "backend": "sklearn",
        }
        rows.extend(_extract_uab_sklearn_rows(payload=payload, source_path=path_like))
    progress.update("uab_sklearn_loaded", row_count=len(rows))

    for path_like in config.nasa_summary_paths:
        payload = _load_json(path_like)
        source_index[path_like] = {
            "dataset_id": payload.get("dataset_id"),
            "backend": "sklearn",
        }
        rows.extend(_extract_nasa_rows(payload=payload, source_path=path_like))
    progress.update("nasa_loaded", row_count=len(rows))

    for path_like in config.uab_torch_summary_paths:
        payload = _load_json(path_like)
        source_index[path_like] = {
            "dataset_id": payload.get("dataset_id"),
            "backend": "torch",
        }
        rows.extend(_extract_uab_torch_rows(payload=payload, source_path=path_like))
    progress.update("uab_torch_loaded", row_count=len(rows))

    table_frame = pd.DataFrame(rows)
    table_path = run_root / "public_adapter_calibration_table.csv"
    table_frame.to_csv(table_path, index=False)
    best_by_category = _best_by_category(rows)
    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "evidence_layer": PUBLIC_ADAPTER_EVIDENCE_LAYER,
        "git_commit": config.git_commit,
        "source_index": source_index,
        "rows": rows,
        "best_by_category": best_by_category,
        "boundary_note": (
            "UAB/NASA rows only support public adapter or calibration evidence. "
            "They do not prove the thesis dual-stream private mainline is fully closed."
        ),
    }
    summary_path = run_root / "public_adapter_calibration_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"stage-i-public-adapter-calibration-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_public_adapter_calibration_report(summary) + "\n",
        encoding="utf-8",
    )
    progress.finish(
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
    )
    return StageIPublicAdapterCalibrationRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_stage_i_public_adapter_calibration_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Public Adapter Calibration - {summary['run_id']}",
        "",
        f"- evidence_layer: `{summary['evidence_layer']}`",
        f"- boundary_note: `{summary['boundary_note']}`",
        "",
        "## Best By Category",
        "",
        "| category | dataset | subset | metric | value | source_path |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for category, row in summary.get("best_by_category", {}).items():
        lines.append(
            "| "
            f"`{category}` | "
            f"`{row['dataset_id']}` | "
            f"`{row['subset_id']}` | "
            f"`{row['primary_metric_name']}` | "
            f"{row['primary_metric_value']:.6f} | "
            f"`{row['source_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| category | dataset | subset | candidate | metric | value |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in summary.get("rows", []):
        lines.append(
            "| "
            f"`{row['source_type']}` | "
            f"`{row['dataset_id']}` | "
            f"`{row['subset_id']}` | "
            f"`{row['candidate_name']}` | "
            f"`{row['primary_metric_name']}` | "
            f"{row['primary_metric_value']:.6f} |"
        )
    return "\n".join(lines)


def _extract_uab_sklearn_rows(*, payload: Mapping[str, object], source_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset_id, subset_payload in (payload.get("subset_results") or {}).items():
        heads = subset_payload.get("heads", {})
        for head_name, metrics in heads.items():
            source_type = (
                "calibration_baseline"
                if _is_calibration_head(head_name)
                else "public_adapter_baseline"
            )
            rows.append(
                {
                    "source_type": source_type,
                    "backend": "sklearn",
                    "dataset_id": str(payload.get("dataset_id")),
                    "run_id": str(payload.get("run_id")),
                    "subset_id": str(subset_id),
                    "candidate_name": str(head_name),
                    "primary_metric_name": "rmse",
                    "primary_metric_value": float(metrics.get("rmse", 0.0)),
                    "secondary_metric_name": "mae",
                    "secondary_metric_value": float(metrics.get("mae", 0.0)),
                    "selected_subset": list(payload.get("selected_subsets", [])),
                    "source_path": source_path,
                }
            )
    return rows


def _extract_nasa_rows(*, payload: Mapping[str, object], source_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset_id, subset_payload in (payload.get("subset_results") or {}).items():
        heads = subset_payload.get("heads", {})
        for head_name, metrics in heads.items():
            rows.append(
                {
                    "source_type": "legacy_public_opt",
                    "backend": "sklearn",
                    "dataset_id": str(payload.get("dataset_id")),
                    "run_id": str(payload.get("run_id")),
                    "subset_id": str(subset_id),
                    "candidate_name": str(head_name),
                    "primary_metric_name": "macro_f1",
                    "primary_metric_value": float(metrics.get("macro_f1", 0.0)),
                    "secondary_metric_name": "balanced_accuracy",
                    "secondary_metric_value": float(metrics.get("balanced_accuracy", 0.0)),
                    "selected_subset": list(payload.get("selected_subsets", [])),
                    "source_path": source_path,
                }
            )
    return rows


def _extract_uab_torch_rows(*, payload: Mapping[str, object], source_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset_id, metrics in (payload.get("final_result", {}) or {}).get("groups", {}).items():
        selection = (
            (payload.get("final_result", {}) or {})
            .get("group_selections", {})
            .get(subset_id, {})
        )
        rows.append(
            {
                "source_type": "torch_uab",
                "backend": "torch",
                "dataset_id": str(payload.get("dataset_id")),
                "run_id": str(payload.get("run_id")),
                "subset_id": str(subset_id),
                "candidate_name": str(selection.get("selected_source_id") or payload.get("winning_candidate", {}).get("candidate_id")),
                "primary_metric_name": "rmse",
                "primary_metric_value": float(metrics.get("rmse", 0.0)),
                "secondary_metric_name": "mae",
                "secondary_metric_value": float(metrics.get("mae", 0.0)),
                "selected_subset": list(
                    ((payload.get("screen_config") or {}).get("selected_subsets")) or []
                ),
                "source_path": source_path,
            }
        )
    return rows


def _best_by_category(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_type"]), []).append(row)
    for source_type, items in grouped.items():
        if source_type == "legacy_public_opt":
            winner = max(items, key=lambda item: (float(item["primary_metric_value"]), float(item["secondary_metric_value"])))
        else:
            winner = min(items, key=lambda item: (float(item["primary_metric_value"]), float(item["secondary_metric_value"])))
        best[source_type] = dict(winner)
    return best


def _is_calibration_head(head_name: str) -> bool:
    lowered = head_name.lower()
    return any(
        token in lowered
        for token in ("target_prior", "heat_prior", "trimmed_mean", "prior")
    )


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def resolve_git_commit(*, cwd: str | Path = ".") -> str | None:
    try:
        payload = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return payload.stdout.strip() or None
