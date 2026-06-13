"""Stage I thesis weak-label multitask sweep."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.dataset.stage_i_private_contracts import StageIPrivateTaskEntry
from chronaris.features.experiment_input import E0ExperimentSample
from chronaris.pipelines.stage_i.stage_i_multitask_train import (
    StageIMultitaskTrainConfig,
    StageIMultitaskTrainRunResult,
    run_stage_i_multitask_train,
)
from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_multitask_sweep"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
THESIS_EVIDENCE_LAYER = "thesis_weak_label"


@dataclass(frozen=True, slots=True)
class StageIMultitaskSweepConfig:
    """Configuration for one bounded thesis weak-label sweep."""

    run_id: str
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    physics_constraint_families: tuple[str, ...] = ("minimal", "full", "rigid_body")
    causal_weights: tuple[float, ...] = (0.0, 0.05, 0.1)
    task_loss_weights: tuple[float, ...] = (0.5, 1.0)
    causal_lag_window_points: tuple[int | None, ...] = (None, 3)
    max_runs: int | None = 4
    epoch_count: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = "cpu"
    task_head_hidden_dim: int = 32
    retrieval_embedding_dim: int = 16
    causal_attention_temperature: float = 1.0
    causal_event_bias_weight: float = 0.25
    git_commit: str | None = None
    source_manifests: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class StageIMultitaskSweepRunResult:
    """Artifacts written by one multitask sweep."""

    run_id: str
    artifact_root: str
    summary_path: str
    table_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_multitask_sweep(
    config: StageIMultitaskSweepConfig,
    *,
    samples: Sequence[E0ExperimentSample],
    task_entries: Sequence[StageIPrivateTaskEntry],
    source_summary: Mapping[str, object] | None = None,
) -> StageIMultitaskSweepRunResult:
    """Run a bounded Stage I thesis weak-label sweep over the shared backbone."""

    if not samples:
        raise ValueError("StageIMultitaskSweep requires at least one sample.")
    if not task_entries:
        raise ValueError("StageIMultitaskSweep requires at least one task entry.")

    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_multitask_sweep",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "max_runs": config.max_runs,
            "physics_constraint_families": list(config.physics_constraint_families),
        },
    ) as progress:
        return _run_stage_i_multitask_sweep_observed(
            config=config,
            run_root=run_root,
            progress=progress,
            samples=tuple(samples),
            task_entries=tuple(task_entries),
            source_summary=source_summary,
        )


def _run_stage_i_multitask_sweep_observed(
    *,
    config: StageIMultitaskSweepConfig,
    run_root: Path,
    progress: StageIRunProgress,
    samples: tuple[E0ExperimentSample, ...],
    task_entries: tuple[StageIPrivateTaskEntry, ...],
    source_summary: Mapping[str, object] | None,
) -> StageIMultitaskSweepRunResult:
    combinations = _build_combinations(config)
    progress.update("grid_ready", combination_count=len(combinations))
    child_root = run_root / "runs"
    child_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    child_results: list[dict[str, object]] = []
    for index, combination in enumerate(combinations, start=1):
        family = str(combination["physics_constraint_family"])
        causal_weight = float(combination["causal_weight"])
        task_loss_weight = float(combination["task_loss_weight"])
        lag_window_points = combination["causal_lag_window_points"]
        child_run_id = (
            f"{config.run_id}-{index:02d}-{family}"
            f"-cw{_slug_float(causal_weight)}"
            f"-tlw{_slug_float(task_loss_weight)}"
            f"-lag{_slug_optional_int(lag_window_points)}"
        )
        train_config = StageIMultitaskTrainConfig(
            run_id=child_run_id,
            output_root=str(child_root),
            task_head_hidden_dim=config.task_head_hidden_dim,
            retrieval_embedding_dim=config.retrieval_embedding_dim,
            task_loss_weight=task_loss_weight,
            causal_weight=causal_weight,
            causal_attention_temperature=config.causal_attention_temperature,
            causal_event_bias_weight=config.causal_event_bias_weight,
            causal_lag_window_points=lag_window_points,
        )
        preview_config = replace(
            train_config.preview_config,
            epoch_count=config.epoch_count,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            physics_constraint_family=family,
        )
        train_config = replace(train_config, preview_config=preview_config)
        child_source_summary = {
            **dict(source_summary or {}),
            "evidence_layer": THESIS_EVIDENCE_LAYER,
            "source_manifests": dict(config.source_manifests or {}),
            "sweep_combination": {
                "physics_constraint_family": family,
                "causal_weight": causal_weight,
                "task_loss_weight": task_loss_weight,
                "causal_lag_window_points": lag_window_points,
            },
            "git_commit": config.git_commit,
        }
        progress.update(
            "combination_started",
            run_index=index,
            child_run_id=child_run_id,
            physics_constraint_family=family,
        )
        result = run_stage_i_multitask_train(
            train_config,
            samples=samples,
            task_entries=task_entries,
            source_summary=child_source_summary,
        )
        row = _summarize_child_result(
            result=result,
            family=family,
            causal_weight=causal_weight,
            task_loss_weight=task_loss_weight,
            lag_window_points=lag_window_points,
            git_commit=config.git_commit,
            source_manifests=config.source_manifests or {},
        )
        run_rows.append(row)
        child_results.append(
            {
                "run_id": child_run_id,
                "summary_path": result.summary_path,
                "checkpoint_path": result.checkpoint_path,
                "task_manifest_path": result.task_manifest_path,
            }
        )
        progress.update(
            "combination_finished",
            run_index=index,
            child_run_id=child_run_id,
            test_total=row["test_total"],
        )

    table_frame = pd.DataFrame(run_rows)
    table_path = run_root / "thesis_weak_label_multitask_ablation.csv"
    table_frame.to_csv(table_path, index=False)
    best_row = _select_best_row(run_rows)
    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "evidence_layer": THESIS_EVIDENCE_LAYER,
        "git_commit": config.git_commit,
        "source_manifests": dict(config.source_manifests or {}),
        "sample_count": len(samples),
        "task_entry_count": len(task_entries),
        "combination_count": len(run_rows),
        "source_summary": dict(source_summary or {}),
        "grid": {
            "physics_constraint_families": list(config.physics_constraint_families),
            "causal_weights": list(config.causal_weights),
            "task_loss_weights": list(config.task_loss_weights),
            "causal_lag_window_points": [
                None if value is None else int(value)
                for value in config.causal_lag_window_points
            ],
            "max_runs": config.max_runs,
        },
        "metric_definition": {
            "test_total": "shared Stage E objective total on the held-out split",
            "test_task_total": "weighted thesis weak-label task loss on the held-out split",
            "test_causal_total": "causal regularization term on the held-out split",
        },
        "rows": run_rows,
        "child_runs": child_results,
        "best_run": best_row,
    }
    summary_path = run_root / "multitask_sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"stage-i-thesis-weak-label-multitask-sweep-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_multitask_sweep_report(summary) + "\n",
        encoding="utf-8",
    )
    progress.finish(
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
    )
    return StageIMultitaskSweepRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_stage_i_multitask_sweep_report(summary: Mapping[str, object]) -> str:
    rows = summary.get("rows", [])
    lines = [
        f"# Stage I Thesis Weak-Label Multitask Sweep - {summary['run_id']}",
        "",
        f"- evidence_layer: `{summary['evidence_layer']}`",
        f"- source_manifests: `{summary.get('source_manifests', {})}`",
        f"- combination_count: `{summary.get('combination_count', 0)}`",
        "",
        "## Reading",
        "",
        "1. 所有结果都属于 `thesis weak-label evidence`，任务仍是 `risk_proxy / workload_proxy / event_replay_tag`，不是人工真值闭环。",
        "2. 当前表按 `test_total` 升序排序，便于快速定位在共享骨干 + 任务监督 + 因果正则组合下的相对稳定配置。",
        "",
        "## Ablation Table",
        "",
        "| run_id | physics_family | causal_weight | task_loss_weight | lag_window | test_total | test_task_total | test_causal_total | checkpoint |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| "
            f"`{row['child_run_id']}` | "
            f"`{row['physics_constraint_family']}` | "
            f"{row['causal_weight']:.2f} | "
            f"{row['task_loss_weight']:.2f} | "
            f"`{row['causal_lag_window_points']}` | "
            f"{row['test_total']:.6f} | "
            f"{row['test_task_total']:.6f} | "
            f"{row['test_causal_total']:.6f} | "
            f"`{row['checkpoint_path']}` |"
        )
    best_run = summary.get("best_run") or {}
    if best_run:
        lines.extend(
            [
                "",
                "## Best Run",
                "",
                f"- child_run_id: `{best_run.get('child_run_id')}`",
                f"- physics_constraint_family: `{best_run.get('physics_constraint_family')}`",
                f"- test_total: `{best_run.get('test_total')}`",
                f"- checkpoint_path: `{best_run.get('checkpoint_path')}`",
            ]
        )
    return "\n".join(lines)


def _build_combinations(config: StageIMultitaskSweepConfig) -> tuple[dict[str, object], ...]:
    rows = [
        {
            "physics_constraint_family": family,
            "causal_weight": causal_weight,
            "task_loss_weight": task_loss_weight,
            "causal_lag_window_points": lag_window_points,
        }
        for family, causal_weight, task_loss_weight, lag_window_points in product(
            config.physics_constraint_families,
            config.causal_weights,
            config.task_loss_weights,
            config.causal_lag_window_points,
        )
    ]
    if config.max_runs is not None:
        rows = rows[: config.max_runs]
    return tuple(rows)


def _summarize_child_result(
    *,
    result: StageIMultitaskTrainRunResult,
    family: str,
    causal_weight: float,
    task_loss_weight: float,
    lag_window_points: int | None,
    git_commit: str | None,
    source_manifests: Mapping[str, str],
) -> dict[str, object]:
    test_metrics = result.summary.get("test_metrics", {})
    validation_metrics = result.summary.get("validation_metrics", {})
    return {
        "child_run_id": result.summary["run_id"],
        "artifact_root": result.artifact_root,
        "summary_path": result.summary_path,
        "checkpoint_path": result.checkpoint_path,
        "task_manifest_path": result.task_manifest_path,
        "evidence_layer": THESIS_EVIDENCE_LAYER,
        "git_commit": git_commit,
        "source_manifests": dict(source_manifests),
        "physics_constraint_family": family,
        "causal_weight": causal_weight,
        "task_loss_weight": task_loss_weight,
        "causal_lag_window_points": lag_window_points,
        "test_total": float(test_metrics.get("total", 0.0)),
        "test_task_total": float(test_metrics.get("task_total", 0.0)),
        "test_causal_total": float(test_metrics.get("causal_total", 0.0)),
        "validation_total": float(validation_metrics.get("total", 0.0))
        if validation_metrics
        else 0.0,
    }


def _select_best_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["test_total"]))


def _slug_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _slug_optional_int(value: int | None) -> str:
    return "none" if value is None else str(int(value))


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
