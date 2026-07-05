"""task evaluation thesis weak-label multitask sweep."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from chronaris.dataset.dingxin_task_contracts import StageIPrivateTaskEntry
from chronaris.features.experiment_input import E0ExperimentSample
from chronaris.modeling.training.multitask_train import (
    StageIMultitaskTrainConfig,
    run_task_eval_multitask_train,
)
from chronaris.modeling.common.run_observer import (
    StageIRunProgress,
    open_task_eval_run_observer,
)
from chronaris.evidence.weak_label_sweep_helpers import (
    _build_child_run_id,
    _build_combinations,
    _build_summary,
    _load_existing_child_result,
    _merge_source_summary,
    _merge_unique_paths,
    _resolve_blocker_context,
    _resolve_existing_child_summary_path,
    _summarize_child_result,
    _write_partial_state,
    discover_existing_child_summary_paths,
    render_task_eval_multitask_sweep_report,
    resolve_git_commit,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
DEFAULT_PARTIAL_SUMMARY_FILENAME = "partial_summary.json"
DEFAULT_PARTIAL_TABLE_FILENAME = "thesis_weak_label_multitask_ablation.partial.csv"
THESIS_EVIDENCE_LAYER = "thesis_weak_label"


class StageIMultitaskSweepRuntimeBudgetExceeded(RuntimeError):
    """Raised when a sweep exceeds the configured runtime budget."""


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
    resume_existing: bool = False
    resume_run_root: str | None = None
    max_runtime_seconds: float | None = None
    partial_summary_filename: str = DEFAULT_PARTIAL_SUMMARY_FILENAME
    partial_table_filename: str = DEFAULT_PARTIAL_TABLE_FILENAME


@dataclass(frozen=True, slots=True)
class StageIMultitaskSweepRunResult:
    """Artifacts written by one multitask sweep."""

    run_id: str
    artifact_root: str
    summary_path: str
    table_path: str
    report_path: str
    partial_summary_path: str
    partial_table_path: str
    summary: Mapping[str, object]


def run_task_eval_multitask_sweep(
    config: StageIMultitaskSweepConfig,
    *,
    samples: Sequence[E0ExperimentSample],
    task_entries: Sequence[StageIPrivateTaskEntry],
    source_summary: Mapping[str, object] | None = None,
) -> StageIMultitaskSweepRunResult:
    """Run a bounded task evaluation thesis weak-label sweep over the shared backbone."""

    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="task_eval_multitask_sweep",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "max_runs": config.max_runs,
            "physics_constraint_families": list(config.physics_constraint_families),
            "resume_existing": config.resume_existing,
            "resume_run_root": config.resume_run_root,
            "max_runtime_seconds": config.max_runtime_seconds,
        },
    ) as progress:
        return _run_task_eval_multitask_sweep_observed(
            config=config,
            run_root=run_root,
            progress=progress,
            samples=tuple(samples),
            task_entries=tuple(task_entries),
            source_summary=source_summary,
        )


def _run_task_eval_multitask_sweep_observed(
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
    resume_root = Path(config.resume_run_root) if config.resume_run_root else None
    runtime_started_at = perf_counter()

    run_rows: list[dict[str, object]] = []
    child_results: list[dict[str, object]] = []
    aggregate_source_summary = dict(source_summary or {})
    aggregate_source_summary_from_child: dict[str, object] | None = None
    blocker_context = _resolve_blocker_context(config=config, run_root=run_root)
    blocked_at_run_index: int | None = None

    try:
        for index, combination in enumerate(combinations, start=1):
            child_run_id = _build_child_run_id(config=config, index=index, combination=combination)
            existing_summary_path = _resolve_existing_child_summary_path(
                child_run_id=child_run_id,
                combination=combination,
                current_run_root=run_root,
                resume_run_root=resume_root,
            )
            if config.resume_existing and existing_summary_path is not None:
                row, child_result, child_source_summary = _load_existing_child_result(
                    summary_path=existing_summary_path,
                    fallback_combination=combination,
                    git_commit=config.git_commit,
                    source_manifests=config.source_manifests or {},
                )
                run_rows.append(row)
                child_results.append(child_result)
                aggregate_source_summary_from_child = aggregate_source_summary_from_child or child_source_summary
                progress.update(
                    "combination_reused",
                    run_index=index,
                    child_run_id=child_run_id,
                    summary_path=str(existing_summary_path),
                )
                _write_partial_state(
                    config=config,
                    run_root=run_root,
                    run_rows=run_rows,
                    child_results=child_results,
                    source_summary=_merge_source_summary(
                        aggregate_source_summary,
                        aggregate_source_summary_from_child,
                    ),
                    total_combination_count=len(combinations),
                    status="partial_progress",
                    blocker_context=blocker_context,
                )
                continue

            if config.max_runtime_seconds is not None:
                elapsed = perf_counter() - runtime_started_at
                if elapsed >= config.max_runtime_seconds:
                    blocked_at_run_index = index
                    raise StageIMultitaskSweepRuntimeBudgetExceeded(
                        "runtime budget reached before starting the next child run: "
                        f"elapsed={elapsed:.3f}s budget={config.max_runtime_seconds:.3f}s"
                    )

            if not samples:
                raise ValueError(
                    "StageIMultitaskSweep requires samples when the requested child runs "
                    "cannot be fully restored from existing summaries."
                )
            if not task_entries:
                raise ValueError(
                    "StageIMultitaskSweep requires task_entries when the requested child runs "
                    "cannot be fully restored from existing summaries."
                )

            family = str(combination["physics_constraint_family"])
            causal_weight = float(combination["causal_weight"])
            task_loss_weight = float(combination["task_loss_weight"])
            lag_window_points = combination["causal_lag_window_points"]
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
                **_merge_source_summary(aggregate_source_summary, aggregate_source_summary_from_child),
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
            result = run_task_eval_multitask_train(
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
            _write_partial_state(
                config=config,
                run_root=run_root,
                run_rows=run_rows,
                child_results=child_results,
                source_summary=_merge_source_summary(aggregate_source_summary, aggregate_source_summary_from_child),
                total_combination_count=len(combinations),
                status="partial_progress",
                blocker_context=blocker_context,
            )
    except BaseException as error:
        blocked_at_run_index = blocked_at_run_index or min(len(run_rows) + 1, len(combinations) or 1)
        blocker_context = {
            **blocker_context,
            "blocked_at_run_index": blocked_at_run_index,
            "blocker_log_path": str(run_root / "run.log"),
            "blocked_attempt_log_paths": _merge_unique_paths(
                blocker_context.get("blocked_attempt_log_paths", []),
                [str(run_root / "run.log")],
            ),
        }
        _write_partial_state(
            config=config,
            run_root=run_root,
            run_rows=run_rows,
            child_results=child_results,
            source_summary=_merge_source_summary(aggregate_source_summary, aggregate_source_summary_from_child),
            total_combination_count=len(combinations),
            status="partial_blocked",
            blocker_context=blocker_context,
        )
        progress.update(
            "partial_blocked",
            completed_child_runs=len(run_rows),
            blocked_at_run_index=blocked_at_run_index,
            blocker_log_path=str(run_root / "run.log"),
        )
        raise error

    summary = _build_summary(
        config=config,
        run_root=run_root,
        run_rows=run_rows,
        child_results=child_results,
        source_summary=_merge_source_summary(aggregate_source_summary, aggregate_source_summary_from_child),
        sample_count=len(samples) if samples else None,
        task_entry_count=len(task_entries) if task_entries else None,
        status="completed",
        blocker_context=blocker_context,
    )
    table_frame = pd.DataFrame(run_rows)
    table_path = run_root / "thesis_weak_label_multitask_ablation.csv"
    table_frame.to_csv(table_path, index=False)
    summary_path = run_root / "multitask_sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"task-eval-thesis-weak-label-multitask-sweep-{config.run_id}.md"
    report_path.write_text(
        render_task_eval_multitask_sweep_report(summary) + "\n",
        encoding="utf-8",
    )
    _write_partial_state(
        config=config,
        run_root=run_root,
        run_rows=run_rows,
        child_results=child_results,
        source_summary=_merge_source_summary(aggregate_source_summary, aggregate_source_summary_from_child),
        total_combination_count=len(combinations),
        status="completed",
        blocker_context=blocker_context,
    )
    progress.finish(
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
        partial_summary_path=str(run_root / config.partial_summary_filename),
        partial_table_path=str(run_root / config.partial_table_filename),
    )
    return StageIMultitaskSweepRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
        partial_summary_path=str(run_root / config.partial_summary_filename),
        partial_table_path=str(run_root / config.partial_table_filename),
        summary=summary,
    )
