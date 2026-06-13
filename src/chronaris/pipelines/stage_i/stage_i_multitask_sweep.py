"""Stage I thesis weak-label multitask sweep."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

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


def run_stage_i_multitask_sweep(
    config: StageIMultitaskSweepConfig,
    *,
    samples: Sequence[E0ExperimentSample],
    task_entries: Sequence[StageIPrivateTaskEntry],
    source_summary: Mapping[str, object] | None = None,
) -> StageIMultitaskSweepRunResult:
    """Run a bounded Stage I thesis weak-label sweep over the shared backbone."""

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
            "resume_existing": config.resume_existing,
            "resume_run_root": config.resume_run_root,
            "max_runtime_seconds": config.max_runtime_seconds,
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


def discover_existing_child_summary_paths(
    config: StageIMultitaskSweepConfig,
) -> dict[str, str]:
    """Return already materialized child summaries reachable from the configured run roots."""

    combinations = _build_combinations(config)
    current_root = Path(config.output_root) / config.run_id
    resume_root = Path(config.resume_run_root) if config.resume_run_root else None
    resolved: dict[str, str] = {}
    for index, combination in enumerate(combinations, start=1):
        child_run_id = _build_child_run_id(config=config, index=index, combination=combination)
        summary_path = _resolve_existing_child_summary_path(
            child_run_id=child_run_id,
            combination=combination,
            current_run_root=current_root,
            resume_run_root=resume_root,
        )
        if summary_path is not None:
            resolved[child_run_id] = str(summary_path)
    return resolved


def render_stage_i_multitask_sweep_report(summary: Mapping[str, object]) -> str:
    rows = summary.get("rows", [])
    lines = [
        f"# Stage I Thesis Weak-Label Multitask Sweep - {summary['run_id']}",
        "",
        f"- status: `{summary.get('status', 'completed')}`",
        f"- evidence_layer: `{summary['evidence_layer']}`",
        f"- source_manifests: `{summary.get('source_manifests', {})}`",
        f"- combination_count: `{summary.get('combination_count', 0)}`",
    ]
    if summary.get("derived_from_run_id"):
        lines.append(f"- derived_from_run_id: `{summary['derived_from_run_id']}`")
    if summary.get("blocked_at_run_index") is not None:
        lines.append(f"- blocked_at_run_index: `{summary['blocked_at_run_index']}`")
    if summary.get("blocked_attempt_log_paths"):
        lines.append(f"- blocked_attempt_log_paths: `{summary['blocked_attempt_log_paths']}`")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "1. 所有结果都属于 `thesis weak-label evidence`，任务仍是 `risk_proxy / workload_proxy / event_replay_tag`，不是人工真值闭环。",
            "2. 当前表按 `test_total` 升序排序，便于快速定位在共享骨干 + 任务监督 + 因果正则组合下的相对稳定配置。",
        ]
    )
    if summary.get("status") == "partial_blocked":
        lines.extend(
            [
                "3. 本次 sweep 只完成了部分 child run；`partial_summary.json` 和临时 CSV 已记录当前可引用证据，不应包装成 completed。",
            ]
        )
    lines.extend(
        [
            "",
            "## Ablation Table",
            "",
            "| run_id | physics_family | causal_weight | task_loss_weight | lag_window | test_total | test_task_total | test_causal_total | checkpoint |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
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
    if summary.get("comparison"):
        comparison = summary["comparison"]
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                f"- proxy_sample_source: `{comparison.get('proxy_sample_source')}`",
                f"- live_sample_source: `{comparison.get('live_sample_source')}`",
                f"- partial_run_blocker: `{comparison.get('partial_run_blocker')}`",
            ]
        )
    return "\n".join(lines)


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
    report_path = report_root / f"stage-i-thesis-weak-label-multitask-sweep-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_multitask_sweep_report(summary) + "\n",
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


def _build_child_run_id(
    *,
    config: StageIMultitaskSweepConfig,
    index: int,
    combination: Mapping[str, object],
) -> str:
    family = str(combination["physics_constraint_family"])
    causal_weight = float(combination["causal_weight"])
    task_loss_weight = float(combination["task_loss_weight"])
    lag_window_points = combination["causal_lag_window_points"]
    return (
        f"{config.run_id}-{index:02d}-{family}"
        f"-cw{_slug_float(causal_weight)}"
        f"-tlw{_slug_float(task_loss_weight)}"
        f"-lag{_slug_optional_int(lag_window_points)}"
    )


def _resolve_existing_child_summary_path(
    *,
    child_run_id: str,
    combination: Mapping[str, object],
    current_run_root: Path,
    resume_run_root: Path | None,
) -> Path | None:
    candidate = current_run_root / "runs" / child_run_id / "multitask_summary.json"
    if candidate.exists():
        return candidate
    if resume_run_root is None:
        return None
    direct_resume_candidate = resume_run_root / "runs" / child_run_id / "multitask_summary.json"
    if direct_resume_candidate.exists():
        return direct_resume_candidate
    matching_resume_candidate = _find_matching_resume_summary(
        resume_run_root=resume_run_root,
        combination=combination,
    )
    if matching_resume_candidate is not None:
        return matching_resume_candidate
    return None


def _find_matching_resume_summary(
    *,
    resume_run_root: Path,
    combination: Mapping[str, object],
) -> Path | None:
    runs_root = resume_run_root / "runs"
    if not runs_root.exists():
        return None
    normalized_target = _normalize_sweep_combination(combination)
    for summary_path in sorted(runs_root.glob("*/multitask_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate = payload.get("source_summary", {}).get("sweep_combination")
        if _normalize_sweep_combination(candidate) == normalized_target:
            return summary_path
    return None


def _normalize_sweep_combination(payload: object) -> tuple[object, object, object, object]:
    if not isinstance(payload, Mapping):
        return (None, None, None, None)
    lag_window_points = payload.get("causal_lag_window_points")
    if lag_window_points is not None:
        lag_window_points = int(lag_window_points)
    return (
        str(payload.get("physics_constraint_family")),
        float(payload.get("causal_weight", 0.0)),
        float(payload.get("task_loss_weight", 0.0)),
        lag_window_points,
    )


def _load_existing_child_result(
    *,
    summary_path: Path,
    fallback_combination: Mapping[str, object],
    git_commit: str | None,
    source_manifests: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    child_source_summary = dict(payload.get("source_summary", {}))
    sweep_combination = dict(child_source_summary.get("sweep_combination") or fallback_combination)
    row = {
        "child_run_id": payload.get("run_id", summary_path.parent.name),
        "artifact_root": payload.get("artifact_root", str(summary_path.parent)),
        "summary_path": str(summary_path),
        "checkpoint_path": payload.get("checkpoint_path", str(summary_path.parent / "multitask_checkpoint.pt")),
        "task_manifest_path": payload.get("task_manifest_path", str(summary_path.parent / "thesis_task_manifest.jsonl")),
        "evidence_layer": THESIS_EVIDENCE_LAYER,
        "git_commit": child_source_summary.get("git_commit", git_commit),
        "source_manifests": dict(child_source_summary.get("source_manifests", source_manifests)),
        "physics_constraint_family": str(sweep_combination.get("physics_constraint_family")),
        "causal_weight": float(sweep_combination.get("causal_weight", 0.0)),
        "task_loss_weight": float(sweep_combination.get("task_loss_weight", 0.0)),
        "causal_lag_window_points": sweep_combination.get("causal_lag_window_points"),
        "test_total": float(payload.get("test_metrics", {}).get("total", 0.0)),
        "test_task_total": float(payload.get("test_metrics", {}).get("task_total", 0.0)),
        "test_causal_total": float(payload.get("test_metrics", {}).get("causal_total", 0.0)),
        "validation_total": float(payload.get("validation_metrics", {}).get("total", 0.0)),
    }
    child_result = {
        "run_id": row["child_run_id"],
        "summary_path": str(summary_path),
        "checkpoint_path": row["checkpoint_path"],
        "task_manifest_path": row["task_manifest_path"],
    }
    child_source_summary.pop("sweep_combination", None)
    return row, child_result, child_source_summary


def _write_partial_state(
    *,
    config: StageIMultitaskSweepConfig,
    run_root: Path,
    run_rows: Sequence[Mapping[str, object]],
    child_results: Sequence[Mapping[str, object]],
    source_summary: Mapping[str, object],
    total_combination_count: int,
    status: str,
    blocker_context: Mapping[str, object],
) -> None:
    partial_frame = pd.DataFrame(run_rows)
    partial_table_path = run_root / config.partial_table_filename
    partial_frame.to_csv(partial_table_path, index=False)
    partial_summary = _build_summary(
        config=config,
        run_root=run_root,
        run_rows=run_rows,
        child_results=child_results,
        source_summary=source_summary,
        sample_count=None,
        task_entry_count=None,
        status=status,
        blocker_context=blocker_context,
    )
    partial_summary["target_combination_count"] = total_combination_count
    partial_summary["combination_count_completed"] = len(run_rows)
    partial_summary["partial_table_path"] = str(partial_table_path)
    partial_summary_path = run_root / config.partial_summary_filename
    partial_summary_path.write_text(
        json.dumps(partial_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_summary(
    *,
    config: StageIMultitaskSweepConfig,
    run_root: Path,
    run_rows: Sequence[Mapping[str, object]],
    child_results: Sequence[Mapping[str, object]],
    source_summary: Mapping[str, object],
    sample_count: int | None,
    task_entry_count: int | None,
    status: str,
    blocker_context: Mapping[str, object],
) -> dict[str, object]:
    best_row = _select_best_row(run_rows)
    resolved_sample_count = sample_count
    resolved_task_entry_count = task_entry_count
    if resolved_sample_count is None:
        resolved_sample_count = _extract_sample_count(source_summary, child_results)
    if resolved_task_entry_count is None:
        resolved_task_entry_count = _extract_task_entry_count(source_summary, child_results)
    completed_child_run_paths = [str(row["summary_path"]) for row in child_results]
    blocked_attempt_log_paths = blocker_context.get("blocked_attempt_log_paths", [])
    summary = {
        "run_id": config.run_id,
        "status": status,
        "artifact_root": str(run_root),
        "evidence_layer": THESIS_EVIDENCE_LAYER,
        "git_commit": config.git_commit,
        "source_manifests": dict(config.source_manifests or {}),
        "sample_count": resolved_sample_count,
        "task_entry_count": resolved_task_entry_count,
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
        "rows": list(run_rows),
        "child_runs": list(child_results),
        "best_run": best_row,
        "completed_child_runs": [str(row["run_id"]) for row in child_results],
        "completed_child_run_paths": completed_child_run_paths,
        "derived_from_run_id": blocker_context.get("derived_from_run_id"),
        "blocked_at_run_index": blocker_context.get("blocked_at_run_index"),
        "blocker_log_path": blocker_context.get("blocker_log_path"),
        "blocked_attempt_log_paths": blocked_attempt_log_paths,
    }
    if blocker_context.get("resume_progress_path"):
        summary["resume_progress_path"] = blocker_context["resume_progress_path"]
    return summary


def _resolve_blocker_context(
    *,
    config: StageIMultitaskSweepConfig,
    run_root: Path,
) -> dict[str, object]:
    roots = [run_root]
    if config.resume_run_root:
        resume_root = Path(config.resume_run_root)
        if resume_root not in roots:
            roots.append(resume_root)
    derived_from_run_id = None
    blocked_at_run_index = None
    blocker_log_path = None
    resume_progress_path = None
    blocked_attempt_log_paths: list[str] = []
    for root in roots:
        partial_path = root / config.partial_summary_filename
        if partial_path.exists():
            payload = json.loads(partial_path.read_text(encoding="utf-8"))
            derived_from_run_id = derived_from_run_id or str(payload.get("run_id") or root.name)
            blocked_at_run_index = blocked_at_run_index or payload.get("blocked_at_run_index")
            blocker_log_path = blocker_log_path or payload.get("blocker_log_path")
            resume_progress_path = resume_progress_path or str(partial_path)
            blocked_attempt_log_paths = _merge_unique_paths(
                blocked_attempt_log_paths,
                payload.get("blocked_attempt_log_paths", []),
            )
            continue
        progress_path = root / "progress.json"
        run_log_path = root / "run.log"
        if progress_path.exists():
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            if payload.get("last_event") in {"failed", "partial_blocked"}:
                derived_from_run_id = derived_from_run_id or root.name
                blocked_at_run_index = blocked_at_run_index or payload.get("blocked_at_run_index") or payload.get("run_index")
                if run_log_path.exists():
                    blocker_log_path = blocker_log_path or str(run_log_path)
                    blocked_attempt_log_paths = _merge_unique_paths(
                        blocked_attempt_log_paths,
                        [str(run_log_path)],
                    )
                resume_progress_path = resume_progress_path or str(progress_path)
    return {
        "derived_from_run_id": derived_from_run_id,
        "blocked_at_run_index": blocked_at_run_index,
        "blocker_log_path": blocker_log_path,
        "blocked_attempt_log_paths": blocked_attempt_log_paths,
        "resume_progress_path": resume_progress_path,
    }


def _extract_sample_count(
    source_summary: Mapping[str, object],
    child_results: Sequence[Mapping[str, object]],
) -> int | None:
    sample_collection = dict(source_summary.get("sample_collection") or {})
    sample_count = sample_collection.get("sample_count")
    if sample_count is not None:
        return int(sample_count)
    for child_result in child_results:
        payload = json.loads(Path(str(child_result["summary_path"])).read_text(encoding="utf-8"))
        value = payload.get("sample_count")
        if value is not None:
            return int(value)
    return None


def _extract_task_entry_count(
    source_summary: Mapping[str, object],
    child_results: Sequence[Mapping[str, object]],
) -> int | None:
    task_summary = dict(source_summary.get("task_payload_summary") or {})
    entry_count = task_summary.get("entry_count")
    if entry_count is not None:
        return int(entry_count)
    for child_result in child_results:
        payload = json.loads(Path(str(child_result["summary_path"])).read_text(encoding="utf-8"))
        task_payload_summary = dict(payload.get("source_summary", {}).get("task_payload_summary") or {})
        value = task_payload_summary.get("entry_count")
        if value is not None:
            return int(value)
    return None


def _merge_source_summary(
    base_source_summary: Mapping[str, object] | None,
    child_source_summary: Mapping[str, object] | None,
) -> dict[str, object]:
    resolved = dict(child_source_summary or {})
    for key, value in dict(base_source_summary or {}).items():
        if key not in resolved:
            resolved[key] = value
            continue
        if isinstance(resolved[key], Mapping) and isinstance(value, Mapping):
            merged_nested = dict(resolved[key])
            for nested_key, nested_value in dict(value).items():
                merged_nested.setdefault(nested_key, nested_value)
            resolved[key] = merged_nested
    resolved.pop("sweep_combination", None)
    return resolved


def _merge_unique_paths(existing: Sequence[str], additional: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for value in list(existing) + list(additional):
        if value and value not in merged:
            merged.append(str(value))
    return merged


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
