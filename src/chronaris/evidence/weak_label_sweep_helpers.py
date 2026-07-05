"""Reporting, resume, and summary helpers for task evaluation weak-label sweeps."""

from __future__ import annotations

import json
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

THESIS_EVIDENCE_LAYER = "thesis_weak_label"


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


def render_task_eval_multitask_sweep_report(summary: Mapping[str, object]) -> str:
    rows = summary.get("rows", [])
    lines = [
        f"# task evaluation Thesis Weak-Label Multitask Sweep - {summary['run_id']}",
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
