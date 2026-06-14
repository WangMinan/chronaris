"""Component ablation for the private proxy Chronaris candidate."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.pipelines.stage_i.private.benchmark_data import (
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    build_variant_feature_frames,
    derive_private_proxy_task_entries,
    load_aligned_private_records,
    merge_task_features,
)
from chronaris.pipelines.stage_i.private.benchmark_models import (
    run_classical_variant,
    run_retrieval_task,
)
from chronaris.pipelines.stage_i.private.optimization import (
    DEFAULT_OPTIMIZED_VARIANT_NAME,
    build_optimized_chronaris_feature_frames,
    optimized_no_mask_variant_name,
    run_optimized_retrieval_variant,
    run_optimized_supervised_variant,
)
from chronaris.pipelines.stage_i.common.run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_private_component_ablation"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
PRIVATE_PROXY_EVIDENCE_LAYER = "private_proxy"


@dataclass(frozen=True, slots=True)
class StageIPrivateComponentAblationConfig:
    """Configuration for one bounded Chronaris component ablation run."""

    run_id: str
    e_run_manifest_path: str
    f_run_manifest_path: str
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    target_variant_name: str = DEFAULT_OPTIMIZED_VARIANT_NAME
    lag_window_points: int = 3
    full_residual_mode: str = "raw_window_stats"
    no_time_residual_mode: str = "none"
    git_commit: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPrivateComponentAblationRunResult:
    """Artifacts written by one private proxy component ablation."""

    run_id: str
    artifact_root: str
    summary_path: str
    table_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_private_component_ablation(
    config: StageIPrivateComponentAblationConfig,
) -> StageIPrivateComponentAblationRunResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_private_component_ablation",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "target_variant_name": config.target_variant_name,
        },
    ) as progress:
        return _run_stage_i_private_component_ablation_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_private_component_ablation_observed(
    *,
    config: StageIPrivateComponentAblationConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIPrivateComponentAblationRunResult:
    records = load_aligned_private_records(
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
    task_payload = derive_private_proxy_task_entries(records)
    progress.update(
        "records_loaded",
        sample_count=int(len(records)),
        view_count=int(records["view_id"].nunique()),
    )

    base_frames, base_diagnostics = build_variant_feature_frames(
        records,
        enable_optimized_chronaris=False,
    )
    full_opt_frames, full_opt_diagnostics = build_variant_feature_frames(
        records,
        enable_optimized_chronaris=True,
        target_variant_name=config.target_variant_name,
        lag_window_points=config.lag_window_points,
        residual_mode=config.full_residual_mode,
    )
    no_residual_frames, no_residual_diagnostics = build_optimized_chronaris_feature_frames(
        records,
        target_variant_name=f"{config.target_variant_name}_no_time_residual_opt",
        lag_window_points=config.lag_window_points,
        residual_mode=config.no_time_residual_mode,
        selected_vehicle_fields=task_payload["summary"]["selected_vehicle_fields"],
        selected_physiology_fields=task_payload["summary"]["selected_physiology_fields"],
    )
    variant_frames = dict(base_frames)
    variant_frames["chronaris_opt"] = full_opt_frames[config.target_variant_name]
    variant_frames["chronaris_opt_no_causal_mask"] = full_opt_frames[
        optimized_no_mask_variant_name(config.target_variant_name)
    ]
    variant_frames["chronaris_opt_no_time_residual"] = no_residual_frames[
        f"{config.target_variant_name}_no_time_residual_opt"
    ]
    variant_frames["chronaris_opt_no_task_head"] = full_opt_frames[config.target_variant_name]
    variant_order = (
        "naive_sync",
        "e_baseline",
        "f_full",
        "g_min",
        "g_no_causal_mask",
        "chronaris_opt",
        "chronaris_opt_no_causal_mask",
        "chronaris_opt_no_time_residual",
        "chronaris_opt_no_task_head",
    )
    variant_specs = {
        "naive_sync": {"component": "module_baseline", "optimized_head": False},
        "e_baseline": {"component": "module_baseline", "optimized_head": False},
        "f_full": {"component": "module_baseline", "optimized_head": False},
        "g_min": {"component": "module_baseline", "optimized_head": False},
        "g_no_causal_mask": {"component": "module_baseline", "optimized_head": False},
        "chronaris_opt": {"component": "full_candidate", "optimized_head": True},
        "chronaris_opt_no_causal_mask": {"component": "remove_causal_mask", "optimized_head": True},
        "chronaris_opt_no_time_residual": {"component": "remove_time_residual", "optimized_head": True},
        "chronaris_opt_no_task_head": {"component": "remove_task_aware_head", "optimized_head": False},
    }
    progress.update("variant_frames_ready", variant_count=len(variant_order))

    task_results: dict[str, object] = {}
    table_rows: list[dict[str, object]] = []
    for task_name, task_entries in task_payload["by_task"].items():
        task_type = str(task_entries[0].task_type)
        variant_results: dict[str, object] = {}
        for variant_name in variant_order:
            frame = variant_frames[variant_name]
            merged = merge_task_features(task_entries, frame, task_type=task_type)
            if task_type == "retrieval":
                if variant_specs[variant_name]["optimized_head"]:
                    variant_result = run_optimized_retrieval_variant(
                        merged,
                        variant_name=variant_name,
                    )
                else:
                    task_result = run_retrieval_task(
                        task_entries=task_entries,
                        variant_feature_frames={variant_name: frame},
                        variant_order=(variant_name,),
                        target_variant_name="__classical__",
                    )
                    variant_result = task_result["variants"].get(variant_name, {"status": "not_run"})
            else:
                if variant_specs[variant_name]["optimized_head"]:
                    variant_result = run_optimized_supervised_variant(
                        merged,
                        task_type=task_type,
                        variant_name=variant_name,
                    )
                else:
                    variant_result = run_classical_variant(merged, task_type=task_type)
            variant_results[variant_name] = variant_result
        full_result = variant_results["chronaris_opt"]
        task_rows = _build_task_rows(
            task_name=task_name,
            task_type=task_type,
            variant_results=variant_results,
            variant_specs=variant_specs,
            full_result=full_result,
        )
        table_rows.extend(task_rows)
        task_results[str(task_name)] = {
            "task_type": task_type,
            "variants": variant_results,
        }
        progress.update("task_finished", task_name=task_name, row_count=len(task_rows))

    table_frame = pd.DataFrame(table_rows)
    table_path = run_root / "chronaris_opt_component_ablation.csv"
    table_frame.to_csv(table_path, index=False)
    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "evidence_layer": PRIVATE_PROXY_EVIDENCE_LAYER,
        "git_commit": config.git_commit,
        "source_manifests": {
            "e_run_manifest_path": config.e_run_manifest_path,
            "f_run_manifest_path": config.f_run_manifest_path,
        },
        "records": {
            "sample_count": int(len(records)),
            "view_count": int(records["view_id"].nunique()),
            "sortie_count": int(records["sortie_id"].nunique()),
        },
        "task_boundary": task_payload["summary"]["thesis_task_boundary"],
        "variant_order": list(variant_order),
        "variant_specs": variant_specs,
        "rows": table_rows,
        "tasks": task_results,
        "diagnostics": {
            "base_variants": base_diagnostics,
            "optimized_full": {
                config.target_variant_name: full_opt_diagnostics.get(config.target_variant_name),
                optimized_no_mask_variant_name(config.target_variant_name): full_opt_diagnostics.get(
                    optimized_no_mask_variant_name(config.target_variant_name)
                ),
            },
            "optimized_no_time_residual": no_residual_diagnostics,
        },
    }
    summary_path = run_root / "chronaris_opt_component_ablation.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"stage-i-private-component-ablation-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_private_component_ablation_report(summary) + "\n",
        encoding="utf-8",
    )
    progress.finish(
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
    )
    return StageIPrivateComponentAblationRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        table_path=str(table_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_stage_i_private_component_ablation_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Private Proxy Component Ablation - {summary['run_id']}",
        "",
        f"- evidence_layer: `{summary['evidence_layer']}`",
        f"- task_boundary: `{summary['task_boundary']}`",
        f"- source_manifests: `{summary['source_manifests']}`",
        "",
        "## Reading",
        "",
        "1. 当前全部结果都属于 `private proxy benchmark evidence`，`T1/T2/T3` 不是人工真值 thesis task fully closed。",
        "2. `chronaris_opt` 作为 full candidate，分别对比移除因果掩码、移除时间残差、移除 task-aware head 后的退化情况。",
        "",
        "## Component Table",
        "",
        "| task | variant | component | primary_metric | value | delta_vs_full | note |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| "
            f"`{row['task_name']}` | "
            f"`{row['variant_name']}` | "
            f"`{row['component']}` | "
            f"`{row['primary_metric_name']}` | "
            f"{row['primary_metric_value']:.6f} | "
            f"{row['delta_vs_full']:.6f} | "
            f"{row['delta_note']} |"
        )
    return "\n".join(lines)


def _build_task_rows(
    *,
    task_name: str,
    task_type: str,
    variant_results: Mapping[str, Mapping[str, object]],
    variant_specs: Mapping[str, Mapping[str, object]],
    full_result: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    full_metric = _extract_primary_metric(task_type=task_type, result=full_result)
    for variant_name, result in variant_results.items():
        if result.get("status", "completed") != "completed":
            continue
        primary_metric = _extract_primary_metric(task_type=task_type, result=result)
        delta = _compute_delta_vs_full(
            task_type=task_type,
            primary_metric=primary_metric["value"],
            full_value=full_metric["value"],
        )
        rows.append(
            {
                "task_name": task_name,
                "task_type": task_type,
                "variant_name": variant_name,
                "component": variant_specs[variant_name]["component"],
                "optimized_head": bool(variant_specs[variant_name]["optimized_head"]),
                "evidence_layer": PRIVATE_PROXY_EVIDENCE_LAYER,
                "primary_metric_name": primary_metric["name"],
                "primary_metric_value": primary_metric["value"],
                "delta_vs_full": delta,
                "delta_note": _delta_note(task_type=task_type),
            }
        )
    return rows


def _extract_primary_metric(*, task_type: str, result: Mapping[str, object]) -> dict[str, float | str]:
    if task_type == "classification":
        best_metrics = result.get("best_metrics", {})
        return {"name": "macro_f1", "value": float(best_metrics.get("macro_f1", 0.0))}
    if task_type == "regression":
        best_metrics = result.get("best_metrics", {})
        return {"name": "rmse", "value": float(best_metrics.get("rmse", 0.0))}
    return {"name": "top1_accuracy", "value": float(result.get("top1_accuracy", 0.0))}


def _compute_delta_vs_full(*, task_type: str, primary_metric: float, full_value: float) -> float:
    if task_type == "regression":
        return primary_metric - full_value
    return full_value - primary_metric


def _delta_note(*, task_type: str) -> str:
    if task_type == "regression":
        return "positive means worse rmse"
    return "positive means score drop"


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
