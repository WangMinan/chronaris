"""Private-task benchmark pipeline over Stage H all-window exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chronaris.dataset.stage_i_private_contracts import dump_stage_i_private_task_entries
from chronaris.pipelines.stage_i_private_benchmark_data import (
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    VARIANT_ORDER,
    build_variant_feature_frames,
    derive_private_task_entries,
    load_aligned_private_records,
)
from chronaris.pipelines.stage_i_private_benchmark_models import (
    build_conclusion,
    json_default,
    run_retrieval_task,
    run_supervised_task,
    write_task_plots,
)
from chronaris.pipelines.stage_i_private_optimization import (
    DEFAULT_OPTIMIZED_VARIANT_NAME,
    build_private_variant_order,
    optimized_no_mask_variant_name,
    write_optimized_candidate_artifacts,
)

DEEP_MODEL_ORDER = ("mult", "contiformer")


@dataclass(frozen=True, slots=True)
class StageIPrivateBenchmarkConfig:
    """Run configuration for the private-task benchmark suite."""

    run_id: str
    e_run_manifest_path: str
    f_run_manifest_path: str
    output_root: str = "docs/reports/assets/stage_i_private"
    report_root: str = "docs/reports"
    deep_model_names: tuple[str, ...] = DEEP_MODEL_ORDER
    deep_epochs: int = 1
    deep_learning_rate: float = 1e-3
    deep_batch_size: int = 32
    deep_hidden_dim: int = 32
    deep_num_heads: int = 2
    deep_layers: int = 1
    deep_dropout: float = 0.1
    max_deep_folds: int | None = None
    seed: int = 42
    enable_optimized_chronaris: bool = False
    target_variant_name: str = DEFAULT_OPTIMIZED_VARIANT_NAME
    lag_window_points: int = 3
    residual_mode: str = "raw_window_stats"


@dataclass(frozen=True, slots=True)
class StageIPrivateBenchmarkRunResult:
    """Artifacts written by one private-task benchmark run."""

    artifact_root: str
    task_manifest_path: str
    task_summary_path: str
    benchmark_summary_path: str
    alignment_report_path: str
    causal_report_path: str
    optimality_report_path: str
    optimization_report_path: str | None
    optimized_candidate_summary_path: str | None
    optimized_candidate_metrics_path: str | None
    summary: Mapping[str, object]


def run_stage_i_private_benchmark(
    config: StageIPrivateBenchmarkConfig,
) -> StageIPrivateBenchmarkRunResult:
    artifact_root = Path(config.output_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    records = load_aligned_private_records(
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
    task_payload = derive_private_task_entries(records)
    task_manifest_path = artifact_root / "private_task_manifest.jsonl"
    task_summary_path = artifact_root / "task_summary.json"
    dump_stage_i_private_task_entries(task_payload["entries"], path=task_manifest_path)
    task_summary_path.write_text(
        json.dumps(task_payload["summary"], ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    resolved_target_variant_name = (
        config.target_variant_name if config.enable_optimized_chronaris else "g_min"
    )
    variant_order = build_private_variant_order(
        VARIANT_ORDER,
        enable_optimized_chronaris=config.enable_optimized_chronaris,
        target_variant_name=config.target_variant_name,
    )
    variant_frames, diagnostic_summary = build_variant_feature_frames(
        records,
        enable_optimized_chronaris=config.enable_optimized_chronaris,
        target_variant_name=config.target_variant_name,
        lag_window_points=config.lag_window_points,
        residual_mode=config.residual_mode,
    )
    task_results = {
        TASK_MANEUVER: run_supervised_task(
            task_name=TASK_MANEUVER,
            task_type="classification",
            task_entries=task_payload["by_task"][TASK_MANEUVER],
            variant_feature_frames=variant_frames,
            variant_order=variant_order,
            target_variant_name=resolved_target_variant_name,
            deep_model_names=config.deep_model_names,
            records=records,
            config=config,
        ),
        TASK_RESPONSE: run_supervised_task(
            task_name=TASK_RESPONSE,
            task_type="regression",
            task_entries=task_payload["by_task"][TASK_RESPONSE],
            variant_feature_frames=variant_frames,
            variant_order=variant_order,
            target_variant_name=resolved_target_variant_name,
            deep_model_names=config.deep_model_names,
            records=records,
            config=config,
        ),
        TASK_RETRIEVAL: run_retrieval_task(
            task_entries=task_payload["by_task"][TASK_RETRIEVAL],
            variant_feature_frames=variant_frames,
            variant_order=variant_order,
            target_variant_name=resolved_target_variant_name,
        ),
    }
    plot_paths = write_task_plots(task_results, artifact_root=artifact_root)
    conclusion = build_conclusion(
        task_results,
        diagnostic_summary,
        target_variant_name=resolved_target_variant_name,
    )
    summary = {
        "run_id": config.run_id,
        "artifact_root": str(artifact_root),
        "task_manifest_path": str(task_manifest_path),
        "task_summary_path": str(task_summary_path),
        "enable_optimized_chronaris": config.enable_optimized_chronaris,
        "target_variant_name": resolved_target_variant_name,
        "private_optimality_supported": conclusion["private_optimality_supported"],
        "criterion_details": conclusion["criterion_details"],
        "task_metrics": _compact_task_metrics(
            task_results,
            target_variant_name=resolved_target_variant_name,
        ),
        "records": {
            "sample_count": int(len(records)),
            "view_count": int(records["view_id"].nunique()),
            "sortie_count": int(records["sortie_id"].nunique()),
        },
        "variant_order": list(variant_order),
        "diagnostics": diagnostic_summary,
        "tasks": task_results,
        "plots": plot_paths,
        "conclusion": conclusion,
    }
    optimized_candidate_summary_path = None
    optimized_candidate_metrics_path = None
    if config.enable_optimized_chronaris:
        (
            optimized_candidate_summary_path,
            optimized_candidate_metrics_path,
        ) = write_optimized_candidate_artifacts(
            summary,
            artifact_root=artifact_root,
            target_variant_name=resolved_target_variant_name,
        )
        summary["optimized_candidate_summary_path"] = optimized_candidate_summary_path
        summary["optimized_candidate_metrics_path"] = optimized_candidate_metrics_path
    benchmark_summary_path = artifact_root / "private_benchmark_summary.json"
    benchmark_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    alignment_report_path = report_root / f"private-alignment-support-{config.run_id}.md"
    causal_report_path = report_root / f"private-causal-fusion-support-{config.run_id}.md"
    optimality_report_path = report_root / f"private-optimality-summary-{config.run_id}.md"
    optimization_report_path = (
        report_root / f"private-optimization-summary-{config.run_id}.md"
        if config.enable_optimized_chronaris
        else None
    )
    alignment_report_path.write_text(_render_alignment_report(summary) + "\n", encoding="utf-8")
    causal_report_path.write_text(_render_causal_report(summary) + "\n", encoding="utf-8")
    optimality_report_path.write_text(_render_optimality_report(summary) + "\n", encoding="utf-8")
    if optimization_report_path is not None:
        optimization_report_path.write_text(
            _render_optimization_report(summary) + "\n",
            encoding="utf-8",
        )
    return StageIPrivateBenchmarkRunResult(
        artifact_root=str(artifact_root),
        task_manifest_path=str(task_manifest_path),
        task_summary_path=str(task_summary_path),
        benchmark_summary_path=str(benchmark_summary_path),
        alignment_report_path=str(alignment_report_path),
        causal_report_path=str(causal_report_path),
        optimality_report_path=str(optimality_report_path),
        optimization_report_path=str(optimization_report_path) if optimization_report_path is not None else None,
        optimized_candidate_summary_path=optimized_candidate_summary_path,
        optimized_candidate_metrics_path=optimized_candidate_metrics_path,
        summary=summary,
    )


def _render_alignment_report(summary: Mapping[str, object]) -> str:
    t1 = summary["tasks"][TASK_MANEUVER]
    t2 = summary["tasks"][TASK_RESPONSE]
    lines = [
        f"# Private Alignment Support - {summary['run_id']}",
        "",
        f"- alignment gain supported: `{summary['conclusion']['alignment_gain_supported']}`",
        "",
        "## T1",
        "",
        "| variant | macro-F1 | balanced accuracy |",
        "| --- | ---: | ---: |",
    ]
    for variant_name in ("naive_sync", "e_baseline", "f_full"):
        payload = t1["variants"][variant_name]
        metrics = payload.get("best_metrics", {})
        lines.append(
            f"| `{variant_name}` | {metrics.get('macro_f1', float('nan')):.6f} | "
            f"{metrics.get('balanced_accuracy', float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## T2",
            "",
            "| variant | RMSE | MAE | Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant_name in ("naive_sync", "e_baseline", "f_full"):
        payload = t2["variants"][variant_name]
        metrics = payload.get("best_metrics", {})
        lines.append(
            f"| `{variant_name}` | {metrics.get('rmse', float('nan')):.6f} | "
            f"{metrics.get('mae', float('nan')):.6f} | {metrics.get('spearman', float('nan')):.6f} |"
        )
    return "\n".join(lines)


def _render_causal_report(summary: Mapping[str, object]) -> str:
    t1 = summary["tasks"][TASK_MANEUVER]
    t2 = summary["tasks"][TASK_RESPONSE]
    target_variant_name = summary["conclusion"]["target_variant_name"]
    no_mask_name = summary["conclusion"]["no_mask_variant_name"]
    lines = [
        f"# Private Causal Fusion Support - {summary['run_id']}",
        "",
        f"- causal gain supported: `{summary['conclusion']['causal_gain_supported']}`",
        f"- diagnostic supported: `{summary['conclusion']['diagnostic_supported']}`",
        f"- target variant: `{target_variant_name}`",
        "",
        "## T1",
        "",
        "| variant | macro-F1 | balanced accuracy |",
        "| --- | ---: | ---: |",
    ]
    for variant_name in ("f_full", target_variant_name, no_mask_name):
        metrics = t1["variants"][variant_name].get("best_metrics", {})
        lines.append(
            f"| `{variant_name}` | {metrics.get('macro_f1', float('nan')):.6f} | "
            f"{metrics.get('balanced_accuracy', float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## T2",
            "",
            "| variant | RMSE | MAE | Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant_name in ("f_full", target_variant_name, no_mask_name):
        metrics = t2["variants"][variant_name].get("best_metrics", {})
        lines.append(
            f"| `{variant_name}` | {metrics.get('rmse', float('nan')):.6f} | "
            f"{metrics.get('mae', float('nan')):.6f} | {metrics.get('spearman', float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| variant | attention entropy | top-event concentration | event-mask interference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant_name in ("f_full", target_variant_name, no_mask_name):
        metrics = summary["diagnostics"][variant_name]
        lines.append(
            f"| `{variant_name}` | {metrics['mean_attention_entropy']:.6f} | "
            f"{metrics['mean_top_event_concentration']:.6f} | {metrics['mean_event_mask_interference']:.6f} |"
        )
    return "\n".join(lines)


def _render_optimization_report(summary: Mapping[str, object]) -> str:
    target_variant_name = summary["target_variant_name"]
    no_mask_name = optimized_no_mask_variant_name(target_variant_name)
    lines = [
        f"# Private Optimization Summary - {summary['run_id']}",
        "",
        f"- target variant: `{target_variant_name}`",
        f"- no-mask variant: `{no_mask_name}`",
        f"- private optimality supported: `{summary['private_optimality_supported']}`",
        "",
        "## Criteria",
        "",
        "| check | pass |",
        "| --- | ---: |",
    ]
    for name, passed in summary["criterion_details"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.extend(
        [
            "",
            "## Target Metrics",
            "",
            "| task | variant | primary metrics |",
            "| --- | --- | --- |",
        ]
    )
    for task_name in (TASK_MANEUVER, TASK_RESPONSE, TASK_RETRIEVAL):
        for variant_name in (target_variant_name, no_mask_name):
            payload = summary["task_metrics"][task_name].get(variant_name, {})
            lines.append(
                f"| `{task_name}` | `{variant_name}` | {_format_metric_payload(payload)} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- optimized candidate summary: `{summary.get('optimized_candidate_summary_path')}`",
            f"- optimized candidate metrics: `{summary.get('optimized_candidate_metrics_path')}`",
        ]
    )
    return "\n".join(lines)


def _compact_task_metrics(
    task_results: Mapping[str, object],
    *,
    target_variant_name: str,
) -> dict[str, object]:
    no_mask_name = (
        "g_no_causal_mask"
        if target_variant_name == "g_min"
        else optimized_no_mask_variant_name(target_variant_name)
    )
    compact: dict[str, object] = {}
    for task_name in (TASK_MANEUVER, TASK_RESPONSE, TASK_RETRIEVAL):
        task = task_results[task_name]
        compact[task_name] = {
            target_variant_name: _extract_variant_metrics(task["variants"].get(target_variant_name)),
            no_mask_name: _extract_variant_metrics(task["variants"].get(no_mask_name)),
            "best_variant": task.get("best_variant"),
        }
    return compact


def _extract_variant_metrics(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping) or payload.get("status") != "completed":
        return {}
    if "best_metrics" in payload:
        return dict(payload["best_metrics"])
    return {
        key: payload[key]
        for key in ("sample_count", "top1_accuracy", "mrr")
        if key in payload
    }


def _format_metric_payload(payload: Mapping[str, object]) -> str:
    if not payload:
        return "`not_run`"
    preferred = (
        "macro_f1",
        "balanced_accuracy",
        "rmse",
        "mae",
        "top1_accuracy",
        "mrr",
    )
    parts = []
    for key in preferred:
        if key in payload:
            value = payload[key]
            parts.append(f"{key}={float(value):.6f}")
    return ", ".join(parts) if parts else "`completed`"


def _render_optimality_report(summary: Mapping[str, object]) -> str:
    t1 = summary["tasks"][TASK_MANEUVER]
    t2 = summary["tasks"][TASK_RESPONSE]
    t3 = summary["tasks"][TASK_RETRIEVAL]
    lines = [
        f"# Private Optimality Summary - {summary['run_id']}",
        "",
        f"- private optimality supported: `{summary['conclusion']['private_optimality_supported']}`",
        f"- best T1 variant: `{t1['best_variant']['name']}`",
        f"- best T2 variant: `{t2['best_variant']['name']}`",
        f"- best T3 variant: `{t3['best_variant']['name']}`",
        f"- best deep T1 model: `{t1['best_deep_model']['name']}`",
        f"- best deep T2 model: `{t2['best_deep_model']['name']}`",
        "",
        "## Retrieval",
        "",
        "| variant | top-1 accuracy | MRR |",
        "| --- | ---: | ---: |",
    ]
    for variant_name, payload in t3["variants"].items():
        if payload.get("status") != "completed":
            continue
        lines.append(
            f"| `{variant_name}` | {payload['top1_accuracy']:.6f} | {payload['mrr']:.6f} |"
        )
    return "\n".join(lines)
