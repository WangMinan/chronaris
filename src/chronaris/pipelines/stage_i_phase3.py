"""Stage I Phase 3 orchestration and closure helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.dataset import (
    StageIDatasetSummary,
    StageITaskEntry,
    build_nasa_csm_task_entries,
    build_uab_task_entries,
    dump_stage_i_summary,
    dump_stage_i_task_entries,
)
from chronaris.features import StageIFeatureTableResult, build_nasa_csm_feature_table, build_uab_feature_table
from chronaris.pipelines.stage_i_baseline import (
    StageIBaselineArtifacts,
    render_stage_i_baseline_report,
    run_stage_i_baselines,
    write_baseline_artifacts,
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


def build_stage_i_dataset_summary(
    entries: Sequence[StageITaskEntry],
    feature_result: StageIFeatureTableResult,
) -> StageIDatasetSummary:
    subset_counts = Counter(entry.subset_id for entry in entries)
    sample_granularity_counts = Counter(entry.sample_granularity for entry in entries)
    training_role_counts = Counter(entry.training_role for entry in entries)
    task_family_counts = Counter((entry.task_family or "unknown") for entry in entries)
    objective_distributions: dict[str, Counter[str]] = defaultdict(Counter)
    label_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    subjective_counts: Counter[str] = Counter()
    subset_source_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for entry in entries:
        source_key = str(entry.context_payload.get("source_partition") or "default")
        subset_source_counts[entry.subset_id][source_key] += 1
        if entry.objective_label_name is not None and entry.objective_label_value is not None:
            objective_distributions[entry.subset_id][str(entry.objective_label_value)] += 1
            label_key = str(entry.label_namespace or entry.objective_label_name)
            label_distribution[label_key][str(entry.objective_label_value)] += 1
        if entry.subjective_target_name is not None and entry.subjective_target_value is not None:
            subjective_counts[entry.subset_id] += 1

    return StageIDatasetSummary(
        dataset_id=entries[0].dataset_id,
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        entry_count=len(entries),
        recording_count=len({entry.recording_id or entry.session_id for entry in entries}),
        window_count=sum(1 for entry in entries if entry.sample_granularity == "window"),
        sample_granularity_counts=dict(sample_granularity_counts),
        subset_counts=dict(subset_counts),
        subset_source_counts={
            subset_id: dict(counter)
            for subset_id, counter in subset_source_counts.items()
        },
        training_role_counts=dict(training_role_counts),
        split_group_count=len({entry.split_group for entry in entries}),
        task_family_counts=dict(task_family_counts),
        label_distribution={key: dict(value) for key, value in label_distribution.items()},
        objective_label_distributions={
            subset_id: dict(counter) for subset_id, counter in objective_distributions.items()
        },
        subjective_target_counts=dict(subjective_counts),
        feature_count=len(feature_result.feature_columns),
        feature_group_counts={
            group_name: len(columns)
            for group_name, columns in feature_result.feature_group_columns.items()
        },
        eeg_feature_count=len(feature_result.eeg_feature_columns),
        ecg_feature_count=len(feature_result.ecg_feature_columns),
        peripheral_feature_count=len(feature_result.peripheral_feature_columns),
        missing_ecg_session_counts=dict(feature_result.missing_ecg_session_counts),
    )


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
    _write_prepared_dataset(
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
    uab_report_path = uab_artifact_root / "baseline_report.md"
    uab_report_path.write_text(
        render_stage_i_baseline_report(
            artifact_root=uab_artifact_root,
            dataset_summary=uab_summary.to_dict(),
            objective_metrics=uab_baselines.objective_metrics,
            subjective_metrics=uab_baselines.subjective_metrics,
            plot_paths=uab_baselines.plot_paths,
        )
        + "\n",
        encoding="utf-8",
    )

    nasa_entries = build_nasa_csm_task_entries(config.dataset_root).entries
    nasa_feature_result = build_nasa_csm_feature_table(config.dataset_root, nasa_entries)
    nasa_summary = build_stage_i_dataset_summary(nasa_entries, nasa_feature_result)
    _write_prepared_dataset(
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
    nasa_report_path = nasa_artifact_root / "baseline_report.md"
    nasa_report_path.write_text(
        render_stage_i_baseline_report(
            artifact_root=nasa_artifact_root,
            dataset_summary=nasa_summary.to_dict(),
            objective_metrics=nasa_baselines.objective_metrics,
            subjective_metrics=None,
            plot_paths=nasa_baselines.plot_paths,
        )
        + "\n",
        encoding="utf-8",
    )

    comparison_summary = _build_uab_session_window_comparison(
        config.prior_uab_session_artifact_root,
        uab_baselines,
    )
    closure_summary = {
        "phase": "stage_i_phase3",
        "run_id": config.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact_root": str(artifact_root),
        "uab_window": _summarize_artifact_bundle(uab_artifact_root, uab_summary, uab_baselines),
        "nasa_attention": _summarize_artifact_bundle(nasa_artifact_root, nasa_summary, nasa_baselines),
        "uab_session_comparison": comparison_summary,
        "reports": {
            "uab_baseline_report": str(uab_report_path),
            "nasa_baseline_report": str(nasa_report_path),
        },
    }
    closure_summary_path = artifact_root / "closure_summary.json"
    closure_summary_path.write_text(json.dumps(closure_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    closure_report_path = artifact_root / "stage_i_phase3_closure_report.md"
    closure_report_path.write_text(render_stage_i_phase3_report(closure_summary) + "\n", encoding="utf-8")

    return StageIPhase3RunResult(
        artifact_root=str(artifact_root),
        uab_artifact_root=str(uab_artifact_root),
        nasa_artifact_root=str(nasa_artifact_root),
        closure_summary_path=str(closure_summary_path),
        closure_report_path=str(closure_report_path),
        closure_summary=closure_summary,
    )


def render_stage_i_phase3_report(summary: Mapping[str, object]) -> str:
    """Render the top-level Stage I Phase 3 closure markdown."""

    lines = [
        "# Stage I Phase 3 Closure",
        "",
        f"- 生成时间：`{summary['generated_at_utc']}`",
        f"- 机器产物根目录：`{summary['artifact_root']}`",
        "",
        "## UAB Window 主线",
        "",
    ]
    lines.extend(_render_baseline_bundle(summary["uab_window"]))
    lines.extend(["## NASA Attention 主线", ""])
    lines.extend(_render_baseline_bundle(summary["nasa_attention"]))
    comparison = summary.get("uab_session_comparison", {})
    if comparison:
        lines.extend(["## UAB Session vs Window", ""])
        for track_name, groups in comparison.items():
            lines.append(f"### {track_name}")
            lines.append("")
            for group_name, payload in groups.items():
                field = "macro_f1" if track_name == "objective" else "rmse"
                lines.append(
                    f"- {group_name}: session `{payload[f'session_{field}']:.4f}` / window `{payload[f'window_{field}']:.4f}` / delta `{payload['delta']:.4f}`"
                )
            lines.append("")
    return "\n".join(lines)


def _write_prepared_dataset(
    *,
    artifact_root: Path,
    entries: Sequence[StageITaskEntry],
    feature_result: StageIFeatureTableResult,
    summary: StageIDatasetSummary,
) -> None:
    dump_stage_i_task_entries(entries, path=artifact_root / "task_manifest.jsonl")
    feature_result.feature_table.to_parquet(artifact_root / "feature_table.parquet", index=False)
    (artifact_root / "feature_schema.json").write_text(
        json.dumps(feature_result.feature_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dump_stage_i_summary(summary, path=artifact_root / "dataset_summary.json")


def _summarize_artifact_bundle(
    artifact_root: Path,
    summary: StageIDatasetSummary,
    baselines: StageIBaselineArtifacts,
) -> dict[str, object]:
    payload = {
        "artifact_root": str(artifact_root),
        "dataset_summary": summary.to_dict(),
        "objective_primary": _extract_primary_metrics(baselines.objective_metrics),
        "subjective_primary": _extract_primary_metrics(baselines.subjective_metrics),
        "plot_paths": dict(baselines.plot_paths),
    }
    return payload


def _extract_primary_metrics(metrics: Mapping[str, object] | None) -> dict[str, dict[str, float]]:
    if metrics is None:
        return {}
    field = "macro_f1" if metrics["track"] == "objective" else "rmse"
    secondary = "balanced_accuracy" if metrics["track"] == "objective" else "mae"
    result: dict[str, dict[str, float]] = {}
    for group_name, payload in metrics["primary_results"].items():
        best_model_name = payload["best_model_name"]
        best_metrics = payload["models"][best_model_name]
        result[group_name] = {
            field: float(best_metrics[field]),
            secondary: float(best_metrics[secondary]),
            "sample_count": float(payload["sample_count"]),
        }
    return result


def _build_uab_session_window_comparison(
    prior_artifact_root: str | None,
    uab_window_baselines: StageIBaselineArtifacts,
) -> dict[str, dict[str, dict[str, float]]]:
    if not prior_artifact_root:
        return {}
    root = Path(prior_artifact_root)
    objective_path = root / "objective_metrics.json"
    subjective_path = root / "subjective_metrics.json"
    if not objective_path.exists() or not subjective_path.exists():
        return {}

    session_objective = json.loads(objective_path.read_text(encoding="utf-8"))
    session_subjective = json.loads(subjective_path.read_text(encoding="utf-8"))
    comparison = {
        "objective": _compare_track(
            session_metrics=session_objective,
            window_metrics=uab_window_baselines.objective_metrics,
            field="macro_f1",
        ),
        "subjective": _compare_track(
            session_metrics=session_subjective,
            window_metrics=uab_window_baselines.subjective_metrics,
            field="rmse",
        ),
    }
    return comparison


def _compare_track(
    *,
    session_metrics: Mapping[str, object],
    window_metrics: Mapping[str, object] | None,
    field: str,
) -> dict[str, dict[str, float]]:
    if window_metrics is None:
        return {}
    comparison: dict[str, dict[str, float]] = {}
    for group_name, payload in window_metrics["primary_results"].items():
        session_payload = session_metrics["primary_results"][group_name]
        session_best = session_payload["models"][session_payload["best_model_name"]]
        window_best = payload["models"][payload["best_model_name"]]
        delta = float(window_best[field]) - float(session_best[field])
        comparison[group_name] = {
            f"session_{field}": float(session_best[field]),
            f"window_{field}": float(window_best[field]),
            "delta": delta,
        }
    return comparison


def _render_baseline_bundle(bundle: Mapping[str, object]) -> list[str]:
    dataset_summary = bundle["dataset_summary"]
    lines = [
        f"- artifact root: `{bundle['artifact_root']}`",
        f"- entry_count: `{dataset_summary['entry_count']}`",
        f"- recording_count: `{dataset_summary['recording_count']}`",
        f"- window_count: `{dataset_summary['window_count']}`",
        f"- subset_counts: `{json.dumps(dataset_summary['subset_counts'], ensure_ascii=False)}`",
        "",
        "### 主结果",
        "",
    ]
    objective_primary = bundle.get("objective_primary", {})
    for group_name, metrics in objective_primary.items():
        if "macro_f1" in metrics:
            lines.append(
                f"- {group_name}: macro-F1 `{metrics['macro_f1']:.4f}`, balanced accuracy `{metrics['balanced_accuracy']:.4f}`"
            )
    subjective_primary = bundle.get("subjective_primary", {})
    for group_name, metrics in subjective_primary.items():
        if "rmse" in metrics:
            lines.append(
                f"- {group_name}: RMSE `{metrics['rmse']:.4f}`, MAE `{metrics['mae']:.4f}`"
            )
    lines.append("")
    return lines
