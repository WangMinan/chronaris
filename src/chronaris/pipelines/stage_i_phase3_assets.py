"""Artifact and summary helpers for Stage I Phase 3 closure."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.dataset import StageIDatasetSummary, StageITaskEntry, dump_stage_i_summary, dump_stage_i_task_entries
from chronaris.features import StageIFeatureTableResult
from chronaris.pipelines.stage_i_baseline import StageIBaselineArtifacts, render_stage_i_baseline_report


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
        generated_at_utc=pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        entry_count=len(entries),
        recording_count=len({entry.recording_id or entry.session_id for entry in entries}),
        window_count=sum(1 for entry in entries if entry.sample_granularity == "window"),
        sample_granularity_counts=dict(sample_granularity_counts),
        subset_counts=dict(subset_counts),
        subset_source_counts={subset_id: dict(counter) for subset_id, counter in subset_source_counts.items()},
        training_role_counts=dict(training_role_counts),
        split_group_count=len({entry.split_group for entry in entries}),
        task_family_counts=dict(task_family_counts),
        label_distribution={key: dict(value) for key, value in label_distribution.items()},
        objective_label_distributions={subset_id: dict(counter) for subset_id, counter in objective_distributions.items()},
        subjective_target_counts=dict(subjective_counts),
        feature_count=len(feature_result.feature_columns),
        feature_group_counts={group_name: len(columns) for group_name, columns in feature_result.feature_group_columns.items()},
        eeg_feature_count=len(feature_result.eeg_feature_columns),
        ecg_feature_count=len(feature_result.ecg_feature_columns),
        peripheral_feature_count=len(feature_result.peripheral_feature_columns),
        missing_ecg_session_counts=dict(feature_result.missing_ecg_session_counts),
    )


def write_prepared_dataset(
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


def write_local_baseline_report(
    *,
    artifact_root: Path,
    dataset_summary: StageIDatasetSummary,
    baselines: StageIBaselineArtifacts,
) -> Path:
    report_path = artifact_root / "baseline_report.md"
    report_path.write_text(
        render_stage_i_baseline_report(
            artifact_root=artifact_root,
            dataset_summary=dataset_summary.to_dict(),
            objective_metrics=baselines.objective_metrics,
            subjective_metrics=baselines.subjective_metrics,
            plot_paths=baselines.plot_paths,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def summarize_artifact_bundle(
    artifact_root: Path,
    summary: StageIDatasetSummary,
    baselines: StageIBaselineArtifacts,
) -> dict[str, object]:
    return {
        "artifact_root": str(artifact_root),
        "dataset_summary": summary.to_dict(),
        "objective_primary": extract_primary_metrics(baselines.objective_metrics),
        "subjective_primary": extract_primary_metrics(baselines.subjective_metrics),
        "plot_paths": dict(baselines.plot_paths),
        "objective_ablation_results": baselines.objective_metrics["ablation_results"] if baselines.objective_metrics else {},
        "subjective_ablation_results": baselines.subjective_metrics["ablation_results"] if baselines.subjective_metrics else {},
    }


def load_stage_i_baseline_artifacts(
    artifact_root: Path,
    *,
    require_subjective: bool = False,
) -> StageIBaselineArtifacts:
    objective_path = artifact_root / "objective_metrics.json"
    subjective_path = artifact_root / "subjective_metrics.json"
    fold_predictions_path = artifact_root / "fold_predictions.csv"
    plot_root = artifact_root / "plots"
    if not objective_path.exists():
        raise FileNotFoundError(f"missing objective metrics for Phase 3 closure: {objective_path}")
    if require_subjective and not subjective_path.exists():
        raise FileNotFoundError(f"missing subjective metrics for Phase 3 closure: {subjective_path}")
    if not fold_predictions_path.exists():
        raise FileNotFoundError(f"missing fold predictions for Phase 3 closure: {fold_predictions_path}")
    return StageIBaselineArtifacts(
        objective_metrics=json.loads(objective_path.read_text(encoding="utf-8")),
        subjective_metrics=json.loads(subjective_path.read_text(encoding="utf-8")) if subjective_path.exists() else None,
        fold_predictions=pd.read_csv(fold_predictions_path),
        plot_paths={path.stem: str(path) for path in sorted(plot_root.glob("*.png"))},
    )


def read_stage_i_dataset_summary(artifact_root: Path) -> StageIDatasetSummary:
    dataset_summary = json.loads((artifact_root / "dataset_summary.json").read_text(encoding="utf-8"))
    return StageIDatasetSummary(
        dataset_id=dataset_summary["dataset_id"],
        generated_at_utc=dataset_summary["generated_at_utc"],
        entry_count=dataset_summary["entry_count"],
        recording_count=dataset_summary["recording_count"],
        window_count=dataset_summary["window_count"],
        sample_granularity_counts=dataset_summary["sample_granularity_counts"],
        subset_counts=dataset_summary["subset_counts"],
        subset_source_counts=dataset_summary["subset_source_counts"],
        training_role_counts=dataset_summary["training_role_counts"],
        split_group_count=dataset_summary["split_group_count"],
        task_family_counts=dataset_summary["task_family_counts"],
        label_distribution=dataset_summary["label_distribution"],
        objective_label_distributions=dataset_summary["objective_label_distributions"],
        subjective_target_counts=dataset_summary["subjective_target_counts"],
        feature_count=dataset_summary["feature_count"],
        feature_group_counts=dataset_summary["feature_group_counts"],
        eeg_feature_count=dataset_summary["eeg_feature_count"],
        ecg_feature_count=dataset_summary["ecg_feature_count"],
        peripheral_feature_count=dataset_summary.get("peripheral_feature_count", 0),
        missing_ecg_session_counts=dataset_summary["missing_ecg_session_counts"],
    )


def extract_primary_metrics(metrics: Mapping[str, object] | None) -> dict[str, dict[str, float]]:
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


def build_uab_session_window_comparison(
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
    return {
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
        comparison[group_name] = {
            f"session_{field}": float(session_best[field]),
            f"window_{field}": float(window_best[field]),
            "delta": float(window_best[field]) - float(session_best[field]),
        }
    return comparison
