"""Deep baseline pipelines for Stage I sequence experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
    save_confusion_matrix_plot,
    save_grouped_bar_plot,
    save_regression_plot,
)
from chronaris.features.stage_i_sequences import STAGE_H_CASE_DATASET_ID
from chronaris.pipelines.stage_i.common.baseline_models import build_loso_splits
from chronaris.pipelines.stage_i.public.deep_baseline_case import (
    _run_real_sortie_case_study as _run_real_sortie_case_study_impl,
)
from chronaris.pipelines.stage_i.public.deep_baseline_reporting import (
    _render_comparison_report,
    _render_deep_baseline_report,
)
from chronaris.pipelines.stage_i.public.deep_baseline_runtime import (
    _build_prediction_frame,
    _deep_model_config_dict,
    _extract_target_values,
    _fit_predict_classification,
    _fit_predict_regression,
    _infer_profile as _infer_profile_runtime,
    _load_prepared_sequence_dataset,
    _load_reference_comparison,
    _safe_regression_fallback,
    _sanitize_classification_logits,
    _sanitize_regression_outputs,
    _select_indices,
)
from chronaris.pipelines.stage_i.common.run_observer import open_stage_i_run_observer
from chronaris.pipelines.torch_runtime import resolve_torch_device_name

DATASET_RUN_ORDER = (STAGE_H_CASE_DATASET_ID, "uab_workload_dataset", "nasa_csm")
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIDeepBaselineConfig:
    model_name: str
    dataset_id: str
    profile: str
    prepared_artifact_root: str
    artifact_root: str
    reference_artifact_root: str | None = None
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 256
    hidden_dim: int = 32
    num_heads: int = 2
    layers: int = 1
    dropout: float = 0.1
    fusion_event_bias_weight: float = 0.25
    fusion_lag_window_points: int | None = None
    fusion_normalize_states: bool = True
    max_folds: int | None = None
    seed: int = 42
    device: str = "auto"
    train_sampling_policy: str = "none"
    regression_loss: str = "mse"
    huber_delta: float = 1.0
    target_transform: str = "none"
    gradient_clip_max_norm: float | None = None
    weight_decay: float = 0.0
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20


@dataclass(frozen=True, slots=True)
class StageIDeepBaselineRunResult:
    dataset_id: str
    model_name: str
    artifact_root: str
    summary_path: str
    report_path: str
    predictions_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StageIDeepComparisonConfig:
    model_names: tuple[str, ...]
    dataset_artifact_roots: Mapping[str, str]
    output_root: str
    reference_artifact_roots: Mapping[str, str] | None = None
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 256
    hidden_dim: int = 32
    num_heads: int = 2
    layers: int = 1
    dropout: float = 0.1
    fusion_event_bias_weight: float = 0.25
    fusion_lag_window_points: int | None = None
    fusion_normalize_states: bool = True
    max_folds: int | None = None
    seed: int = 42
    device: str = "auto"
    train_sampling_policy: str = "none"
    regression_loss: str = "mse"
    huber_delta: float = 1.0
    target_transform: str = "none"
    gradient_clip_max_norm: float | None = None
    weight_decay: float = 0.0
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20


@dataclass(frozen=True, slots=True)
class StageIDeepComparisonRunResult:
    artifact_root: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def _infer_profile(dataset_id: str) -> str:
    return _infer_profile_runtime(
        dataset_id,
        real_sortie_dataset_id=STAGE_H_CASE_DATASET_ID,
    )


def run_stage_i_deep_baseline(
    config: StageIDeepBaselineConfig,
) -> StageIDeepBaselineRunResult:
    artifact_root = Path(config.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=artifact_root,
        run_id=artifact_root.name,
        stage_name="stage_i_deep_baseline",
        logger=LOGGER,
        initial_progress={
            "dataset_id": config.dataset_id,
            "model_name": config.model_name,
            "requested_device": config.device,
            "artifact_root": str(artifact_root),
            "epochs": config.epochs,
            "max_folds": config.max_folds,
            "heartbeat_seconds": config.heartbeat_seconds,
            "batch_log_interval": config.batch_log_interval,
        },
    ) as progress:
        dataset = _load_prepared_sequence_dataset(config.prepared_artifact_root)
        if dataset["dataset_id"] != config.dataset_id:
            raise ValueError(
                f"prepared dataset mismatch: expected {config.dataset_id}, got {dataset['dataset_id']}"
            )
        progress.update(
            "dataset_loaded",
            prepared_artifact_root=config.prepared_artifact_root,
            runtime_device=resolve_torch_device_name(config.device),
        )
        if config.dataset_id == STAGE_H_CASE_DATASET_ID:
            summary, predictions = _run_real_sortie_case_study(dataset=dataset, config=config)
        else:
            summary, predictions = _run_public_deep_baseline(dataset=dataset, config=config)
        training_curves = _extract_training_curves(summary)
        training_curves_path = artifact_root / "training_curves.csv"
        pd.DataFrame(training_curves).to_csv(training_curves_path, index=False)
        summary["training_curves_path"] = str(training_curves_path)
        summary["training_curve_count"] = len(training_curves)
        summary_path = artifact_root / "deep_baseline_summary.json"
        report_path = artifact_root / "deep_baseline_report.md"
        predictions_path = artifact_root / "fold_predictions.csv"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            _render_deep_baseline_report(summary) + "\n",
            encoding="utf-8",
        )
        predictions.to_csv(predictions_path, index=False)
        progress.finish(
            summary_path=str(summary_path),
            report_path=str(report_path),
            predictions_path=str(predictions_path),
        )
        return StageIDeepBaselineRunResult(
            dataset_id=config.dataset_id,
            model_name=config.model_name,
            artifact_root=str(artifact_root),
            summary_path=str(summary_path),
            report_path=str(report_path),
            predictions_path=str(predictions_path),
            summary=summary,
        )


def run_stage_i_deep_comparison(
    config: StageIDeepComparisonConfig,
) -> StageIDeepComparisonRunResult:
    artifact_root = Path(config.output_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_results: dict[str, object] = {}
    for dataset_id in DATASET_RUN_ORDER:
        prepared_root = config.dataset_artifact_roots.get(dataset_id)
        if not prepared_root:
            dataset_results[dataset_id] = {"status": "not_run"}
            continue
        dataset_model_results: dict[str, object] = {}
        for model_name in config.model_names:
            model_root = artifact_root / dataset_id / model_name
            result = run_stage_i_deep_baseline(
                StageIDeepBaselineConfig(
                    model_name=model_name,
                    dataset_id=dataset_id,
                    profile=_infer_profile(dataset_id),
                    prepared_artifact_root=prepared_root,
                    artifact_root=str(model_root),
                    reference_artifact_root=(
                        config.reference_artifact_roots or {}
                    ).get(dataset_id),
                    epochs=config.epochs,
                    learning_rate=config.learning_rate,
                    batch_size=config.batch_size,
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    layers=config.layers,
                    dropout=config.dropout,
                    fusion_event_bias_weight=config.fusion_event_bias_weight,
                    fusion_lag_window_points=config.fusion_lag_window_points,
                    fusion_normalize_states=config.fusion_normalize_states,
                    max_folds=config.max_folds,
                    seed=config.seed,
                    device=config.device,
                    train_sampling_policy=config.train_sampling_policy,
                    regression_loss=config.regression_loss,
                    huber_delta=config.huber_delta,
                    target_transform=config.target_transform,
                    gradient_clip_max_norm=config.gradient_clip_max_norm,
                    weight_decay=config.weight_decay,
                    heartbeat_seconds=config.heartbeat_seconds,
                    batch_log_interval=config.batch_log_interval,
                ),
            )
            dataset_model_results[model_name] = {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "report_path": result.report_path,
                "summary": result.summary,
            }
        dataset_results[dataset_id] = {
            "status": "completed",
            "models": dataset_model_results,
        }
    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace(
            "+00:00",
            "Z",
        ),
        "artifact_root": str(artifact_root),
        "runtime_device": resolve_torch_device_name(config.device),
        "dataset_order": list(DATASET_RUN_ORDER),
        "model_names": list(config.model_names),
        "datasets": dataset_results,
    }
    summary_path = artifact_root / "comparison_summary.json"
    report_path = artifact_root / "comparison_report.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_comparison_report(summary) + "\n", encoding="utf-8")
    return StageIDeepComparisonRunResult(
        artifact_root=str(artifact_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )


def _run_public_deep_baseline(
    *,
    dataset: Mapping[str, object],
    config: StageIDeepBaselineConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    bundle = dataset["bundle"]
    entries = dataset["entries"]
    ordered_modalities = tuple(entries[0].modality_schema)
    if config.dataset_id == "uab_workload_dataset":
        objective_groups = {
            "n_back": _select_indices(entries, subset_id="n_back", training_role="primary"),
            "heat_the_chair": _select_indices(
                entries,
                subset_id="heat_the_chair",
                training_role="primary",
            ),
        }
        subjective_groups = objective_groups
        objective_summary, objective_predictions = _run_public_track(
            dataset=dataset,
            config=config,
            track="objective",
            evaluation_groups=objective_groups,
            ordered_modalities=ordered_modalities,
            label_order_by_group=None,
        )
        subjective_summary, subjective_predictions = _run_public_track(
            dataset=dataset,
            config=config,
            track="subjective",
            evaluation_groups=subjective_groups,
            ordered_modalities=ordered_modalities,
            label_order_by_group=None,
        )
        predictions = pd.concat(
            (objective_predictions, subjective_predictions),
            axis=0,
            ignore_index=True,
        )
    elif config.dataset_id == "nasa_csm":
        objective_groups = {
            "benchmark_only": _select_indices(
                entries,
                subset_id="benchmark",
                training_role="primary",
            ),
            "loft_only": _select_indices(entries, subset_id="loft", training_role="primary"),
            "combined": _select_indices(
                entries,
                subset_ids=("benchmark", "loft"),
                training_role="primary",
            ),
        }
        label_order_by_group = {
            group_name: (1, 2, 5)
            for group_name in objective_groups
        }
        objective_summary, predictions = _run_public_track(
            dataset=dataset,
            config=config,
            track="objective",
            evaluation_groups=objective_groups,
            ordered_modalities=ordered_modalities,
            label_order_by_group=label_order_by_group,
        )
        subjective_summary = None
    else:
        raise ValueError(f"unsupported public deep dataset: {config.dataset_id}")

    reference_comparison = _load_reference_comparison(
        dataset_id=config.dataset_id,
        reference_artifact_root=config.reference_artifact_root,
    )
    summary = {
        "dataset_id": config.dataset_id,
        "profile": config.profile,
        "model_name": config.model_name,
        "model_config": _deep_model_config_dict(config),
        "runtime_device": resolve_torch_device_name(config.device),
        "artifact_root": str(Path(config.artifact_root)),
        "prepared_artifact_root": config.prepared_artifact_root,
        "objective": objective_summary,
        "subjective": subjective_summary,
        "reference_comparison": reference_comparison,
    }
    return summary, predictions


def _run_public_track(
    *,
    dataset: Mapping[str, object],
    config: StageIDeepBaselineConfig,
    track: str,
    evaluation_groups: Mapping[str, np.ndarray],
    ordered_modalities: Sequence[str],
    label_order_by_group: Mapping[str, Sequence[int]] | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    artifact_root = Path(config.artifact_root)
    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    bundle = dataset["bundle"]
    entries = dataset["entries"]
    predictions_frames: list[pd.DataFrame] = []
    group_metrics: dict[str, object] = {}
    plot_paths: dict[str, str] = {}
    training_curves: list[dict[str, object]] = []
    for group_name, indices in evaluation_groups.items():
        if len(indices) == 0:
            continue
        group_entries = [entries[index] for index in indices]
        split_groups = np.asarray(
            [entry.split_group for entry in group_entries],
            dtype=object,
        )
        loso_splits = build_loso_splits(split_groups)
        if config.max_folds is not None:
            loso_splits = loso_splits[: config.max_folds]
        target_values = _extract_target_values(group_entries, track=track)
        if track == "objective":
            label_order = tuple(
                label_order_by_group[group_name]
                if label_order_by_group is not None and group_name in label_order_by_group
                else sorted(set(int(value) for value in target_values))
            )
            predictions = _fit_predict_classification(
                bundle=bundle,
                entries=group_entries,
                indices=indices,
                ordered_modalities=ordered_modalities,
                labels=np.asarray(target_values, dtype=int),
                label_order=label_order,
                config=config,
                loso_splits=loso_splits,
                evaluation_group=group_name,
            )
            metrics = evaluate_classification_predictions(
                predictions,
                label_order=label_order,
            )
            plot_key = f"{track}_{group_name}_confusion_matrix"
            plot_paths[plot_key] = save_confusion_matrix_plot(
                metrics,
                path=plot_root / f"{plot_key}.png",
                title=f"{config.model_name} {group_name}",
            )
        else:
            predictions = _fit_predict_regression(
                bundle=bundle,
                entries=group_entries,
                indices=indices,
                ordered_modalities=ordered_modalities,
                targets=np.asarray(target_values, dtype=np.float32),
                config=config,
                loso_splits=loso_splits,
                evaluation_group=group_name,
            )
            metrics = evaluate_regression_predictions(predictions)
            plot_key = f"{track}_{group_name}_regression"
            plot_paths[plot_key] = save_regression_plot(
                predictions,
                path=plot_root / f"{plot_key}.png",
                title=f"{config.model_name} {group_name}",
            )
        training_curves.extend(
            [
                dict(row, dataset_id=config.dataset_id, model_name=config.model_name)
                for row in predictions.attrs.get("training_curves", [])
            ]
        )
        predictions_frames.append(predictions)
        group_metrics[group_name] = metrics
    if track == "objective":
        comparison_values = {
            group_name: {
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
            for group_name, metrics in group_metrics.items()
        }
        if comparison_values:
            plot_paths["objective_primary_metrics"] = save_grouped_bar_plot(
                comparison_values,
                path=plot_root / "objective_primary_metrics.png",
                title=f"{config.model_name} objective metrics",
                ylabel="score",
            )
    else:
        comparison_values = {
            group_name: {
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
            }
            for group_name, metrics in group_metrics.items()
        }
        if comparison_values:
            plot_paths["subjective_primary_metrics"] = save_grouped_bar_plot(
                comparison_values,
                path=plot_root / "subjective_primary_metrics.png",
                title=f"{config.model_name} subjective metrics",
                ylabel="value",
            )
    predictions_frame = (
        pd.concat(predictions_frames, axis=0, ignore_index=True)
        if predictions_frames
        else pd.DataFrame()
    )
    return {
        "track": track,
        "model_name": config.model_name,
        "groups": group_metrics,
        "plot_paths": plot_paths,
        "training_curves": training_curves,
    }, predictions_frame


def _run_real_sortie_case_study(
    *,
    dataset: Mapping[str, object],
    config: StageIDeepBaselineConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    return _run_real_sortie_case_study_impl(dataset=dataset, config=config)


def _extract_training_curves(summary: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for track in ("objective", "subjective"):
        payload = summary.get(track)
        if not isinstance(payload, Mapping):
            continue
        for row in payload.get("training_curves", []):
            if isinstance(row, Mapping):
                rows.append(dict(row))
    for row in summary.get("training_curves", []):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows
