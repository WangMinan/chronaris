"""Deep baseline pipelines for Stage I sequence experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from chronaris.dataset import (
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    load_stage_i_sequence_summary,
)
from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
    save_bar_plot,
    save_confusion_matrix_plot,
    save_grouped_bar_plot,
    save_regression_plot,
)
from chronaris.features.stage_i_sequences import STAGE_H_CASE_DATASET_ID
from chronaris.pipelines.stage_i_baseline_models import build_loso_splits
from chronaris.pipelines.stage_i_deep_models import build_stage_i_deep_model
from chronaris.pipelines.stage_i_phase3_assets import (
    extract_primary_metrics,
    load_stage_i_baseline_artifacts,
)

DATASET_RUN_ORDER = (STAGE_H_CASE_DATASET_ID, "uab_workload_dataset", "nasa_csm")


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
    max_folds: int | None = None
    seed: int = 42


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
    max_folds: int | None = None
    seed: int = 42


@dataclass(frozen=True, slots=True)
class StageIDeepComparisonRunResult:
    artifact_root: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_deep_baseline(
    config: StageIDeepBaselineConfig,
) -> StageIDeepBaselineRunResult:
    artifact_root = Path(config.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = _load_prepared_sequence_dataset(config.prepared_artifact_root)
    if dataset["dataset_id"] != config.dataset_id:
        raise ValueError(
            f"prepared dataset mismatch: expected {config.dataset_id}, got {dataset['dataset_id']}"
        )
    if config.dataset_id == STAGE_H_CASE_DATASET_ID:
        summary, predictions = _run_real_sortie_case_study(dataset=dataset, config=config)
    else:
        summary, predictions = _run_public_deep_baseline(dataset=dataset, config=config)
    summary_path = artifact_root / "deep_baseline_summary.json"
    report_path = artifact_root / "deep_baseline_report.md"
    predictions_path = artifact_root / "fold_predictions.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_deep_baseline_report(summary) + "\n", encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
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
                    max_folds=config.max_folds,
                    seed=config.seed,
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
    }, predictions_frame


def _run_real_sortie_case_study(
    *,
    dataset: Mapping[str, object],
    config: StageIDeepBaselineConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    bundle = dataset["bundle"]
    entries = dataset["entries"]
    artifact_root = Path(config.artifact_root)
    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    ordered_modalities = tuple(entries[0].modality_schema)
    normalized_arrays = _normalize_modalities(
        modality_arrays=bundle.modality_arrays,
        modality_masks=bundle.modality_masks,
        ordered_modalities=ordered_modalities,
        train_indices=np.arange(bundle.entry_count, dtype=int),
    )
    labels = bundle.objective_label_values.astype(int)
    model = _train_model(
        model_name=config.model_name,
        ordered_modalities=ordered_modalities,
        modality_arrays=normalized_arrays,
        modality_masks=bundle.modality_masks,
        time_axis=bundle.time_axis,
        train_indices=np.arange(bundle.entry_count, dtype=int),
        train_targets=labels,
        output_dim=max(int(labels.max()) + 1, 2),
        task="classification",
        config=config,
    )
    base_output = _forward_dataset(
        model=model,
        ordered_modalities=ordered_modalities,
        modality_arrays=normalized_arrays,
        modality_masks=bundle.modality_masks,
        time_axis=bundle.time_axis,
        indices=np.arange(bundle.entry_count, dtype=int),
    )
    perturbed_vehicle_values, perturbed_vehicle_masks = _mask_top_event_steps(bundle)
    perturbed_arrays = dict(normalized_arrays)
    perturbed_arrays["vehicle"] = perturbed_vehicle_values
    perturbed_masks = dict(bundle.modality_masks)
    perturbed_masks["vehicle"] = perturbed_vehicle_masks
    perturbed_output = _forward_dataset(
        model=model,
        ordered_modalities=ordered_modalities,
        modality_arrays=perturbed_arrays,
        modality_masks=perturbed_masks,
        time_axis=bundle.time_axis,
        indices=np.arange(bundle.entry_count, dtype=int),
    )
    sample_frame = _build_real_sortie_sample_frame(
        entries=entries,
        bundle=bundle,
        base_output=base_output,
        perturbed_output=perturbed_output,
    )
    view_summary = _build_real_sortie_view_summary(sample_frame)
    pilot_summary = _build_real_sortie_pilot_summary(view_summary)
    plot_paths = {
        "view_representation_stability": save_bar_plot(
            dict(
                zip(
                    view_summary["view_id"],
                    view_summary["representation_stability"],
                    strict=True,
                ),
            ),
            path=plot_root / "view_representation_stability.png",
            title=f"{config.model_name} real-sortie stability",
            ylabel="cosine",
        ),
        "event_mask_interference": save_bar_plot(
            dict(
                zip(
                    view_summary["view_id"],
                    view_summary["event_mask_interference"],
                    strict=True,
                ),
            ),
            path=plot_root / "event_mask_interference.png",
            title=f"{config.model_name} event-mask interference",
            ylabel="1-cosine",
        ),
    }
    summary = {
        "dataset_id": STAGE_H_CASE_DATASET_ID,
        "profile": config.profile,
        "model_name": config.model_name,
        "artifact_root": str(artifact_root),
        "prepared_artifact_root": config.prepared_artifact_root,
        "smoke_training_target": "projection_diagnostics_verdict_code",
        "view_count": int(view_summary.shape[0]),
        "sample_count": int(sample_frame.shape[0]),
        "view_summary_csv": str(artifact_root / "view_summary.csv"),
        "sample_summary_csv": str(artifact_root / "sample_summary.csv"),
        "pilot_summary_csv": str(artifact_root / "pilot_summary.csv"),
        "plot_paths": plot_paths,
        "view_metrics": view_summary.to_dict(orient="records"),
        "pilot_metrics": pilot_summary.to_dict(orient="records"),
        "verdict_counts": (
            sample_frame["projection_diagnostics_verdict"].value_counts().to_dict()
        ),
    }
    sample_frame.to_csv(artifact_root / "sample_summary.csv", index=False)
    view_summary.to_csv(artifact_root / "view_summary.csv", index=False)
    pilot_summary.to_csv(artifact_root / "pilot_summary.csv", index=False)
    return summary, sample_frame


def _fit_predict_classification(
    *,
    bundle,
    entries: Sequence[object],
    indices: np.ndarray,
    ordered_modalities: Sequence[str],
    labels: np.ndarray,
    label_order: Sequence[int],
    config: StageIDeepBaselineConfig,
    loso_splits,
    evaluation_group: str,
) -> pd.DataFrame:
    label_to_index = {int(label): position for position, label in enumerate(label_order)}
    frames: list[pd.DataFrame] = []
    for split in loso_splits:
        model = _train_model(
            model_name=config.model_name,
            ordered_modalities=ordered_modalities,
            modality_arrays=bundle.modality_arrays,
            modality_masks=bundle.modality_masks,
            time_axis=bundle.time_axis,
            train_indices=indices[split.train_indices],
            train_targets=np.asarray(
                [label_to_index[int(value)] for value in labels[split.train_indices]],
                dtype=int,
            ),
            output_dim=len(label_order),
            task="classification",
            config=config,
        )
        output = _forward_dataset(
            model=model,
            ordered_modalities=ordered_modalities,
            modality_arrays=_normalize_modalities(
                modality_arrays=bundle.modality_arrays,
                modality_masks=bundle.modality_masks,
                ordered_modalities=ordered_modalities,
                train_indices=indices[split.train_indices],
            ),
            modality_masks=bundle.modality_masks,
            time_axis=bundle.time_axis,
            indices=indices[split.test_indices],
        )
        logits = _sanitize_classification_logits(
            output.logits.detach().cpu().numpy(),
        )
        predicted_indices = logits.argmax(axis=1)
        predicted_labels = np.asarray(
            [label_order[index] for index in predicted_indices],
            dtype=int,
        )
        frames.append(
            _build_prediction_frame(
                entries=[entries[index] for index in split.test_indices],
                evaluation_group=evaluation_group,
                track="objective",
                model_name=config.model_name,
                y_true=labels[split.test_indices],
                y_pred=predicted_labels,
            ),
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _fit_predict_regression(
    *,
    bundle,
    entries: Sequence[object],
    indices: np.ndarray,
    ordered_modalities: Sequence[str],
    targets: np.ndarray,
    config: StageIDeepBaselineConfig,
    loso_splits,
    evaluation_group: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in loso_splits:
        model = _train_model(
            model_name=config.model_name,
            ordered_modalities=ordered_modalities,
            modality_arrays=bundle.modality_arrays,
            modality_masks=bundle.modality_masks,
            time_axis=bundle.time_axis,
            train_indices=indices[split.train_indices],
            train_targets=targets[split.train_indices],
            output_dim=1,
            task="regression",
            config=config,
        )
        output = _forward_dataset(
            model=model,
            ordered_modalities=ordered_modalities,
            modality_arrays=_normalize_modalities(
                modality_arrays=bundle.modality_arrays,
                modality_masks=bundle.modality_masks,
                ordered_modalities=ordered_modalities,
                train_indices=indices[split.train_indices],
            ),
            modality_masks=bundle.modality_masks,
            time_axis=bundle.time_axis,
            indices=indices[split.test_indices],
        )
        prediction_values, nonfinite_mask = _sanitize_regression_outputs(
            output.logits.detach().cpu().numpy().reshape(-1),
            fallback_value=_safe_regression_fallback(targets[split.train_indices]),
        )
        frames.append(
            _build_prediction_frame(
                entries=[entries[index] for index in split.test_indices],
                evaluation_group=evaluation_group,
                track="subjective",
                model_name=config.model_name,
                y_true=targets[split.test_indices],
                y_pred=prediction_values,
                extra_columns={
                    "prediction_was_nonfinite": nonfinite_mask.astype(int),
                },
            ),
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _train_model(
    *,
    model_name: str,
    ordered_modalities: Sequence[str],
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    train_indices: np.ndarray,
    train_targets: np.ndarray,
    output_dim: int,
    task: str,
    config: StageIDeepBaselineConfig,
) -> nn.Module:
    normalized_arrays = _normalize_modalities(
        modality_arrays=modality_arrays,
        modality_masks=modality_masks,
        ordered_modalities=ordered_modalities,
        train_indices=train_indices,
    )
    torch.manual_seed(config.seed)
    model = build_stage_i_deep_model(
        model_name=model_name,
        ordered_modalities=ordered_modalities,
        modality_input_dims={
            name: normalized_arrays[name].shape[-1]
            for name in ordered_modalities
        },
        output_dim=output_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        layers=config.layers,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = (
        nn.CrossEntropyLoss()
        if task == "classification"
        else nn.MSELoss()
    )
    if len(train_indices) == 0:
        return model
    for epoch in range(config.epochs):
        for batch_indices in _iter_batches(
            len(train_indices),
            batch_size=config.batch_size,
            seed=config.seed + epoch,
        ):
            global_batch = train_indices[batch_indices]
            batch_targets = train_targets[batch_indices]
            optimizer.zero_grad()
            output = _forward_dataset(
                model=model,
                ordered_modalities=ordered_modalities,
                modality_arrays=normalized_arrays,
                modality_masks=modality_masks,
                time_axis=time_axis,
                indices=global_batch,
                training=True,
            )
            logits = output.logits
            if logits is None:
                raise ValueError("deep model returned no logits during supervised training.")
            if task == "classification":
                target_tensor = torch.as_tensor(batch_targets, dtype=torch.long)
                loss = criterion(logits, target_tensor)
            else:
                target_tensor = torch.as_tensor(batch_targets, dtype=torch.float32).view(-1, 1)
                loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
    return model


def _forward_dataset(
    *,
    model: nn.Module,
    ordered_modalities: Sequence[str],
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    indices: np.ndarray,
    training: bool = False,
):
    modality_tensor_map = {
        name: torch.as_tensor(modality_arrays[name][indices], dtype=torch.float32)
        for name in ordered_modalities
    }
    mask_tensor_map = {
        name: torch.as_tensor(modality_masks[name][indices], dtype=torch.float32)
        for name in ordered_modalities
    }
    time_tensor = torch.as_tensor(time_axis[indices], dtype=torch.float32)
    if training:
        model.train()
        return model(
            modality_tensor_map,
            time_axis=time_tensor,
            modality_masks=mask_tensor_map,
        )
    model.eval()
    with torch.no_grad():
        return model(
            modality_tensor_map,
            time_axis=time_tensor,
            modality_masks=mask_tensor_map,
        )


def _normalize_modalities(
    *,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    ordered_modalities: Sequence[str],
    train_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for modality_name in ordered_modalities:
        values = modality_arrays[modality_name].astype(np.float32, copy=True)
        mask = modality_masks[modality_name].astype(bool, copy=False)
        train_values = values[train_indices]
        train_mask = mask[train_indices]
        valid = train_mask[:, :, None] & np.isfinite(train_values)
        count = valid.sum(axis=(0, 1)).astype(np.float32)
        safe_count = np.maximum(count, 1.0)
        mean = np.where(
            count > 0,
            np.where(valid, train_values, 0.0).sum(axis=(0, 1)) / safe_count,
            0.0,
        )
        centered = np.where(valid, train_values - mean.reshape(1, 1, -1), 0.0)
        variance = np.where(
            count > 0,
            np.square(centered).sum(axis=(0, 1)) / safe_count,
            1.0,
        )
        std = np.sqrt(np.maximum(variance, 1e-6))
        transformed = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        normalized[modality_name] = np.where(mask[:, :, None], transformed, 0.0).astype(
            np.float32,
        )
    return normalized


def _build_real_sortie_sample_frame(
    *,
    entries: Sequence[object],
    bundle,
    base_output,
    perturbed_output,
) -> pd.DataFrame:
    attention_entropy = _attention_entropy(base_output.attention_map)
    top_concentration = base_output.attention_map.max(dim=-1).values.mean(dim=1).cpu().numpy()
    cosine_shift = 1.0 - _cosine_similarity(
        base_output.pooled_embedding,
        perturbed_output.pooled_embedding,
    ).cpu().numpy()
    rows: list[dict[str, object]] = []
    metadata_by_sample = {
        entry.sample_id: json.loads(metadata)
        for entry, metadata in zip(entries, bundle.metadata_json, strict=True)
    }
    for index, entry in enumerate(entries):
        metadata = metadata_by_sample[entry.sample_id]
        rows.append(
            {
                "sample_id": entry.sample_id,
                "view_id": metadata["view_id"],
                "sortie_id": metadata["sortie_id"],
                "pilot_id": metadata["pilot_id"],
                "projection_diagnostics_verdict": metadata[
                    "projection_diagnostics_verdict"
                ],
                "embedding_norm": float(
                    torch.linalg.norm(base_output.pooled_embedding[index]).item(),
                ),
                "attention_entropy": float(attention_entropy[index]),
                "top_event_concentration": float(top_concentration[index]),
                "event_mask_interference": float(cosine_shift[index]),
            },
        )
    return pd.DataFrame(rows)


def _build_real_sortie_view_summary(sample_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for view_id, frame in sample_frame.groupby("view_id", sort=False):
        norms = frame["embedding_norm"].to_numpy(dtype=float)
        stability = float(np.mean(norms / max(np.max(norms), 1e-6)))
        records.append(
            {
                "view_id": view_id,
                "sortie_id": frame["sortie_id"].iloc[0],
                "pilot_id": int(frame["pilot_id"].iloc[0]),
                "projection_diagnostics_verdict": frame[
                    "projection_diagnostics_verdict"
                ].iloc[0],
                "sample_count": int(len(frame)),
                "representation_stability": stability,
                "mean_attention_entropy": float(frame["attention_entropy"].mean()),
                "top_event_concentration": float(
                    frame["top_event_concentration"].mean(),
                ),
                "event_mask_interference": float(
                    frame["event_mask_interference"].mean(),
                ),
            },
        )
    return pd.DataFrame(records)


def _build_real_sortie_pilot_summary(view_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sortie_id, frame in view_summary.groupby("sortie_id", sort=False):
        if len(frame) < 2:
            continue
        ordered = frame.sort_values("pilot_id").reset_index(drop=True)
        reference = ordered.iloc[0]
        comparison = ordered.iloc[1]
        rows.append(
            {
                "sortie_id": sortie_id,
                "reference_pilot_id": int(reference["pilot_id"]),
                "comparison_pilot_id": int(comparison["pilot_id"]),
                "delta_representation_stability": float(
                    comparison["representation_stability"]
                    - reference["representation_stability"]
                ),
                "delta_attention_entropy": float(
                    comparison["mean_attention_entropy"]
                    - reference["mean_attention_entropy"]
                ),
                "delta_top_event_concentration": float(
                    comparison["top_event_concentration"]
                    - reference["top_event_concentration"]
                ),
                "delta_event_mask_interference": float(
                    comparison["event_mask_interference"]
                    - reference["event_mask_interference"]
                ),
            },
        )
    return pd.DataFrame(rows)


def _mask_top_event_steps(bundle) -> tuple[np.ndarray, np.ndarray]:
    vehicle_values = bundle.modality_arrays["vehicle"].copy()
    vehicle_masks = bundle.modality_masks["vehicle"].copy()
    event_scores = bundle.extras["vehicle_event_scores"].astype(np.float32)
    for sample_index in range(event_scores.shape[0]):
        threshold = float(np.quantile(event_scores[sample_index], 0.75))
        active = event_scores[sample_index] >= threshold
        vehicle_values[sample_index, active, :] = 0.0
        vehicle_masks[sample_index, active] = 0
    return vehicle_values, vehicle_masks


def _load_prepared_sequence_dataset(artifact_root: str | Path) -> dict[str, object]:
    root = Path(artifact_root)
    entries = load_stage_i_sequence_entries(root / "task_manifest.jsonl")
    bundle = load_stage_i_sequence_bundle(root / "sequence_bundle.npz")
    summary = load_stage_i_sequence_summary(root / "dataset_summary.json")
    schema = json.loads((root / "sequence_schema.json").read_text(encoding="utf-8"))
    return {
        "artifact_root": str(root),
        "dataset_id": summary.dataset_id,
        "entries": entries,
        "bundle": bundle,
        "summary": summary.to_dict(),
        "schema": schema,
    }


def _load_reference_comparison(
    *,
    dataset_id: str,
    reference_artifact_root: str | None,
) -> dict[str, object] | None:
    if not reference_artifact_root:
        return None
    root = Path(reference_artifact_root)
    if not root.exists():
        return None
    artifacts = load_stage_i_baseline_artifacts(
        root,
        require_subjective=(dataset_id == "uab_workload_dataset"),
    )
    return {
        "objective": extract_primary_metrics(artifacts.objective_metrics),
        "subjective": extract_primary_metrics(artifacts.subjective_metrics),
    }


def _render_deep_baseline_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Deep Baseline - {summary['dataset_id']} - {summary['model_name']}",
        "",
        f"- profile: `{summary['profile']}`",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- prepared root: `{summary['prepared_artifact_root']}`",
        "",
    ]
    if summary["dataset_id"] == STAGE_H_CASE_DATASET_ID:
        lines.extend(
            [
                "## Real Sortie Summary",
                "",
                f"- view count: `{summary['view_count']}`",
                f"- sample count: `{summary['sample_count']}`",
                f"- smoke training target: `{summary['smoke_training_target']}`",
                "",
                "| view | sortie | pilot | verdict | samples | stability | attention entropy | top concentration | event-mask interference |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ],
        )
        for row in summary["view_metrics"]:
            lines.append(
                f"| `{row['view_id']}` | `{row['sortie_id']}` | {row['pilot_id']} | "
                f"`{row['projection_diagnostics_verdict']}` | {row['sample_count']} | "
                f"{row['representation_stability']:.6f} | {row['mean_attention_entropy']:.6f} | "
                f"{row['top_event_concentration']:.6f} | {row['event_mask_interference']:.6f} |"
            )
        if summary["pilot_metrics"]:
            lines.extend(
                [
                    "",
                    "## Same-Sortie Dual-Pilot Delta",
                    "",
                    "| sortie | reference pilot | comparison pilot | delta stability | delta entropy | delta top concentration | delta event-mask interference |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ],
            )
            for row in summary["pilot_metrics"]:
                lines.append(
                    f"| `{row['sortie_id']}` | {row['reference_pilot_id']} | {row['comparison_pilot_id']} | "
                    f"{row['delta_representation_stability']:+.6f} | {row['delta_attention_entropy']:+.6f} | "
                    f"{row['delta_top_event_concentration']:+.6f} | {row['delta_event_mask_interference']:+.6f} |"
                )
    else:
        for track_name in ("objective", "subjective"):
            track = summary.get(track_name)
            if not track:
                continue
            lines.extend(
                [
                    f"## {track_name.title()}",
                    "",
                ],
            )
            if track_name == "objective":
                lines.extend(
                    [
                        "| group | macro-F1 | balanced accuracy | samples | folds |",
                        "| --- | ---: | ---: | ---: | ---: |",
                    ],
                )
                for group_name, metrics in track["groups"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['macro_f1']:.6f} | "
                        f"{metrics['balanced_accuracy']:.6f} | {metrics['sample_count']} | "
                        f"{metrics['fold_count']} |"
                    )
            else:
                lines.extend(
                    [
                        "| group | RMSE | MAE | R2 | Spearman | samples | folds |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ],
                )
                for group_name, metrics in track["groups"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['rmse']:.6f} | {metrics['mae']:.6f} | "
                        f"{metrics['r2']:.6f} | {metrics['spearman']:.6f} | "
                        f"{metrics['sample_count']} | {metrics['fold_count']} |"
                    )
            lines.append("")
        if summary.get("reference_comparison"):
            lines.extend(["## Existing Classical Reference", ""])
            reference = summary["reference_comparison"]
            if reference.get("objective"):
                lines.extend(
                    [
                        "| group | classical macro-F1 | classical balanced acc |",
                        "| --- | ---: | ---: |",
                    ],
                )
                for group_name, metrics in reference["objective"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['macro_f1']:.6f} | "
                        f"{metrics['balanced_accuracy']:.6f} |"
                    )
                lines.append("")
            if reference.get("subjective"):
                lines.extend(
                    [
                        "| group | classical RMSE | classical MAE |",
                        "| --- | ---: | ---: |",
                    ],
                )
                for group_name, metrics in reference["subjective"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['rmse']:.6f} | {metrics['mae']:.6f} |"
                    )
    return "\n".join(lines)


def _render_comparison_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Stage I Deep Comparison",
        "",
        f"- generated at UTC: `{summary['generated_at_utc']}`",
        f"- artifact root: `{summary['artifact_root']}`",
        "",
    ]
    for dataset_id in summary["dataset_order"]:
        dataset_payload = summary["datasets"].get(dataset_id, {"status": "not_run"})
        lines.append(f"## {dataset_id}")
        lines.append("")
        if dataset_payload.get("status") != "completed":
            lines.append("- 本轮未运行。")
            lines.append("")
            continue
        lines.extend(
            [
                "| model | artifact root | summary path | report path |",
                "| --- | --- | --- | --- |",
            ],
        )
        for model_name, payload in dataset_payload["models"].items():
            lines.append(
                f"| `{model_name}` | `{payload['artifact_root']}` | "
                f"`{payload['summary_path']}` | `{payload['report_path']}` |"
            )
        lines.append("")
    return "\n".join(lines)


def _build_prediction_frame(
    *,
    entries: Sequence[object],
    evaluation_group: str,
    track: str,
    model_name: str,
    y_true: Sequence[float | int],
    y_pred: Sequence[float | int],
    extra_columns: Mapping[str, Sequence[object]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    normalized_extra = {
        key: list(values)
        for key, values in (extra_columns or {}).items()
    }
    for row_index, (entry, truth, prediction) in enumerate(
        zip(entries, y_true, y_pred, strict=True),
    ):
        row = {
            "sample_id": entry.sample_id,
            "dataset_id": entry.dataset_id,
            "subset_id": entry.subset_id,
            "subject_id": entry.subject_id,
            "split_group": entry.split_group,
            "evaluation_group": evaluation_group,
            "track": track,
            "model_name": model_name,
            "y_true": truth,
            "y_pred": prediction,
        }
        for key, values in normalized_extra.items():
            row[key] = values[row_index]
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_target_values(entries: Sequence[object], *, track: str) -> list[float]:
    if track == "objective":
        return [float(entry.objective_label_value) for entry in entries]
    return [float(entry.subjective_target_value) for entry in entries]


def _select_indices(
    entries: Sequence[object],
    *,
    subset_id: str | None = None,
    subset_ids: Sequence[str] | None = None,
    training_role: str | None = None,
) -> np.ndarray:
    active_subset_ids = set(subset_ids or ([] if subset_id is None else [subset_id]))
    return np.asarray(
        [
            index
            for index, entry in enumerate(entries)
            if (not active_subset_ids or entry.subset_id in active_subset_ids)
            and (training_role is None or entry.training_role == training_role)
        ],
        dtype=int,
    )


def _iter_batches(
    length: int,
    *,
    batch_size: int,
    seed: int,
) -> Sequence[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(length)
    return tuple(
        order[start : start + batch_size]
        for start in range(0, length, max(batch_size, 1))
    )


def _infer_profile(dataset_id: str) -> str:
    return "real_sortie_v1" if dataset_id == STAGE_H_CASE_DATASET_ID else "window_v2"


def _sanitize_classification_logits(logits: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        logits,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _sanitize_regression_outputs(
    values: np.ndarray,
    *,
    fallback_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    sanitized = np.asarray(values, dtype=np.float32).copy()
    nonfinite_mask = ~np.isfinite(sanitized)
    if np.any(nonfinite_mask):
        sanitized[nonfinite_mask] = float(fallback_value)
    return sanitized, nonfinite_mask


def _safe_regression_fallback(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float32)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def _attention_entropy(attention_map: torch.Tensor) -> np.ndarray:
    probabilities = attention_map.clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=-1)
    return entropy.detach().cpu().numpy()


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    right_norm = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (left_norm * right_norm).sum(dim=-1)
