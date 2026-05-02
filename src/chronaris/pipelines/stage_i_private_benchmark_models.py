"""Modeling and reporting helpers for the private benchmark."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from chronaris.dataset import StageISequenceEntry
from chronaris.dataset.stage_i_private_contracts import StageIPrivateTaskEntry
from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
    save_grouped_bar_plot,
)
from chronaris.pipelines.stage_i_baseline_models import (
    build_fold_cache,
    build_loso_splits,
    objective_model_specs,
    subjective_model_specs,
)
from chronaris.pipelines.stage_i_deep_baseline import (
    _build_prediction_frame,
    _safe_regression_fallback,
    _sanitize_classification_logits,
    _sanitize_regression_outputs,
)
from chronaris.pipelines.stage_i_deep_models import build_stage_i_deep_model
from chronaris.pipelines.stage_i_private_benchmark_data import (
    CLASS_LABEL_TO_ID,
    PRIVATE_DATASET_ID,
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    build_private_sequence_frame,
    merge_task_features,
)
from chronaris.pipelines.stage_i_private_optimization import (
    is_optimized_variant_name,
    optimized_no_mask_variant_name,
    run_optimized_retrieval_variant,
    run_optimized_supervised_variant,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chronaris.pipelines.stage_i_private_benchmark import StageIPrivateBenchmarkConfig


def run_supervised_task(
    *,
    task_name: str,
    task_type: str,
    task_entries: Sequence[StageIPrivateTaskEntry],
    variant_feature_frames: Mapping[str, pd.DataFrame],
    variant_order: Sequence[str],
    target_variant_name: str,
    deep_model_names: Sequence[str],
    records: pd.DataFrame,
    config: StageIPrivateBenchmarkConfig,
) -> dict[str, object]:
    variants: dict[str, object] = {}
    for variant_name in variant_order:
        frame = merge_task_features(task_entries, variant_feature_frames[variant_name], task_type=task_type)
        if is_optimized_variant_name(variant_name, target_variant_name=target_variant_name):
            variants[variant_name] = run_optimized_supervised_variant(
                frame,
                task_type=task_type,
                variant_name=variant_name,
            )
        else:
            variants[variant_name] = run_classical_variant(frame, task_type=task_type)

    deep = run_deep_baselines(
        task_name=task_name,
        task_type=task_type,
        task_entries=task_entries,
        records=records,
        model_names=deep_model_names,
        config=config,
    )
    best_variant_name, best_variant_metrics = select_best_variant(variants, task_type=task_type)
    best_deep_name, best_deep_metrics = select_best_variant(deep, task_type=task_type)
    return {
        "task_type": task_type,
        "variants": variants,
        "deep_models": deep,
        "best_variant": {
            "name": best_variant_name,
            "metrics": best_variant_metrics,
        },
        "best_deep_model": {
            "name": best_deep_name,
            "metrics": best_deep_metrics,
        },
    }


def run_classical_variant(frame: pd.DataFrame, *, task_type: str) -> dict[str, object]:
    if frame.empty:
        return {"status": "not_run"}

    feature_columns = [column for column in frame.columns if column.startswith("feat__")]
    if not feature_columns:
        return {"status": "not_run"}
    feature_matrix = frame[feature_columns].to_numpy(dtype=float)
    split_groups = frame["split_group"].to_numpy(dtype=object)
    loso_splits = build_loso_splits(split_groups)
    model_specs = objective_model_specs("window_v2") if task_type == "classification" else subjective_model_specs("window_v2")
    cached_folds = build_fold_cache(
        feature_matrix=feature_matrix,
        loso_splits=loso_splits,
        preprocess_modes={spec.preprocessing for spec in model_specs.values()},
    )
    metrics_by_model: dict[str, object] = {}
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for model_name, model_spec in model_specs.items():
        frames: list[pd.DataFrame] = []
        for split, prepared in cached_folds:
            estimator = clone(model_spec.estimator)
            train_x, test_x = prepared[model_spec.preprocessing]
            train_y = frame.iloc[split.train_indices]["y_label"].to_numpy()
            test_y = frame.iloc[split.test_indices]["y_label"].to_numpy()
            estimator.fit(train_x, train_y)
            predicted = estimator.predict(test_x)
            frames.append(
                _build_prediction_frame(
                    entries=[
                        task_entry_to_sequence_entry(task_entry=frame.iloc[index]["task_entry"])
                        for index in split.test_indices
                    ],
                    evaluation_group="leave_one_view_out",
                    track="objective" if task_type == "classification" else "subjective",
                    model_name=model_name,
                    y_true=test_y,
                    y_pred=predicted,
                )
            )
        predictions = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        predictions_by_model[model_name] = predictions
        metrics_by_model[model_name] = (
            evaluate_classification_predictions(predictions, label_order=(0, 1, 2))
            if task_type == "classification"
            else evaluate_regression_predictions(predictions)
        )
    best_model_name, best_metrics = select_best_variant(metrics_by_model, task_type=task_type)
    return {
        "status": "completed",
        "sample_count": int(len(frame)),
        "model_metrics": metrics_by_model,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "predictions": {
            model_name: predictions.to_dict(orient="records")
            for model_name, predictions in predictions_by_model.items()
        },
    }


def run_deep_baselines(
    *,
    task_name: str,
    task_type: str,
    task_entries: Sequence[StageIPrivateTaskEntry],
    records: pd.DataFrame,
    model_names: Sequence[str],
    config: StageIPrivateBenchmarkConfig,
) -> dict[str, object]:
    frame = build_private_sequence_frame(task_entries, records, task_type=task_type)
    if frame.empty:
        return {model_name: {"status": "not_run"} for model_name in model_names}

    modality_arrays = {
        "physiology": np.stack(frame["physiology_sequence"].to_list(), axis=0).astype(np.float32),
        "vehicle": np.stack(frame["vehicle_sequence"].to_list(), axis=0).astype(np.float32),
    }
    modality_masks = {
        "physiology": np.stack(frame["physiology_mask"].to_list(), axis=0).astype(np.uint8),
        "vehicle": np.stack(frame["vehicle_mask"].to_list(), axis=0).astype(np.uint8),
    }
    time_axis = np.stack(frame["time_axis"].to_list(), axis=0).astype(np.float32)
    split_groups = frame["split_group"].to_numpy(dtype=object)
    loso_splits = build_loso_splits(split_groups)
    if config.max_deep_folds is not None:
        loso_splits = loso_splits[: config.max_deep_folds]

    entries = [
        task_entry_to_sequence_entry(task_entry=item)
        for item in frame["task_entry"].tolist()
    ]
    results: dict[str, object] = {}
    for model_name in model_names:
        predictions = run_one_deep_model(
            entries=entries,
            modality_arrays=modality_arrays,
            modality_masks=modality_masks,
            time_axis=time_axis,
            labels=frame["y_label"].to_numpy(),
            task_type=task_type,
            model_name=model_name,
            config=config,
            loso_splits=loso_splits,
        )
        metrics = (
            evaluate_classification_predictions(predictions, label_order=(0, 1, 2))
            if task_type == "classification"
            else evaluate_regression_predictions(predictions)
        )
        results[model_name] = {
            "status": "completed",
            "sample_count": int(len(frame)),
            "metrics": metrics,
            "predictions": predictions.to_dict(orient="records"),
        }
    return results


def run_one_deep_model(
    *,
    entries: Sequence[StageISequenceEntry],
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    model_name: str,
    config: StageIPrivateBenchmarkConfig,
    loso_splits,
) -> pd.DataFrame:
    ordered_modalities = ("physiology", "vehicle")
    frames: list[pd.DataFrame] = []
    for fold_index, split in enumerate(loso_splits):
        normalized_arrays = normalize_modalities(
            modality_arrays=modality_arrays,
            modality_masks=modality_masks,
            ordered_modalities=ordered_modalities,
            train_indices=split.train_indices,
        )
        model = build_stage_i_deep_model(
            model_name=model_name,
            ordered_modalities=ordered_modalities,
            modality_input_dims={
                name: normalized_arrays[name].shape[-1]
                for name in ordered_modalities
            },
            output_dim=3 if task_type == "classification" else 1,
            hidden_dim=config.deep_hidden_dim,
            num_heads=config.deep_num_heads,
            layers=config.deep_layers,
            dropout=config.deep_dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.deep_learning_rate)
        criterion = torch.nn.CrossEntropyLoss() if task_type == "classification" else torch.nn.MSELoss()
        train_deep_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            modality_arrays=normalized_arrays,
            modality_masks=modality_masks,
            time_axis=time_axis,
            labels=labels,
            task_type=task_type,
            train_indices=split.train_indices,
            batch_size=config.deep_batch_size,
            epochs=config.deep_epochs,
            seed=config.seed + fold_index,
        )
        output = forward_deep_model(
            model=model,
            modality_arrays=normalized_arrays,
            modality_masks=modality_masks,
            time_axis=time_axis,
            indices=split.test_indices,
        )
        if task_type == "classification":
            logits = _sanitize_classification_logits(output.logits.detach().cpu().numpy())
            predicted = logits.argmax(axis=1)
        else:
            predicted, _ = _sanitize_regression_outputs(
                output.logits.detach().cpu().numpy().reshape(-1),
                fallback_value=_safe_regression_fallback(labels[split.train_indices]),
            )
        frames.append(
            _build_prediction_frame(
                entries=[entries[index] for index in split.test_indices],
                evaluation_group="leave_one_view_out",
                track="objective" if task_type == "classification" else "subjective",
                model_name=model_name,
                y_true=labels[split.test_indices],
                y_pred=predicted,
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def run_retrieval_task(
    *,
    task_entries: Sequence[StageIPrivateTaskEntry],
    variant_feature_frames: Mapping[str, pd.DataFrame],
    variant_order: Sequence[str] | None = None,
    target_variant_name: str = "g_min",
) -> dict[str, object]:
    results: dict[str, object] = {}
    ordered_names = tuple(variant_order or variant_feature_frames.keys())
    for variant_name in ordered_names:
        frame = variant_feature_frames[variant_name]
        merged = merge_task_features(task_entries, frame, task_type="retrieval")
        if merged.empty:
            results[variant_name] = {"status": "not_run"}
            continue
        if is_optimized_variant_name(variant_name, target_variant_name=target_variant_name):
            results[variant_name] = run_optimized_retrieval_variant(
                merged,
                variant_name=variant_name,
            )
            continue
        rows = []
        for sortie_id, sortie_frame in merged.groupby("sortie_id", sort=False):
            for row in sortie_frame.itertuples(index=False):
                candidates = sortie_frame.loc[
                    sortie_frame["pilot_id"] != row.pilot_id
                ].copy()
                if candidates.empty or row.paired_sample_id not in set(candidates["sample_id"]):
                    continue
                query = row.feature_vector
                candidate_matrix = np.stack(candidates["feature_vector"].to_list(), axis=0)
                similarities = _cosine_similarity(query, candidate_matrix)
                ranked = candidates.iloc[np.argsort(-similarities)].reset_index(drop=True)
                rank = int(ranked.index[ranked["sample_id"] == row.paired_sample_id][0]) + 1
                rows.append(
                    {
                        "sortie_id": sortie_id,
                        "sample_id": row.sample_id,
                        "paired_sample_id": row.paired_sample_id,
                        "top1_hit": int(rank == 1),
                        "reciprocal_rank": 1.0 / rank,
                    }
                )
        metric_frame = pd.DataFrame(rows)
        if metric_frame.empty:
            results[variant_name] = {"status": "not_run"}
            continue
        results[variant_name] = {
            "status": "completed",
            "sample_count": int(metric_frame.shape[0]),
            "top1_accuracy": float(metric_frame["top1_hit"].mean()),
            "mrr": float(metric_frame["reciprocal_rank"].mean()),
        }
    best_variant_name, best_metrics = select_best_variant(results, task_type="retrieval")
    return {
        "task_type": "retrieval",
        "variants": results,
        "best_variant": {
            "name": best_variant_name,
            "metrics": best_metrics,
        },
    }


def write_task_plots(
    task_results: Mapping[str, object],
    *,
    artifact_root,
) -> dict[str, str]:
    plots_root = artifact_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    t1_groups = {
        variant_name: {
            "macro_f1": payload["best_metrics"]["macro_f1"],
            "balanced_accuracy": payload["best_metrics"]["balanced_accuracy"],
        }
        for variant_name, payload in task_results[TASK_MANEUVER]["variants"].items()
        if payload.get("status") == "completed"
    }
    t2_groups = {
        variant_name: {
            "rmse": payload["best_metrics"]["rmse"],
            "mae": payload["best_metrics"]["mae"],
        }
        for variant_name, payload in task_results[TASK_RESPONSE]["variants"].items()
        if payload.get("status") == "completed"
    }
    t3_groups = {
        variant_name: {
            "top1_accuracy": payload["top1_accuracy"],
            "mrr": payload["mrr"],
        }
        for variant_name, payload in task_results[TASK_RETRIEVAL]["variants"].items()
        if payload.get("status") == "completed"
    }
    return {
        "t1_metrics": save_grouped_bar_plot(
            t1_groups,
            path=plots_root / "t1_maneuver_metrics.png",
            title="Private T1 maneuver metrics",
            ylabel="score",
        ),
        "t2_metrics": save_grouped_bar_plot(
            t2_groups,
            path=plots_root / "t2_response_metrics.png",
            title="Private T2 response metrics",
            ylabel="score",
        ),
        "t3_metrics": save_grouped_bar_plot(
            t3_groups,
            path=plots_root / "t3_retrieval_metrics.png",
            title="Private T3 retrieval metrics",
            ylabel="score",
        ),
    }


def build_conclusion(
    task_results: Mapping[str, object],
    diagnostics: Mapping[str, object],
    *,
    target_variant_name: str = "g_min",
) -> dict[str, object]:
    t1_variants = task_results[TASK_MANEUVER]["variants"]
    t2_variants = task_results[TASK_RESPONSE]["variants"]
    t3_variants = task_results[TASK_RETRIEVAL]["variants"]
    no_mask_name = (
        "g_no_causal_mask"
        if target_variant_name == "g_min"
        else optimized_no_mask_variant_name(target_variant_name)
    )
    target_t1 = t1_variants.get(target_variant_name, {}).get("best_metrics")
    target_t2 = t2_variants.get(target_variant_name, {}).get("best_metrics")
    target_t3 = t3_variants.get(target_variant_name)
    deep_t1 = task_results[TASK_MANEUVER]["best_deep_model"]["metrics"]
    deep_t2 = task_results[TASK_RESPONSE]["best_deep_model"]["metrics"]
    alignment_gain_supported = is_better_classification(
        t1_variants.get("f_full", {}).get("best_metrics"),
        t1_variants.get("e_baseline", {}).get("best_metrics"),
    ) and is_better_classification(
        t1_variants.get("e_baseline", {}).get("best_metrics"),
        t1_variants.get("naive_sync", {}).get("best_metrics"),
    )
    if not alignment_gain_supported:
        alignment_gain_supported = is_better_regression(
            t2_variants.get("f_full", {}).get("best_metrics"),
            t2_variants.get("e_baseline", {}).get("best_metrics"),
        ) and is_better_regression(
            t2_variants.get("e_baseline", {}).get("best_metrics"),
            t2_variants.get("naive_sync", {}).get("best_metrics"),
        )
    t1_target_beats_module_baselines = metrics_better_than_all(
        target_t1,
        (
            t1_variants.get("naive_sync", {}).get("best_metrics"),
            t1_variants.get("e_baseline", {}).get("best_metrics"),
            t1_variants.get("f_full", {}).get("best_metrics"),
            t1_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        task_type="classification",
    )
    t2_target_beats_module_baselines = metrics_better_than_all(
        target_t2,
        (
            t2_variants.get("naive_sync", {}).get("best_metrics"),
            t2_variants.get("e_baseline", {}).get("best_metrics"),
            t2_variants.get("f_full", {}).get("best_metrics"),
            t2_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        task_type="regression",
    )
    t3_target_beats_module_baselines = metrics_better_than_all(
        target_t3,
        (
            t3_variants.get("naive_sync"),
            t3_variants.get("e_baseline"),
            t3_variants.get("f_full"),
            t3_variants.get(no_mask_name),
        ),
        task_type="retrieval",
    )
    causal_gain_supported = (
        is_better_classification(target_t1, t1_variants.get("f_full", {}).get("best_metrics"))
        and is_better_classification(target_t1, t1_variants.get(no_mask_name, {}).get("best_metrics"))
    ) or (
        is_better_regression(target_t2, t2_variants.get("f_full", {}).get("best_metrics"))
        and is_better_regression(target_t2, t2_variants.get(no_mask_name, {}).get("best_metrics"))
    ) or (
        is_better_retrieval(target_t3, t3_variants.get("f_full"))
        and is_better_retrieval(target_t3, t3_variants.get(no_mask_name))
    )
    target_diagnostics = diagnostics.get(target_variant_name, {})
    no_mask_diagnostics = diagnostics.get(no_mask_name, {})
    diagnostic_supported = bool(
        (
            target_diagnostics.get("mean_top_event_concentration", 0.0)
            > no_mask_diagnostics.get("mean_top_event_concentration", 0.0)
            and target_diagnostics.get("mean_event_mask_interference", 0.0)
            > no_mask_diagnostics.get("mean_event_mask_interference", 0.0)
        )
        or (
            target_diagnostics.get("mean_causal_residual_gate", 0.0)
            > no_mask_diagnostics.get("mean_causal_residual_gate", 0.0)
        )
    )
    t1_target_beats_deep = is_better_classification(target_t1, deep_t1)
    t2_target_beats_deep = is_better_regression(target_t2, deep_t2)
    private_optimality_supported = (
        t1_target_beats_module_baselines
        and t2_target_beats_module_baselines
        and t3_target_beats_module_baselines
        and t1_target_beats_deep
        and t2_target_beats_deep
    )
    criterion_details = {
        f"t1_{target_variant_name}_beats_module_baselines": t1_target_beats_module_baselines,
        f"t2_{target_variant_name}_beats_module_baselines": t2_target_beats_module_baselines,
        f"t3_{target_variant_name}_beats_module_baselines": t3_target_beats_module_baselines,
        f"t1_{target_variant_name}_beats_best_deep": t1_target_beats_deep,
        f"t2_{target_variant_name}_beats_best_deep": t2_target_beats_deep,
        f"t1_{target_variant_name}_beats_{no_mask_name}": is_better_classification(
            target_t1,
            t1_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        f"t2_{target_variant_name}_beats_{no_mask_name}": is_better_regression(
            target_t2,
            t2_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        f"t3_{target_variant_name}_beats_{no_mask_name}": is_better_retrieval(
            target_t3,
            t3_variants.get(no_mask_name),
        ),
    }
    if target_variant_name == "g_min":
        criterion_details = {
            "t1_g_min_beats_module_baselines": t1_target_beats_module_baselines,
            "t2_g_min_beats_module_baselines": t2_target_beats_module_baselines,
            "t3_g_min_beats_module_baselines": t3_target_beats_module_baselines,
            "t1_g_min_beats_best_deep": t1_target_beats_deep,
            "t2_g_min_beats_best_deep": t2_target_beats_deep,
        }
    return {
        "target_variant_name": target_variant_name,
        "no_mask_variant_name": no_mask_name,
        "alignment_gain_supported": alignment_gain_supported,
        "causal_gain_supported": causal_gain_supported,
        "diagnostic_supported": diagnostic_supported,
        "private_optimality_supported": private_optimality_supported,
        "criterion_details": criterion_details,
    }


def task_entry_to_sequence_entry(task_entry: StageIPrivateTaskEntry) -> StageISequenceEntry:
    return StageISequenceEntry(
        sample_id=task_entry.sample_id,
        dataset_id=PRIVATE_DATASET_ID,
        subset_id=task_entry.task_name,
        subject_id=f"pilot_{task_entry.pilot_id}",
        session_id=task_entry.view_id,
        split_group=task_entry.view_id,
        training_role="primary",
        sequence_bundle_path="",
        sequence_length=0,
        modality_schema={},
        source_origin="stage_h_private_benchmark",
        task_family=task_entry.task_name,
        label_namespace=task_entry.label_name,
        objective_label_name=task_entry.label_name if task_entry.task_type == "classification" else None,
        objective_label_value=(
            CLASS_LABEL_TO_ID[str(task_entry.label_value)]
            if task_entry.task_type == "classification" and task_entry.label_value is not None
            else None
        ),
        subjective_target_name=task_entry.label_name if task_entry.task_type == "regression" else None,
        subjective_target_value=(
            float(task_entry.label_value)
            if task_entry.task_type == "regression" and task_entry.label_value is not None
            else None
        ),
        window_index=task_entry.window_index,
        context_payload=task_entry.context_payload,
    )


def select_best_variant(
    metrics_by_name: Mapping[str, object],
    *,
    task_type: str,
) -> tuple[str | None, Mapping[str, object] | None]:
    completed = {
        name: payload["best_metrics"] if "best_metrics" in payload else payload.get("metrics", payload)
        for name, payload in metrics_by_name.items()
        if payload.get("status", "completed") == "completed"
    }
    if not completed:
        return None, None
    if task_type == "classification":
        best_name = max(completed, key=lambda name: (completed[name]["macro_f1"], completed[name]["balanced_accuracy"]))
    elif task_type == "regression":
        best_name = min(completed, key=lambda name: (completed[name]["rmse"], completed[name]["mae"]))
    else:
        best_name = max(completed, key=lambda name: (completed[name]["top1_accuracy"], completed[name]["mrr"]))
    return best_name, completed[best_name]


def normalize_modalities(
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
        normalized[modality_name] = np.where(mask[:, :, None], transformed, 0.0).astype(np.float32)
    return normalized


def train_deep_model(
    *,
    model,
    optimizer,
    criterion,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    train_indices: np.ndarray,
    batch_size: int,
    epochs: int,
    seed: int,
) -> None:
    if len(train_indices) == 0:
        return
    for epoch in range(epochs):
        for batch_indices in iter_batches(len(train_indices), batch_size=batch_size, seed=seed + epoch):
            global_indices = train_indices[batch_indices]
            optimizer.zero_grad()
            output = forward_deep_model(
                model=model,
                modality_arrays=modality_arrays,
                modality_masks=modality_masks,
                time_axis=time_axis,
                indices=global_indices,
                training=True,
            )
            if task_type == "classification":
                targets = torch.as_tensor(labels[global_indices], dtype=torch.long)
                loss = criterion(output.logits, targets)
            else:
                targets = torch.as_tensor(labels[global_indices], dtype=torch.float32).view(-1, 1)
                loss = criterion(output.logits, targets)
            loss.backward()
            optimizer.step()


def forward_deep_model(
    *,
    model,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    indices: np.ndarray,
    training: bool = False,
):
    modality_tensor_map = {
        name: torch.as_tensor(modality_arrays[name][indices], dtype=torch.float32)
        for name in ("physiology", "vehicle")
    }
    mask_tensor_map = {
        name: torch.as_tensor(modality_masks[name][indices], dtype=torch.float32)
        for name in ("physiology", "vehicle")
    }
    time_tensor = torch.as_tensor(time_axis[indices], dtype=torch.float32)
    if training:
        model.train()
        return model(modality_tensor_map, time_axis=time_tensor, modality_masks=mask_tensor_map)
    model.eval()
    with torch.no_grad():
        return model(modality_tensor_map, time_axis=time_tensor, modality_masks=mask_tensor_map)


def iter_batches(length: int, *, batch_size: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(length)
    return tuple(order[start : start + batch_size] for start in range(0, length, max(batch_size, 1)))


def is_better_classification(current: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> bool:
    if current is None or reference is None:
        return False
    return (
        current["macro_f1"] > reference["macro_f1"]
        or (
            math.isclose(current["macro_f1"], reference["macro_f1"])
            and current["balanced_accuracy"] > reference["balanced_accuracy"]
        )
    )


def is_better_regression(current: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> bool:
    if current is None or reference is None:
        return False
    return (
        current["rmse"] < reference["rmse"]
        or (
            math.isclose(current["rmse"], reference["rmse"])
            and current["mae"] < reference["mae"]
        )
    )


def is_better_retrieval(current: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> bool:
    if current is None or reference is None:
        return False
    if current.get("status") != "completed" or reference.get("status") != "completed":
        return False
    return (
        current["top1_accuracy"] > reference["top1_accuracy"]
        or (
            math.isclose(current["top1_accuracy"], reference["top1_accuracy"])
            and current["mrr"] > reference["mrr"]
        )
    )


def metrics_better_than_all(
    current: Mapping[str, object] | None,
    references: Sequence[Mapping[str, object] | None],
    *,
    task_type: str,
) -> bool:
    if current is None:
        return False
    comparators = {
        "classification": is_better_classification,
        "regression": is_better_regression,
        "retrieval": is_better_retrieval,
    }
    comparator = comparators[task_type]
    return all(comparator(current, reference) for reference in references)


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_norm = query / np.clip(np.linalg.norm(query), a_min=1e-8, a_max=None)
    candidate_norm = candidates / np.clip(np.linalg.norm(candidates, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    return candidate_norm @ query_norm
