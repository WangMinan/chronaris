"""Runtime helpers for Stage I deep baseline pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from chronaris.dataset import (
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    load_stage_i_sequence_summary,
)
from chronaris.pipelines.stage_i.stage_i_deep_models import build_stage_i_deep_model
from chronaris.pipelines.stage_i.stage_i_phase3_assets import (
    extract_primary_metrics,
    load_stage_i_baseline_artifacts,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

if TYPE_CHECKING:
    from chronaris.pipelines.stage_i.stage_i_deep_baseline import StageIDeepBaselineConfig


def _deep_model_config_dict(config: "StageIDeepBaselineConfig") -> dict[str, object]:
    return {
        "hidden_dim": int(config.hidden_dim),
        "num_heads": int(config.num_heads),
        "layers": int(config.layers),
        "dropout": float(config.dropout),
        "epochs": int(config.epochs),
        "learning_rate": float(config.learning_rate),
        "batch_size": int(config.batch_size),
        "fusion_event_bias_weight": float(config.fusion_event_bias_weight),
        "fusion_lag_window_points": (
            int(config.fusion_lag_window_points)
            if config.fusion_lag_window_points is not None
            else None
        ),
        "fusion_normalize_states": bool(config.fusion_normalize_states),
        "train_sampling_policy": str(config.train_sampling_policy),
    }


def _fit_predict_classification(
    *,
    bundle,
    entries: Sequence[object],
    indices: np.ndarray,
    ordered_modalities: Sequence[str],
    labels: np.ndarray,
    label_order: Sequence[int],
    config: "StageIDeepBaselineConfig",
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
        logits = _sanitize_classification_logits(output.logits.detach().cpu().numpy())
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
    config: "StageIDeepBaselineConfig",
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
    config: "StageIDeepBaselineConfig",
) -> nn.Module:
    runtime_device = resolve_torch_device_name(config.device)
    normalized_arrays = _normalize_modalities(
        modality_arrays=modality_arrays,
        modality_masks=modality_masks,
        ordered_modalities=ordered_modalities,
        train_indices=train_indices,
    )
    seed_torch(config.seed, device=runtime_device)
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
        fusion_event_bias_weight=config.fusion_event_bias_weight,
        fusion_lag_window_points=config.fusion_lag_window_points,
        fusion_normalize_states=config.fusion_normalize_states,
    ).to(device=runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = (
        _classification_loss(train_targets, device=runtime_device, output_dim=output_dim)
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
            sampling_policy=config.train_sampling_policy,
            labels=train_targets if task == "classification" else None,
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
                target_tensor = torch.as_tensor(
                    batch_targets,
                    dtype=torch.long,
                    device=runtime_device,
                )
                loss = criterion(logits, target_tensor)
            else:
                target_tensor = torch.as_tensor(
                    batch_targets,
                    dtype=torch.float32,
                    device=runtime_device,
                ).view(-1, 1)
                loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
    return model


def _classification_loss(
    train_targets: np.ndarray,
    *,
    device: str,
    output_dim: int,
) -> nn.Module:
    if len(train_targets) == 0:
        return nn.CrossEntropyLoss()
    counts = np.bincount(np.asarray(train_targets, dtype=int), minlength=output_dim).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * float(output_dim))
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


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
    runtime_device = next(model.parameters()).device
    modality_tensor_map = {
        name: torch.as_tensor(
            modality_arrays[name][indices],
            dtype=torch.float32,
            device=runtime_device,
        )
        for name in ordered_modalities
    }
    mask_tensor_map = {
        name: torch.as_tensor(
            modality_masks[name][indices],
            dtype=torch.float32,
            device=runtime_device,
        )
        for name in ordered_modalities
    }
    time_tensor = torch.as_tensor(
        time_axis[indices],
        dtype=torch.float32,
        device=runtime_device,
    )
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
    normalized_extra = {key: list(values) for key, values in (extra_columns or {}).items()}
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
    sampling_policy: str = "none",
    labels: np.ndarray | None = None,
) -> Sequence[np.ndarray]:
    rng = np.random.default_rng(seed)
    if sampling_policy == "balanced_class":
        if labels is None or len(labels) != length:
            raise ValueError(
                "balanced_class sampling requires classification labels aligned with the train set."
            )
        numeric_labels = np.asarray(labels, dtype=int)
        class_counts = np.bincount(numeric_labels).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        sample_weights = 1.0 / class_counts[numeric_labels]
        sample_weights = sample_weights / sample_weights.sum()
        order = rng.choice(
            np.arange(length, dtype=int),
            size=length,
            replace=True,
            p=sample_weights,
        )
    elif sampling_policy == "none":
        order = rng.permutation(length)
    else:
        raise ValueError(f"unsupported train sampling policy: {sampling_policy}")
    return tuple(
        order[start : start + batch_size]
        for start in range(0, length, max(batch_size, 1))
    )


def _infer_profile(dataset_id: str, *, real_sortie_dataset_id: str) -> str:
    return "real_sortie_v1" if dataset_id == real_sortie_dataset_id else "window_v2"


def _sanitize_classification_logits(logits: np.ndarray) -> np.ndarray:
    return np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


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
