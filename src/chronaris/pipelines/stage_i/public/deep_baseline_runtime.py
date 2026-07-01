"""Runtime helpers for Stage I deep baseline pipelines."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import time
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
from chronaris.pipelines.stage_i.common.deep_models import build_stage_i_deep_model
from chronaris.pipelines.stage_i.legacy.phase3_assets import (
    extract_primary_metrics,
    load_stage_i_baseline_artifacts,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

if TYPE_CHECKING:
    from chronaris.pipelines.stage_i.public.deep_baseline import StageIDeepBaselineConfig

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deep_model_config_dict(config: "StageIDeepBaselineConfig") -> dict[str, object]:
    payload = {
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
        "regression_loss": str(config.regression_loss),
        "huber_delta": float(config.huber_delta),
        "target_transform": str(config.target_transform),
        "gradient_clip_max_norm": (
            float(config.gradient_clip_max_norm)
            if config.gradient_clip_max_norm is not None
            else None
        ),
        "weight_decay": float(config.weight_decay),
        "heartbeat_seconds": float(config.heartbeat_seconds),
        "batch_log_interval": int(config.batch_log_interval),
    }
    return payload


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
    curve_rows: list[dict[str, object]] = []
    for fold_index, split in enumerate(loso_splits, start=1):
        train_label_indices = np.asarray(
            [label_to_index[int(value)] for value in labels[split.train_indices]],
            dtype=int,
        )
        model, fold_curve_rows = _train_model(
            model_name=config.model_name,
            ordered_modalities=ordered_modalities,
            modality_arrays=bundle.modality_arrays,
            modality_masks=bundle.modality_masks,
            time_axis=bundle.time_axis,
            train_indices=indices[split.train_indices],
            train_targets=train_label_indices,
            output_dim=len(label_order),
            task="classification",
            config=config,
            progress_context={
                "dataset_id": config.dataset_id,
                "track": "objective",
                "evaluation_group": evaluation_group,
                "fold_index": fold_index,
                "fold_count": len(loso_splits),
                "split_group": split.split_group,
            },
        )
        curve_rows.extend(
            _attach_curve_context(
                fold_curve_rows,
                track="objective",
                evaluation_group=evaluation_group,
                fold_index=fold_index,
                split_group=split.split_group,
                seed=config.seed,
            )
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
        logits = _apply_classification_logit_adjustment(
            logits,
            train_targets=train_label_indices,
            output_dim=len(label_order),
            sampling_policy=config.train_sampling_policy,
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
    predictions = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    predictions.attrs["training_curves"] = curve_rows
    return predictions


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
    curve_rows: list[dict[str, object]] = []
    for fold_index, split in enumerate(loso_splits, start=1):
        target_transform = _fit_target_transform(
            targets[split.train_indices],
            transform_name=config.target_transform,
        )
        transformed_train_targets = _transform_targets(
            targets[split.train_indices],
            target_transform,
        )
        model, fold_curve_rows = _train_model(
            model_name=config.model_name,
            ordered_modalities=ordered_modalities,
            modality_arrays=bundle.modality_arrays,
            modality_masks=bundle.modality_masks,
            time_axis=bundle.time_axis,
            train_indices=indices[split.train_indices],
            train_targets=transformed_train_targets,
            output_dim=1,
            task="regression",
            config=config,
            progress_context={
                "dataset_id": config.dataset_id,
                "track": "subjective",
                "evaluation_group": evaluation_group,
                "fold_index": fold_index,
                "fold_count": len(loso_splits),
                "split_group": split.split_group,
                "target_transform_name": target_transform["name"],
                "target_transform_center": target_transform["center"],
                "target_transform_scale": target_transform["scale"],
            },
        )
        curve_rows.extend(
            _attach_curve_context(
                fold_curve_rows,
                track="subjective",
                evaluation_group=evaluation_group,
                fold_index=fold_index,
                split_group=split.split_group,
                seed=config.seed,
            )
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
        transformed_prediction_values, nonfinite_mask = _sanitize_regression_outputs(
            output.logits.detach().cpu().numpy().reshape(-1),
            fallback_value=_safe_regression_fallback(transformed_train_targets),
        )
        prediction_values = _inverse_transform_targets(
            transformed_prediction_values,
            target_transform,
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
                    "target_transform": [target_transform["name"]] * len(nonfinite_mask),
                    "target_transform_center": [
                        target_transform["center"]
                    ] * len(nonfinite_mask),
                    "target_transform_scale": [
                        target_transform["scale"]
                    ] * len(nonfinite_mask),
                },
            ),
        )
    predictions = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    predictions.attrs["training_curves"] = curve_rows
    return predictions


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
    progress_context: Mapping[str, object] | None = None,
) -> tuple[nn.Module, list[dict[str, object]]]:
    context = dict(progress_context or {})
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=float(config.weight_decay),
    )
    criterion = (
        _classification_loss(
            train_targets,
            device=runtime_device,
            output_dim=output_dim,
            sampling_policy=config.train_sampling_policy,
        )
        if task == "classification"
        else _regression_loss(config.regression_loss, huber_delta=config.huber_delta)
    )
    if len(train_indices) == 0:
        return model, []
    curve_rows: list[dict[str, object]] = []
    checkpoint_dir = Path(config.artifact_root) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_index = int(context.get("fold_index", 1))
    fold_count = int(context.get("fold_count", 1))
    batch_log_interval = max(int(config.batch_log_interval), 1)
    heartbeat_seconds = max(float(config.heartbeat_seconds), 1.0)
    LOGGER.info(
        "deep_train fold_start dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d split_group=%s task=%s train_count=%d epochs=%d batch_size=%d",
        context.get("dataset_id", config.dataset_id),
        model_name,
        context.get("track", task),
        context.get("evaluation_group", "unknown"),
        config.seed,
        fold_index,
        fold_count,
        context.get("split_group", "unknown"),
        task,
        len(train_indices),
        config.epochs,
        config.batch_size,
    )
    _write_training_progress(
        config,
        "fold_start",
        **_context_progress_fields(context),
        model_name=model_name,
        task=task,
        seed=config.seed,
        fold_index=fold_index,
        fold_count=fold_count,
        train_count=len(train_indices),
        epochs=config.epochs,
        batch_size=config.batch_size,
    )
    fold_started_at = time.monotonic()
    for epoch in range(config.epochs):
        epoch_started_at = time.monotonic()
        last_heartbeat_at = epoch_started_at
        last_loss = float("nan")
        batch_losses: list[float] = []
        batches = _iter_batches(
            len(train_indices),
            batch_size=config.batch_size,
            seed=config.seed + epoch,
            sampling_policy=config.train_sampling_policy,
            labels=train_targets if task == "classification" else None,
        )
        batch_count = len(batches)
        LOGGER.info(
            "deep_train epoch_start dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d epoch=%03d/%03d batch_count=%d",
            context.get("dataset_id", config.dataset_id),
            model_name,
            context.get("track", task),
            context.get("evaluation_group", "unknown"),
            config.seed,
            fold_index,
            fold_count,
            epoch + 1,
            config.epochs,
            batch_count,
        )
        _write_training_progress(
            config,
            "epoch_start",
            **_context_progress_fields(context),
            model_name=model_name,
            task=task,
            seed=config.seed,
            fold_index=fold_index,
            fold_count=fold_count,
            epoch=epoch + 1,
            epochs=config.epochs,
            batch_count=batch_count,
        )
        for batch_number, batch_indices in enumerate(batches, start=1):
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
            if (
                config.gradient_clip_max_norm is not None
                and float(config.gradient_clip_max_norm) > 0.0
            ):
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.gradient_clip_max_norm),
                )
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            if not np.isfinite(last_loss):
                LOGGER.warning(
                    "deep_train nonfinite_loss dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d epoch=%03d/%03d batch=%03d/%03d loss=%s",
                    context.get("dataset_id", config.dataset_id),
                    model_name,
                    context.get("track", task),
                    context.get("evaluation_group", "unknown"),
                    config.seed,
                    fold_index,
                    fold_count,
                    epoch + 1,
                    config.epochs,
                    batch_number,
                    batch_count,
                    last_loss,
                )
                _write_training_progress(
                    config,
                    "nonfinite_loss",
                    **_context_progress_fields(context),
                    model_name=model_name,
                    task=task,
                    seed=config.seed,
                    fold_index=fold_index,
                    fold_count=fold_count,
                    epoch=epoch + 1,
                    epochs=config.epochs,
                    batch=batch_number,
                    batch_count=batch_count,
                    loss=last_loss,
                )
            batch_losses.append(last_loss)
            now = time.monotonic()
            should_log_batch = (
                batch_number == 1
                or batch_number == batch_count
                or batch_number % batch_log_interval == 0
            )
            should_heartbeat = now - last_heartbeat_at >= heartbeat_seconds
            if should_log_batch or should_heartbeat:
                LOGGER.info(
                    "%s dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d epoch=%03d/%03d batch=%03d/%03d loss=%.6f elapsed_s=%.1f status=running",
                    "deep_train heartbeat" if should_heartbeat and not should_log_batch else "deep_train batch",
                    context.get("dataset_id", config.dataset_id),
                    model_name,
                    context.get("track", task),
                    context.get("evaluation_group", "unknown"),
                    config.seed,
                    fold_index,
                    fold_count,
                    epoch + 1,
                    config.epochs,
                    batch_number,
                    batch_count,
                    last_loss,
                    now - fold_started_at,
                )
                _write_training_progress(
                    config,
                    "heartbeat" if should_heartbeat else "batch_progress",
                    **_context_progress_fields(context),
                    model_name=model_name,
                    task=task,
                    seed=config.seed,
                    fold_index=fold_index,
                    fold_count=fold_count,
                    epoch=epoch + 1,
                    epochs=config.epochs,
                    batch=batch_number,
                    batch_count=batch_count,
                    loss=last_loss,
                    elapsed_s=now - fold_started_at,
                )
                if should_heartbeat:
                    last_heartbeat_at = now
        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        LOGGER.info(
            "deep_train epoch_done dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d epoch=%03d/%03d train_loss=%.6f elapsed_s=%.1f",
            context.get("dataset_id", config.dataset_id),
            model_name,
            context.get("track", task),
            context.get("evaluation_group", "unknown"),
            config.seed,
            fold_index,
            fold_count,
            epoch + 1,
            config.epochs,
            epoch_loss,
            time.monotonic() - epoch_started_at,
        )
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch + 1),
            "train_loss": epoch_loss,
            "config": _deep_model_config_dict(config),
            "progress_context": context,
        }
        torch.save(checkpoint_payload, checkpoint_dir / "checkpoint_last.pt")
        fold_checkpoint_path = (
            checkpoint_dir
            / (
                f"checkpoint_last_{context.get('track', task)}_"
                f"{context.get('evaluation_group', 'group')}_fold{fold_index:03d}.pt"
            )
        )
        torch.save(
            checkpoint_payload,
            fold_checkpoint_path,
        )
        _write_training_progress(
            config,
            "epoch_done",
            **_context_progress_fields(context),
            model_name=model_name,
            task=task,
            seed=config.seed,
            fold_index=fold_index,
            fold_count=fold_count,
            epoch=epoch + 1,
            epochs=config.epochs,
            train_loss=epoch_loss,
            checkpoint_last_path=str(checkpoint_dir / "checkpoint_last.pt"),
            checkpoint_fold_path=str(fold_checkpoint_path),
        )
        curve_rows.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": epoch_loss,
                "batch_count": int(len(batch_losses)),
                "task": task,
                "model_name": model_name,
                "learning_rate": float(config.learning_rate),
                "batch_size": int(config.batch_size),
                "weight_decay": float(config.weight_decay),
                "regression_loss": str(config.regression_loss),
                "target_transform": str(config.target_transform),
                "checkpoint_last_path": str(checkpoint_dir / "checkpoint_last.pt"),
            }
        )
    LOGGER.info(
        "deep_train fold_done dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d elapsed_s=%.1f",
        context.get("dataset_id", config.dataset_id),
        model_name,
        context.get("track", task),
        context.get("evaluation_group", "unknown"),
        config.seed,
        fold_index,
        fold_count,
        time.monotonic() - fold_started_at,
    )
    _write_training_progress(
        config,
        "fold_done",
        **_context_progress_fields(context),
        model_name=model_name,
        task=task,
        seed=config.seed,
        fold_index=fold_index,
        fold_count=fold_count,
        elapsed_s=time.monotonic() - fold_started_at,
    )
    return model, curve_rows


def _write_training_progress(
    config: "StageIDeepBaselineConfig",
    event: str,
    **fields: object,
) -> None:
    progress_path = Path(config.artifact_root) / "progress.json"
    try:
        state = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {}
        )
    except json.JSONDecodeError:
        state = {}
    record = {
        "timestamp_utc": _utc_now(),
        "event": event,
        **{key: _jsonable(value) for key, value in fields.items()},
    }
    events = list(state.get("events") or [])
    events.append(record)
    state.update(record)
    state["last_event"] = event
    state["updated_at_utc"] = record["timestamp_utc"]
    state["events"] = events[-200:]
    progress_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _context_progress_fields(context: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in context.items()
        if key not in {"fold_index", "fold_count"}
    }


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _regression_loss(loss_name: str, *, huber_delta: float) -> nn.Module:
    normalized = loss_name.strip().lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "smooth_l1":
        return nn.SmoothL1Loss(beta=float(huber_delta))
    if normalized == "huber":
        return nn.HuberLoss(delta=float(huber_delta))
    raise ValueError(f"unsupported regression loss: {loss_name}")


def _fit_target_transform(
    train_targets: np.ndarray,
    *,
    transform_name: str,
) -> dict[str, float | str]:
    normalized = transform_name.strip().lower()
    finite = np.asarray(train_targets, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or normalized == "none":
        return {"name": "none", "center": 0.0, "scale": 1.0}
    if normalized == "zscore_train":
        center = float(finite.mean())
        scale = float(finite.std())
    elif normalized == "robust_train":
        center = float(np.median(finite))
        q25, q75 = np.percentile(finite, [25, 75])
        scale = float(q75 - q25)
    else:
        raise ValueError(f"unsupported target transform: {transform_name}")
    if not np.isfinite(scale) or abs(scale) < 1e-6:
        scale = 1.0
    return {"name": normalized, "center": center, "scale": scale}


def _transform_targets(
    values: np.ndarray,
    target_transform: Mapping[str, float | str],
) -> np.ndarray:
    center = float(target_transform["center"])
    scale = float(target_transform["scale"])
    return ((np.asarray(values, dtype=np.float32) - center) / scale).astype(np.float32)


def _inverse_transform_targets(
    values: np.ndarray,
    target_transform: Mapping[str, float | str],
) -> np.ndarray:
    center = float(target_transform["center"])
    scale = float(target_transform["scale"])
    return (np.asarray(values, dtype=np.float32) * scale + center).astype(np.float32)


def _attach_curve_context(
    rows: Sequence[Mapping[str, object]],
    *,
    track: str,
    evaluation_group: str,
    fold_index: int,
    split_group: str,
    seed: int,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        payload = dict(row)
        payload.update(
            {
                "track": track,
                "evaluation_group": evaluation_group,
                "fold_index": int(fold_index),
                "split_group": split_group,
                "seed": int(seed),
            }
        )
        enriched.append(payload)
    return enriched


def _classification_loss(
    train_targets: np.ndarray,
    *,
    device: str,
    output_dim: int,
    sampling_policy: str = "none",
) -> nn.Module:
    if sampling_policy == "balanced_class":
        return nn.CrossEntropyLoss()
    if len(train_targets) == 0:
        return nn.CrossEntropyLoss()
    counts = np.bincount(np.asarray(train_targets, dtype=int), minlength=output_dim).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * float(output_dim))
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def _apply_classification_logit_adjustment(
    logits: np.ndarray,
    *,
    train_targets: np.ndarray,
    output_dim: int,
    sampling_policy: str,
) -> np.ndarray:
    if sampling_policy != "balanced_class" or len(train_targets) == 0:
        return logits
    class_counts = np.bincount(np.asarray(train_targets, dtype=int), minlength=output_dim).astype(
        np.float64
    )
    class_counts = np.maximum(class_counts, 1.0)
    priors = class_counts / class_counts.sum()
    return logits + np.log(priors).reshape(1, -1)


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
