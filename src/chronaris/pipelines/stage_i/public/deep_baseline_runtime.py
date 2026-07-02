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
from chronaris.pipelines.stage_i.common.deep_models import StageIDeepForwardResult
from chronaris.pipelines.stage_i.common.gpu_runtime import (
    AmpRuntime,
    get_train_batch,
    gpu_runtime_snapshot,
    iter_eval_batches,
    is_cuda_oom,
    make_grad_scaler,
    prepare_fold_tensors,
    resolve_amp_runtime,
)
from chronaris.pipelines.stage_i.legacy.phase3_assets import (
    extract_primary_metrics,
    load_stage_i_baseline_artifacts,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

if TYPE_CHECKING:
    from chronaris.pipelines.stage_i.public.deep_baseline import StageIDeepBaselineConfig

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

_AUTO_BATCH_MEMORY_PRESSURE_LIMIT = 0.88
_AUTO_BATCH_SELECTION_CACHE: dict[tuple[object, ...], tuple[int, list[dict[str, object]]]] = {}


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
        "tensor_cache": str(config.tensor_cache),
        "max_cache_gb": float(config.max_cache_gb),
        "pin_memory": bool(config.pin_memory),
        "non_blocking_copy": bool(config.non_blocking_copy),
        "auto_batch_size": bool(config.auto_batch_size),
        "batch_size_candidates": [int(value) for value in config.batch_size_candidates],
        "amp": str(config.amp),
        "amp_eval": bool(config.amp_eval),
        "torch_compile": str(config.torch_compile),
        "profile_gpu": bool(config.profile_gpu),
        "eval_batch_size": (
            int(config.eval_batch_size)
            if config.eval_batch_size is not None
            else None
        ),
        "checkpoint_policy": str(config.checkpoint_policy),
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
        model, fold_curve_rows, prepared = _train_model(
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
        output = _forward_prepared_dataset(
            model=model,
            prepared=prepared,
            indices=indices[split.test_indices],
            batch_size=_resolve_eval_batch_size(config),
            amp=_eval_amp(config, str(next(model.parameters()).device).split(":")[0]),
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
        model, fold_curve_rows, prepared = _train_model(
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
        output = _forward_prepared_dataset(
            model=model,
            prepared=prepared,
            indices=indices[split.test_indices],
            batch_size=_resolve_eval_batch_size(config),
            amp=_eval_amp(config, str(next(model.parameters()).device).split(":")[0]),
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
) -> tuple[nn.Module, list[dict[str, object]], object]:
    context = dict(progress_context or {})
    runtime_device = resolve_torch_device_name(config.device)
    seed_torch(config.seed, device=runtime_device)
    model = build_stage_i_deep_model(
        model_name=model_name,
        ordered_modalities=ordered_modalities,
        modality_input_dims={
            name: modality_arrays[name].shape[-1]
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
        dataset_id=config.dataset_id,
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
        empty_targets = np.zeros(next(iter(modality_arrays.values())).shape[0], dtype=np.float32)
        prepared = prepare_fold_tensors(
            modality_arrays=modality_arrays,
            modality_masks=modality_masks,
            time_axis=time_axis,
            ordered_modalities=ordered_modalities,
            train_indices=train_indices,
            targets=empty_targets,
            requested_mode=config.tensor_cache,
            device=runtime_device,
            max_cache_gb=config.max_cache_gb,
            pin_memory=config.pin_memory,
            non_blocking_copy=config.non_blocking_copy,
        )
        return model, [], prepared
    target_full = np.zeros(next(iter(modality_arrays.values())).shape[0], dtype=np.float32)
    target_full[np.asarray(train_indices, dtype=int)] = np.asarray(train_targets, dtype=np.float32)
    prepared = prepare_fold_tensors(
        modality_arrays=modality_arrays,
        modality_masks=modality_masks,
        time_axis=time_axis,
        ordered_modalities=ordered_modalities,
        train_indices=train_indices,
        targets=target_full,
        requested_mode=config.tensor_cache,
        device=runtime_device,
        max_cache_gb=config.max_cache_gb,
        pin_memory=config.pin_memory,
        non_blocking_copy=config.non_blocking_copy,
    )
    amp = resolve_amp_runtime(
        requested_mode=config.amp,
        device=runtime_device,
        grad_scaler=config.grad_scaler,
    )
    scaler = make_grad_scaler(amp, device=runtime_device)
    selected_batch_size, batch_attempts = _select_train_batch_size(
        config=config,
        model=model,
        model_name=model_name,
        criterion=criterion,
        prepared=prepared,
        train_indices=train_indices,
        train_targets=train_targets,
        output_dim=output_dim,
        task=task,
        amp=amp,
        runtime_device=runtime_device,
        context=context,
    )
    model, compile_info = _maybe_compile_deep_model(
        config=config,
        model=model,
        prepared=prepared,
        train_indices=train_indices,
        runtime_device=runtime_device,
    )
    curve_rows: list[dict[str, object]] = []
    checkpoint_policy = _checkpoint_policy(config.checkpoint_policy)
    checkpoint_dir = Path(config.artifact_root) / "checkpoints"
    if checkpoint_policy != "off":
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
        selected_batch_size,
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
        batch_size=selected_batch_size,
        requested_batch_size=config.batch_size,
        tensor_cache_mode=prepared.tensor_cache_mode,
        amp_mode=amp.resolved_mode,
        compile_mode=compile_info["compile_mode"],
        compile_status=compile_info["compile_status"],
    )
    fold_started_at = time.monotonic()
    for epoch in range(config.epochs):
        epoch_started_at = time.monotonic()
        last_heartbeat_at = epoch_started_at
        last_loss = float("nan")
        batch_losses: list[float] = []
        batches = _iter_batches(
            len(train_indices),
            batch_size=selected_batch_size,
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
            batch_started_at = time.monotonic()
            global_batch = train_indices[batch_indices]
            optimizer.zero_grad()
            modality_batch, mask_batch, time_batch, target_tensor = get_train_batch(
                prepared,
                global_batch,
                device=runtime_device,
            )
            with amp.autocast(device=runtime_device):
                output = model(
                    modality_batch,
                    time_axis=time_batch,
                    modality_masks=mask_batch,
                )
                logits = output.logits
            if logits is None:
                raise ValueError("deep model returned no logits during supervised training.")
            if target_tensor is None:
                raise ValueError("prepared fold tensor cache did not include targets.")
            if task == "classification":
                loss = _compute_classification_loss(criterion, logits, target_tensor)
            else:
                loss = _compute_regression_loss(criterion, logits, target_tensor)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (
                config.gradient_clip_max_norm is not None
                and float(config.gradient_clip_max_norm) > 0.0
            ):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.gradient_clip_max_norm),
                )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _sync_runtime(runtime_device)
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
            batch_elapsed_s = now - batch_started_at
            samples_per_sec = (
                float(len(global_batch) / batch_elapsed_s)
                if batch_elapsed_s > 0
                else float("nan")
            )
            snapshot = _training_gpu_snapshot(runtime_device) if config.profile_gpu else {}
            gpu_mem_gb = snapshot.get("gpu_memory_allocated_gb")
            should_log_batch = (
                batch_number == 1
                or batch_number == batch_count
                or batch_number % batch_log_interval == 0
            )
            should_heartbeat = now - last_heartbeat_at >= heartbeat_seconds
            if should_log_batch or should_heartbeat:
                LOGGER.info(
                    "%s run_id=%s dataset=%s model=%s track=%s group=%s seed=%s fold=%03d/%03d epoch=%03d/%03d batch=%03d/%03d loss=%.6f samples_per_sec=%.2f gpu_mem_gb=%.3f elapsed_s=%.1f device=%s status=running",
                    "deep_train heartbeat" if should_heartbeat and not should_log_batch else "deep_train batch",
                    Path(config.artifact_root).name,
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
                    samples_per_sec,
                    float(gpu_mem_gb or 0.0),
                    now - fold_started_at,
                    runtime_device,
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
                    samples_per_sec=samples_per_sec,
                    gpu_mem_gb=gpu_mem_gb,
                    elapsed_s=now - fold_started_at,
                    device=runtime_device,
                    tensor_cache_mode=prepared.tensor_cache_mode,
                    amp_mode=amp.resolved_mode,
                    compile_status=compile_info["compile_status"],
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
        checkpoint_last_path: Path | None = None
        checkpoint_fold_path: Path | None = None
        if checkpoint_policy != "off":
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": int(epoch + 1),
                "train_loss": epoch_loss,
                "config": _deep_model_config_dict(config),
                "progress_context": context,
                "gpuopt": {
                    "tensor_cache_mode": prepared.tensor_cache_mode,
                    "amp_mode": amp.resolved_mode,
                    "auto_batch_size": bool(config.auto_batch_size),
                    "selected_batch_size": int(selected_batch_size),
                    "requested_batch_size": int(config.batch_size),
                    "batch_size_attempts": batch_attempts,
                    **compile_info,
                },
            }
            checkpoint_last_path = checkpoint_dir / "checkpoint_last.pt"
            torch.save(checkpoint_payload, checkpoint_last_path)
            if checkpoint_policy == "epoch_and_fold":
                checkpoint_fold_path = (
                    checkpoint_dir
                    / (
                        f"checkpoint_last_{context.get('track', task)}_"
                        f"{context.get('evaluation_group', 'group')}_fold{fold_index:03d}.pt"
                    )
                )
                torch.save(checkpoint_payload, checkpoint_fold_path)
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
            checkpoint_policy=checkpoint_policy,
            checkpoint_last_path=str(checkpoint_last_path) if checkpoint_last_path else None,
            checkpoint_fold_path=str(checkpoint_fold_path) if checkpoint_fold_path else None,
        )
        curve_rows.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": epoch_loss,
                "batch_count": int(len(batch_losses)),
                "task": task,
                "model_name": model_name,
                "learning_rate": float(config.learning_rate),
                "batch_size": int(selected_batch_size),
                "requested_batch_size": int(config.batch_size),
                "weight_decay": float(config.weight_decay),
                "regression_loss": str(config.regression_loss),
                "target_transform": str(config.target_transform),
                "checkpoint_last_path": str(checkpoint_last_path) if checkpoint_last_path else None,
                "checkpoint_policy": checkpoint_policy,
                "tensor_cache_mode": prepared.tensor_cache_mode,
                "requested_tensor_cache": str(config.tensor_cache),
                "cache_build_time_s": float(prepared.cache_build_time_s),
                "estimated_cache_gb": float(prepared.estimated_cache_gb),
                "actual_cache_gb": float(prepared.actual_cache_gb),
                "cache_fallback_reason": prepared.fallback_reason,
                "amp_mode": amp.resolved_mode,
                "amp_fallback_reason": amp.fallback_reason,
                "auto_batch_size": bool(config.auto_batch_size),
                "batch_size_attempts": json.dumps(batch_attempts),
                "torch_compile": str(config.torch_compile),
                **compile_info,
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
    return model, curve_rows, prepared


def _select_train_batch_size(
    *,
    config: "StageIDeepBaselineConfig",
    model: nn.Module,
    model_name: str,
    criterion: nn.Module,
    prepared,
    train_indices: np.ndarray,
    train_targets: np.ndarray,
    output_dim: int,
    task: str,
    amp: AmpRuntime,
    runtime_device: str,
    context: Mapping[str, object],
) -> tuple[int, list[dict[str, object]]]:
    del train_targets, output_dim
    if not config.auto_batch_size or runtime_device != "cuda":
        return int(config.batch_size), [{"batch_size": int(config.batch_size), "status": "fixed"}]

    candidates = _batch_size_candidates(config)
    if not candidates:
        return int(config.batch_size), [{"batch_size": int(config.batch_size), "status": "fallback"}]
    cache_key = _auto_batch_cache_key(
        config=config,
        model_name=model_name,
        task=task,
        amp=amp,
        prepared=prepared,
        train_count=len(train_indices),
        context=context,
        candidates=candidates,
    )
    cached = _AUTO_BATCH_SELECTION_CACHE.get(cache_key)
    if cached is not None:
        selected, attempts = cached
        cached_attempts = [dict(row) for row in attempts]
        cached_attempts.append({"batch_size": int(selected), "status": "cache_hit"})
        return int(selected), cached_attempts

    attempts: list[dict[str, object]] = []

    def _probe(batch_size: int, *, record: bool = True) -> dict[str, object] | None:
        sample_size = min(int(batch_size), len(train_indices))
        sample = train_indices[:sample_size]
        if len(sample) == 0:
            return None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            modality_batch, mask_batch, time_batch, target_tensor = get_train_batch(
                prepared,
                sample,
                device=runtime_device,
            )
            if target_tensor is None:
                raise ValueError("prepared fold tensor cache did not include targets.")
            model.zero_grad(set_to_none=True)
            _sync_runtime(runtime_device)
            started = time.monotonic()
            with amp.autocast(device=runtime_device):
                logits = model(
                    modality_batch,
                    time_axis=time_batch,
                    modality_masks=mask_batch,
                ).logits
                if logits is None:
                    raise ValueError("deep model returned no logits during auto batch probing.")
                if task == "classification":
                    loss = _compute_classification_loss(criterion, logits, target_tensor)
                else:
                    loss = _compute_regression_loss(criterion, logits, target_tensor)
            loss.backward()
            _sync_runtime(runtime_device)
            elapsed_s = max(time.monotonic() - started, 1e-9)
            loss_value = float(loss.detach().cpu().item())
            model.zero_grad(set_to_none=True)
            peak_allocated_gb, peak_reserved_gb, memory_pressure = _cuda_probe_memory_pressure()
            samples_per_sec = float(sample_size / elapsed_s)
            estimated_epoch_s = float(np.ceil(len(train_indices) / max(int(batch_size), 1)) * elapsed_s)
            status = (
                "ok"
                if memory_pressure <= _AUTO_BATCH_MEMORY_PRESSURE_LIMIT
                else "ok_memory_pressure"
            )
            row: dict[str, object] = {
                "batch_size": int(batch_size),
                "sample_size": int(sample_size),
                "status": status,
                "elapsed_s": elapsed_s,
                "samples_per_sec": samples_per_sec,
                "estimated_epoch_s": estimated_epoch_s,
                "loss": loss_value,
                "gpu_peak_allocated_gb": peak_allocated_gb,
                "gpu_peak_reserved_gb": peak_reserved_gb,
                "gpu_memory_pressure": memory_pressure,
                "memory_pressure_limit": _AUTO_BATCH_MEMORY_PRESSURE_LIMIT,
            }
            if record:
                attempts.append(row)
            return row
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if not is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            row = {
                "batch_size": int(batch_size),
                "sample_size": int(sample_size),
                "status": "oom",
                "reason": str(exc).splitlines()[0],
            }
            if record:
                attempts.append(row)
            return row
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    warmup_size = min(max(min(candidates), 1), len(train_indices), 128)
    if warmup_size > 0:
        _probe(warmup_size, record=False)

    for candidate in candidates:
        _probe(candidate)

    ok_attempts = [
        row
        for row in attempts
        if str(row.get("status")) in {"ok", "ok_memory_pressure"}
        and np.isfinite(float(row.get("estimated_epoch_s", float("nan"))))
    ]
    if not ok_attempts:
        attempts.append({"batch_size": int(config.batch_size), "status": "fallback"})
        selected = int(config.batch_size)
    else:
        eligible_attempts = [
            row for row in ok_attempts if str(row.get("status")) == "ok"
        ] or ok_attempts
        selected_row = min(
            eligible_attempts,
            key=lambda row: (
                float(row.get("estimated_epoch_s", float("inf"))),
                -int(row.get("batch_size", 0)),
            ),
        )
        selected = int(selected_row["batch_size"])
        attempts.append(
            {
                "batch_size": selected,
                "status": "selected",
                "selection_policy": (
                    "min_estimated_epoch_seconds_with_memory_pressure_filter"
                ),
            }
        )
    _AUTO_BATCH_SELECTION_CACHE[cache_key] = (selected, [dict(row) for row in attempts])
    LOGGER.info(
        "deep_train auto_batch_selected dataset=%s model=%s task=%s group=%s train_count=%d batch_size=%d attempts=%s",
        context.get("dataset_id", config.dataset_id),
        model_name,
        task,
        context.get("evaluation_group", "unknown"),
        len(train_indices),
        selected,
        json.dumps(attempts, ensure_ascii=True),
    )
    return selected, attempts


def _batch_size_candidates(config: "StageIDeepBaselineConfig") -> list[int]:
    candidates = [int(value) for value in config.batch_size_candidates if int(value) > 0]
    if not candidates:
        candidates = [int(config.batch_size)]
    return list(dict.fromkeys(candidates))


def _auto_batch_cache_key(
    *,
    config: "StageIDeepBaselineConfig",
    model_name: str,
    task: str,
    amp: AmpRuntime,
    prepared,
    train_count: int,
    context: Mapping[str, object],
    candidates: Sequence[int],
) -> tuple[object, ...]:
    modality_shapes = tuple(
        (name, tuple(int(value) for value in tensor.shape[1:]))
        for name, tensor in sorted(prepared.modality_tensors.items())
    )
    train_count_bucket = int(np.ceil(max(int(train_count), 1) / 512.0) * 512)
    return (
        str(context.get("dataset_id", config.dataset_id)),
        str(model_name),
        str(task),
        str(context.get("track", "")),
        str(context.get("evaluation_group", "")),
        int(config.hidden_dim),
        int(config.layers),
        int(config.num_heads),
        amp.resolved_mode,
        prepared.tensor_cache_mode,
        tuple(int(value) for value in candidates),
        train_count_bucket,
        modality_shapes,
    )


def _cuda_probe_memory_pressure() -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    peak_allocated_gb = float(torch.cuda.max_memory_allocated() / (1024.0**3))
    peak_reserved_gb = float(torch.cuda.max_memory_reserved() / (1024.0**3))
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_gb = float((total_bytes - free_bytes) / (1024.0**3))
        total_gb = float(total_bytes / (1024.0**3))
        pressure = max(peak_reserved_gb, used_gb) / total_gb if total_gb > 0 else 0.0
    except Exception:  # pragma: no cover - host/runtime dependent.
        pressure = 0.0
    return peak_allocated_gb, peak_reserved_gb, float(pressure)


def _training_gpu_snapshot(runtime_device: str) -> dict[str, object]:
    """Cheap per-batch GPU stats; avoid spawning nvidia-smi inside training loops."""

    if runtime_device != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "gpu_memory_allocated_gb": float(torch.cuda.memory_allocated() / (1024.0**3)),
        "gpu_memory_reserved_gb": float(torch.cuda.memory_reserved() / (1024.0**3)),
        "gpu_max_memory_allocated_gb": float(
            torch.cuda.max_memory_allocated() / (1024.0**3),
        ),
    }


def _maybe_compile_deep_model(
    *,
    config: "StageIDeepBaselineConfig",
    model: nn.Module,
    prepared,
    train_indices: np.ndarray,
    runtime_device: str,
) -> tuple[nn.Module, dict[str, object]]:
    if (
        config.torch_compile == "off"
        or runtime_device != "cuda"
        or not hasattr(torch, "compile")
    ):
        return model, {
            "compile_mode": str(config.torch_compile),
            "compile_status": "off",
            "compile_warmup_time_s": 0.0,
            "compile_fallback_reason": None,
        }
    started = time.monotonic()
    try:
        compiled = torch.compile(
            model,
            mode=None if config.torch_compile == "default" else config.torch_compile,
        )
        sample = train_indices[: min(8, len(train_indices))]
        modality_batch, mask_batch, time_batch, _target_tensor = get_train_batch(
            prepared,
            sample,
            device=runtime_device,
        )
        model.eval()
        compiled.eval()
        with torch.inference_mode():
            eager_logits = model(
                modality_batch,
                time_axis=time_batch,
                modality_masks=mask_batch,
            ).logits
            compiled_logits = compiled(
                modality_batch,
                time_axis=time_batch,
                modality_masks=mask_batch,
            ).logits
        if eager_logits is not None and compiled_logits is not None:
            max_diff = float((eager_logits - compiled_logits).abs().max().detach().cpu().item())
            if not np.isfinite(max_diff) or max_diff > 1e-3:
                return model, {
                    "compile_mode": str(config.torch_compile),
                    "compile_status": "fallback",
                    "compile_warmup_time_s": time.monotonic() - started,
                    "compile_fallback_reason": f"compile_sanity_diff={max_diff}",
                }
        return compiled, {
            "compile_mode": str(config.torch_compile),
            "compile_status": "enabled",
            "compile_warmup_time_s": time.monotonic() - started,
            "compile_fallback_reason": None,
        }
    except Exception as exc:  # pragma: no cover - host/runtime dependent.
        return model, {
            "compile_mode": str(config.torch_compile),
            "compile_status": "fallback",
            "compile_warmup_time_s": time.monotonic() - started,
            "compile_fallback_reason": type(exc).__name__ + ":" + str(exc),
        }


def _eval_amp(config: "StageIDeepBaselineConfig", runtime_device: str) -> AmpRuntime:
    requested = config.amp if config.amp_eval else "off"
    return resolve_amp_runtime(
        requested_mode=requested,
        device=runtime_device,
        grad_scaler=False,
    )


def _resolve_eval_batch_size(config: "StageIDeepBaselineConfig") -> int:
    if config.eval_batch_size is not None:
        return max(int(config.eval_batch_size), 1)
    if config.auto_batch_size and config.batch_size_candidates:
        return max(int(max(config.batch_size_candidates)), 1)
    return max(int(config.batch_size), 1)


def _forward_prepared_dataset(
    *,
    model: nn.Module,
    prepared,
    indices: np.ndarray,
    batch_size: int,
    amp: AmpRuntime,
) -> StageIDeepForwardResult:
    runtime_device = str(next(model.parameters()).device).split(":")[0]
    outputs: list[StageIDeepForwardResult] = []
    model.eval()
    with torch.inference_mode():
        for modality_batch, mask_batch, time_batch, _target_tensor in iter_eval_batches(
            prepared,
            np.asarray(indices, dtype=int),
            batch_size=batch_size,
            device=runtime_device,
        ):
            with amp.autocast(device=runtime_device):
                outputs.append(
                    model(
                        modality_batch,
                        time_axis=time_batch,
                        modality_masks=mask_batch,
                    )
                )
    return _concat_forward_outputs(outputs)


def _concat_forward_outputs(outputs: Sequence[StageIDeepForwardResult]) -> StageIDeepForwardResult:
    if not outputs:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return StageIDeepForwardResult(
            pooled_embedding=empty,
            sequence_embedding=torch.empty((0, 0, 0), dtype=torch.float32),
            attention_map=torch.empty((0, 0, 0), dtype=torch.float32),
            logits=empty,
        )
    logits = None
    if outputs[0].logits is not None:
        logits = torch.cat([output.logits for output in outputs if output.logits is not None], dim=0)
    return StageIDeepForwardResult(
        pooled_embedding=torch.cat([output.pooled_embedding.detach().float().cpu() for output in outputs], dim=0),
        sequence_embedding=torch.cat([output.sequence_embedding.detach().float().cpu() for output in outputs], dim=0),
        attention_map=torch.cat([output.attention_map.detach().float().cpu() for output in outputs], dim=0),
        logits=logits.detach().float().cpu() if logits is not None else None,
    )


def _sync_runtime(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _checkpoint_policy(value: str) -> str:
    normalized = str(value).strip().lower()
    choices = {"off", "last", "epoch_and_fold"}
    if normalized not in choices:
        raise ValueError(f"unsupported checkpoint_policy '{value}'; expected one of {tuple(sorted(choices))}")
    return normalized


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


def _compute_classification_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    target = target_tensor.to(dtype=torch.long)
    if isinstance(criterion, nn.CrossEntropyLoss):
        weight = criterion.weight
        if weight is not None and weight.dtype != logits.dtype:
            weight = weight.to(dtype=logits.dtype)
        return torch.nn.functional.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=criterion.ignore_index,
            reduction=criterion.reduction,
            label_smoothing=float(criterion.label_smoothing),
        )
    return criterion(logits, target)


def _compute_regression_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    target = target_tensor.to(dtype=logits.dtype).view_as(logits)
    return criterion(logits, target)


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
