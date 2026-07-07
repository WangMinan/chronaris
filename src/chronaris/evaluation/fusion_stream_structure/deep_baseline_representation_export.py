"""OOF pooled representation export for Dingxin E3 deep baselines."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    TASK_MANEUVER,
    TASK_RESPONSE,
    build_private_sequence_frame,
    derive_private_proxy_task_entries,
    load_aligned_private_records,
)
from chronaris.evaluation.dingxin.pipelines.thirdparty_comparison import (
    StageIPrivateThirdPartyComparisonConfig,
    _build_private_model,
    _sequence_arrays,
    _supervised_metrics,
    _train_supervised_model,
)
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_io import (
    build_four_method_e3_input_table,
    checkpoint_manifest_row,
    checkpoint_path,
    extract_pooled_embeddings_for_indices,
    fold_embedding_path,
    fold_status_row,
    json_default,
    modality_input_dims,
    model_hyperparameters,
    resolve_path,
    rows_from_pooled_embeddings,
    source_manifest_paths,
    utc_now,
    validate_checkpoint_manifest_frame,
    validate_deep_baseline_representation_frame,
    write_blocked_result,
    write_deep_baseline_representation_long_table,
    write_export_outputs,
    write_resume_command,
    write_training_protocol,
    load_deep_baseline_representation_long_table,
)
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_types import (
    DEFAULT_EXPORT_RUN_ID,
    DEFAULT_FOUR_METHOD_RUN_ID,
    DeepBaselineRepresentationExportConfig,
    DeepBaselineRepresentationExportResult,
)
from chronaris.modeling.common.gpu_runtime import iter_eval_batches, prepare_fold_tensors, resolve_amp_runtime
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def build_deep_baseline_sequence_frame(config: DeepBaselineRepresentationExportConfig) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Build the T2 sequence frame used for OOF pooled embedding export."""

    records = load_aligned_private_records(
        e_run_manifest_path=str(resolve_path(config.e_run_manifest_path)),
        f_run_manifest_path=str(resolve_path(config.f_run_manifest_path)),
    )
    payload = derive_private_proxy_task_entries(records)
    base = build_private_sequence_frame(tuple(payload["by_task"].get(config.task_name, ())), records, task_type=config.task_type)
    if base.empty:
        return base, {"records": records, "task_payload": payload}
    record_by_sample = records.set_index("sample_id", drop=False)
    enriched = base.copy()
    for column in ("raw_sample_id", "sortie_id", "view_id", "pilot_id", "window_index", "start_offset_ms", "end_offset_ms", "sample_partition"):
        enriched[column] = [record_by_sample.loc[str(row.sample_id)][column] for row in enriched.itertuples(index=False)]
    enriched["source_task_name"] = config.task_name
    enriched["source_task_type"] = config.task_type
    enriched["source_split_strategy"] = config.split_strategy
    enriched["source_seed"] = int(config.seed)
    enriched = _attach_posthoc_metadata(enriched, payload)
    return enriched.reset_index(drop=True), {"records": records, "task_payload": payload, "source_record_count": int(len(records))}


def train_fold_and_export_embeddings(
    *,
    config: DeepBaselineRepresentationExportConfig,
    thirdparty_config: StageIPrivateThirdPartyComparisonConfig,
    frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    fold_index: int,
    fold_group: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    runtime_device: str,
    run_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Train one fold and emit held-out pooled embeddings for that fold."""

    checkpoint = checkpoint_path(run_root, model_name, config.seed, fold_index)
    fold_embedding = fold_embedding_path(run_root, model_name, config.seed, fold_index)
    dims = modality_input_dims(arrays)
    if config.skip_completed and checkpoint.exists() and fold_embedding.exists():
        fold_frame = pd.read_csv(fold_embedding)
        return (
            fold_frame.to_dict(orient="records"),
            [],
            checkpoint_manifest_row(config, model_name, fold_index, fold_group, checkpoint, train_idx, test_idx, frame, runtime_device, dims, "completed_skipped"),
            [fold_status_row(config, model_name, fold_index, fold_group, train_idx, test_idx, "completed_skipped", checkpoint=checkpoint, embedding_row_count=len(fold_frame))],
        )

    seed_torch(int(config.seed) + int(fold_index), device=runtime_device)
    model = _build_private_model(model_name, config.task_type, thirdparty_config, runtime_device, arrays)
    prepared = prepare_fold_tensors(
        modality_arrays=arrays,
        modality_masks=masks,
        time_axis=time_axis,
        ordered_modalities=("physiology", "vehicle"),
        train_indices=train_idx,
        targets=labels.astype(np.float32),
        requested_mode=config.tensor_cache,
        device=runtime_device,
        max_cache_gb=config.max_cache_gb,
        pin_memory=config.pin_memory,
        non_blocking_copy=config.non_blocking_copy,
    )
    started = time.monotonic()
    curves = _train_supervised_model(
        model=model,
        prepared=prepared,
        labels=labels,
        train_indices=train_idx,
        task_type=config.task_type,
        config=thirdparty_config,
        seed=int(config.seed) + int(fold_index),
        context={
            "task_name": config.task_name,
            "task_type": config.task_type,
            "model_name": model_name,
            "split_strategy": config.split_strategy,
            "seed": int(config.seed),
            "fold_index": int(fold_index),
            "fold_group": str(fold_group),
            "representation_family": config.representation_family,
        },
    )
    embeddings = extract_pooled_embeddings_for_indices(
        model,
        prepared,
        test_idx,
        batch_size=thirdparty_config.eval_batch_size or thirdparty_config.batch_size,
        amp_mode=thirdparty_config.amp if thirdparty_config.amp_eval else "off",
    )
    metrics = _supervised_metrics(labels[test_idx], _predict_values(model, prepared, test_idx, labels, thirdparty_config), config.task_type)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format": "chronaris_deep_baseline_representation_v1",
            "model_state_dict": model.state_dict(),
            "task_name": config.task_name,
            "task_type": config.task_type,
            "model_name": model_name,
            "split_strategy": config.split_strategy,
            "seed": int(config.seed),
            "fold_index": int(fold_index),
            "fold_group": str(fold_group),
            "representation_family": config.representation_family,
            "model_hyperparameters": model_hyperparameters(config),
            "modality_input_dims": dims,
            "source_manifest_paths": source_manifest_paths(config),
            "train_sample_ids": frame.iloc[train_idx]["sample_id"].astype(str).to_list(),
            "test_sample_ids": frame.iloc[test_idx]["sample_id"].astype(str).to_list(),
            "prepared_tensors": prepared.to_jsonable(),
            "epoch_count": int(config.epochs),
            "saved_at_utc": utc_now(),
        },
        checkpoint,
    )
    rows = rows_from_pooled_embeddings(
        frame.iloc[test_idx],
        embeddings,
        config=config,
        model_name=model_name,
        fold_index=fold_index,
        fold_group=fold_group,
        checkpoint_path=checkpoint,
        train_sample_count=len(train_idx),
        test_sample_count=len(test_idx),
    )
    fold_embedding.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(fold_embedding, index=False)
    return (
        rows,
        curves,
        checkpoint_manifest_row(config, model_name, fold_index, fold_group, checkpoint, train_idx, test_idx, frame, runtime_device, dims, "completed"),
        [fold_status_row(config, model_name, fold_index, fold_group, train_idx, test_idx, "completed", checkpoint=checkpoint, embedding_row_count=len(rows), elapsed_s=time.monotonic() - started, metrics=metrics)],
    )


def export_oof_pooled_embeddings(config: DeepBaselineRepresentationExportConfig) -> DeepBaselineRepresentationExportResult:
    """Run the configured Dingxin deep baseline OOF representation export."""

    run_root = resolve_path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_resume_command(run_root, config)
    write_training_protocol(run_root, config)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="deep_baseline_representation_export",
        logger=LOGGER,
        initial_progress={"artifact_root": str(run_root), "representation_family": config.representation_family, "training_invoked": True, "confirmed_metrics_changed": False},
    ) as progress:
        try:
            runtime_device = resolve_torch_device_name(config.device)
            if config.require_cuda and runtime_device != "cuda":
                return write_blocked_result(config, run_root, reason="cuda_unavailable", details={"resolved_device": runtime_device})
            frame, build_context = build_deep_baseline_sequence_frame(config)
            if frame.empty:
                return write_blocked_result(config, run_root, reason="empty_sequence_frame")
            arrays, masks, time_axis, labels = _sequence_arrays(frame)
            groups = tuple(sorted(frame["view_id"].astype(str).unique()))
            if config.max_folds is not None:
                groups = groups[: max(int(config.max_folds), 0)]
            expected = len(tuple(config.models)) * len(groups)
            thirdparty_config = _thirdparty_config(config)
            progress.update("sequence_frame_ready", sample_count=int(len(frame)), fold_count=int(len(groups)), expected_fold_count=int(expected), runtime_device=runtime_device)
            embeddings: list[dict[str, object]] = []
            curves: list[dict[str, object]] = []
            checkpoints: list[dict[str, object]] = []
            statuses: list[dict[str, object]] = []
            for model_name in tuple(config.models):
                for fold_index, fold_group in enumerate(groups, start=1):
                    train_idx = frame.index[frame["view_id"].astype(str) != fold_group].to_numpy(dtype=int)
                    test_idx = frame.index[frame["view_id"].astype(str) == fold_group].to_numpy(dtype=int)
                    progress.update("fold_started", model_name=model_name, fold_index=int(fold_index), fold_group=str(fold_group), train_count=int(len(train_idx)), test_count=int(len(test_idx)))
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        statuses.append(fold_status_row(config, model_name, fold_index, fold_group, train_idx, test_idx, "skipped", reason="empty_train_or_test"))
                        continue
                    try:
                        fold_embeddings, fold_curves, checkpoint_row, fold_status = train_fold_and_export_embeddings(
                            config=config,
                            thirdparty_config=thirdparty_config,
                            frame=frame,
                            arrays=arrays,
                            masks=masks,
                            time_axis=time_axis,
                            labels=labels,
                            model_name=model_name,
                            fold_index=fold_index,
                            fold_group=fold_group,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            runtime_device=runtime_device,
                            run_root=run_root,
                        )
                    except Exception as exc:  # pragma: no cover - integration/runtime path.
                        fold_embeddings, fold_curves, checkpoint_row = [], [], {}
                        fold_status = [fold_status_row(config, model_name, fold_index, fold_group, train_idx, test_idx, "failed", reason=type(exc).__name__ + ":" + str(exc), traceback_text=traceback.format_exc())]
                        if not config.allow_partial:
                            raise
                    embeddings.extend(fold_embeddings)
                    curves.extend(fold_curves)
                    if checkpoint_row:
                        checkpoints.append(checkpoint_row)
                    statuses.extend(fold_status)
                    progress.update("fold_finished", model_name=model_name, fold_index=int(fold_index), fold_group=str(fold_group), status=str(fold_status[-1].get("status") if fold_status else "unknown"))
            return write_export_outputs(
                config=config,
                run_root=run_root,
                runtime_device=runtime_device,
                sequence_frame=frame,
                build_context=build_context,
                embedding_rows=embeddings,
                curve_rows=curves,
                checkpoint_rows=checkpoints,
                fold_status_rows=statuses,
                expected_fold_count=expected,
            )
        except Exception as exc:  # pragma: no cover - host/data dependent.
            return write_blocked_result(config, run_root, reason=type(exc).__name__ + ":" + str(exc), details={"traceback": traceback.format_exc()})


def _thirdparty_config(config: DeepBaselineRepresentationExportConfig) -> StageIPrivateThirdPartyComparisonConfig:
    return StageIPrivateThirdPartyComparisonConfig(
        run_id=config.run_id,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
        output_root=config.output_root,
        report_root=config.output_root,
        models=tuple(config.models),
        seeds=(int(config.seed),),
        split_strategy=(config.split_strategy,),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        hidden_dim=int(config.hidden_dim),
        num_heads=int(config.num_heads),
        layers=int(config.layers),
        dropout=float(config.dropout),
        weight_decay=float(config.weight_decay),
        grad_clip_norm=float(config.grad_clip_norm),
        device=config.device,
        require_cuda=bool(config.require_cuda),
        resume=bool(config.resume),
        skip_completed=bool(config.skip_completed),
        allow_partial=bool(config.allow_partial),
        heartbeat_seconds=float(config.heartbeat_seconds),
        batch_log_interval=int(config.batch_log_interval),
        tensor_cache=config.tensor_cache,
        max_cache_gb=float(config.max_cache_gb),
        pin_memory=bool(config.pin_memory),
        non_blocking_copy=bool(config.non_blocking_copy),
        auto_batch_size=bool(config.auto_batch_size),
        batch_size_candidates=tuple(int(value) for value in config.batch_size_candidates),
        amp=config.amp,
        grad_scaler=bool(config.grad_scaler),
        amp_eval=bool(config.amp_eval),
        torch_compile=config.torch_compile,
        profile_gpu=bool(config.profile_gpu),
        eval_batch_size=config.eval_batch_size,
        num_workers=int(config.num_workers),
        checkpoint_policy=config.checkpoint_policy,
    )


def _attach_posthoc_metadata(frame: pd.DataFrame, payload: Mapping[str, object]) -> pd.DataFrame:
    maneuver_by_sample = {entry.sample_id: entry.label_value for entry in payload["by_task"].get(TASK_MANEUVER, ())}
    response_by_sample = {entry.sample_id: entry.label_value for entry in payload["by_task"].get(TASK_RESPONSE, ()) if entry.label_value is not None}
    response_values = np.asarray([float(value) for value in response_by_sample.values()], dtype=float)
    threshold = float(np.nanquantile(response_values, 0.75)) if response_values.size else None
    enriched = frame.copy()
    enriched["maneuver_proxy_label"] = [maneuver_by_sample.get(sample_id) for sample_id in enriched["sample_id"]]
    enriched["physio_fluctuation_interval"] = [bool(threshold is not None and response_by_sample.get(sample_id) is not None and float(response_by_sample[sample_id]) >= threshold) for sample_id in enriched["sample_id"]]
    return enriched


def _predict_values(model, prepared, indices: np.ndarray, labels: np.ndarray, config: StageIPrivateThirdPartyComparisonConfig) -> np.ndarray:
    device_name = next(model.parameters()).device.type
    amp = resolve_amp_runtime(requested_mode=config.amp if config.amp_eval else "off", device=device_name, grad_scaler=False)
    logits = []
    model.eval()
    with torch.inference_mode():
        for modality_batch, mask_batch, time_batch, _target_tensor in iter_eval_batches(prepared, np.asarray(indices, dtype=int), batch_size=config.eval_batch_size or config.batch_size, device=device_name):
            with amp.autocast(device=device_name):
                logits.append(model(modality_batch, time_axis=time_batch, modality_masks=mask_batch).logits.detach().float().cpu())
    values = torch.cat(logits, dim=0).numpy() if logits else np.empty((0, 1), dtype=np.float32)
    return np.nan_to_num(values.reshape(-1), nan=float(np.nanmean(labels)) if len(labels) else 0.0)


__all__ = [
    "DEFAULT_EXPORT_RUN_ID",
    "DEFAULT_FOUR_METHOD_RUN_ID",
    "DeepBaselineRepresentationExportConfig",
    "DeepBaselineRepresentationExportResult",
    "build_deep_baseline_sequence_frame",
    "build_four_method_e3_input_table",
    "export_oof_pooled_embeddings",
    "extract_pooled_embeddings_for_indices",
    "load_deep_baseline_representation_long_table",
    "rows_from_pooled_embeddings",
    "train_fold_and_export_embeddings",
    "validate_checkpoint_manifest_frame",
    "validate_deep_baseline_representation_frame",
    "write_deep_baseline_representation_long_table",
]
