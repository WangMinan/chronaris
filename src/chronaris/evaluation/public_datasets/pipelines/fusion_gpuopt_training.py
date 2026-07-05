"""Training/profiling primitives for P28 GPU optimization."""

from __future__ import annotations

import json
import math
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from chronaris.evaluation.public_datasets.pipelines.deep_baseline_runtime import (
    _apply_classification_logit_adjustment,
    _safe_regression_fallback,
    _sanitize_classification_logits,
    _sanitize_regression_outputs,
    _inverse_transform_targets,
)
from chronaris.modeling.common.gpu_runtime import (
    AmpRuntime,
    choose_auto_batch_size,
    get_train_batch,
    gpu_runtime_snapshot,
    iter_eval_batches,
)


def run_profile_batch(
    *,
    config,
    candidate,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    amp,
    prepared,
    workload,
    ordered_modalities: Sequence[str],
    relative_indices: np.ndarray,
    runtime_device: str,
    profile_stage: str,
    epoch: int,
    batch_number: int,
    batch_count: int,
    batch_size: int,
    compile_info: Mapping[str, object],
    batch_attempts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], float]:
    total_started = time.monotonic()
    data_started = time.monotonic()
    global_batch = workload.train_global_indices[relative_indices]
    data_time = time.monotonic() - data_started
    copy_started = time.monotonic()
    modalities, masks, time_values, target_values = fetch_batch(
        prepared=prepared,
        ordered_modalities=ordered_modalities,
        global_batch=global_batch,
        runtime_device=runtime_device,
        profile_stage=profile_stage,
    )
    _sync(runtime_device)
    copy_time = time.monotonic() - copy_started
    optimizer.zero_grad(set_to_none=(profile_stage == "optimized"))
    forward_started = time.monotonic()
    with amp.autocast(device=runtime_device):
        output = model(modalities, time_axis=time_values, modality_masks=masks)
        logits = output.logits
    _sync(runtime_device)
    forward_time = time.monotonic() - forward_started
    if logits is None:
        raise ValueError("model returned no logits")
    loss_started = time.monotonic()
    if workload.task == "classification":
        loss = criterion(logits, target_values.to(dtype=torch.long))
    else:
        loss = criterion(logits, target_values.to(dtype=torch.float32).view(-1, 1))
    _sync(runtime_device)
    loss_time = time.monotonic() - loss_started
    backward_started = time.monotonic()
    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    _sync(runtime_device)
    backward_time = time.monotonic() - backward_started
    step_started = time.monotonic()
    if candidate.gradient_clip_max_norm is not None and float(candidate.gradient_clip_max_norm) > 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(candidate.gradient_clip_max_norm))
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    _sync(runtime_device)
    step_time = time.monotonic() - step_started
    loss_value = float(loss.detach().cpu().item())
    snapshot = gpu_runtime_snapshot()
    batch_total = time.monotonic() - total_started
    row = {
        "run_id": config.run_id,
        "base_p28_run_id": config.base_p28_run_id,
        "profile_stage": profile_stage,
        "dataset_id": workload.dataset_id,
        "candidate_id": candidate.candidate_id,
        "seed": 42,
        "track": workload.track,
        "evaluation_group": workload.evaluation_group,
        "split_group": workload.split_group,
        "fold_index": workload.fold_index,
        "fold_count": workload.fold_count,
        "epoch": epoch,
        "batch_index": batch_number,
        "batch_count": batch_count,
        "gpu_name": snapshot.get("gpu_name"),
        "cuda_available": snapshot.get("cuda_available"),
        "cuda_version": snapshot.get("cuda_version"),
        "torch_version": snapshot.get("torch_version"),
        "requested_device": config.device,
        "resolved_device": runtime_device,
        "batch_size": batch_size,
        "hidden_dim": candidate.hidden_dim,
        "layers": candidate.layers,
        "num_heads": candidate.num_heads,
        "dropout": candidate.dropout,
        "regression_loss": candidate.regression_loss,
        "target_transform": workload.target_transform["name"] if workload.target_transform else "none",
        "sequence_length": int(next(iter(prepared.modality_tensors.values())).shape[1]),
        "modality_shapes": json.dumps(
            {name: list(tensor.shape) for name, tensor in prepared.modality_tensors.items()}
        ),
        "train_sample_count": int(len(workload.train_global_indices)),
        "test_sample_count": int(len(workload.test_global_indices)),
        "sample_count": int(len(global_batch)),
        "data_time_s": data_time,
        "copy_time_s": copy_time,
        "forward_time_s": forward_time,
        "loss_time_s": loss_time,
        "backward_time_s": backward_time,
        "step_time_s": step_time,
        "evaluation_time_s": 0.0,
        "batch_total_time_s": batch_total,
        "samples_per_sec": float(len(global_batch) / batch_total) if batch_total > 0 else None,
        "loss": loss_value,
        "loss_finite": bool(np.isfinite(loss_value)),
        "tensor_cache_mode": prepared.tensor_cache_mode,
        "estimated_cache_gb": prepared.estimated_cache_gb,
        "cache_build_time_s": prepared.cache_build_time_s,
        "cache_fallback_reason": prepared.fallback_reason,
        "amp_mode": amp.resolved_mode,
        "auto_batch_attempts": json.dumps(list(batch_attempts)),
        "gpu_memory_allocated_gb": snapshot.get("gpu_memory_allocated_gb"),
        "gpu_memory_reserved_gb": snapshot.get("gpu_memory_reserved_gb"),
        "max_gpu_memory_allocated_gb": snapshot.get("gpu_max_memory_allocated_gb"),
        "nvidia_smi_util_pct": snapshot.get("gpu_util_pct"),
        **compile_info,
    }
    return row, loss_value


def fetch_batch(
    *,
    prepared,
    ordered_modalities: Sequence[str],
    global_batch: np.ndarray,
    runtime_device: str,
    profile_stage: str,
):
    if profile_stage == "baseline":
        modalities = {
            name: torch.as_tensor(
                prepared.modality_tensors[name].detach().cpu().numpy()[global_batch],
                dtype=torch.float32,
                device=runtime_device,
            )
            for name in ordered_modalities
        }
        masks = {
            name: torch.as_tensor(
                prepared.mask_tensors[name].detach().cpu().numpy()[global_batch],
                dtype=torch.float32,
                device=runtime_device,
            )
            for name in ordered_modalities
        }
        time_values = torch.as_tensor(
            prepared.time_tensor.detach().cpu().numpy()[global_batch],
            dtype=torch.float32,
            device=runtime_device,
        )
        targets = torch.as_tensor(
            prepared.target_tensor.detach().cpu().numpy()[global_batch],
            dtype=torch.float32,
            device=runtime_device,
        )
        return modalities, masks, time_values, targets
    return get_train_batch(prepared, global_batch, device=runtime_device)


def evaluate_workload(*, model, prepared, workload, runtime_device, batch_size, amp):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for modalities, masks, time_values, _targets in iter_eval_batches(
            prepared,
            workload.test_global_indices,
            batch_size=batch_size,
            device=runtime_device,
        ):
            with amp.autocast(device=runtime_device):
                output = model(modalities, time_axis=time_values, modality_masks=masks)
            outputs.append(output.logits.detach().float().cpu())
    logits = torch.cat(outputs, dim=0).numpy() if outputs else np.empty((0, 1), dtype=np.float32)
    return prediction_frame_from_logits(workload, logits)


def prediction_frame_from_logits(workload, logits: np.ndarray) -> pd.DataFrame:
    if workload.task == "classification":
        clean_logits = _sanitize_classification_logits(logits)
        adjusted = _apply_classification_logit_adjustment(
            clean_logits,
            train_targets=workload.train_targets,
            output_dim=len(workload.label_order or ()),
            sampling_policy=_sampling_policy(workload.dataset_id),
        )
        pred = np.asarray(
            [workload.label_order[index] for index in adjusted.argmax(axis=1)],
            dtype=int,
        )
        truth = workload.truth_values.astype(int)
    else:
        transformed, _nonfinite = _sanitize_regression_outputs(
            logits.reshape(-1),
            fallback_value=_safe_regression_fallback(workload.train_targets),
        )
        pred = _inverse_transform_targets(
            transformed,
            workload.target_transform or {"center": 0.0, "scale": 1.0},
        )
        truth = workload.truth_values.astype(np.float32)
    rows = pd.DataFrame({"y_true": truth, "y_pred": pred})
    rows["dataset_id"] = workload.dataset_id
    rows["evaluation_group"] = workload.evaluation_group
    rows["track"] = workload.track
    rows["split_group"] = workload.split_group
    return rows


def select_batch_size(
    *,
    config,
    candidate,
    model,
    criterion,
    prepared,
    workload,
    runtime_device,
    profile_stage,
    amp,
):
    if profile_stage == "baseline" or not config.auto_batch_size:
        return int(candidate.batch_size), [{"batch_size": int(candidate.batch_size), "status": "fixed"}]

    def _try(batch_size: int) -> None:
        relative = np.arange(min(batch_size, len(workload.train_global_indices)), dtype=int)
        global_batch = workload.train_global_indices[relative]
        modalities, masks, time_values, targets = fetch_batch(
            prepared=prepared,
            ordered_modalities=prepared.modality_tensors.keys(),
            global_batch=global_batch,
            runtime_device=runtime_device,
            profile_stage="optimized",
        )
        with amp.autocast(device=runtime_device):
            logits = model(modalities, time_axis=time_values, modality_masks=masks).logits
            loss = criterion(
                logits,
                targets.long() if workload.task == "classification" else targets.view(-1, 1),
            )
        loss.backward()
        model.zero_grad(set_to_none=True)
        _sync(runtime_device)

    return choose_auto_batch_size(
        candidates=config.batch_size_candidates,
        try_batch=_try,
        fallback=candidate.batch_size,
    )


def maybe_compile_model(*, config, model, prepared, workload, runtime_device):
    if config.torch_compile == "off" or runtime_device != "cuda" or not hasattr(torch, "compile"):
        return model, {
            "compile_mode": config.torch_compile,
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
        relative = np.arange(min(8, len(workload.train_global_indices)), dtype=int)
        modalities, masks, time_values, _targets = get_train_batch(
            prepared,
            workload.train_global_indices[relative],
            device=runtime_device,
        )
        with torch.inference_mode():
            eager_logits = model(modalities, time_axis=time_values, modality_masks=masks).logits
            compiled_logits = compiled(modalities, time_axis=time_values, modality_masks=masks).logits
        max_diff = float((eager_logits - compiled_logits).abs().max().detach().cpu().item())
        if not math.isfinite(max_diff) or max_diff > 1e-3:
            return model, {
                "compile_mode": config.torch_compile,
                "compile_status": "fallback",
                "compile_warmup_time_s": time.monotonic() - started,
                "compile_fallback_reason": f"compile_sanity_diff={max_diff}",
            }
        return compiled, {
            "compile_mode": config.torch_compile,
            "compile_status": "enabled",
            "compile_warmup_time_s": time.monotonic() - started,
            "compile_fallback_reason": None,
        }
    except Exception as exc:
        return model, {
            "compile_mode": config.torch_compile,
            "compile_status": "fallback",
            "compile_warmup_time_s": time.monotonic() - started,
            "compile_fallback_reason": type(exc).__name__ + ":" + str(exc),
        }


def _sampling_policy(dataset_id: str) -> str:
    return "balanced_class" if dataset_id == "nasa_csm" else "none"


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
