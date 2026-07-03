"""GPU runtime helpers for Stage I public fusion profiling."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import subprocess
import time
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import torch


TENSOR_CACHE_MODES = ("auto", "cuda", "cpu", "off")
AMP_MODES = ("off", "fp16", "bf16")


@dataclass(slots=True)
class FoldNormalizationStats:
    """Train-fold-only normalization statistics for each modality."""

    modality_mean: dict[str, np.ndarray]
    modality_std: dict[str, np.ndarray]
    train_indices: np.ndarray

    def to_jsonable(self) -> dict[str, object]:
        return {
            "train_index_count": int(len(self.train_indices)),
            "modalities": {
                name: {
                    "mean": self.modality_mean[name].astype(float).tolist(),
                    "std": self.modality_std[name].astype(float).tolist(),
                }
                for name in sorted(self.modality_mean)
            },
        }


@dataclass(slots=True)
class FoldPreparedTensors:
    """Normalized fold tensors cached on CUDA or pinned CPU."""

    tensor_cache_mode: str
    requested_mode: str
    device: str
    modality_tensors: dict[str, torch.Tensor]
    mask_tensors: dict[str, torch.Tensor]
    time_tensor: torch.Tensor
    target_tensor: torch.Tensor | None
    normalization_stats: FoldNormalizationStats
    estimated_cache_gb: float
    actual_cache_gb: float
    cache_build_time_s: float
    fallback_reason: str | None = None
    non_blocking_copy: bool = True
    cache_hit_count: int = 0

    def to_jsonable(self) -> dict[str, object]:
        return {
            "tensor_cache_mode": self.tensor_cache_mode,
            "requested_mode": self.requested_mode,
            "device": self.device,
            "estimated_cache_gb": self.estimated_cache_gb,
            "actual_cache_gb": self.actual_cache_gb,
            "cache_build_time_s": self.cache_build_time_s,
            "cache_hit_count": int(self.cache_hit_count),
            "cache_fallback_reason": self.fallback_reason,
            "normalization": self.normalization_stats.to_jsonable(),
        }


@dataclass(frozen=True, slots=True)
class AmpRuntime:
    requested_mode: str
    resolved_mode: str
    enabled: bool
    dtype: torch.dtype | None
    use_grad_scaler: bool
    fallback_reason: str | None = None

    def autocast(self, *, device: str):
        if not self.enabled or self.dtype is None or device != "cuda":
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.dtype)


def compute_fold_normalization_stats(
    *,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    ordered_modalities: Sequence[str],
    train_indices: np.ndarray,
) -> FoldNormalizationStats:
    """Compute modality stats using train fold indices only."""

    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    train_indices = np.asarray(train_indices, dtype=int)
    for modality_name in ordered_modalities:
        values = modality_arrays[modality_name].astype(np.float32, copy=False)
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
        ).astype(np.float32)
        centered = np.where(valid, train_values - mean.reshape(1, 1, -1), 0.0)
        variance = np.where(
            count > 0,
            np.square(centered).sum(axis=(0, 1)) / safe_count,
            1.0,
        ).astype(np.float32)
        std = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
        means[modality_name] = mean
        stds[modality_name] = std
    return FoldNormalizationStats(
        modality_mean=means,
        modality_std=stds,
        train_indices=train_indices.copy(),
    )


def apply_fold_normalization(
    *,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    ordered_modalities: Sequence[str],
    stats: FoldNormalizationStats,
) -> dict[str, np.ndarray]:
    """Apply train-fold stats to the full fold-visible dataset."""

    normalized: dict[str, np.ndarray] = {}
    for modality_name in ordered_modalities:
        values = modality_arrays[modality_name].astype(np.float32, copy=False)
        mask = modality_masks[modality_name].astype(bool, copy=False)
        mean = stats.modality_mean[modality_name].reshape(1, 1, -1)
        std = stats.modality_std[modality_name].reshape(1, 1, -1)
        transformed = (values - mean) / std
        normalized[modality_name] = np.where(mask[:, :, None], transformed, 0.0).astype(
            np.float32,
        )
    return normalized


def estimate_tensor_cache_bytes(
    *,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    ordered_modalities: Sequence[str],
    time_axis: np.ndarray,
    targets: np.ndarray | None = None,
) -> int:
    total = int(np.asarray(time_axis, dtype=np.float32).nbytes)
    for modality_name in ordered_modalities:
        total += int(np.asarray(modality_arrays[modality_name], dtype=np.float32).nbytes)
        total += int(np.asarray(modality_masks[modality_name], dtype=np.float32).nbytes)
    if targets is not None:
        total += int(np.asarray(targets, dtype=np.float32).nbytes)
    return total


def prepare_fold_tensors(
    *,
    modality_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    ordered_modalities: Sequence[str],
    train_indices: np.ndarray,
    targets: np.ndarray | None,
    requested_mode: str,
    device: str,
    max_cache_gb: float = 18.0,
    pin_memory: bool = True,
    non_blocking_copy: bool = True,
) -> FoldPreparedTensors:
    """Normalize once per fold and cache tensors on CUDA or pinned CPU."""

    requested_mode = _validate_choice(requested_mode, TENSOR_CACHE_MODES, "tensor_cache")
    build_started = time.monotonic()
    stats = compute_fold_normalization_stats(
        modality_arrays=modality_arrays,
        modality_masks=modality_masks,
        ordered_modalities=ordered_modalities,
        train_indices=train_indices,
    )
    normalized_arrays = apply_fold_normalization(
        modality_arrays=modality_arrays,
        modality_masks=modality_masks,
        ordered_modalities=ordered_modalities,
        stats=stats,
    )
    estimated_bytes = estimate_tensor_cache_bytes(
        modality_arrays=normalized_arrays,
        modality_masks=modality_masks,
        ordered_modalities=ordered_modalities,
        time_axis=time_axis,
        targets=targets,
    )
    estimated_gb = bytes_to_gb(estimated_bytes)
    target_mode = _resolve_tensor_cache_mode(
        requested_mode=requested_mode,
        device=device,
        estimated_gb=estimated_gb,
        max_cache_gb=max_cache_gb,
    )
    try:
        prepared = _build_prepared_tensors(
            normalized_arrays=normalized_arrays,
            modality_masks=modality_masks,
            time_axis=time_axis,
            ordered_modalities=ordered_modalities,
            targets=targets,
            mode=target_mode,
            device=device,
            requested_mode=requested_mode,
            estimated_gb=estimated_gb,
            stats=stats,
            started_at=build_started,
            pin_memory=pin_memory,
            non_blocking_copy=non_blocking_copy,
            fallback_reason=None,
        )
    except RuntimeError as exc:
        if target_mode != "cuda" or not is_cuda_oom(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        prepared = _build_prepared_tensors(
            normalized_arrays=normalized_arrays,
            modality_masks=modality_masks,
            time_axis=time_axis,
            ordered_modalities=ordered_modalities,
            targets=targets,
            mode="cpu",
            device=device,
            requested_mode=requested_mode,
            estimated_gb=estimated_gb,
            stats=stats,
            started_at=build_started,
            pin_memory=pin_memory,
            non_blocking_copy=non_blocking_copy,
            fallback_reason=f"cuda_oom:{str(exc).splitlines()[0]}",
        )
    return prepared


def get_train_batch(
    prepared: FoldPreparedTensors,
    indices: np.ndarray | torch.Tensor,
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    prepared.cache_hit_count += 1
    index_tensor = _index_tensor(indices, source_device=prepared.time_tensor.device)
    modalities = {
        name: _move_if_needed(tensor.index_select(0, index_tensor), device, prepared)
        for name, tensor in prepared.modality_tensors.items()
    }
    masks = {
        name: _move_if_needed(tensor.index_select(0, index_tensor), device, prepared)
        for name, tensor in prepared.mask_tensors.items()
    }
    time_values = _move_if_needed(
        prepared.time_tensor.index_select(0, index_tensor),
        device,
        prepared,
    )
    targets = (
        _move_if_needed(prepared.target_tensor.index_select(0, index_tensor), device, prepared)
        if prepared.target_tensor is not None
        else None
    )
    return modalities, masks, time_values, targets


def iter_eval_batches(
    prepared: FoldPreparedTensors,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]]:
    for start in range(0, len(indices), max(int(batch_size), 1)):
        yield get_train_batch(
            prepared,
            np.asarray(indices[start : start + batch_size], dtype=int),
            device=device,
        )


def resolve_amp_runtime(
    *,
    requested_mode: str,
    device: str,
    grad_scaler: bool = True,
) -> AmpRuntime:
    requested_mode = _validate_choice(requested_mode, AMP_MODES, "amp")
    if requested_mode == "off" or device != "cuda" or not torch.cuda.is_available():
        reason = None if requested_mode == "off" else "amp_requires_cuda"
        return AmpRuntime(requested_mode, "off", False, None, False, reason)
    if requested_mode == "bf16":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return AmpRuntime("bf16", "bf16", True, torch.bfloat16, False, None)
        return AmpRuntime(
            "bf16",
            "fp16",
            True,
            torch.float16,
            bool(grad_scaler),
            "bf16_not_supported_fallback_fp16",
        )
    return AmpRuntime("fp16", "fp16", True, torch.float16, bool(grad_scaler), None)


def make_grad_scaler(amp: AmpRuntime, *, device: str):
    enabled = amp.enabled and amp.use_grad_scaler and device == "cuda"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def choose_auto_batch_size(
    *,
    candidates: Sequence[int],
    try_batch: Callable[[int], object],
    fallback: int = 128,
) -> tuple[int, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            try_batch(int(candidate))
            attempts.append({"batch_size": int(candidate), "status": "ok"})
            return int(candidate), attempts
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            attempts.append(
                {
                    "batch_size": int(candidate),
                    "status": "oom",
                    "reason": str(exc).splitlines()[0],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    attempts.append({"batch_size": int(fallback), "status": "fallback"})
    return int(fallback), attempts


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def gpu_runtime_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_util_pct": None,
        "gpu_memory_used_gb": None,
        "gpu_memory_total_gb": None,
        "gpu_snapshot_reason": None,
    }
    if torch.cuda.is_available():
        snapshot.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated_gb": bytes_to_gb(torch.cuda.memory_allocated()),
                "gpu_memory_reserved_gb": bytes_to_gb(torch.cuda.memory_reserved()),
                "gpu_max_memory_allocated_gb": bytes_to_gb(
                    torch.cuda.max_memory_allocated(),
                ),
            }
        )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
        if output:
            first = output.splitlines()[0]
            name, total, used, util = [part.strip() for part in first.split(",")[:4]]
            snapshot.update(
                {
                    "gpu_name": name,
                    "gpu_memory_total_gb": float(total) / 1024.0,
                    "gpu_memory_used_gb": float(used) / 1024.0,
                    "gpu_util_pct": float(util),
                }
            )
    except Exception as exc:  # pragma: no cover - depends on host nvidia-smi.
        snapshot["gpu_snapshot_reason"] = type(exc).__name__ + ":" + str(exc)
    return snapshot


def completed_profile_key(
    *,
    dataset_id: str,
    candidate_id: str,
    seed: int,
    track: str,
    evaluation_group: str,
    split_group: str,
) -> str:
    return "|".join(
        [
            dataset_id,
            candidate_id,
            str(int(seed)),
            track,
            evaluation_group,
            split_group,
        ]
    )


def remaining_profile_keys(
    requested_keys: Sequence[str],
    completed_keys: set[str],
    *,
    skip_completed: bool,
) -> list[str]:
    if not skip_completed:
        return list(requested_keys)
    return [key for key in requested_keys if key not in completed_keys]


def validate_optimization_summary_schema(summary: Mapping[str, object]) -> tuple[bool, list[str]]:
    required = {
        "run_id",
        "base_p28_run_id",
        "status",
        "generated_at_utc",
        "gpu_name",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "optimization_enabled",
        "throughput",
        "timing",
        "memory",
        "utilization",
        "fallbacks",
        "resume_command",
    }
    missing = sorted(key for key in required if key not in summary)
    return not missing, missing


def bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024.0 ** 3)


def json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _resolve_tensor_cache_mode(
    *,
    requested_mode: str,
    device: str,
    estimated_gb: float,
    max_cache_gb: float,
) -> str:
    if requested_mode == "off":
        return "off"
    if requested_mode == "cpu":
        return "cpu"
    if requested_mode == "cuda":
        return "cuda"
    if device == "cuda" and torch.cuda.is_available() and estimated_gb <= float(max_cache_gb):
        return "cuda"
    return "cpu"


def _build_prepared_tensors(
    *,
    normalized_arrays: Mapping[str, np.ndarray],
    modality_masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    ordered_modalities: Sequence[str],
    targets: np.ndarray | None,
    mode: str,
    device: str,
    requested_mode: str,
    estimated_gb: float,
    stats: FoldNormalizationStats,
    started_at: float,
    pin_memory: bool,
    non_blocking_copy: bool,
    fallback_reason: str | None,
) -> FoldPreparedTensors:
    tensor_device = "cuda" if mode == "cuda" else "cpu"
    modality_tensors = {
        name: _as_cache_tensor(
            normalized_arrays[name],
            tensor_device=tensor_device,
            pin_memory=pin_memory,
        )
        for name in ordered_modalities
    }
    mask_tensors = {
        name: _as_cache_tensor(
            modality_masks[name],
            tensor_device=tensor_device,
            pin_memory=pin_memory,
        )
        for name in ordered_modalities
    }
    time_tensor = _as_cache_tensor(time_axis, tensor_device=tensor_device, pin_memory=pin_memory)
    target_tensor = (
        _as_cache_tensor(targets, tensor_device=tensor_device, pin_memory=pin_memory)
        if targets is not None
        else None
    )
    actual_bytes = int(time_tensor.element_size() * time_tensor.nelement())
    for tensor in list(modality_tensors.values()) + list(mask_tensors.values()):
        actual_bytes += int(tensor.element_size() * tensor.nelement())
    if target_tensor is not None:
        actual_bytes += int(target_tensor.element_size() * target_tensor.nelement())
    return FoldPreparedTensors(
        tensor_cache_mode=mode,
        requested_mode=requested_mode,
        device=device,
        modality_tensors=modality_tensors,
        mask_tensors=mask_tensors,
        time_tensor=time_tensor,
        target_tensor=target_tensor,
        normalization_stats=stats,
        estimated_cache_gb=estimated_gb,
        actual_cache_gb=bytes_to_gb(actual_bytes),
        cache_build_time_s=time.monotonic() - started_at,
        fallback_reason=fallback_reason,
        non_blocking_copy=non_blocking_copy,
    )


def _as_cache_tensor(
    values: np.ndarray,
    *,
    tensor_device: str,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)
    if tensor_device == "cuda":
        return tensor.to(device="cuda")
    if pin_memory and torch.cuda.is_available():
        return tensor.pin_memory()
    return tensor


def _move_if_needed(
    tensor: torch.Tensor,
    device: str,
    prepared: FoldPreparedTensors,
) -> torch.Tensor:
    if str(tensor.device).startswith(device):
        return tensor
    return tensor.to(device=device, non_blocking=prepared.non_blocking_copy)


def _index_tensor(indices: np.ndarray | torch.Tensor, *, source_device: torch.device) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        tensor = indices.to(dtype=torch.long)
        return tensor.to(device=source_device) if tensor.device != source_device else tensor
    return torch.as_tensor(indices, dtype=torch.long, device=source_device)


def _validate_choice(value: str, choices: Sequence[str], name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} '{value}'; expected one of {tuple(choices)}")
    return normalized
