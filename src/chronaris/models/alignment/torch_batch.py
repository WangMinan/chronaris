"""Torch-side tensor adapters for Stage E alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from chronaris.models.alignment.batching import AlignmentBatch, AlignmentStreamBatch


@dataclass(frozen=True, slots=True)
class TorchAlignmentStreamBatch:
    """A torch-ready stream batch derived from an AlignmentStreamBatch."""

    values: torch.Tensor
    mask: torch.Tensor
    feature_valid_mask: torch.Tensor
    offsets_ms: torch.Tensor
    offsets_s: torch.Tensor
    delta_t_s: torch.Tensor
    point_counts: torch.Tensor
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TorchAlignmentBatch:
    """A torch-ready dual-stream batch for Stage E prototype models."""

    sample_ids: tuple[str, ...]
    physiology: TorchAlignmentStreamBatch
    vehicle: TorchAlignmentStreamBatch


def build_torch_alignment_batch(
    batch: AlignmentBatch,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchAlignmentBatch:
    """Convert numpy-backed alignment batches into torch-backed tensors."""

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    physiology = _build_stream_batch(batch.physiology, device=resolved_device, dtype=dtype)
    vehicle = _build_stream_batch(batch.vehicle, device=resolved_device, dtype=dtype)

    return TorchAlignmentBatch(
        sample_ids=batch.sample_ids,
        physiology=physiology,
        vehicle=vehicle,
    )


def _build_stream_batch(
    stream: AlignmentStreamBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TorchAlignmentStreamBatch:
    point_mask_np = stream.mask.astype(bool, copy=False)
    feature_valid_mask_np = np.isfinite(stream.values) & point_mask_np[:, :, np.newaxis]
    values_np = np.nan_to_num(stream.values, nan=0.0, posinf=0.0, neginf=0.0)
    offsets_ms_np = stream.offsets_ms.astype(np.int64, copy=False)
    offsets_s_np = np.where(point_mask_np, offsets_ms_np.astype(np.float32) / 1000.0, 0.0)
    delta_t_s_np = _compute_delta_t_s(offsets_ms_np, point_mask_np)

    return TorchAlignmentStreamBatch(
        values=torch.as_tensor(values_np, dtype=dtype, device=device),
        mask=torch.as_tensor(point_mask_np, dtype=torch.bool, device=device),
        feature_valid_mask=torch.as_tensor(feature_valid_mask_np, dtype=torch.bool, device=device),
        offsets_ms=torch.as_tensor(offsets_ms_np, dtype=torch.int64, device=device),
        offsets_s=torch.as_tensor(offsets_s_np, dtype=dtype, device=device),
        delta_t_s=torch.as_tensor(delta_t_s_np, dtype=dtype, device=device),
        point_counts=torch.as_tensor(stream.point_counts, dtype=torch.int64, device=device),
        feature_names=stream.feature_names,
    )


def _compute_delta_t_s(offsets_ms: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta_t_s = np.zeros_like(offsets_ms, dtype=np.float32)
    batch_size, point_limit = offsets_ms.shape

    for sample_index in range(batch_size):
        previous_offset_ms: int | None = None
        for point_index in range(point_limit):
            if not mask[sample_index, point_index]:
                continue

            current_offset_ms = int(offsets_ms[sample_index, point_index])
            if previous_offset_ms is None:
                delta_t_s[sample_index, point_index] = 0.0
            else:
                delta_t_s[sample_index, point_index] = max(
                    0.0,
                    (current_offset_ms - previous_offset_ms) / 1000.0,
                )
            previous_offset_ms = current_offset_ms

    return delta_t_s
