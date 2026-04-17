"""Minimal loss functions for Stage E alignment prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chronaris.models.alignment.prototype import DualStreamPrototypeOutput, StreamPrototypeOutput
from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch


@dataclass(frozen=True, slots=True)
class ReconstructionLossBreakdown:
    """Per-stream reconstruction losses for the Stage E prototype."""

    physiology: torch.Tensor
    vehicle: torch.Tensor
    total: torch.Tensor


def masked_mean_squared_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only on positions marked valid."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if valid_mask.shape != predictions.shape:
        raise ValueError("valid_mask must match predictions shape.")

    weighted_mask = valid_mask.to(dtype=predictions.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return predictions.new_zeros(())

    squared_error = (predictions - targets) ** 2
    return (squared_error * weighted_mask).sum() / valid_count


def stream_reconstruction_loss(
    output: StreamPrototypeOutput,
    stream_batch: TorchAlignmentStreamBatch,
) -> torch.Tensor:
    """Compute reconstruction MSE for one stream."""

    valid_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)
    return masked_mean_squared_error(
        output.reconstructions,
        stream_batch.values,
        valid_mask,
    )


def dual_stream_reconstruction_loss(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
) -> ReconstructionLossBreakdown:
    """Compute reconstruction losses for both streams."""

    physiology = stream_reconstruction_loss(output.physiology, batch.physiology)
    vehicle = stream_reconstruction_loss(output.vehicle, batch.vehicle)
    return ReconstructionLossBreakdown(
        physiology=physiology,
        vehicle=vehicle,
        total=physiology + vehicle,
    )
