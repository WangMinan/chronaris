"""Minimal loss functions for Stage E alignment prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from chronaris.models.alignment.prototype import DualStreamPrototypeOutput, StreamPrototypeOutput
from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch


@dataclass(frozen=True, slots=True)
class ReconstructionLossBreakdown:
    """Per-stream reconstruction losses for the Stage E prototype."""

    physiology: torch.Tensor
    vehicle: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True, slots=True)
class AlignmentLossBreakdown:
    """Alignment loss summary on the shared reference grid."""

    alignment: torch.Tensor
    mode: str


@dataclass(frozen=True, slots=True)
class StageEObjectiveBreakdown:
    """Combined reconstruction + alignment objective summary."""

    physiology_reconstruction: torch.Tensor
    vehicle_reconstruction: torch.Tensor
    reconstruction_total: torch.Tensor
    alignment: torch.Tensor
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


def projection_alignment_loss(
    physiology_projection: torch.Tensor,
    vehicle_projection: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    mode: str = "mse",
) -> torch.Tensor:
    """Compare two projected trajectories on the shared reference grid."""

    if physiology_projection.shape != vehicle_projection.shape:
        raise ValueError("Projection tensors must have the same shape.")
    if physiology_projection.ndim != 3:
        raise ValueError("Projection tensors must have shape [B, R, P].")
    if valid_mask is None:
        valid_mask = torch.ones(
            physiology_projection.shape[:-1],
            dtype=torch.bool,
            device=physiology_projection.device,
        )
    if valid_mask.shape != physiology_projection.shape[:-1]:
        raise ValueError("valid_mask must have shape [B, R].")

    expanded_mask = valid_mask.unsqueeze(-1).expand_as(physiology_projection)
    if mode == "mse":
        return masked_mean_squared_error(
            physiology_projection,
            vehicle_projection,
            expanded_mask,
        )
    if mode == "cosine":
        cosine = F.cosine_similarity(physiology_projection, vehicle_projection, dim=-1)
        weighted_mask = valid_mask.to(dtype=physiology_projection.dtype)
        valid_count = weighted_mask.sum()
        if torch.is_nonzero(valid_count <= 0):
            return physiology_projection.new_zeros(())
        return ((1.0 - cosine) * weighted_mask).sum() / valid_count
    raise ValueError("mode must be either 'mse' or 'cosine'.")


def dual_stream_alignment_loss(
    output: DualStreamPrototypeOutput,
    *,
    mode: str = "mse",
) -> AlignmentLossBreakdown:
    """Compute the shared-reference alignment loss for the dual-stream prototype."""

    physiology_projection = output.physiology.reference_projected_states
    vehicle_projection = output.vehicle.reference_projected_states
    if physiology_projection is None or vehicle_projection is None:
        raise ValueError("Dual-stream alignment loss requires reference_projected_states for both streams.")

    if output.physiology.reference_offsets_s is None or output.vehicle.reference_offsets_s is None:
        raise ValueError("Dual-stream alignment loss requires reference_offsets_s for both streams.")
    if not torch.allclose(
        output.physiology.reference_offsets_s,
        output.vehicle.reference_offsets_s,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Physiology and vehicle reference grids must match before computing alignment loss.")

    return AlignmentLossBreakdown(
        alignment=projection_alignment_loss(
            physiology_projection,
            vehicle_projection,
            mode=mode,
        ),
        mode=mode,
    )


def build_stage_e_objective(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    alignment_mode: str = "mse",
    physiology_weight: float = 1.0,
    vehicle_weight: float = 1.0,
    alignment_weight: float = 1.0,
) -> StageEObjectiveBreakdown:
    """Combine reconstruction and alignment losses into one minimal Stage E objective."""

    if physiology_weight < 0 or vehicle_weight < 0 or alignment_weight < 0:
        raise ValueError("All objective weights must be non-negative.")

    reconstruction = dual_stream_reconstruction_loss(output, batch)
    alignment = dual_stream_alignment_loss(output, mode=alignment_mode)
    total = (
        (physiology_weight * reconstruction.physiology)
        + (vehicle_weight * reconstruction.vehicle)
        + (alignment_weight * alignment.alignment)
    )
    return StageEObjectiveBreakdown(
        physiology_reconstruction=reconstruction.physiology,
        vehicle_reconstruction=reconstruction.vehicle,
        reconstruction_total=reconstruction.total,
        alignment=alignment.alignment,
        total=total,
    )
