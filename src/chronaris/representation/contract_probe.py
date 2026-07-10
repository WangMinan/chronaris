"""Deterministic interface probe used only to validate six-method plumbing."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    FusionStreamBatch,
)


SIX_METHOD_NAMES = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)


class ContractProbeEncoder:
    """Exercise an adapter slot without claiming to implement the named model."""

    output_dim = FUSION_OUTPUT_DIM

    def __init__(
        self,
        *,
        method_name: str,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        if method_name not in SIX_METHOD_NAMES:
            raise ValueError(f"unsupported six-method contract slot: {method_name}")
        self.method_name = method_name
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        physiology = _causal_stream_summary(
            values=batch.physiology_values,
            timestamps=batch.physiology_timestamps_s,
            point_mask=batch.physiology_point_mask,
            feature_mask=batch.physiology_feature_mask,
            query=batch.query_timestamps_s,
        )
        vehicle = _causal_stream_summary(
            values=batch.vehicle_values,
            timestamps=batch.vehicle_timestamps_s,
            point_mask=batch.vehicle_point_mask,
            feature_mask=batch.vehicle_feature_mask,
            query=batch.query_timestamps_s,
        )
        time = batch.query_timestamps_s.to(torch.float32)
        base = torch.cat(
            [
                physiology,
                vehicle,
                torch.sin(time / 5.0).unsqueeze(-1),
                torch.cos(time / 5.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        projection = torch.from_numpy(
            _projection_matrix(self.method_name, base.shape[-1])
        ).to(device=base.device, dtype=base.dtype)
        sequence = torch.tanh(base @ projection)
        valid = (physiology[..., 3] > 0) | (vehicle[..., 3] > 0)
        counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (
            sequence * valid.unsqueeze(-1).to(sequence.dtype)
        ).sum(dim=1) / counts.to(sequence.dtype)
        return FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s,
            sequence_embedding=sequence,
            valid_mask=valid,
            pooled_embedding=pooled,
            method_name=self.method_name,
            fold_id=self.fold_id,
            checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes,
        )


def _causal_stream_summary(
    *,
    values: torch.Tensor,
    timestamps: torch.Tensor,
    point_mask: torch.Tensor,
    feature_mask: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    batch_size, query_count = query.shape
    summary = torch.zeros(
        (batch_size, query_count, 4),
        dtype=torch.float32,
        device=values.device,
    )
    for row in range(batch_size):
        valid_times = timestamps[row][point_mask[row]]
        valid_values = values[row][point_mask[row]].to(torch.float32)
        valid_features = feature_mask[row][point_mask[row]]
        if valid_times.numel() == 0:
            continue
        indices = torch.searchsorted(valid_times, query[row], right=True) - 1
        available = indices >= 0
        selected_index = indices.clamp_min(0)
        selected_values = valid_values[selected_index]
        selected_mask = valid_features[selected_index]
        counts = selected_mask.sum(dim=-1).clamp_min(1)
        mean = (
            selected_values * selected_mask.to(selected_values.dtype)
        ).sum(dim=-1) / counts
        centered = (selected_values - mean.unsqueeze(-1)) * selected_mask.to(
            selected_values.dtype
        )
        std = torch.sqrt((centered.square().sum(dim=-1) / counts).clamp_min(0))
        age = (query[row] - valid_times[selected_index]).to(torch.float32).clamp_min(0)
        ratio = selected_mask.to(torch.float32).mean(dim=-1)
        summary[row, :, 0] = torch.where(available, mean, 0)
        summary[row, :, 1] = torch.where(available, std, 0)
        summary[row, :, 2] = torch.where(available, age, 0)
        summary[row, :, 3] = torch.where(available, ratio, 0)
    return summary


def _projection_matrix(method_name: str, input_dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(method_name.encode()).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    return generator.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(input_dim, 1)),
        size=(input_dim, FUSION_OUTPUT_DIM),
    ).astype(np.float32)
