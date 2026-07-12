"""Lag-conditioned, private-retention objectives for Chronaris v2."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream
from chronaris.representation.contracts import DualStreamObservationBatch, masked_mean_pool


FUTURE_HORIZONS_S = (1.0, 3.0, 5.0, 10.0)
LAG_BIN_CENTERS_S = (1.0, 3.5, 7.5, 15.0, 25.0)


@dataclass(frozen=True, slots=True)
class ChronarisV2ObjectiveWeights:
    vehicle_private_retention: float
    physiology_private_retention: float
    future_physiology_delta: float
    lag_bin_classification: float
    clean_corruption_consistency: float
    physical_consistency: float


@dataclass(frozen=True, slots=True)
class ChronarisV2ObjectiveTargets:
    vehicle_query_values: torch.Tensor
    vehicle_query_mask: torch.Tensor
    physiology_query_values: torch.Tensor
    physiology_query_mask: torch.Tensor
    future_physiology_delta: torch.Tensor
    future_physiology_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ChronarisV2LossTerm:
    name: str
    raw_loss: torch.Tensor
    weight: float
    count: int

    @property
    def weighted_loss(self) -> torch.Tensor:
        return self.raw_loss * self.weight


@dataclass(frozen=True, slots=True)
class ChronarisV2ObjectiveOutput:
    total_loss: torch.Tensor
    terms: tuple[ChronarisV2LossTerm, ...]


class ChronarisV2ObjectiveHeads(nn.Module):
    """Task-independent heads attached to explicit v2 representation slices."""

    def __init__(
        self,
        *,
        physiology_feature_count: int,
        vehicle_feature_count: int,
    ) -> None:
        super().__init__()
        if physiology_feature_count <= 0 or vehicle_feature_count <= 0:
            raise ValueError("Chronaris v2 objective feature counts must be positive")
        self.physiology_feature_count = physiology_feature_count
        self.vehicle_feature_count = vehicle_feature_count
        self.vehicle_reconstruction = nn.Linear(24, vehicle_feature_count)
        self.physiology_reconstruction = nn.Linear(16, physiology_feature_count)
        self.future_physiology = nn.Linear(
            24,
            len(FUTURE_HORIZONS_S) * physiology_feature_count,
        )
        self.lag_classifier = nn.Sequential(
            nn.LayerNorm(24),
            nn.Linear(24, len(LAG_BIN_CENTERS_S)),
        )

    def forward(
        self,
        clean_encoding,
        targets: ChronarisV2ObjectiveTargets,
        *,
        weights: ChronarisV2ObjectiveWeights,
        corrupted_encoding=None,
        lag_encoding=None,
        lag_labels: torch.Tensor | None = None,
    ) -> ChronarisV2ObjectiveOutput:
        vehicle_prediction = self.vehicle_reconstruction(
            clean_encoding.vehicle_private
        )
        physiology_prediction = self.physiology_reconstruction(
            clean_encoding.physiology_private
        )
        future_prediction = self.future_physiology(
            clean_encoding.causal_shared
        ).reshape(
            *clean_encoding.causal_shared.shape[:2],
            len(FUTURE_HORIZONS_S),
            self.physiology_feature_count,
        )
        terms = [
            _masked_huber_term(
                "vehicle_private_retention",
                vehicle_prediction,
                targets.vehicle_query_values,
                targets.vehicle_query_mask,
                weights.vehicle_private_retention,
            ),
            _masked_huber_term(
                "physiology_private_retention",
                physiology_prediction,
                targets.physiology_query_values,
                targets.physiology_query_mask,
                weights.physiology_private_retention,
            ),
            _masked_huber_term(
                "future_physiology_delta",
                future_prediction,
                targets.future_physiology_delta,
                targets.future_physiology_mask,
                weights.future_physiology_delta,
            ),
        ]
        if corrupted_encoding is not None:
            shared_valid = (
                clean_encoding.modality_available_mask
                & corrupted_encoding.modality_available_mask
            )
            terms.append(
                _masked_huber_term(
                    "clean_corruption_consistency",
                    corrupted_encoding.sequence_embedding,
                    clean_encoding.sequence_embedding.detach(),
                    shared_valid.unsqueeze(-1).expand_as(
                        clean_encoding.sequence_embedding
                    ),
                    weights.clean_corruption_consistency,
                )
            )
        if lag_encoding is not None and lag_labels is not None:
            pooled = masked_mean_pool(
                lag_encoding.causal_shared,
                lag_encoding.modality_available_mask,
            )
            logits = self.lag_classifier(pooled)
            raw = F.cross_entropy(logits, lag_labels)
            terms.append(
                ChronarisV2LossTerm(
                    name="lag_bin_classification",
                    raw_loss=raw,
                    weight=weights.lag_bin_classification,
                    count=int(lag_labels.numel()),
                )
            )
        physics_components = tuple(
            component.raw_value
            for component in clean_encoding.physics_audit.components
            if component.active and component.raw_value is not None
        )
        physical_loss = (
            torch.stack(physics_components).mean()
            if physics_components
            else clean_encoding.sequence_embedding.sum() * 0.0
        )
        terms.append(
            ChronarisV2LossTerm(
                name="physical_consistency",
                raw_loss=physical_loss,
                weight=weights.physical_consistency,
                count=len(physics_components),
            )
        )
        active = tuple(term.weighted_loss for term in terms if term.weight > 0)
        total = (
            torch.stack(active).sum()
            if active
            else clean_encoding.sequence_embedding.sum() * 0.0
        )
        return ChronarisV2ObjectiveOutput(total_loss=total, terms=tuple(terms))


def chronaris_v2_objective_weight_schedule(
    epoch: int,
) -> ChronarisV2ObjectiveWeights:
    if epoch <= 0:
        raise ValueError("objective epoch must be one-based")
    if epoch <= 10:
        fraction = 0.0
    elif epoch >= 20:
        fraction = 1.0
    else:
        fraction = (epoch - 10) / 10.0
    return ChronarisV2ObjectiveWeights(
        vehicle_private_retention=0.30 * fraction,
        physiology_private_retention=0.30 * fraction,
        future_physiology_delta=0.50 * fraction,
        lag_bin_classification=0.20 * fraction,
        clean_corruption_consistency=0.20 * fraction,
        physical_consistency=0.10 * fraction,
    )


def build_chronaris_v2_objective_targets(
    clean_batch: DualStreamObservationBatch,
) -> ChronarisV2ObjectiveTargets:
    physiology = causal_query_stream(clean_batch, stream_name="physiology")
    vehicle = causal_query_stream(clean_batch, stream_name="vehicle")
    future_delta, future_mask = _future_delta_targets(
        physiology.values,
        physiology.feature_mask,
        clean_batch.query_timestamps_s.to(physiology.values.device),
    )
    return ChronarisV2ObjectiveTargets(
        vehicle_query_values=vehicle.values,
        vehicle_query_mask=vehicle.feature_mask,
        physiology_query_values=physiology.values,
        physiology_query_mask=physiology.feature_mask,
        future_physiology_delta=future_delta,
        future_physiology_mask=future_mask,
    )


def build_known_lag_batch(
    batch: DualStreamObservationBatch,
    lag_labels: torch.Tensor,
) -> DualStreamObservationBatch:
    if lag_labels.shape != (len(batch.sample_ids),):
        raise ValueError("lag labels must have shape [B]")
    if lag_labels.dtype != torch.long:
        raise ValueError("lag labels must use torch.long")
    if bool(((lag_labels < 0) | (lag_labels >= len(LAG_BIN_CENTERS_S))).any()):
        raise ValueError("lag labels are outside the five configured bins")
    centers = batch.vehicle_timestamps_s.new_tensor(LAG_BIN_CENTERS_S)
    shifts = centers.index_select(0, lag_labels.to(centers.device)).unsqueeze(-1)
    timestamps = batch.vehicle_timestamps_s + shifts
    return replace(batch, vehicle_timestamps_s=timestamps)


def _future_delta_targets(values, valid_mask, timestamps):
    batch_size, query_count, feature_count = values.shape
    deltas = values.new_zeros(
        (batch_size, query_count, len(FUTURE_HORIZONS_S), feature_count)
    )
    masks = torch.zeros_like(deltas, dtype=torch.bool)
    for horizon_index, horizon in enumerate(FUTURE_HORIZONS_S):
        target_times = timestamps + horizon
        indices = torch.searchsorted(timestamps, target_times, right=False)
        inside = indices < query_count
        safe_indices = indices.clamp(max=query_count - 1)
        future_values = torch.gather(
            values,
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, feature_count),
        )
        future_valid = torch.gather(
            valid_mask,
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, feature_count),
        )
        valid = valid_mask & future_valid & inside.unsqueeze(-1)
        deltas[:, :, horizon_index] = future_values - values
        masks[:, :, horizon_index] = valid
    return deltas, masks


def _masked_huber_term(name, prediction, target, mask, weight):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f"{name} target shape mismatch")
    count = int(mask.sum())
    raw = (
        F.smooth_l1_loss(prediction[mask], target[mask])
        if count
        else prediction.sum() * 0.0
    )
    return ChronarisV2LossTerm(
        name=name,
        raw_loss=raw,
        weight=float(weight),
        count=count,
    )
