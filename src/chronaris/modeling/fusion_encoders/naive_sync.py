"""Leakage-safe causal time synchronization with train-fold-only PCA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from chronaris.modeling.fusion_encoders.causal_query import (
    CausalQueryStream,
    causal_query_stream,
)
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    FusionStreamBatch,
    RepresentationContractError,
)
from chronaris.representation.normalization import (
    TrainOnlyPCAProjector,
    TrainOnlyRobustNormalizer,
)


@dataclass(frozen=True, slots=True)
class NaiveTimeSyncConfig:
    output_dim: int = FUSION_OUTPUT_DIM
    maximum_age_s: float = 30.0

    def __post_init__(self) -> None:
        if self.output_dim != FUSION_OUTPUT_DIM or self.maximum_age_s <= 0:
            raise ValueError("naive time-sync config is invalid")


class NaiveTimeSyncEncoder:
    """Causal forward-fill followed by unsupervised train-only projection."""

    method_name = "naive_time_sync"
    output_dim = FUSION_OUTPUT_DIM

    def __init__(self, config: NaiveTimeSyncConfig | None = None) -> None:
        self.config = config or NaiveTimeSyncConfig()
        self.normalizer = TrainOnlyRobustNormalizer()
        self.projector = TrainOnlyPCAProjector(output_dim=self.config.output_dim)

    def fit(
        self,
        batch: DualStreamObservationBatch,
        *,
        train_sample_ids: Sequence[str],
        held_out_sample_ids: Sequence[str],
        normalizer: TrainOnlyRobustNormalizer | None = None,
    ) -> "NaiveTimeSyncEncoder":
        if normalizer is None:
            self.normalizer.fit(
                batch,
                train_sample_ids=train_sample_ids,
                held_out_sample_ids=held_out_sample_ids,
            )
        else:
            if tuple(sorted(train_sample_ids)) != normalizer.fit_sample_ids:
                raise RepresentationContractError(
                    "shared normalizer fit samples do not match naive time-sync train samples"
                )
            if set(normalizer.fit_sample_ids) & set(held_out_sample_ids):
                raise RepresentationContractError(
                    "shared normalizer includes held-out samples"
                )
            self.normalizer = normalizer
        normalized = self.normalizer.transform(batch)
        features, _valid = self._query_features(normalized)
        batch_size, query_count, feature_count = features.shape
        row_sample_ids = tuple(
            sample_id
            for sample_id in batch.sample_ids
            for _ in range(query_count)
        )
        self.projector.fit(
            features.detach().cpu().numpy().reshape(batch_size * query_count, feature_count),
            row_sample_ids=row_sample_ids,
            train_sample_ids=train_sample_ids,
            held_out_sample_ids=held_out_sample_ids,
        )
        return self

    def encode(
        self,
        batch: DualStreamObservationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalizer.transform(batch)
        features, valid = self._query_features(normalized)
        batch_size, query_count, feature_count = features.shape
        projected = self.projector.transform(
            features.detach().cpu().numpy().reshape(batch_size * query_count, feature_count)
        ).reshape(batch_size, query_count, self.output_dim)
        sequence = torch.from_numpy(projected).to(batch.query_timestamps_s.device)
        sequence = sequence * valid.unsqueeze(-1).to(sequence.dtype)
        return sequence, valid

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "method_name": self.method_name,
            "config": asdict(self.config),
            "normalizer": self.normalizer.to_manifest(),
            "pca_projector": self.projector.to_manifest(),
            "parameter_count": 0,
            "interpolation_policy": "past_or_present_forward_fill_only",
            "label_used_for_encoder_training": False,
        }

    def _query_features(
        self,
        batch: DualStreamObservationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physiology = causal_query_stream(batch, stream_name="physiology")
        vehicle = causal_query_stream(batch, stream_name="vehicle")
        features = torch.cat(
            (
                _stream_features(physiology, maximum_age_s=self.config.maximum_age_s),
                _stream_features(vehicle, maximum_age_s=self.config.maximum_age_s),
            ),
            dim=-1,
        )
        valid = physiology.modality_mask | vehicle.modality_mask
        return features, valid


class NaiveTimeSyncFusionAdapter:
    output_dim = FUSION_OUTPUT_DIM
    method_name = "naive_time_sync"

    def __init__(
        self,
        *,
        encoder: NaiveTimeSyncEncoder,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self.encoder = encoder
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        sequence, _modality_available = self.encoder.encode(batch)
        query_valid = torch.ones(
            sequence.shape[:2],
            dtype=torch.bool,
            device=sequence.device,
        )
        pooled = sequence.mean(dim=1)
        return FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s,
            sequence_embedding=sequence,
            valid_mask=query_valid,
            pooled_embedding=pooled,
            method_name=self.method_name,
            fold_id=self.fold_id,
            checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes,
        )

    def to_manifest(self) -> Mapping[str, object]:
        return {
            **self.encoder.to_manifest(),
            "fold_id": self.fold_id,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def save_naive_time_sync_checkpoint(
    path: str | Path,
    *,
    encoder: NaiveTimeSyncEncoder,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    payload = {
        "format": "chronaris.naive_time_sync.v1",
        "config": asdict(encoder.config),
        "normalizer": dict(encoder.normalizer.to_manifest()),
        "pca_state": dict(encoder.projector.state_dict()),
        "label_used_for_encoder_training": False,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def load_naive_time_sync_checkpoint(
    path: str | Path,
) -> NaiveTimeSyncEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "chronaris.naive_time_sync.v1":
        raise RepresentationContractError("unsupported naive time-sync checkpoint format")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("naive time-sync checkpoint used downstream labels")
    encoder = NaiveTimeSyncEncoder(NaiveTimeSyncConfig(**dict(payload["config"])))
    encoder.normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    encoder.projector = TrainOnlyPCAProjector.from_state_dict(payload["pca_state"])
    return encoder


def _stream_features(
    stream: CausalQueryStream,
    *,
    maximum_age_s: float,
) -> torch.Tensor:
    mask_float = stream.feature_mask.to(stream.values.dtype)
    age = torch.where(
        stream.feature_mask,
        stream.observation_age_s.clamp(max=maximum_age_s),
        torch.zeros_like(stream.observation_age_s),
    )
    age = torch.log1p(age) / math.log1p(maximum_age_s)
    return torch.cat((stream.values, mask_float, age), dim=-1)
