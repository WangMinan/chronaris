"""Shared continuous-time backbone for physiology-only and vehicle-only baselines."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    FusionStreamBatch,
    RepresentationContractError,
)
from chronaris.representation.normalization import TrainOnlyRobustNormalizer


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.contiformer.physiopro.network import ContiFormerEncoder


@dataclass(frozen=True, slots=True)
class SingleStreamEncoderConfig:
    active_stream: str
    input_feature_dim: int
    hidden_dim: int = FUSION_OUTPUT_DIM
    num_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    maximum_age_s: float = 30.0

    def __post_init__(self) -> None:
        if self.active_stream not in {"physiology", "vehicle"}:
            raise ValueError("active_stream must be physiology or vehicle")
        if self.input_feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("single-stream dimensions are invalid")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.layers <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("single-stream depth/dropout is invalid")
        if self.maximum_age_s <= 0:
            raise ValueError("maximum_age_s must be positive")


@dataclass(frozen=True, slots=True)
class SingleStreamEncoding:
    sequence_embedding: torch.Tensor
    valid_mask: torch.Tensor


class ContinuousTimeSingleStreamEncoder(nn.Module):
    """One causal ContiFormer backbone reused by both single-stream baselines."""

    def __init__(self, config: SingleStreamEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ContiFormerEncoder(
            input_dim=config.input_feature_dim * 3,
            model_dim=config.hidden_dim,
            num_heads=config.num_heads,
            depth=config.layers,
            dropout=config.dropout,
            causal=True,
        )
        self.contract_projection = (
            nn.Identity()
            if config.hidden_dim == FUSION_OUTPUT_DIM
            else nn.Linear(config.hidden_dim, FUSION_OUTPUT_DIM)
        )

    def forward(self, batch: DualStreamObservationBatch) -> SingleStreamEncoding:
        queried = causal_query_stream(batch, stream_name=self.config.active_stream)
        if queried.values.shape[-1] != self.config.input_feature_dim:
            raise RepresentationContractError(
                "single-stream input dimension does not match encoder config"
            )
        mask_float = queried.feature_mask.to(queried.values.dtype)
        age = torch.where(
            queried.feature_mask,
            queried.observation_age_s.clamp(max=self.config.maximum_age_s),
            torch.zeros_like(queried.observation_age_s),
        )
        age = torch.log1p(age) / math.log1p(self.config.maximum_age_s)
        inputs = torch.cat((queried.values, mask_float, age), dim=-1)
        attention_mask = queried.modality_mask.clone()
        attention_mask[:, 0] = True
        encoded, _attention = self.encoder(
            inputs,
            time_axis=queried.timestamps_s.to(inputs.dtype),
            mask=attention_mask,
        )
        sequence = torch.nan_to_num(self.contract_projection(encoded))
        sequence = sequence * queried.modality_mask.unsqueeze(-1).to(sequence.dtype)
        return SingleStreamEncoding(
            sequence_embedding=sequence,
            valid_mask=queried.modality_mask,
        )


class SingleStreamFusionAdapter:
    """Bind one fitted transform and locked checkpoint to the shared backbone."""

    output_dim = FUSION_OUTPUT_DIM

    def __init__(
        self,
        *,
        backbone: ContinuousTimeSingleStreamEncoder,
        normalizer: TrainOnlyRobustNormalizer,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self.backbone = backbone
        self.normalizer = normalizer
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.method_name = (
            "physiology_only"
            if backbone.config.active_stream == "physiology"
            else "vehicle_only"
        )

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        device = next(self.backbone.parameters()).device
        moved = move_observation_batch(batch, device=device)
        normalized = self.normalizer.transform(moved)
        self.backbone.eval()
        with torch.inference_mode():
            encoded = self.backbone(normalized)
        query_valid = torch.ones_like(encoded.valid_mask)
        pooled = encoded.sequence_embedding.mean(dim=1)
        return FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s.to(device),
            sequence_embedding=encoded.sequence_embedding,
            valid_mask=query_valid,
            pooled_embedding=pooled,
            method_name=self.method_name,
            fold_id=self.fold_id,
            checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters())

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "method_name": self.method_name,
            "backbone_class": type(self.backbone).__name__,
            "backbone_config": asdict(self.backbone.config),
            "parameter_count": self.parameter_count,
            "normalizer": self.normalizer.to_manifest(),
            "fold_id": self.fold_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "label_used_for_encoder_training": False,
        }


def save_single_stream_checkpoint(
    path: str | Path,
    *,
    backbone: ContinuousTimeSingleStreamEncoder,
    normalizer: TrainOnlyRobustNormalizer,
    seed: int,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    payload = {
        "format": "chronaris.single_stream_encoder.v1",
        "config": asdict(backbone.config),
        "model_state_dict": backbone.state_dict(),
        "normalizer": dict(normalizer.to_manifest()),
        "seed": int(seed),
        "label_used_for_encoder_training": False,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def load_single_stream_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ContinuousTimeSingleStreamEncoder, TrainOnlyRobustNormalizer, Mapping[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "chronaris.single_stream_encoder.v1":
        raise RepresentationContractError("unsupported single-stream checkpoint format")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("single-stream checkpoint used downstream labels")
    config = SingleStreamEncoderConfig(**dict(payload["config"]))
    backbone = ContinuousTimeSingleStreamEncoder(config).to(device)
    backbone.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    metadata = {
        "format": payload["format"],
        "seed": int(payload["seed"]),
        "label_used_for_encoder_training": False,
    }
    return backbone, normalizer, metadata


def move_observation_batch(
    batch: DualStreamObservationBatch,
    *,
    device: str | torch.device,
) -> DualStreamObservationBatch:
    resolved = torch.device(device)
    return DualStreamObservationBatch(
        sample_ids=batch.sample_ids,
        group_ids=batch.group_ids,
        physiology_values=batch.physiology_values.to(resolved),
        physiology_timestamps_s=batch.physiology_timestamps_s.to(resolved),
        physiology_point_mask=batch.physiology_point_mask.to(resolved),
        physiology_feature_mask=batch.physiology_feature_mask.to(resolved),
        physiology_observation_age_s=batch.physiology_observation_age_s.to(resolved),
        vehicle_values=batch.vehicle_values.to(resolved),
        vehicle_timestamps_s=batch.vehicle_timestamps_s.to(resolved),
        vehicle_point_mask=batch.vehicle_point_mask.to(resolved),
        vehicle_feature_mask=batch.vehicle_feature_mask.to(resolved),
        vehicle_observation_age_s=batch.vehicle_observation_age_s.to(resolved),
        query_timestamps_s=batch.query_timestamps_s.to(resolved),
        source_sample_hashes=batch.source_sample_hashes,
    )
