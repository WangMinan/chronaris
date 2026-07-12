"""Task-head-free causal MulT and ContiFormer production adapters."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.causal_query import (
    CausalQueryStream,
    causal_query_stream,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    FusionStreamBatch,
    RepresentationContractError,
    masked_mean_pool,
)
from chronaris.representation.normalization import TrainOnlyRobustNormalizer


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.contiformer.physiopro.network import ContiFormerEncoder
from third_party.mult.modules.transformer import TransformerEncoder


@dataclass(frozen=True, slots=True)
class DeepBaselineEncoderConfig:
    method_name: str
    physiology_feature_dim: int
    vehicle_feature_dim: int
    hidden_dim: int = FUSION_OUTPUT_DIM
    num_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    maximum_age_s: float = 30.0

    def __post_init__(self) -> None:
        if self.method_name not in {"mult", "contiformer"}:
            raise ValueError("deep baseline method must be mult or contiformer")
        if self.physiology_feature_dim <= 0 or self.vehicle_feature_dim <= 0:
            raise ValueError("deep baseline input dimensions must be positive")
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("deep baseline hidden/head dimensions are invalid")
        if self.layers <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("deep baseline depth/dropout is invalid")
        if self.maximum_age_s <= 0:
            raise ValueError("maximum_age_s must be positive")


@dataclass(frozen=True, slots=True)
class DeepBaselineEncoding:
    sequence_embedding: torch.Tensor
    modality_available_mask: torch.Tensor


class CausalMulTFusionEncoder(nn.Module):
    """Bidirectional cross-modal MulT with causal masks in every attention block."""

    def __init__(self, config: DeepBaselineEncoderConfig) -> None:
        super().__init__()
        if config.method_name != "mult":
            raise ValueError("CausalMulTFusionEncoder requires method_name=mult")
        self.config = config
        self.physiology_projection = nn.Linear(
            config.physiology_feature_dim * 3 + 2,
            config.hidden_dim,
        )
        self.vehicle_projection = nn.Linear(
            config.vehicle_feature_dim * 3 + 2,
            config.hidden_dim,
        )
        encoder_kwargs = {
            "embed_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "layers": config.layers,
            "attn_dropout": config.dropout,
            "relu_dropout": config.dropout,
            "res_dropout": config.dropout,
            "embed_dropout": config.dropout,
            "attn_mask": True,
        }
        self.physiology_from_vehicle = TransformerEncoder(**encoder_kwargs)
        self.vehicle_from_physiology = TransformerEncoder(**encoder_kwargs)
        memory_kwargs = {
            **encoder_kwargs,
            "embed_dim": config.hidden_dim * 2,
        }
        self.physiology_memory = TransformerEncoder(**memory_kwargs)
        self.vehicle_memory = TransformerEncoder(**memory_kwargs)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 4),
            nn.Linear(config.hidden_dim * 4, FUSION_OUTPUT_DIM),
        )

    def forward(self, batch: DualStreamObservationBatch) -> DeepBaselineEncoding:
        physiology, vehicle = _dual_query_inputs(batch, self.config)
        time_features = _time_features(
            batch.query_timestamps_s.to(physiology.values.device),
            maximum_time_s=self.config.maximum_age_s,
            dtype=physiology.values.dtype,
        )
        physiology_input = self.physiology_projection(
            torch.cat((_stream_features(physiology, self.config), time_features), dim=-1)
        )
        vehicle_input = self.vehicle_projection(
            torch.cat((_stream_features(vehicle, self.config), time_features), dim=-1)
        )
        physiology_input = physiology_input * physiology.modality_mask.unsqueeze(-1)
        vehicle_input = vehicle_input * vehicle.modality_mask.unsqueeze(-1)
        safe_physiology = _safe_attention_mask(physiology.modality_mask)
        safe_vehicle = _safe_attention_mask(vehicle.modality_mask)
        physiology_seq = physiology_input.transpose(0, 1)
        vehicle_seq = vehicle_input.transpose(0, 1)
        cross_physiology = self.physiology_from_vehicle(
            physiology_seq,
            vehicle_seq,
            vehicle_seq,
            key_padding_mask=~safe_vehicle,
        )
        cross_vehicle = self.vehicle_from_physiology(
            vehicle_seq,
            physiology_seq,
            physiology_seq,
            key_padding_mask=~safe_physiology,
        )
        physiology_memory = self.physiology_memory(
            torch.cat((physiology_seq, cross_physiology), dim=-1),
            key_padding_mask=~safe_physiology,
        )
        vehicle_memory = self.vehicle_memory(
            torch.cat((vehicle_seq, cross_vehicle), dim=-1),
            key_padding_mask=~safe_vehicle,
        )
        merged = torch.cat((physiology_memory, vehicle_memory), dim=-1).transpose(0, 1)
        available = physiology.modality_mask | vehicle.modality_mask
        sequence = torch.nan_to_num(self.output_projection(merged))
        sequence = sequence * available.unsqueeze(-1).to(sequence.dtype)
        return DeepBaselineEncoding(sequence, available)


class CausalContiFormerFusionEncoder(nn.Module):
    """Task-head-free causal ContiFormer over two causally queried streams."""

    def __init__(self, config: DeepBaselineEncoderConfig) -> None:
        super().__init__()
        if config.method_name != "contiformer":
            raise ValueError(
                "CausalContiFormerFusionEncoder requires method_name=contiformer"
            )
        self.config = config
        input_dim = 3 * (
            config.physiology_feature_dim + config.vehicle_feature_dim
        )
        self.encoder = ContiFormerEncoder(
            input_dim=input_dim,
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

    def forward(self, batch: DualStreamObservationBatch) -> DeepBaselineEncoding:
        physiology, vehicle = _dual_query_inputs(batch, self.config)
        inputs = torch.cat(
            (
                _stream_features(physiology, self.config),
                _stream_features(vehicle, self.config),
            ),
            dim=-1,
        )
        available = physiology.modality_mask | vehicle.modality_mask
        encoded, _attention = self.encoder(
            inputs,
            time_axis=batch.query_timestamps_s.to(
                device=inputs.device,
                dtype=inputs.dtype,
            ),
            mask=_safe_attention_mask(available),
        )
        sequence = torch.nan_to_num(self.contract_projection(encoded))
        sequence = sequence * available.unsqueeze(-1).to(sequence.dtype)
        return DeepBaselineEncoding(sequence, available)


class DeepBaselineFusionAdapter:
    output_dim = FUSION_OUTPUT_DIM

    def __init__(
        self,
        *,
        backbone: CausalMulTFusionEncoder | CausalContiFormerFusionEncoder,
        normalizer: TrainOnlyRobustNormalizer,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self.backbone = backbone
        self.normalizer = normalizer
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.method_name = backbone.config.method_name

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        device = next(self.backbone.parameters()).device
        normalized = self.normalizer.transform(
            move_observation_batch(batch, device=device)
        )
        self.backbone.eval()
        with torch.inference_mode():
            encoded = self.backbone(normalized)
        query_valid = encoded.modality_available_mask
        pooled = masked_mean_pool(encoded.sequence_embedding, query_valid)
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
            "sequence_source": "task_head_free",
            "causal_attention": True,
            "label_used_for_encoder_training": False,
        }


def save_deep_baseline_checkpoint(
    path: str | Path,
    *,
    backbone: CausalMulTFusionEncoder | CausalContiFormerFusionEncoder,
    normalizer: TrainOnlyRobustNormalizer,
    seed: int,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    payload = {
        "format": "chronaris.deep_baseline_encoder.v1",
        "config": asdict(backbone.config),
        "model_state_dict": backbone.state_dict(),
        "normalizer": dict(normalizer.to_manifest()),
        "seed": int(seed),
        "sequence_source": "task_head_free",
        "label_used_for_encoder_training": False,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def load_deep_baseline_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    CausalMulTFusionEncoder | CausalContiFormerFusionEncoder,
    TrainOnlyRobustNormalizer,
    Mapping[str, object],
]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "chronaris.deep_baseline_encoder.v1":
        raise RepresentationContractError("unsupported deep baseline checkpoint format")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("deep baseline checkpoint used downstream labels")
    if payload.get("sequence_source") != "task_head_free":
        raise RepresentationContractError("deep baseline checkpoint is not task-head-free")
    config = DeepBaselineEncoderConfig(**dict(payload["config"]))
    backbone = (
        CausalMulTFusionEncoder(config)
        if config.method_name == "mult"
        else CausalContiFormerFusionEncoder(config)
    ).to(device)
    backbone.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    metadata = {
        "format": payload["format"],
        "seed": int(payload["seed"]),
        "sequence_source": payload["sequence_source"],
        "label_used_for_encoder_training": False,
    }
    return backbone, normalizer, metadata


def _dual_query_inputs(
    batch: DualStreamObservationBatch,
    config: DeepBaselineEncoderConfig,
) -> tuple[CausalQueryStream, CausalQueryStream]:
    physiology = causal_query_stream(batch, stream_name="physiology")
    vehicle = causal_query_stream(batch, stream_name="vehicle")
    if physiology.values.shape[-1] != config.physiology_feature_dim:
        raise RepresentationContractError("deep baseline physiology dimension mismatch")
    if vehicle.values.shape[-1] != config.vehicle_feature_dim:
        raise RepresentationContractError("deep baseline vehicle dimension mismatch")
    return physiology, vehicle


def _stream_features(
    stream: CausalQueryStream,
    config: DeepBaselineEncoderConfig,
) -> torch.Tensor:
    mask_float = stream.feature_mask.to(stream.values.dtype)
    age = torch.where(
        stream.feature_mask,
        stream.observation_age_s.clamp(max=config.maximum_age_s),
        torch.zeros_like(stream.observation_age_s),
    )
    age = torch.log1p(age) / math.log1p(config.maximum_age_s)
    return torch.cat((stream.values, mask_float, age), dim=-1)


def _time_features(
    timestamps: torch.Tensor,
    *,
    maximum_time_s: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    time = timestamps.to(dtype=dtype) / maximum_time_s
    delta = torch.zeros_like(time)
    delta[:, 1:] = time[:, 1:] - time[:, :-1]
    return torch.stack((time, delta), dim=-1)


def _safe_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    safe = mask.clone()
    safe[:, 0] = True
    return safe
