"""One differentiable interface over the five trainable fusion encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.chronaris_continuous import (
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousFusionEncoder,
)
from chronaris.modeling.fusion_encoders.deep_baselines import (
    CausalContiFormerFusionEncoder,
    CausalMulTFusionEncoder,
    DeepBaselineEncoderConfig,
)
from chronaris.modeling.fusion_encoders.single_stream import (
    ContinuousTimeSingleStreamEncoder,
    SingleStreamEncoderConfig,
)
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
)


TRAINABLE_FUSION_METHODS = (
    "physiology_only",
    "vehicle_only",
    "mult",
    "contiformer",
    "chronaris",
)


@dataclass(frozen=True, slots=True)
class EncoderCandidateConfig:
    """One equal-budget architecture candidate from the frozen screen."""

    candidate_id: str = "A"
    hidden_dim: int = FUSION_OUTPUT_DIM
    learning_rate: float = 1e-3
    dropout: float = 0.1
    layers: int = 2
    num_heads: int = 4

    def __post_init__(self) -> None:
        if self.candidate_id not in {"A", "B", "C", "D"}:
            raise ValueError("encoder candidate id must be A, B, C, or D")
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("encoder candidate hidden/head dimensions are invalid")
        if self.learning_rate <= 0 or self.layers <= 0:
            raise ValueError("encoder candidate optimizer/depth is invalid")
        if not 0 <= self.dropout < 1:
            raise ValueError("encoder candidate dropout is invalid")


ENCODER_SCREEN_CANDIDATES = (
    EncoderCandidateConfig(candidate_id="A"),
    EncoderCandidateConfig(candidate_id="B", learning_rate=3e-4),
    EncoderCandidateConfig(candidate_id="C", hidden_dim=32),
    EncoderCandidateConfig(candidate_id="D", dropout=0.2),
)


@dataclass(frozen=True, slots=True)
class PretrainingEncoderOutput:
    method_name: str
    sequence_embedding: torch.Tensor
    modality_available_mask: torch.Tensor
    auxiliary: Mapping[str, object]


class TrainableFusionEncoder(nn.Module):
    """Dispatch five production backbones without inference-only adapter logic."""

    def __init__(self, *, method_name: str, backbone: nn.Module) -> None:
        super().__init__()
        if method_name not in TRAINABLE_FUSION_METHODS:
            raise ValueError(f"unsupported trainable fusion method: {method_name}")
        self.method_name = method_name
        self.backbone = backbone

    def forward(
        self,
        batch: DualStreamObservationBatch,
        *,
        compute_chronaris_diagnostics: bool = False,
    ) -> PretrainingEncoderOutput:
        encoded = (
            self.backbone(
                batch,
                compute_diagnostics=compute_chronaris_diagnostics,
            )
            if self.method_name == "chronaris"
            else self.backbone(batch)
        )
        if self.method_name in {"physiology_only", "vehicle_only"}:
            sequence = encoded.sequence_embedding
            available = encoded.valid_mask
            auxiliary = {
                "continuous_alignment": "not_applicable",
                "physical_consistency": "not_applicable",
                "causal_direction": "not_applicable",
            }
        elif self.method_name in {"mult", "contiformer"}:
            sequence = encoded.sequence_embedding
            available = encoded.modality_available_mask
            auxiliary = {
                "continuous_alignment": "not_applicable",
                "physical_consistency": "not_applicable",
                "causal_direction": "not_applicable",
            }
        else:
            sequence = encoded.sequence_embedding
            available = encoded.modality_available_mask
            auxiliary = {
                "continuous_alignment": "available",
                "physical_consistency": encoded.physics_audit,
                "causal_direction": "available",
                "alignment_output": encoded.alignment_output,
                "fusion_output": encoded.fusion_output,
            }
        if sequence.shape[-1] != FUSION_OUTPUT_DIM:
            raise ValueError("pretraining encoder violated 64-dimensional contract")
        if not torch.isfinite(sequence).all():
            raise ValueError("pretraining encoder produced non-finite representation")
        return PretrainingEncoderOutput(
            method_name=self.method_name,
            sequence_embedding=sequence,
            modality_available_mask=available,
            auxiliary=auxiliary,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters())

    def config_manifest(self) -> Mapping[str, object]:
        config = self.backbone.config
        payload = (
            dict(config.to_checkpoint_dict())
            if hasattr(config, "to_checkpoint_dict")
            else asdict(config)
        )
        return {
            "method_name": self.method_name,
            "backbone_class": type(self.backbone).__name__,
            "backbone_config": payload,
            "parameter_count": self.parameter_count,
        }


def build_trainable_fusion_encoder(
    method_name: str,
    *,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...] = (),
    candidate_config: EncoderCandidateConfig | None = None,
    chronaris_variant: str = "full",
    chronaris_fusion_kind: str = "multiscale",
    chronaris_max_ode_step_s: float | None = None,
) -> TrainableFusionEncoder:
    candidate = candidate_config or ENCODER_SCREEN_CANDIDATES[0]
    if method_name == "physiology_only":
        backbone = ContinuousTimeSingleStreamEncoder(
            SingleStreamEncoderConfig(
                active_stream="physiology",
                input_feature_dim=len(physiology_feature_names),
                hidden_dim=candidate.hidden_dim,
                num_heads=candidate.num_heads,
                layers=candidate.layers,
                dropout=candidate.dropout,
            )
        )
    elif method_name == "vehicle_only":
        backbone = ContinuousTimeSingleStreamEncoder(
            SingleStreamEncoderConfig(
                active_stream="vehicle",
                input_feature_dim=len(vehicle_feature_names),
                hidden_dim=candidate.hidden_dim,
                num_heads=candidate.num_heads,
                layers=candidate.layers,
                dropout=candidate.dropout,
            )
        )
    elif method_name in {"mult", "contiformer"}:
        config = DeepBaselineEncoderConfig(
            method_name=method_name,
            physiology_feature_dim=len(physiology_feature_names),
            vehicle_feature_dim=len(vehicle_feature_names),
            hidden_dim=candidate.hidden_dim,
            num_heads=candidate.num_heads,
            layers=candidate.layers,
            dropout=candidate.dropout,
        )
        backbone = (
            CausalMulTFusionEncoder(config)
            if method_name == "mult"
            else CausalContiFormerFusionEncoder(config)
        )
    elif method_name == "chronaris":
        backbone = ChronarisContinuousFusionEncoder(
            ChronarisContinuousEncoderConfig(
                physiology_feature_names=physiology_feature_names,
                vehicle_feature_names=vehicle_feature_names,
                field_labels=vehicle_field_labels,
                variant=chronaris_variant,
                fusion_kind=chronaris_fusion_kind,
                max_ode_step_s=chronaris_max_ode_step_s,
                hidden_dim=candidate.hidden_dim,
                embedding_dim=candidate.hidden_dim,
                encoder_hidden_dim=candidate.hidden_dim,
                decoder_hidden_dim=candidate.hidden_dim,
                dynamics_hidden_dim=candidate.hidden_dim,
                dropout=candidate.dropout,
            )
        )
    else:
        raise ValueError(f"unsupported trainable fusion method: {method_name}")
    return TrainableFusionEncoder(method_name=method_name, backbone=backbone)
