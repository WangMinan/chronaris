"""Stage G minimal causal fusion pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from chronaris.models.fusion import (
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalMaskedCrossModalFusion,
    attention_entropy,
)
from chronaris.pipelines.alignment_preview import AlignmentPreviewIntermediateExport

FusionStateSource = Literal["hidden", "projection"]
FusionOutputMode = Literal["concat", "pooled_with_residual"]
FusionResidualMode = Literal["none", "raw_window_stats"]


@dataclass(frozen=True, slots=True)
class StageGCausalFusionConfig:
    """Configuration for the Stage G minimal fusion summary pipeline."""

    state_source: FusionStateSource = "hidden"
    attention_temperature: float = 1.0
    event_bias_weight: float = 0.25
    causal_epsilon_s: float = 1e-6
    normalize_states: bool = True
    use_causal_mask: bool = True
    lag_window_points: int | None = None
    fusion_output_mode: FusionOutputMode = "concat"
    residual_mode: FusionResidualMode = "none"

    def __post_init__(self) -> None:
        if self.state_source not in {"hidden", "projection"}:
            raise ValueError("state_source must be one of: hidden, projection.")
        if self.fusion_output_mode not in {"concat", "pooled_with_residual"}:
            raise ValueError("fusion_output_mode must be one of: concat, pooled_with_residual.")
        if self.residual_mode not in {"none", "raw_window_stats"}:
            raise ValueError("residual_mode must be one of: none, raw_window_stats.")
        CausalFusionConfig(
            attention_temperature=self.attention_temperature,
            event_bias_weight=self.event_bias_weight,
            causal_epsilon_s=self.causal_epsilon_s,
            normalize_states=self.normalize_states,
            use_causal_mask=self.use_causal_mask,
            lag_window_points=self.lag_window_points,
        )


@dataclass(frozen=True, slots=True)
class StageGCausalFusionSample:
    """Per-sample Stage G attention and contribution diagnostics."""

    sample_id: str
    reference_point_count: int
    state_dim: int
    fused_dim: int
    mean_attention_entropy: float
    mean_max_attention: float
    mean_causal_option_count: float
    top_event_offset_s: float
    top_event_score: float
    top_contribution_offset_s: float
    top_contribution_score: float
    attention_weights: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    vehicle_event_scores: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class StageGCausalFusionTensorExport:
    """Tensor-shaped Stage G outputs serialized as Python tuples."""

    sample_ids: tuple[str, ...]
    fused_states: tuple[tuple[tuple[float, ...], ...], ...]
    attention_weights: tuple[tuple[tuple[float, ...], ...], ...]
    vehicle_event_scores: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class StageGCausalFusionResult:
    """Stage G minimal causal fusion output for exported alignment intermediates."""

    config: StageGCausalFusionConfig
    partition: str
    sample_count: int
    reference_point_count: int
    state_dim: int
    fused_dim: int
    mean_attention_entropy: float
    mean_max_attention: float
    mean_causal_option_count: float
    mean_top_event_score: float
    mean_top_contribution_score: float
    samples: tuple[StageGCausalFusionSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "state_source": self.config.state_source,
                "attention_temperature": self.config.attention_temperature,
                "event_bias_weight": self.config.event_bias_weight,
                "causal_epsilon_s": self.config.causal_epsilon_s,
                "normalize_states": self.config.normalize_states,
                "use_causal_mask": self.config.use_causal_mask,
                "lag_window_points": self.config.lag_window_points,
                "fusion_output_mode": self.config.fusion_output_mode,
                "residual_mode": self.config.residual_mode,
            },
            "partition": self.partition,
            "sample_count": self.sample_count,
            "reference_point_count": self.reference_point_count,
            "state_dim": self.state_dim,
            "fused_dim": self.fused_dim,
            "mean_attention_entropy": self.mean_attention_entropy,
            "mean_max_attention": self.mean_max_attention,
            "mean_causal_option_count": self.mean_causal_option_count,
            "mean_top_event_score": self.mean_top_event_score,
            "mean_top_contribution_score": self.mean_top_contribution_score,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "reference_point_count": sample.reference_point_count,
                    "state_dim": sample.state_dim,
                    "fused_dim": sample.fused_dim,
                    "mean_attention_entropy": sample.mean_attention_entropy,
                    "mean_max_attention": sample.mean_max_attention,
                    "mean_causal_option_count": sample.mean_causal_option_count,
                    "top_event_offset_s": sample.top_event_offset_s,
                    "top_event_score": sample.top_event_score,
                    "top_contribution_offset_s": sample.top_contribution_offset_s,
                    "top_contribution_score": sample.top_contribution_score,
                    "attention_weights": sample.attention_weights,
                    "vehicle_event_scores": sample.vehicle_event_scores,
                }
                for sample in self.samples
            ],
        }


def export_stage_g_causal_fusion_tensors(
    intermediate_export: AlignmentPreviewIntermediateExport,
    *,
    config: StageGCausalFusionConfig | None = None,
) -> StageGCausalFusionTensorExport:
    """Export deterministic Stage G tensors for downstream feature packaging."""

    resolved_config = config or StageGCausalFusionConfig()
    if not intermediate_export.samples:
        return StageGCausalFusionTensorExport(
            sample_ids=(),
            fused_states=(),
            attention_weights=(),
            vehicle_event_scores=(),
        )

    tensor_input = _build_tensor_input(intermediate_export, config=resolved_config)
    model = CausalMaskedCrossModalFusion(
        CausalFusionConfig(
            attention_temperature=resolved_config.attention_temperature,
            event_bias_weight=resolved_config.event_bias_weight,
            causal_epsilon_s=resolved_config.causal_epsilon_s,
            normalize_states=resolved_config.normalize_states,
            use_causal_mask=resolved_config.use_causal_mask,
            lag_window_points=resolved_config.lag_window_points,
        )
    )
    with torch.no_grad():
        output = model(tensor_input)

    return StageGCausalFusionTensorExport(
        sample_ids=tuple(sample.sample_id for sample in intermediate_export.samples),
        fused_states=tuple(
            _tensor_2d_to_tuple(output.fused_states[sample_index])
            for sample_index in range(output.fused_states.shape[0])
        ),
        attention_weights=tuple(
            _tensor_2d_to_tuple(output.attention_weights[sample_index])
            for sample_index in range(output.attention_weights.shape[0])
        ),
        vehicle_event_scores=tuple(
            _tensor_1d_to_tuple(output.vehicle_event_scores[sample_index])
            for sample_index in range(output.vehicle_event_scores.shape[0])
        ),
    )


def run_stage_g_causal_fusion(
    intermediate_export: AlignmentPreviewIntermediateExport,
    *,
    config: StageGCausalFusionConfig | None = None,
) -> StageGCausalFusionResult:
    """Run deterministic Stage G(min) fusion over exported reference-grid states."""

    resolved_config = config or StageGCausalFusionConfig()
    if not intermediate_export.samples:
        return StageGCausalFusionResult(
            config=resolved_config,
            partition=intermediate_export.partition,
            sample_count=0,
            reference_point_count=intermediate_export.reference_point_count,
            state_dim=0,
            fused_dim=0,
            mean_attention_entropy=0.0,
            mean_max_attention=0.0,
            mean_causal_option_count=0.0,
            mean_top_event_score=0.0,
            mean_top_contribution_score=0.0,
            samples=(),
        )

    tensor_input = _build_tensor_input(intermediate_export, config=resolved_config)
    tensor_export = export_stage_g_causal_fusion_tensors(
        intermediate_export,
        config=resolved_config,
    )
    model = CausalMaskedCrossModalFusion(
        CausalFusionConfig(
            attention_temperature=resolved_config.attention_temperature,
            event_bias_weight=resolved_config.event_bias_weight,
            causal_epsilon_s=resolved_config.causal_epsilon_s,
            normalize_states=resolved_config.normalize_states,
            use_causal_mask=resolved_config.use_causal_mask,
            lag_window_points=resolved_config.lag_window_points,
        )
    )
    with torch.no_grad():
        output = model(tensor_input)

    entropy = attention_entropy(output.attention_weights, output.causal_mask)
    max_attention = output.attention_weights.max(dim=-1).values
    causal_option_count = output.causal_mask.sum(dim=-1).to(dtype=output.attention_weights.dtype)
    contribution_scores = output.attention_weights.sum(dim=1) * output.vehicle_event_scores

    samples: list[StageGCausalFusionSample] = []
    for sample_index, source_sample in enumerate(intermediate_export.samples):
        event_scores = output.vehicle_event_scores[sample_index]
        top_event_index = int(torch.argmax(event_scores).detach().cpu())
        sample_contributions = contribution_scores[sample_index]
        top_contribution_index = int(torch.argmax(sample_contributions).detach().cpu())
        vehicle_offsets = tensor_input.vehicle_offsets_s[sample_index]
        samples.append(
            StageGCausalFusionSample(
                sample_id=source_sample.sample_id,
                reference_point_count=int(tensor_input.physiology_states.shape[1]),
                state_dim=int(tensor_input.physiology_states.shape[-1]),
                fused_dim=int(output.fused_states.shape[-1]),
                mean_attention_entropy=float(entropy[sample_index].mean().detach().cpu()),
                mean_max_attention=float(max_attention[sample_index].mean().detach().cpu()),
                mean_causal_option_count=float(causal_option_count[sample_index].mean().detach().cpu()),
                top_event_offset_s=float(vehicle_offsets[top_event_index].detach().cpu()),
                top_event_score=float(event_scores[top_event_index].detach().cpu()),
                top_contribution_offset_s=float(vehicle_offsets[top_contribution_index].detach().cpu()),
                top_contribution_score=float(sample_contributions[top_contribution_index].detach().cpu()),
                attention_weights=tensor_export.attention_weights[sample_index],
                vehicle_event_scores=tensor_export.vehicle_event_scores[sample_index],
            )
        )

    return StageGCausalFusionResult(
        config=resolved_config,
        partition=intermediate_export.partition,
        sample_count=len(samples),
        reference_point_count=int(tensor_input.physiology_states.shape[1]),
        state_dim=int(tensor_input.physiology_states.shape[-1]),
        fused_dim=int(output.fused_states.shape[-1]),
        mean_attention_entropy=_mean(tuple(sample.mean_attention_entropy for sample in samples)),
        mean_max_attention=_mean(tuple(sample.mean_max_attention for sample in samples)),
        mean_causal_option_count=_mean(tuple(sample.mean_causal_option_count for sample in samples)),
        mean_top_event_score=_mean(tuple(sample.top_event_score for sample in samples)),
        mean_top_contribution_score=_mean(tuple(sample.top_contribution_score for sample in samples)),
        samples=tuple(samples),
    )


def render_stage_g_causal_fusion_markdown(result: StageGCausalFusionResult) -> str:
    """Render a compact Stage G(min) diagnostics section."""

    lines = [
        "## Stage G Causal Fusion Diagnostics",
        "",
        f"- enabled: `{result.sample_count > 0}`",
        f"- partition: `{result.partition}`",
        f"- state source: `{result.config.state_source}`",
        f"- sample count: `{result.sample_count}`",
        f"- reference point count: `{result.reference_point_count}`",
        f"- state dim: `{result.state_dim}`",
        f"- fused dim: `{result.fused_dim}`",
        f"- use causal mask: `{result.config.use_causal_mask}`",
        f"- lag window points: `{result.config.lag_window_points}`",
        f"- fusion output mode: `{result.config.fusion_output_mode}`",
        f"- residual mode: `{result.config.residual_mode}`",
        f"- mean attention entropy: `{result.mean_attention_entropy:.6f}`",
        f"- mean max attention: `{result.mean_max_attention:.6f}`",
        f"- mean causal option count: `{result.mean_causal_option_count:.6f}`",
        "",
    ]

    if result.samples:
        lines.extend(
            [
                "### Event Contribution Samples",
                "",
                "| sample | top event offset s | event score | top contribution offset s | contribution score |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for sample in result.samples:
            lines.append(
                f"| `{sample.sample_id}` | {sample.top_event_offset_s:.6f} | "
                f"{sample.top_event_score:.6f} | {sample.top_contribution_offset_s:.6f} | "
                f"{sample.top_contribution_score:.6f} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_tensor_input(
    intermediate_export: AlignmentPreviewIntermediateExport,
    *,
    config: StageGCausalFusionConfig,
) -> CausalFusionTensorInput:
    physiology_states = []
    vehicle_states = []
    physiology_offsets = []
    vehicle_offsets = []
    for sample in intermediate_export.samples:
        if config.state_source == "hidden":
            physiology_state_rows = sample.physiology.reference_hidden_states
            vehicle_state_rows = sample.vehicle.reference_hidden_states
        else:
            physiology_state_rows = sample.physiology.reference_projected_states
            vehicle_state_rows = sample.vehicle.reference_projected_states
        physiology_states.append(physiology_state_rows)
        vehicle_states.append(vehicle_state_rows)
        physiology_offsets.append(sample.physiology.reference_offsets_s)
        vehicle_offsets.append(sample.vehicle.reference_offsets_s)

    return CausalFusionTensorInput(
        physiology_states=torch.as_tensor(physiology_states, dtype=torch.float32),
        vehicle_states=torch.as_tensor(vehicle_states, dtype=torch.float32),
        physiology_offsets_s=torch.as_tensor(physiology_offsets, dtype=torch.float32),
        vehicle_offsets_s=torch.as_tensor(vehicle_offsets, dtype=torch.float32),
    )


def _tensor_1d_to_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu().tolist())


def _tensor_2d_to_tuple(values: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values.detach().cpu().tolist())


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
