"""Stage G minimal causal fusion pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from chronaris.features import load_stage_h_feature_run
from chronaris.models.fusion.causal import (
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalMaskedCrossModalFusion,
    attention_entropy,
)
from chronaris.models.fusion.semantic_event import (
    CausalEventFusion,
    CausalEventFusionConfig,
    SemanticEventTensorInput,
    semantic_query_entropy,
)
from chronaris.pipelines.alignment_preview import AlignmentPreviewIntermediateExport
from chronaris.pipelines.torch_runtime import (
    TORCH_DEVICE_CHOICES,
    resolve_torch_device_name,
)

FusionStateSource = Literal["hidden", "projection"]
FusionOutputMode = Literal["concat", "pooled_with_residual"]
FusionResidualMode = Literal["none", "raw_window_stats"]
SemanticSupportStateSource = Literal["hidden_with_projection_fallback", "hidden", "projection"]


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
    semantic_event_top_k: int = 4
    semantic_event_score_quantile: float = 0.75
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.state_source not in {"hidden", "projection"}:
            raise ValueError("state_source must be one of: hidden, projection.")
        if self.fusion_output_mode not in {"concat", "pooled_with_residual"}:
            raise ValueError("fusion_output_mode must be one of: concat, pooled_with_residual.")
        if self.residual_mode not in {"none", "raw_window_stats"}:
            raise ValueError("residual_mode must be one of: none, raw_window_stats.")
        if self.semantic_event_top_k <= 0:
            raise ValueError("semantic_event_top_k must be positive.")
        if not 0.0 < self.semantic_event_score_quantile <= 1.0:
            raise ValueError("semantic_event_score_quantile must be in (0, 1].")
        if self.device not in TORCH_DEVICE_CHOICES:
            raise ValueError(f"device must be one of: {', '.join(TORCH_DEVICE_CHOICES)}.")
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
    semantic_event_summary: dict[str, object] | None = None

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
                "semantic_event_top_k": self.config.semantic_event_top_k,
                "semantic_event_score_quantile": self.config.semantic_event_score_quantile,
                "device": self.config.device,
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
            "semantic_event": self.semantic_event_summary,
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
    resolved_device = resolve_torch_device_name(resolved_config.device)
    if not intermediate_export.samples:
        return StageGCausalFusionTensorExport(
            sample_ids=(),
            fused_states=(),
            attention_weights=(),
            vehicle_event_scores=(),
        )

    tensor_input = _build_tensor_input(
        intermediate_export,
        config=resolved_config,
        device=resolved_device,
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
    ).to(device=resolved_device)
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
    resolved_device = resolve_torch_device_name(resolved_config.device)
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

    tensor_input = _build_tensor_input(
        intermediate_export,
        config=resolved_config,
        device=resolved_device,
    )
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
    ).to(device=resolved_device)
    with torch.no_grad():
        output = model(tensor_input)

    entropy = attention_entropy(output.attention_weights, output.causal_mask)
    max_attention = output.attention_weights.max(dim=-1).values
    causal_option_count = output.causal_mask.sum(dim=-1).to(dtype=output.attention_weights.dtype)
    contribution_scores = output.attention_weights.sum(dim=1) * output.vehicle_event_scores
    semantic_fusion = CausalEventFusion(
        CausalEventFusionConfig(
            attention_temperature=resolved_config.attention_temperature,
            event_score_bias_weight=resolved_config.event_bias_weight,
            event_top_k=resolved_config.semantic_event_top_k,
            event_score_quantile=resolved_config.semantic_event_score_quantile,
        )
    ).to(device=resolved_device)
    with torch.no_grad():
        semantic_output = semantic_fusion(
            inputs=_build_semantic_event_input(tensor_input=tensor_input, causal_output=output)
        )

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
        semantic_event_summary=_build_semantic_event_summary(
            sample_ids=tuple(sample.sample_id for sample in intermediate_export.samples),
            semantic_output=semantic_output,
        ),
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

    semantic_event = result.semantic_event_summary or {}
    if semantic_event:
        lines.extend(
            [
                "### Semantic Event Fusion",
                "",
                f"- query names: `{semantic_event['query_names']}`",
                f"- mean_event_token_count: `{semantic_event['mean_event_token_count']:.6f}`",
                f"- mean_query_entropy: `{semantic_event['mean_query_entropy']:.6f}`",
                f"- mean_top_event_attribution: `{semantic_event['mean_top_event_attribution']:.6f}`",
                "",
                "| sample | top query | top query event offset s | top event attribution |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in semantic_event.get("samples", []):
            lines.append(
                f"| `{row['sample_id']}` | `{row['top_query_name']}` | "
                f"{row['top_query_event_offset_s']:.6f} | {row['top_event_attribution']:.6f} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class StageGSemanticEventSupportConfig:
    """Configuration for view-level semantic event support aggregation."""

    state_source: SemanticSupportStateSource = "hidden_with_projection_fallback"
    attention_temperature: float = 1.0
    event_score_bias_weight: float = 0.25
    event_top_k: int = 4
    event_score_quantile: float = 0.75
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.state_source not in {"hidden_with_projection_fallback", "hidden", "projection"}:
            raise ValueError("unsupported state_source for semantic event support.")
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")
        if self.event_score_bias_weight < 0:
            raise ValueError("event_score_bias_weight must be non-negative.")
        if self.event_top_k <= 0:
            raise ValueError("event_top_k must be positive.")
        if not 0.0 < self.event_score_quantile <= 1.0:
            raise ValueError("event_score_quantile must be in (0, 1].")
        if self.device not in TORCH_DEVICE_CHOICES:
            raise ValueError(f"device must be one of: {', '.join(TORCH_DEVICE_CHOICES)}.")


def build_stage_h_semantic_event_support(
    run_manifest_path: str,
    *,
    config: StageGSemanticEventSupportConfig | None = None,
) -> dict[str, object]:
    """Aggregate semantic event support across all views in one Stage H run."""

    resolved_config = config or StageGSemanticEventSupportConfig()
    run = load_stage_h_feature_run(run_manifest_path)
    resolved_device = resolve_torch_device_name(resolved_config.device)
    fusion = CausalEventFusion(
        CausalEventFusionConfig(
            attention_temperature=resolved_config.attention_temperature,
            event_score_bias_weight=resolved_config.event_score_bias_weight,
            event_top_k=resolved_config.event_top_k,
            event_score_quantile=resolved_config.event_score_quantile,
        )
    ).to(device=resolved_device)
    causal_fusion = CausalMaskedCrossModalFusion(
        CausalFusionConfig(
            attention_temperature=resolved_config.attention_temperature,
            event_bias_weight=resolved_config.event_score_bias_weight,
            use_causal_mask=True,
        )
    ).to(device=resolved_device)
    view_rows: list[dict[str, object]] = []
    query_names: list[str] = []
    event_token_counts: list[float] = []
    query_entropies: list[float] = []
    top_event_attributions: list[float] = []
    top_query_scores: list[float] = []
    sample_rows: list[dict[str, object]] = []
    attention_entropies: list[float] = []
    max_attentions: list[float] = []
    top_event_scores: list[float] = []
    top_contribution_scores: list[float] = []
    total_sample_count = 0
    for view in run.views:
        physiology_states, vehicle_states, state_source_used = _resolve_view_state_tensors(
            view,
            state_source=resolved_config.state_source,
            device=resolved_device,
        )
        attention_weights = torch.as_tensor(view.attention_weights, dtype=torch.float32, device=resolved_device)
        vehicle_event_scores = torch.as_tensor(view.vehicle_event_scores, dtype=torch.float32, device=resolved_device)
        if attention_weights.ndim != 3 or vehicle_event_scores.ndim != 2:
            with torch.no_grad():
                causal_output = causal_fusion(
                    CausalFusionTensorInput(
                        physiology_states=physiology_states,
                        vehicle_states=vehicle_states,
                        physiology_offsets_s=torch.as_tensor(view.reference_offsets_s, dtype=torch.float32, device=resolved_device),
                        vehicle_offsets_s=torch.as_tensor(view.reference_offsets_s, dtype=torch.float32, device=resolved_device),
                    )
                )
            attention_weights = causal_output.attention_weights
            vehicle_event_scores = causal_output.vehicle_event_scores
        semantic_input = SemanticEventTensorInput(
            physiology_states=physiology_states,
            vehicle_states=vehicle_states,
            attention_weights=attention_weights,
            vehicle_event_scores=vehicle_event_scores,
            vehicle_offsets_s=torch.as_tensor(view.reference_offsets_s, dtype=torch.float32, device=resolved_device),
        )
        with torch.no_grad():
            semantic_output = fusion(semantic_input)
        semantic_summary = _build_semantic_event_summary(
            sample_ids=view.sample_ids,
            semantic_output=semantic_output,
        )
        raw_attention = attention_weights
        raw_event_scores = vehicle_event_scores
        raw_contributions = raw_attention.sum(dim=1) * raw_event_scores
        raw_entropy = -(raw_attention * torch.log(torch.clamp(raw_attention, min=1e-12))).sum(dim=-1)
        query_names = semantic_summary["query_names"]
        sample_rows.extend(list(semantic_summary["samples"]))
        event_token_counts.append(float(semantic_summary["mean_event_token_count"]))
        query_entropies.append(float(semantic_summary["mean_query_entropy"]))
        top_query_scores.append(float(semantic_summary["mean_top_query_score"]))
        top_event_attributions.append(float(semantic_summary["mean_top_event_attribution"]))
        attention_entropies.append(float(raw_entropy.mean().detach().cpu()))
        max_attentions.append(float(raw_attention.max(dim=-1).values.mean().detach().cpu()))
        top_event_scores.append(float(raw_event_scores.max(dim=-1).values.mean().detach().cpu()))
        top_contribution_scores.append(float(raw_contributions.max(dim=-1).values.mean().detach().cpu()))
        total_sample_count += len(view.sample_ids)
        top_sample = max(
            semantic_summary["samples"],
            key=lambda row: float(row["top_event_attribution"]),
        )
        dominant_query_name = _mode_name(row["top_query_name"] for row in semantic_summary["samples"])
        view_rows.append(
            {
                "view_id": view.view_id,
                "sortie_id": view.sortie_id,
                "pilot_id": view.pilot_id,
                "source_summary_path": str(view.view_manifest["artifact_paths"].get("causal_fusion_summary_json") or ""),
                "state_source": state_source_used,
                "sample_count": len(view.sample_ids),
                "query_names": semantic_summary["query_names"],
                "mean_event_token_count": semantic_summary["mean_event_token_count"],
                "mean_query_entropy": semantic_summary["mean_query_entropy"],
                "mean_top_query_score": semantic_summary["mean_top_query_score"],
                "mean_top_event_attribution": semantic_summary["mean_top_event_attribution"],
                "mean_attention_entropy": float(raw_entropy.mean().detach().cpu()),
                "mean_max_attention": float(raw_attention.max(dim=-1).values.mean().detach().cpu()),
                "mean_top_event_score": float(raw_event_scores.max(dim=-1).values.mean().detach().cpu()),
                "mean_top_contribution_score": float(raw_contributions.max(dim=-1).values.mean().detach().cpu()),
                "dominant_query_name": dominant_query_name,
                "top_sample_id": top_sample["sample_id"],
                "top_sample_query_name": top_sample["top_query_name"],
                "top_sample_query_event_offset_s": top_sample["top_query_event_offset_s"],
                "top_sample_event_attribution": top_sample["top_event_attribution"],
            }
        )
    top_view = max(view_rows, key=lambda row: float(row["mean_top_event_attribution"])) if view_rows else None
    return {
        "run_manifest_path": str(run_manifest_path),
        "sample_count": total_sample_count,
        "mean_attention_entropy": _mean(tuple(attention_entropies)),
        "mean_max_attention": _mean(tuple(max_attentions)),
        "mean_top_event_score": _mean(tuple(top_event_scores)),
        "mean_top_contribution_score": _mean(tuple(top_contribution_scores)),
        "view_count": len(view_rows),
        "generated_view_ids": [view.view_id for view in run.views],
        "query_names": query_names,
        "query_count": len(query_names),
        "mean_event_token_count": _mean(tuple(event_token_counts)),
        "mean_query_entropy": _mean(tuple(query_entropies)),
        "mean_top_query_score": _mean(tuple(top_query_scores)),
        "mean_top_event_attribution": _mean(tuple(top_event_attributions)),
        "top_view_id": None if top_view is None else top_view["view_id"],
        "samples": sample_rows,
        "view_rows": view_rows,
    }


def _build_tensor_input(
    intermediate_export: AlignmentPreviewIntermediateExport,
    *,
    config: StageGCausalFusionConfig,
    device: str,
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
        physiology_states=torch.as_tensor(
            physiology_states,
            dtype=torch.float32,
            device=device,
        ),
        vehicle_states=torch.as_tensor(
            vehicle_states,
            dtype=torch.float32,
            device=device,
        ),
        physiology_offsets_s=torch.as_tensor(
            physiology_offsets,
            dtype=torch.float32,
            device=device,
        ),
        vehicle_offsets_s=torch.as_tensor(
            vehicle_offsets,
            dtype=torch.float32,
            device=device,
        ),
    )


def _resolve_view_state_tensors(
    view,
    *,
    state_source: SemanticSupportStateSource,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if state_source == "projection":
        return (
            torch.as_tensor(view.physiology_reference_projection, dtype=torch.float32, device=device),
            torch.as_tensor(view.vehicle_reference_projection, dtype=torch.float32, device=device),
            "projection",
        )
    if state_source == "hidden":
        if view.physiology_reference_hidden is None or view.vehicle_reference_hidden is None:
            raise ValueError(f"view {view.view_id} is missing hidden-state tensors.")
        return (
            torch.as_tensor(view.physiology_reference_hidden, dtype=torch.float32, device=device),
            torch.as_tensor(view.vehicle_reference_hidden, dtype=torch.float32, device=device),
            "hidden",
        )
    if view.physiology_reference_hidden is not None and view.vehicle_reference_hidden is not None:
        return (
            torch.as_tensor(view.physiology_reference_hidden, dtype=torch.float32, device=device),
            torch.as_tensor(view.vehicle_reference_hidden, dtype=torch.float32, device=device),
            "hidden",
        )
    return (
        torch.as_tensor(view.physiology_reference_projection, dtype=torch.float32, device=device),
        torch.as_tensor(view.vehicle_reference_projection, dtype=torch.float32, device=device),
        "projection_fallback",
    )


def _build_semantic_event_input(
    *,
    tensor_input: CausalFusionTensorInput,
    causal_output,
):
    from chronaris.models.fusion.semantic_event import SemanticEventTensorInput

    return SemanticEventTensorInput(
        physiology_states=tensor_input.physiology_states,
        vehicle_states=tensor_input.vehicle_states,
        attention_weights=causal_output.attention_weights,
        vehicle_event_scores=causal_output.vehicle_event_scores,
        vehicle_offsets_s=tensor_input.vehicle_offsets_s,
    )


def _build_semantic_event_summary(
    *,
    sample_ids: tuple[str, ...],
    semantic_output,
) -> dict[str, object]:
    query_entropy = semantic_query_entropy(
        semantic_output.query_to_event_attention,
        semantic_output.event_token_mask,
    )
    token_count = semantic_output.event_token_mask.sum(dim=-1)
    samples: list[dict[str, object]] = []
    top_query_scores, top_event_scores = [], []
    for sample_index, sample_id in enumerate(sample_ids):
        query_index = int(torch.argmax(semantic_output.query_attribution_scores[sample_index]).detach().cpu())
        event_index = int(torch.argmax(semantic_output.event_attribution_scores[sample_index]).detach().cpu())
        top_query_scores.append(float(semantic_output.query_attribution_scores[sample_index, query_index].detach().cpu()))
        top_event_scores.append(float(semantic_output.event_attribution_scores[sample_index, event_index].detach().cpu()))
        samples.append(
            {
                "sample_id": sample_id,
                "event_token_count": int(token_count[sample_index].detach().cpu()),
                "top_query_name": semantic_output.query_names[query_index],
                "top_query_score": float(semantic_output.query_attribution_scores[sample_index, query_index].detach().cpu()),
                "top_query_event_offset_s": float(
                    semantic_output.event_token_center_offsets_s[sample_index, event_index].detach().cpu()
                ),
                "top_event_attribution": float(
                    semantic_output.event_attribution_scores[sample_index, event_index].detach().cpu()
                ),
            }
        )
    return {
        "query_names": list(semantic_output.query_names),
        "query_count": len(semantic_output.query_names),
        "mean_event_token_count": float(token_count.to(dtype=torch.float32).mean().detach().cpu()),
        "mean_query_entropy": float(query_entropy.mean().detach().cpu()),
        "mean_top_query_score": _mean(tuple(top_query_scores)),
        "mean_top_event_attribution": _mean(tuple(top_event_scores)),
        "samples": samples,
    }


def _tensor_1d_to_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu().tolist())


def _tensor_2d_to_tuple(values: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values.detach().cpu().tolist())


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mode_name(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
