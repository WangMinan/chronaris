"""Phase 2 case-study metrics over frozen Stage H assets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from chronaris.features.stage_i_case import StageICaseStudyViewInput, StageICaseStudyWindowRow
from chronaris.models.fusion import CausalFusionConfig, CausalFusionTensorInput, CausalMaskedCrossModalFusion, attention_entropy


@dataclass(frozen=True, slots=True)
class StageICaseStudySampleMetric:
    """Per-window metrics from one bundle-only causal-fusion pass."""

    sample_id: str
    top_event_offset_s: float
    top_event_score: float
    top_contribution_offset_s: float
    top_contribution_score: float
    mean_attention_entropy: float
    mean_max_attention: float
    mean_fused_l2_norm: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "top_event_offset_s": self.top_event_offset_s,
            "top_event_score": self.top_event_score,
            "top_contribution_offset_s": self.top_contribution_offset_s,
            "top_contribution_score": self.top_contribution_score,
            "mean_attention_entropy": self.mean_attention_entropy,
            "mean_max_attention": self.mean_max_attention,
            "mean_fused_l2_norm": self.mean_fused_l2_norm,
        }


@dataclass(frozen=True, slots=True)
class StageICaseStudyAblationMetrics:
    """Aggregated metrics for one Phase 2 ablation path."""

    name: str
    sample_count: int
    reference_point_count: int
    mean_attention_entropy: float
    mean_max_attention: float
    mean_causal_option_count: float
    mean_top_event_score: float
    mean_top_contribution_score: float
    mean_fused_l2_norm: float
    mean_fused_cosine_to_projection_baseline: float
    delta_mean_attention_entropy: float
    delta_mean_max_attention: float
    delta_mean_top_event_score: float
    delta_mean_top_contribution_score: float
    delta_fused_l2_norm: float
    delta_fused_cosine_to_projection_baseline: float
    sample_metrics: tuple[StageICaseStudySampleMetric, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "reference_point_count": self.reference_point_count,
            "mean_attention_entropy": self.mean_attention_entropy,
            "mean_max_attention": self.mean_max_attention,
            "mean_causal_option_count": self.mean_causal_option_count,
            "mean_top_event_score": self.mean_top_event_score,
            "mean_top_contribution_score": self.mean_top_contribution_score,
            "mean_fused_l2_norm": self.mean_fused_l2_norm,
            "mean_fused_cosine_to_projection_baseline": self.mean_fused_cosine_to_projection_baseline,
            "delta_mean_attention_entropy": self.delta_mean_attention_entropy,
            "delta_mean_max_attention": self.delta_mean_max_attention,
            "delta_mean_top_event_score": self.delta_mean_top_event_score,
            "delta_mean_top_contribution_score": self.delta_mean_top_contribution_score,
            "delta_fused_l2_norm": self.delta_fused_l2_norm,
            "delta_fused_cosine_to_projection_baseline": self.delta_fused_cosine_to_projection_baseline,
            "sample_metrics": [sample.to_dict() for sample in self.sample_metrics],
        }


@dataclass(frozen=True, slots=True)
class StageICaseStudyWindowRanking:
    """One top-ranked case-study window."""

    view_id: str
    sortie_id: str
    pilot_id: int
    sample_id: str
    window_index: int
    start_offset_ms: int
    end_offset_ms: int
    selected_for_model: bool
    top_event_offset_s: float
    top_event_score: float
    top_contribution_offset_s: float
    top_contribution_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "sortie_id": self.sortie_id,
            "pilot_id": self.pilot_id,
            "sample_id": self.sample_id,
            "window_index": self.window_index,
            "start_offset_ms": self.start_offset_ms,
            "end_offset_ms": self.end_offset_ms,
            "selected_for_model": self.selected_for_model,
            "top_event_offset_s": self.top_event_offset_s,
            "top_event_score": self.top_event_score,
            "top_contribution_offset_s": self.top_contribution_offset_s,
            "top_contribution_score": self.top_contribution_score,
        }


@dataclass(frozen=True, slots=True)
class StageICaseStudyViewSummary:
    """One view-level Phase 2 summary row."""

    view_id: str
    sortie_id: str
    pilot_id: int
    verdict: str
    window_count: int
    selected_window_count: int
    case_partition_sample_count: int
    mean_projection_cosine: float
    projection_cosine_cv: float
    mean_projection_l2_gap: float
    projection_l2_gap_cv: float
    mean_attention_entropy: float
    mean_max_attention: float
    mean_top_event_score: float
    mean_top_contribution_score: float
    exported_hidden_fused_vs_projection_baseline_cosine: float

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "sortie_id": self.sortie_id,
            "pilot_id": self.pilot_id,
            "verdict": self.verdict,
            "window_count": self.window_count,
            "selected_window_count": self.selected_window_count,
            "case_partition_sample_count": self.case_partition_sample_count,
            "mean_projection_cosine": self.mean_projection_cosine,
            "projection_cosine_cv": self.projection_cosine_cv,
            "mean_projection_l2_gap": self.mean_projection_l2_gap,
            "projection_l2_gap_cv": self.projection_l2_gap_cv,
            "mean_attention_entropy": self.mean_attention_entropy,
            "mean_max_attention": self.mean_max_attention,
            "mean_top_event_score": self.mean_top_event_score,
            "mean_top_contribution_score": self.mean_top_contribution_score,
            "exported_hidden_fused_vs_projection_baseline_cosine": self.exported_hidden_fused_vs_projection_baseline_cosine,
        }


@dataclass(frozen=True, slots=True)
class StageICaseStudyPilotComparison:
    """Comparison between two views from the same sortie."""

    sortie_id: str
    reference_view_id: str
    comparison_view_id: str
    reference_pilot_id: int
    comparison_pilot_id: int
    reference_verdict: str
    comparison_verdict: str
    delta_mean_projection_cosine: float
    delta_projection_cosine_cv: float
    delta_mean_projection_l2_gap: float
    delta_projection_l2_gap_cv: float
    delta_mean_attention_entropy: float
    delta_mean_top_contribution_score: float
    delta_exported_hidden_fused_vs_projection_baseline_cosine: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sortie_id": self.sortie_id,
            "reference_view_id": self.reference_view_id,
            "comparison_view_id": self.comparison_view_id,
            "reference_pilot_id": self.reference_pilot_id,
            "comparison_pilot_id": self.comparison_pilot_id,
            "reference_verdict": self.reference_verdict,
            "comparison_verdict": self.comparison_verdict,
            "delta_mean_projection_cosine": self.delta_mean_projection_cosine,
            "delta_projection_cosine_cv": self.delta_projection_cosine_cv,
            "delta_mean_projection_l2_gap": self.delta_mean_projection_l2_gap,
            "delta_projection_l2_gap_cv": self.delta_projection_l2_gap_cv,
            "delta_mean_attention_entropy": self.delta_mean_attention_entropy,
            "delta_mean_top_contribution_score": self.delta_mean_top_contribution_score,
            "delta_exported_hidden_fused_vs_projection_baseline_cosine": self.delta_exported_hidden_fused_vs_projection_baseline_cosine,
        }


def compute_case_study_ablations(
    view: StageICaseStudyViewInput,
) -> tuple[tuple[StageICaseStudyAblationMetrics, ...], np.ndarray]:
    """Run the fixed Phase 2 bundle-only ablation family for one view."""

    physiology_states = np.asarray(view.stage_h_view.physiology_reference_projection, dtype=np.float32)
    vehicle_states = np.asarray(view.stage_h_view.vehicle_reference_projection, dtype=np.float32)
    offsets = np.asarray(view.stage_h_view.reference_offsets_s, dtype=np.float32)

    baseline = _run_one_ablation(
        name="projection_refusion_baseline",
        sample_ids=view.sample_ids,
        physiology_states=physiology_states,
        vehicle_states=vehicle_states,
        physiology_offsets_s=offsets,
        vehicle_offsets_s=offsets,
        config=CausalFusionConfig(
            attention_temperature=1.0,
            event_bias_weight=0.25,
            normalize_states=True,
        ),
    )
    no_event_bias = _run_one_ablation(
        name="no_event_bias",
        sample_ids=view.sample_ids,
        physiology_states=physiology_states,
        vehicle_states=vehicle_states,
        physiology_offsets_s=offsets,
        vehicle_offsets_s=offsets,
        config=CausalFusionConfig(
            attention_temperature=1.0,
            event_bias_weight=0.0,
            normalize_states=True,
        ),
        baseline_fused=baseline["fused_states"],
    )
    no_state_normalization = _run_one_ablation(
        name="no_state_normalization",
        sample_ids=view.sample_ids,
        physiology_states=physiology_states,
        vehicle_states=vehicle_states,
        physiology_offsets_s=offsets,
        vehicle_offsets_s=offsets,
        config=CausalFusionConfig(
            attention_temperature=1.0,
            event_bias_weight=0.25,
            normalize_states=False,
        ),
        baseline_fused=baseline["fused_states"],
    )
    first_vehicle = np.repeat(vehicle_states[:, :1, :], repeats=vehicle_states.shape[1], axis=1)
    vehicle_delta_suppressed = _run_one_ablation(
        name="vehicle_delta_suppressed",
        sample_ids=view.sample_ids,
        physiology_states=physiology_states,
        vehicle_states=first_vehicle,
        physiology_offsets_s=offsets,
        vehicle_offsets_s=offsets,
        config=CausalFusionConfig(
            attention_temperature=1.0,
            event_bias_weight=0.25,
            normalize_states=True,
        ),
        baseline_fused=baseline["fused_states"],
    )
    baseline_metrics = _to_ablation_metrics(
        name="projection_refusion_baseline",
        sample_metrics=baseline["sample_metrics"],
        aggregate=baseline["aggregate"],
        baseline_aggregate=baseline["aggregate"],
        baseline_fused=baseline["fused_states"],
        current_fused=baseline["fused_states"],
    )
    return (
        (
            baseline_metrics,
            _to_ablation_metrics(
                name="no_event_bias",
                sample_metrics=no_event_bias["sample_metrics"],
                aggregate=no_event_bias["aggregate"],
                baseline_aggregate=baseline["aggregate"],
                baseline_fused=baseline["fused_states"],
                current_fused=no_event_bias["fused_states"],
            ),
            _to_ablation_metrics(
                name="no_state_normalization",
                sample_metrics=no_state_normalization["sample_metrics"],
                aggregate=no_state_normalization["aggregate"],
                baseline_aggregate=baseline["aggregate"],
                baseline_fused=baseline["fused_states"],
                current_fused=no_state_normalization["fused_states"],
            ),
            _to_ablation_metrics(
                name="vehicle_delta_suppressed",
                sample_metrics=vehicle_delta_suppressed["sample_metrics"],
                aggregate=vehicle_delta_suppressed["aggregate"],
                baseline_aggregate=baseline["aggregate"],
                baseline_fused=baseline["fused_states"],
                current_fused=vehicle_delta_suppressed["fused_states"],
            ),
        ),
        baseline["fused_states"],
    )


def build_view_summary(
    view: StageICaseStudyViewInput,
    *,
    baseline: StageICaseStudyAblationMetrics,
    baseline_fused_states: np.ndarray,
) -> StageICaseStudyViewSummary:
    """Summarize one view for the Phase 2 report."""

    projection_summary = view.projection_summary.get("summary", {})
    return StageICaseStudyViewSummary(
        view_id=view.view_id,
        sortie_id=view.sortie_id,
        pilot_id=view.pilot_id,
        verdict=view.projection_diagnostics_verdict,
        window_count=view.window_count,
        selected_window_count=view.selected_window_count,
        case_partition_sample_count=view.case_partition_sample_count,
        mean_projection_cosine=float(projection_summary.get("mean_projection_cosine", 0.0)),
        projection_cosine_cv=float(projection_summary.get("cv_projection_cosine", 0.0)),
        mean_projection_l2_gap=float(projection_summary.get("mean_projection_l2_gap", 0.0)),
        projection_l2_gap_cv=float(projection_summary.get("cv_projection_l2_gap", 0.0)),
        mean_attention_entropy=baseline.mean_attention_entropy,
        mean_max_attention=baseline.mean_max_attention,
        mean_top_event_score=baseline.mean_top_event_score,
        mean_top_contribution_score=baseline.mean_top_contribution_score,
        exported_hidden_fused_vs_projection_baseline_cosine=_mean_hidden_vs_projection_cosine(
            exported_fused=np.asarray(view.stage_h_view.fused_representation, dtype=np.float32),
            rerun_baseline_fused=baseline_fused_states,
        ),
    )


def build_window_rankings(
    view: StageICaseStudyViewInput,
    *,
    baseline: StageICaseStudyAblationMetrics,
    top_k: int,
) -> tuple[StageICaseStudyWindowRanking, ...]:
    """Return top-k windows ranked by baseline top contribution score."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    row_by_sample_id = {row.sample_id: row for row in view.case_window_rows}
    ranked = sorted(
        baseline.sample_metrics,
        key=lambda item: item.top_contribution_score,
        reverse=True,
    )[:top_k]
    result: list[StageICaseStudyWindowRanking] = []
    for sample_metric in ranked:
        row = row_by_sample_id[sample_metric.sample_id]
        result.append(
            StageICaseStudyWindowRanking(
                view_id=view.view_id,
                sortie_id=view.sortie_id,
                pilot_id=view.pilot_id,
                sample_id=sample_metric.sample_id,
                window_index=row.window_index,
                start_offset_ms=row.start_offset_ms,
                end_offset_ms=row.end_offset_ms,
                selected_for_model=row.selected_for_model,
                top_event_offset_s=sample_metric.top_event_offset_s,
                top_event_score=sample_metric.top_event_score,
                top_contribution_offset_s=sample_metric.top_contribution_offset_s,
                top_contribution_score=sample_metric.top_contribution_score,
            )
        )
    return tuple(result)


def build_pilot_comparisons(
    view_summaries: Sequence[StageICaseStudyViewSummary],
) -> tuple[StageICaseStudyPilotComparison, ...]:
    """Build same-sortie multi-pilot comparison rows."""

    by_sortie: dict[str, list[StageICaseStudyViewSummary]] = {}
    for summary in view_summaries:
        by_sortie.setdefault(summary.sortie_id, []).append(summary)
    comparisons: list[StageICaseStudyPilotComparison] = []
    for sortie_id, summaries in by_sortie.items():
        if len(summaries) < 2:
            continue
        ordered = sorted(
            summaries,
            key=lambda item: (0 if item.verdict == "PASS" else 1, item.pilot_id),
        )
        reference = ordered[0]
        for current in ordered[1:]:
            comparisons.append(
                StageICaseStudyPilotComparison(
                    sortie_id=sortie_id,
                    reference_view_id=reference.view_id,
                    comparison_view_id=current.view_id,
                    reference_pilot_id=reference.pilot_id,
                    comparison_pilot_id=current.pilot_id,
                    reference_verdict=reference.verdict,
                    comparison_verdict=current.verdict,
                    delta_mean_projection_cosine=current.mean_projection_cosine - reference.mean_projection_cosine,
                    delta_projection_cosine_cv=current.projection_cosine_cv - reference.projection_cosine_cv,
                    delta_mean_projection_l2_gap=current.mean_projection_l2_gap - reference.mean_projection_l2_gap,
                    delta_projection_l2_gap_cv=current.projection_l2_gap_cv - reference.projection_l2_gap_cv,
                    delta_mean_attention_entropy=current.mean_attention_entropy - reference.mean_attention_entropy,
                    delta_mean_top_contribution_score=current.mean_top_contribution_score - reference.mean_top_contribution_score,
                    delta_exported_hidden_fused_vs_projection_baseline_cosine=(
                        current.exported_hidden_fused_vs_projection_baseline_cosine
                        - reference.exported_hidden_fused_vs_projection_baseline_cosine
                    ),
                )
            )
    return tuple(comparisons)


def explain_warn_view(
    view: StageICaseStudyViewInput,
    *,
    view_summary: StageICaseStudyViewSummary,
    ablations: Sequence[StageICaseStudyAblationMetrics],
    pilot_comparisons: Sequence[StageICaseStudyPilotComparison],
) -> str:
    """Render a compact explanation for one WARN view."""

    failing_checks = tuple(
        check
        for check in view.threshold_evaluation.get("checks", ())
        if isinstance(check, Mapping) and not bool(check.get("passed"))
    )
    lines = [f"- 该视图 `verdict={view.projection_diagnostics_verdict}`。"]
    if failing_checks:
        rendered = ", ".join(
            f"{check['name']}={float(check['actual']):.6f} {check['operator']} {float(check['expected']):.6f}"
            for check in failing_checks
        )
        lines.append(f"- 触发的阈值项：{rendered}。")
    related_comparison = next(
        (
            item
            for item in pilot_comparisons
            if item.comparison_view_id == view.view_id or item.reference_view_id == view.view_id
        ),
        None,
    )
    if related_comparison is not None:
        lines.append(
            "- 同 sortie 对比显示："
            f"projection cosine 差值 `{related_comparison.delta_mean_projection_cosine:+.6f}`，"
            f"projection cosine CV 差值 `{related_comparison.delta_projection_cosine_cv:+.6f}`，"
            f"projection L2 gap CV 差值 `{related_comparison.delta_projection_l2_gap_cv:+.6f}`。"
        )
    nonbaseline = tuple(item for item in ablations if item.name != "projection_refusion_baseline")
    if nonbaseline:
        most_sensitive = max(nonbaseline, key=lambda item: abs(item.delta_mean_top_contribution_score))
        lines.append(
            "- 在 bundle-only 消融中，对 top contribution 影响最大的路径为 "
            f"`{most_sensitive.name}`，delta=`{most_sensitive.delta_mean_top_contribution_score:+.6f}`。"
        )
    lines.append(
        "- 这说明该视图不是导出失败，而是对投影一致性与事件驱动变化更敏感，需要在 Phase 3 与第二公开数据集结果一起综合解释。"
    )
    return "\n".join(lines)


def _run_one_ablation(
    *,
    name: str,
    sample_ids: Sequence[str],
    physiology_states: np.ndarray,
    vehicle_states: np.ndarray,
    physiology_offsets_s: np.ndarray,
    vehicle_offsets_s: np.ndarray,
    config: CausalFusionConfig,
    baseline_fused: np.ndarray | None = None,
) -> dict[str, object]:
    model = CausalMaskedCrossModalFusion(config)
    inputs = CausalFusionTensorInput(
        physiology_states=torch.from_numpy(physiology_states),
        vehicle_states=torch.from_numpy(vehicle_states),
        physiology_offsets_s=torch.from_numpy(physiology_offsets_s),
        vehicle_offsets_s=torch.from_numpy(vehicle_offsets_s),
    )
    with torch.no_grad():
        output = model(inputs)
    event_scores = output.vehicle_event_scores
    contribution_scores = output.attention_weights.sum(dim=1) * output.vehicle_event_scores
    entropy = attention_entropy(output.attention_weights, output.causal_mask)
    max_attention = output.attention_weights.max(dim=-1).values
    fused_states = output.fused_states.detach().cpu().numpy()
    sample_metrics: list[StageICaseStudySampleMetric] = []
    for sample_index, sample_id in enumerate(sample_ids):
        top_event_index = int(torch.argmax(event_scores[sample_index]).detach().cpu())
        top_contribution_index = int(torch.argmax(contribution_scores[sample_index]).detach().cpu())
        sample_metrics.append(
            StageICaseStudySampleMetric(
                sample_id=str(sample_id),
                top_event_offset_s=float(vehicle_offsets_s[sample_index, top_event_index]),
                top_event_score=float(event_scores[sample_index, top_event_index].detach().cpu()),
                top_contribution_offset_s=float(vehicle_offsets_s[sample_index, top_contribution_index]),
                top_contribution_score=float(contribution_scores[sample_index, top_contribution_index].detach().cpu()),
                mean_attention_entropy=float(entropy[sample_index].mean().detach().cpu()),
                mean_max_attention=float(max_attention[sample_index].mean().detach().cpu()),
                mean_fused_l2_norm=float(
                    torch.linalg.vector_norm(output.fused_states[sample_index], dim=-1).mean().detach().cpu()
                ),
            )
        )
    aggregate = {
        "sample_count": len(sample_metrics),
        "reference_point_count": int(output.fused_states.shape[1]),
        "mean_attention_entropy": _mean(sample.mean_attention_entropy for sample in sample_metrics),
        "mean_max_attention": _mean(sample.mean_max_attention for sample in sample_metrics),
        "mean_causal_option_count": float(output.causal_mask.sum(dim=-1).to(dtype=torch.float32).mean().detach().cpu()),
        "mean_top_event_score": _mean(sample.top_event_score for sample in sample_metrics),
        "mean_top_contribution_score": _mean(sample.top_contribution_score for sample in sample_metrics),
        "mean_fused_l2_norm": _mean(sample.mean_fused_l2_norm for sample in sample_metrics),
    }
    if baseline_fused is None:
        mean_cosine_to_baseline = 1.0
    else:
        mean_cosine_to_baseline = _mean_state_cosine(
            left=fused_states,
            right=baseline_fused,
        )
    aggregate["mean_fused_cosine_to_projection_baseline"] = mean_cosine_to_baseline
    return {
        "name": name,
        "sample_metrics": tuple(sample_metrics),
        "aggregate": aggregate,
        "fused_states": fused_states,
    }


def _to_ablation_metrics(
    *,
    name: str,
    sample_metrics: Sequence[StageICaseStudySampleMetric],
    aggregate: Mapping[str, float],
    baseline_aggregate: Mapping[str, float],
    baseline_fused: np.ndarray,
    current_fused: np.ndarray,
) -> StageICaseStudyAblationMetrics:
    return StageICaseStudyAblationMetrics(
        name=name,
        sample_count=int(aggregate["sample_count"]),
        reference_point_count=int(aggregate["reference_point_count"]),
        mean_attention_entropy=float(aggregate["mean_attention_entropy"]),
        mean_max_attention=float(aggregate["mean_max_attention"]),
        mean_causal_option_count=float(aggregate["mean_causal_option_count"]),
        mean_top_event_score=float(aggregate["mean_top_event_score"]),
        mean_top_contribution_score=float(aggregate["mean_top_contribution_score"]),
        mean_fused_l2_norm=float(aggregate["mean_fused_l2_norm"]),
        mean_fused_cosine_to_projection_baseline=float(aggregate["mean_fused_cosine_to_projection_baseline"]),
        delta_mean_attention_entropy=float(aggregate["mean_attention_entropy"] - baseline_aggregate["mean_attention_entropy"]),
        delta_mean_max_attention=float(aggregate["mean_max_attention"] - baseline_aggregate["mean_max_attention"]),
        delta_mean_top_event_score=float(aggregate["mean_top_event_score"] - baseline_aggregate["mean_top_event_score"]),
        delta_mean_top_contribution_score=float(
            aggregate["mean_top_contribution_score"] - baseline_aggregate["mean_top_contribution_score"]
        ),
        delta_fused_l2_norm=float(aggregate["mean_fused_l2_norm"] - baseline_aggregate["mean_fused_l2_norm"]),
        delta_fused_cosine_to_projection_baseline=float(
            aggregate["mean_fused_cosine_to_projection_baseline"] - 1.0
        ),
        sample_metrics=tuple(sample_metrics),
    )


def _mean_state_cosine(*, left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("left and right fused states must share the same shape.")
    left_flat = left.reshape(-1, left.shape[-1])
    right_flat = right.reshape(-1, right.shape[-1])
    numerator = np.sum(left_flat * right_flat, axis=-1)
    denominator = np.linalg.norm(left_flat, axis=-1) * np.linalg.norm(right_flat, axis=-1)
    safe_denominator = np.where(denominator > 0.0, denominator, 1.0)
    cosine = np.where(denominator > 0.0, numerator / safe_denominator, 0.0)
    return float(np.mean(cosine))


def _mean_hidden_vs_projection_cosine(
    *,
    exported_fused: np.ndarray,
    rerun_baseline_fused: np.ndarray,
) -> float:
    if exported_fused.shape[-1] != rerun_baseline_fused.shape[-1]:
        exported_profile = np.linalg.norm(exported_fused, axis=-1)
        rerun_profile = np.linalg.norm(rerun_baseline_fused, axis=-1)
        return _mean_profile_cosine(exported_profile, rerun_profile)
    return _mean_state_cosine(left=exported_fused, right=rerun_baseline_fused)


def _mean_profile_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("left and right norm profiles must share the same shape.")
    left_flat = left.reshape(1, -1)
    right_flat = right.reshape(1, -1)
    numerator = np.sum(left_flat * right_flat, axis=-1)
    denominator = np.linalg.norm(left_flat, axis=-1) * np.linalg.norm(right_flat, axis=-1)
    safe_denominator = np.where(denominator > 0.0, denominator, 1.0)
    cosine = np.where(denominator > 0.0, numerator / safe_denominator, 0.0)
    return float(np.mean(cosine))


def _mean(values: Sequence[float] | Mapping | object) -> float:
    if isinstance(values, Mapping):
        seq = tuple(float(value) for value in values.values())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        seq = tuple(float(value) for value in values)
    else:
        seq = tuple(float(value) for value in values)
    if not seq:
        return 0.0
    return float(sum(seq) / len(seq))
