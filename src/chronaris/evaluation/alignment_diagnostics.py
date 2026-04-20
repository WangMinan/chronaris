"""Sample-level alignment diagnostics on shared reference projections."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampleAlignmentProjectionDiagnostic:
    """Sample-level projection diagnostics on the shared reference grid."""

    sample_id: str
    reference_point_count: int
    mean_projection_cosine: float
    min_projection_cosine: float
    max_projection_cosine: float
    physiology_projection_l2_mean: float
    vehicle_projection_l2_mean: float
    projection_l2_gap_mean: float
    projection_l2_ratio_mean: float


@dataclass(frozen=True, slots=True)
class AlignmentProjectionDiagnosticsSummary:
    """Aggregated diagnostics over an intermediate export partition."""

    sample_count: int
    reference_point_count: int
    mean_projection_cosine: float
    min_projection_cosine: float
    max_projection_cosine: float
    mean_projection_l2_gap: float
    mean_projection_l2_ratio: float
    samples: tuple[SampleAlignmentProjectionDiagnostic, ...]


def summarize_alignment_projection_diagnostics(
    intermediate_export,
    *,
    ratio_epsilon: float = 1e-6,
) -> AlignmentProjectionDiagnosticsSummary:
    """Build sample-level diagnostics from one intermediate export object."""

    if ratio_epsilon <= 0:
        raise ValueError("ratio_epsilon must be positive.")
    if intermediate_export is None or not getattr(intermediate_export, "samples", ()):
        return AlignmentProjectionDiagnosticsSummary(
            sample_count=0,
            reference_point_count=0,
            mean_projection_cosine=0.0,
            min_projection_cosine=0.0,
            max_projection_cosine=0.0,
            mean_projection_l2_gap=0.0,
            mean_projection_l2_ratio=0.0,
            samples=(),
        )

    sample_diagnostics = tuple(
        _summarize_one_sample(sample, ratio_epsilon=ratio_epsilon)
        for sample in intermediate_export.samples
    )
    return AlignmentProjectionDiagnosticsSummary(
        sample_count=len(sample_diagnostics),
        reference_point_count=int(getattr(intermediate_export, "reference_point_count", 0)),
        mean_projection_cosine=_mean(item.mean_projection_cosine for item in sample_diagnostics),
        min_projection_cosine=min((item.min_projection_cosine for item in sample_diagnostics), default=0.0),
        max_projection_cosine=max((item.max_projection_cosine for item in sample_diagnostics), default=0.0),
        mean_projection_l2_gap=_mean(item.projection_l2_gap_mean for item in sample_diagnostics),
        mean_projection_l2_ratio=_mean(item.projection_l2_ratio_mean for item in sample_diagnostics),
        samples=sample_diagnostics,
    )


def render_alignment_projection_diagnostics_markdown(
    diagnostics: AlignmentProjectionDiagnosticsSummary,
    *,
    title: str = "Sample-Level Projection Diagnostics",
    max_samples: int = 10,
) -> str:
    """Render diagnostics as a compact Markdown section."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    lines = [
        f"## {title}",
        "",
        f"- sample count: `{diagnostics.sample_count}`",
        f"- reference point count: `{diagnostics.reference_point_count}`",
        f"- mean projection cosine: `{diagnostics.mean_projection_cosine:.6f}`",
        f"- min projection cosine: `{diagnostics.min_projection_cosine:.6f}`",
        f"- max projection cosine: `{diagnostics.max_projection_cosine:.6f}`",
        f"- mean projection L2 gap: `{diagnostics.mean_projection_l2_gap:.6f}`",
        f"- mean projection L2 ratio (vehicle/physiology): `{diagnostics.mean_projection_l2_ratio:.6f}`",
        "",
    ]
    if not diagnostics.samples:
        lines.append("- no exported samples")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| sample id | mean cosine | min cosine | max cosine | mean L2 gap | mean L2 ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for sample in diagnostics.samples[:max_samples]:
        lines.append(
            "| "
            f"{sample.sample_id} | "
            f"{sample.mean_projection_cosine:.6f} | "
            f"{sample.min_projection_cosine:.6f} | "
            f"{sample.max_projection_cosine:.6f} | "
            f"{sample.projection_l2_gap_mean:.6f} | "
            f"{sample.projection_l2_ratio_mean:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summarize_one_sample(
    sample_intermediate,
    *,
    ratio_epsilon: float,
) -> SampleAlignmentProjectionDiagnostic:
    physiology_rows = tuple(sample_intermediate.physiology.reference_projected_states)
    vehicle_rows = tuple(sample_intermediate.vehicle.reference_projected_states)
    point_count = min(len(physiology_rows), len(vehicle_rows))
    if point_count <= 0:
        return SampleAlignmentProjectionDiagnostic(
            sample_id=sample_intermediate.sample_id,
            reference_point_count=0,
            mean_projection_cosine=0.0,
            min_projection_cosine=0.0,
            max_projection_cosine=0.0,
            physiology_projection_l2_mean=0.0,
            vehicle_projection_l2_mean=0.0,
            projection_l2_gap_mean=0.0,
            projection_l2_ratio_mean=0.0,
        )

    cosines: list[float] = []
    physiology_l2: list[float] = []
    vehicle_l2: list[float] = []
    l2_gap: list[float] = []
    l2_ratio: list[float] = []

    for physiology_row, vehicle_row in zip(physiology_rows[:point_count], vehicle_rows[:point_count]):
        phys_norm = _l2_norm(physiology_row)
        veh_norm = _l2_norm(vehicle_row)
        cosines.append(_cosine_similarity(physiology_row, vehicle_row))
        physiology_l2.append(phys_norm)
        vehicle_l2.append(veh_norm)
        l2_gap.append(abs(veh_norm - phys_norm))
        l2_ratio.append(veh_norm / max(phys_norm, ratio_epsilon))

    return SampleAlignmentProjectionDiagnostic(
        sample_id=sample_intermediate.sample_id,
        reference_point_count=point_count,
        mean_projection_cosine=_mean(cosines),
        min_projection_cosine=min(cosines),
        max_projection_cosine=max(cosines),
        physiology_projection_l2_mean=_mean(physiology_l2),
        vehicle_projection_l2_mean=_mean(vehicle_l2),
        projection_l2_gap_mean=_mean(l2_gap),
        projection_l2_ratio_mean=_mean(l2_ratio),
    )


def _mean(values) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return sum(values_tuple) / len(values_tuple)


def _l2_norm(values) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _cosine_similarity(left, right) -> float:
    numerator = 0.0
    left_sum = 0.0
    right_sum = 0.0
    for left_value, right_value in zip(left, right):
        left_value_f = float(left_value)
        right_value_f = float(right_value)
        numerator += left_value_f * right_value_f
        left_sum += left_value_f * left_value_f
        right_sum += right_value_f * right_value_f

    denominator = math.sqrt(left_sum * right_sum)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator
