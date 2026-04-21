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
    std_projection_cosine: float
    cv_projection_cosine: float
    std_projection_l2_gap: float
    cv_projection_l2_gap: float
    std_projection_l2_ratio: float
    cv_projection_l2_ratio: float
    samples: tuple[SampleAlignmentProjectionDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class AlignmentProjectionThresholdConfig:
    """Threshold rules used by Stage E projection diagnostics."""

    min_sample_count: int = 1
    min_mean_projection_cosine: float = 0.65
    enforce_min_projection_cosine: bool = False
    min_min_projection_cosine: float = 0.10
    max_mean_projection_l2_gap: float = 0.25
    max_mean_projection_l2_ratio_deviation: float = 0.30
    max_projection_cosine_cv: float = 0.15
    max_projection_l2_gap_cv: float = 0.25

    def __post_init__(self) -> None:
        if self.min_sample_count <= 0:
            raise ValueError("min_sample_count must be positive.")
        if not 0.0 <= self.min_mean_projection_cosine <= 1.0:
            raise ValueError("min_mean_projection_cosine must be between 0 and 1.")
        if not 0.0 <= self.min_min_projection_cosine <= 1.0:
            raise ValueError("min_min_projection_cosine must be between 0 and 1.")
        if self.max_mean_projection_l2_gap < 0.0:
            raise ValueError("max_mean_projection_l2_gap must be non-negative.")
        if self.max_mean_projection_l2_ratio_deviation < 0.0:
            raise ValueError("max_mean_projection_l2_ratio_deviation must be non-negative.")
        if self.max_projection_cosine_cv < 0.0:
            raise ValueError("max_projection_cosine_cv must be non-negative.")
        if self.max_projection_l2_gap_cv < 0.0:
            raise ValueError("max_projection_l2_gap_cv must be non-negative.")


@dataclass(frozen=True, slots=True)
class AlignmentProjectionThresholdCheck:
    """One threshold check over a diagnostics summary."""

    name: str
    passed: bool
    actual: float
    operator: str
    expected: float


@dataclass(frozen=True, slots=True)
class AlignmentProjectionThresholdEvaluation:
    """Full threshold evaluation result for one diagnostics summary."""

    passed: bool
    verdict: str
    checks: tuple[AlignmentProjectionThresholdCheck, ...]


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
            std_projection_cosine=0.0,
            cv_projection_cosine=0.0,
            std_projection_l2_gap=0.0,
            cv_projection_l2_gap=0.0,
            std_projection_l2_ratio=0.0,
            cv_projection_l2_ratio=0.0,
            samples=(),
        )

    sample_diagnostics = tuple(
        _summarize_one_sample(sample, ratio_epsilon=ratio_epsilon)
        for sample in intermediate_export.samples
    )
    cosine_values = tuple(item.mean_projection_cosine for item in sample_diagnostics)
    l2_gap_values = tuple(item.projection_l2_gap_mean for item in sample_diagnostics)
    l2_ratio_values = tuple(item.projection_l2_ratio_mean for item in sample_diagnostics)
    mean_projection_cosine = _mean(cosine_values)
    mean_projection_l2_gap = _mean(l2_gap_values)
    mean_projection_l2_ratio = _mean(l2_ratio_values)

    return AlignmentProjectionDiagnosticsSummary(
        sample_count=len(sample_diagnostics),
        reference_point_count=int(getattr(intermediate_export, "reference_point_count", 0)),
        mean_projection_cosine=mean_projection_cosine,
        min_projection_cosine=min((item.min_projection_cosine for item in sample_diagnostics), default=0.0),
        max_projection_cosine=max((item.max_projection_cosine for item in sample_diagnostics), default=0.0),
        mean_projection_l2_gap=mean_projection_l2_gap,
        mean_projection_l2_ratio=mean_projection_l2_ratio,
        std_projection_cosine=_std(cosine_values),
        cv_projection_cosine=_coefficient_of_variation(mean_projection_cosine, _std(cosine_values)),
        std_projection_l2_gap=_std(l2_gap_values),
        cv_projection_l2_gap=_coefficient_of_variation(mean_projection_l2_gap, _std(l2_gap_values)),
        std_projection_l2_ratio=_std(l2_ratio_values),
        cv_projection_l2_ratio=_coefficient_of_variation(mean_projection_l2_ratio, _std(l2_ratio_values)),
        samples=sample_diagnostics,
    )


def evaluate_alignment_projection_thresholds(
    diagnostics: AlignmentProjectionDiagnosticsSummary,
    *,
    config: AlignmentProjectionThresholdConfig | None = None,
) -> AlignmentProjectionThresholdEvaluation:
    """Evaluate Stage E projection diagnostics against one threshold template."""

    active_config = config or AlignmentProjectionThresholdConfig()
    ratio_deviation = abs(diagnostics.mean_projection_l2_ratio - 1.0)
    checks: list[AlignmentProjectionThresholdCheck] = [
        AlignmentProjectionThresholdCheck(
            name="sample_count",
            passed=diagnostics.sample_count >= active_config.min_sample_count,
            actual=float(diagnostics.sample_count),
            operator=">=",
            expected=float(active_config.min_sample_count),
        ),
        AlignmentProjectionThresholdCheck(
            name="mean_projection_cosine",
            passed=diagnostics.mean_projection_cosine >= active_config.min_mean_projection_cosine,
            actual=diagnostics.mean_projection_cosine,
            operator=">=",
            expected=active_config.min_mean_projection_cosine,
        ),
        AlignmentProjectionThresholdCheck(
            name="mean_projection_l2_gap",
            passed=diagnostics.mean_projection_l2_gap <= active_config.max_mean_projection_l2_gap,
            actual=diagnostics.mean_projection_l2_gap,
            operator="<=",
            expected=active_config.max_mean_projection_l2_gap,
        ),
        AlignmentProjectionThresholdCheck(
            name="mean_projection_l2_ratio_deviation",
            passed=ratio_deviation <= active_config.max_mean_projection_l2_ratio_deviation,
            actual=ratio_deviation,
            operator="<=",
            expected=active_config.max_mean_projection_l2_ratio_deviation,
        ),
        AlignmentProjectionThresholdCheck(
            name="projection_cosine_cv",
            passed=diagnostics.cv_projection_cosine <= active_config.max_projection_cosine_cv,
            actual=diagnostics.cv_projection_cosine,
            operator="<=",
            expected=active_config.max_projection_cosine_cv,
        ),
        AlignmentProjectionThresholdCheck(
            name="projection_l2_gap_cv",
            passed=diagnostics.cv_projection_l2_gap <= active_config.max_projection_l2_gap_cv,
            actual=diagnostics.cv_projection_l2_gap,
            operator="<=",
            expected=active_config.max_projection_l2_gap_cv,
        ),
    ]
    if active_config.enforce_min_projection_cosine:
        checks.insert(
            2,
            AlignmentProjectionThresholdCheck(
                name="min_projection_cosine",
                passed=diagnostics.min_projection_cosine >= active_config.min_min_projection_cosine,
                actual=diagnostics.min_projection_cosine,
                operator=">=",
                expected=active_config.min_min_projection_cosine,
            ),
        )
    passed = all(check.passed for check in checks)
    return AlignmentProjectionThresholdEvaluation(
        passed=passed,
        verdict="PASS" if passed else "WARN",
        checks=tuple(checks),
    )


def render_alignment_projection_diagnostics_markdown(
    diagnostics: AlignmentProjectionDiagnosticsSummary,
    *,
    title: str = "Sample-Level Projection Diagnostics",
    max_samples: int = 10,
    threshold_evaluation: AlignmentProjectionThresholdEvaluation | None = None,
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
        f"- std projection cosine (cross-sample): `{diagnostics.std_projection_cosine:.6f}`",
        f"- cv projection cosine (cross-sample): `{diagnostics.cv_projection_cosine:.6f}`",
        f"- std projection L2 gap (cross-sample): `{diagnostics.std_projection_l2_gap:.6f}`",
        f"- cv projection L2 gap (cross-sample): `{diagnostics.cv_projection_l2_gap:.6f}`",
        f"- std projection L2 ratio (cross-sample): `{diagnostics.std_projection_l2_ratio:.6f}`",
        f"- cv projection L2 ratio (cross-sample): `{diagnostics.cv_projection_l2_ratio:.6f}`",
        "",
    ]
    if threshold_evaluation is not None:
        lines.extend(
            [
                "### Threshold Evaluation",
                "",
                f"- verdict: `{threshold_evaluation.verdict}`",
                "",
                "| check | actual | operator | expected | result |",
                "| --- | ---: | :---: | ---: | :---: |",
            ]
        )
        for check in threshold_evaluation.checks:
            lines.append(
                "| "
                f"{check.name} | "
                f"{check.actual:.6f} | "
                f"{check.operator} | "
                f"{check.expected:.6f} | "
                f"{'PASS' if check.passed else 'WARN'} |"
            )
        lines.append("")

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


def _std(values) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    mean_value = _mean(values_tuple)
    variance = sum((item - mean_value) ** 2 for item in values_tuple) / len(values_tuple)
    return math.sqrt(max(0.0, variance))


def _coefficient_of_variation(mean_value: float, std_value: float, *, epsilon: float = 1e-6) -> float:
    return std_value / max(abs(mean_value), epsilon)


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
