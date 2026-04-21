"""Evaluation, ablation, and case-review helpers."""

from chronaris.evaluation.alignment_diagnostics import (
    AlignmentProjectionThresholdCheck,
    AlignmentProjectionThresholdConfig,
    AlignmentProjectionThresholdEvaluation,
    AlignmentProjectionDiagnosticsSummary,
    SampleAlignmentProjectionDiagnostic,
    evaluate_alignment_projection_thresholds,
    render_alignment_projection_diagnostics_markdown,
    summarize_alignment_projection_diagnostics,
)
from chronaris.evaluation.sortie_validation import (
    CrossStreamTimingSummary,
    MeasurementCoverageSummary,
    SortieValidationSummary,
    StreamCoverageSummary,
    WindowTrialSummary,
    render_validation_markdown,
    summarize_cross_stream_timing,
    summarize_stream,
    summarize_window_trial,
    validate_sortie_bundle,
)

__all__ = [
    "AlignmentProjectionDiagnosticsSummary",
    "AlignmentProjectionThresholdCheck",
    "AlignmentProjectionThresholdConfig",
    "AlignmentProjectionThresholdEvaluation",
    "CrossStreamTimingSummary",
    "MeasurementCoverageSummary",
    "SampleAlignmentProjectionDiagnostic",
    "SortieValidationSummary",
    "StreamCoverageSummary",
    "WindowTrialSummary",
    "evaluate_alignment_projection_thresholds",
    "render_alignment_projection_diagnostics_markdown",
    "render_validation_markdown",
    "summarize_alignment_projection_diagnostics",
    "summarize_cross_stream_timing",
    "summarize_stream",
    "summarize_window_trial",
    "validate_sortie_bundle",
]
