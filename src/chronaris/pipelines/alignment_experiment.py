"""Composed Stage E experiment pipeline and report helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from chronaris.features.experiment_input import E0ExperimentSample, summarize_e0_samples
from chronaris.pipelines.alignment_preview import AlignmentPreviewPipeline, AlignmentPreviewRunResult
from chronaris.pipelines.e0_preview import E0PreviewPipeline
from chronaris.schema.models import SortieLocator


@dataclass(frozen=True, slots=True)
class AlignmentExperimentSampleSummary:
    """Compact E0 sample summary for one Stage E experiment run."""

    sample_count: int
    physiology_feature_count_max: int
    vehicle_feature_count_max: int


@dataclass(frozen=True, slots=True)
class AlignmentExperimentRunResult:
    """Outputs for one composed Stage E experiment run."""

    locator: SortieLocator
    samples: tuple[E0ExperimentSample, ...]
    sample_summary: AlignmentExperimentSampleSummary
    preview_result: AlignmentPreviewRunResult
    report_markdown: str


@dataclass(slots=True)
class AlignmentExperimentPipeline:
    """Compose E0 preview generation with the Stage E preview trainer."""

    e0_pipeline: E0PreviewPipeline
    alignment_preview_pipeline: AlignmentPreviewPipeline = field(default_factory=AlignmentPreviewPipeline)

    def run(self, locator: SortieLocator) -> AlignmentExperimentRunResult:
        """Build E0 samples, run preview training, and render one Markdown summary."""

        samples = self.e0_pipeline.run(locator)
        sample_summary_dict = summarize_e0_samples(samples)
        sample_summary = AlignmentExperimentSampleSummary(
            sample_count=sample_summary_dict["sample_count"],
            physiology_feature_count_max=sample_summary_dict["physiology_feature_count_max"],
            vehicle_feature_count_max=sample_summary_dict["vehicle_feature_count_max"],
        )
        preview_result = self.alignment_preview_pipeline.run(samples)
        report_markdown = render_alignment_experiment_report(
            locator=locator,
            sample_summary=sample_summary,
            preview_result=preview_result,
        )
        return AlignmentExperimentRunResult(
            locator=locator,
            samples=samples,
            sample_summary=sample_summary,
            preview_result=preview_result,
            report_markdown=report_markdown,
        )


def render_alignment_experiment_report(
    *,
    locator: SortieLocator,
    sample_summary: AlignmentExperimentSampleSummary,
    preview_result: AlignmentPreviewRunResult,
) -> str:
    """Render a compact Markdown report for one Stage E preview experiment."""

    final_train = preview_result.train_history[-1] if preview_result.train_history else None
    final_validation = preview_result.validation_history[-1] if preview_result.validation_history else None
    test_metrics = preview_result.test_metrics
    intermediate_export = preview_result.intermediate_export

    lines = [
        f"# Alignment Preview - {locator.sortie_id}",
        "",
        "## Sample Summary",
        "",
        f"- sample count: `{sample_summary.sample_count}`",
        f"- max physiology feature count: `{sample_summary.physiology_feature_count_max}`",
        f"- max vehicle feature count: `{sample_summary.vehicle_feature_count_max}`",
        "",
        "## Split Summary",
        "",
        f"- train: `{len(preview_result.split.train)}`",
        f"- validation: `{len(preview_result.split.validation)}`",
        f"- test: `{len(preview_result.split.test)}`",
        f"- skipped between train/validation: `{len(preview_result.split.skipped_between_train_validation)}`",
        f"- skipped between validation/test: `{len(preview_result.split.skipped_between_validation_test)}`",
        "",
    ]

    if final_train is not None:
        lines.extend(
            [
                "## Final Train Metrics",
                "",
                f"- physiology reconstruction: `{final_train.physiology_reconstruction:.6f}`",
                f"- vehicle reconstruction: `{final_train.vehicle_reconstruction:.6f}`",
                f"- reconstruction total: `{final_train.reconstruction_total:.6f}`",
                f"- alignment: `{final_train.alignment:.6f}`",
                f"- vehicle physics: `{final_train.vehicle_physics:.6f}`",
                f"- physiology physics: `{final_train.physiology_physics:.6f}`",
                f"- physics total: `{final_train.physics_total:.6f}`",
                f"- total: `{final_train.total:.6f}`",
                "",
            ]
        )

    if final_validation is not None:
        lines.extend(
            [
                "## Final Validation Metrics",
                "",
                f"- physiology reconstruction: `{final_validation.physiology_reconstruction:.6f}`",
                f"- vehicle reconstruction: `{final_validation.vehicle_reconstruction:.6f}`",
                f"- reconstruction total: `{final_validation.reconstruction_total:.6f}`",
                f"- alignment: `{final_validation.alignment:.6f}`",
                f"- vehicle physics: `{final_validation.vehicle_physics:.6f}`",
                f"- physiology physics: `{final_validation.physiology_physics:.6f}`",
                f"- physics total: `{final_validation.physics_total:.6f}`",
                f"- total: `{final_validation.total:.6f}`",
                "",
            ]
        )

    if intermediate_export is not None:
        exported_sample_ids = ", ".join(sample.sample_id for sample in intermediate_export.samples)
        physiology_projection_l2_values = tuple(
            sample.physiology.mean_reference_projection_l2 for sample in intermediate_export.samples
        )
        vehicle_projection_l2_values = tuple(
            sample.vehicle.mean_reference_projection_l2 for sample in intermediate_export.samples
        )
        projection_cosine_values = tuple(
            sample.mean_reference_projection_cosine for sample in intermediate_export.samples
        )

        lines.extend(
            [
                "## Reference Intermediate Export",
                "",
                f"- partition: `{intermediate_export.partition}`",
                f"- exported sample count: `{intermediate_export.sample_count}`",
                f"- reference point count: `{intermediate_export.reference_point_count}`",
                (
                    f"- exported sample ids: `{exported_sample_ids}`"
                    if exported_sample_ids
                    else "- exported sample ids: `(none)`"
                ),
                f"- physiology mean reference projection L2: `{_mean(physiology_projection_l2_values):.6f}`",
                f"- vehicle mean reference projection L2: `{_mean(vehicle_projection_l2_values):.6f}`",
                f"- mean cross-stream projection cosine: `{_mean(projection_cosine_values):.6f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Test Metrics",
            "",
            f"- physiology reconstruction: `{test_metrics.physiology_reconstruction:.6f}`",
            f"- vehicle reconstruction: `{test_metrics.vehicle_reconstruction:.6f}`",
            f"- reconstruction total: `{test_metrics.reconstruction_total:.6f}`",
            f"- alignment: `{test_metrics.alignment:.6f}`",
            f"- vehicle physics: `{test_metrics.vehicle_physics:.6f}`",
            f"- physiology physics: `{test_metrics.physiology_physics:.6f}`",
            f"- physics total: `{test_metrics.physics_total:.6f}`",
            f"- total: `{test_metrics.total:.6f}`",
        ]
    )

    return "\n".join(lines) + "\n"


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
