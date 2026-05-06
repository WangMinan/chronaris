"""Training, export, and validation pipelines."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "AlignmentExperimentPipeline": (
        "chronaris.pipelines.alignment_experiment",
        "AlignmentExperimentPipeline",
    ),
    "AlignmentExperimentRunResult": (
        "chronaris.pipelines.alignment_experiment",
        "AlignmentExperimentRunResult",
    ),
    "AlignmentExperimentSampleSummary": (
        "chronaris.pipelines.alignment_experiment",
        "AlignmentExperimentSampleSummary",
    ),
    "render_alignment_experiment_report": (
        "chronaris.pipelines.alignment_experiment",
        "render_alignment_experiment_report",
    ),
    "AlignmentPreviewConfig": (
        "chronaris.pipelines.alignment_preview",
        "AlignmentPreviewConfig",
    ),
    "AlignmentPreviewPipeline": (
        "chronaris.pipelines.alignment_preview",
        "AlignmentPreviewPipeline",
    ),
    "StageGCausalFusionConfig": (
        "chronaris.pipelines.causal_fusion",
        "StageGCausalFusionConfig",
    ),
    "StageGCausalFusionResult": (
        "chronaris.pipelines.causal_fusion",
        "StageGCausalFusionResult",
    ),
    "StageGCausalFusionSample": (
        "chronaris.pipelines.causal_fusion",
        "StageGCausalFusionSample",
    ),
    "StageGCausalFusionTensorExport": (
        "chronaris.pipelines.causal_fusion",
        "StageGCausalFusionTensorExport",
    ),
    "export_stage_g_causal_fusion_tensors": (
        "chronaris.pipelines.causal_fusion",
        "export_stage_g_causal_fusion_tensors",
    ),
    "render_stage_g_causal_fusion_markdown": (
        "chronaris.pipelines.causal_fusion",
        "render_stage_g_causal_fusion_markdown",
    ),
    "run_stage_g_causal_fusion": (
        "chronaris.pipelines.causal_fusion",
        "run_stage_g_causal_fusion",
    ),
    "DatasetPipelineV1": ("chronaris.pipelines.dataset_v1", "DatasetPipelineV1"),
    "E0PreviewPipeline": ("chronaris.pipelines.e0_preview", "E0PreviewPipeline"),
    "InfluxPartialVehiclePointProvider": (
        "chronaris.pipelines.partial_data",
        "InfluxPartialVehiclePointProvider",
    ),
    "MySQLPartialVehicleMetadataProvider": (
        "chronaris.pipelines.partial_data",
        "MySQLPartialVehicleMetadataProvider",
    ),
    "PartialDataBuilder": ("chronaris.pipelines.partial_data", "PartialDataBuilder"),
    "PartialDataBuildResult": (
        "chronaris.pipelines.partial_data",
        "PartialDataBuildResult",
    ),
    "PartialDataConfig": ("chronaris.pipelines.partial_data", "PartialDataConfig"),
    "PartialDataEntry": ("chronaris.pipelines.partial_data", "PartialDataEntry"),
    "PartialDataManifest": (
        "chronaris.pipelines.partial_data",
        "PartialDataManifest",
    ),
    "PartialMeasurementMetadata": (
        "chronaris.pipelines.partial_data",
        "PartialMeasurementMetadata",
    ),
    "PartialPointChunk": ("chronaris.pipelines.partial_data", "PartialPointChunk"),
    "PartialStreamSample": ("chronaris.pipelines.partial_data", "PartialStreamSample"),
    "VEHICLE_ONLY_FEATURE_BUNDLE_KEYS": (
        "chronaris.pipelines.partial_data",
        "VEHICLE_ONLY_FEATURE_BUNDLE_KEYS",
    ),
    "dump_partial_data_entries": (
        "chronaris.pipelines.partial_data",
        "dump_partial_data_entries",
    ),
    "load_partial_data_entries": (
        "chronaris.pipelines.partial_data",
        "load_partial_data_entries",
    ),
    "AlignmentStageHViewRunner": (
        "chronaris.pipelines.stage_h.export",
        "AlignmentStageHViewRunner",
    ),
    "StageHExportConfig": (
        "chronaris.pipelines.stage_h.export",
        "StageHExportConfig",
    ),
    "StageHExportProfile": (
        "chronaris.pipelines.stage_h.export",
        "StageHExportProfile",
    ),
    "StageHExportPipeline": (
        "chronaris.pipelines.stage_h.export",
        "StageHExportPipeline",
    ),
    "StageHExportRunResult": (
        "chronaris.pipelines.stage_h.export",
        "StageHExportRunResult",
    ),
    "StageHRunManifest": (
        "chronaris.pipelines.stage_h.export",
        "StageHRunManifest",
    ),
    "StageHSortieManifest": (
        "chronaris.pipelines.stage_h.export",
        "StageHSortieManifest",
    ),
    "StageHViewExecutionResult": (
        "chronaris.pipelines.stage_h.export",
        "StageHViewExecutionResult",
    ),
    "StageHViewManifest": (
        "chronaris.pipelines.stage_h.export",
        "StageHViewManifest",
    ),
    "render_stage_h_report": (
        "chronaris.pipelines.stage_h.export",
        "render_stage_h_report",
    ),
    "StageIBaselineArtifacts": (
        "chronaris.pipelines.stage_i.stage_i_baseline",
        "StageIBaselineArtifacts",
    ),
    "render_stage_i_baseline_report": (
        "chronaris.pipelines.stage_i.stage_i_baseline",
        "render_stage_i_baseline_report",
    ),
    "render_uab_baseline_report": (
        "chronaris.pipelines.stage_i.stage_i_baseline",
        "render_uab_baseline_report",
    ),
    "run_stage_i_baselines": (
        "chronaris.pipelines.stage_i.stage_i_baseline",
        "run_stage_i_baselines",
    ),
    "run_uab_baselines": (
        "chronaris.pipelines.stage_i.stage_i_baseline",
        "run_uab_baselines",
    ),
    "write_baseline_artifacts": (
        "chronaris.pipelines.stage_i.stage_i_baseline",
        "write_baseline_artifacts",
    ),
    "StageICaseStudyConfig": (
        "chronaris.pipelines.stage_i.stage_i_case_study",
        "StageICaseStudyConfig",
    ),
    "StageICaseStudyRunResult": (
        "chronaris.pipelines.stage_i.stage_i_case_study",
        "StageICaseStudyRunResult",
    ),
    "render_stage_i_case_study_report": (
        "chronaris.pipelines.stage_i.stage_i_case_study",
        "render_stage_i_case_study_report",
    ),
    "run_stage_i_case_study": (
        "chronaris.pipelines.stage_i.stage_i_case_study",
        "run_stage_i_case_study",
    ),
    "write_stage_i_case_study_report": (
        "chronaris.pipelines.stage_i.stage_i_case_study",
        "write_stage_i_case_study_report",
    ),
    "StageIDeepBaselineConfig": (
        "chronaris.pipelines.stage_i.stage_i_deep_baseline",
        "StageIDeepBaselineConfig",
    ),
    "StageIDeepBaselineRunResult": (
        "chronaris.pipelines.stage_i.stage_i_deep_baseline",
        "StageIDeepBaselineRunResult",
    ),
    "StageIDeepComparisonConfig": (
        "chronaris.pipelines.stage_i.stage_i_deep_baseline",
        "StageIDeepComparisonConfig",
    ),
    "StageIDeepComparisonRunResult": (
        "chronaris.pipelines.stage_i.stage_i_deep_baseline",
        "StageIDeepComparisonRunResult",
    ),
    "run_stage_i_deep_baseline": (
        "chronaris.pipelines.stage_i.stage_i_deep_baseline",
        "run_stage_i_deep_baseline",
    ),
    "run_stage_i_deep_comparison": (
        "chronaris.pipelines.stage_i.stage_i_deep_baseline",
        "run_stage_i_deep_comparison",
    ),
    "StageIPhase3Config": (
        "chronaris.pipelines.stage_i.stage_i_phase3",
        "StageIPhase3Config",
    ),
    "StageIPhase3RunResult": (
        "chronaris.pipelines.stage_i.stage_i_phase3",
        "StageIPhase3RunResult",
    ),
    "build_stage_i_dataset_summary": (
        "chronaris.pipelines.stage_i.stage_i_phase3",
        "build_stage_i_dataset_summary",
    ),
    "compose_stage_i_phase3_closure": (
        "chronaris.pipelines.stage_i.stage_i_phase3",
        "compose_stage_i_phase3_closure",
    ),
    "render_stage_i_phase3_report": (
        "chronaris.pipelines.stage_i.stage_i_phase3",
        "render_stage_i_phase3_report",
    ),
    "run_stage_i_phase3": (
        "chronaris.pipelines.stage_i.stage_i_phase3",
        "run_stage_i_phase3",
    ),
    "StageIPublicOptConfig": (
        "chronaris.pipelines.stage_i.stage_i_public_opt",
        "StageIPublicOptConfig",
    ),
    "StageIPublicOptRunResult": (
        "chronaris.pipelines.stage_i.stage_i_public_opt",
        "StageIPublicOptRunResult",
    ),
    "run_stage_i_public_opt": (
        "chronaris.pipelines.stage_i.stage_i_public_opt",
        "run_stage_i_public_opt",
    ),
    "StageIPublicOptTorchUABConfig": (
        "chronaris.pipelines.stage_i.stage_i_public_opt_torch",
        "StageIPublicOptTorchUABConfig",
    ),
    "StageIPublicOptTorchUABRunResult": (
        "chronaris.pipelines.stage_i.stage_i_public_opt_torch",
        "StageIPublicOptTorchUABRunResult",
    ),
    "run_stage_i_public_opt_torch_uab": (
        "chronaris.pipelines.stage_i.stage_i_public_opt_torch",
        "run_stage_i_public_opt_torch_uab",
    ),
    "StageIPublicMainlineReportConfig": (
        "chronaris.pipelines.stage_i.stage_i_public_mainline_report",
        "StageIPublicMainlineReportConfig",
    ),
    "StageIPublicMainlineReportRunResult": (
        "chronaris.pipelines.stage_i.stage_i_public_mainline_report",
        "StageIPublicMainlineReportRunResult",
    ),
    "run_stage_i_public_mainline_report": (
        "chronaris.pipelines.stage_i.stage_i_public_mainline_report",
        "run_stage_i_public_mainline_report",
    ),
    "DEFAULT_PUBLIC_FUSION_CANDIDATES": (
        "chronaris.pipelines.stage_i.stage_i_public_fusion_screen",
        "DEFAULT_PUBLIC_FUSION_CANDIDATES",
    ),
    "StageIPublicFusionCandidate": (
        "chronaris.pipelines.stage_i.stage_i_public_fusion_screen",
        "StageIPublicFusionCandidate",
    ),
    "StageIPublicFusionScreenConfig": (
        "chronaris.pipelines.stage_i.stage_i_public_fusion_screen",
        "StageIPublicFusionScreenConfig",
    ),
    "StageIPublicFusionScreenRunResult": (
        "chronaris.pipelines.stage_i.stage_i_public_fusion_screen",
        "StageIPublicFusionScreenRunResult",
    ),
    "run_stage_i_public_fusion_screen": (
        "chronaris.pipelines.stage_i.stage_i_public_fusion_screen",
        "run_stage_i_public_fusion_screen",
    ),
    "StageIPrivateBenchmarkConfig": (
        "chronaris.pipelines.stage_i.stage_i_private_benchmark",
        "StageIPrivateBenchmarkConfig",
    ),
    "StageIPrivateBenchmarkRunResult": (
        "chronaris.pipelines.stage_i.stage_i_private_benchmark",
        "StageIPrivateBenchmarkRunResult",
    ),
    "run_stage_i_private_benchmark": (
        "chronaris.pipelines.stage_i.stage_i_private_benchmark",
        "run_stage_i_private_benchmark",
    ),
    "StageIPrivateOptimizedPackageResult": (
        "chronaris.pipelines.stage_i.stage_i_private_optimized_package",
        "StageIPrivateOptimizedPackageResult",
    ),
    "StageISequencePreparationConfig": (
        "chronaris.pipelines.stage_i.stage_i_sequence_preparation",
        "StageISequencePreparationConfig",
    ),
    "StageISequencePreparationRunResult": (
        "chronaris.pipelines.stage_i.stage_i_sequence_preparation",
        "StageISequencePreparationRunResult",
    ),
    "run_stage_i_sequence_preparation": (
        "chronaris.pipelines.stage_i.stage_i_sequence_preparation",
        "run_stage_i_sequence_preparation",
    ),
    "StageISupportConfig": (
        "chronaris.pipelines.stage_i.stage_i_support",
        "StageISupportConfig",
    ),
    "StageISupportRunResult": (
        "chronaris.pipelines.stage_i.stage_i_support",
        "StageISupportRunResult",
    ),
    "run_stage_i_support": (
        "chronaris.pipelines.stage_i.stage_i_support",
        "run_stage_i_support",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
