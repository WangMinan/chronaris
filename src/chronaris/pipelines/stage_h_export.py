"""Stage H multi-sortie export pipeline and artifact packaging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, MutableMapping, Sequence

import numpy as np

from chronaris.access import (
    DirectInfluxScopeConfig,
    MySQLRealBusContextReader,
    OverlapPreviewSortieLoaderConfig,
    build_overlap_preview_sortie_loader,
)
from chronaris.access.stage_h_profile import StageHSortieProfile, StageHViewProfile
from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.evaluation import (
    AlignmentProjectionThresholdConfig,
    evaluate_alignment_projection_thresholds,
    summarize_alignment_projection_diagnostics,
)
from chronaris.features import E0InputConfig, build_e0_experiment_samples
from chronaris.pipelines.alignment_preview import AlignmentPreviewConfig, AlignmentPreviewPipeline
from chronaris.pipelines.causal_fusion import (
    StageGCausalFusionConfig,
    StageGCausalFusionResult,
    StageGCausalFusionTensorExport,
    export_stage_g_causal_fusion_tensors,
    run_stage_g_causal_fusion,
)
from chronaris.pipelines.partial_data import (
    PartialDataBuildResult,
    PartialDataBuilder,
    PartialDataConfig,
    PartialDataEntry,
)
from chronaris.pipelines.stage_h_export_helpers import (
    build_intermediate_summary,
    build_run_config_summary,
    render_stage_h_report,
    resolve_export_scope,
    write_json,
    write_view_artifacts,
)
from chronaris.schema.models import DatasetBuildResult, SortieLocator, WindowConfig

StageHExportProfile = Literal["preview", "validation", "full_clip"]


def _default_stage_h_preview_config() -> AlignmentPreviewConfig:
    return AlignmentPreviewConfig(
        epoch_count=3,
        batch_size=8,
        learning_rate=1e-3,
        reconstruction_loss_mode="relative_mse",
        input_normalization_mode="zscore_train",
        alignment_loss_mode="mse",
        enable_physics_constraints=True,
        physics_constraint_mode="feature_first_with_latent_fallback",
        physics_constraint_family="full",
        vehicle_physics_weight=0.1,
        physiology_physics_weight=0.1,
        physics_huber_delta=1.0,
        vehicle_envelope_quantile=0.95,
        physiology_envelope_quantile=0.95,
        export_intermediate_states=True,
        intermediate_sample_limit=256,
        intermediate_partition="test",
    )


@dataclass(frozen=True, slots=True)
class StageHExportConfig:
    """Frozen Stage H export settings for one run."""

    run_id: str
    sortie_ids: tuple[str, ...]
    output_root: str | Path = "docs/reports/assets/stage_h"
    report_path: str | Path = "docs/reports/stage-h-export-v1.md"
    export_version: str = "stage-h-v1"
    export_profile: StageHExportProfile = "preview"
    window_config: WindowConfig = field(
        default_factory=lambda: WindowConfig(duration_ms=5_000, stride_ms=5_000)
    )
    preview_config: AlignmentPreviewConfig = field(default_factory=_default_stage_h_preview_config)
    causal_fusion_enabled: bool = True
    causal_fusion_config: StageGCausalFusionConfig = field(default_factory=StageGCausalFusionConfig)
    threshold_config: AlignmentProjectionThresholdConfig = field(
        default_factory=AlignmentProjectionThresholdConfig
    )
    bus_access_rule_id: int = 6000019510066
    preview_point_limit_per_measurement: int | None = None
    physiology_point_limit_per_measurement: int | None = None
    vehicle_point_limit_per_measurement: int | None = None
    export_scope_overrides_utc: Mapping[str, tuple[datetime, datetime]] = field(default_factory=dict)
    partial_data_config: PartialDataConfig = field(default_factory=PartialDataConfig)
    partial_data_entries: tuple[PartialDataEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.export_profile not in {"preview", "validation", "full_clip"}:
            raise ValueError("export_profile must be one of: preview, validation, full_clip.")
        for field_name in (
            "preview_point_limit_per_measurement",
            "physiology_point_limit_per_measurement",
            "vehicle_point_limit_per_measurement",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided.")

    @property
    def resolved_physiology_point_limit_per_measurement(self) -> int | None:
        if self.physiology_point_limit_per_measurement is not None:
            return self.physiology_point_limit_per_measurement
        return self.preview_point_limit_per_measurement

    @property
    def resolved_vehicle_point_limit_per_measurement(self) -> int | None:
        if self.vehicle_point_limit_per_measurement is not None:
            return self.vehicle_point_limit_per_measurement
        return self.preview_point_limit_per_measurement

    @property
    def point_limit_note(self) -> str:
        if (
            self.resolved_physiology_point_limit_per_measurement is None
            and self.resolved_vehicle_point_limit_per_measurement is None
        ):
            return "no per-measurement point cap"
        if self.export_profile == "preview":
            return "preview query guard, not a Stage H closure standard"
        return "explicit per-measurement query cap"


@dataclass(frozen=True, slots=True)
class StageHViewManifest:
    """One exported view package inside a Stage H run."""

    view_id: str
    sortie_id: str
    pilot_id: int
    flight_task_id: int
    collect_task_id: int
    physiology_measurements: tuple[str, ...]
    vehicle_measurements: tuple[str, ...]
    clip_start_utc: str
    clip_stop_utc: str
    export_start_utc: str
    export_stop_utc: str
    window_count: int
    model_sample_count: int
    split_summary: Mapping[str, int]
    intermediate_summary: Mapping[str, object]
    projection_diagnostics_verdict: str
    stage_g_enabled: bool
    stage_g_available: bool
    vehicle_field_metadata: Mapping[str, object]
    artifact_paths: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "sortie_id": self.sortie_id,
            "pilot_id": self.pilot_id,
            "flight_task_id": self.flight_task_id,
            "collect_task_id": self.collect_task_id,
            "physiology_measurements": list(self.physiology_measurements),
            "vehicle_measurements": list(self.vehicle_measurements),
            "clip_start_utc": self.clip_start_utc,
            "clip_stop_utc": self.clip_stop_utc,
            "export_start_utc": self.export_start_utc,
            "export_stop_utc": self.export_stop_utc,
            "window_count": self.window_count,
            "model_sample_count": self.model_sample_count,
            "split_summary": dict(self.split_summary),
            "intermediate_summary": dict(self.intermediate_summary),
            "projection_diagnostics_verdict": self.projection_diagnostics_verdict,
            "stage_g_enabled": self.stage_g_enabled,
            "stage_g_available": self.stage_g_available,
            "vehicle_field_metadata": dict(self.vehicle_field_metadata),
            "artifact_paths": dict(self.artifact_paths),
        }


@dataclass(frozen=True, slots=True)
class StageHSortieManifest:
    """One sortie-level Stage H manifest."""

    sortie_id: str
    flight_task_id: int
    collect_task_id: int
    pilot_ids: tuple[int, ...]
    physiology_measurements: tuple[str, ...]
    vehicle_measurements: tuple[str, ...]
    clip_start_utc: str
    clip_stop_utc: str
    exported_view_ids: tuple[str, ...]
    view_manifest_paths: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "sortie_id": self.sortie_id,
            "flight_task_id": self.flight_task_id,
            "collect_task_id": self.collect_task_id,
            "pilot_ids": list(self.pilot_ids),
            "physiology_measurements": list(self.physiology_measurements),
            "vehicle_measurements": list(self.vehicle_measurements),
            "clip_start_utc": self.clip_start_utc,
            "clip_stop_utc": self.clip_stop_utc,
            "exported_view_ids": list(self.exported_view_ids),
            "view_manifest_paths": dict(self.view_manifest_paths),
        }


@dataclass(frozen=True, slots=True)
class StageHRunManifest:
    """Run-level Stage H manifest spanning all sorties and partial-data outputs."""

    run_id: str
    export_version: str
    generated_at_utc: str
    output_root: str
    report_path: str
    sortie_ids: tuple[str, ...]
    generated_view_count: int
    generated_view_ids: tuple[str, ...]
    config: Mapping[str, object]
    sortie_manifest_paths: Mapping[str, str]
    failures: tuple[Mapping[str, str], ...]
    skipped: tuple[Mapping[str, str], ...]
    partial_data: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "export_version": self.export_version,
            "generated_at_utc": self.generated_at_utc,
            "output_root": self.output_root,
            "report_path": self.report_path,
            "sortie_ids": list(self.sortie_ids),
            "generated_view_count": self.generated_view_count,
            "generated_view_ids": list(self.generated_view_ids),
            "config": dict(self.config),
            "sortie_manifest_paths": dict(self.sortie_manifest_paths),
            "failures": [dict(item) for item in self.failures],
            "skipped": [dict(item) for item in self.skipped],
            "partial_data": None if self.partial_data is None else dict(self.partial_data),
        }


@dataclass(frozen=True, slots=True)
class StageHViewExecutionResult:
    """In-memory result of exporting one Stage H view before serialization."""

    dataset_result: DatasetBuildResult
    sample_ids: tuple[str, ...]
    split_summary: Mapping[str, int]
    train_metrics: Mapping[str, object]
    validation_metrics: Mapping[str, object]
    test_metrics: Mapping[str, object]
    intermediate_export: object | None
    diagnostics_summary: object
    threshold_evaluation: object
    stage_g_result: StageGCausalFusionResult | None
    stage_g_tensor_export: StageGCausalFusionTensorExport | None
    vehicle_field_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StageHExportRunResult:
    """Final Stage H run outputs."""

    run_manifest_path: str
    report_path: str
    output_root: str
    generated_view_ids: tuple[str, ...]
    partial_data_result: PartialDataBuildResult | None = None


@dataclass(slots=True)
class AlignmentStageHViewRunner:
    """Default Stage H executor based on the frozen Stage E/F/G preview stack."""

    config: StageHExportConfig
    influx_settings: object | None = None
    influx_runner: object | None = None
    vehicle_context_reader: MySQLRealBusContextReader | None = None
    _vehicle_field_label_cache: MutableMapping[str, tuple[dict[str, str], dict[str, object]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (self.influx_settings is None) == (self.influx_runner is None):
            raise ValueError("Provide exactly one of influx_settings or influx_runner.")

    def run(
        self,
        profile: StageHSortieProfile,
        view: StageHViewProfile,
        *,
        export_start_utc: datetime,
        export_stop_utc: datetime,
    ) -> StageHViewExecutionResult:
        vehicle_field_labels, vehicle_field_metadata = self._resolve_vehicle_field_labels(profile)
        preview_config = replace(
            self.config.preview_config,
            vehicle_field_labels=vehicle_field_labels,
        )
        loader = build_overlap_preview_sortie_loader(
            OverlapPreviewSortieLoaderConfig(
                sortie_id=profile.sortie_id,
                physiology_scope=DirectInfluxScopeConfig(
                    bucket=profile.physiology_bucket,
                    measurements=profile.model_physiology_measurements,
                    start_time_utc=export_start_utc,
                    stop_time_utc=export_stop_utc,
                    tag_filters={
                        "collect_task_id": str(profile.collect_task_id),
                        "pilot_id": str(view.pilot_id),
                    },
                    point_limit_per_measurement=(
                        self.config.resolved_physiology_point_limit_per_measurement
                    ),
                ),
                vehicle_scope=DirectInfluxScopeConfig(
                    bucket=profile.vehicle_bucket,
                    measurements=profile.vehicle_measurements,
                    start_time_utc=export_start_utc,
                    stop_time_utc=export_stop_utc,
                    tag_filters={"sortie_number": profile.sortie_id},
                    point_limit_per_measurement=(
                        self.config.resolved_vehicle_point_limit_per_measurement
                    ),
                ),
                metadata=profile.to_sortie_metadata(),
            ),
            influx_settings=self.influx_settings,
            runner=self.influx_runner,
        )
        locator = SortieLocator(sortie_id=profile.sortie_id, pilot_id=str(view.pilot_id))
        bundle = loader.load(locator)
        dataset_result = SortieDatasetBuilder(window_config=self.config.window_config).build(bundle)
        samples = build_e0_experiment_samples(
            dataset_result,
            config=E0InputConfig(
                physiology_measurements=profile.model_physiology_measurements,
                vehicle_measurements=profile.vehicle_measurements,
            ),
        )
        preview_result = AlignmentPreviewPipeline(config=preview_config).run(samples)
        diagnostics_summary = summarize_alignment_projection_diagnostics(
            preview_result.intermediate_export
        )
        threshold_evaluation = evaluate_alignment_projection_thresholds(
            diagnostics_summary,
            config=self.config.threshold_config,
        )
        stage_g_result = None
        stage_g_tensor_export = None
        if self.config.causal_fusion_enabled and preview_result.intermediate_export is not None:
            stage_g_result = run_stage_g_causal_fusion(
                preview_result.intermediate_export,
                config=self.config.causal_fusion_config,
            )
            stage_g_tensor_export = export_stage_g_causal_fusion_tensors(
                preview_result.intermediate_export,
                config=self.config.causal_fusion_config,
            )
        return StageHViewExecutionResult(
            dataset_result=dataset_result,
            sample_ids=tuple(sample.sample_id for sample in samples),
            split_summary={
                "train": len(preview_result.split.train),
                "validation": len(preview_result.split.validation),
                "test": len(preview_result.split.test),
                "skipped_between_train_validation": len(
                    preview_result.split.skipped_between_train_validation
                ),
                "skipped_between_validation_test": len(
                    preview_result.split.skipped_between_validation_test
                ),
            },
            train_metrics=(
                asdict(preview_result.train_history[-1])
                if preview_result.train_history
                else {}
            ),
            validation_metrics=(
                asdict(preview_result.validation_history[-1])
                if preview_result.validation_history
                else {}
            ),
            test_metrics=asdict(preview_result.test_metrics),
            intermediate_export=preview_result.intermediate_export,
            diagnostics_summary=diagnostics_summary,
            threshold_evaluation=threshold_evaluation,
            stage_g_result=stage_g_result,
            stage_g_tensor_export=stage_g_tensor_export,
            vehicle_field_metadata=vehicle_field_metadata,
        )

    def _resolve_vehicle_field_labels(
        self,
        profile: StageHSortieProfile,
    ) -> tuple[dict[str, str], dict[str, object]]:
        cached = self._vehicle_field_label_cache.get(profile.sortie_id)
        if cached is not None:
            return cached
        if self.vehicle_context_reader is None:
            result = ({}, {"status": "skipped", "field_count": 0, "errors": []})
            self._vehicle_field_label_cache[profile.sortie_id] = result
            return result

        labels: dict[str, str] = {}
        errors: list[str] = []
        for measurement, analysis_id in profile.vehicle_analysis_ids.items():
            try:
                context = self.vehicle_context_reader.fetch_context(
                    locator=SortieLocator(sortie_id=profile.sortie_id),
                    access_rule_id=self.config.bus_access_rule_id,
                    analysis_id=analysis_id,
                )
            except Exception as exc:  # pragma: no cover - live-only fallback.
                errors.append(f"{measurement}: {exc}")
                continue
            for detail in context.detail_list:
                labels[detail.col_field] = detail.col_name
                labels[f"{measurement}.{detail.col_field}"] = detail.col_name
            for structure in context.structure_list:
                labels.setdefault(structure.col_field, structure.col_name)
                labels.setdefault(f"{measurement}.{structure.col_field}", structure.col_name)
            for access_detail in context.access_rule_details:
                if access_detail.col_name:
                    labels.setdefault(access_detail.col_field, access_detail.col_name)
                    labels.setdefault(
                        f"{measurement}.{access_detail.col_field}",
                        access_detail.col_name,
                    )

        metadata = {
            "status": "loaded" if labels and not errors else ("partial" if labels else "unavailable"),
            "field_count": len(labels),
            "error_count": len(errors),
            "errors": errors,
        }
        result = (labels, metadata)
        self._vehicle_field_label_cache[profile.sortie_id] = result
        return result


@dataclass(slots=True)
class StageHExportPipeline:
    """Write Stage H run/sortie/view assets plus the partial-data sidecar."""

    config: StageHExportConfig
    profile_resolver: object
    view_runner: object
    partial_data_builder: PartialDataBuilder | None = None

    def run(self) -> StageHExportRunResult:
        generated_at_utc = datetime.now(timezone.utc).isoformat()
        run_root = Path(self.config.output_root) / self.config.run_id
        run_root.mkdir(parents=True, exist_ok=True)
        report_path = Path(self.config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        sortie_manifests: list[StageHSortieManifest] = []
        view_manifests: list[StageHViewManifest] = []
        failures: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        sortie_manifest_paths: dict[str, str] = {}

        profiles = self.profile_resolver.resolve_many(self.config.sortie_ids)
        for profile in profiles:
            sortie_dir = run_root / "sorties" / profile.sortie_id
            views_dir = sortie_dir / "views"
            views_dir.mkdir(parents=True, exist_ok=True)
            export_start_utc, export_stop_utc = resolve_export_scope(
                profile,
                overrides=self.config.export_scope_overrides_utc,
            )

            exported_view_ids: list[str] = []
            view_manifest_paths: dict[str, str] = {}
            for view in profile.views:
                try:
                    execution = self.view_runner.run(
                        profile,
                        view,
                        export_start_utc=export_start_utc,
                        export_stop_utc=export_stop_utc,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "scope": "view",
                            "sortie_id": profile.sortie_id,
                            "view_id": view.view_id,
                            "message": str(exc),
                        }
                    )
                    continue
                if execution.intermediate_export is None:
                    skipped.append(
                        {
                            "scope": "view",
                            "sortie_id": profile.sortie_id,
                            "view_id": view.view_id,
                            "reason": "intermediate_export_missing",
                        }
                    )
                    continue

                view_dir = views_dir / view.view_id
                view_dir.mkdir(parents=True, exist_ok=True)
                artifact_paths = write_view_artifacts(
                    view_dir=view_dir,
                    execution=execution,
                )
                intermediate_summary = build_intermediate_summary(
                    execution.intermediate_export
                )
                view_manifest = StageHViewManifest(
                    view_id=view.view_id,
                    sortie_id=profile.sortie_id,
                    pilot_id=view.pilot_id,
                    flight_task_id=profile.flight_task.flight_task_id,
                    collect_task_id=profile.collect_task_id,
                    physiology_measurements=profile.model_physiology_measurements,
                    vehicle_measurements=profile.vehicle_measurements,
                    clip_start_utc=profile.clip_start_utc.isoformat(),
                    clip_stop_utc=profile.clip_stop_utc.isoformat(),
                    export_start_utc=export_start_utc.isoformat(),
                    export_stop_utc=export_stop_utc.isoformat(),
                    window_count=len(execution.dataset_result.windows),
                    model_sample_count=len(execution.sample_ids),
                    split_summary=execution.split_summary,
                    intermediate_summary=intermediate_summary,
                    projection_diagnostics_verdict=execution.threshold_evaluation.verdict,
                    stage_g_enabled=self.config.causal_fusion_enabled,
                    stage_g_available=execution.stage_g_result is not None,
                    vehicle_field_metadata=execution.vehicle_field_metadata,
                    artifact_paths=artifact_paths,
                )
                view_manifest_path = view_dir / "view_manifest.json"
                write_json(view_manifest_path, view_manifest.to_dict())
                exported_view_ids.append(view.view_id)
                view_manifest_paths[view.view_id] = str(view_manifest_path)
                view_manifests.append(view_manifest)

            sortie_manifest = StageHSortieManifest(
                sortie_id=profile.sortie_id,
                flight_task_id=profile.flight_task.flight_task_id,
                collect_task_id=profile.collect_task_id,
                pilot_ids=profile.pilot_ids,
                physiology_measurements=profile.available_physiology_measurements,
                vehicle_measurements=profile.vehicle_measurements,
                clip_start_utc=profile.clip_start_utc.isoformat(),
                clip_stop_utc=profile.clip_stop_utc.isoformat(),
                exported_view_ids=tuple(exported_view_ids),
                view_manifest_paths=view_manifest_paths,
            )
            sortie_manifest_path = sortie_dir / "sortie_manifest.json"
            write_json(sortie_manifest_path, sortie_manifest.to_dict())
            sortie_manifests.append(sortie_manifest)
            sortie_manifest_paths[profile.sortie_id] = str(sortie_manifest_path)

        partial_data_result = None
        partial_summary = None
        if self.partial_data_builder is not None and self.config.partial_data_entries:
            partial_data_result = self.partial_data_builder.run(
                self.config.partial_data_entries,
                output_root=run_root / "partial_data",
            )
            partial_summary = {
                "manifest_path": partial_data_result.manifest_path,
                "window_manifest_path": partial_data_result.window_manifest_path,
                "feature_bundle_path": partial_data_result.feature_bundle_path,
                "entry_count": partial_data_result.manifest.entry_count,
                "built_entry_count": partial_data_result.manifest.built_entry_count,
                "skipped_entry_count": partial_data_result.manifest.skipped_entry_count,
            }

        run_manifest = StageHRunManifest(
            run_id=self.config.run_id,
            export_version=self.config.export_version,
            generated_at_utc=generated_at_utc,
            output_root=str(run_root),
            report_path=str(report_path),
            sortie_ids=self.config.sortie_ids,
            generated_view_count=len(view_manifests),
            generated_view_ids=tuple(view_manifest.view_id for view_manifest in view_manifests),
            config=build_run_config_summary(self.config),
            sortie_manifest_paths=sortie_manifest_paths,
            failures=tuple(failures),
            skipped=tuple(skipped),
            partial_data=partial_summary,
        )
        run_manifest_path = run_root / "run_manifest.json"
        write_json(run_manifest_path, run_manifest.to_dict())
        report_path.write_text(
            render_stage_h_report(
                config=self.config,
                run_manifest=run_manifest,
                sortie_manifests=tuple(sortie_manifests),
                view_manifests=tuple(view_manifests),
            ),
            encoding="utf-8",
        )
        return StageHExportRunResult(
            run_manifest_path=str(run_manifest_path),
            report_path=str(report_path),
            output_root=str(run_root),
            generated_view_ids=run_manifest.generated_view_ids,
            partial_data_result=partial_data_result,
        )
