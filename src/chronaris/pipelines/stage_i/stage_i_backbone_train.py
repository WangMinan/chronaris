"""Reusable alignment-backbone training entry for the Stage I thesis mainline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.access import (
    DirectInfluxScopeConfig,
    OverlapPreviewSortieLoaderConfig,
    build_overlap_preview_sortie_loader,
)
from chronaris.access.stage_h_profile import StageHSortieProfile
from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.features import E0InputConfig, build_e0_experiment_samples
from chronaris.features.experiment_input import E0ExperimentSample
from chronaris.pipelines.alignment_preview import (
    AlignmentPreviewConfig,
    AlignmentPreviewPipeline,
    save_alignment_preview_checkpoint,
)
from chronaris.pipelines.stage_h.export_helpers import resolve_export_scope
from chronaris.schema.models import SortieLocator, WindowConfig


def _default_backbone_preview_config() -> AlignmentPreviewConfig:
    return AlignmentPreviewConfig(
        epoch_count=3,
        batch_size=8,
        learning_rate=1e-3,
        device="auto",
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
        export_intermediate_states=False,
        intermediate_sample_limit=None,
        intermediate_partition="all",
    )


@dataclass(frozen=True, slots=True)
class StageIBackboneTrainConfig:
    """Configuration for one reusable backbone-training run."""

    run_id: str
    output_root: str | Path = "docs/reports/assets/stage_i_backbone"
    preview_config: AlignmentPreviewConfig = field(
        default_factory=_default_backbone_preview_config
    )
    window_config: WindowConfig = field(
        default_factory=lambda: WindowConfig(duration_ms=5_000, stride_ms=5_000)
    )
    checkpoint_filename: str = "alignment_backbone_checkpoint.pt"
    summary_filename: str = "backbone_summary.json"


@dataclass(frozen=True, slots=True)
class StageIBackboneTrainRunResult:
    """Artifacts written by one backbone-training run."""

    artifact_root: str
    checkpoint_path: str
    summary_path: str
    summary: Mapping[str, object]


@dataclass(slots=True)
class StageIBackboneTrainPipeline:
    """Train one reusable Stage E/F alignment backbone and export its checkpoint."""

    config: StageIBackboneTrainConfig

    def run(
        self,
        samples: tuple[E0ExperimentSample, ...],
        *,
        source_summary: Mapping[str, object] | None = None,
    ) -> StageIBackboneTrainRunResult:
        if not samples:
            raise ValueError("StageIBackboneTrainPipeline requires at least one sample.")

        artifact_root = Path(self.config.output_root) / self.config.run_id
        artifact_root.mkdir(parents=True, exist_ok=True)

        preview_result = AlignmentPreviewPipeline(config=self.config.preview_config).run(samples)
        checkpoint_path = artifact_root / self.config.checkpoint_filename
        checkpoint_metadata = save_alignment_preview_checkpoint(
            checkpoint_path,
            run_id=self.config.run_id,
            model=preview_result.model,
            config=self.config.preview_config,
            samples=samples,
            input_normalization_stats=preview_result.input_normalization_stats,
            train_history=preview_result.train_history,
            validation_history=preview_result.validation_history,
            test_metrics=preview_result.test_metrics,
        )

        summary = {
            "run_id": self.config.run_id,
            "artifact_root": str(artifact_root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "sample_count": len(samples),
            "split_counts": {
                "train": len(preview_result.split.train),
                "validation": len(preview_result.split.validation),
                "test": len(preview_result.split.test),
            },
            "preview_config": {
                "input_normalization_mode": self.config.preview_config.input_normalization_mode,
                "physics_constraints_enabled": self.config.preview_config.enable_physics_constraints,
                "physics_constraint_family": self.config.preview_config.physics_constraint_family,
                "intermediate_partition": self.config.preview_config.intermediate_partition,
                "intermediate_sample_limit": self.config.preview_config.intermediate_sample_limit,
            },
            "train_metrics": (
                asdict(preview_result.train_history[-1])
                if preview_result.train_history
                else None
            ),
            "validation_metrics": (
                asdict(preview_result.validation_history[-1])
                if preview_result.validation_history
                else None
            ),
            "test_metrics": asdict(preview_result.test_metrics),
            "source_summary": dict(source_summary or {}),
        }
        summary_path = artifact_root / self.config.summary_filename
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return StageIBackboneTrainRunResult(
            artifact_root=str(artifact_root),
            checkpoint_path=str(checkpoint_path),
            summary_path=str(summary_path),
            summary=summary,
        )


def run_stage_i_backbone_train(
    config: StageIBackboneTrainConfig,
    samples: tuple[E0ExperimentSample, ...],
    *,
    source_summary: Mapping[str, object] | None = None,
) -> StageIBackboneTrainRunResult:
    return StageIBackboneTrainPipeline(config=config).run(
        samples,
        source_summary=source_summary,
    )


def collect_stage_i_backbone_samples(
    *,
    profiles: Sequence[StageHSortieProfile],
    window_config: WindowConfig,
    physiology_point_limit_per_measurement: int | None = None,
    vehicle_point_limit_per_measurement: int | None = None,
    export_scope_overrides_utc: Mapping[str, tuple[object, object]] | None = None,
    influx_settings: object | None = None,
    runner: object | None = None,
) -> tuple[tuple[E0ExperimentSample, ...], dict[str, object]]:
    if (influx_settings is None) == (runner is None):
        raise ValueError("Provide exactly one of influx_settings or runner.")

    samples: list[E0ExperimentSample] = []
    view_summaries: list[dict[str, object]] = []
    for profile in profiles:
        export_start_utc, export_stop_utc = resolve_export_scope(
            profile,
            overrides=export_scope_overrides_utc or {},
        )
        for view in profile.views:
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
                        point_limit_per_measurement=physiology_point_limit_per_measurement,
                    ),
                    vehicle_scope=DirectInfluxScopeConfig(
                        bucket=profile.vehicle_bucket,
                        measurements=profile.vehicle_measurements,
                        start_time_utc=export_start_utc,
                        stop_time_utc=export_stop_utc,
                        tag_filters={"sortie_number": profile.sortie_id},
                        point_limit_per_measurement=vehicle_point_limit_per_measurement,
                    ),
                    metadata=profile.to_sortie_metadata(),
                ),
                influx_settings=influx_settings,
                runner=runner,
            )
            locator = SortieLocator(
                sortie_id=profile.sortie_id,
                pilot_id=str(view.pilot_id),
            )
            bundle = loader.load(locator)
            dataset_result = SortieDatasetBuilder(window_config=window_config).build(bundle)
            view_samples = build_e0_experiment_samples(
                dataset_result,
                config=E0InputConfig(
                    physiology_measurements=profile.model_physiology_measurements,
                    vehicle_measurements=profile.vehicle_measurements,
                ),
            )
            samples.extend(view_samples)
            view_summaries.append(
                {
                    "sortie_id": profile.sortie_id,
                    "view_id": view.view_id,
                    "pilot_id": view.pilot_id,
                    "sample_count": len(view_samples),
                    "export_start_utc": export_start_utc.isoformat(),
                    "export_stop_utc": export_stop_utc.isoformat(),
                }
            )
    source_summary = {
        "sortie_count": len({profile.sortie_id for profile in profiles}),
        "view_count": len(view_summaries),
        "sample_count": len(samples),
        "view_summaries": view_summaries,
    }
    return tuple(samples), source_summary
