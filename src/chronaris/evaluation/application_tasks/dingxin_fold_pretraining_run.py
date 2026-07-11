"""Run one real Dingxin outer fold through six-method frozen representation smoke."""

from __future__ import annotations

import logging
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_audit import (
    build_dingxin_fold_pretraining_acceptance_rows,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_reporting import (
    write_dingxin_fold_pretraining_outputs,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.modeling.training import (
    TRAINABLE_FUSION_METHODS,
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import (
    AugmentationPolicy,
    CheckpointRegistry,
    ResumableOOFExporter,
    TrainOnlyRobustNormalizer,
    build_checkpoint_record,
    load_fusion_stream_batch,
    validate_fusion_method_alignment,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_fold_pretraining")
LOGGER.addHandler(logging.NullHandler())
SIX_METHODS = (
    *TRAINABLE_FUSION_METHODS[:2],
    "naive_time_sync",
    *TRAINABLE_FUSION_METHODS[2:],
)


@dataclass(frozen=True, slots=True)
class DingxinFoldPretrainingConfig:
    run_id: str = "2026-07-11_dingxin-fold-pretraining-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    fold_id: str = "leave_one_view_out__fold01"
    seed: int = 17
    normalizer_batch_size: int = 2
    training_batch_size: int = 1
    export_batch_size: int = 2
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinFoldPretrainingResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    fold_id: str
    trainable_checkpoint_count: int
    representation_export_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_fold_pretraining_smoke(
    config: DingxinFoldPretrainingConfig,
) -> DingxinFoldPretrainingResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_real_fold_pretraining_smoke",
        logger=LOGGER,
        initial_progress={
            "fold_id": config.fold_id,
            "training_epochs": 1,
            "downstream_targets_opened": False,
            "outer_test_metrics_opened": False,
            "confirmed_metrics_changed": False,
        },
    ) as progress:
        data = load_dingxin_fold_pretraining_data(
            fold_id=config.fold_id,
            snapshot_root=config.snapshot_root,
            fixed_audit_root=config.fixed_audit_root,
            inner_split_root=config.inner_split_root,
        )
        fold = data.fold
        provider = data.index.load_batch
        normalization_started = time.perf_counter()
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
            provider,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=(
                fold.validation_sample_ids + fold.held_out_sample_ids
            ),
            batch_size=config.normalizer_batch_size,
        )
        normalization_elapsed = time.perf_counter() - normalization_started
        progress.update(
            "inner_train_normalizer_fitted",
            fit_sample_count=len(normalizer.fit_sample_ids),
            elapsed_s=normalization_elapsed,
        )
        training_config = CommonPretrainingConfig(
            epochs=1,
            batch_size=config.training_batch_size,
            seed=config.seed,
        )
        augmentation_policy = AugmentationPolicy()
        checkpoint_root = heavy_root / "checkpoints" / fold.fold_id
        training_results = []
        resource_rows = []
        for method_name in TRAINABLE_FUSION_METHODS:
            result = train_common_pretext_method(
                method_name,
                batch=None,
                batch_provider=provider,
                fold=fold,
                physiology_feature_names=(
                    data.index.plan.schema.physiology_feature_names
                ),
                vehicle_feature_names=data.index.plan.schema.vehicle_feature_names,
                vehicle_field_labels=data.vehicle_field_labels,
                normalizer=normalizer,
                output_root=checkpoint_root,
                config=training_config,
                augmentation_policy=augmentation_policy,
                resume=config.resume,
            )
            training_results.append(result)
            resource_rows.append(
                {
                    "method_name": method_name,
                    "parameter_count": result.parameter_count,
                    "head_parameter_count": result.head_parameter_count,
                    "training_elapsed_s": result.training_elapsed_s,
                    "step_count": result.step_count,
                    "maximum_rss_mb": _maximum_rss_mb(),
                }
            )
            progress.update(
                "method_pretraining_complete",
                method_name=method_name,
                status=result.status,
                step_count=result.step_count,
            )
        resumed_training_results = [
            train_common_pretext_method(
                method_name,
                batch=None,
                batch_provider=provider,
                fold=fold,
                physiology_feature_names=(
                    data.index.plan.schema.physiology_feature_names
                ),
                vehicle_feature_names=data.index.plan.schema.vehicle_feature_names,
                vehicle_field_labels=data.vehicle_field_labels,
                normalizer=normalizer,
                output_root=checkpoint_root,
                config=training_config,
                augmentation_policy=augmentation_policy,
                resume=True,
            )
            for method_name in TRAINABLE_FUSION_METHODS
        ]
        registry = CheckpointRegistry(compact_root / "checkpoint_registry.json")
        adapters = {}
        for result in training_results:
            checkpoint = build_checkpoint_record(
                method_name=result.method_name,
                fold=fold,
                checkpoint_path=result.best_checkpoint_path,
                seed=config.seed,
            )
            registry.register(checkpoint, replace_existing=True)
            encoder, _heads, loaded_normalizer, _payload = (
                load_common_pretraining_checkpoint(result.best_checkpoint_path)
            )
            adapters[result.method_name] = TrainedFusionAdapter(
                encoder=encoder,
                normalizer=loaded_normalizer,
                fold_id=fold.fold_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
            )
        naive_started = time.perf_counter()
        naive_path = checkpoint_root / "naive_time_sync" / "best.pt"
        had_naive_checkpoint = naive_path.exists()
        rebuild_naive = not had_naive_checkpoint or not config.resume
        if not rebuild_naive:
            existing_naive = load_naive_time_sync_checkpoint(naive_path)
            rebuild_naive = (
                existing_naive.projector.solver != "randomized"
                or existing_naive.normalizer.fit_sample_ids
                != normalizer.fit_sample_ids
            )
        if rebuild_naive:
            naive_encoder = NaiveTimeSyncEncoder().fit_from_batch_provider(
                provider,
                train_sample_ids=fold.train_sample_ids,
                held_out_sample_ids=(
                    fold.validation_sample_ids + fold.held_out_sample_ids
                ),
                normalizer=normalizer,
                batch_size=config.normalizer_batch_size,
            )
            save_naive_time_sync_checkpoint(naive_path, encoder=naive_encoder)
            naive_status = (
                "rebuilt_incompatible" if had_naive_checkpoint else "completed"
            )
        else:
            naive_status = "resumed"
        naive_encoder = load_naive_time_sync_checkpoint(naive_path)
        if naive_encoder.normalizer.fit_sample_ids != normalizer.fit_sample_ids:
            raise ValueError("naive checkpoint fit samples differ from current fold")
        naive_elapsed = time.perf_counter() - naive_started
        resource_rows.append(
            {
                "method_name": "naive_time_sync",
                "parameter_count": 0,
                "head_parameter_count": 0,
                "training_elapsed_s": naive_elapsed,
                "step_count": 0,
                "maximum_rss_mb": _maximum_rss_mb(),
            }
        )
        resource_rows = _merge_resource_history(
            compact_root / "resource_budget.csv",
            resource_rows,
        )
        naive_checkpoint = build_checkpoint_record(
            method_name="naive_time_sync",
            fold=fold,
            checkpoint_path=naive_path,
            seed=config.seed,
        )
        registry.register(naive_checkpoint, replace_existing=True)
        adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
            encoder=naive_encoder,
            fold_id=fold.fold_id,
            checkpoint_sha256=naive_checkpoint.checkpoint_sha256,
        )
        exporter = ResumableOOFExporter(
            heavy_root / "representations",
            resume=config.resume,
        )
        initial_exports = []
        outputs = {method: {} for method in SIX_METHODS}
        for method_name in SIX_METHODS:
            checkpoint = registry.require(method_name, fold.fold_id)
            for role in ("train", "validation", "held_out"):
                result = exporter.export_from_batch_provider(
                    encoder=adapters[method_name],
                    batch_provider=provider,
                    checkpoint=checkpoint,
                    export_role=role,
                    batch_size=config.export_batch_size,
                )
                initial_exports.append(result)
                outputs[method_name][role] = load_fusion_stream_batch(
                    result.output_root
                )
            progress.update(
                "method_representation_export_complete",
                method_name=method_name,
            )
        alignment_hashes = {
            role: validate_fusion_method_alignment(
                [outputs[method][role] for method in SIX_METHODS]
            )
            for role in ("train", "validation", "held_out")
        }
        resume_exporter = ResumableOOFExporter(
            heavy_root / "representations",
            resume=True,
        )
        resumed_exports = [
            resume_exporter.export_from_batch_provider(
                encoder=adapters[method_name],
                batch_provider=provider,
                checkpoint=registry.require(method_name, fold.fold_id),
                export_role=role,
                batch_size=config.export_batch_size,
            )
            for method_name in SIX_METHODS
            for role in ("train", "validation", "held_out")
        ]
        maximum_rss = max(
            _maximum_rss_mb(),
            *(float(row["maximum_rss_mb"]) for row in resource_rows),
        )
        acceptance_rows = build_dingxin_fold_pretraining_acceptance_rows(
            fold=fold,
            normalizer=normalizer,
            training_results=training_results,
            resumed_training_results=resumed_training_results,
            registry=registry,
            initial_exports=initial_exports,
            resumed_exports=resumed_exports,
            alignment_hashes=alignment_hashes,
            naive_manifest=naive_encoder.to_manifest(),
            maximum_rss_mb=maximum_rss,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        training_status_rows = [
            {
                "method_name": initial.method_name,
                "initial_status": initial.status,
                "resume_status": resumed.status,
                "protocol_sha256": initial.protocol_sha256,
                "step_count": initial.step_count,
            }
            for initial, resumed in zip(
                training_results,
                resumed_training_results,
                strict=True,
            )
        ] + [
            {
                "method_name": "naive_time_sync",
                "initial_status": naive_status,
                "resume_status": "checkpoint_reused",
                "protocol_sha256": None,
                "step_count": 0,
            }
        ]
        training_rows = [
            dict(row) for result in training_results for row in result.training_rows
        ]
        representation_manifest = {
            "format": "chronaris.dingxin_fold_representation_exports.v1",
            "fold_id": fold.fold_id,
            "export_count": len(initial_exports),
            "initial_built_count": sum(
                result.status == "completed" for result in initial_exports
            ),
            "initial_reused_count": sum(
                result.status == "resumed" for result in initial_exports
            ),
            "resume_reused_count": sum(
                result.status == "resumed" for result in resumed_exports
            ),
            "alignment_sha256": alignment_hashes,
            "exports": [result.to_dict() for result in initial_exports],
        }
        paths = write_dingxin_fold_pretraining_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            source_manifest={
                "format": "chronaris.dingxin_fold_pretraining_sources.v1",
                "source_hashes": data.source_hashes,
                "schema_sha256": data.index.plan.schema.schema_sha256,
                "physiology_feature_count": len(
                    data.index.plan.schema.physiology_feature_names
                ),
                "vehicle_feature_count": len(
                    data.index.plan.schema.vehicle_feature_names
                ),
                "downstream_targets_opened": False,
            },
            split_manifest=fold.to_dict(),
            training_protocol={
                "format": "chronaris.dingxin_fold_pretraining_protocol.v1",
                "config": asdict(training_config),
                "augmentation_policy": asdict(augmentation_policy),
                "normalizer_fit_sample_hash": normalizer.fit_sample_hash,
                "normalizer_transform_sha256": normalizer.to_manifest()[
                    "transform_sha256"
                ],
                "normalizer_elapsed_s": normalization_elapsed,
                "normalizer_batch_size": config.normalizer_batch_size,
                "export_batch_size": config.export_batch_size,
                "label_used_for_encoder_training": False,
                "outer_test_metrics_opened": False,
            },
            training_status_rows=training_status_rows,
            training_rows=training_rows,
            resource_rows=resource_rows,
            representation_manifest=representation_manifest,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            representation_export_count=len(initial_exports),
            maximum_rss_mb=maximum_rss,
        )
        return DingxinFoldPretrainingResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            fold_id=fold.fold_id,
            trainable_checkpoint_count=len(training_results),
            representation_export_count=len(initial_exports),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _maximum_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _merge_resource_history(path: Path, current_rows):
    if not path.is_file():
        return current_rows
    previous = {
        str(row.method_name): row
        for row in pd.read_csv(path).itertuples(index=False)
    }
    merged = []
    for row in current_rows:
        prior = previous.get(str(row["method_name"]))
        if prior is not None:
            row = {
                **row,
                "training_elapsed_s": max(
                    float(row["training_elapsed_s"]),
                    float(prior.training_elapsed_s),
                ),
                "maximum_rss_mb": max(
                    float(row["maximum_rss_mb"]),
                    float(prior.maximum_rss_mb),
                ),
            }
        merged.append(row)
    return merged
