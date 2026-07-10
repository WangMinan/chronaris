"""Run six methods through common pretraining, OOF export, and linear consumers."""

from __future__ import annotations

import logging
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.common_pretraining_smoke_audit import (
    build_augmentation_alignment_rows,
    build_common_pretraining_acceptance_rows,
)
from chronaris.evaluation.application_tasks.common_pretraining_smoke_reporting import (
    write_common_pretraining_smoke_outputs,
)
from chronaris.evaluation.application_tasks.pretraining_smoke_data import (
    load_pretraining_smoke_data,
)
from chronaris.evaluation.application_tasks.pretraining_smoke_downstream import (
    load_guarded_pretraining_smoke_targets,
    run_fixed_linear_smoke_consumers,
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
    select_observation_batch,
    validate_fusion_method_alignment,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.common_pretraining_loop_smoke")
LOGGER.addHandler(logging.NullHandler())
SIX_METHODS = (*TRAINABLE_FUSION_METHODS[:2], "naive_time_sync", *TRAINABLE_FUSION_METHODS[2:])


@dataclass(frozen=True, slots=True)
class CommonPretrainingLoopSmokeConfig:
    run_id: str = "2026-07-11_common-pretraining-loop-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    seed: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class CommonPretrainingLoopSmokeResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    acceptance_pass_count: int
    acceptance_check_count: int
    trainable_checkpoint_count: int
    representation_export_count: int
    metric_count: int
    report_path: str
    evidence_manifest_path: str


def run_common_pretraining_loop_smoke(
    config: CommonPretrainingLoopSmokeConfig,
) -> CommonPretrainingLoopSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="common_pretraining_loop_smoke",
        logger=LOGGER,
        initial_progress={
            "training_epochs": 1,
            "confirmed_metrics_changed": False,
            "pretraining_oracle_opened": False,
            "downstream_metrics_smoke_only": True,
        },
    ) as progress:
        data = load_pretraining_smoke_data(config.simulation_root)
        normalizer = TrainOnlyRobustNormalizer().fit(
            data.batch,
            train_sample_ids=data.fold.train_sample_ids,
            held_out_sample_ids=data.fold.held_out_sample_ids,
        )
        training_config = CommonPretrainingConfig(seed=config.seed)
        augmentation_policy = AugmentationPolicy()
        vehicle_labels = tuple(
            (name, name) for name in data.schema.vehicle_feature_names
        )
        training_root = heavy_root / "checkpoints"
        initial_training_results = []
        for method_name in TRAINABLE_FUSION_METHODS:
            result = train_common_pretext_method(
                method_name,
                batch=data.batch,
                fold=data.fold,
                physiology_feature_names=data.schema.physiology_feature_names,
                vehicle_feature_names=data.schema.vehicle_feature_names,
                vehicle_field_labels=vehicle_labels,
                normalizer=normalizer,
                output_root=training_root,
                config=training_config,
                augmentation_policy=augmentation_policy,
                resume=config.resume,
            )
            initial_training_results.append(result)
            progress.update(
                "method_pretraining_complete",
                method_name=method_name,
                status=result.status,
                step_count=result.step_count,
            )
        resumed_training_results = [
            train_common_pretext_method(
                method_name,
                batch=data.batch,
                fold=data.fold,
                physiology_feature_names=data.schema.physiology_feature_names,
                vehicle_feature_names=data.schema.vehicle_feature_names,
                vehicle_field_labels=vehicle_labels,
                normalizer=normalizer,
                output_root=training_root,
                config=training_config,
                augmentation_policy=augmentation_policy,
                resume=True,
            )
            for method_name in TRAINABLE_FUSION_METHODS
        ]
        registry = CheckpointRegistry(compact_root / "checkpoint_registry.json")
        adapters = {}
        for result in initial_training_results:
            checkpoint = build_checkpoint_record(
                method_name=result.method_name,
                fold=data.fold,
                checkpoint_path=result.best_checkpoint_path,
                seed=config.seed,
            )
            registry.register(checkpoint)
            encoder, _heads, loaded_normalizer, _payload = (
                load_common_pretraining_checkpoint(result.best_checkpoint_path)
            )
            adapters[result.method_name] = TrainedFusionAdapter(
                encoder=encoder,
                normalizer=loaded_normalizer,
                fold_id=data.fold.fold_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
            )
        naive_path = training_root / "naive_time_sync" / "best.pt"
        if not naive_path.exists():
            naive_encoder = NaiveTimeSyncEncoder().fit(
                data.batch,
                train_sample_ids=data.fold.train_sample_ids,
                held_out_sample_ids=data.fold.held_out_sample_ids,
                normalizer=normalizer,
            )
            save_naive_time_sync_checkpoint(naive_path, encoder=naive_encoder)
        naive_checkpoint = build_checkpoint_record(
            method_name="naive_time_sync",
            fold=data.fold,
            checkpoint_path=naive_path,
            seed=config.seed,
        )
        registry.register(naive_checkpoint)
        naive_encoder = load_naive_time_sync_checkpoint(naive_path)
        adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
            encoder=naive_encoder,
            fold_id=data.fold.fold_id,
            checkpoint_sha256=naive_checkpoint.checkpoint_sha256,
        )

        target_frame, target_manifest = load_guarded_pretraining_smoke_targets(
            data_manifest_rows=data.data_manifest_rows,
            fold=data.fold,
            completed_pretraining_checkpoints=tuple(
                result.best_checkpoint_path for result in initial_training_results
            ),
        )
        progress.update(
            "downstream_targets_opened_after_pretraining",
            target_count=len(target_frame),
        )
        exporter = ResumableOOFExporter(
            heavy_root / "representations",
            resume=config.resume,
        )
        initial_exports = []
        outputs = {method: {} for method in SIX_METHODS}
        for method_name in SIX_METHODS:
            adapter = adapters[method_name]
            checkpoint = registry.require(method_name, data.fold.fold_id)
            for role in ("train", "validation", "held_out"):
                role_batch = select_observation_batch(
                    data.batch,
                    data.fold.sample_ids_for_role(role),
                )
                result = exporter.export(
                    encoder=adapter,
                    batch=role_batch,
                    checkpoint=checkpoint,
                    export_role=role,
                )
                initial_exports.append(result)
                outputs[method_name][role] = load_fusion_stream_batch(
                    result.output_root
                )
        recovery_audit = _verify_single_export_recovery(
            exporter=exporter,
            adapters=adapters,
            registry=registry,
            data=data,
            initial_exports=initial_exports,
            heavy_root=heavy_root,
        )
        alignment_hashes = {
            role: validate_fusion_method_alignment(
                [outputs[method][role] for method in SIX_METHODS]
            )
            for role in ("train", "validation", "held_out")
        }
        resume_exports = [
            exporter.export(
                encoder=adapters[method_name],
                batch=select_observation_batch(
                    data.batch,
                    data.fold.sample_ids_for_role(role),
                ),
                checkpoint=registry.require(method_name, data.fold.fold_id),
                export_role=role,
            )
            for method_name in SIX_METHODS
            for role in ("train", "validation", "held_out")
        ]
        metric_rows, prediction_rows = run_fixed_linear_smoke_consumers(
            outputs=outputs,
            targets=target_frame,
            fold=data.fold,
            seed=config.seed,
        )
        prediction_path = heavy_root / "downstream_predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)

        training_rows = [
            dict(row)
            for result in initial_training_results
            for row in result.training_rows
        ]
        augmentation_rows = [
            dict(row)
            for result in initial_training_results
            for row in result.augmentation_rows
        ]
        augmentation_alignment = build_augmentation_alignment_rows(
            augmentation_rows
        )
        transform_manifest = {
            "format": "chronaris.common_pretraining_transforms.v1",
            "fold": data.fold.to_dict(),
            "methods": {
                method: {
                    "normalizer_sha256": normalizer.to_manifest()[
                        "transform_sha256"
                    ],
                    "fit_sample_hash": normalizer.fit_sample_hash,
                }
                for method in SIX_METHODS
            },
        }
        representation_manifest = {
            "format": "chronaris.common_pretraining_exports.v1",
            "available_export_count": len(initial_exports),
            "current_run_built_count": sum(
                result.status == "completed" for result in initial_exports
            ),
            "current_run_reused_count": sum(
                result.status == "resumed" for result in initial_exports
            ),
            "resume_verification_reused_count": sum(
                result.status == "resumed" for result in resume_exports
            ),
            "alignment_sha256": alignment_hashes,
            "exports": [result.to_dict() for result in initial_exports],
            "transform_manifest": transform_manifest,
            "single_item_recovery": recovery_audit,
        }
        acceptance_rows = build_common_pretraining_acceptance_rows(
            data_manifest_rows=data.data_manifest_rows,
            fold=data.fold,
            initial_training_results=initial_training_results,
            resumed_training_results=resumed_training_results,
            registry=registry,
            training_rows=training_rows,
            augmentation_alignment_rows=augmentation_alignment,
            export_manifest=representation_manifest,
            alignment_hashes=alignment_hashes,
            metric_rows=metric_rows,
            target_manifest=target_manifest,
            transform_manifest=transform_manifest,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        training_status_rows = [
            {
                "method_name": initial.method_name,
                "initial_status": initial.status,
                "resume_verification_status": resumed.status,
                "protocol_sha256": initial.protocol_sha256,
                "step_count": initial.step_count,
            }
            for initial, resumed in zip(
                initial_training_results,
                resumed_training_results,
                strict=True,
            )
        ] + [
            {
                "method_name": "naive_time_sync",
                "initial_status": "unsupervised_transform_only",
                "resume_verification_status": "checkpoint_reused",
                "protocol_sha256": None,
                "step_count": 0,
            }
        ]
        resource_rows = [
            {
                "method_name": result.method_name,
                "parameter_count": result.parameter_count,
                "head_parameter_count": result.head_parameter_count,
                "training_elapsed_s": result.training_elapsed_s,
                "step_count": result.step_count,
            }
            for result in initial_training_results
        ]
        paths = write_common_pretraining_smoke_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            data_manifest_rows=data.data_manifest_rows,
            split_manifest=data.fold.to_dict(),
            augmentation_protocol={
                "format": "chronaris.augmentation_protocol.v1",
                "policy": asdict(augmentation_policy),
                "method_parameter_allowed": False,
                "method_count": 5,
            },
            augmentation_alignment_rows=augmentation_alignment,
            pretext_target_manifest=target_manifest,
            training_protocol={
                "format": "chronaris.common_pretraining_protocol.v1",
                "config": asdict(training_config),
                "common_loss_weights": {
                    "masked_reconstruction": 1.0,
                    "short_horizon_prediction": 0.5,
                    "lag_discrimination": 0.2,
                },
                "chronaris_epoch_one_auxiliary_weights": {
                    "continuous_alignment": 0.0,
                    "physical_consistency": 0.0,
                    "causal_direction": 0.0,
                },
                "label_used_for_encoder_training": False,
                "pretraining_oracle_opened": False,
            },
            training_rows=training_rows,
            training_status_rows=training_status_rows,
            resource_rows=resource_rows,
            representation_manifest=representation_manifest,
            downstream_protocol={
                "format": "chronaris.linear_smoke_consumer.v1",
                "classification": {"estimator": "LogisticRegression", "C": 1.0},
                "regression": {"estimator": "Ridge", "alpha": 1.0},
                "scaler_fit_role": "train",
                "threshold_fit_role": "train",
                "evaluation_roles": ["validation", "held_out"],
                "smoke_only": True,
            },
            metric_rows=metric_rows,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
            prediction_path=str(prediction_path),
        )
        pass_count = sum(bool(row["passed"]) for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            representation_export_count=len(initial_exports),
            metric_count=len(metric_rows),
        )
        return CommonPretrainingLoopSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            trainable_checkpoint_count=len(initial_training_results),
            representation_export_count=len(initial_exports),
            metric_count=len(metric_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _verify_single_export_recovery(
    *, exporter, adapters, registry, data, initial_exports, heavy_root
):
    marker = heavy_root / "single_item_recovery_audit.json"
    target = next(
        result
        for result in initial_exports
        if result.method_name == "chronaris" and result.export_role == "held_out"
    )
    if marker.exists() and Path(target.representation_path).exists() and Path(
        target.manifest_path
    ).exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    original_hash = target.representation_sha256
    Path(target.representation_path).unlink(missing_ok=True)
    Path(target.manifest_path).unlink(missing_ok=True)
    held_out = select_observation_batch(
        data.batch,
        data.fold.held_out_sample_ids,
    )
    rebuilt = exporter.export(
        encoder=adapters["chronaris"],
        batch=held_out,
        checkpoint=registry.require("chronaris", data.fold.fold_id),
        export_role="held_out",
    )
    payload = {
        "method_name": "chronaris",
        "export_role": "held_out",
        "removed_file_count": 2,
        "rebuild_status": rebuilt.status,
        "original_representation_sha256": original_hash,
        "rebuilt_representation_sha256": rebuilt.representation_sha256,
        "hash_match": original_hash == rebuilt.representation_sha256,
        "non_target_initial_export_count": len(initial_exports) - 1,
    }
    marker.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
