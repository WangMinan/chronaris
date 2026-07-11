"""Run six frozen representations through application-facing downstream consumers."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
    export_application_context_representations,
    load_frozen_pretraining_adapters,
)
from chronaris.evaluation.application_tasks.application_consumer_runtime import (
    ApplicationConsumerProtocol,
    run_application_method_consumers,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_audit import (
    build_application_consumer_acceptance_rows,
    build_paired_unit_statistic_rows,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    build_guarded_application_consumer_targets,
    load_application_consumer_smoke_data,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_reporting import (
    write_application_consumer_smoke_outputs,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.application_consumer_smoke")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class ApplicationConsumerSmokeConfig:
    run_id: str = "2026-07-11_application-consumer-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    pretraining_run_id: str = "2026-07-11_common-pretraining-loop-smoke"
    seed: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class ApplicationConsumerSmokeResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    acceptance_pass_count: int
    acceptance_check_count: int
    representation_export_count: int
    metric_count: int
    fusion_gain_count: int
    paired_statistic_count: int
    report_path: str
    evidence_manifest_path: str


def run_application_consumer_smoke(
    config: ApplicationConsumerSmokeConfig,
) -> ApplicationConsumerSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_compact = Path(config.compact_output_root) / config.pretraining_run_id
    pretraining_heavy = Path(config.heavy_output_root) / config.pretraining_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="application_consumer_smoke",
        logger=LOGGER,
        initial_progress={
            "encoder_training_invoked": False,
            "target_oracle_opened": False,
            "confirmed_metrics_changed": False,
            "smoke_only": True,
        },
    ) as progress:
        adapters, checkpoint_rows, fold_id = load_frozen_pretraining_adapters(
            pretraining_heavy_root=pretraining_heavy,
            pretraining_compact_root=pretraining_compact,
        )
        checkpoint_hashes_before = {
            row["method_name"]: sha256_file(row["checkpoint_path"])
            for row in checkpoint_rows
        }
        data = load_application_consumer_smoke_data(config.simulation_root)
        outputs, initial_export_rows, alignment_hashes = (
            export_application_context_representations(
                adapters=adapters,
                batch=data.batch,
                role_sample_ids=data.role_sample_ids,
                output_root=heavy_root / "representations",
                resume=config.resume,
            )
        )
        progress.update(
            "application_representations_complete",
            export_count=len(initial_export_rows),
            target_oracle_opened=False,
        )
        outputs, recovery_export_rows, representation_recovery = (
            _verify_representation_recovery(
                adapters=adapters,
                data=data,
                output_root=heavy_root / "representations",
                initial_export_rows=initial_export_rows,
            )
        )
        _, resume_export_rows, resume_alignment = (
            export_application_context_representations(
                adapters=adapters,
                batch=data.batch,
                role_sample_ids=data.role_sample_ids,
                output_root=heavy_root / "representations",
                resume=True,
            )
        )
        if alignment_hashes != resume_alignment:
            raise RuntimeError("application representation alignment changed on resume")
        completed_trainable_paths = tuple(
            row["checkpoint_path"]
            for row in checkpoint_rows
            if row["training_status"] == "completed"
        )
        targets = build_guarded_application_consumer_targets(
            data,
            completed_pretraining_checkpoints=completed_trainable_paths,
        )
        progress.update(
            "application_targets_opened_after_representation",
            target_count=len(targets.sample_ids),
            target_oracle_opened=True,
        )
        protocol = ApplicationConsumerProtocol()
        method_results = []
        for method_name in APPLICATION_METHODS:
            result = run_application_method_consumers(
                method_name=method_name,
                outputs=outputs[method_name],
                targets=targets,
                output_root=heavy_root / "consumers",
                fold_id=fold_id,
                protocol=protocol,
                resume=config.resume,
            )
            method_results.append(result)
            progress.update(
                "method_consumers_complete",
                method_name=method_name,
                status=result.status,
                metric_count=len(result.metric_rows),
            )
        resumed_method_results = [
            run_application_method_consumers(
                method_name=method_name,
                outputs=outputs[method_name],
                targets=targets,
                output_root=heavy_root / "consumers",
                fold_id=fold_id,
                protocol=protocol,
                resume=True,
            )
            for method_name in APPLICATION_METHODS
        ]
        consumer_recovery = _verify_consumer_component_recovery(
            chronaris_result=resumed_method_results[-1],
            outputs=outputs["chronaris"],
            targets=targets,
            output_root=heavy_root / "consumers",
            fold_id=fold_id,
            protocol=protocol,
        )
        metric_rows = [dict(row) for result in method_results for row in result.metric_rows]
        workload_rows = [
            dict(row)
            for result in method_results
            for row in result.workload_prediction_rows
        ]
        unit_rows = [dict(row) for result in method_results for row in result.unit_score_rows]
        tcn_training_rows = [
            {"method": result.method_name, **dict(row)}
            for result in method_results
            for row in result.tcn_training_rows
        ]
        resource_rows = [
            dict(row) for result in method_results for row in result.resource_rows
        ]
        fusion_gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=("naive_time_sync", "mult", "contiformer", "chronaris"),
        )
        paired_rows = build_paired_unit_statistic_rows(
            unit_rows,
            sample_manifest_rows=data.sample_manifest_rows,
            seed=config.seed,
        )
        workload_path = heavy_root / "workload_predictions.csv"
        unit_path = heavy_root / "trajectory_unit_scores.csv"
        pd.DataFrame(workload_rows).to_csv(workload_path, index=False)
        pd.DataFrame(unit_rows).to_csv(unit_path, index=False)
        checkpoint_hashes_after = {
            row["method_name"]: sha256_file(row["checkpoint_path"])
            for row in checkpoint_rows
        }
        acceptance_rows = build_application_consumer_acceptance_rows(
            data=data,
            checkpoint_rows=checkpoint_rows,
            target_manifest=targets.manifest,
            initial_export_rows=initial_export_rows,
            recovery_export_rows=recovery_export_rows,
            resume_export_rows=resume_export_rows,
            alignment_hashes=alignment_hashes,
            method_results=method_results,
            resumed_method_results=resumed_method_results,
            metric_rows=metric_rows,
            paired_rows=paired_rows,
            representation_recovery=representation_recovery,
            consumer_recovery=consumer_recovery,
            checkpoint_hashes_unchanged=(
                checkpoint_hashes_before == checkpoint_hashes_after
            ),
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        representation_manifest = {
            "format": "chronaris.application_context_representations.v1",
            "initial_exports": initial_export_rows,
            "single_item_recovery_exports": recovery_export_rows,
            "resume_exports": resume_export_rows,
            "alignment_sha256": alignment_hashes,
            "single_item_recovery": representation_recovery,
        }
        downstream_protocol = {
            "format": "chronaris.application_consumer_protocol.v2",
            "consumer_runtime_revision": "validation_selected_residual_tcn.v3",
            "config": asdict(protocol),
            "methods": list(APPLICATION_METHODS),
            "consumer_configuration_method_invariant": True,
            "fit_role": "train",
            "evaluation_roles": ["validation", "held_out"],
            "duration_parameters_fit_role": "train",
            "threshold_parameters_fit_role": "train",
            "minirocket_variance_filter_fit_role": "train",
            "minirocket_variance_filter_rule": (
                "retain channels whose within-window standard deviation exceeds "
                "the fixed threshold for every training context"
            ),
            "smoke_only": True,
        }
        paths = write_application_consumer_smoke_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            data_manifest={
                "format": "chronaris.application_context_data.v1",
                "sample_count": len(data.batch.sample_ids),
                "rows": list(data.sample_manifest_rows),
                "oracle_opened_for_representation": False,
            },
            target_manifest=targets.manifest,
            checkpoint_manifest={
                "format": "chronaris.application_checkpoint_reuse.v1",
                "pretraining_run_id": config.pretraining_run_id,
                "rows": list(checkpoint_rows),
                "hashes_before": checkpoint_hashes_before,
                "hashes_after": checkpoint_hashes_after,
            },
            representation_manifest=representation_manifest,
            downstream_protocol=downstream_protocol,
            model_manifests={
                "format": "chronaris.application_consumer_model_set.v1",
                "models": [result.model_manifest for result in method_results],
                "prediction_single_item_recovery": consumer_recovery,
            },
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            paired_rows=paired_rows,
            tcn_training_rows=tcn_training_rows,
            resource_rows=resource_rows,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
            workload_prediction_path=str(workload_path),
            unit_score_path=str(unit_path),
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            metric_count=len(metric_rows),
        )
        return ApplicationConsumerSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            representation_export_count=len(initial_export_rows),
            metric_count=len(metric_rows),
            fusion_gain_count=len(fusion_gain_rows),
            paired_statistic_count=len(paired_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _verify_representation_recovery(*, adapters, data, output_root, initial_export_rows):
    target = next(
        row
        for row in initial_export_rows
        if row["method_name"] == "chronaris" and row["role"] == "held_out"
    )
    destination = Path(target["output_root"])
    (destination / "fusion_stream.npz").unlink(missing_ok=True)
    (destination / "representation_manifest.json").unlink(missing_ok=True)
    outputs, rows, _ = export_application_context_representations(
        adapters=adapters,
        batch=data.batch,
        role_sample_ids=data.role_sample_ids,
        output_root=output_root,
        resume=True,
    )
    rebuilt = next(
        row
        for row in rows
        if row["method_name"] == "chronaris" and row["role"] == "held_out"
    )
    return outputs, rows, {
        "method_name": "chronaris",
        "role": "held_out",
        "original_sha256": target["representation_sha256"],
        "rebuilt_sha256": rebuilt["representation_sha256"],
        "hash_match": target["representation_sha256"] == rebuilt["representation_sha256"],
        "rebuild_status": rebuilt["status"],
    }


def _verify_consumer_component_recovery(
    *, chronaris_result, outputs, targets, output_root, fold_id, protocol
):
    original = chronaris_result.model_manifest
    Path(original["model_files"]["minirocket"]["path"]).unlink(missing_ok=True)
    rocket_rebuilt = run_application_method_consumers(
        method_name="chronaris",
        outputs=outputs,
        targets=targets,
        output_root=output_root,
        fold_id=fold_id,
        protocol=protocol,
        resume=True,
    )
    after_rocket = rocket_rebuilt.model_manifest
    Path(after_rocket["model_files"]["tcn"]["path"]).unlink(missing_ok=True)
    tcn_rebuilt = run_application_method_consumers(
        method_name="chronaris",
        outputs=outputs,
        targets=targets,
        output_root=output_root,
        fold_id=fold_id,
        protocol=protocol,
        resume=True,
    )
    after_tcn = tcn_rebuilt.model_manifest
    return {
        "method_name": "chronaris",
        "minirocket": {
            "component_status": rocket_rebuilt.component_status,
            "unrelated_linear_hash_unchanged": (
                original["model_files"]["linear"]["sha256"]
                == after_rocket["model_files"]["linear"]["sha256"]
            ),
            "unrelated_tcn_hash_unchanged": (
                original["model_files"]["tcn"]["sha256"]
                == after_rocket["model_files"]["tcn"]["sha256"]
            ),
            "prediction_hash_match": (
                original["prediction_sha256"] == after_rocket["prediction_sha256"]
            ),
        },
        "tcn": {
            "component_status": tcn_rebuilt.component_status,
            "unrelated_linear_hash_unchanged": (
                after_rocket["model_files"]["linear"]["sha256"]
                == after_tcn["model_files"]["linear"]["sha256"]
            ),
            "unrelated_minirocket_hash_unchanged": (
                after_rocket["model_files"]["minirocket"]["sha256"]
                == after_tcn["model_files"]["minirocket"]["sha256"]
            ),
            "prediction_hash_match": (
                after_rocket["prediction_sha256"] == after_tcn["prediction_sha256"]
            ),
        },
    }
