"""Run the fixed-data representation infrastructure smoke without model training."""

from __future__ import annotations

import importlib.metadata
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from chronaris.evaluation.application_tasks.representation_contract_reporting import (
    write_representation_contract_outputs,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.representation import (
    SIX_METHOD_NAMES,
    CheckpointRegistry,
    ContractProbeEncoder,
    FoldLineage,
    ResumableOOFExporter,
    TrainOnlyPCAProjector,
    TrainOnlyRobustNormalizer,
    build_checkpoint_record,
    build_dingxin_observation_schema_plan,
    collate_observation_samples,
    load_dingxin_observed_context,
    load_fusion_stream_batch,
    load_simulation_observed_context,
    select_observation_batch,
    validate_fusion_method_alignment,
)
from chronaris.representation.contracts import (
    FORBIDDEN_REPRESENTATION_FIELDS,
    FUSION_OUTPUT_DIM,
    QUERY_POINT_COUNT,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.representation_contract_smoke")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class RepresentationContractSmokeConfig:
    run_id: str = "2026-07-11_representation-contract-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    dingxin_snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    field_role_manifest_path: str = (
        "docs/artifacts/runs/2026-07-10_fixed-data-audit/field_role_manifest.csv"
    )
    context_manifest_path: str = (
        "docs/artifacts/runs/2026-07-10_fixed-data-audit/context_sample_manifest.jsonl"
    )
    seed: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class RepresentationContractSmokeResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    acceptance_pass_count: int
    acceptance_check_count: int
    representation_export_count: int
    resume_reused_count: int
    report_path: str
    evidence_manifest_path: str


def run_representation_contract_smoke(
    config: RepresentationContractSmokeConfig,
) -> RepresentationContractSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="representation_contract_smoke",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "probe_outputs_are_model_results": False,
        },
    ) as progress:
        simulation_paths = _select_simulation_paths(Path(config.simulation_root))
        simulation_samples = tuple(
            load_simulation_observed_context(path, context_start_s=0.0)
            for path in simulation_paths.values()
        )
        simulation_batch = collate_observation_samples(simulation_samples)
        progress.update(
            "simulation_batch_ready",
            sample_count=len(simulation_batch.sample_ids),
        )

        dingxin_plan = build_dingxin_observation_schema_plan(
            snapshot_root=config.dingxin_snapshot_root,
            field_role_manifest_path=config.field_role_manifest_path,
        )
        dingxin_context = _first_eligible_context(Path(config.context_manifest_path))
        dingxin_sample = load_dingxin_observed_context(dingxin_plan, dingxin_context)
        dingxin_batch = collate_observation_samples([dingxin_sample])
        progress.update(
            "dingxin_batch_ready",
            physiology_feature_count=len(dingxin_plan.schema.physiology_feature_names),
            vehicle_feature_count=len(dingxin_plan.schema.vehicle_feature_names),
        )

        train_id, validation_id, held_out_id = simulation_batch.sample_ids
        fold = FoldLineage(
            fold_id="simulation_out_of_family_smoke",
            train_sample_ids=(train_id,),
            validation_sample_ids=(validation_id,),
            held_out_sample_ids=(held_out_id,),
        )
        normalizer = TrainOnlyRobustNormalizer().fit(
            simulation_batch,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=fold.held_out_sample_ids,
        )
        normalized_batch = normalizer.transform(simulation_batch)
        pca_values = _batch_feature_means(normalized_batch)
        pca = TrainOnlyPCAProjector().fit(
            pca_values,
            row_sample_ids=normalized_batch.sample_ids,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=fold.held_out_sample_ids,
        )
        pca_held_out = pca.transform(pca_values[[-1]])
        held_out_batch = select_observation_batch(
            normalized_batch,
            fold.held_out_sample_ids,
        )

        registry = CheckpointRegistry(compact_root / "checkpoint_registry.json")
        exporter = ResumableOOFExporter(
            heavy_root / "representations",
            resume=config.resume,
        )
        checkpoints = {}
        initial_results = []
        for method_name in SIX_METHOD_NAMES:
            checkpoint_path = heavy_root / "checkpoints" / f"{method_name}.probe"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint_path.exists():
                checkpoint_path.write_text(
                    f"contract-probe-only\nmethod={method_name}\nseed={config.seed}\n",
                    encoding="utf-8",
                )
            checkpoint = build_checkpoint_record(
                method_name=method_name,
                fold=fold,
                checkpoint_path=checkpoint_path,
                seed=config.seed,
            )
            registry.register(checkpoint)
            checkpoints[method_name] = checkpoint
            initial_results.append(
                exporter.export(
                    encoder=ContractProbeEncoder(
                        method_name=method_name,
                        fold_id=fold.fold_id,
                        checkpoint_sha256=checkpoint.checkpoint_sha256,
                    ),
                    batch=held_out_batch,
                    checkpoint=checkpoint,
                    export_role="held_out",
                )
            )
        progress.update(
            "initial_exports_complete",
            export_count=len(initial_results),
        )
        resume_results = [
            exporter.export(
                encoder=ContractProbeEncoder(
                    method_name=method_name,
                    fold_id=fold.fold_id,
                    checkpoint_sha256=checkpoints[method_name].checkpoint_sha256,
                ),
                batch=held_out_batch,
                checkpoint=checkpoints[method_name],
                export_role="held_out",
            )
            for method_name in SIX_METHOD_NAMES
        ]
        outputs = [
            load_fusion_stream_batch(result.output_root) for result in initial_results
        ]
        alignment_hash = validate_fusion_method_alignment(outputs)
        aeon_version = _optional_package_version("aeon")

        input_schema = _input_schema_payload(
            simulation_samples=simulation_samples,
            simulation_paths=simulation_paths,
            simulation_batch=simulation_batch,
            dingxin_sample=dingxin_sample,
            dingxin_batch=dingxin_batch,
            dingxin_plan=dingxin_plan,
        )
        representation_schema = {
            "format": "chronaris.fusion_stream.v1",
            "output_dim": FUSION_OUTPUT_DIM,
            "query_point_count": QUERY_POINT_COUNT,
            "forbidden_fields": sorted(FORBIDDEN_REPRESENTATION_FIELDS),
            "label_used_for_encoder_training": False,
            "contract_probe_only": True,
            "probe_outputs_are_model_results": False,
            "method_slots": list(SIX_METHOD_NAMES),
        }
        sample_order_manifest = {
            "fold": fold.to_dict(),
            "simulation_sample_order": list(simulation_batch.sample_ids),
            "held_out_export_order": list(held_out_batch.sample_ids),
            "held_out_source_hashes": list(held_out_batch.source_sample_hashes),
            "dingxin_smoke_sample_id": dingxin_sample.sample_id,
            "alignment_sha256": alignment_hash,
        }
        fold_transform_manifest = {
            "fold_id": fold.fold_id,
            "robust_normalizer": normalizer.to_manifest(),
            "pca_projector": pca.to_manifest(),
            "pca_held_out_shape": list(pca_held_out.shape),
            "held_out_sample_ids": list(fold.held_out_sample_ids),
        }
        oof_export_manifest = {
            "format": "chronaris.oof_export_smoke.v1",
            "contract_probe_only": True,
            "available_export_count": len(initial_results),
            "current_run_built_count": sum(
                result.status == "completed" for result in initial_results
            ),
            "current_run_initial_reused_count": sum(
                result.status == "resumed" for result in initial_results
            ),
            "resume_verification_reused_count": sum(
                result.status == "resumed" for result in resume_results
            ),
            "exports": [result.to_dict() for result in initial_results],
        }
        acceptance_rows = _acceptance_rows(
            simulation_batch=simulation_batch,
            dingxin_batch=dingxin_batch,
            dingxin_excluded_count=len(dingxin_plan.schema.excluded_feature_names),
            normalizer=normalizer,
            pca=pca,
            fold=fold,
            registry=registry,
            outputs=outputs,
            initial_results=initial_results,
            resume_results=resume_results,
            alignment_hash=alignment_hash,
            aeon_version=aeon_version,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = write_representation_contract_outputs(
            compact_root=compact_root,
            run_id=config.run_id,
            status=status,
            input_schema=input_schema,
            representation_schema=representation_schema,
            sample_order_manifest=sample_order_manifest,
            fold_transform_manifest=fold_transform_manifest,
            oof_export_manifest=oof_export_manifest,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
        )
        pass_count = sum(bool(row["passed"]) for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            representation_export_count=len(initial_results),
            resume_reused_count=oof_export_manifest["resume_verification_reused_count"],
        )
        return RepresentationContractSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            representation_export_count=len(initial_results),
            resume_reused_count=int(
                oof_export_manifest["resume_verification_reused_count"]
            ),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _select_simulation_paths(root: Path) -> Mapping[str, Path]:
    selected = {}
    for split_id in ("train", "validation", "locked_test"):
        candidates = sorted(
            (root / split_id).glob("**/clean_asynchronous/raw_dual_stream.npz")
        )
        if not candidates:
            raise FileNotFoundError(
                f"no clean simulation observed archive for split {split_id} under {root}"
            )
        selected[split_id] = candidates[0]
    return selected


def _first_eligible_context(path: Path) -> Mapping[str, object]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("classification_eligible") and payload.get("response_eligible"):
            return payload
    raise ValueError(f"no eligible Dingxin context in {path}")


def _batch_feature_means(batch) -> np.ndarray:
    parts = []
    for values, mask in (
        (batch.physiology_values, batch.physiology_feature_mask),
        (batch.vehicle_values, batch.vehicle_feature_mask),
    ):
        counts = mask.sum(dim=1).clamp_min(1)
        mean = (values * mask.to(values.dtype)).sum(dim=1) / counts
        parts.append(mean.detach().cpu().numpy())
    return np.concatenate(parts, axis=1).astype(np.float64)


def _input_schema_payload(
    *,
    simulation_samples,
    simulation_paths,
    simulation_batch,
    dingxin_sample,
    dingxin_batch,
    dingxin_plan,
) -> Mapping[str, object]:
    return {
        "format": "chronaris.dual_stream_observation_smoke.v1",
        "simulation": {
            "schema": simulation_samples[0].schema.to_dict(),
            "schema_sha256": simulation_samples[0].schema.schema_sha256,
            "physiology_feature_count": len(
                simulation_samples[0].schema.physiology_feature_names
            ),
            "vehicle_feature_count": len(
                simulation_samples[0].schema.vehicle_feature_names
            ),
            "sample_count": len(simulation_batch.sample_ids),
            "observed_archive_paths": {
                key: str(value) for key, value in simulation_paths.items()
            },
            "oracle_archive_opened": False,
        },
        "dingxin": {
            "schema": dingxin_plan.schema.to_dict(),
            "schema_sha256": dingxin_plan.schema.schema_sha256,
            "physiology_feature_count": len(
                dingxin_plan.schema.physiology_feature_names
            ),
            "vehicle_feature_count": len(dingxin_plan.schema.vehicle_feature_names),
            "excluded_feature_count": len(dingxin_plan.schema.excluded_feature_names),
            "sample_id": dingxin_sample.sample_id,
            "physiology_point_count": int(dingxin_batch.physiology_point_mask.sum()),
            "vehicle_point_count": int(dingxin_batch.vehicle_point_mask.sum()),
            "label_or_target_returned_by_loader": False,
        },
        "query_point_count": QUERY_POINT_COUNT,
        "context_duration_s": 30.0,
    }


def _acceptance_rows(
    *,
    simulation_batch,
    dingxin_batch,
    dingxin_excluded_count,
    normalizer,
    pca,
    fold,
    registry,
    outputs,
    initial_results,
    resume_results,
    alignment_hash,
    aeon_version,
) -> list[dict[str, object]]:
    return [
        _check("simulation_observed_batch", len(simulation_batch.sample_ids) == 3, len(simulation_batch.sample_ids), 3),
        _check("dingxin_observed_batch", len(dingxin_batch.sample_ids) == 1, len(dingxin_batch.sample_ids), 1),
        _check("query_contract_96", simulation_batch.query_timestamps_s.shape[1] == QUERY_POINT_COUNT and dingxin_batch.query_timestamps_s.shape[1] == QUERY_POINT_COUNT, QUERY_POINT_COUNT, QUERY_POINT_COUNT),
        _check("dingxin_label_sources_excluded", dingxin_excluded_count == 20, dingxin_excluded_count, 20),
        _check("normalizer_train_only", set(normalizer.fit_sample_ids).isdisjoint(fold.held_out_sample_ids), list(normalizer.fit_sample_ids), list(fold.held_out_sample_ids)),
        _check("pca_train_only", set(pca.fit_sample_ids).isdisjoint(fold.held_out_sample_ids), list(pca.fit_sample_ids), list(fold.held_out_sample_ids)),
        _check("checkpoint_registry_complete", len(registry.records) == len(SIX_METHOD_NAMES), len(registry.records), len(SIX_METHOD_NAMES)),
        _check("six_method_slot_exports", len(initial_results) == len(SIX_METHOD_NAMES), len(initial_results), len(SIX_METHOD_NAMES)),
        _check("resume_reuses_complete_exports", all(result.status == "resumed" for result in resume_results), [result.status for result in resume_results], "all resumed"),
        _check("method_alignment", len(alignment_hash) == 64, alignment_hash, "SHA-256"),
        _check("fixed_output_dimension", all(output.sequence_embedding.shape[-1] == FUSION_OUTPUT_DIM for output in outputs), [output.sequence_embedding.shape[-1] for output in outputs], FUSION_OUTPUT_DIM),
        _check("held_out_checkpoint_lineage", all(output.sample_ids == fold.held_out_sample_ids for output in outputs), [list(output.sample_ids) for output in outputs], list(fold.held_out_sample_ids)),
        _check("task_labels_not_used", all(not record.label_used_for_encoder_training for record in registry.records.values()), False, False),
        _check("aeon_optional_dependency", aeon_version == "1.5.0", aeon_version, "1.5.0"),
    ]


def _check(check_id: str, passed: bool, actual: object, expected: object) -> dict[str, object]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _optional_package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
