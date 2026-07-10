"""Smoke the two single-stream and naive time-sync production adapters."""

from __future__ import annotations

import json
import logging
import time
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from chronaris.evaluation.application_tasks.representation_contract_smoke import (
    _select_simulation_paths,
)
from chronaris.evaluation.application_tasks.shallow_adapter_reporting import (
    write_shallow_adapter_outputs,
)
from chronaris.evaluation.application_tasks.shallow_adapter_audit import (
    build_shallow_acceptance_rows,
    causal_audit,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import (
    ContinuousTimeSingleStreamEncoder,
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    SingleStreamEncoderConfig,
    SingleStreamFusionAdapter,
    load_naive_time_sync_checkpoint,
    load_single_stream_checkpoint,
    save_naive_time_sync_checkpoint,
    save_single_stream_checkpoint,
)
from chronaris.representation import (
    CheckpointRegistry,
    FoldLineage,
    ResumableOOFExporter,
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


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.shallow_adapter_smoke")
LOGGER.addHandler(logging.NullHandler())
SHALLOW_METHODS = ("physiology_only", "vehicle_only", "naive_time_sync")
METHOD_LABELS = {
    "physiology_only": "生理单流",
    "vehicle_only": "航电单流",
    "naive_time_sync": "朴素时间同步",
}
DATASET_LABELS = {"simulation": "仿真", "dingxin": "鼎新"}


@dataclass(frozen=True, slots=True)
class ShallowAdapterSmokeConfig:
    run_id: str = "2026-07-11_shallow-baseline-adapter-smoke"
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
class ShallowAdapterSmokeResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    acceptance_pass_count: int
    acceptance_check_count: int
    export_count: int
    resume_reused_count: int
    report_path: str
    evidence_manifest_path: str


def run_shallow_adapter_smoke(
    config: ShallowAdapterSmokeConfig,
) -> ShallowAdapterSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="shallow_adapter_smoke",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "downstream_metrics_produced": False,
        },
    ) as progress:
        datasets, dataset_metadata = _load_smoke_datasets(config)
        registry = CheckpointRegistry(compact_root / "checkpoint_registry.json")
        all_initial_results = []
        all_resume_results = []
        parameter_rows = []
        causal_rows = []
        transform_manifest: dict[str, object] = {
            "format": "chronaris.shallow_adapter_transforms.v1",
            "datasets": {},
        }
        alignment_hashes = {}

        for dataset_id, batch in datasets.items():
            fold = FoldLineage(
                fold_id=f"{dataset_id}_shallow_adapter_smoke",
                train_sample_ids=(batch.sample_ids[0],),
                validation_sample_ids=(batch.sample_ids[1],),
                held_out_sample_ids=(batch.sample_ids[2],),
            )
            adapters = _build_or_load_adapters(
                dataset_id=dataset_id,
                batch=batch,
                fold=fold,
                heavy_root=heavy_root,
                registry=registry,
                seed=config.seed,
            )
            held_out_batch = select_observation_batch(batch, fold.held_out_sample_ids)
            exporter = ResumableOOFExporter(
                heavy_root / "representations",
                resume=config.resume,
            )
            dataset_results = []
            dataset_outputs = []
            dataset_transforms = {}
            for method_name in SHALLOW_METHODS:
                adapter = adapters[method_name]
                checkpoint = registry.require(method_name, fold.fold_id)
                started = time.perf_counter()
                result = exporter.export(
                    encoder=adapter,
                    batch=held_out_batch,
                    checkpoint=checkpoint,
                    export_role="held_out",
                )
                elapsed = time.perf_counter() - started
                output = load_fusion_stream_batch(result.output_root)
                dataset_results.append(result)
                dataset_outputs.append(output)
                causal_row = causal_audit(
                    dataset_id=dataset_id,
                    method_name=method_name,
                    adapter=adapter,
                    held_out_batch=held_out_batch,
                    baseline_output=output,
                )
                causal_rows.append(causal_row)
                parameter_rows.append(
                    _parameter_row(
                        dataset_id=dataset_id,
                        method_name=method_name,
                        adapter=adapter,
                        batch=batch,
                        elapsed_s=elapsed,
                        output=output,
                    )
                )
                dataset_transforms[method_name] = _transform_summary(adapter)
            alignment_hashes[dataset_id] = validate_fusion_method_alignment(
                dataset_outputs
            )
            all_initial_results.extend(dataset_results)
            all_resume_results.extend(
                exporter.export(
                    encoder=adapters[method_name],
                    batch=held_out_batch,
                    checkpoint=registry.require(method_name, fold.fold_id),
                    export_role="held_out",
                )
                for method_name in SHALLOW_METHODS
            )
            transform_manifest["datasets"][dataset_id] = {
                "fold": fold.to_dict(),
                "methods": dataset_transforms,
            }
            progress.update(
                "dataset_adapters_complete",
                dataset_id=dataset_id,
                export_count=len(dataset_results),
            )

        export_manifest = {
            "format": "chronaris.shallow_adapter_exports.v1",
            "available_export_count": len(all_initial_results),
            "current_run_built_count": sum(
                result.status == "completed" for result in all_initial_results
            ),
            "current_run_reused_count": sum(
                result.status == "resumed" for result in all_initial_results
            ),
            "resume_verification_reused_count": sum(
                result.status == "resumed" for result in all_resume_results
            ),
            "exports": [result.to_dict() for result in all_initial_results],
            "alignment_sha256": alignment_hashes,
        }
        adapter_protocol = {
            "format": "chronaris.shallow_adapter_protocol.v1",
            "methods": list(SHALLOW_METHODS),
            "single_stream_backbone_class": "ContinuousTimeSingleStreamEncoder",
            "single_stream_continuous_block": "ContiFormerEncoder(causal=True)",
            "naive_sync_policy": "past_or_present_forward_fill_only",
            "query_point_count": 96,
            "output_dim": 64,
            "label_used_for_encoder_training": False,
            "training_invoked": False,
            "dataset_metadata": dataset_metadata,
        }
        acceptance_rows = build_shallow_acceptance_rows(
            registry=registry,
            parameter_rows=parameter_rows,
            causal_rows=causal_rows,
            export_manifest=export_manifest,
            alignment_hashes=alignment_hashes,
            dataset_metadata=dataset_metadata,
            transform_manifest=transform_manifest,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = write_shallow_adapter_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            adapter_protocol=adapter_protocol,
            causal_rows=causal_rows,
            parameter_rows=parameter_rows,
            transform_manifest=transform_manifest,
            export_manifest=export_manifest,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
        )
        pass_count = sum(bool(row["passed"]) for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            export_count=len(all_initial_results),
            resume_reused_count=export_manifest["resume_verification_reused_count"],
        )
        return ShallowAdapterSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            export_count=len(all_initial_results),
            resume_reused_count=int(
                export_manifest["resume_verification_reused_count"]
            ),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _load_smoke_datasets(config: ShallowAdapterSmokeConfig):
    simulation_paths = _select_simulation_paths(Path(config.simulation_root))
    simulation_samples = tuple(
        load_simulation_observed_context(path, context_start_s=0.0)
        for path in simulation_paths.values()
    )
    dingxin_plan = build_dingxin_observation_schema_plan(
        snapshot_root=config.dingxin_snapshot_root,
        field_role_manifest_path=config.field_role_manifest_path,
    )
    dingxin_contexts = _one_context_per_view(Path(config.context_manifest_path))
    dingxin_samples = tuple(
        load_dingxin_observed_context(dingxin_plan, context)
        for context in dingxin_contexts
    )
    datasets = {
        "simulation": collate_observation_samples(simulation_samples),
        "dingxin": collate_observation_samples(dingxin_samples),
    }
    metadata = {
        "simulation": {
            "sample_count": len(simulation_samples),
            "physiology_feature_count": len(
                simulation_samples[0].schema.physiology_feature_names
            ),
            "vehicle_feature_count": len(
                simulation_samples[0].schema.vehicle_feature_names
            ),
            "oracle_opened": False,
        },
        "dingxin": {
            "sample_count": len(dingxin_samples),
            "view_count": len({sample.group_id for sample in dingxin_samples}),
            "physiology_feature_count": len(
                dingxin_plan.schema.physiology_feature_names
            ),
            "vehicle_feature_count": len(dingxin_plan.schema.vehicle_feature_names),
            "excluded_label_source_count": len(
                dingxin_plan.schema.excluded_feature_names
            ),
            "label_or_target_opened": False,
        },
    }
    return datasets, metadata


def _one_context_per_view(path: Path) -> tuple[Mapping[str, object], ...]:
    by_view = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("classification_eligible") or not payload.get("response_eligible"):
            continue
        by_view.setdefault(str(payload["view_id"]), payload)
    if len(by_view) < 3:
        raise ValueError("shallow adapter smoke requires three distinct Dingxin views")
    return tuple(by_view[key] for key in sorted(by_view)[:3])


def _build_or_load_adapters(
    *,
    dataset_id,
    batch,
    fold,
    heavy_root,
    registry,
    seed,
):
    checkpoint_root = heavy_root / "checkpoints" / dataset_id
    checkpoint_paths = {
        method: checkpoint_root / f"{method}.pt" for method in SHALLOW_METHODS
    }
    common_normalizer = None

    def require_normalizer():
        nonlocal common_normalizer
        if common_normalizer is None:
            common_normalizer = TrainOnlyRobustNormalizer().fit(
                batch,
                train_sample_ids=fold.train_sample_ids,
                held_out_sample_ids=fold.held_out_sample_ids,
            )
        return common_normalizer

    adapters = {}
    for method_name in SHALLOW_METHODS:
        path = checkpoint_paths[method_name]
        if not path.exists():
            if method_name == "naive_time_sync":
                encoder = NaiveTimeSyncEncoder().fit(
                    batch,
                    train_sample_ids=fold.train_sample_ids,
                    held_out_sample_ids=fold.held_out_sample_ids,
                    normalizer=require_normalizer(),
                )
                save_naive_time_sync_checkpoint(path, encoder=encoder)
            else:
                active_stream = (
                    "physiology" if method_name == "physiology_only" else "vehicle"
                )
                input_dim = (
                    batch.physiology_values.shape[-1]
                    if active_stream == "physiology"
                    else batch.vehicle_values.shape[-1]
                )
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    backbone = ContinuousTimeSingleStreamEncoder(
                        SingleStreamEncoderConfig(
                            active_stream=active_stream,
                            input_feature_dim=int(input_dim),
                        )
                    )
                save_single_stream_checkpoint(
                    path,
                    backbone=backbone,
                    normalizer=require_normalizer(),
                    seed=seed,
                )
        checkpoint = build_checkpoint_record(
            method_name=method_name,
            fold=fold,
            checkpoint_path=path,
            seed=seed,
        )
        registry.register(checkpoint)
        if method_name == "naive_time_sync":
            adapters[method_name] = NaiveTimeSyncFusionAdapter(
                encoder=load_naive_time_sync_checkpoint(path),
                fold_id=fold.fold_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
            )
        else:
            backbone, normalizer, _metadata = load_single_stream_checkpoint(path)
            adapters[method_name] = SingleStreamFusionAdapter(
                backbone=backbone,
                normalizer=normalizer,
                fold_id=fold.fold_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
            )
    return adapters


def _parameter_row(*, dataset_id, method_name, adapter, batch, elapsed_s, output):
    if method_name == "physiology_only":
        input_count = int(batch.physiology_values.shape[-1])
        parameters = adapter.parameter_count
    elif method_name == "vehicle_only":
        input_count = int(batch.vehicle_values.shape[-1])
        parameters = adapter.parameter_count
    else:
        input_count = int(
            batch.physiology_values.shape[-1] + batch.vehicle_values.shape[-1]
        )
        parameters = 0
    return {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "method_name": method_name,
        "method_label": METHOD_LABELS[method_name],
        "input_feature_count": input_count,
        "parameter_count": int(parameters),
        "output_dim": int(output.sequence_embedding.shape[-1]),
        "query_point_count": int(output.sequence_embedding.shape[1]),
        "export_elapsed_s": float(elapsed_s),
        "process_peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def _transform_summary(adapter):
    manifest = adapter.to_manifest()
    normalizer = manifest["normalizer"]
    result = {
        "fit_sample_hash": normalizer["fit_sample_hash"],
        "normalizer_sha256": normalizer["transform_sha256"],
        "label_used_for_encoder_training": False,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    if "pca_projector" in manifest:
        result["pca_fit_sample_hash"] = manifest["pca_projector"]["fit_sample_hash"]
        result["pca_transform_sha256"] = manifest["pca_projector"]["transform_sha256"]
        result["pca_component_count"] = manifest["pca_projector"]["component_count"]
    return result
