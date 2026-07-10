"""Smoke task-head-free causal MulT and ContiFormer production adapters."""

from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.adapter_smoke_inputs import (
    load_adapter_smoke_datasets,
)
from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import (
    audit_deep_baseline_output,
    build_deep_baseline_acceptance_rows,
)
from chronaris.evaluation.application_tasks.deep_baseline_adapter_reporting import (
    write_deep_baseline_adapter_outputs,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import (
    CausalContiFormerFusionEncoder,
    CausalMulTFusionEncoder,
    DeepBaselineEncoderConfig,
    DeepBaselineFusionAdapter,
    load_deep_baseline_checkpoint,
    save_deep_baseline_checkpoint,
)
from chronaris.representation import (
    CheckpointRegistry,
    FoldLineage,
    ResumableOOFExporter,
    TrainOnlyRobustNormalizer,
    build_checkpoint_record,
    load_fusion_stream_batch,
    select_observation_batch,
    validate_fusion_method_alignment,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.deep_baseline_adapter_smoke")
LOGGER.addHandler(logging.NullHandler())
DEEP_BASELINE_METHODS = ("mult", "contiformer")
METHOD_LABELS = {"mult": "MulT", "contiformer": "ContiFormer"}
DATASET_LABELS = {"simulation": "仿真", "dingxin": "鼎新"}


@dataclass(frozen=True, slots=True)
class DeepBaselineAdapterSmokeConfig:
    run_id: str = "2026-07-11_deep-baseline-adapter-smoke"
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
class DeepBaselineAdapterSmokeResult:
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


def run_deep_baseline_adapter_smoke(
    config: DeepBaselineAdapterSmokeConfig,
) -> DeepBaselineAdapterSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="deep_baseline_adapter_smoke",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "downstream_metrics_produced": False,
        },
    ) as progress:
        datasets, dataset_metadata = load_adapter_smoke_datasets(
            simulation_root=config.simulation_root,
            dingxin_snapshot_root=config.dingxin_snapshot_root,
            field_role_manifest_path=config.field_role_manifest_path,
            context_manifest_path=config.context_manifest_path,
        )
        registry = CheckpointRegistry(compact_root / "checkpoint_registry.json")
        initial_results = []
        resume_results = []
        causality_rows = []
        sensitivity_rows = []
        parameter_rows = []
        transform_manifest: dict[str, object] = {
            "format": "chronaris.deep_baseline_transforms.v1",
            "datasets": {},
        }
        alignment_hashes = {}

        for dataset_id, batch in datasets.items():
            fold = FoldLineage(
                fold_id=f"{dataset_id}_deep_baseline_smoke",
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
            outputs = []
            dataset_transforms = {}
            for method_name in DEEP_BASELINE_METHODS:
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
                initial_results.append(result)
                outputs.append(output)
                causality, sensitivity = audit_deep_baseline_output(
                    dataset_id=dataset_id,
                    method_name=method_name,
                    adapter=adapter,
                    held_out_batch=held_out_batch,
                    baseline_output=output,
                )
                causality_rows.append(causality)
                sensitivity_rows.append(sensitivity)
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
            alignment_hashes[dataset_id] = validate_fusion_method_alignment(outputs)
            resume_results.extend(
                exporter.export(
                    encoder=adapters[method_name],
                    batch=held_out_batch,
                    checkpoint=registry.require(method_name, fold.fold_id),
                    export_role="held_out",
                )
                for method_name in DEEP_BASELINE_METHODS
            )
            transform_manifest["datasets"][dataset_id] = {
                "fold": fold.to_dict(),
                "methods": dataset_transforms,
            }
            progress.update(
                "dataset_deep_baselines_complete",
                dataset_id=dataset_id,
                export_count=len(outputs),
            )

        export_manifest = {
            "format": "chronaris.deep_baseline_exports.v1",
            "available_export_count": len(initial_results),
            "current_run_built_count": sum(
                result.status == "completed" for result in initial_results
            ),
            "current_run_reused_count": sum(
                result.status == "resumed" for result in initial_results
            ),
            "resume_verification_reused_count": sum(
                result.status == "resumed" for result in resume_results
            ),
            "exports": [result.to_dict() for result in initial_results],
            "alignment_sha256": alignment_hashes,
        }
        adapter_protocol = {
            "format": "chronaris.deep_baseline_adapter_protocol.v1",
            "methods": list(DEEP_BASELINE_METHODS),
            "sequence_source": "task_head_free",
            "causal_query": True,
            "causal_attention": True,
            "hidden_dim": 64,
            "layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "query_point_count": 96,
            "output_dim": 64,
            "seed": config.seed,
            "label_used_for_encoder_training": False,
            "training_invoked": False,
            "dataset_metadata": dataset_metadata,
        }
        acceptance_rows = build_deep_baseline_acceptance_rows(
            registry=registry,
            parameter_rows=parameter_rows,
            causality_rows=causality_rows,
            sensitivity_rows=sensitivity_rows,
            export_manifest=export_manifest,
            alignment_hashes=alignment_hashes,
            dataset_metadata=dataset_metadata,
            transform_manifest=transform_manifest,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = write_deep_baseline_adapter_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            adapter_protocol=adapter_protocol,
            causality_rows=causality_rows,
            sensitivity_rows=sensitivity_rows,
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
            export_count=len(initial_results),
            resume_reused_count=export_manifest["resume_verification_reused_count"],
        )
        return DeepBaselineAdapterSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            export_count=len(initial_results),
            resume_reused_count=int(
                export_manifest["resume_verification_reused_count"]
            ),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


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
    paths = {
        method: checkpoint_root / f"{method}.pt" for method in DEEP_BASELINE_METHODS
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
    for method_name, path in paths.items():
        if not path.exists():
            config = DeepBaselineEncoderConfig(
                method_name=method_name,
                physiology_feature_dim=int(batch.physiology_values.shape[-1]),
                vehicle_feature_dim=int(batch.vehicle_values.shape[-1]),
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                backbone = (
                    CausalMulTFusionEncoder(config)
                    if method_name == "mult"
                    else CausalContiFormerFusionEncoder(config)
                )
            save_deep_baseline_checkpoint(
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
        backbone, normalizer, _metadata = load_deep_baseline_checkpoint(path)
        adapters[method_name] = DeepBaselineFusionAdapter(
            backbone=backbone,
            normalizer=normalizer,
            fold_id=fold.fold_id,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
        )
    return adapters


def _parameter_row(*, dataset_id, method_name, adapter, batch, elapsed_s, output):
    return {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "method_name": method_name,
        "method_label": METHOD_LABELS[method_name],
        "input_feature_count": int(
            batch.physiology_values.shape[-1] + batch.vehicle_values.shape[-1]
        ),
        "parameter_count": adapter.parameter_count,
        "output_dim": int(output.sequence_embedding.shape[-1]),
        "query_point_count": int(output.sequence_embedding.shape[1]),
        "export_elapsed_s": float(elapsed_s),
        "process_peak_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "sequence_source": "task_head_free",
        "causal_attention": True,
    }


def _transform_summary(adapter):
    manifest = adapter.to_manifest()
    normalizer = manifest["normalizer"]
    return {
        "fit_sample_hash": normalizer["fit_sample_hash"],
        "normalizer_sha256": normalizer["transform_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "sequence_source": manifest["sequence_source"],
        "causal_attention": manifest["causal_attention"],
        "label_used_for_encoder_training": False,
    }
