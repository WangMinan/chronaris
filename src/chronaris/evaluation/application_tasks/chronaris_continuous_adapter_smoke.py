"""Smoke the complete task-independent Chronaris continuous fusion path."""

from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.adapter_smoke_inputs import (
    load_adapter_smoke_datasets,
    load_adapter_smoke_schema_contexts,
)
from chronaris.evaluation.application_tasks.chronaris_continuous_adapter_audit import (
    audit_chronaris_output,
    build_chronaris_acceptance_rows,
    build_lag_boundary_audit_rows,
)
from chronaris.evaluation.application_tasks.chronaris_continuous_adapter_reporting import (
    write_chronaris_continuous_adapter_outputs,
)
from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import (
    perturb_future_observations,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import (
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousFusionAdapter,
    ChronarisContinuousFusionEncoder,
    build_chronaris_ablation_configs,
    load_chronaris_continuous_checkpoint,
    save_chronaris_continuous_checkpoint,
    validate_chronaris_ablation_diff,
)
from chronaris.representation import (
    CheckpointRegistry,
    FoldLineage,
    ResumableOOFExporter,
    TrainOnlyRobustNormalizer,
    build_checkpoint_record,
    load_fusion_stream_batch,
    select_observation_batch,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.chronaris_continuous_adapter_smoke")
LOGGER.addHandler(logging.NullHandler())
DATASET_LABELS = {"simulation": "仿真", "dingxin": "鼎新"}


@dataclass(frozen=True, slots=True)
class ChronarisContinuousAdapterSmokeConfig:
    run_id: str = "2026-07-11_chronaris-continuous-adapter-smoke"
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
class ChronarisContinuousAdapterSmokeResult:
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


def run_chronaris_continuous_adapter_smoke(
    config: ChronarisContinuousAdapterSmokeConfig,
) -> ChronarisContinuousAdapterSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="chronaris_continuous_adapter_smoke",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "downstream_metrics_produced": False,
            "simulation_oracle_opened": False,
        },
    ) as progress:
        datasets, dataset_metadata = load_adapter_smoke_datasets(
            simulation_root=config.simulation_root,
            dingxin_snapshot_root=config.dingxin_snapshot_root,
            field_role_manifest_path=config.field_role_manifest_path,
            context_manifest_path=config.context_manifest_path,
        )
        schema_contexts = load_adapter_smoke_schema_contexts(
            simulation_root=config.simulation_root,
            dingxin_snapshot_root=config.dingxin_snapshot_root,
            field_role_manifest_path=config.field_role_manifest_path,
        )
        registry = CheckpointRegistry(compact_root / "checkpoint_registry.json")
        initial_results = []
        resume_results = []
        path_rows = []
        physics_rows = []
        ablation_rows = []
        causality_rows = []
        sensitivity_rows = []
        parameter_rows = []
        transform_manifest: dict[str, object] = {
            "format": "chronaris.continuous_adapter_transforms.v1",
            "datasets": {},
        }

        for dataset_id, batch in datasets.items():
            fold = FoldLineage(
                fold_id=f"{dataset_id}_chronaris_continuous_smoke",
                train_sample_ids=(batch.sample_ids[0],),
                validation_sample_ids=(batch.sample_ids[1],),
                held_out_sample_ids=(batch.sample_ids[2],),
            )
            adapter = _build_or_load_adapter(
                dataset_id=dataset_id,
                batch=batch,
                fold=fold,
                schema_context=schema_contexts[dataset_id],
                heavy_root=heavy_root,
                registry=registry,
                seed=config.seed,
            )
            held_out = select_observation_batch(batch, fold.held_out_sample_ids)
            checkpoint = registry.require("chronaris", fold.fold_id)
            exporter = ResumableOOFExporter(
                heavy_root / "representations",
                resume=config.resume,
            )
            result = exporter.export(
                encoder=adapter,
                batch=held_out,
                checkpoint=checkpoint,
                export_role="held_out",
            )
            initial_results.append(result)
            output = load_fusion_stream_batch(result.output_root)
            started = time.perf_counter()
            audit_output = adapter(held_out)
            elapsed = time.perf_counter() - started
            baseline_encoding = adapter.last_encoding
            if baseline_encoding is None:
                raise RuntimeError("Chronaris adapter did not retain mechanism audit")
            if not torch.equal(output.sequence_embedding, audit_output.sequence_embedding):
                raise RuntimeError("exported Chronaris output differs from direct audit output")
            causality, sensitivity, paths, physics = audit_chronaris_output(
                dataset_id=dataset_id,
                adapter=adapter,
                held_out_batch=held_out,
                baseline_output=audit_output,
                baseline_encoding=baseline_encoding,
            )
            causality_rows.append(causality)
            sensitivity_rows.append(sensitivity)
            path_rows.extend(paths)
            physics_rows.extend(physics)
            parameter_rows.append(
                _parameter_row(
                    dataset_id=dataset_id,
                    adapter=adapter,
                    batch=batch,
                    output=output,
                    elapsed_s=elapsed,
                )
            )
            ablation_rows.extend(
                _audit_ablations(
                    dataset_id=dataset_id,
                    full_config=adapter.backbone.config,
                    normalizer=adapter.normalizer,
                    held_out_batch=held_out,
                    seed=config.seed,
                )
            )
            resume_results.append(
                exporter.export(
                    encoder=adapter,
                    batch=held_out,
                    checkpoint=checkpoint,
                    export_role="held_out",
                )
            )
            normalizer_manifest = adapter.normalizer.to_manifest()
            transform_manifest["datasets"][dataset_id] = {
                "fold": fold.to_dict(),
                "method": {
                    "fit_sample_hash": normalizer_manifest["fit_sample_hash"],
                    "normalizer_sha256": normalizer_manifest["transform_sha256"],
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "label_used_for_encoder_training": False,
                },
            }
            progress.update(
                "dataset_chronaris_complete",
                dataset_id=dataset_id,
                physics_active_count=baseline_encoding.physics_audit.active_component_count,
            )

        export_manifest = {
            "format": "chronaris.continuous_adapter_exports.v1",
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
        }
        lag_rows = build_lag_boundary_audit_rows()
        acceptance_rows = build_chronaris_acceptance_rows(
            registry=registry,
            parameter_rows=parameter_rows,
            causality_rows=causality_rows,
            sensitivity_rows=sensitivity_rows,
            path_rows=path_rows,
            physics_rows=physics_rows,
            ablation_rows=ablation_rows,
            lag_boundary_rows=lag_rows,
            export_manifest=export_manifest,
            dataset_metadata=dataset_metadata,
            transform_manifest=transform_manifest,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        adapter_protocol = {
            "format": "chronaris.continuous_adapter_protocol.v1",
            "method_name": "chronaris",
            "sequence_source": "task_head_free_continuous_fusion",
            "query_point_count": 96,
            "output_dim": 64,
            "ode_method": "euler",
            "lag_ranges_s": [[0.0, 5.0], [5.0, 15.0], [15.0, 30.0]],
            "physics_weight": 0.1,
            "fixed_ablations": [
                "no_continuous_evolution",
                "no_physics",
                "no_causal_mask",
                "single_scale_lag",
            ],
            "seed": config.seed,
            "training_invoked": False,
            "label_used_for_encoder_training": False,
            "simulation_oracle_opened": False,
            "dataset_metadata": dataset_metadata,
        }
        paths = write_chronaris_continuous_adapter_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            adapter_protocol=adapter_protocol,
            path_rows=path_rows,
            lag_rows=lag_rows,
            physics_rows=physics_rows,
            ablation_rows=ablation_rows,
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
        return ChronarisContinuousAdapterSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            export_count=len(initial_results),
            resume_reused_count=export_manifest["resume_verification_reused_count"],
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _build_or_load_adapter(
    *, dataset_id, batch, fold, schema_context, heavy_root, registry, seed
):
    path = heavy_root / "checkpoints" / dataset_id / "chronaris.pt"
    if not path.exists():
        normalizer = TrainOnlyRobustNormalizer().fit(
            batch,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=fold.held_out_sample_ids,
        )
        config = ChronarisContinuousEncoderConfig(
            physiology_feature_names=schema_context.physiology_feature_names,
            vehicle_feature_names=schema_context.vehicle_feature_names,
            field_labels=schema_context.vehicle_field_labels,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            backbone = ChronarisContinuousFusionEncoder(config)
        save_chronaris_continuous_checkpoint(
            path,
            backbone=backbone,
            normalizer=normalizer,
            seed=seed,
        )
    checkpoint = build_checkpoint_record(
        method_name="chronaris",
        fold=fold,
        checkpoint_path=path,
        seed=seed,
    )
    registry.register(checkpoint)
    backbone, normalizer, _metadata = load_chronaris_continuous_checkpoint(path)
    return ChronarisContinuousFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id=fold.fold_id,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
    )


def _audit_ablations(
    *, dataset_id, full_config, normalizer, held_out_batch, seed
):
    normalized = normalizer.transform(held_out_batch)
    future_normalized = normalizer.transform(
        perturb_future_observations(held_out_batch, cutoff_s=15.0)
    )
    rows = []
    for config in build_chronaris_ablation_configs(full_config)[1:]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            backbone = ChronarisContinuousFusionEncoder(config).eval()
        started = time.perf_counter()
        with torch.inference_mode():
            encoding = backbone(normalized)
        elapsed = time.perf_counter() - started
        diff = validate_chronaris_ablation_diff(full_config, config)
        future_delta = None
        if config.variant == "no_causal_mask":
            with torch.inference_mode():
                future_encoding = backbone(future_normalized)
            past = normalized.query_timestamps_s <= 15.0
            future_delta = float(
                (
                    encoding.sequence_embedding[past]
                    - future_encoding.sequence_embedding[past]
                )
                .abs()
                .max()
                .item()
            )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": DATASET_LABELS[dataset_id],
                "variant": config.variant,
                "changed_fields": ",".join(sorted(diff)),
                "diff_valid": True,
                "finite_output": bool(torch.isfinite(encoding.sequence_embedding).all()),
                "forward_elapsed_s": elapsed,
                "continuous_evolution_enabled": config.continuous_evolution_enabled,
                "physics_active_count": encoding.physics_audit.active_component_count,
                "causal_mask_enabled": config.causal_mask_enabled,
                "lag_scale_count": len(config.lag_ranges_s),
                "scale_gate_enabled": config.scale_gate_enabled,
                "future_counterfactual_max_abs_delta": future_delta,
            }
        )
    return rows


def _parameter_row(*, dataset_id, adapter, batch, output, elapsed_s):
    return {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "input_feature_count": int(
            batch.physiology_values.shape[-1] + batch.vehicle_values.shape[-1]
        ),
        "parameter_count": adapter.parameter_count,
        "output_dim": int(output.sequence_embedding.shape[-1]),
        "query_point_count": int(output.sequence_embedding.shape[1]),
        "forward_elapsed_s": float(elapsed_s),
        "process_peak_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "ode_method": adapter.backbone.config.ode_method,
        "sequence_source": "task_head_free_continuous_fusion",
    }
