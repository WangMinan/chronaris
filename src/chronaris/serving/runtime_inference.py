"""Checkpoint-backed Stage I runtime inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping, Sequence

import pandas as pd
import torch

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.models.alignment import (
    AlignmentPrototypeConfig,
    ChronologicalSplitConfig,
    ReferenceGridConfig,
    StageITaskHeadBatch,
    StageITaskHeadSet,
    StageITaskHeadSpec,
)
from chronaris.models.fusion import (
    CausalEventFusion,
    CausalEventFusionConfig,
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalMaskedCrossModalFusion,
    SemanticEventTensorInput,
    semantic_query_entropy,
)
from chronaris.pipelines.alignment_preview import (
    AlignmentInputNormalizationStats,
    AlignmentPreviewConfig,
    AlignmentPreviewPipeline,
    StreamInputNormalizationStats,
)
from chronaris.schema.models import StreamKind

ReplayMode = Literal["batch", "incremental", "both"]


@dataclass(frozen=True, slots=True)
class StageIRuntimeInferenceConfig:
    """Configuration for one checkpoint-backed inference run."""

    run_id: str
    checkpoint_path: str
    artifact_root: str = "docs/artifacts/assets/stage_i_runtime_inference"
    report_root: str = "docs/artifacts/stage_i"
    device: str = "auto"
    export_predictions_csv: bool = True
    emit_predictions_jsonl: bool = False
    semantic_event_top_k: int = 4
    semantic_event_score_quantile: float = 0.75
    batch_size: int | None = None
    max_windows: int | None = None
    strict_feature_schema: bool = False
    replay_mode: ReplayMode = "batch"

    def __post_init__(self) -> None:
        if self.semantic_event_top_k <= 0:
            raise ValueError("semantic_event_top_k must be positive.")
        if not 0.0 < self.semantic_event_score_quantile <= 1.0:
            raise ValueError("semantic_event_score_quantile must be in (0, 1].")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided.")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive when provided.")
        if self.replay_mode not in {"batch", "incremental", "both"}:
            raise ValueError("replay_mode must be one of: batch, incremental, both.")


@dataclass(frozen=True, slots=True)
class StageIRuntimeInferenceRunResult:
    """Artifacts emitted by one runtime inference invocation."""

    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    predictions_csv_path: str | None
    predictions_jsonl_path: str | None
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _RuntimeSession:
    checkpoint: Mapping[str, object]
    preview_pipeline: AlignmentPreviewPipeline
    model: object
    task_heads: StageITaskHeadSet
    task_metadata: Mapping[str, object]
    reverse_label_maps: Mapping[str, Mapping[int, str]]
    causal_fusion: CausalMaskedCrossModalFusion
    semantic_fusion: CausalEventFusion
    normalization_stats: AlignmentInputNormalizationStats | None


def run_stage_i_runtime_inference(
    config: StageIRuntimeInferenceConfig,
    *,
    samples: Sequence[E0ExperimentSample],
) -> StageIRuntimeInferenceRunResult:
    """Run checkpoint-backed inference in batch, incremental, or both modes."""

    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("runtime inference requires at least one sample.")

    checkpoint = torch.load(Path(config.checkpoint_path), map_location="cpu")
    normalization_stats = _deserialize_input_normalization_stats(
        checkpoint.get("input_normalization_stats")
    )
    prepared_samples, schema_diagnostics = _prepare_runtime_samples(
        config=config,
        checkpoint=checkpoint,
        samples=sample_tuple,
    )
    session = _load_runtime_session(
        config=config,
        checkpoint=checkpoint,
        normalization_stats=normalization_stats,
        reference_samples=prepared_samples,
    )

    batch_summary = None
    incremental_summary = None
    if config.replay_mode in {"batch", "both"}:
        batch_summary = _run_replay_mode(
            session=session,
            config=config,
            samples=prepared_samples,
            replay_mode="batch",
            schema_diagnostics=schema_diagnostics,
        )
    if config.replay_mode in {"incremental", "both"}:
        incremental_summary = _run_replay_mode(
            session=session,
            config=config,
            samples=prepared_samples,
            replay_mode="incremental",
            schema_diagnostics=schema_diagnostics,
        )

    if config.replay_mode == "batch":
        summary = batch_summary
    elif config.replay_mode == "incremental":
        summary = incremental_summary
    else:
        assert batch_summary is not None and incremental_summary is not None
        summary = dict(batch_summary)
        summary["incremental_consistency"] = {
            "sample_count_match": batch_summary["sample_count"] == incremental_summary["sample_count"],
            "batch_sample_count": batch_summary["sample_count"],
            "incremental_sample_count": incremental_summary["sample_count"],
            "incremental_diagnostics": incremental_summary["diagnostics"],
        }

    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "runtime_inference_summary.json"
    report_path = report_root / f"stage-i-runtime-inference-{config.run_id}.md"
    predictions_csv_path: str | None = None
    predictions_jsonl_path: str | None = None
    if config.export_predictions_csv:
        csv_path = run_root / "runtime_inference_predictions.csv"
        pd.DataFrame(summary["samples"]).to_csv(csv_path, index=False)
        predictions_csv_path = str(csv_path)
    if config.emit_predictions_jsonl:
        jsonl_path = run_root / "runtime_inference_predictions.jsonl"
        _dump_prediction_rows_jsonl(summary["samples"], path=jsonl_path)
        predictions_jsonl_path = str(jsonl_path)

    summary = {
        **summary,
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "checkpoint_path": str(Path(config.checkpoint_path)),
        "artifact_root": str(run_root),
        "predictions_csv_path": predictions_csv_path,
        "predictions_jsonl_path": predictions_jsonl_path,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_stage_i_runtime_inference_report(summary) + "\n",
        encoding="utf-8",
    )
    return StageIRuntimeInferenceRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        predictions_csv_path=predictions_csv_path,
        predictions_jsonl_path=predictions_jsonl_path,
        summary=summary,
    )


def run_stage_i_runtime_inference_batch(
    config: StageIRuntimeInferenceConfig,
    *,
    samples: Sequence[E0ExperimentSample],
) -> StageIRuntimeInferenceRunResult:
    """Convenience wrapper for batch replay only."""

    return run_stage_i_runtime_inference(
        _replace_config(config, replay_mode="batch"),
        samples=samples,
    )


def run_stage_i_runtime_inference_incremental(
    config: StageIRuntimeInferenceConfig,
    *,
    samples: Sequence[E0ExperimentSample],
) -> StageIRuntimeInferenceRunResult:
    """Convenience wrapper for incremental replay only."""

    return run_stage_i_runtime_inference(
        _replace_config(config, replay_mode="incremental"),
        samples=samples,
    )


def render_stage_i_runtime_inference_report(summary: Mapping[str, object]) -> str:
    """Render a compact runtime inference report."""

    diagnostics = summary["diagnostics"]
    lines = [
        f"# Stage I Runtime Inference - {summary['run_id']}",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- checkpoint_path: `{summary['checkpoint_path']}`",
        f"- sample_count: `{summary['sample_count']}`",
        f"- replay_mode: `{summary['replay_mode']}`",
    ]
    if summary.get("predictions_csv_path"):
        lines.append(f"- predictions_csv_path: `{summary['predictions_csv_path']}`")
    if summary.get("predictions_jsonl_path"):
        lines.append(f"- predictions_jsonl_path: `{summary['predictions_jsonl_path']}`")
    lines.extend(
        [
            "",
            "## Replay Diagnostics",
            "",
            f"- feature_schema_status: `{diagnostics['feature_schema_status']}`",
            f"- feature_schema_source: `{diagnostics['feature_schema_source']}`",
            f"- truncated_by_max_windows: `{diagnostics['truncated_by_max_windows']}`",
            f"- batch_size: `{diagnostics['batch_size']}`",
            f"- chunk_count: `{diagnostics['chunk_count']}`",
            f"- latency_seconds: `{diagnostics['latency_seconds']:.6f}`",
            f"- throughput_samples_per_second: `{diagnostics['throughput_samples_per_second']:.6f}`",
            "",
            "## Batch Summary",
            "",
            f"- mean_attention_entropy: `{summary['mean_attention_entropy']:.6f}`",
            f"- mean_top_event_score: `{summary['mean_top_event_score']:.6f}`",
            f"- mean_top_contribution_score: `{summary['mean_top_contribution_score']:.6f}`",
            "",
            "## Semantic Event Summary",
            "",
            f"- query_names: `{summary['semantic_event']['query_names']}`",
            f"- mean_event_token_count: `{summary['semantic_event']['mean_event_token_count']:.6f}`",
            f"- mean_query_entropy: `{summary['semantic_event']['mean_query_entropy']:.6f}`",
            f"- mean_top_event_attribution: `{summary['semantic_event']['mean_top_event_attribution']:.6f}`",
        ]
    )
    if summary.get("incremental_consistency"):
        comparison = summary["incremental_consistency"]
        lines.extend(
            [
                "",
                "## Incremental Consistency",
                "",
                f"- sample_count_match: `{comparison['sample_count_match']}`",
                f"- batch_sample_count: `{comparison['batch_sample_count']}`",
                f"- incremental_sample_count: `{comparison['incremental_sample_count']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Sample Predictions",
            "",
            "| sample | risk_proxy | workload_proxy | event_replay_tag | top event offset s | top contribution offset s | top semantic query | top event attribution |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for row in summary["samples"]:
        lines.append(
            f"| `{row['sample_id']}` | `{row.get('risk_proxy_prediction', '-')}` | "
            f"{_fmt_optional_float(row.get('workload_proxy_prediction'))} | "
            f"`{row.get('event_replay_tag_prediction', '-')}` | "
            f"{row['top_event_offset_s']:.6f} | {row['top_contribution_offset_s']:.6f} | "
            f"`{row['semantic_top_query_name']}` | {row['semantic_top_event_attribution']:.6f} |"
        )
    return "\n".join(lines)


def dump_runtime_samples_jsonl(
    samples: Sequence[E0ExperimentSample],
    *,
    path: str | Path,
) -> None:
    """Serialize runtime samples for CLI replay."""

    Path(path).write_text(
        "".join(json.dumps(_serialize_sample(sample), ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )


def load_runtime_samples_jsonl(path: str | Path) -> tuple[E0ExperimentSample, ...]:
    """Load runtime samples serialized by `dump_runtime_samples_jsonl`."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(_deserialize_sample(row) for row in rows)


def _run_replay_mode(
    *,
    session: _RuntimeSession,
    config: StageIRuntimeInferenceConfig,
    samples: tuple[E0ExperimentSample, ...],
    replay_mode: Literal["batch", "incremental"],
    schema_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    batch_size = 1 if replay_mode == "incremental" else (config.batch_size or len(samples))
    chunk_results: list[dict[str, object]] = []
    start = perf_counter()
    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start : chunk_start + batch_size]
        chunk_results.append(
            _infer_sample_batch(
                session=session,
                samples=chunk,
                replay_mode=replay_mode,
            )
        )
    elapsed = perf_counter() - start
    sample_rows = [row for chunk in chunk_results for row in chunk["sample_rows"]]
    causal_rows = [row for chunk in chunk_results for row in chunk["causal_rows"]]
    semantic_rows = [row for chunk in chunk_results for row in chunk["semantic_summary"]["samples"]]
    query_names = []
    if chunk_results:
        query_names = list(chunk_results[0]["semantic_summary"]["query_names"])
    semantic_event = {
        "query_names": query_names,
        "query_count": len(query_names),
        "mean_event_token_count": _weighted_chunk_mean(chunk_results, "semantic_summary", "mean_event_token_count"),
        "mean_query_entropy": _weighted_chunk_mean(chunk_results, "semantic_summary", "mean_query_entropy"),
        "mean_top_event_attribution": _weighted_chunk_mean(chunk_results, "semantic_summary", "mean_top_event_attribution"),
        "samples": semantic_rows,
    }
    return {
        "sample_count": len(sample_rows),
        "replay_mode": replay_mode,
        "task_heads": {
            task_name: {
                "task_type": payload["task_type"],
                "label_to_id": dict(payload.get("label_to_id", {})),
            }
            for task_name, payload in session.task_metadata.items()
        },
        "mean_attention_entropy": _mean(row["attention_entropy"] for row in causal_rows),
        "mean_top_event_score": _mean(row["top_event_score"] for row in causal_rows),
        "mean_top_contribution_score": _mean(row["top_contribution_score"] for row in causal_rows),
        "semantic_event": semantic_event,
        "samples": sample_rows,
        "diagnostics": {
            **dict(schema_diagnostics),
            "batch_size": batch_size,
            "chunk_count": len(chunk_results),
            "latency_seconds": elapsed,
            "throughput_samples_per_second": (len(sample_rows) / elapsed) if elapsed > 0 else 0.0,
            "retrieval_context": "per_chunk_only",
        },
    }


def _infer_sample_batch(
    *,
    session: _RuntimeSession,
    samples: tuple[E0ExperimentSample, ...],
    replay_mode: Literal["batch", "incremental"],
) -> dict[str, object]:
    sample_ids = tuple(sample.sample_id for sample in samples)
    task_batches = _build_inference_task_batches(
        sample_ids=sample_ids,
        task_metadata=session.task_metadata,
    )
    torch_batch = session.preview_pipeline._build_torch_batch(samples)
    torch_batch = session.preview_pipeline._apply_input_normalization(
        torch_batch,
        normalization_stats=session.normalization_stats,
    )
    reference_offsets_s = session.preview_pipeline._build_reference_offsets_s_tensor(samples)

    session.model.train(False)
    session.task_heads.train(False)
    with torch.no_grad():
        output = session.model(torch_batch, reference_offsets_s=reference_offsets_s)
        if (
            output.physiology.reference_projected_states is None
            or output.vehicle.reference_projected_states is None
            or output.physiology.reference_offsets_s is None
            or output.vehicle.reference_offsets_s is None
        ):
            raise RuntimeError("runtime inference requires reference-grid projections for both streams.")
        causal_output = session.causal_fusion(
            CausalFusionTensorInput(
                physiology_states=output.physiology.reference_projected_states,
                vehicle_states=output.vehicle.reference_projected_states,
                physiology_offsets_s=output.physiology.reference_offsets_s,
                vehicle_offsets_s=output.vehicle.reference_offsets_s,
            )
        )
        pooled = causal_output.fused_states.mean(dim=1)
        task_outputs = session.task_heads(pooled, task_batches)
        semantic_output = session.semantic_fusion(
            SemanticEventTensorInput(
                physiology_states=output.physiology.reference_projected_states,
                vehicle_states=output.vehicle.reference_projected_states,
                attention_weights=causal_output.attention_weights,
                vehicle_event_scores=causal_output.vehicle_event_scores,
                vehicle_offsets_s=output.vehicle.reference_offsets_s,
            )
        )

    task_predictions = _summarize_task_predictions(
        task_outputs=task_outputs,
        reverse_label_maps=session.reverse_label_maps,
        replay_mode=replay_mode,
    )
    semantic_summary = _summarize_semantic_output(
        sample_ids=sample_ids,
        semantic_output=semantic_output,
    )
    causal_rows = _build_causal_rows(
        sample_ids=sample_ids,
        causal_output=causal_output,
        reference_offsets_s=output.vehicle.reference_offsets_s,
    )
    sample_rows = _merge_prediction_rows(
        sample_ids=sample_ids,
        task_predictions=task_predictions,
        causal_rows=causal_rows,
        semantic_summary=semantic_summary,
    )
    return {
        "sample_rows": sample_rows,
        "causal_rows": causal_rows,
        "semantic_summary": semantic_summary,
    }


def _load_runtime_session(
    *,
    config: StageIRuntimeInferenceConfig,
    checkpoint: Mapping[str, object],
    normalization_stats: AlignmentInputNormalizationStats | None,
    reference_samples: tuple[E0ExperimentSample, ...],
) -> _RuntimeSession:
    preview_pipeline = _build_preview_pipeline(
        checkpoint=checkpoint,
        config=config,
        sample_count=len(reference_samples),
        normalization_stats=normalization_stats,
    )
    model = preview_pipeline._build_model(
        reference_samples,
        prototype_config=AlignmentPrototypeConfig(**dict(checkpoint["preview_config"]["prototype_config"])),
    )
    model.load_state_dict(dict(checkpoint["model_state_dict"]))
    model.to(device=preview_pipeline._resolved_device_name(), dtype=preview_pipeline.config.dtype)

    task_specs = [
        StageITaskHeadSpec(**dict(payload["spec"]))
        for payload in checkpoint.get("task_heads", {}).values()
    ]
    task_heads = StageITaskHeadSet(task_specs).to(
        device=preview_pipeline._resolved_device_name(),
        dtype=preview_pipeline.config.dtype,
    )
    task_heads.load_state_dict(dict(checkpoint["task_head_state_dict"]))
    task_metadata = dict(checkpoint.get("task_heads") or {})
    reverse_label_maps = {
        task_name: {int(value): label for label, value in dict(payload.get("label_to_id", {})).items()}
        for task_name, payload in task_metadata.items()
    }
    causal_fusion = CausalMaskedCrossModalFusion(
        CausalFusionConfig(
            attention_temperature=float(checkpoint["multitask_config"]["causal_attention_temperature"]),
            event_bias_weight=float(checkpoint["multitask_config"]["causal_event_bias_weight"]),
            use_causal_mask=True,
            lag_window_points=checkpoint["multitask_config"].get("causal_lag_window_points"),
        )
    ).to(device=preview_pipeline._resolved_device_name())
    semantic_fusion = CausalEventFusion(
        CausalEventFusionConfig(
            attention_temperature=float(checkpoint["multitask_config"]["causal_attention_temperature"]),
            event_score_bias_weight=float(checkpoint["multitask_config"]["causal_event_bias_weight"]),
            event_top_k=config.semantic_event_top_k,
            event_score_quantile=config.semantic_event_score_quantile,
        )
    ).to(device=preview_pipeline._resolved_device_name())
    return _RuntimeSession(
        checkpoint=checkpoint,
        preview_pipeline=preview_pipeline,
        model=model,
        task_heads=task_heads,
        task_metadata=task_metadata,
        reverse_label_maps=reverse_label_maps,
        causal_fusion=causal_fusion,
        semantic_fusion=semantic_fusion,
        normalization_stats=normalization_stats,
    )


def _prepare_runtime_samples(
    *,
    config: StageIRuntimeInferenceConfig,
    checkpoint: Mapping[str, object],
    samples: tuple[E0ExperimentSample, ...],
) -> tuple[tuple[E0ExperimentSample, ...], dict[str, object]]:
    prepared = samples[: config.max_windows] if config.max_windows is not None else samples
    feature_schema = dict(checkpoint.get("feature_schema") or {})
    expected_physiology = tuple(feature_schema.get("physiology_feature_names", ()))
    expected_vehicle = tuple(feature_schema.get("vehicle_feature_names", ()))
    schema_source = "feature_schema"
    if not expected_physiology or not expected_vehicle:
        normalization_stats = _deserialize_input_normalization_stats(
            checkpoint.get("input_normalization_stats")
        )
        if normalization_stats is not None:
            expected_physiology = expected_physiology or tuple(normalization_stats.physiology.feature_names)
            expected_vehicle = expected_vehicle or tuple(normalization_stats.vehicle.feature_names)
            schema_source = "input_normalization_stats"
    extra_physiology: set[str] = set()
    missing_physiology: set[str] = set()
    extra_vehicle: set[str] = set()
    missing_vehicle: set[str] = set()
    aligned_samples: list[E0ExperimentSample] = []
    schema_status = "exact"
    for sample in prepared:
        physiology, physiology_diag = _align_stream_to_schema(
            sample.physiology,
            expected_feature_names=expected_physiology,
        )
        vehicle, vehicle_diag = _align_stream_to_schema(
            sample.vehicle,
            expected_feature_names=expected_vehicle,
        )
        extra_physiology.update(physiology_diag["extra_features"])
        missing_physiology.update(physiology_diag["missing_features"])
        extra_vehicle.update(vehicle_diag["extra_features"])
        missing_vehicle.update(vehicle_diag["missing_features"])
        aligned_samples.append(
            E0ExperimentSample(
                sample_id=sample.sample_id,
                sortie_id=sample.sortie_id,
                start_offset_ms=sample.start_offset_ms,
                end_offset_ms=sample.end_offset_ms,
                physiology=physiology,
                vehicle=vehicle,
                notes=sample.notes,
            )
        )
    if extra_physiology or missing_physiology or extra_vehicle or missing_vehicle:
        if config.strict_feature_schema:
            raise ValueError(
                "runtime feature schema mismatch: "
                f"missing physiology={sorted(missing_physiology)}, extra physiology={sorted(extra_physiology)}, "
                f"missing vehicle={sorted(missing_vehicle)}, extra vehicle={sorted(extra_vehicle)}"
            )
        schema_status = "aligned"
    return (
        tuple(aligned_samples),
        {
            "requested_sample_count": len(samples),
            "truncated_by_max_windows": config.max_windows is not None and len(prepared) < len(samples),
            "feature_schema_status": schema_status,
            "feature_schema_source": schema_source if (expected_physiology or expected_vehicle) else "input_samples",
            "missing_physiology_features": sorted(missing_physiology),
            "extra_physiology_features": sorted(extra_physiology),
            "missing_vehicle_features": sorted(missing_vehicle),
            "extra_vehicle_features": sorted(extra_vehicle),
            "strict_feature_schema": config.strict_feature_schema,
        },
    )


def _align_stream_to_schema(
    stream: NumericStreamMatrix,
    *,
    expected_feature_names: tuple[str, ...],
) -> tuple[NumericStreamMatrix, dict[str, object]]:
    if not expected_feature_names:
        return stream, {"missing_features": (), "extra_features": ()}
    current_index = {name: index for index, name in enumerate(stream.feature_names)}
    missing = tuple(name for name in expected_feature_names if name not in current_index)
    extra = tuple(name for name in stream.feature_names if name not in set(expected_feature_names))
    if not missing and not extra and stream.feature_names == expected_feature_names:
        return stream, {"missing_features": (), "extra_features": ()}
    aligned_rows = []
    for row in stream.values:
        aligned_row = []
        for feature_name in expected_feature_names:
            feature_index = current_index.get(feature_name)
            aligned_row.append(float("nan") if feature_index is None else row[feature_index])
        aligned_rows.append(tuple(aligned_row))
    return (
        NumericStreamMatrix(
            stream_kind=stream.stream_kind,
            point_count=stream.point_count,
            feature_names=expected_feature_names,
            point_offsets_ms=stream.point_offsets_ms,
            point_measurements=stream.point_measurements,
            values=tuple(aligned_rows),
            dropped_fields=stream.dropped_fields,
        ),
        {"missing_features": missing, "extra_features": extra},
    )


def _build_preview_pipeline(
    *,
    checkpoint: Mapping[str, object],
    config: StageIRuntimeInferenceConfig,
    sample_count: int,
    normalization_stats: AlignmentInputNormalizationStats | None,
) -> AlignmentPreviewPipeline:
    preview_config = checkpoint.get("preview_config") or {}
    training_config = preview_config.get("training") or {}
    return AlignmentPreviewPipeline(
        config=AlignmentPreviewConfig(
            prototype_config=AlignmentPrototypeConfig(**dict(preview_config["prototype_config"])),
            split_config=ChronologicalSplitConfig(**dict(preview_config["split_config"])),
            reference_grid_config=ReferenceGridConfig(**dict(preview_config["reference_grid_config"])),
            epoch_count=1,
            batch_size=max(config.batch_size or sample_count, 1),
            learning_rate=float(training_config.get("learning_rate", 1e-3)),
            device=config.device,
            input_normalization_mode="zscore_train" if normalization_stats is not None else "none",
            enable_physics_constraints=bool(training_config.get("enable_physics_constraints", False)),
            physics_constraint_family=str(training_config.get("physics_constraint_family", "minimal")),
            export_intermediate_states=False,
            intermediate_partition="all",
        )
    )


def _build_inference_task_batches(
    *,
    sample_ids: tuple[str, ...],
    task_metadata: Mapping[str, object],
) -> tuple[StageITaskHeadBatch, ...]:
    indices = tuple(range(len(sample_ids)))
    return tuple(
        StageITaskHeadBatch(
            task_name=str(task_name),
            task_type=str(payload["task_type"]),
            sample_ids=sample_ids,
            sample_indices=indices,
            paired_sample_ids=tuple(None for _ in sample_ids),
        )
        for task_name, payload in task_metadata.items()
    )


def _summarize_task_predictions(
    *,
    task_outputs,
    reverse_label_maps: Mapping[str, Mapping[int, str]],
    replay_mode: Literal["batch", "incremental"],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for task_output in task_outputs:
        if task_output.task_type == "classification":
            probabilities = torch.softmax(task_output.logits, dim=-1)
            predicted_ids = probabilities.argmax(dim=-1)
            reverse_map = reverse_label_maps.get(task_output.task_name, {})
            rows = []
            for row_index in range(probabilities.shape[0]):
                label_id = int(predicted_ids[row_index].detach().cpu())
                rows.append(
                    {
                        "prediction": reverse_map.get(label_id, str(label_id)),
                        "confidence": float(probabilities[row_index, label_id].detach().cpu()),
                    }
                )
            summary[task_output.task_name] = {
                "task_type": task_output.task_type,
                "rows": rows,
            }
            continue
        if task_output.task_type == "regression":
            summary[task_output.task_name] = {
                "task_type": task_output.task_type,
                "rows": [
                    {"prediction": float(value)}
                    for value in task_output.logits.reshape(-1).detach().cpu().tolist()
                ],
            }
            continue
        if replay_mode == "incremental" and len(task_output.sample_ids) <= 1:
            summary[task_output.task_name] = {
                "task_type": task_output.task_type,
                "rows": [{"prediction": None, "score": None} for _ in task_output.sample_ids],
            }
            continue
        similarity = torch.matmul(task_output.logits, task_output.logits.transpose(-1, -2))
        rows = []
        for row_index, _sample_id in enumerate(task_output.sample_ids):
            masked_similarity = similarity[row_index].clone()
            masked_similarity[row_index] = torch.finfo(masked_similarity.dtype).min
            if masked_similarity.numel() <= 1:
                rows.append({"prediction": None, "score": None})
                continue
            best_index = int(torch.argmax(masked_similarity).detach().cpu())
            rows.append(
                {
                    "prediction": str(task_output.sample_ids[best_index]),
                    "score": float(masked_similarity[best_index].detach().cpu()),
                }
            )
        summary[task_output.task_name] = {
            "task_type": task_output.task_type,
            "rows": rows,
        }
    return summary


def _build_causal_rows(
    *,
    sample_ids: tuple[str, ...],
    causal_output,
    reference_offsets_s: torch.Tensor,
) -> list[dict[str, float | str]]:
    contribution_scores = causal_output.attention_weights.sum(dim=1) * causal_output.vehicle_event_scores
    entropy = -(causal_output.attention_weights * torch.log(torch.clamp(causal_output.attention_weights, min=1e-12))).sum(dim=-1)
    rows: list[dict[str, float | str]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        top_event_index = int(torch.argmax(causal_output.vehicle_event_scores[sample_index]).detach().cpu())
        top_contribution_index = int(torch.argmax(contribution_scores[sample_index]).detach().cpu())
        rows.append(
            {
                "sample_id": sample_id,
                "attention_entropy": float(entropy[sample_index].mean().detach().cpu()),
                "top_event_offset_s": float(reference_offsets_s[sample_index, top_event_index].detach().cpu()),
                "top_event_score": float(causal_output.vehicle_event_scores[sample_index, top_event_index].detach().cpu()),
                "top_contribution_offset_s": float(reference_offsets_s[sample_index, top_contribution_index].detach().cpu()),
                "top_contribution_score": float(contribution_scores[sample_index, top_contribution_index].detach().cpu()),
            }
        )
    return rows


def _summarize_semantic_output(
    *,
    sample_ids: tuple[str, ...],
    semantic_output,
) -> dict[str, object]:
    query_entropy = semantic_query_entropy(
        semantic_output.query_to_event_attention,
        semantic_output.event_token_mask,
    )
    token_count = semantic_output.event_token_mask.sum(dim=-1)
    samples = []
    top_event_attributions = []
    for sample_index, sample_id in enumerate(sample_ids):
        top_query_index = int(torch.argmax(semantic_output.query_attribution_scores[sample_index]).detach().cpu())
        query_attention = semantic_output.query_to_event_attention[sample_index, top_query_index]
        top_event_index = int(torch.argmax(query_attention).detach().cpu())
        top_event_attribution = float(
            semantic_output.event_attribution_scores[sample_index, top_event_index].detach().cpu()
        )
        top_event_attributions.append(top_event_attribution)
        samples.append(
            {
                "sample_id": sample_id,
                "event_token_count": int(token_count[sample_index].detach().cpu()),
                "top_query_name": semantic_output.query_names[top_query_index],
                "top_query_score": float(
                    semantic_output.query_attribution_scores[sample_index, top_query_index].detach().cpu()
                ),
                "top_query_event_offset_s": float(
                    semantic_output.event_token_center_offsets_s[sample_index, top_event_index].detach().cpu()
                ),
                "top_event_attribution": top_event_attributions[-1],
            }
        )
    return {
        "query_names": list(semantic_output.query_names),
        "query_count": len(semantic_output.query_names),
        "mean_event_token_count": float(token_count.to(dtype=torch.float32).mean().detach().cpu()),
        "mean_query_entropy": float(query_entropy.mean().detach().cpu()),
        "mean_top_query_score": _mean(row["top_query_score"] for row in samples),
        "mean_top_event_attribution": _mean(top_event_attributions),
        "samples": samples,
    }


def _merge_prediction_rows(
    *,
    sample_ids: tuple[str, ...],
    task_predictions: Mapping[str, Mapping[str, object]],
    causal_rows: Sequence[Mapping[str, object]],
    semantic_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    causal_by_sample = {str(row["sample_id"]): dict(row) for row in causal_rows}
    semantic_by_sample = {
        str(row["sample_id"]): dict(row)
        for row in semantic_summary.get("samples", [])
    }
    rows: list[dict[str, object]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        row = {
            "sample_id": sample_id,
            **causal_by_sample[sample_id],
            **{
                "semantic_top_query_name": semantic_by_sample[sample_id]["top_query_name"],
                "semantic_top_event_attribution": semantic_by_sample[sample_id]["top_event_attribution"],
                "semantic_top_query_event_offset_s": semantic_by_sample[sample_id]["top_query_event_offset_s"],
            },
        }
        for task_name, payload in task_predictions.items():
            prediction_row = payload["rows"][sample_index]
            row[f"{task_name}_prediction"] = prediction_row.get("prediction")
            for key, value in prediction_row.items():
                if key == "prediction":
                    continue
                row[f"{task_name}_{key}"] = value
        rows.append(row)
    return rows


def _deserialize_input_normalization_stats(
    payload: object,
) -> AlignmentInputNormalizationStats | None:
    if not isinstance(payload, Mapping):
        return None
    physiology = payload.get("physiology") or {}
    vehicle = payload.get("vehicle") or {}
    physiology_mean = physiology.get("mean")
    physiology_std = physiology.get("std")
    vehicle_mean = vehicle.get("mean")
    vehicle_std = vehicle.get("std")
    if physiology_mean is None or physiology_std is None or vehicle_mean is None or vehicle_std is None:
        return None
    return AlignmentInputNormalizationStats(
        mode=str(payload.get("mode") or "none"),
        physiology=StreamInputNormalizationStats(
            feature_names=tuple(physiology.get("feature_names", ())),
            mean=torch.as_tensor(physiology_mean),
            std=torch.as_tensor(physiology_std),
        ),
        vehicle=StreamInputNormalizationStats(
            feature_names=tuple(vehicle.get("feature_names", ())),
            mean=torch.as_tensor(vehicle_mean),
            std=torch.as_tensor(vehicle_std),
        ),
    )


def dump_runtime_predictions_jsonl(
    rows: Sequence[Mapping[str, object]],
    *,
    path: str | Path,
) -> None:
    """Write prediction rows as line-delimited JSON."""

    _dump_prediction_rows_jsonl(rows, path=path)


def _dump_prediction_rows_jsonl(
    rows: Sequence[Mapping[str, object]],
    *,
    path: str | Path,
) -> None:
    Path(path).write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _serialize_sample(sample: E0ExperimentSample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "sortie_id": sample.sortie_id,
        "start_offset_ms": sample.start_offset_ms,
        "end_offset_ms": sample.end_offset_ms,
        "physiology": _serialize_stream(sample.physiology),
        "vehicle": _serialize_stream(sample.vehicle),
        "notes": list(sample.notes),
    }


def _serialize_stream(stream: NumericStreamMatrix) -> dict[str, object]:
    return {
        "stream_kind": stream.stream_kind.value,
        "point_count": stream.point_count,
        "feature_names": list(stream.feature_names),
        "point_offsets_ms": list(stream.point_offsets_ms),
        "point_measurements": list(stream.point_measurements),
        "values": [list(row) for row in stream.values],
        "dropped_fields": list(stream.dropped_fields),
    }


def _deserialize_sample(payload: Mapping[str, object]) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=str(payload["sample_id"]),
        sortie_id=str(payload["sortie_id"]),
        start_offset_ms=int(payload["start_offset_ms"]),
        end_offset_ms=int(payload["end_offset_ms"]),
        physiology=_deserialize_stream(payload["physiology"]),
        vehicle=_deserialize_stream(payload["vehicle"]),
        notes=tuple(str(value) for value in payload.get("notes", [])),
    )


def _deserialize_stream(payload: Mapping[str, object]) -> NumericStreamMatrix:
    return NumericStreamMatrix(
        stream_kind=StreamKind(str(payload["stream_kind"])),
        point_count=int(payload["point_count"]),
        feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
        point_offsets_ms=tuple(int(value) for value in payload.get("point_offsets_ms", [])),
        point_measurements=tuple(str(value) for value in payload.get("point_measurements", [])),
        values=tuple(tuple(float(value) for value in row) for row in payload.get("values", [])),
        dropped_fields=tuple(str(value) for value in payload.get("dropped_fields", [])),
    )


def _weighted_chunk_mean(
    chunk_results: Sequence[Mapping[str, object]],
    section_key: str,
    metric_key: str,
) -> float:
    weighted_total = 0.0
    sample_count = 0
    for chunk in chunk_results:
        section = chunk[section_key]
        current_count = len(chunk["sample_rows"])
        weighted_total += float(section[metric_key]) * current_count
        sample_count += current_count
    if sample_count <= 0:
        return 0.0
    return weighted_total / sample_count


def _replace_config(
    config: StageIRuntimeInferenceConfig,
    *,
    replay_mode: ReplayMode,
) -> StageIRuntimeInferenceConfig:
    return StageIRuntimeInferenceConfig(
        run_id=config.run_id,
        checkpoint_path=config.checkpoint_path,
        artifact_root=config.artifact_root,
        report_root=config.report_root,
        device=config.device,
        export_predictions_csv=config.export_predictions_csv,
        emit_predictions_jsonl=config.emit_predictions_jsonl,
        semantic_event_top_k=config.semantic_event_top_k,
        semantic_event_score_quantile=config.semantic_event_score_quantile,
        batch_size=config.batch_size,
        max_windows=config.max_windows,
        strict_feature_schema=config.strict_feature_schema,
        replay_mode=replay_mode,
    )


def _mean(values) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _fmt_optional_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"
