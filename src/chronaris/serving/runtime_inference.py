"""Checkpoint-backed Stage I runtime inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


@dataclass(frozen=True, slots=True)
class StageIRuntimeInferenceConfig:
    """Configuration for one checkpoint-backed inference run."""

    run_id: str
    checkpoint_path: str
    artifact_root: str = "docs/artifacts/assets/stage_i_runtime_inference"
    report_root: str = "docs/artifacts/stage_i"
    device: str = "auto"
    export_predictions_csv: bool = True
    semantic_event_top_k: int = 4
    semantic_event_score_quantile: float = 0.75

    def __post_init__(self) -> None:
        if self.semantic_event_top_k <= 0:
            raise ValueError("semantic_event_top_k must be positive.")
        if not 0.0 < self.semantic_event_score_quantile <= 1.0:
            raise ValueError("semantic_event_score_quantile must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class StageIRuntimeInferenceRunResult:
    """Artifacts emitted by one runtime inference invocation."""

    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    predictions_csv_path: str | None
    summary: Mapping[str, object]


def run_stage_i_runtime_inference(
    config: StageIRuntimeInferenceConfig,
    *,
    samples: Sequence[E0ExperimentSample],
) -> StageIRuntimeInferenceRunResult:
    """Run multitask checkpoint inference over one sample batch."""

    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("runtime inference requires at least one sample.")

    checkpoint = torch.load(Path(config.checkpoint_path), map_location="cpu")
    normalization_stats = _deserialize_input_normalization_stats(
        checkpoint.get("input_normalization_stats")
    )
    preview_pipeline = _build_preview_pipeline(
        checkpoint=checkpoint,
        config=config,
        sample_count=len(sample_tuple),
        normalization_stats=normalization_stats,
    )
    _validate_feature_schema(sample_tuple, preview_pipeline=preview_pipeline, checkpoint=checkpoint)
    model = preview_pipeline._build_model(
        sample_tuple,
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

    sample_ids = tuple(sample.sample_id for sample in sample_tuple)
    task_batches = _build_inference_task_batches(
        sample_ids=sample_ids,
        task_metadata=task_metadata,
    )
    torch_batch = preview_pipeline._build_torch_batch(sample_tuple)
    torch_batch = preview_pipeline._apply_input_normalization(
        torch_batch,
        normalization_stats=normalization_stats,
    )
    reference_offsets_s = preview_pipeline._build_reference_offsets_s_tensor(sample_tuple)

    model.train(False)
    task_heads.train(False)
    with torch.no_grad():
        output = model(torch_batch, reference_offsets_s=reference_offsets_s)
        if (
            output.physiology.reference_projected_states is None
            or output.vehicle.reference_projected_states is None
            or output.physiology.reference_offsets_s is None
            or output.vehicle.reference_offsets_s is None
        ):
            raise RuntimeError("runtime inference requires reference-grid projections for both streams.")
        causal_output = causal_fusion(
            CausalFusionTensorInput(
                physiology_states=output.physiology.reference_projected_states,
                vehicle_states=output.vehicle.reference_projected_states,
                physiology_offsets_s=output.physiology.reference_offsets_s,
                vehicle_offsets_s=output.vehicle.reference_offsets_s,
            )
        )
        pooled = causal_output.fused_states.mean(dim=1)
        task_outputs = task_heads(pooled, task_batches)
        semantic_output = semantic_fusion(
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
        reverse_label_maps=reverse_label_maps,
    )
    semantic_summary = _summarize_semantic_output(
        sample_ids=sample_ids,
        semantic_output=semantic_output,
    )
    causal_rows = _build_causal_rows(sample_ids=sample_ids, causal_output=causal_output, reference_offsets_s=output.vehicle.reference_offsets_s)
    sample_rows = _merge_prediction_rows(
        sample_ids=sample_ids,
        task_predictions=task_predictions,
        causal_rows=causal_rows,
        semantic_summary=semantic_summary,
    )

    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "runtime_inference_summary.json"
    report_path = report_root / f"stage-i-runtime-inference-{config.run_id}.md"
    predictions_csv_path: str | None = None
    if config.export_predictions_csv:
        csv_path = run_root / "runtime_inference_predictions.csv"
        pd.DataFrame(sample_rows).to_csv(csv_path, index=False)
        predictions_csv_path = str(csv_path)

    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "checkpoint_path": str(Path(config.checkpoint_path)),
        "artifact_root": str(run_root),
        "sample_count": len(sample_tuple),
        "task_heads": {
            task_name: {
                "task_type": payload["task_type"],
                "label_to_id": dict(payload.get("label_to_id", {})),
            }
            for task_name, payload in task_metadata.items()
        },
        "predictions_csv_path": predictions_csv_path,
        "mean_attention_entropy": _mean(row["attention_entropy"] for row in causal_rows),
        "mean_top_event_score": _mean(row["top_event_score"] for row in causal_rows),
        "mean_top_contribution_score": _mean(row["top_contribution_score"] for row in causal_rows),
        "semantic_event": semantic_summary,
        "samples": sample_rows,
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
        summary=summary,
    )


def render_stage_i_runtime_inference_report(summary: Mapping[str, object]) -> str:
    """Render a compact runtime inference report."""

    lines = [
        f"# Stage I Runtime Inference - {summary['run_id']}",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- checkpoint_path: `{summary['checkpoint_path']}`",
        f"- sample_count: `{summary['sample_count']}`",
    ]
    if summary.get("predictions_csv_path"):
        lines.append(f"- predictions_csv_path: `{summary['predictions_csv_path']}`")
    lines.extend(
        [
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
            batch_size=max(sample_count, 1),
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
        similarity = torch.matmul(task_output.logits, task_output.logits.transpose(-1, -2))
        rows = []
        for row_index, sample_id in enumerate(task_output.sample_ids):
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
        row = {"sample_id": sample_id, **causal_by_sample[sample_id], **{
            "semantic_top_query_name": semantic_by_sample[sample_id]["top_query_name"],
            "semantic_top_event_attribution": semantic_by_sample[sample_id]["top_event_attribution"],
            "semantic_top_query_event_offset_s": semantic_by_sample[sample_id]["top_query_event_offset_s"],
        }}
        for task_name, payload in task_predictions.items():
            prediction_row = payload["rows"][sample_index]
            row[f"{task_name}_prediction"] = prediction_row.get("prediction")
            for key, value in prediction_row.items():
                if key == "prediction":
                    continue
                row[f"{task_name}_{key}"] = value
        rows.append(row)
    return rows


def _validate_feature_schema(
    samples: Sequence[E0ExperimentSample],
    *,
    preview_pipeline: AlignmentPreviewPipeline,
    checkpoint: Mapping[str, object],
) -> None:
    torch_batch = preview_pipeline._build_torch_batch(tuple(samples))
    feature_schema = dict(checkpoint.get("feature_schema") or {})
    expected_physiology = tuple(feature_schema.get("physiology_feature_names", ()))
    expected_vehicle = tuple(feature_schema.get("vehicle_feature_names", ()))
    if expected_physiology and torch_batch.physiology.feature_names != expected_physiology:
        raise ValueError("checkpoint physiology feature schema does not match runtime samples.")
    if expected_vehicle and torch_batch.vehicle.feature_names != expected_vehicle:
        raise ValueError("checkpoint vehicle feature schema does not match runtime samples.")


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


def _mean(values) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _fmt_optional_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"
