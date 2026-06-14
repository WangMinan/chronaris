"""Real-sortie case-study helpers for Stage I deep baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation import save_bar_plot
from chronaris.features.stage_i_sequences import STAGE_H_CASE_DATASET_ID
from chronaris.pipelines.stage_i.public.deep_baseline_runtime import (
    _deep_model_config_dict,
    _forward_dataset,
    _normalize_modalities,
    _train_model,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name

if TYPE_CHECKING:
    from chronaris.pipelines.stage_i.public.deep_baseline import StageIDeepBaselineConfig


def _run_real_sortie_case_study(
    *,
    dataset: Mapping[str, object],
    config: "StageIDeepBaselineConfig",
) -> tuple[dict[str, object], pd.DataFrame]:
    bundle = dataset["bundle"]
    entries = dataset["entries"]
    artifact_root = Path(config.artifact_root)
    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    ordered_modalities = tuple(entries[0].modality_schema)
    normalized_arrays = _normalize_modalities(
        modality_arrays=bundle.modality_arrays,
        modality_masks=bundle.modality_masks,
        ordered_modalities=ordered_modalities,
        train_indices=np.arange(bundle.entry_count, dtype=int),
    )
    labels = bundle.objective_label_values.astype(int)
    model = _train_model(
        model_name=config.model_name,
        ordered_modalities=ordered_modalities,
        modality_arrays=normalized_arrays,
        modality_masks=bundle.modality_masks,
        time_axis=bundle.time_axis,
        train_indices=np.arange(bundle.entry_count, dtype=int),
        train_targets=labels,
        output_dim=max(int(labels.max()) + 1, 2),
        task="classification",
        config=config,
    )
    base_output = _forward_dataset(
        model=model,
        ordered_modalities=ordered_modalities,
        modality_arrays=normalized_arrays,
        modality_masks=bundle.modality_masks,
        time_axis=bundle.time_axis,
        indices=np.arange(bundle.entry_count, dtype=int),
    )
    perturbed_vehicle_values, perturbed_vehicle_masks = _mask_top_event_steps(bundle)
    perturbed_arrays = dict(normalized_arrays)
    perturbed_arrays["vehicle"] = perturbed_vehicle_values
    perturbed_masks = dict(bundle.modality_masks)
    perturbed_masks["vehicle"] = perturbed_vehicle_masks
    perturbed_output = _forward_dataset(
        model=model,
        ordered_modalities=ordered_modalities,
        modality_arrays=perturbed_arrays,
        modality_masks=perturbed_masks,
        time_axis=bundle.time_axis,
        indices=np.arange(bundle.entry_count, dtype=int),
    )
    sample_frame = _build_real_sortie_sample_frame(
        entries=entries,
        bundle=bundle,
        base_output=base_output,
        perturbed_output=perturbed_output,
    )
    view_summary = _build_real_sortie_view_summary(sample_frame)
    pilot_summary = _build_real_sortie_pilot_summary(view_summary)
    plot_paths = {
        "view_representation_stability": save_bar_plot(
            dict(
                zip(
                    view_summary["view_id"],
                    view_summary["representation_stability"],
                    strict=True,
                ),
            ),
            path=plot_root / "view_representation_stability.png",
            title=f"{config.model_name} real-sortie stability",
            ylabel="cosine",
        ),
        "event_mask_interference": save_bar_plot(
            dict(
                zip(
                    view_summary["view_id"],
                    view_summary["event_mask_interference"],
                    strict=True,
                ),
            ),
            path=plot_root / "event_mask_interference.png",
            title=f"{config.model_name} event-mask interference",
            ylabel="1-cosine",
        ),
    }
    summary = {
        "dataset_id": STAGE_H_CASE_DATASET_ID,
        "profile": config.profile,
        "model_name": config.model_name,
        "model_config": _deep_model_config_dict(config),
        "runtime_device": resolve_torch_device_name(config.device),
        "artifact_root": str(artifact_root),
        "prepared_artifact_root": config.prepared_artifact_root,
        "smoke_training_target": "projection_diagnostics_verdict_code",
        "view_count": int(view_summary.shape[0]),
        "sample_count": int(sample_frame.shape[0]),
        "view_summary_csv": str(artifact_root / "view_summary.csv"),
        "sample_summary_csv": str(artifact_root / "sample_summary.csv"),
        "pilot_summary_csv": str(artifact_root / "pilot_summary.csv"),
        "plot_paths": plot_paths,
        "view_metrics": view_summary.to_dict(orient="records"),
        "pilot_metrics": pilot_summary.to_dict(orient="records"),
        "verdict_counts": (
            sample_frame["projection_diagnostics_verdict"].value_counts().to_dict()
        ),
    }
    sample_frame.to_csv(artifact_root / "sample_summary.csv", index=False)
    view_summary.to_csv(artifact_root / "view_summary.csv", index=False)
    pilot_summary.to_csv(artifact_root / "pilot_summary.csv", index=False)
    return summary, sample_frame

def _build_real_sortie_sample_frame(
    *,
    entries: Sequence[object],
    bundle,
    base_output,
    perturbed_output,
) -> pd.DataFrame:
    attention_entropy = _attention_entropy(base_output.attention_map)
    top_concentration = base_output.attention_map.max(dim=-1).values.mean(dim=1).cpu().numpy()
    cosine_shift = 1.0 - _cosine_similarity(
        base_output.pooled_embedding,
        perturbed_output.pooled_embedding,
    ).cpu().numpy()
    rows: list[dict[str, object]] = []
    metadata_by_sample = {
        entry.sample_id: json.loads(metadata)
        for entry, metadata in zip(entries, bundle.metadata_json, strict=True)
    }
    for index, entry in enumerate(entries):
        metadata = metadata_by_sample[entry.sample_id]
        rows.append(
            {
                "sample_id": entry.sample_id,
                "view_id": metadata["view_id"],
                "sortie_id": metadata["sortie_id"],
                "pilot_id": metadata["pilot_id"],
                "projection_diagnostics_verdict": metadata[
                    "projection_diagnostics_verdict"
                ],
                "embedding_norm": float(
                    torch.linalg.norm(base_output.pooled_embedding[index]).item(),
                ),
                "attention_entropy": float(attention_entropy[index]),
                "top_event_concentration": float(top_concentration[index]),
                "event_mask_interference": float(cosine_shift[index]),
            },
        )
    return pd.DataFrame(rows)


def _build_real_sortie_view_summary(sample_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for view_id, frame in sample_frame.groupby("view_id", sort=False):
        norms = frame["embedding_norm"].to_numpy(dtype=float)
        stability = float(np.mean(norms / max(np.max(norms), 1e-6)))
        records.append(
            {
                "view_id": view_id,
                "sortie_id": frame["sortie_id"].iloc[0],
                "pilot_id": int(frame["pilot_id"].iloc[0]),
                "projection_diagnostics_verdict": frame[
                    "projection_diagnostics_verdict"
                ].iloc[0],
                "sample_count": int(len(frame)),
                "representation_stability": stability,
                "mean_attention_entropy": float(frame["attention_entropy"].mean()),
                "top_event_concentration": float(
                    frame["top_event_concentration"].mean(),
                ),
                "event_mask_interference": float(
                    frame["event_mask_interference"].mean(),
                ),
            },
        )
    return pd.DataFrame(records)


def _build_real_sortie_pilot_summary(view_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sortie_id, frame in view_summary.groupby("sortie_id", sort=False):
        if len(frame) < 2:
            continue
        ordered = frame.sort_values("pilot_id").reset_index(drop=True)
        reference = ordered.iloc[0]
        comparison = ordered.iloc[1]
        rows.append(
            {
                "sortie_id": sortie_id,
                "reference_pilot_id": int(reference["pilot_id"]),
                "comparison_pilot_id": int(comparison["pilot_id"]),
                "delta_representation_stability": float(
                    comparison["representation_stability"]
                    - reference["representation_stability"]
                ),
                "delta_attention_entropy": float(
                    comparison["mean_attention_entropy"]
                    - reference["mean_attention_entropy"]
                ),
                "delta_top_event_concentration": float(
                    comparison["top_event_concentration"]
                    - reference["top_event_concentration"]
                ),
                "delta_event_mask_interference": float(
                    comparison["event_mask_interference"]
                    - reference["event_mask_interference"]
                ),
            },
        )
    return pd.DataFrame(rows)


def _mask_top_event_steps(bundle) -> tuple[np.ndarray, np.ndarray]:
    vehicle_values = bundle.modality_arrays["vehicle"].copy()
    vehicle_masks = bundle.modality_masks["vehicle"].copy()
    event_scores = bundle.extras["vehicle_event_scores"].astype(np.float32)
    for sample_index in range(event_scores.shape[0]):
        threshold = float(np.quantile(event_scores[sample_index], 0.75))
        active = event_scores[sample_index] >= threshold
        vehicle_values[sample_index, active, :] = 0.0
        vehicle_masks[sample_index, active] = 0
    return vehicle_values, vehicle_masks


def _attention_entropy(attention_map: torch.Tensor) -> np.ndarray:
    probabilities = attention_map.clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=-1)
    return entropy.detach().cpu().numpy()


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    right_norm = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (left_norm * right_norm).sum(dim=-1)
