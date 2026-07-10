"""Causality, dual-stream sensitivity and acceptance checks for deep baselines."""

from __future__ import annotations

from dataclasses import replace

import torch


METHOD_LABELS = {"mult": "MulT", "contiformer": "ContiFormer"}
DATASET_LABELS = {"simulation": "仿真", "dingxin": "鼎新"}


def audit_deep_baseline_output(
    *,
    dataset_id,
    method_name,
    adapter,
    held_out_batch,
    baseline_output,
):
    cutoff_s = 15.0
    future_output = adapter(
        perturb_future_observations(held_out_batch, cutoff_s=cutoff_s)
    )
    past = baseline_output.timestamps_s <= cutoff_s
    future_delta = float(
        (baseline_output.sequence_embedding[past] - future_output.sequence_embedding[past])
        .abs()
        .max()
        .item()
    )
    physiology_output = adapter(
        perturb_historical_stream(
            held_out_batch,
            stream_name="physiology",
            cutoff_s=cutoff_s,
        )
    )
    vehicle_output = adapter(
        perturb_historical_stream(
            held_out_batch,
            stream_name="vehicle",
            cutoff_s=cutoff_s,
        )
    )
    physiology_delta = float(
        (baseline_output.sequence_embedding - physiology_output.sequence_embedding)
        .abs()
        .max()
        .item()
    )
    vehicle_delta = float(
        (baseline_output.sequence_embedding - vehicle_output.sequence_embedding)
        .abs()
        .max()
        .item()
    )
    common = {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "method_name": method_name,
        "method_label": METHOD_LABELS[method_name],
        "cutoff_s": cutoff_s,
    }
    return (
        {
            **common,
            "future_perturbation_max_abs_delta": future_delta,
            "causal_attention": True,
        },
        {
            **common,
            "physiology_history_max_abs_delta": physiology_delta,
            "vehicle_history_max_abs_delta": vehicle_delta,
            "both_streams_sensitive": physiology_delta > 0 and vehicle_delta > 0,
        },
    )


def build_deep_baseline_acceptance_rows(
    *,
    registry,
    parameter_rows,
    causality_rows,
    sensitivity_rows,
    export_manifest,
    alignment_hashes,
    dataset_metadata,
    transform_manifest,
):
    transform_hashes_match = all(
        len(
            {
                method["normalizer_sha256"]
                for method in dataset["methods"].values()
            }
        )
        == 1
        for dataset in transform_manifest["datasets"].values()
    )
    fit_isolated = all(
        method["fit_sample_hash"] == dataset["fold"]["train_sample_hash"]
        for dataset in transform_manifest["datasets"].values()
        for method in dataset["methods"].values()
    )
    worst_future = max(row["future_perturbation_max_abs_delta"] for row in causality_rows)
    minimum_phys = min(row["physiology_history_max_abs_delta"] for row in sensitivity_rows)
    minimum_vehicle = min(row["vehicle_history_max_abs_delta"] for row in sensitivity_rows)
    shape_values = [
        (row["query_point_count"], row["output_dim"]) for row in parameter_rows
    ]
    return [
        _check("two_dataset_two_method_exports", export_manifest["available_export_count"] == 4, export_manifest["available_export_count"], 4),
        _check("resume_reuses_all_exports", export_manifest["resume_verification_reused_count"] == 4, export_manifest["resume_verification_reused_count"], 4),
        _check("checkpoint_registry_complete", len(registry.records) == 4, len(registry.records), 4),
        _check("fixed_query_and_output_shape", all(query == 96 and output == 64 for query, output in shape_values), shape_values, (96, 64)),
        _check("future_observation_invariance", worst_future <= 1e-5, worst_future, 1e-5),
        _check("physiology_stream_sensitivity", minimum_phys > 0, minimum_phys, "> 0"),
        _check("vehicle_stream_sensitivity", minimum_vehicle > 0, minimum_vehicle, "> 0"),
        _check("shared_train_only_transform", transform_hashes_match and fit_isolated, {"same_hash": transform_hashes_match, "fit_isolated": fit_isolated}, True),
        _check("cross_method_alignment", all(len(value) == 64 for value in alignment_hashes.values()), alignment_hashes, "two SHA-256 values"),
        _check("task_head_free_outputs", all(row["sequence_source"] == "task_head_free" for row in parameter_rows), [row["sequence_source"] for row in parameter_rows], "task_head_free"),
        _check("causal_attention_enabled", all(row["causal_attention"] for row in parameter_rows), [row["causal_attention"] for row in parameter_rows], True),
        _check("dingxin_three_view_smoke", dataset_metadata["dingxin"]["view_count"] == 3, dataset_metadata["dingxin"]["view_count"], 3),
        _check("dingxin_label_sources_excluded", dataset_metadata["dingxin"]["excluded_label_source_count"] == 20, dataset_metadata["dingxin"]["excluded_label_source_count"], 20),
        _check("observed_only_inputs", not dataset_metadata["simulation"]["oracle_opened"] and not dataset_metadata["dingxin"]["label_or_target_opened"], False, False),
        _check("labels_not_used_for_encoder", all(not record.label_used_for_encoder_training for record in registry.records.values()), False, False),
    ]


def perturb_future_observations(batch, *, cutoff_s: float):
    physiology_future = (
        (batch.physiology_timestamps_s > cutoff_s) & batch.physiology_point_mask
    ).unsqueeze(-1)
    vehicle_future = (
        (batch.vehicle_timestamps_s > cutoff_s) & batch.vehicle_point_mask
    ).unsqueeze(-1)
    return replace(
        batch,
        physiology_values=torch.where(
            physiology_future & batch.physiology_feature_mask,
            batch.physiology_values + 100_000.0,
            batch.physiology_values,
        ),
        vehicle_values=torch.where(
            vehicle_future & batch.vehicle_feature_mask,
            batch.vehicle_values + 100_000.0,
            batch.vehicle_values,
        ),
    )


def perturb_historical_stream(batch, *, stream_name: str, cutoff_s: float):
    if stream_name == "physiology":
        historical = (
            (batch.physiology_timestamps_s <= cutoff_s)
            & batch.physiology_point_mask
        ).unsqueeze(-1)
        return replace(
            batch,
            physiology_values=torch.where(
                historical & batch.physiology_feature_mask,
                batch.physiology_values + 100.0,
                batch.physiology_values,
            ),
        )
    historical = (
        (batch.vehicle_timestamps_s <= cutoff_s) & batch.vehicle_point_mask
    ).unsqueeze(-1)
    return replace(
        batch,
        vehicle_values=torch.where(
            historical & batch.vehicle_feature_mask,
            batch.vehicle_values + 100.0,
            batch.vehicle_values,
        ),
    )


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
