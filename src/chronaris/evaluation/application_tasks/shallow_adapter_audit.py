"""Causality and acceptance audits for shallow production adapters."""

from __future__ import annotations

from dataclasses import replace

import torch


METHOD_LABELS = {
    "physiology_only": "生理单流",
    "vehicle_only": "航电单流",
    "naive_time_sync": "朴素时间同步",
}
DATASET_LABELS = {"simulation": "仿真", "dingxin": "鼎新"}


def causal_audit(
    *,
    dataset_id,
    method_name,
    adapter,
    held_out_batch,
    baseline_output,
):
    cutoff_s = 15.0
    future_batch = _perturb_future_values(held_out_batch, cutoff_s=cutoff_s)
    future_output = adapter(future_batch)
    past = baseline_output.timestamps_s <= cutoff_s
    future_delta = float(
        (baseline_output.sequence_embedding[past] - future_output.sequence_embedding[past])
        .abs()
        .max()
        .item()
    )
    inactive_delta = 0.0
    inactive_applicable = method_name in {"physiology_only", "vehicle_only"}
    if inactive_applicable:
        inactive_output = adapter(_perturb_inactive_stream(held_out_batch, method_name))
        inactive_delta = float(
            (baseline_output.sequence_embedding - inactive_output.sequence_embedding)
            .abs()
            .max()
            .item()
        )
    return {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "method_name": method_name,
        "method_label": METHOD_LABELS[method_name],
        "cutoff_s": cutoff_s,
        "future_perturbation_max_abs_delta": future_delta,
        "inactive_stream_check_applicable": inactive_applicable,
        "inactive_stream_max_abs_delta": inactive_delta,
        "query_valid_ratio": float(baseline_output.valid_mask.float().mean()),
    }


def build_shallow_acceptance_rows(
    *,
    registry,
    parameter_rows,
    causal_rows,
    export_manifest,
    alignment_hashes,
    dataset_metadata,
    transform_manifest,
):
    fit_isolated = all(
        method["fit_sample_hash"] == dataset["fold"]["train_sample_hash"]
        and method.get("pca_fit_sample_hash", dataset["fold"]["train_sample_hash"])
        == dataset["fold"]["train_sample_hash"]
        for dataset in transform_manifest["datasets"].values()
        for method in dataset["methods"].values()
    )
    shape_values = [
        (row["query_point_count"], row["output_dim"]) for row in parameter_rows
    ]
    parameter_values = [row["parameter_count"] for row in parameter_rows]
    naive_parameters = [
        row["parameter_count"]
        for row in parameter_rows
        if row["method_name"] == "naive_time_sync"
    ]
    worst_future = max(row["future_perturbation_max_abs_delta"] for row in causal_rows)
    worst_inactive = max(row["inactive_stream_max_abs_delta"] for row in causal_rows)
    return [
        _check(
            "two_dataset_three_method_exports",
            export_manifest["available_export_count"] == 6,
            export_manifest["available_export_count"],
            6,
        ),
        _check(
            "resume_reuses_all_exports",
            export_manifest["resume_verification_reused_count"] == 6,
            export_manifest["resume_verification_reused_count"],
            6,
        ),
        _check("checkpoint_registry_complete", len(registry.records) == 6, len(registry.records), 6),
        _check(
            "fixed_query_and_output_shape",
            all(query == 96 and output == 64 for query, output in shape_values),
            shape_values,
            (96, 64),
        ),
        _check("future_observation_invariance", worst_future <= 1e-6, worst_future, 1e-6),
        _check("inactive_stream_invariance", worst_inactive <= 1e-6, worst_inactive, 1e-6),
        _check("train_only_transform_lineage", fit_isolated, fit_isolated, True),
        _check(
            "cross_method_alignment",
            all(len(value) == 64 for value in alignment_hashes.values()),
            alignment_hashes,
            "two SHA-256 values",
        ),
        _check(
            "single_stream_shared_backbone",
            all(
                row["parameter_count"] > 0
                for row in parameter_rows
                if row["method_name"] != "naive_time_sync"
            ),
            parameter_values,
            "single streams >0, naive=0",
        ),
        _check(
            "naive_has_no_trainable_parameters",
            all(value == 0 for value in naive_parameters),
            naive_parameters,
            0,
        ),
        _check(
            "dingxin_three_view_smoke",
            dataset_metadata["dingxin"]["view_count"] == 3,
            dataset_metadata["dingxin"]["view_count"],
            3,
        ),
        _check(
            "dingxin_label_sources_excluded",
            dataset_metadata["dingxin"]["excluded_label_source_count"] == 20,
            dataset_metadata["dingxin"]["excluded_label_source_count"],
            20,
        ),
        _check(
            "observed_only_inputs",
            not dataset_metadata["simulation"]["oracle_opened"]
            and not dataset_metadata["dingxin"]["label_or_target_opened"],
            False,
            False,
        ),
        _check(
            "labels_not_used_for_encoder",
            all(
                not record.label_used_for_encoder_training
                for record in registry.records.values()
            ),
            False,
            False,
        ),
    ]


def _perturb_future_values(batch, *, cutoff_s: float):
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


def _perturb_inactive_stream(batch, method_name: str):
    if method_name == "physiology_only":
        return replace(batch, vehicle_values=batch.vehicle_values + 100_000.0)
    return replace(batch, physiology_values=batch.physiology_values + 100_000.0)


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
