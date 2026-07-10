"""Mechanism and acceptance audits for the Chronaris continuous adapter."""

from __future__ import annotations

import torch

from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import (
    perturb_future_observations,
    perturb_historical_stream,
)
from chronaris.modeling.fusion_encoders import (
    build_seconds_lag_mask,
    physics_audit_to_rows,
)


DATASET_LABELS = {"simulation": "仿真", "dingxin": "鼎新"}


def audit_chronaris_output(
    *,
    dataset_id,
    adapter,
    held_out_batch,
    baseline_output,
    baseline_encoding,
):
    cutoff_s = 15.0
    future = adapter(
        perturb_future_observations(held_out_batch, cutoff_s=cutoff_s)
    )
    past = baseline_output.timestamps_s <= cutoff_s
    future_delta = _maximum_delta(baseline_output, future, mask=past)
    physiology = adapter(
        perturb_historical_stream(
            held_out_batch,
            stream_name="physiology",
            cutoff_s=cutoff_s,
        )
    )
    vehicle = adapter(
        perturb_historical_stream(
            held_out_batch,
            stream_name="vehicle",
            cutoff_s=cutoff_s,
        )
    )
    causality = {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "cutoff_s": cutoff_s,
        "future_perturbation_max_abs_delta": future_delta,
        "causal_attention": True,
    }
    sensitivity = {
        "dataset_id": dataset_id,
        "dataset_label": DATASET_LABELS[dataset_id],
        "physiology_history_max_abs_delta": _maximum_delta(
            baseline_output,
            physiology,
        ),
        "vehicle_history_max_abs_delta": _maximum_delta(
            baseline_output,
            vehicle,
        ),
    }
    path_rows = []
    for stream_name, stream_output in (
        ("physiology", baseline_encoding.alignment_output.physiology),
        ("vehicle", baseline_encoding.alignment_output.vehicle),
    ):
        trace = stream_output.path_trace
        if trace is None:
            raise ValueError("Chronaris continuous path trace is missing")
        path_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": DATASET_LABELS[dataset_id],
                "stream_name": stream_name,
                "continuous_evolution_enabled": trace.continuous_evolution_enabled,
                "observation_update_count": trace.observation_update_count,
                "observation_positive_evolution_count": trace.observation_positive_evolution_count,
                "reference_query_count": trace.reference_query_count,
                "reference_positive_evolution_count": trace.reference_positive_evolution_count,
                "maximum_positive_delta_t_s": trace.maximum_positive_delta_t_s,
            }
        )
    physics_rows = tuple(
        {
            "dataset_id": dataset_id,
            "dataset_label": DATASET_LABELS[dataset_id],
            **row,
        }
        for row in physics_audit_to_rows(baseline_encoding.physics_audit)
    )
    return causality, sensitivity, tuple(path_rows), physics_rows


def build_lag_boundary_audit_rows():
    query = torch.tensor([[30.0]])
    keys = torch.tensor([[30.0, 25.0, 15.0, 0.0, -1.0]])
    valid_query = torch.ones_like(query, dtype=torch.bool)
    valid_key = torch.ones_like(keys, dtype=torch.bool)
    rows = []
    for index, (lower, upper) in enumerate(
        ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))
    ):
        mask = build_seconds_lag_mask(
            query,
            keys,
            query_valid_mask=valid_query,
            key_valid_mask=valid_key,
            lower_s=lower,
            upper_s=upper,
            range_index=index,
            use_causal_mask=True,
        )[0, 0]
        for key_time, selected in zip(keys[0].tolist(), mask.tolist(), strict=True):
            rows.append(
                {
                    "scale_index": index,
                    "lower_s": lower,
                    "upper_s": upper,
                    "query_time_s": 30.0,
                    "key_time_s": key_time,
                    "lag_s": 30.0 - key_time,
                    "selected": selected,
                }
            )
    return rows


def build_chronaris_acceptance_rows(
    *,
    registry,
    parameter_rows,
    causality_rows,
    sensitivity_rows,
    path_rows,
    physics_rows,
    ablation_rows,
    lag_boundary_rows,
    export_manifest,
    dataset_metadata,
    transform_manifest,
):
    worst_future = max(row["future_perturbation_max_abs_delta"] for row in causality_rows)
    minimum_phys = min(row["physiology_history_max_abs_delta"] for row in sensitivity_rows)
    minimum_vehicle = min(row["vehicle_history_max_abs_delta"] for row in sensitivity_rows)
    fit_isolated = all(
        dataset["method"]["fit_sample_hash"]
        == dataset["fold"]["train_sample_hash"]
        for dataset in transform_manifest["datasets"].values()
    )
    simulation_active = sum(
        row["status"] == "active"
        for row in physics_rows
        if row["dataset_id"] == "simulation"
    )
    unavailable_are_null = all(
        row["raw_value"] is None and row["weighted_value"] is None
        for row in physics_rows
        if row["status"] == "unavailable"
    )
    boundary_membership = {}
    for row in lag_boundary_rows:
        if row["selected"]:
            boundary_membership.setdefault(row["lag_s"], []).append(row["scale_index"])
    expected_boundaries = {0.0: [0], 5.0: [0], 15.0: [1], 30.0: [2]}
    ablation_variants = {row["variant"] for row in ablation_rows}
    no_causal_counterfactual = [
        row["future_counterfactual_max_abs_delta"]
        for row in ablation_rows
        if row["variant"] == "no_causal_mask"
    ]
    return [
        _check("two_dataset_exports", export_manifest["available_export_count"] == 2, export_manifest["available_export_count"], 2),
        _check("resume_reuses_all_exports", export_manifest["resume_verification_reused_count"] == 2, export_manifest["resume_verification_reused_count"], 2),
        _check("checkpoint_registry_complete", len(registry.records) == 2, len(registry.records), 2),
        _check("fixed_query_and_output_shape", all(row["query_point_count"] == 96 and row["output_dim"] == 64 for row in parameter_rows), [(row["query_point_count"], row["output_dim"]) for row in parameter_rows], (96, 64)),
        _check("future_observation_invariance", worst_future <= 1e-5, worst_future, 1e-5),
        _check("physiology_stream_sensitivity", minimum_phys > 0, minimum_phys, "> 0"),
        _check("vehicle_stream_sensitivity", minimum_vehicle > 0, minimum_vehicle, "> 0"),
        _check("continuous_evolution_executed", all(row["continuous_evolution_enabled"] and row["observation_positive_evolution_count"] > 0 and row["reference_positive_evolution_count"] > 0 for row in path_rows), len(path_rows), 4),
        _check("reference_query_count", all(row["reference_query_count"] == 96 for row in path_rows), [row["reference_query_count"] for row in path_rows], 96),
        _check("seconds_lag_boundaries", boundary_membership == expected_boundaries, boundary_membership, expected_boundaries),
        _check("simulation_physics_active", simulation_active >= 3, simulation_active, ">= 3"),
        _check("unavailable_physics_not_zero", unavailable_are_null, unavailable_are_null, True),
        _check("four_fixed_ablations", ablation_variants == {"no_continuous_evolution", "no_physics", "no_causal_mask", "single_scale_lag"}, sorted(ablation_variants), "four fixed variants"),
        _check("ablation_forward_finite", all(row["finite_output"] for row in ablation_rows), [row["finite_output"] for row in ablation_rows], True),
        _check("ablation_diff_exact", all(row["diff_valid"] for row in ablation_rows), [row["changed_fields"] for row in ablation_rows], "declared fields only"),
        _check("no_causal_ablation_exposes_future", len(no_causal_counterfactual) == 2 and all(value is not None and value > 0 for value in no_causal_counterfactual), no_causal_counterfactual, "> 0 for both datasets"),
        _check("train_only_transform", fit_isolated, fit_isolated, True),
        _check("dingxin_three_view_smoke", dataset_metadata["dingxin"]["view_count"] == 3, dataset_metadata["dingxin"]["view_count"], 3),
        _check("dingxin_label_sources_excluded", dataset_metadata["dingxin"]["excluded_label_source_count"] == 20, dataset_metadata["dingxin"]["excluded_label_source_count"], 20),
        _check("observed_only_inputs", not dataset_metadata["simulation"]["oracle_opened"] and not dataset_metadata["dingxin"]["label_or_target_opened"], False, False),
        _check("labels_not_used_for_encoder", all(not record.label_used_for_encoder_training for record in registry.records.values()), False, False),
    ]


def _maximum_delta(first, second, *, mask=None) -> float:
    delta = (first.sequence_embedding - second.sequence_embedding).abs()
    if mask is not None:
        delta = delta[mask]
    return float(delta.max().item())


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
