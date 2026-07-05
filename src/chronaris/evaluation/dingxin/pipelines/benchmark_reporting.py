"""Reporting and decision helpers for the task evaluation private benchmark."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from chronaris.evaluation import save_grouped_bar_plot
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
)
from chronaris.evaluation.dingxin.pipelines.optimization import optimized_no_mask_variant_name


def write_task_plots(
    task_results: Mapping[str, object],
    *,
    artifact_root,
) -> dict[str, str]:
    plots_root = artifact_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    t1_groups = {
        variant_name: {
            "macro_f1": payload["best_metrics"]["macro_f1"],
            "balanced_accuracy": payload["best_metrics"]["balanced_accuracy"],
        }
        for variant_name, payload in task_results[TASK_MANEUVER]["variants"].items()
        if payload.get("status") == "completed"
    }
    t2_groups = {
        variant_name: {
            "rmse": payload["best_metrics"]["rmse"],
            "mae": payload["best_metrics"]["mae"],
        }
        for variant_name, payload in task_results[TASK_RESPONSE]["variants"].items()
        if payload.get("status") == "completed"
    }
    t3_groups = {
        variant_name: {
            "top1_accuracy": payload["top1_accuracy"],
            "mrr": payload["mrr"],
        }
        for variant_name, payload in task_results[TASK_RETRIEVAL]["variants"].items()
        if payload.get("status") == "completed"
    }
    return {
        "t1_metrics": save_grouped_bar_plot(
            t1_groups,
            path=plots_root / "t1_maneuver_metrics.png",
            title="Private T1 maneuver metrics",
            ylabel="score",
        ),
        "t2_metrics": save_grouped_bar_plot(
            t2_groups,
            path=plots_root / "t2_response_metrics.png",
            title="Private T2 response metrics",
            ylabel="score",
        ),
        "t3_metrics": save_grouped_bar_plot(
            t3_groups,
            path=plots_root / "t3_retrieval_metrics.png",
            title="Private T3 retrieval metrics",
            ylabel="score",
        ),
    }


def build_conclusion(
    task_results: Mapping[str, object],
    diagnostics: Mapping[str, object],
    *,
    target_variant_name: str = "g_min",
) -> dict[str, object]:
    t1_variants = task_results[TASK_MANEUVER]["variants"]
    t2_variants = task_results[TASK_RESPONSE]["variants"]
    t3_variants = task_results[TASK_RETRIEVAL]["variants"]
    no_mask_name = (
        "g_no_causal_mask"
        if target_variant_name == "g_min"
        else optimized_no_mask_variant_name(target_variant_name)
    )
    target_t1 = t1_variants.get(target_variant_name, {}).get("best_metrics")
    target_t2 = t2_variants.get(target_variant_name, {}).get("best_metrics")
    target_t3 = t3_variants.get(target_variant_name)
    deep_t1 = task_results[TASK_MANEUVER]["best_deep_model"]["metrics"]
    deep_t2 = task_results[TASK_RESPONSE]["best_deep_model"]["metrics"]
    alignment_gain_supported = is_better_classification(
        t1_variants.get("f_full", {}).get("best_metrics"),
        t1_variants.get("e_baseline", {}).get("best_metrics"),
    ) and is_better_classification(
        t1_variants.get("e_baseline", {}).get("best_metrics"),
        t1_variants.get("naive_sync", {}).get("best_metrics"),
    )
    if not alignment_gain_supported:
        alignment_gain_supported = is_better_regression(
            t2_variants.get("f_full", {}).get("best_metrics"),
            t2_variants.get("e_baseline", {}).get("best_metrics"),
        ) and is_better_regression(
            t2_variants.get("e_baseline", {}).get("best_metrics"),
            t2_variants.get("naive_sync", {}).get("best_metrics"),
        )
    t1_target_beats_module_baselines = metrics_better_than_all(
        target_t1,
        (
            t1_variants.get("naive_sync", {}).get("best_metrics"),
            t1_variants.get("e_baseline", {}).get("best_metrics"),
            t1_variants.get("f_full", {}).get("best_metrics"),
            t1_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        task_type="classification",
    )
    t2_target_beats_module_baselines = metrics_better_than_all(
        target_t2,
        (
            t2_variants.get("naive_sync", {}).get("best_metrics"),
            t2_variants.get("e_baseline", {}).get("best_metrics"),
            t2_variants.get("f_full", {}).get("best_metrics"),
            t2_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        task_type="regression",
    )
    t3_target_beats_module_baselines = metrics_better_than_all(
        target_t3,
        (
            t3_variants.get("naive_sync"),
            t3_variants.get("e_baseline"),
            t3_variants.get("f_full"),
            t3_variants.get(no_mask_name),
        ),
        task_type="retrieval",
    )
    causal_gain_supported = (
        is_better_classification(target_t1, t1_variants.get("f_full", {}).get("best_metrics"))
        and is_better_classification(target_t1, t1_variants.get(no_mask_name, {}).get("best_metrics"))
    ) or (
        is_better_regression(target_t2, t2_variants.get("f_full", {}).get("best_metrics"))
        and is_better_regression(target_t2, t2_variants.get(no_mask_name, {}).get("best_metrics"))
    ) or (
        is_better_retrieval(target_t3, t3_variants.get("f_full"))
        and is_better_retrieval(target_t3, t3_variants.get(no_mask_name))
    )
    target_diagnostics = diagnostics.get(target_variant_name, {})
    no_mask_diagnostics = diagnostics.get(no_mask_name, {})
    diagnostic_supported = bool(
        (
            target_diagnostics.get("mean_top_event_concentration", 0.0)
            > no_mask_diagnostics.get("mean_top_event_concentration", 0.0)
            and target_diagnostics.get("mean_event_mask_interference", 0.0)
            > no_mask_diagnostics.get("mean_event_mask_interference", 0.0)
        )
        or (
            target_diagnostics.get("mean_causal_residual_gate", 0.0)
            > no_mask_diagnostics.get("mean_causal_residual_gate", 0.0)
        )
    )
    t1_target_beats_deep = is_better_classification(target_t1, deep_t1)
    t2_target_beats_deep = is_better_regression(target_t2, deep_t2)
    private_optimality_supported = (
        t1_target_beats_module_baselines
        and t2_target_beats_module_baselines
        and t3_target_beats_module_baselines
        and t1_target_beats_deep
        and t2_target_beats_deep
    )
    criterion_details = {
        f"t1_{target_variant_name}_beats_module_baselines": t1_target_beats_module_baselines,
        f"t2_{target_variant_name}_beats_module_baselines": t2_target_beats_module_baselines,
        f"t3_{target_variant_name}_beats_module_baselines": t3_target_beats_module_baselines,
        f"t1_{target_variant_name}_beats_best_deep": t1_target_beats_deep,
        f"t2_{target_variant_name}_beats_best_deep": t2_target_beats_deep,
        f"t1_{target_variant_name}_beats_{no_mask_name}": is_better_classification(
            target_t1,
            t1_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        f"t2_{target_variant_name}_beats_{no_mask_name}": is_better_regression(
            target_t2,
            t2_variants.get(no_mask_name, {}).get("best_metrics"),
        ),
        f"t3_{target_variant_name}_beats_{no_mask_name}": is_better_retrieval(
            target_t3,
            t3_variants.get(no_mask_name),
        ),
    }
    if target_variant_name == "g_min":
        criterion_details = {
            "t1_g_min_beats_module_baselines": t1_target_beats_module_baselines,
            "t2_g_min_beats_module_baselines": t2_target_beats_module_baselines,
            "t3_g_min_beats_module_baselines": t3_target_beats_module_baselines,
            "t1_g_min_beats_best_deep": t1_target_beats_deep,
            "t2_g_min_beats_best_deep": t2_target_beats_deep,
        }
    return {
        "target_variant_name": target_variant_name,
        "no_mask_variant_name": no_mask_name,
        "alignment_gain_supported": alignment_gain_supported,
        "causal_gain_supported": causal_gain_supported,
        "diagnostic_supported": diagnostic_supported,
        "private_optimality_supported": private_optimality_supported,
        "criterion_details": criterion_details,
    }


def is_better_classification(current: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> bool:
    if current is None or reference is None:
        return False
    return (
        current["macro_f1"] > reference["macro_f1"]
        or (
            math.isclose(current["macro_f1"], reference["macro_f1"])
            and current["balanced_accuracy"] > reference["balanced_accuracy"]
        )
    )


def is_better_regression(current: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> bool:
    if current is None or reference is None:
        return False
    return (
        current["rmse"] < reference["rmse"]
        or (
            math.isclose(current["rmse"], reference["rmse"])
            and current["mae"] < reference["mae"]
        )
    )


def is_better_retrieval(current: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> bool:
    if current is None or reference is None:
        return False
    if current.get("status") != "completed" or reference.get("status") != "completed":
        return False
    return (
        current["top1_accuracy"] > reference["top1_accuracy"]
        or (
            math.isclose(current["top1_accuracy"], reference["top1_accuracy"])
            and current["mrr"] > reference["mrr"]
        )
    )


def metrics_better_than_all(
    current: Mapping[str, object] | None,
    references: Sequence[Mapping[str, object] | None],
    *,
    task_type: str,
) -> bool:
    if current is None:
        return False
    comparators = {
        "classification": is_better_classification,
        "regression": is_better_regression,
        "retrieval": is_better_retrieval,
    }
    comparator = comparators[task_type]
    return all(comparator(current, reference) for reference in references)


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
