from __future__ import annotations

import pandas as pd

from chronaris.evaluation.application_tasks.dingxin_inner_splits import (
    build_dingxin_inner_split_plans,
    build_inner_task_coverage_rows,
    interval_overlap_count,
)


def _contexts():
    rows = []
    for sortie_id, view_ids in (
        ("sortie_a", ("view_a", "view_b")),
        ("sortie_b", ("view_c",)),
    ):
        for view_id in view_ids:
            for index in range(31):
                rows.append(
                    {
                        "context_id": f"{view_id}_{index:02d}",
                        "sortie_id": sortie_id,
                        "view_id": view_id,
                        "start_offset_ms": index * 5_000,
                        "end_offset_ms": index * 5_000 + 30_000,
                        "input_fully_observed": True,
                    }
                )
    return pd.DataFrame(rows)


def _outer_manifest():
    view_a = tuple(f"view_a_{index:02d}" for index in range(31))
    view_b = tuple(f"view_b_{index:02d}" for index in range(31))
    view_c = tuple(f"view_c_{index:02d}" for index in range(31))
    return {
        "split_protocols": [
            {
                "fold_id": "leave_one_view_out__fold01",
                "split_strategy": "leave_one_view_out",
                "classification_train_context_ids": list(view_b + view_c),
                "classification_test_context_ids": list(view_a),
            },
            {
                "fold_id": "leave_one_sortie_out__fold01",
                "split_strategy": "leave_one_sortie_out",
                "classification_train_context_ids": list(view_a + view_b),
                "classification_test_context_ids": list(view_c),
            },
        ]
    }


def test_inner_splits_use_sortie_group_or_temporal_overlap_embargo() -> None:
    plans, roles = build_dingxin_inner_split_plans(
        context_catalog=_contexts(),
        outer_split_manifest=_outer_manifest(),
    )
    by_fold = {plan.fold.fold_id: plan for plan in plans}
    group = by_fold["leave_one_view_out__fold01"]
    temporal = by_fold["leave_one_sortie_out__fold01"]

    assert group.inner_split_strategy == "leave_one_outer_train_sortie_out"
    assert (len(group.fold.train_sample_ids), len(group.fold.validation_sample_ids)) == (
        31,
        31,
    )
    assert temporal.inner_split_strategy.startswith("shared_vehicle_temporal")
    assert (
        len(temporal.fold.train_sample_ids),
        len(temporal.fold.validation_sample_ids),
        len(temporal.embargo_sample_ids),
    ) == (38, 14, 10)
    assert interval_overlap_count(roles) == 0


def test_task_coverage_uses_split_roles_without_changing_targets() -> None:
    contexts = _contexts()
    plans, _roles = build_dingxin_inner_split_plans(
        context_catalog=contexts,
        outer_split_manifest=_outer_manifest(),
    )
    rows = []
    for plan in plans:
        all_ids = (
            plan.fold.train_sample_ids
            + plan.fold.validation_sample_ids
            + plan.fold.held_out_sample_ids
        )
        for task_slug in (
            "maneuver_intensity_classification",
            "physiology_response_prediction",
        ):
            for context_id in all_ids:
                index = int(context_id.rsplit("_", 1)[1])
                rows.append(
                    {
                        "fold_id": plan.fold.fold_id,
                        "task_slug": task_slug,
                        "context_id": context_id,
                        "binding_status": "available",
                        "class_target": index % 3 if task_slug.startswith("maneuver") else -1,
                        "binary_target": index % 2 if task_slug.startswith("physiology") else -1,
                        "continuous_target": float(index) if task_slug.startswith("physiology") else float("nan"),
                    }
                )
    coverage = build_inner_task_coverage_rows(
        plans=plans,
        task_bindings=pd.DataFrame(rows),
    )

    assert len(coverage) == 12
    assert all(row["context_count"] > 0 for row in coverage)
    assert all(row["target_threshold_scope"] == "outer_train_smoke_only" for row in coverage)
