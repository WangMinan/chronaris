"""Leakage-safe inner validation plans inside fixed Dingxin outer folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from chronaris.representation import FoldLineage


@dataclass(frozen=True, slots=True)
class DingxinInnerSplitPlan:
    fold: FoldLineage
    outer_split_strategy: str
    inner_split_strategy: str
    validation_group_ids: tuple[str, ...]
    embargo_sample_ids: tuple[str, ...]

    def to_dict(self):
        return {
            **self.fold.to_dict(),
            "outer_split_strategy": self.outer_split_strategy,
            "inner_split_strategy": self.inner_split_strategy,
            "validation_group_ids": list(self.validation_group_ids),
            "embargo_sample_ids": list(self.embargo_sample_ids),
        }


def build_dingxin_inner_split_plans(
    *,
    context_catalog: pd.DataFrame,
    outer_split_manifest: Mapping[str, object],
    temporal_validation_block_count: int = 7,
):
    available = context_catalog[context_catalog["input_fully_observed"]].copy()
    context_index = available.set_index("context_id")
    plans = []
    role_rows = []
    for outer in outer_split_manifest["split_protocols"]:
        fold_id = str(outer["fold_id"])
        outer_train_ids = tuple(
            context_id
            for context_id in outer["classification_train_context_ids"]
            if context_id in context_index.index
        )
        held_out_ids = tuple(
            context_id
            for context_id in outer["classification_test_context_ids"]
            if context_id in context_index.index
        )
        train_frame = context_index.loc[list(outer_train_ids)].reset_index()
        train_sorties = tuple(sorted(train_frame["sortie_id"].astype(str).unique()))
        if len(train_sorties) >= 2:
            validation_sortie = train_sorties[-1]
            inner_validation = tuple(
                train_frame[train_frame["sortie_id"] == validation_sortie]
                .sort_values(["view_id", "start_offset_ms"])["context_id"]
                .astype(str)
            )
            inner_train = tuple(
                train_frame[train_frame["sortie_id"] != validation_sortie]
                .sort_values(["view_id", "start_offset_ms"])["context_id"]
                .astype(str)
            )
            embargo = ()
            inner_strategy = "leave_one_outer_train_sortie_out"
            validation_groups = (validation_sortie,)
        else:
            inner_train, inner_validation, embargo = _temporal_split(
                train_frame,
                validation_block_count=temporal_validation_block_count,
            )
            inner_strategy = "shared_vehicle_temporal_block_with_30s_embargo"
            validation_groups = train_sorties
        plan = DingxinInnerSplitPlan(
            fold=FoldLineage(
                fold_id=fold_id,
                train_sample_ids=inner_train,
                validation_sample_ids=inner_validation,
                held_out_sample_ids=held_out_ids,
            ),
            outer_split_strategy=str(outer["split_strategy"]),
            inner_split_strategy=inner_strategy,
            validation_group_ids=validation_groups,
            embargo_sample_ids=embargo,
        )
        plans.append(plan)
        role_by_id = {
            **{value: "inner_train" for value in inner_train},
            **{value: "validation" for value in inner_validation},
            **{value: "embargo" for value in embargo},
            **{value: "outer_test" for value in held_out_ids},
        }
        for context_id, role in role_by_id.items():
            context = context_index.loc[context_id]
            role_rows.append(
                {
                    "fold_id": fold_id,
                    "outer_split_strategy": outer["split_strategy"],
                    "inner_split_strategy": inner_strategy,
                    "context_id": context_id,
                    "role": role,
                    "sortie_id": context.sortie_id,
                    "view_id": context.view_id,
                    "start_offset_ms": int(context.start_offset_ms),
                    "end_offset_ms": int(context.end_offset_ms),
                }
            )
    return tuple(plans), pd.DataFrame(role_rows)


def build_inner_task_coverage_rows(
    *,
    plans: Sequence[DingxinInnerSplitPlan],
    task_bindings: pd.DataFrame,
):
    rows = []
    for plan in plans:
        role_ids = {
            "inner_train": set(plan.fold.train_sample_ids),
            "validation": set(plan.fold.validation_sample_ids),
            "outer_test": set(plan.fold.held_out_sample_ids),
        }
        fold_bindings = task_bindings[
            (task_bindings["fold_id"] == plan.fold.fold_id)
            & (task_bindings["binding_status"] == "available")
        ]
        for task_slug, task_frame in fold_bindings.groupby("task_slug"):
            for role, context_ids in role_ids.items():
                selected = task_frame[task_frame["context_id"].isin(context_ids)]
                class_values = sorted(
                    int(value)
                    for value in selected["class_target"].astype(int).unique()
                    if value >= 0
                )
                binary_values = sorted(
                    int(value)
                    for value in selected["binary_target"].astype(int).unique()
                    if value >= 0
                )
                rows.append(
                    {
                        "fold_id": plan.fold.fold_id,
                        "task_slug": task_slug,
                        "role": role,
                        "context_count": len(selected),
                        "class_values": class_values,
                        "binary_values": binary_values,
                        "continuous_finite_count": int(
                            selected["continuous_target"].notna().sum()
                        ),
                        "target_threshold_scope": "outer_train_smoke_only",
                    }
                )
    return rows


def interval_overlap_count(role_rows: pd.DataFrame) -> int:
    overlap_count = 0
    for (_fold_id, _sortie_id), frame in role_rows.groupby(
        ["fold_id", "sortie_id"]
    ):
        train = frame[frame["role"] == "inner_train"]
        validation = frame[frame["role"] == "validation"]
        for left in train.itertuples(index=False):
            for right in validation.itertuples(index=False):
                if max(left.start_offset_ms, right.start_offset_ms) < min(
                    left.end_offset_ms, right.end_offset_ms
                ):
                    overlap_count += 1
    return overlap_count


def _temporal_split(frame, *, validation_block_count):
    blocks = (
        frame[["start_offset_ms", "end_offset_ms"]]
        .drop_duplicates()
        .sort_values(["start_offset_ms", "end_offset_ms"])
        .reset_index(drop=True)
    )
    if len(blocks) <= validation_block_count + 1:
        raise ValueError("not enough temporal blocks for inner validation")
    validation_blocks = blocks.iloc[-validation_block_count:]
    validation_start = int(validation_blocks["start_offset_ms"].min())
    validation_pairs = set(
        zip(
            validation_blocks["start_offset_ms"],
            validation_blocks["end_offset_ms"],
            strict=True,
        )
    )
    ordered = frame.sort_values(["view_id", "start_offset_ms"])
    train_ids = []
    validation_ids = []
    embargo_ids = []
    for row in ordered.itertuples(index=False):
        pair = (row.start_offset_ms, row.end_offset_ms)
        if pair in validation_pairs:
            validation_ids.append(str(row.context_id))
        elif int(row.end_offset_ms) <= validation_start:
            train_ids.append(str(row.context_id))
        else:
            embargo_ids.append(str(row.context_id))
    if not train_ids or not validation_ids or not embargo_ids:
        raise ValueError("temporal inner split failed to create train/validation/embargo")
    return tuple(train_ids), tuple(validation_ids), tuple(embargo_ids)
