"""Construct leakage-safe, support-deduplicated Dingxin development splits."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash
from chronaris.dataset.application_evaluation.labels import _build_maneuver_labels
from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    PRIMARY_FOLDS,
    purge_train_for_full_support,
)
from chronaris.evaluation.application_tasks.dingxin_nested_target_data import (
    CLASS_MAPPING,
    _build_nested_response_rows,
    _load_contexts,
    _load_field_roles,
    build_dingxin_nested_targets,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    build_raw_median_response_targets,
    load_dingxin_target_source_data,
)
from chronaris.evaluation.application_tasks.task_stability_contracts import (
    maneuver_metric_ceiling,
    stable_sha256,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    load_aligned_private_records,
)


SUPPORT_DURATION_MS = 35_000
MINIMUM_SUPPORT_EMBARGO_MS = 35_000
MINIMUM_ANCHOR_SEPARATION_MS = SUPPORT_DURATION_MS + MINIMUM_SUPPORT_EMBARGO_MS


def build_development_splits(
    *,
    fixed_root: str | Path,
    legacy_inner_root: str | Path,
    heavy_root: str | Path,
    snapshot_root: str | Path,
    e_run_manifest_path: str,
    f_run_manifest_path: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate candidates, fit train-only targets, and select six structural splits."""

    fixed = Path(fixed_root)
    heavy = Path(heavy_root)
    heavy.mkdir(parents=True, exist_ok=True)
    context = pd.read_json(fixed / "context_sample_manifest.jsonl", lines=True)
    context = context[
        context["classification_eligible"].astype(bool)
        & (context["end_window_index"].astype(int) <= 35)
    ].copy()
    outer = json.loads((fixed / "split_manifest.json").read_text(encoding="utf-8"))[
        "split_protocols"
    ][:3]
    candidates = _candidate_plans(outer, context)
    candidate_root = heavy / "split_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    _write_json(candidate_root / "split_manifest.json", {"folds": candidates})
    candidate_targets, candidate_thresholds, candidate_errors = (
        _fit_candidate_targets_resilient(
        fixed_root=fixed,
        snapshot_root=Path(snapshot_root),
        plans=candidates,
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
        )
    )
    candidate_targets.to_csv(candidate_root / "nested_targets.csv", index=False)
    candidate_thresholds.to_csv(candidate_root / "nested_thresholds.csv", index=False)
    validity = _candidate_validity(candidates, candidate_targets, candidate_errors)
    selected, pool_status = _select_structural_splits(candidates, validity)

    legacy_payload = json.loads(
        (Path(legacy_inner_root) / "split_manifest.json").read_text(encoding="utf-8")
    )
    legacy_by_id = {
        str(row["fold_id"]): row
        for row in legacy_payload["folds"]
        if str(row["fold_id"]) in PRIMARY_FOLDS
    }
    diagnostics = []
    for fold_id in PRIMARY_FOLDS:
        plan = purge_train_for_full_support(
            plan=legacy_by_id[fold_id],
            context_catalog=context,
        )
        plan.update(
            {
                "fold_id": f"legacy_diagnostic__{fold_id}",
                "outer_pool_id": fold_id,
                "split_kind": "distribution_pressure_diagnostic",
                "main_selection": False,
                "held_out_sample_ids": [],
            }
        )
        plan.update(support_audit(plan, context))
        diagnostics.append(plan)

    canonical_plans = [
        {
            **plan,
            "split_kind": "main_selection",
            "main_selection": True,
            **support_audit(plan, context),
        }
        for plan in selected
    ] + diagnostics
    canonical_root = heavy / "development_splits"
    canonical_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "chronaris.dingxin_task_stability_splits.v1",
        "support_definition": "30s_input_plus_current_query_plus_5s_target",
        "support_interval_convention": "half_open",
        "minimum_support_embargo_ms": MINIMUM_SUPPORT_EMBARGO_MS,
        "minimum_anchor_separation_ms": MINIMUM_ANCHOR_SEPARATION_MS,
        "outer_test_opened": False,
        "outer_identifiers_embedded": False,
        "main_split_count": sum(bool(row["main_selection"]) for row in canonical_plans),
        "unique_main_validation_support_count": len(
            {
                row["validation_support_hash"]
                for row in canonical_plans
                if row["main_selection"]
            }
        ),
        "outer_pool_status": pool_status,
        "folds": canonical_plans,
    }
    _write_json(canonical_root / "split_manifest.json", manifest)
    targets, thresholds, fold_frame = build_dingxin_nested_targets(
        fixed_audit_root=fixed,
        snapshot_root=snapshot_root,
        inner_split_root=canonical_root,
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
    )
    targets.to_csv(canonical_root / "nested_targets.csv", index=False)
    thresholds.to_csv(canonical_root / "nested_thresholds.csv", index=False)
    fold_frame.to_csv(canonical_root / "nested_target_folds.csv", index=False)
    return manifest, validity, targets, thresholds, _dedup_rows(canonical_plans)


def support_audit(plan: dict[str, object], context: pd.DataFrame) -> dict[str, object]:
    lookup = context.set_index("context_id")
    roles = {}
    for role, key in (("train", "train_sample_ids"), ("validation", "validation_sample_ids")):
        for sample_id in plan[key]:
            row = lookup.loc[str(sample_id)]
            unit = (str(row["sortie_id"]), int(row["start_offset_ms"]))
            prior = roles.setdefault(unit, role)
            if prior != role:
                raise PermissionError("shared vehicle support unit crosses roles")
    train = lookup.loc[list(plan["train_sample_ids"])]
    validation = lookup.loc[list(plan["validation_sample_ids"])]
    minimum_anchor_gap = None
    minimum_support_gap = None
    overlap_count = 0
    for left in train.itertuples():
        for right in validation.itertuples():
            if str(left.sortie_id) != str(right.sortie_id):
                continue
            gap = abs(int(left.start_offset_ms) - int(right.start_offset_ms))
            minimum_anchor_gap = gap if minimum_anchor_gap is None else min(
                minimum_anchor_gap, gap
            )
            left_interval = (int(left.start_offset_ms), int(left.start_offset_ms) + SUPPORT_DURATION_MS)
            right_interval = (
                int(right.start_offset_ms),
                int(right.start_offset_ms) + SUPPORT_DURATION_MS,
            )
            if max(left_interval[0], right_interval[0]) < min(
                left_interval[1], right_interval[1]
            ):
                overlap_count += 1
                interval_gap = 0
            else:
                interval_gap = max(
                    left_interval[0], right_interval[0]
                ) - min(left_interval[1], right_interval[1])
            minimum_support_gap = (
                interval_gap
                if minimum_support_gap is None
                else min(minimum_support_gap, interval_gap)
            )
    support_hash = validation_support_hash(plan["validation_sample_ids"], context)
    return {
        "validation_support_hash": support_hash,
        "validation_sample_hash": stable_sha256(sorted(plan["validation_sample_ids"])),
        "train_sample_hash": stable_sha256(sorted(plan["train_sample_ids"])),
        "support_overlap_count": overlap_count,
        "minimum_anchor_separation_ms": minimum_anchor_gap,
        "minimum_support_embargo_ms": minimum_support_gap,
        "support_isolated": overlap_count == 0,
        "shared_vehicle_unit_cross_role_count": 0,
        "outer_test_opened": False,
    }


def validation_support_hash(sample_ids, context: pd.DataFrame) -> str:
    lookup = context.set_index("context_id")
    units = sorted(
        {
            (
                str(lookup.loc[str(sample_id), "sortie_id"]),
                int(lookup.loc[str(sample_id), "start_offset_ms"]),
                int(lookup.loc[str(sample_id), "start_offset_ms"])
                + SUPPORT_DURATION_MS,
            )
            for sample_id in sample_ids
        }
    )
    return stable_sha256(units)


def _candidate_plans(outer, context: pd.DataFrame) -> list[dict[str, object]]:
    lookup = context.set_index("context_id")
    plans = []
    for outer_plan in outer:
        pool_id = str(outer_plan["fold_id"])
        pool_ids = [
            str(value)
            for value in outer_plan["classification_train_context_ids"]
            if str(value) in lookup.index
        ]
        pool = lookup.loc[pool_ids].reset_index()
        units = []
        for (sortie_id, start, _end), frame in pool.groupby(
            ["sortie_id", "start_offset_ms", "end_offset_ms"], sort=True
        ):
            units.append(
                {
                    "sortie_id": str(sortie_id),
                    "start_offset_ms": int(start),
                    "sample_ids": tuple(sorted(frame["context_id"].astype(str))),
                }
            )
        by_sortie = {
            sortie: sorted(
                [row for row in units if row["sortie_id"] == sortie],
                key=lambda row: row["start_offset_ms"],
            )
            for sortie in sorted({row["sortie_id"] for row in units})
        }
        validation_panels = []
        if len(by_sortie) == 2:
            sorties = tuple(by_sortie)
            for length in (5, 6):
                left_starts = range(0, len(by_sortie[sorties[0]]) - length + 1, 2)
                right_starts = range(0, len(by_sortie[sorties[1]]) - length + 1, 2)
                for left, right in itertools.product(left_starts, right_starts):
                    validation_panels.append(
                        by_sortie[sorties[0]][left : left + length]
                        + by_sortie[sorties[1]][right : right + length]
                    )
        else:
            sortie = next(iter(by_sortie))
            for length in (5, 6, 7, 8, 9, 10):
                for start in range(0, len(by_sortie[sortie]) - length + 1):
                    validation_panels.append(by_sortie[sortie][start : start + length])
        for index, validation_units in enumerate(validation_panels):
            train_units = [
                row
                for row in units
                if row not in validation_units
                and all(
                    row["sortie_id"] != held["sortie_id"]
                    or abs(row["start_offset_ms"] - held["start_offset_ms"])
                    >= MINIMUM_ANCHOR_SEPARATION_MS
                    for held in validation_units
                )
            ]
            train_counts = {
                sortie: sum(row["sortie_id"] == sortie for row in train_units)
                for sortie in by_sortie
            }
            if not train_counts or min(train_counts.values()) < 6:
                continue
            train_ids = [value for row in train_units for value in row["sample_ids"]]
            validation_ids = [
                value for row in validation_units for value in row["sample_ids"]
            ]
            plan = {
                "fold_id": f"{pool_id}__candidate_{index:03d}",
                "outer_pool_id": pool_id,
                "split_kind": "candidate",
                "train_sample_ids": train_ids,
                "validation_sample_ids": validation_ids,
                "held_out_sample_ids": [],
            }
            plan.update(support_audit(plan, context))
            plans.append(plan)
    return plans


def _candidate_validity(plans, targets: pd.DataFrame, errors=None) -> pd.DataFrame:
    errors = errors or {}
    rows = []
    for plan in plans:
        fold_id = plan["fold_id"]
        maneuver = targets[
            (targets["fold_id"] == fold_id)
            & (targets["task_slug"] == "maneuver_intensity_classification")
            & (targets["role"] == "validation")
            & (targets["status"] == "completed")
        ]
        response = targets[
            (targets["fold_id"] == fold_id)
            & (targets["task_slug"] == "physiology_response_prediction")
            & (targets["role"] == "validation")
            & (targets["status"] == "completed")
        ]
        ceiling = maneuver_metric_ceiling(maneuver["class_target"].astype(int))
        positive = int((response["binary_target"] == 1).sum())
        negative = int((response["binary_target"] == 0).sum())
        prevalence = positive / len(response) if len(response) else np.nan
        valid = bool(
            ceiling["complete_class_support"]
            and ceiling["minimum_class_count"] >= 2
            and len(response) >= 10
            and positive >= 3
            and negative >= 3
            and 0.20 <= prevalence <= 0.80
            and plan["support_overlap_count"] == 0
            and plan["minimum_support_embargo_ms"] >= MINIMUM_SUPPORT_EMBARGO_MS
        )
        counts = ceiling["class_counts"]
        rows.append(
            {
                "split_id": fold_id,
                "outer_pool_id": plan["outer_pool_id"],
                "validation_support_hash": plan["validation_support_hash"],
                "train_count": len(plan["train_sample_ids"]),
                "validation_count": len(plan["validation_sample_ids"]),
                "response_count": len(response),
                "maneuver_low_count": counts[0],
                "maneuver_medium_count": counts[1],
                "maneuver_high_count": counts[2],
                "minimum_maneuver_class_count": ceiling["minimum_class_count"],
                "high_response_positive_count": positive,
                "high_response_negative_count": negative,
                "high_response_prevalence": prevalence,
                "support_overlap_count": plan["support_overlap_count"],
                "minimum_anchor_separation_ms": plan[
                    "minimum_anchor_separation_ms"
                ],
                "minimum_support_embargo_ms": plan["minimum_support_embargo_ms"],
                "outer_test_opened": False,
                "valid_main_split": valid,
                "target_fit_status": "unavailable" if fold_id in errors else "completed",
                "target_fit_error": errors.get(fold_id),
            }
        )
    return pd.DataFrame(rows)


def _fit_candidate_targets_resilient(
    *, fixed_root: Path, snapshot_root: Path, plans,
    e_run_manifest_path: str, f_run_manifest_path: str,
):
    contexts = _load_contexts(fixed_root / "context_sample_manifest.jsonl")
    context_by_id = {context.context_id: context for context in contexts}
    roles = _load_field_roles(fixed_root / "field_role_manifest.csv")
    records = load_aligned_private_records(
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
    )
    record_by_sample = {
        str(row.sample_id): row for row in records.itertuples(index=False)
    }
    response_source = load_dingxin_target_source_data(
        fixed_audit_root=fixed_root,
        snapshot_root=snapshot_root,
    )
    response_raw = build_raw_median_response_targets(
        response_source, snapshot_root=snapshot_root
    )
    delta_frame = response_raw.field_delta_rows
    delta_index = {
        (str(row.context_id), str(row.feature_name)): float(row.absolute_delta)
        for row in delta_frame.itertuples(index=False)
        if np.isfinite(row.absolute_delta)
    }
    candidate_fields = tuple(sorted(delta_frame["feature_name"].astype(str).unique()))
    label_rows = []
    threshold_rows = []
    errors = {}
    for plan in plans:
        fold_id = str(plan["fold_id"])
        role_ids = {
            "train": tuple(str(value) for value in plan["train_sample_ids"]),
            "validation": tuple(
                str(value) for value in plan["validation_sample_ids"]
            ),
            "held_out": (),
        }
        fit_hash = stable_sample_hash(role_ids["train"])
        try:
            maneuver_rows, maneuver_thresholds = _build_maneuver_labels(
                record_by_sample=record_by_sample,
                context_by_id=context_by_id,
                train_context_ids=role_ids["train"],
                test_context_ids=role_ids["validation"],
                roles=roles,
                fit_sample_hash=fit_hash,
                minimum_semantic_count=4,
                eps=1e-6,
            )
            response_rows, response_thresholds, _ = _build_nested_response_rows(
                fold_id=fold_id,
                role_ids=role_ids,
                delta_index=delta_index,
                candidate_fields=candidate_fields,
                minimum_field_count=2,
                train_valid_ratio=0.80,
                eps=1e-6,
            )
        except (ValueError, RuntimeError) as exc:
            errors[fold_id] = f"{type(exc).__name__}: {exc}"
            continue
        role_by_id = {
            context_id: role
            for role, sample_ids in role_ids.items()
            for context_id in sample_ids
        }
        label_rows.extend(
            {
                "fold_id": fold_id,
                "task_slug": "maneuver_intensity_classification",
                "role": role_by_id[row["context_id"]],
                "context_id": row["context_id"],
                "class_target": CLASS_MAPPING.get(row["class_label"], -1),
                "continuous_target": None,
                "binary_target": -1,
                "status": row["status"],
                "fit_sample_hash": fit_hash,
                "threshold_scope": "inner_train_nested",
            }
            for row in maneuver_rows
        )
        threshold_rows.extend(
            {
                "fold_id": fold_id,
                "task_slug": "maneuver_intensity_classification",
                "threshold_scope": "inner_train_nested",
                **row,
            }
            for row in maneuver_thresholds
        )
        label_rows.extend(response_rows)
        threshold_rows.extend(response_thresholds)
    return pd.DataFrame(label_rows), pd.DataFrame(threshold_rows), errors


def _select_structural_splits(plans, validity: pd.DataFrame):
    plan_by_id = {row["fold_id"]: row for row in plans}
    frame = validity[validity["valid_main_split"]].copy()
    frame["class_imbalance"] = (
        frame[["maneuver_low_count", "maneuver_medium_count", "maneuver_high_count"]]
        .max(axis=1)
        - frame[["maneuver_low_count", "maneuver_medium_count", "maneuver_high_count"]]
        .min(axis=1)
    )
    frame["prevalence_distance"] = abs(frame["high_response_prevalence"] - 0.5)
    frame = frame.sort_values(
        [
            "minimum_maneuver_class_count",
            "class_imbalance",
            "prevalence_distance",
            "validation_count",
            "train_count",
            "validation_support_hash",
        ],
        ascending=[False, True, True, False, False, True],
    )
    selected_ids = []
    selected_hashes = set()
    pool_status = []
    for pool_id in PRIMARY_FOLDS:
        pool = frame[frame["outer_pool_id"] == pool_id]
        picked = []
        for row in pool.itertuples(index=False):
            if row.validation_support_hash in selected_hashes:
                continue
            picked.append(row.split_id)
            selected_hashes.add(row.validation_support_hash)
            if len(picked) == 2:
                break
        selected_ids.extend(picked)
        pool_status.append(
            {
                "outer_pool_id": pool_id,
                "valid_candidate_count": int(len(pool)),
                "initial_selected_count": len(picked),
                "status": "available" if picked else "split_unavailable",
            }
        )
    if len(selected_ids) < 6:
        for row in frame.itertuples(index=False):
            if row.split_id in selected_ids or row.validation_support_hash in selected_hashes:
                continue
            selected_ids.append(row.split_id)
            selected_hashes.add(row.validation_support_hash)
            if len(selected_ids) == 6:
                break
    if len(selected_ids) < 6:
        return [], pool_status
    counts = defaultdict(int)
    for split_id in selected_ids:
        counts[plan_by_id[split_id]["outer_pool_id"]] += 1
    for row in pool_status:
        row["final_selected_count"] = counts[row["outer_pool_id"]]
    return [plan_by_id[value] for value in selected_ids], pool_status


def _dedup_rows(plans) -> pd.DataFrame:
    seen = {}
    rows = []
    for plan in plans:
        support_hash = plan["validation_support_hash"]
        duplicate = seen.get(support_hash)
        if duplicate is None:
            seen[support_hash] = plan["fold_id"]
        rows.append(
            {
                "split_id": plan["fold_id"],
                "outer_pool_id": plan["outer_pool_id"],
                "split_kind": plan["split_kind"],
                "validation_support_hash": support_hash,
                "duplicate_of_split_id": duplicate,
                "effective_weight": 0.0 if duplicate else 1.0,
                "included_in_main_ranking": bool(plan["main_selection"] and not duplicate),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
