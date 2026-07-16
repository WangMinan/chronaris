"""Compatibility audit for reusing historical Dingxin representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd


COMPATIBILITY_STATUSES = (
    "reusable_as_primary",
    "reusable_as_sensitivity_only",
    "retraining_required",
)


def audit_representation_compatibility(
    *,
    expected_context_ids: Sequence[str],
    inventory_path: str | Path,
    protocol_path: str | Path,
    field_role_manifest_path: str | Path,
    required_methods: Sequence[str],
    required_seeds: Sequence[int],
    repository_root: str | Path = ".",
) -> dict[str, object]:
    inventory_file = Path(inventory_path)
    protocol_file = Path(protocol_path)
    roles_file = Path(field_role_manifest_path)
    repo_root = Path(repository_root)
    inventory = pd.read_json(inventory_file, lines=True)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    roles = pd.read_csv(roles_file)
    expected = set(str(value) for value in expected_context_ids)
    loso = inventory[
        inventory["fold_id"].astype(str).str.startswith("leave_one_sortie_out__")
    ].copy()
    observed_methods = set(loso["method_name"].astype(str))
    observed_seeds = set(loso["seed"].astype(int))
    missing_units = []
    context_coverage_failures = []
    label_training_violations = []
    output_contract_violations = []
    manifest_count = 0
    for seed in required_seeds:
        for fold_id in sorted(loso["fold_id"].astype(str).unique()):
            for method in required_methods:
                unit = loso[
                    (loso["seed"].astype(int) == int(seed))
                    & (loso["fold_id"].astype(str) == str(fold_id))
                    & (loso["method_name"].astype(str) == str(method))
                ]
                if set(unit["export_role"].astype(str)) != {
                    "train",
                    "validation",
                    "held_out",
                }:
                    missing_units.append(
                        {"seed": int(seed), "fold_id": fold_id, "method": method}
                    )
                    continue
                unit_contexts = {
                    str(value)
                    for values in unit["sample_ids"]
                    for value in values
                }
                missing_contexts = sorted(expected - unit_contexts)
                if missing_contexts:
                    context_coverage_failures.append(
                        {
                            "seed": int(seed),
                            "fold_id": fold_id,
                            "method": method,
                            "missing_count": len(missing_contexts),
                            "examples": missing_contexts[:5],
                        }
                    )
                for manifest_path in unit["manifest_path"].astype(str):
                    manifest_file = Path(manifest_path)
                    if not manifest_file.is_absolute():
                        manifest_file = repo_root / manifest_file
                    manifest = json.loads(
                        manifest_file.read_text(encoding="utf-8")
                    )
                    manifest_count += 1
                    if bool(manifest.get("label_used_for_encoder_training")):
                        label_training_violations.append(str(manifest_file))
                    if int(manifest.get("output_dim", -1)) != 64 or int(
                        manifest.get("query_point_count", -1)
                    ) != 96:
                        output_contract_violations.append(str(manifest_file))
    maneuver_sources = roles[
        roles["selected_for_maneuver_label"].astype(bool)
    ].copy()
    excluded_historical_source_count = int(
        (~maneuver_sources["allowed_in_maneuver_input"].astype(bool)).sum()
    )
    checks = {
        "required_methods_present": set(required_methods) <= observed_methods,
        "required_seeds_present": set(int(value) for value in required_seeds)
        <= observed_seeds,
        "three_roles_per_unit": not missing_units,
        "new_contexts_covered": not context_coverage_failures,
        "encoder_task_label_isolation": not label_training_violations
        and protocol.get("task_targets_opened") is False,
        "uniform_64d_96point_contract": not output_contract_violations,
        "task_agnostic_representation_family": protocol.get("representation_family")
        == "frozen_task_agnostic_v1",
        "historical_maneuver_fields_allowed": excluded_historical_source_count == 0,
        "outer_metrics_closed_during_export": protocol.get("outer_test_metrics_opened")
        is False,
    }
    if not checks["historical_maneuver_fields_allowed"]:
        status = "retraining_required"
        reason = (
            "historical representations exclude maneuver-source fields that are legal "
            "past predictors in the simplified future-task protocol"
        )
    elif all(checks.values()):
        status = "reusable_as_primary"
        reason = "all simplified primary-representation checks passed"
    else:
        status = "reusable_as_sensitivity_only"
        reason = "one or more non-field compatibility checks failed"
    if status not in COMPATIBILITY_STATUSES:  # pragma: no cover - invariant.
        raise AssertionError(status)
    return {
        "format": "chronaris.simple_representation_compatibility.v1",
        "status": status,
        "reason": reason,
        "checks": checks,
        "expected_context_count": len(expected),
        "inventory_row_count": len(inventory),
        "loso_inventory_row_count": len(loso),
        "manifest_count_checked": manifest_count,
        "maneuver_source_field_count": len(maneuver_sources),
        "excluded_historical_maneuver_source_count": excluded_historical_source_count,
        "missing_units": missing_units,
        "context_coverage_failures": context_coverage_failures,
        "label_training_violations": label_training_violations,
        "output_contract_violations": output_contract_violations,
        "source_paths": {
            "inventory": str(inventory_file),
            "protocol": str(protocol_file),
            "field_roles": str(roles_file),
        },
    }
