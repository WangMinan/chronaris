"""Protocol and access guards for Dingxin core-task feasibility screening."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash


PRIMARY_FOLDS = (
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
)

UPPER_BOUND_GATES = {
    "maneuver": {"metric": "macro_f1", "direction": "higher", "threshold": 0.950},
    "response": {"metric": "rmse", "direction": "lower", "threshold": 0.285},
    "high_response": {
        "metric": "auprc",
        "direction": "higher",
        "threshold": 0.900,
    },
}

SAFE_FUSION_GATES = {
    "maneuver": {"metric": "macro_f1", "direction": "higher", "threshold": 0.900},
    "response": {"metric": "rmse", "direction": "lower", "threshold": 0.305},
    "high_response": {
        "metric": "auprc",
        "direction": "higher",
        "threshold": 0.880,
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class InnerRoleAccessGuard:
    """Fail closed when development code asks for a held-out role or sample."""

    role_by_sample_id: Mapping[str, str]
    audit_rows: list[dict[str, object]] = field(default_factory=list)

    def request(
        self,
        *,
        fold_id: str,
        role: str,
        sample_ids: Sequence[str],
        purpose: str,
    ) -> tuple[str, ...]:
        normalized_role = str(role).strip().lower()
        identifiers = tuple(str(value) for value in sample_ids)
        allowed_role = normalized_role in {"train", "inner_train", "validation"}
        actual_roles = {
            self.role_by_sample_id.get(sample_id, "unknown") for sample_id in identifiers
        }
        roles_allowed = actual_roles <= {"train", "validation"}
        allowed = allowed_role and roles_allowed and bool(identifiers)
        self.audit_rows.append(
            {
                "fold_id": fold_id,
                "requested_role": normalized_role,
                "purpose": purpose,
                "sample_count": len(identifiers),
                "actual_roles": ";".join(sorted(actual_roles)),
                "allowed": allowed,
                "outer_test_accessed": False,
                "reason": (
                    "inner_role_allowed"
                    if allowed
                    else "held_out_or_unknown_role_rejected"
                ),
            }
        )
        if not allowed:
            raise PermissionError(
                f"development access rejected for {fold_id}: role={role}, "
                f"actual_roles={sorted(actual_roles)}"
            )
        return identifiers

    def record_blocked_probe(self, *, fold_id: str, role: str) -> None:
        """Exercise the guard without exposing held-out sample identifiers."""

        try:
            self.request(
                fold_id=fold_id,
                role=role,
                sample_ids=("__outer_probe__",),
                purpose="fail_closed_self_test",
            )
        except PermissionError:
            return
        raise AssertionError("outer role probe unexpectedly passed")


def role_map_for_fold(plan: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for role, key in (
        ("train", "train_sample_ids"),
        ("validation", "validation_sample_ids"),
        ("held_out", "held_out_sample_ids"),
    ):
        for value in plan[key]:
            sample_id = str(value)
            if sample_id in result:
                raise ValueError(f"sample appears in multiple roles: {sample_id}")
            result[sample_id] = role
    return result


def audit_train_validation_support(
    *,
    plan: Mapping[str, object],
    context_catalog: pd.DataFrame,
    target_horizon_ms: int = 5_000,
) -> dict[str, object]:
    """Check complete 30 s input plus 5 s target support isolation."""

    required = {
        "context_id",
        "sortie_id",
        "start_offset_ms",
        "end_offset_ms",
    }
    missing = sorted(required - set(context_catalog.columns))
    if missing:
        raise ValueError(f"context catalog missing support columns: {missing}")
    lookup = context_catalog.set_index("context_id")
    train = _support_rows(plan["train_sample_ids"], lookup, target_horizon_ms)
    validation = _support_rows(
        plan["validation_sample_ids"], lookup, target_horizon_ms
    )
    overlap_pairs = []
    for left in train:
        for right in validation:
            if left[1] != right[1]:
                continue
            if max(left[2], right[2]) < min(left[3], right[3]):
                overlap_pairs.append((left[0], right[0]))
    return {
        "fold_id": str(plan["fold_id"]),
        "train_context_count": len(train),
        "validation_context_count": len(validation),
        "support_overlap_pair_count": len(overlap_pairs),
        "support_isolated": not overlap_pairs,
        "outer_test_accessed": False,
    }


def purge_train_for_full_support(
    *,
    plan: Mapping[str, object],
    context_catalog: pd.DataFrame,
    target_horizon_ms: int = 5_000,
) -> dict[str, object]:
    """Derive an inner-only plan with complete 30 s plus 5 s support isolation."""

    lookup = context_catalog.set_index("context_id")
    train = _support_rows(plan["train_sample_ids"], lookup, target_horizon_ms)
    validation = _support_rows(
        plan["validation_sample_ids"], lookup, target_horizon_ms
    )
    kept = []
    purged = []
    for left in train:
        overlaps = any(
            left[1] == right[1]
            and max(left[2], right[2]) < min(left[3], right[3])
            for right in validation
        )
        (purged if overlaps else kept).append(left[0])
    if not kept:
        raise ValueError(f"support purge removed every train sample: {plan['fold_id']}")
    validation_ids = [str(value) for value in plan["validation_sample_ids"]]
    return {
        "fold_id": str(plan["fold_id"]),
        "outer_split_strategy": str(plan.get("outer_split_strategy", "unknown")),
        "inner_split_strategy": (
            f"{plan.get('inner_split_strategy', 'unknown')}__purged_full_35s_support"
        ),
        "train_sample_ids": kept,
        "train_sample_hash": stable_sample_hash(kept),
        "validation_sample_ids": validation_ids,
        "validation_sample_hash": stable_sample_hash(validation_ids),
        "held_out_sample_ids": [],
        "held_out_sample_hash": stable_sample_hash(()),
        "outer_test_sample_ids_removed_from_development_manifest": len(
            plan["held_out_sample_ids"]
        ),
        "purged_for_full_35s_support_sample_ids": purged,
        "development_roles": ["train", "validation"],
        "outer_test_opened": False,
    }


def gate_passed(value: float | None, specification: Mapping[str, object]) -> bool:
    if value is None:
        return False
    threshold = float(specification["threshold"])
    if specification["direction"] == "higher":
        return float(value) >= threshold
    return float(value) <= threshold


def aggregate_gate_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    gates: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    outputs = []
    for task, specification in gates.items():
        subset = frame[
            (frame["task"] == task)
            & (frame["metric"] == specification["metric"])
            & frame["value"].notna()
        ]
        if subset.empty:
            best_value = None
            candidate_id = None
            fold_pass_count = 0
        else:
            ascending = specification["direction"] == "lower"
            candidate_summary = (
                subset.groupby("candidate_id", as_index=False)["value"].mean()
                .sort_values("value", ascending=ascending, kind="mergesort")
                .reset_index(drop=True)
            )
            candidate_id = str(candidate_summary.iloc[0]["candidate_id"])
            selected = subset[subset["candidate_id"] == candidate_id]
            best_value = float(selected["value"].mean())
            fold_pass_count = sum(
                gate_passed(float(value), specification)
                for value in selected["value"]
            )
        outputs.append(
            {
                "task": task,
                "metric": specification["metric"],
                "direction": specification["direction"],
                "threshold": specification["threshold"],
                "best_candidate_id": candidate_id,
                "best_mean_value": best_value,
                "mean_gate_passed": gate_passed(best_value, specification),
                "fold_gate_pass_count": fold_pass_count,
                "required_fold_count": len(PRIMARY_FOLDS),
            }
        )
    return outputs


def _support_rows(sample_ids, lookup, target_horizon_ms):
    rows = []
    for sample_id in sample_ids:
        row = lookup.loc[str(sample_id)]
        rows.append(
            (
                str(sample_id),
                str(row.sortie_id),
                int(row.start_offset_ms),
                int(row.end_offset_ms) + int(target_horizon_ms),
            )
        )
    return rows
