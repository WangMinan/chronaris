"""Lock the development winner and authorize one outer confirmation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.core_recovery_protocol import (
    CORE_RECOVERY_METHODS,
    OuterTestAccessGuard,
    sha256_file,
    stable_mapping_sha256,
)


@dataclass(frozen=True, slots=True)
class CoreRecoveryLockConfig:
    protocol_root: str = "docs/artifacts/runs/2026-07-13_chronaris-core-task-recovery"
    development_root: str = (
        "docs/artifacts/runs/2026-07-13_chronaris-core-task-recovery-development"
    )
    nested_target_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-nested-targets/nested_targets.csv"
    )


def lock_core_recovery_development(config: CoreRecoveryLockConfig):
    protocol_root = Path(config.protocol_root)
    development_root = Path(config.development_root)
    protocol_path = protocol_root / "protocol_lock.json"
    development_path = development_root / "development_result.json"
    summary_path = development_root / "consumer_route_summary.csv"
    nested_target_path = Path(config.nested_target_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development.get("status") != "completed":
        raise ValueError("development result is incomplete")
    if not bool(development.get("development_gate_passed")):
        raise PermissionError("development gate failed; outer confirmation stays closed")
    rows = pd.read_csv(summary_path)
    selected = {
        method: {
            task: _select_best(rows, method=method, task=task)
            for task in ("maneuver", "response", "high_response")
        }
        for method in CORE_RECOVERY_METHODS
    }
    payload = {
        "format": "chronaris.core_task_recovery_locked_configuration.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_path": str(protocol_path),
        "development_result_path": str(development_path),
        "development_result_sha256": sha256_file(development_path),
        "consumer_summary_path": str(summary_path),
        "consumer_summary_sha256": sha256_file(summary_path),
        "nested_target_path": str(nested_target_path),
        "nested_target_sha256": sha256_file(nested_target_path),
        "methods": selected,
        "confirmation_seeds": [17, 29, 43],
        "confirmation_folds": [
            "leave_one_view_out__fold01",
            "leave_one_view_out__fold02",
            "leave_one_view_out__fold03",
        ],
        "task_definition": development["task_definition"],
        "hard_confirmation_thresholds": {
            "maneuver_macro_f1": {"direction": "higher", "strict": 0.9409316261641733},
            "response_rmse": {"direction": "lower", "strict": 0.2912},
            "high_response_auprc": {"direction": "higher", "strict": 0.8839},
            "all_three_rank_first": True,
        },
        "outer_test_open_count": 1,
    }
    payload["locked_configuration_sha256"] = stable_mapping_sha256(payload)
    lock_path = protocol_root / "locked_configuration.json"
    lock_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    guard = OuterTestAccessGuard(protocol_sha256=protocol["protocol_sha256"])
    authorization = guard.authorize(
        locked_configuration_sha256=payload["locked_configuration_sha256"],
        development_gate_passed=True,
    )
    authorization_path = protocol_root / "outer_test_access_lock.json"
    authorization_path.write_text(
        json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "locked",
        "locked_configuration_path": str(lock_path),
        "locked_configuration_sha256": payload["locked_configuration_sha256"],
        "authorization_path": str(authorization_path),
        "authorized_open_count": 1,
    }


def _select_best(rows: pd.DataFrame, *, method: str, task: str):
    candidates = rows[
        (rows["method_name"].astype(str) == method)
        & (rows["task"].astype(str) == task)
        & (rows["completed_fold_count"] == rows["required_fold_count"])
    ].copy()
    if candidates.empty:
        raise ValueError(f"no complete development consumer for {method}/{task}")
    ascending = str(candidates.iloc[0]["direction"]) == "lower"
    candidates = candidates.sort_values(
        ["mean_value", "n_kernels", "head_config_id"],
        ascending=[ascending, True, True],
        kind="mergesort",
    )
    row = candidates.iloc[0]
    return {
        "sequence_mode": str(row["sequence_mode"]),
        "n_kernels": int(row["n_kernels"]),
        "head_config_id": str(row["head_config_id"]),
        "development_mean": float(row["mean_value"]),
        "development_worst_fold": float(row["worst_value"]),
        "direction": str(row["direction"]),
    }
