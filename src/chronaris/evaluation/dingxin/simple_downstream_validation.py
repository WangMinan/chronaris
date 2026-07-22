"""Independent validation helpers for the simplified Dingxin report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score

from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def summary_max_difference(metrics: pd.DataFrame, published: pd.DataFrame) -> float:
    """Recompute the published aggregate table and return its maximum difference."""
    computed = (
        metrics.groupby(["method_name", "task_name", "metric_name"], as_index=False)
        .agg(
            mean=("metric_value", "mean"),
            std=("metric_value", "std"),
            minimum=("metric_value", "min"),
            maximum=("metric_value", "max"),
            unit_count=("metric_value", "count"),
        )
        .sort_values(["method_name", "task_name", "metric_name"])
        .reset_index(drop=True)
    )
    expected = published.sort_values(
        ["method_name", "task_name", "metric_name"]
    ).reset_index(drop=True)
    index_columns = ["method_name", "task_name", "metric_name"]
    if computed[index_columns].to_dict("records") != expected[index_columns].to_dict(
        "records"
    ):
        return float("inf")
    value_columns = ["mean", "std", "minimum", "maximum", "unit_count"]
    return float(
        np.nanmax(
            np.abs(
                computed[value_columns].to_numpy()
                - expected[value_columns].to_numpy()
            )
        )
    )


def validate_raw_predictions(metrics: pd.DataFrame, inventory: pd.DataFrame) -> dict:
    """Verify prediction hashes and independently recompute core unit metrics."""
    lookup = metrics.set_index(
        ["seed", "fold_id", "method_name", "task_name", "metric_name"]
    )["metric_value"]
    max_difference = 0.0
    checked_metric_count = 0
    hash_count = 0
    for row in inventory.itertuples(index=False):
        for path_field, hash_field in (
            ("maneuver_predictions_path", "maneuver_predictions_sha256"),
            ("physiology_predictions_path", "physiology_predictions_sha256"),
            ("physiology_field_metrics_path", "physiology_field_metrics_sha256"),
        ):
            if sha256_file(Path(getattr(row, path_field))) != getattr(row, hash_field):
                return {"hashes_match": False, "max_metric_difference": float("inf")}
            hash_count += 1

        maneuver = pd.read_csv(row.maneuver_predictions_path)
        truth = maneuver["future_maneuver_score"].to_numpy(dtype=float)
        predicted = maneuver["score_prediction"].to_numpy(dtype=float)
        current = maneuver["current_maneuver_score"].to_numpy(dtype=float)
        maneuver_values = {
            "independent_vehicle_context_count": len(maneuver),
            "spearman": spearmanr(truth, predicted).statistic,
            "normalized_mae": np.mean(np.abs(truth - predicted))
            / float(maneuver["train_target_iqr"].iloc[0]),
            "skill_vs_current_maneuver": 1.0
            - np.sum((truth - predicted) ** 2) / np.sum((truth - current) ** 2),
            "macro_f1": f1_score(
                maneuver["future_maneuver_class"],
                maneuver["class_prediction"],
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                maneuver["future_maneuver_class"], maneuver["class_prediction"]
            ),
        }

        physiology = pd.read_csv(row.physiology_predictions_path)
        fields = pd.read_csv(row.physiology_field_metrics_path)
        residual = physiology["future_standardized"] - physiology["prediction_standardized"]
        persistence = physiology["future_standardized"] - physiology["current_standardized"]
        physiology_values = {
            "view_context_count": physiology["context_id"].nunique(),
            "field_count": physiology["field_name"].nunique(),
            "standardized_rmse_macro": fields["rmse"].mean(),
            "standardized_mae_macro": fields["mae"].mean(),
            "skill_vs_persistence": 1.0
            - np.sum(residual**2) / np.sum(persistence**2),
            "positive_skill_field_ratio": (fields["skill_vs_persistence"] > 0).mean(),
            "eeg_rmse_macro": fields.loc[
                fields["semantic_category"] == "eeg", "rmse"
            ].mean(),
            "spo2_rmse_macro": fields.loc[
                fields["semantic_category"] == "spo2", "rmse"
            ].mean(),
        }
        for task_name, values in (
            ("future_maneuver", maneuver_values),
            ("future_physiology", physiology_values),
        ):
            for metric_name, value in values.items():
                expected = float(
                    lookup.loc[
                        (
                            row.seed,
                            row.fold_id,
                            row.method_name,
                            task_name,
                            metric_name,
                        )
                    ]
                )
                max_difference = max(max_difference, abs(float(value) - expected))
                checked_metric_count += 1
    return {
        "hashes_match": True,
        "hash_count": hash_count,
        "checked_metric_count": checked_metric_count,
        "max_metric_difference": max_difference,
    }


def validate_task_targets(task_root: Path) -> dict:
    """Validate context counts, future boundaries, and held-out class support."""
    context = pd.read_csv(task_root / "context_manifest.csv")
    maneuver = pd.read_csv(task_root / "maneuver_targets.csv")
    boundaries = bool(
        (context["input_end_exclusive_ms"] == context["target_start_offset_ms"]).all()
        and (
            (context["target_end_exclusive_ms"] - context["target_start_offset_ms"])
            == 5_000
        ).all()
        and (
            context["target_end_exclusive_ms"] <= context["snapshot_stop_offset_ms"]
        ).all()
    )
    support = {}
    for fold_id, fold in maneuver[maneuver["split_role"] == "held_out"].groupby(
        "fold_id"
    ):
        unique = fold.drop_duplicates("vehicle_context_id")
        support[str(fold_id)] = {
            str(int(label)): int(count)
            for label, count in unique["future_maneuver_class"]
            .value_counts()
            .sort_index()
            .items()
        }
    return {
        "view_context_count": int(context["context_id"].nunique()),
        "vehicle_context_count": int(context["vehicle_context_id"].nunique()),
        "strict_future_boundaries": boundaries,
        "held_out_class_support": support,
    }


def validate_protocol_chain(compact: Path, config: Any) -> dict:
    """Check that targets and outer metrics stayed closed until confirmation."""
    pretraining = _read_json(compact / config.pretraining_run_id / "protocol.json")
    representations = _read_json(
        compact / config.representation_run_id / "protocol.json"
    )
    confirmation = _read_json(compact / config.confirmation_run_id / "protocol.json")
    return {
        "pretraining_targets_closed": pretraining.get("task_targets_opened") is False,
        "pretraining_outer_test_closed": pretraining.get("outer_test_accessed") is False,
        "representation_targets_closed": representations.get("task_targets_opened")
        is False,
        "representation_metrics_closed": representations.get("outer_metrics_opened")
        is False,
        "formal_result_opened": confirmation.get("outer_metrics_opened") is True,
        "result_driven_tuning_disabled": confirmation.get(
            "result_driven_tuning_allowed"
        )
        is False,
    }


def build_source_inventory(formal: Path, prior: Path, config: Any) -> pd.DataFrame:
    """Hash every compact input that directly supports the report."""
    paths = (
        formal / "metrics_long.csv",
        formal / "metrics_summary.csv",
        formal / "prediction_inventory.csv",
        formal / "protocol.json",
        Path(config.compact_output_root) / config.pretraining_run_id / "protocol.json",
        Path(config.compact_output_root) / config.representation_run_id / "protocol.json",
        prior / "tables" / "mechanism_recovery_mae.csv",
        prior / "tables" / "stress_degradation_slopes.csv",
        prior / "tables" / "chronaris_ablation_advantage.csv",
    )
    return pd.DataFrame(
        {
            "source_path": [str(path) for path in paths],
            "sha256": [sha256_file(path) for path in paths],
        }
    )


def build_acceptance_rows(**values: Any) -> tuple[dict, ...]:
    """Build the bounded pass/fail summary for the report run."""
    units = values["units"]
    metrics = values["metrics"]
    raw = values["raw_validation"]
    target = values["target_validation"]
    protocol = values["protocol_validation"]
    support_ok = all(
        value == {"0": 10, "1": 10, "2": 10}
        for value in target["held_out_class_support"].values()
    )
    return (
        _check(
            "formal_units_complete",
            len(units) == 36 and set(units["status"]) == {"completed"},
            len(units),
            36,
        ),
        _check(
            "formal_metrics_finite",
            len(metrics) == 504 and np.isfinite(metrics["metric_value"]).all(),
            len(metrics),
            504,
        ),
        _check(
            "published_summary_recomputed",
            values["summary_difference"] <= 1e-12,
            values["summary_difference"],
            "<=1e-12",
        ),
        _check(
            "raw_prediction_metrics_recomputed",
            raw["hashes_match"] and raw["max_metric_difference"] <= 1e-10,
            raw,
            "all hashes and <=1e-10",
        ),
        _check(
            "future_context_contract",
            target["view_context_count"] == 90
            and target["vehicle_context_count"] == 60
            and target["strict_future_boundaries"],
            target,
            "90 views, 60 vehicles, strict future",
        ),
        _check(
            "held_out_class_support",
            support_ok,
            target["held_out_class_support"],
            {"0": 10, "1": 10, "2": 10},
        ),
        _check(
            "protocol_chain_closed_before_confirmation",
            all(protocol.values()),
            protocol,
            "all true",
        ),
    )


def _check(check_id: str, passed: bool, actual: Any, expected: Any) -> dict:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
