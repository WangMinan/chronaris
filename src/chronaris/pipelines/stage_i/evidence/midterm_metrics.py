"""Shared metric extraction helpers for Stage I midterm evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd


def _primary_task_metric(task_id: str, payload: Mapping[str, object]) -> tuple[str, float]:
    if task_id == "T1_maneuver_intensity_class":
        best_metrics = payload["best_metrics"]
        return "macro_f1", float(best_metrics["macro_f1"])
    if task_id == "T2_next_window_physiology_response":
        best_metrics = payload["best_metrics"]
        return "rmse", float(best_metrics["rmse"])
    return "top1_accuracy", float(payload["top1_accuracy"])


def _extract_uab_fairness_metrics(
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float | str]]:
    historical_groups = (
        sources["deep_comparison"]["payload"]["datasets"]["uab_workload_dataset"]["models"]["contiformer"]["summary"]["subjective"]["groups"]
    )
    if "uab_fairness" not in sources:
        return {
            subset_id: {
                "rmse": float(metrics["rmse"]),
                "historical_rmse": float(metrics["rmse"]),
                "source_path": sources["deep_comparison"]["path"],
            }
            for subset_id, metrics in historical_groups.items()
        }
    fairness_summary = sources["uab_fairness"]["payload"]
    confirm_groups = fairness_summary["subjective"]["groups"]
    return {
        subset_id: {
            "rmse": float(confirm_groups[subset_id]["rmse"]),
            "historical_rmse": float(historical_groups[subset_id]["rmse"]),
            "source_path": sources["uab_fairness"]["path"],
        }
        for subset_id in historical_groups
    }


def _extract_nasa_fusion_metrics(
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, float | str]:
    key = "nasa_fusion_confirm" if "nasa_fusion_confirm" in sources else "public_fusion_screen"
    payload = sources[key]["payload"]
    if "per_dataset_rankings" in payload:
        best_row = payload["per_dataset_rankings"]["nasa_csm"][0]
        return {
            "candidate_id": best_row["candidate_id"],
            "macro_f1": float(best_row["selection_score"]),
            "balanced_accuracy": float(best_row["secondary_score"]),
            "source_path": sources[key]["path"],
            "runtime_device": str(payload.get("runtime_device", "unknown")),
            "source_type": "fusion_screen",
        }
    combined = payload["objective"]["groups"]["combined"]
    return {
        "candidate_id": str(Path(str(payload["artifact_root"])).name),
        "macro_f1": float(combined["macro_f1"]),
        "balanced_accuracy": float(combined["balanced_accuracy"]),
        "source_path": sources[key]["path"],
        "runtime_device": str(payload.get("runtime_device", "unknown")),
        "source_type": "full_confirm",
    }


def _infer_timestamp(payload: Mapping[str, object], source_path: str) -> pd.Timestamp | None:
    candidates = [
        payload.get("generated_at_utc"),
        payload.get("generated_at"),
        payload.get("run_id"),
        Path(source_path).parent.name,
        Path(source_path).name,
    ]
    for candidate in candidates:
        timestamp = _parse_timestamp_candidate(candidate)
        if timestamp is not None:
            return timestamp
    return pd.Timestamp(Path(source_path).stat().st_mtime, unit="s", tz="UTC")


def _parse_timestamp_candidate(candidate: object) -> pd.Timestamp | None:
    if not candidate or not isinstance(candidate, str):
        return None
    if candidate.endswith("Z") and "T" in candidate:
        try:
            return pd.Timestamp(candidate)
        except Exception:
            pass
    match = re.search(r"(20\d{6}T\d{6}Z)", candidate)
    if not match:
        return None
    try:
        return pd.Timestamp(match.group(1))
    except Exception:
        return None
