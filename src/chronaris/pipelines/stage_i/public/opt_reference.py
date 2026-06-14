"""Reference-loading helpers for Stage I public-opt reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from chronaris.dataset import (
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    load_stage_i_sequence_summary,
)


def load_public_opt_prepared_dataset(artifact_root: str | Path) -> dict[str, object]:
    root = Path(artifact_root)
    entries = load_stage_i_sequence_entries(root / "task_manifest.jsonl")
    bundle = load_stage_i_sequence_bundle(root / "sequence_bundle.npz")
    summary = load_stage_i_sequence_summary(root / "dataset_summary.json")
    schema = json.loads((root / "sequence_schema.json").read_text(encoding="utf-8"))
    return {
        "artifact_root": str(root),
        "dataset_id": summary.dataset_id,
        "entries": entries,
        "bundle": bundle,
        "summary": summary.to_dict(),
        "schema": schema,
    }


def validate_public_opt_prepared_dataset_contract(
    prepared: Mapping[str, object],
    *,
    dataset_id: str,
) -> None:
    """Fail fast on stale public prepared assets before long runs start."""

    if dataset_id != "nasa_csm":
        return
    schema = prepared.get("schema") or {}
    if not isinstance(schema, Mapping):
        raise ValueError("NASA prepared asset is missing sequence_schema.json.")
    modalities = tuple((schema.get("modalities") or {}).keys())
    expected_modalities = ("physiology", "scenario_context")
    if modalities != expected_modalities:
        raise ValueError(
            "NASA prepared asset must use Chronaris-public modalities "
            f"{expected_modalities}, got {modalities}. Rebuild NASA sequence assets."
        )
    guard = schema.get("label_leakage_guard")
    if not isinstance(guard, Mapping):
        raise ValueError(
            "NASA prepared asset is missing label_leakage_guard. "
            "Rebuild NASA sequence assets before public-opt/public-fusion runs."
        )
    context_feature_names = tuple(
        str(value) for value in guard.get("context_feature_names", ())
    )
    forbidden = {"event_code", "objective_label_text"}
    leaked_features = tuple(
        name
        for name in context_feature_names
        if any(forbidden_name in name for forbidden_name in forbidden)
    )
    if leaked_features:
        raise ValueError(
            "NASA prepared asset leaks label-like context features: "
            + ", ".join(leaked_features)
        )
    entries = tuple(prepared.get("entries") or ())
    for entry in entries:
        leaked_context = sorted(forbidden.intersection(entry.context_payload))
        if leaked_context:
            raise ValueError(
                "NASA prepared asset context_payload contains label-like fields: "
                + ", ".join(leaked_context)
            )


def build_public_opt_reference_comparison(
    *,
    dataset_id: str,
    track: str,
    subset_results: Mapping[str, object],
    phase3_closure_summary_path: str | None,
    deep_comparison_summary_path: str | None,
) -> dict[str, object]:
    classical = _load_classical_reference(
        dataset_id=dataset_id,
        track=track,
        phase3_closure_summary_path=phase3_closure_summary_path,
    )
    deep = _load_deep_reference(
        dataset_id=dataset_id,
        track=track,
        deep_comparison_summary_path=deep_comparison_summary_path,
    )
    if not classical and not deep:
        return {}
    groups: dict[str, object] = {}
    for group_name, payload in subset_results.items():
        groups[group_name] = {
            "public_opt": {
                "best_head": payload["best_head"],
                "best_metrics": payload["heads"][payload["best_head"]],
            },
            "classical": classical.get(group_name),
            "deep_models": deep.get(group_name, {}),
        }
    return {
        "dataset_id": dataset_id,
        "track": track,
        "groups": groups,
    }


def evaluate_public_opt_winning_margins(
    *,
    track: str,
    reference_comparison: Mapping[str, object],
    policy: str,
) -> tuple[dict[str, object], bool]:
    if policy == "none" or not reference_comparison:
        return {}, False
    primary_field = "rmse" if track == "subjective" else "macro_f1"
    groups: dict[str, object] = {}
    needs_deep_rerun = False
    for group_name, payload in reference_comparison.get("groups", {}).items():
        deep_models = payload.get("deep_models") or {}
        public_metrics = payload["public_opt"]["best_metrics"]
        deep_primary = {
            model_name: float(metrics.get(primary_field, 0.0))
            for model_name, metrics in deep_models.items()
        }
        if not deep_primary:
            continue
        if track == "subjective":
            best_deep_name, best_deep_value = min(
                deep_primary.items(),
                key=lambda item: item[1],
            )
            public_value = float(public_metrics.get(primary_field, 0.0))
            margin = best_deep_value - public_value
            threshold = max(0.02, 0.01 * best_deep_value)
            gate_passed = margin > 0.0
            group_needs_rerun = abs(margin) < threshold
        else:
            best_deep_name, best_deep_value = max(
                deep_primary.items(),
                key=lambda item: item[1],
            )
            public_value = float(public_metrics.get(primary_field, 0.0))
            margin = public_value - best_deep_value
            threshold = 0.005
            gate_passed = margin > 0.0
            group_needs_rerun = abs(margin) < threshold
        groups[group_name] = {
            "best_public_head": payload["public_opt"]["best_head"],
            "best_public_value": public_value,
            "best_deep_model": best_deep_name,
            "best_deep_value": best_deep_value,
            "margin_vs_best_deep": margin,
            "gate_passed": gate_passed,
            "rerun_threshold": threshold,
            "needs_deep_rerun": group_needs_rerun,
        }
        needs_deep_rerun = needs_deep_rerun or group_needs_rerun
    return groups, needs_deep_rerun


def _load_classical_reference(
    *,
    dataset_id: str,
    track: str,
    phase3_closure_summary_path: str | None,
) -> dict[str, object]:
    if not phase3_closure_summary_path:
        return {}
    path = Path(phase3_closure_summary_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if dataset_id == "uab_workload_dataset":
        branch = payload.get("uab_window", {})
        if track == "subjective":
            return dict(branch.get("subjective_primary", {}))
        return dict(branch.get("objective_primary", {}))
    if dataset_id == "nasa_csm":
        branch = payload.get("nasa_attention", {})
        return dict(branch.get("objective_primary", {}))
    return {}


def _load_deep_reference(
    *,
    dataset_id: str,
    track: str,
    deep_comparison_summary_path: str | None,
) -> dict[str, dict[str, object]]:
    if not deep_comparison_summary_path:
        return {}
    path = Path(deep_comparison_summary_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_payload = payload.get("datasets", {}).get(dataset_id)
    if not dataset_payload or dataset_payload.get("status") != "completed":
        return {}
    groups: dict[str, dict[str, object]] = {}
    for model_name, model_payload in dataset_payload.get("models", {}).items():
        summary = model_payload.get("summary", {})
        track_payload = summary.get(track) or {}
        for group_name, metrics in track_payload.get("groups", {}).items():
            groups.setdefault(group_name, {})[model_name] = metrics
    return groups
