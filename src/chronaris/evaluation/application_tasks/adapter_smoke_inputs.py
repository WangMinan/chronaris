"""Shared observed-only dataset loader for production adapter smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from chronaris.representation import (
    build_dingxin_observation_schema_plan,
    collate_observation_samples,
    load_dingxin_observed_context,
    load_simulation_observed_context,
)


def load_adapter_smoke_datasets(
    *,
    simulation_root: str,
    dingxin_snapshot_root: str,
    field_role_manifest_path: str,
    context_manifest_path: str,
):
    simulation_paths = select_simulation_observed_paths(Path(simulation_root))
    simulation_samples = tuple(
        load_simulation_observed_context(path, context_start_s=0.0)
        for path in simulation_paths.values()
    )
    dingxin_plan = build_dingxin_observation_schema_plan(
        snapshot_root=dingxin_snapshot_root,
        field_role_manifest_path=field_role_manifest_path,
    )
    dingxin_contexts = one_eligible_context_per_view(Path(context_manifest_path))
    dingxin_samples = tuple(
        load_dingxin_observed_context(dingxin_plan, context)
        for context in dingxin_contexts
    )
    datasets = {
        "simulation": collate_observation_samples(simulation_samples),
        "dingxin": collate_observation_samples(dingxin_samples),
    }
    metadata = {
        "simulation": {
            "sample_count": len(simulation_samples),
            "physiology_feature_count": len(
                simulation_samples[0].schema.physiology_feature_names
            ),
            "vehicle_feature_count": len(
                simulation_samples[0].schema.vehicle_feature_names
            ),
            "oracle_opened": False,
        },
        "dingxin": {
            "sample_count": len(dingxin_samples),
            "view_count": len({sample.group_id for sample in dingxin_samples}),
            "physiology_feature_count": len(
                dingxin_plan.schema.physiology_feature_names
            ),
            "vehicle_feature_count": len(dingxin_plan.schema.vehicle_feature_names),
            "excluded_label_source_count": len(
                dingxin_plan.schema.excluded_feature_names
            ),
            "label_or_target_opened": False,
        },
    }
    return datasets, metadata


def select_simulation_observed_paths(root: Path) -> Mapping[str, Path]:
    selected = {}
    for split_id in ("train", "validation", "locked_test"):
        candidates = sorted(
            (root / split_id).glob("**/clean_asynchronous/raw_dual_stream.npz")
        )
        if not candidates:
            raise FileNotFoundError(
                f"no clean simulation observed archive for split {split_id} under {root}"
            )
        selected[split_id] = candidates[0]
    return selected


def one_eligible_context_per_view(path: Path) -> tuple[Mapping[str, object], ...]:
    by_view = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("classification_eligible") or not payload.get(
            "response_eligible"
        ):
            continue
        by_view.setdefault(str(payload["view_id"]), payload)
    if len(by_view) < 3:
        raise ValueError("adapter smoke requires three distinct Dingxin views")
    return tuple(by_view[key] for key in sorted(by_view)[:3])
