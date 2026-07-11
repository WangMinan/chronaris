"""Observed-only fold input for Dingxin common pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from chronaris.evaluation.application_tasks.dingxin_context_data import (
    DingxinLazyContextIndex,
    build_dingxin_lazy_context_index,
    dingxin_vehicle_field_labels,
)
from chronaris.representation import FoldLineage
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class DingxinFoldPretrainingData:
    index: DingxinLazyContextIndex
    fold: FoldLineage
    vehicle_field_labels: tuple[tuple[str, str], ...]
    source_hashes: dict[str, str]


def load_dingxin_fold_pretraining_data(
    *,
    fold_id: str,
    snapshot_root: str | Path,
    fixed_audit_root: str | Path,
    inner_split_root: str | Path,
) -> DingxinFoldPretrainingData:
    fixed_root = Path(fixed_audit_root)
    split_root = Path(inner_split_root)
    paths = {
        "field_role_manifest": fixed_root / "field_role_manifest.csv",
        "context_manifest": fixed_root / "context_sample_manifest.jsonl",
        "inner_split_manifest": split_root / "split_manifest.json",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Dingxin fold pretraining sources missing: {missing}")
    split_payload = json.loads(
        paths["inner_split_manifest"].read_text(encoding="utf-8")
    )
    candidates = [
        item for item in split_payload["folds"] if str(item["fold_id"]) == fold_id
    ]
    if len(candidates) != 1:
        raise ValueError(f"Dingxin inner split has no unique fold: {fold_id}")
    item = candidates[0]
    fold = FoldLineage(
        fold_id=fold_id,
        train_sample_ids=tuple(str(value) for value in item["train_sample_ids"]),
        validation_sample_ids=tuple(
            str(value) for value in item["validation_sample_ids"]
        ),
        held_out_sample_ids=tuple(
            str(value) for value in item["held_out_sample_ids"]
        ),
    )
    index = build_dingxin_lazy_context_index(
        snapshot_root=snapshot_root,
        field_role_manifest_path=paths["field_role_manifest"],
        context_manifest_path=paths["context_manifest"],
    )
    expected = set(
        fold.train_sample_ids
        + fold.validation_sample_ids
        + fold.held_out_sample_ids
    )
    available = set(
        index.contexts[index.contexts["input_fully_observed"]]["context_id"].astype(str)
    )
    missing_contexts = sorted(expected - available)
    if missing_contexts:
        raise ValueError(
            f"Dingxin fold contains unavailable raw contexts: {missing_contexts[:5]}"
        )
    return DingxinFoldPretrainingData(
        index=index,
        fold=fold,
        vehicle_field_labels=dingxin_vehicle_field_labels(
            index,
            field_role_manifest_path=paths["field_role_manifest"],
        ),
        source_hashes={name: sha256_file(path) for name, path in paths.items()},
    )
