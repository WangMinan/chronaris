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
from chronaris.representation import (
    DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY,
    FoldLineage,
    coalesce_observation_batch,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


DINGXIN_MODEL_INPUT_BIN_WIDTH_S = 0.1


def ensure_dingxin_model_input_contract(
    root: str | Path,
    *,
    maneuver_history_policy: str = DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY,
) -> Path:
    root = Path(root)
    path = root / "model_input_contract.json"
    expected = {
        "format": (
            "chronaris.dingxin_model_input_contract.v1"
            if maneuver_history_policy == DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY
            else "chronaris.dingxin_model_input_contract.v2"
        ),
        "model_input_bin_width_s": DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
        "causal_bin_timestamp": "last_real_observation",
        "duplicate_feature_reducer": "arithmetic_mean",
    }
    if maneuver_history_policy != DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY:
        expected["maneuver_history_policy"] = maneuver_history_policy
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("Dingxin model-input temporal contract changed")
        return path
    if any(root.rglob("*.pt")):
        raise ValueError(
            "Dingxin checkpoints predate the temporal-coalescing contract; "
            "use a new run_id instead of mixing model inputs"
        )
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True, slots=True)
class DingxinFoldPretrainingData:
    index: DingxinLazyContextIndex
    fold: FoldLineage
    vehicle_field_labels: tuple[tuple[str, str], ...]
    source_hashes: dict[str, str]

    def load_batch(self, sample_ids):
        return coalesce_observation_batch(
            self.index.load_batch(sample_ids),
            bin_width_s=DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
        )


def load_dingxin_fold_pretraining_data(
    *,
    fold_id: str,
    snapshot_root: str | Path,
    fixed_audit_root: str | Path,
    inner_split_root: str | Path,
    maneuver_history_policy: str = DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY,
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
        maneuver_history_policy=maneuver_history_policy,
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
