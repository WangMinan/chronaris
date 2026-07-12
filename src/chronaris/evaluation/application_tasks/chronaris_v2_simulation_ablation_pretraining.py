"""Locked G1 retraining for isolatable Chronaris v2 mechanism ablations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.chronaris_v2_locked_dingxin_pretraining import (
    _load_locked_candidate,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.training import ChronarisV2TrainingConfig, train_chronaris_v2_candidate
from chronaris.representation import TrainOnlyRobustNormalizer


V2_FORMAL_ABLATION_STRUCTURES = {
    "no_corrected_physics": "structure_07_missing_curriculum",
    "no_missingness_curriculum": "structure_06_corrected_physics",
}


@dataclass(frozen=True, slots=True)
class ChronarisV2SimulationAblationConfig:
    run_id: str = "2026-07-13_chronaris-v2-simulation-ablation-pretraining"
    locked_configuration_path: str = (
        "docs/artifacts/runs/2026-07-12_chronaris-v2-dingxin-inner-"
        "confirmation-r3/locked_configuration.json"
    )
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    variants: tuple[str, ...] = tuple(V2_FORMAL_ABLATION_STRUCTURES)
    max_epochs: int = 50
    batch_size: int = 128
    device: str = "cuda"
    resume: bool = True

    def __post_init__(self):
        if not self.variants or not set(self.variants).issubset(V2_FORMAL_ABLATION_STRUCTURES):
            raise ValueError("unsupported Chronaris v2 formal ablation")


def run_chronaris_v2_simulation_ablation_pretraining(
    config: ChronarisV2SimulationAblationConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2SimulationAblationConfig()
    locked_candidate, _lock = _load_locked_candidate(resolved.locked_configuration_path)
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    data = load_simulation_locked_pretraining_data(resolved.simulation_root)
    normalizer = TrainOnlyRobustNormalizer().fit(
        data.batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=data.fold.validation_sample_ids + data.fold.held_out_sample_ids,
    )
    rows = []
    for seed in resolved.seeds:
        for variant in resolved.variants:
            structure = V2_FORMAL_ABLATION_STRUCTURES[variant]
            candidate = replace(
                locked_candidate,
                candidate_id=f"{locked_candidate.candidate_id}__{variant}",
                structure_candidate_id=structure,
            )
            result = train_chronaris_v2_candidate(
                candidate=candidate,
                batch=data.batch,
                fold=data.fold,
                physiology_feature_names=data.schema.physiology_feature_names,
                vehicle_feature_names=data.schema.vehicle_feature_names,
                vehicle_field_labels=tuple((name, name) for name in data.schema.vehicle_feature_names),
                normalizer=normalizer,
                output_root=heavy_root / "checkpoints" / f"seed_{seed}" / variant / "chronaris",
                config=ChronarisV2TrainingConfig(
                    max_epochs=resolved.max_epochs,
                    batch_size=resolved.batch_size,
                    patience=min(8, resolved.max_epochs),
                    seed=seed,
                    device=resolved.device,
                ),
                resume=resolved.resume,
            )
            checkpoint = Path(result.last_checkpoint_path)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            rows.append({
                "seed": seed,
                "variant": variant,
                "structure_candidate_id": structure,
                "candidate_id": candidate.candidate_id,
                "status": result.status,
                "completed_epochs": result.completed_epochs,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "task_labels_opened": bool(payload["label_used_for_encoder_training"]),
                "simulation_oracle_opened": bool(payload["simulation_oracle_opened"]),
                "sealed_confirmation_opened": bool(payload["locked_test_opened"]),
            })
    expected = len(resolved.seeds) * len(resolved.variants)
    acceptance = (
        _check("all_variant_seed_checkpoints", len(rows) == expected),
        _check("all_training_complete", all(row["status"] in {"completed", "resumed"} for row in rows)),
        _check("forbidden_sources_closed", all(
            not row["task_labels_opened"]
            and not row["simulation_oracle_opened"]
            and not row["sealed_confirmation_opened"] for row in rows
        )),
    )
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"
    pd.DataFrame(rows).to_csv(compact_root / "checkpoint_inventory.csv", index=False)
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(compact_root / "protocol.json", {
        "format": "chronaris.v2_simulation_ablation_pretraining_protocol.v1",
        "config": asdict(resolved),
        "locked_candidate": asdict(locked_candidate),
        "isolated_comparisons": {
            "corrected_physics": ["structure_08_complete_v2", "structure_07_missing_curriculum"],
            "missingness_curriculum": ["structure_08_complete_v2", "structure_06_corrected_physics"],
        },
        "downstream_results_used_for_configuration": False,
    })
    _write_json(compact_root / "evidence_manifest.json", {
        "format": "chronaris.v2_simulation_ablation_pretraining_evidence.v1",
        "run_id": resolved.run_id,
        "status": status,
        "checkpoint_count": len(rows),
        "heavy_run_root": str(heavy_root),
        "confirmed_v1_evidence_changed": False,
    })
    return compact_root


def _check(name, passed):
    return {"check": name, "passed": bool(passed)}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
