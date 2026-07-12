"""Locked three-seed G1 retraining for the selected Chronaris v2 config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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
from chronaris.modeling.training import (
    ChronarisV2TrainingConfig,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


@dataclass(frozen=True, slots=True)
class ChronarisV2LockedSimulationConfig:
    run_id: str = "2026-07-13_chronaris-v2-simulation-locked-pretraining"
    locked_configuration_path: str = (
        "docs/artifacts/runs/"
        "2026-07-12_chronaris-v2-dingxin-inner-confirmation-r3/"
        "locked_configuration.json"
    )
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    max_epochs: int = 50
    batch_size: int = 128
    device: str = "cuda"
    resume: bool = True


def run_chronaris_v2_locked_simulation_pretraining(
    config: ChronarisV2LockedSimulationConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2LockedSimulationConfig()
    candidate, _lock = _load_locked_candidate(resolved.locked_configuration_path)
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    data = load_simulation_locked_pretraining_data(resolved.simulation_root)
    normalizer = TrainOnlyRobustNormalizer().fit(
        data.batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=(
            data.fold.validation_sample_ids + data.fold.held_out_sample_ids
        ),
    )
    rows = []
    for seed in resolved.seeds:
        result = train_chronaris_v2_candidate(
            candidate=candidate,
            batch=data.batch,
            fold=data.fold,
            physiology_feature_names=data.schema.physiology_feature_names,
            vehicle_feature_names=data.schema.vehicle_feature_names,
            vehicle_field_labels=tuple(
                (name, name) for name in data.schema.vehicle_feature_names
            ),
            normalizer=normalizer,
            output_root=(
                heavy_root / "checkpoints" / f"seed_{seed}" / "chronaris"
            ),
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
        rows.append(
            {
                "seed": seed,
                "candidate_id": candidate.candidate_id,
                "status": result.status,
                "completed_epochs": result.completed_epochs,
                "best_epoch": result.best_epoch,
                "public_self_supervised_validation_loss": (
                    result.best_public_selection_loss
                ),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "task_labels_opened": bool(
                    payload["label_used_for_encoder_training"]
                ),
                "simulation_oracle_opened": bool(
                    payload["simulation_oracle_opened"]
                ),
                "locked_test_opened": bool(payload["locked_test_opened"]),
            }
        )
    acceptance = (
        _check("three_seed_checkpoints", len(rows) == len(resolved.seeds)),
        _check(
            "all_training_complete",
            all(row["status"] in {"completed", "resumed"} for row in rows),
        ),
        _check(
            "forbidden_sources_closed",
            all(
                not row["task_labels_opened"]
                and not row["simulation_oracle_opened"]
                and not row["locked_test_opened"]
                for row in rows
            ),
        ),
    )
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"
    pd.DataFrame(rows).to_csv(compact_root / "checkpoint_inventory.csv", index=False)
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(
        compact_root / "protocol.json",
        {
            "format": "chronaris.v2_simulation_locked_pretraining_protocol.v1",
            "config": asdict(resolved),
            "locked_candidate": asdict(candidate),
            "locked_configuration_sha256": _sha256_file(
                Path(resolved.locked_configuration_path)
            ),
            "representation_family": "frozen_task_agnostic_v2",
            "task_oracle_opened": False,
            "sealed_confirmation_opened": False,
        },
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_simulation_locked_pretraining_evidence.v1",
            "run_id": resolved.run_id,
            "status": status,
            "checkpoint_count": len(rows),
            "heavy_run_root": str(heavy_root),
            "confirmed_metrics_changed": False,
        },
    )
    return compact_root


def _check(name: str, passed: bool):
    return {"check": name, "passed": bool(passed)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
