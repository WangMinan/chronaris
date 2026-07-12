"""Resumable 24-candidate Chronaris v2 hyperparameter screen."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.training import (
    ChronarisV2TrainingConfig,
    chronaris_v2_hyperparameter_grid,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


@dataclass(frozen=True, slots=True)
class ChronarisV2HyperparameterScreenConfig:
    run_id: str = "2026-07-12_chronaris-v2-hyperparameter-screen-seed17"
    architecture_gate_run_id: str = (
        "2026-07-12_chronaris-v2-direct-residual-repair-seed17-r1"
    )
    architecture_gate_candidate_id: str = "direct_residual_01_causal_query"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    device: str = "cuda"
    max_epochs: int = 50
    batch_size: int = 128
    max_candidates: int = 24

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= 24:
            raise ValueError("hyperparameter max_candidates must be in [1,24]")


def run_chronaris_v2_hyperparameter_screen(
    config: ChronarisV2HyperparameterScreenConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2HyperparameterScreenConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    _assert_architecture_gate_passed(resolved)
    data = load_simulation_locked_pretraining_data(resolved.simulation_root)
    normalizer = TrainOnlyRobustNormalizer().fit(
        data.batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=(
            data.fold.validation_sample_ids + data.fold.held_out_sample_ids
        ),
    )
    candidates = chronaris_v2_hyperparameter_grid(
        physiology_residual_mode="direct_causal_query"
    )[: resolved.max_candidates]
    rows = []
    for candidate in candidates:
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
            output_root=heavy_root / "checkpoints",
            config=ChronarisV2TrainingConfig(
                max_epochs=resolved.max_epochs,
                batch_size=resolved.batch_size,
                patience=min(8, resolved.max_epochs),
                seed=17,
                device=resolved.device,
            ),
            resume=True,
        )
        payload = torch.load(
            result.best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        rows.append(
            {
                **asdict(candidate),
                "status": result.status,
                "completed_epochs": result.completed_epochs,
                "best_epoch": result.best_epoch,
                "public_selection_loss": result.best_public_selection_loss,
                "parameter_count": int(payload["parameter_count"]),
                "checkpoint_path": result.last_checkpoint_path,
                "checkpoint_role": "final_task_independent_training_state",
                "checkpoint_sha256": _sha256_file(
                    Path(result.last_checkpoint_path)
                ),
                "task_labels_opened": bool(
                    payload["label_used_for_encoder_training"]
                ),
                "simulation_oracle_opened": bool(
                    payload["simulation_oracle_opened"]
                ),
                "locked_test_opened": bool(payload["locked_test_opened"]),
            }
        )
        pd.DataFrame(rows).to_csv(
            compact_root / "candidate_training.csv",
            index=False,
        )
        _write_json(
            compact_root / "progress.json",
            {
                "completed_candidate_count": len(rows),
                "requested_candidate_count": len(candidates),
                "last_candidate_id": candidate.candidate_id,
            },
        )
    acceptance = (
        _check(
            "exact_grid_registered",
            len(
                chronaris_v2_hyperparameter_grid(
                    physiology_residual_mode="direct_causal_query"
                )
            )
            == 24,
        ),
        _check("requested_candidates_completed", len(rows) == len(candidates)),
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
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(
        compact_root / "screen_protocol.json",
        {
            "format": "chronaris.v2_hyperparameter_screen_protocol.v1",
            "config": asdict(resolved),
            "candidate_grid": [
                asdict(value)
                for value in chronaris_v2_hyperparameter_grid(
                    physiology_residual_mode="direct_causal_query"
                )
            ],
            "architecture_gate_verified_before_training": True,
            "task_labels_opened": False,
            "outer_test_opened": False,
            "sealed_confirmation_opened": False,
            "selection_status": "awaiting_task_independent_gate_diagnostics",
        },
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_hyperparameter_screen_evidence.v1",
            "run_id": resolved.run_id,
            "status": "training_complete",
            "selection_status": "awaiting_task_independent_gate_diagnostics",
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
        },
    )
    return compact_root


def _assert_architecture_gate_passed(config) -> None:
    root = Path(config.compact_output_root) / config.architecture_gate_run_id
    manifest = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(root / "task_independent_ranking.csv")
    selected = ranking[
        ranking["candidate_id"] == config.architecture_gate_candidate_id
    ]
    manifest_selected = set(manifest.get("selected_candidate_ids", ()))
    if manifest.get("status") != "gates_passed" or len(selected) != 1:
        raise PermissionError("hyperparameter screen is closed before architecture gate")
    if config.architecture_gate_candidate_id not in manifest_selected:
        raise PermissionError("architecture candidate was not selected by the gate run")
    row = selected.iloc[0]
    if not bool(row["gate_passed"]):
        raise PermissionError("selected v2 architecture did not pass the gate")
    forbidden = (
        "task_labels_opened",
        "outer_test_opened",
        "sealed_confirmation_opened",
    )
    if any(bool(row.get(name, False)) for name in forbidden):
        raise PermissionError("architecture gate opened a forbidden evidence source")


# Compatibility alias for older callers; the check now targets the versioned
# architecture gate instead of silently reopening the failed structure gate.
_assert_complete_v2_passed_structure_gate = _assert_architecture_gate_passed


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
