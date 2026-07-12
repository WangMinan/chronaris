"""Task-independent diagnostics and top-three selection for the v2 grid."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_candidate_diagnostics import (
    diagnose_chronaris_candidate,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.training import rank_task_independent_candidates


@dataclass(frozen=True, slots=True)
class ChronarisV2HyperparameterDiagnosticsConfig:
    run_id: str = "2026-07-12_chronaris-v2-hyperparameter-diagnostics-seed17"
    hyperparameter_training_run_id: str = (
        "2026-07-12_chronaris-v2-hyperparameter-screen-seed17"
    )
    structure_diagnostics_run_id: str = (
        "2026-07-12_chronaris-v2-structure-diagnostics-seed17"
    )
    compact_output_root: str = "docs/artifacts/runs"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    physiology_reference_checkpoint: str = (
        "artifacts/application_evaluation/"
        "2026-07-11_encoder-candidate-screen-seed17/"
        "checkpoints/physiology_only/A/best.pt"
    )
    vehicle_reference_checkpoint: str = (
        "artifacts/application_evaluation/"
        "2026-07-11_encoder-candidate-screen-seed17/"
        "checkpoints/vehicle_only/C/best.pt"
    )
    device: str = "cuda"


def run_chronaris_v2_hyperparameter_diagnostics(
    config: ChronarisV2HyperparameterDiagnosticsConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2HyperparameterDiagnosticsConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    training_root = (
        Path(resolved.compact_output_root)
        / resolved.hyperparameter_training_run_id
    )
    rows = pd.read_csv(training_root / "candidate_training.csv").to_dict(
        orient="records"
    )
    if len(rows) != 24 or any(
        row["status"] not in {"completed", "resumed"} for row in rows
    ):
        raise ValueError("hyperparameter diagnostics require all 24 checkpoints")
    structure_root = (
        Path(resolved.compact_output_root) / resolved.structure_diagnostics_run_id
    )
    structure = pd.read_csv(structure_root / "candidate_diagnostics.csv")
    v1 = structure[structure["candidate_id"] == "structure_01_v1"]
    if len(v1) != 1:
        raise ValueError("hyperparameter diagnostics require one v1 time baseline")
    clock_baseline = float(v1.iloc[0]["clock_offset_mae_s"])
    lag_baseline = float(v1.iloc[0]["response_lag_mae_s"])
    data = load_simulation_locked_pretraining_data(resolved.simulation_root)
    results = []
    for row in rows:
        results.append(
            diagnose_chronaris_candidate(
                candidate_checkpoint=row["checkpoint_path"],
                candidate_id=str(row["candidate_id"]),
                physiology_reference_checkpoint=resolved.physiology_reference_checkpoint,
                vehicle_reference_checkpoint=resolved.vehicle_reference_checkpoint,
                batch=data.batch,
                train_sample_ids=data.fold.train_sample_ids,
                validation_sample_ids=data.fold.validation_sample_ids,
                fold_id=data.fold.fold_id,
                device=resolved.device,
                v1_clock_offset_mae_s=clock_baseline,
                v1_response_lag_mae_s=lag_baseline,
            )
        )
    ranking = rank_task_independent_candidates(
        tuple(result.evidence for result in results),
        top_k=3,
    )
    selected = [row for row in ranking if row["selected_for_inner_validation"]]
    pd.DataFrame(result.evidence.to_dict() for result in results).to_csv(
        compact_root / "candidate_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(ranking).to_csv(
        compact_root / "task_independent_ranking.csv",
        index=False,
    )
    selected_configs = {
        row["candidate_id"]: next(
            candidate
            for candidate in rows
            if candidate["candidate_id"] == row["candidate_id"]
        )
        for row in selected
    }
    _write_json(compact_root / "selected_top_three.json", selected_configs)
    _write_json(
        compact_root / "diagnostic_details.json",
        {
            result.evidence.candidate_id: result.details for result in results
        },
    )
    status = "selected_top_three" if len(selected) == 3 else "gates_failed"
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_hyperparameter_diagnostics_evidence.v1",
            "run_id": resolved.run_id,
            "status": status,
            "selected_candidate_ids": [row["candidate_id"] for row in selected],
            "task_labels_opened": False,
            "outer_test_opened": False,
            "sealed_confirmation_opened": False,
            "confirmed_metrics_changed": False,
        },
    )
    _write_json(
        compact_root / "protocol.json",
        {
            "format": "chronaris.v2_hyperparameter_diagnostics_protocol.v1",
            "config": asdict(resolved),
            "lexicographic_order": [
                "public_self_supervised_validation_loss",
                "negative_worst_fold_fidelity",
                "time_mechanism_ratio",
                "parameter_count",
                "candidate_id",
            ],
            "downstream_labels_opened": False,
        },
    )
    return compact_root


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
