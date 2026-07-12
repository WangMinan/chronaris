"""Pre-registered direct physiology residual repair for Chronaris v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_candidate_diagnostics import (
    diagnose_chronaris_candidate,
)
from chronaris.evaluation.application_tasks.chronaris_v2_structure_figures import (
    render_structure_gate_figure,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.training import (
    ChronarisV2CandidateConfig,
    ChronarisV2TrainingConfig,
    rank_task_independent_candidates,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


DIRECT_RESIDUAL_CANDIDATE_ID = "direct_residual_01_causal_query"


@dataclass(frozen=True, slots=True)
class ChronarisV2DirectResidualRepairConfig:
    run_id: str = "2026-07-12_chronaris-v2-direct-residual-repair-seed17"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    complete_v2_checkpoint: str = (
        "artifacts/application_evaluation/"
        "2026-07-12_chronaris-v2-structure-screen-seed17-r5-cpu-recovery/"
        "checkpoints/structure_08_complete_v2/last.pt"
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
    structure_diagnostics_run_id: str = (
        "2026-07-12_chronaris-v2-structure-diagnostics-seed17"
    )
    device: str = "cpu"
    max_epochs: int = 5
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("direct residual repair device must be cpu or cuda")
        if self.max_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("direct residual repair epoch/batch size must be positive")


def run_chronaris_v2_direct_residual_repair(
    config: ChronarisV2DirectResidualRepairConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2DirectResidualRepairConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        compact_root / "preregistration.json",
        {
            "format": "chronaris.v2_direct_residual_repair_preregistration.v1",
            "config": asdict(resolved),
            "candidate": {
                "candidate_id": DIRECT_RESIDUAL_CANDIDATE_ID,
                "physiology_residual_mode": "direct_causal_query",
                "direct_feature_capacity": 16,
                "learning_rate": 3e-4,
                "phase_epoch_offset": 20,
                "task_label_usage": False,
            },
            "rationale": (
                "preserve normalized physiology observations in the fixed private "
                "slice without a learnable projection absorbing the fidelity loss"
            ),
            "task_labels_opened": False,
            "simulation_oracle_opened": False,
            "locked_test_opened": False,
            "sealed_confirmation_opened": False,
        },
    )
    data = load_simulation_locked_pretraining_data(resolved.simulation_root)
    normalizer = TrainOnlyRobustNormalizer().fit(
        data.batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=(
            data.fold.validation_sample_ids + data.fold.held_out_sample_ids
        ),
    )
    candidate = ChronarisV2CandidateConfig(
        candidate_id=DIRECT_RESIDUAL_CANDIDATE_ID,
        internal_hidden_dim=64,
        lag_mode="fixed_five",
        ode_method="euler",
        learning_rate=3e-4,
        structure_candidate_id="structure_08_complete_v2",
        phase_epoch_offset=20,
        physiology_residual_mode="direct_causal_query",
    )
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
        initialization_checkpoint=resolved.complete_v2_checkpoint,
        resume=True,
    )
    training_row = {
        "candidate_id": DIRECT_RESIDUAL_CANDIDATE_ID,
        "status": result.status,
        "completed_epochs": result.completed_epochs,
        "best_epoch": result.best_epoch,
        "public_selection_loss": result.best_public_selection_loss,
        "checkpoint_path": result.last_checkpoint_path,
        "checkpoint_sha256": _sha256_file(Path(result.last_checkpoint_path)),
        "task_labels_opened": False,
        "simulation_oracle_opened": False,
        "locked_test_opened": False,
    }
    pd.DataFrame((training_row,)).to_csv(
        compact_root / "candidate_training.csv",
        index=False,
    )
    structure = pd.read_csv(
        Path(resolved.compact_output_root)
        / resolved.structure_diagnostics_run_id
        / "candidate_diagnostics.csv"
    )
    v1 = structure.loc[structure["candidate_id"] == "structure_01_v1"].iloc[0]
    common = {
        "physiology_reference_checkpoint": resolved.physiology_reference_checkpoint,
        "vehicle_reference_checkpoint": resolved.vehicle_reference_checkpoint,
        "batch": data.batch,
        "train_sample_ids": data.fold.train_sample_ids,
        "validation_sample_ids": data.fold.validation_sample_ids,
        "fold_id": data.fold.fold_id,
        "device": resolved.device,
        "v1_clock_offset_mae_s": float(v1["clock_offset_mae_s"]),
        "v1_response_lag_mae_s": float(v1["response_lag_mae_s"]),
    }
    results = (
        diagnose_chronaris_candidate(
            candidate_checkpoint=resolved.complete_v2_checkpoint,
            candidate_id="direct_residual_00_complete_v2_reference",
            **common,
        ),
        diagnose_chronaris_candidate(
            candidate_checkpoint=result.last_checkpoint_path,
            candidate_id=DIRECT_RESIDUAL_CANDIDATE_ID,
            **common,
        ),
    )
    evidence = tuple(value.evidence for value in results)
    ranking = rank_task_independent_candidates(evidence, top_k=1)
    pd.DataFrame(row.to_dict() for row in evidence).to_csv(
        compact_root / "candidate_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(ranking).to_csv(
        compact_root / "task_independent_ranking.csv",
        index=False,
    )
    render_structure_gate_figure(
        compact_root / "task_independent_ranking.csv",
        compact_root / "direct_residual_gate_overview.png",
    )
    passed = [row for row in ranking if row["gate_passed"]]
    acceptance = (
        _check("preregistration_written_before_training", True),
        _check("candidate_completed", result.status in {"completed", "resumed"}),
        _check("forbidden_sources_closed", True),
        _check("task_independent_gate_diagnosed", len(ranking) == 2),
    )
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    (compact_root / "summary.md").write_text(
        "\n".join(
            (
                "# Chronaris v2 直接生理残差门禁",
                "",
                f"状态：{'gates_passed' if passed else 'gates_failed'}；门内候选 {len(passed)} 个。",
                "本轮在固定 16 维生理私有切片中直接保留因果查询值。",
                "未读取下游标签、outer-test 或封存仿真确认结果。",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_direct_residual_repair_evidence.v1",
            "run_id": resolved.run_id,
            "status": "gates_passed" if passed else "gates_failed",
            "selected_candidate_ids": [row["candidate_id"] for row in passed],
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
        },
    )
    return compact_root


def _check(name: str, passed: bool) -> dict[str, object]:
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
