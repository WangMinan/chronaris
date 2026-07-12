"""Pre-registered task-independent physiology-fidelity repair round."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

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
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    rank_task_independent_candidates,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


@dataclass(frozen=True, slots=True)
class FidelityRepairCandidate:
    candidate_id: str
    physiology_teacher_mode: str
    physiology_teacher_weight: float
    display_name: str


FIDELITY_REPAIR_CANDIDATES = (
    FidelityRepairCandidate(
        "fidelity_01_private_teacher_w050",
        "private",
        0.5,
        "生理私有子空间蒸馏",
    ),
    FidelityRepairCandidate(
        "fidelity_02_path_teacher_w050",
        "physiology_path",
        0.5,
        "生理路径蒸馏（权重 0.5）",
    ),
    FidelityRepairCandidate(
        "fidelity_03_path_teacher_w100",
        "physiology_path",
        1.0,
        "生理路径蒸馏（权重 1.0）",
    ),
)


@dataclass(frozen=True, slots=True)
class ChronarisV2FidelityRepairConfig:
    run_id: str = "2026-07-12_chronaris-v2-fidelity-repair-seed17"
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
    physiology_teacher_checkpoint: str = (
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
    device: str = "cuda"
    max_epochs: int = 30
    batch_size: int = 128
    max_candidates: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= len(FIDELITY_REPAIR_CANDIDATES):
            raise ValueError("fidelity repair max_candidates must be in [1,3]")
        if self.max_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("fidelity repair epoch/batch size must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("fidelity repair device must be cpu or cuda")


def run_chronaris_v2_fidelity_repair(
    config: ChronarisV2FidelityRepairConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2FidelityRepairConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    selected_specs = FIDELITY_REPAIR_CANDIDATES[: resolved.max_candidates]
    _write_json(
        compact_root / "preregistration.json",
        {
            "format": "chronaris.v2_fidelity_repair_preregistration.v1",
            "config": asdict(resolved),
            "candidates": [asdict(value) for value in selected_specs],
            "fixed_architecture": {
                "structure_candidate_id": "structure_08_complete_v2",
                "hidden_dim": 64,
                "lag_mode": "fixed_five",
                "solver": "euler",
                "learning_rate": 3e-4,
                "phase_epoch_offset": 20,
                "subspace_slices": {
                    "vehicle_private": [0, 24],
                    "physiology_private": [24, 40],
                    "causal_shared": [40, 64],
                },
            },
            "selection_inputs": [
                "public_self_supervised_validation_loss",
                "single_stream_fidelity",
                "representation_health",
                "synthetic_time_mechanism_recovery",
                "causal_and_pooling_invariants",
                "parameter_count",
            ],
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
    complete_checkpoint = Path(resolved.complete_v2_checkpoint)
    physiology_checkpoint = Path(resolved.physiology_teacher_checkpoint)
    vehicle_checkpoint = Path(resolved.vehicle_reference_checkpoint)
    for path in (complete_checkpoint, physiology_checkpoint, vehicle_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"fidelity repair dependency is missing: {path}")
    teacher_encoder, _teacher_heads, teacher_normalizer, teacher_payload = (
        load_common_pretraining_checkpoint(
            physiology_checkpoint,
            device=resolved.device,
        )
    )
    if teacher_payload["method_name"] != "physiology_only":
        raise ValueError("fidelity repair requires a physiology-only teacher")
    teacher_adapter = TrainedFusionAdapter(
        encoder=teacher_encoder,
        normalizer=teacher_normalizer,
        fold_id=data.fold.fold_id,
        checkpoint_sha256=_sha256_file(physiology_checkpoint),
    )
    physiology_teacher_targets = teacher_adapter(data.batch)
    rows = []
    for spec in selected_specs:
        candidate = ChronarisV2CandidateConfig(
            candidate_id=spec.candidate_id,
            internal_hidden_dim=64,
            lag_mode="fixed_five",
            ode_method="euler",
            learning_rate=3e-4,
            structure_candidate_id="structure_08_complete_v2",
            physiology_teacher_mode=spec.physiology_teacher_mode,
            physiology_teacher_weight=spec.physiology_teacher_weight,
            phase_epoch_offset=20,
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
            physiology_teacher_targets=physiology_teacher_targets,
            initialization_checkpoint=complete_checkpoint,
            resume=True,
        )
        payload = torch.load(
            result.last_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "display_name": spec.display_name,
                "status": result.status,
                "completed_epochs": result.completed_epochs,
                "best_epoch": result.best_epoch,
                "public_selection_loss": result.best_public_selection_loss,
                "checkpoint_path": result.last_checkpoint_path,
                "checkpoint_sha256": _sha256_file(Path(result.last_checkpoint_path)),
                "initialization_checkpoint_sha256": payload[
                    "initialization_manifest"
                ]["checkpoint_sha256"],
                "teacher_checkpoint_sha256": payload[
                    "physiology_teacher_manifest"
                ]["checkpoint_sha256"],
                "teacher_target_sha256": payload[
                    "physiology_teacher_manifest"
                ]["target_sha256"],
                "task_labels_opened": False,
                "simulation_oracle_opened": False,
                "locked_test_opened": False,
            }
        )
        pd.DataFrame(rows).to_csv(compact_root / "candidate_training.csv", index=False)
    full_round = len(rows) == len(FIDELITY_REPAIR_CANDIDATES)
    if full_round:
        _diagnose_full_round(
            resolved=resolved,
            compact_root=compact_root,
            data=data,
            rows=rows,
        )
    acceptance = (
        _check("preregistration_written_before_ranking", True),
        _check("requested_candidates_completed", len(rows) == len(selected_specs)),
        _check(
            "forbidden_sources_closed",
            all(
                not row["task_labels_opened"]
                and not row["simulation_oracle_opened"]
                and not row["locked_test_opened"]
                for row in rows
            ),
        ),
        _check("full_round_diagnosed", full_round),
    )
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_fidelity_repair_evidence.v1",
            "run_id": resolved.run_id,
            "status": "diagnosed" if full_round else "smoke_complete",
            "candidate_count": len(rows),
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
        },
    )
    return compact_root


def _diagnose_full_round(*, resolved, compact_root, data, rows) -> None:
    structure = pd.read_csv(
        Path(resolved.compact_output_root)
        / resolved.structure_diagnostics_run_id
        / "candidate_diagnostics.csv"
    )
    v1 = structure.loc[structure["candidate_id"] == "structure_01_v1"].iloc[0]
    common = {
        "physiology_reference_checkpoint": resolved.physiology_teacher_checkpoint,
        "vehicle_reference_checkpoint": resolved.vehicle_reference_checkpoint,
        "batch": data.batch,
        "train_sample_ids": data.fold.train_sample_ids,
        "validation_sample_ids": data.fold.validation_sample_ids,
        "fold_id": data.fold.fold_id,
        "device": resolved.device,
        "v1_clock_offset_mae_s": float(v1["clock_offset_mae_s"]),
        "v1_response_lag_mae_s": float(v1["response_lag_mae_s"]),
    }
    results = [
        diagnose_chronaris_candidate(
            candidate_checkpoint=resolved.complete_v2_checkpoint,
            candidate_id="fidelity_00_complete_v2_reference",
            **common,
        )
    ]
    results.extend(
        diagnose_chronaris_candidate(
            candidate_checkpoint=row["checkpoint_path"],
            candidate_id=row["candidate_id"],
            **common,
        )
        for row in rows
    )
    evidence = tuple(result.evidence for result in results)
    ranking = rank_task_independent_candidates(evidence, top_k=1)
    pd.DataFrame(row.to_dict() for row in evidence).to_csv(
        compact_root / "candidate_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(ranking).to_csv(
        compact_root / "task_independent_ranking.csv",
        index=False,
    )
    _write_json(
        compact_root / "diagnostic_details.json",
        {result.evidence.candidate_id: result.details for result in results},
    )
    render_structure_gate_figure(
        compact_root / "task_independent_ranking.csv",
        compact_root / "fidelity_gate_overview.png",
    )
    passed = [row for row in ranking if row["gate_passed"]]
    (compact_root / "summary.md").write_text(
        "\n".join(
            (
                "# Chronaris v2 生理保真修复门禁",
                "",
                f"状态：{'gates_passed' if passed else 'gates_failed'}；门内候选 {len(passed)} 个。",
                "本轮只使用冻结生理单流教师、公共自监督验证损失和既有任务无关机制门禁。",
                "未读取下游标签、outer-test 或封存仿真确认结果。",
                "",
            )
        ),
        encoding="utf-8",
    )


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
