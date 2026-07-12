"""Resumable observed-only execution of the Chronaris v2 structure screen."""

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
    chronaris_v2_structure_candidates,
    chronaris_v2_structure_training_grid,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


@dataclass(frozen=True, slots=True)
class ChronarisV2StructureScreenConfig:
    run_id: str = "2026-07-12_chronaris-v2-structure-screen-seed17"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    v1_reference_checkpoint: str = (
        "artifacts/application_evaluation/"
        "2026-07-11_encoder-candidate-screen-seed17/checkpoints/chronaris/A/best.pt"
    )
    device: str = "cuda"
    max_epochs: int = 50
    batch_size: int = 128
    max_candidates: int = 7
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= 7:
            raise ValueError("structure screen max_candidates must be in [1,7]")
        registered = {
            value.candidate_id for value in chronaris_v2_structure_training_grid()
        }
        if self.candidate_ids and (
            len(self.candidate_ids) != len(set(self.candidate_ids))
            or not set(self.candidate_ids) <= registered
        ):
            raise ValueError("structure screen candidate_ids are invalid")


def run_chronaris_v2_structure_screen(
    config: ChronarisV2StructureScreenConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2StructureScreenConfig()
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
    reference = Path(resolved.v1_reference_checkpoint)
    if not reference.is_file():
        raise FileNotFoundError(f"v1 structure reference is missing: {reference}")
    reference_payload = torch.load(reference, map_location="cpu", weights_only=True)
    if reference_payload.get("format") != "chronaris.common_pretraining_checkpoint.v1":
        raise ValueError("v1 structure reference checkpoint format changed")
    rows = [
        {
            "candidate_id": "structure_01_v1",
            "architecture_version": "v1",
            "status": "immutable_reference",
            "completed_epochs": int(reference_payload["completed_epochs"]),
            "best_epoch": int(reference_payload["best_epoch"]),
            "public_selection_loss": float(
                reference_payload["best_public_selection_loss"]
            ),
            "parameter_count": int(reference_payload["parameter_count"]),
            "checkpoint_path": str(reference),
            "checkpoint_sha256": _sha256_file(reference),
            "task_labels_opened": False,
            "simulation_oracle_opened": False,
            "locked_test_opened": False,
        }
    ]
    grid = chronaris_v2_structure_training_grid()
    candidates = (
        tuple(
            candidate
            for candidate in grid
            if candidate.candidate_id in resolved.candidate_ids
        )
        if resolved.candidate_ids
        else grid[: resolved.max_candidates]
    )
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
                "candidate_id": candidate.candidate_id,
                "architecture_version": "v2",
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
                "completed_v2_candidate_count": len(rows) - 1,
                "requested_v2_candidate_count": len(candidates),
                "last_candidate_id": candidate.candidate_id,
            },
        )
    acceptance = (
        _check("v1_reference_preserved", rows[0]["status"] == "immutable_reference"),
        _check("requested_candidates_completed", len(rows) == len(candidates) + 1),
        _check(
            "forbidden_sources_closed",
            all(
                not row["task_labels_opened"]
                and not row["simulation_oracle_opened"]
                and not row["locked_test_opened"]
                for row in rows
            ),
        ),
        _check("registered_structure_count", len(chronaris_v2_structure_candidates()) == 8),
    )
    pd.DataFrame(rows).to_csv(compact_root / "candidate_training.csv", index=False)
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(
        compact_root / "screen_protocol.json",
        {
            "format": "chronaris.v2_structure_screen_protocol.v1",
            "config": asdict(resolved),
            "structure_candidates": [
                asdict(value) for value in chronaris_v2_structure_candidates()
            ],
            "selection_status": "awaiting_task_independent_gate_diagnostics",
            "task_labels_opened": False,
            "simulation_oracle_opened": False,
            "locked_test_opened": False,
        },
    )
    passed = sum(row["passed"] for row in acceptance)
    (compact_root / "summary.md").write_text(
        "\n".join(
            (
                "# Chronaris v2 结构筛选训练",
                "",
                f"训练层验收 {passed}/{len(acceptance)}；已执行 {len(candidates)}/7 个 v2 结构候选。",
                "当前只形成公共自监督训练 checkpoint；保真、表示健康和时间机制门禁完成前不生成结构排名。",
                "任务标签、仿真任务真值和锁定确认族保持关闭。",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_structure_screen_evidence.v1",
            "run_id": resolved.run_id,
            "status": "training_complete" if passed == len(acceptance) else "partial",
            "selection_status": "awaiting_task_independent_gate_diagnostics",
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
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
