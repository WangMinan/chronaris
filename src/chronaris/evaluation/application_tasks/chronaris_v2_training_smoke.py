"""Observed-only G1 smoke run for the Chronaris v2 training protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.training import (
    ChronarisV2TrainingConfig,
    chronaris_v2_hyperparameter_grid,
    load_common_pretraining_checkpoint,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


@dataclass(frozen=True, slots=True)
class ChronarisV2TrainingSmokeConfig:
    run_id: str = "2026-07-12_chronaris-v2-training-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    device: str = "cpu"
    max_epochs: int = 1
    batch_size: int = 128


def run_chronaris_v2_training_smoke(
    config: ChronarisV2TrainingSmokeConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2TrainingSmokeConfig()
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
    candidate = chronaris_v2_hyperparameter_grid()[0]
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
            patience=max(1, resolved.max_epochs),
            seed=17,
            device=resolved.device,
        ),
        resume=True,
    )
    encoder, _heads, _normalizer, payload = load_common_pretraining_checkpoint(
        result.best_checkpoint_path,
        device=resolved.device,
    )
    acceptance = (
        _check("training_completed", result.status in {"completed", "resumed"}, result.status, "completed_or_resumed"),
        _check("v2_checkpoint_format", payload["format"] == "chronaris.common_pretraining_checkpoint.v2", payload["format"], "chronaris.common_pretraining_checkpoint.v2"),
        _check("architecture_version", encoder.backbone.config.architecture_version == "v2", encoder.backbone.config.architecture_version, "v2"),
        _check("task_labels_closed", payload["label_used_for_encoder_training"] is False, payload["label_used_for_encoder_training"], False),
        _check("simulation_oracle_closed", payload["simulation_oracle_opened"] is False, payload["simulation_oracle_opened"], False),
        _check("locked_test_closed", payload["locked_test_opened"] is False, payload["locked_test_opened"], False),
        _check("grid_has_twenty_four_candidates", len(chronaris_v2_hyperparameter_grid()) == 24, len(chronaris_v2_hyperparameter_grid()), 24),
    )
    pd.DataFrame(result.epoch_rows).to_json(
        compact_root / "epoch_metrics.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    pd.DataFrame(result.gradient_rows).to_csv(
        compact_root / "gradient_conflicts.csv",
        index=False,
    )
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    protocol = {
        "format": "chronaris.v2_training_smoke.v1",
        "config": asdict(resolved),
        "candidate": asdict(candidate),
        "candidate_grid": [
            asdict(value) for value in chronaris_v2_hyperparameter_grid()
        ],
        "task_labels_opened": False,
        "simulation_oracle_opened": False,
        "locked_test_opened": False,
        "checkpoint_path": result.best_checkpoint_path,
        "checkpoint_format": payload["format"],
    }
    _write_json(compact_root / "protocol.json", protocol)
    passed = sum(row["passed"] for row in acceptance)
    (compact_root / "report.md").write_text(
        "\n".join(
            (
                "# Chronaris v2 任务无关训练协议冒烟",
                "",
                f"状态：{'completed' if passed == len(acceptance) else 'partial'}；验收 {passed}/{len(acceptance)}。",
                f"候选 {candidate.candidate_id} 完成 {result.completed_epochs} epoch，最佳公共选择损失 {result.best_public_selection_loss:.6f}。",
                "训练只读取 G1 observed-only 双流；任务标签、仿真 oracle 与锁定测试保持关闭。",
                "",
            )
        ),
        encoding="utf-8",
    )
    (compact_root / "resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_chronaris_v2_training_smoke.py "
        f"--run-id {resolved.run_id} --device {resolved.device} "
        f"--max-epochs {resolved.max_epochs} --batch-size {resolved.batch_size}\n",
        encoding="utf-8",
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_training_smoke_evidence.v1",
            "run_id": resolved.run_id,
            "status": "completed" if passed == len(acceptance) else "partial",
            "acceptance_pass_count": passed,
            "acceptance_check_count": len(acceptance),
            "compact_run_root": str(compact_root),
            "heavy_run_root": str(heavy_root),
        },
    )
    return compact_root


def _check(name, passed, observed, expected):
    return {
        "check": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
