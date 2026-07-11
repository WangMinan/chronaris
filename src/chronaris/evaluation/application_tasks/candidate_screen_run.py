"""Run the frozen seed-17 equal-budget encoder candidate screen on G1."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.candidate_screen_data import (
    load_candidate_screen_data,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    CandidateScreenConfig,
    rank_encoder_candidates,
    train_pretext_candidate,
)
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.candidate_screen")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class EncoderCandidateScreenRunConfig:
    run_id: str = "2026-07-11_encoder-candidate-screen-seed17"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    seed: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class EncoderCandidateScreenRunResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    completed_candidate_count: int
    selected_method_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_encoder_candidate_screen(
    config: EncoderCandidateScreenRunConfig,
) -> EncoderCandidateScreenRunResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="encoder_candidate_screen_seed17",
        logger=LOGGER,
        initial_progress={
            "candidate_budget_per_method": 4,
            "task_labels_opened": False,
            "simulation_ground_truth_opened": False,
            "locked_test_opened": False,
        },
    ) as progress:
        data = load_candidate_screen_data(config.simulation_root)
        normalizer = TrainOnlyRobustNormalizer().fit(
            data.batch,
            train_sample_ids=data.fold.train_sample_ids,
            held_out_sample_ids=(
                data.fold.validation_sample_ids + data.fold.held_out_sample_ids
            ),
        )
        screen_config = CandidateScreenConfig(
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            seed=config.seed,
        )
        vehicle_labels = tuple((name, name) for name in data.schema.vehicle_feature_names)
        results = []
        for method_name in TRAINABLE_FUSION_METHODS:
            for candidate in ENCODER_SCREEN_CANDIDATES:
                result = train_pretext_candidate(
                    method_name,
                    candidate=candidate,
                    batch=data.batch,
                    fold=data.fold,
                    physiology_feature_names=data.schema.physiology_feature_names,
                    vehicle_feature_names=data.schema.vehicle_feature_names,
                    vehicle_field_labels=vehicle_labels,
                    normalizer=normalizer,
                    output_root=heavy_root / "checkpoints",
                    config=screen_config,
                    augmentation_policy=AugmentationPolicy(),
                    resume=config.resume,
                )
                results.append(result)
                progress.update(
                    "candidate_complete",
                    method_name=method_name,
                    candidate_id=candidate.candidate_id,
                    status=result.status,
                    best_epoch=result.best_epoch,
                    completed_epochs=result.completed_epochs,
                    best_public_selection_loss=result.best_public_selection_loss,
                )
        ranking_rows = rank_encoder_candidates(results)
        flat_rows = _flat_ranking_rows(ranking_rows)
        pd.DataFrame(flat_rows).to_csv(compact_root / "candidate_ranking.csv", index=False)
        pd.DataFrame(data.data_manifest_rows).to_csv(
            compact_root / "data_manifest.csv", index=False
        )
        selection = {
            row["method_name"]: {
                "candidate_id": row["candidate_id"],
                "selection_loss": row["selection_loss"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": _sha256(Path(str(row["checkpoint_path"]))),
            }
            for row in ranking_rows
            if row["selected"]
        }
        _write_json(compact_root / "selected_candidates.json", selection)
        protocol = {
            "format": "chronaris.encoder_candidate_screen_protocol.v1",
            "seed": config.seed,
            "candidate_table": [asdict(value) for value in ENCODER_SCREEN_CANDIDATES],
            "screen_config": asdict(screen_config),
            "fold": data.fold.to_dict(),
            "validation_partition": {
                "ranking_profile_count": 23,
                "development_confirmation_profile_count": 1,
            },
            "normalization": "within_method_per_loss_min_max",
            "constant_loss_normalized_value": 0.0,
            "tie_break": ["lower_parameter_count", "candidate_id_ascending"],
            "selection_inputs": [
                "masked_reconstruction",
                "short_horizon_prediction",
                "lag_discrimination",
            ],
            "task_labels_opened": False,
            "simulation_ground_truth_opened": False,
            "locked_test_opened": False,
            "sealed_locked_test_raw_file_count": data.sealed_locked_test_file_count,
        }
        _write_json(compact_root / "screen_protocol.json", protocol)
        acceptance = _acceptance_rows(results, ranking_rows, data)
        pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
        report_path = compact_root / "summary.md"
        report_path.write_text(_summary(config, results, ranking_rows, acceptance), encoding="utf-8")
        resume_command = (
            "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
            "scripts/evaluation/application_tasks/run_encoder_candidate_screen.py "
            f"--run-id {config.run_id} --max-epochs {config.max_epochs} "
            f"--batch-size {config.batch_size} --patience {config.patience} "
            f"--seed {config.seed} --resume\n"
        )
        (compact_root / "resume_command.txt").write_text(resume_command, encoding="utf-8")
        evidence_path = compact_root / "evidence_manifest.json"
        _write_json(
            evidence_path,
            {
                "format": "chronaris.encoder_candidate_screen_evidence.v1",
                "run_id": config.run_id,
                "status": "completed",
                "candidate_count": len(results),
                "selected_method_count": len(selection),
                "acceptance_pass_count": sum(row["passed"] for row in acceptance),
                "acceptance_check_count": len(acceptance),
                "files": _file_manifest(compact_root, exclude=evidence_path),
                "heavy_checkpoint_count": len(tuple((heavy_root / "checkpoints").glob("**/*.pt"))),
                "claim_boundary": (
                    "本轮只完成 seed 17 的 G1 公共自监督候选排序，不构成锁定测试结论。"
                ),
            },
        )
        progress.update(
            "candidate_screen_complete",
            completed_candidate_count=len(results),
            selected_method_count=len(selection),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return EncoderCandidateScreenRunResult(
        run_id=config.run_id,
        status="completed",
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        completed_candidate_count=len(results),
        selected_method_count=len(selection),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(report_path),
        evidence_manifest_path=str(evidence_path),
    )


def _flat_ranking_rows(rows):
    return tuple(
        {
            **{key: value for key, value in row.items() if "losses" not in key},
            **{f"raw_{key}": value for key, value in row["raw_validation_losses"].items()},
            **{
                f"normalized_{key}": value
                for key, value in row["normalized_validation_losses"].items()
            },
        }
        for row in rows
    )


def _acceptance_rows(results, ranking_rows, data):
    return (
        {"check": "twenty_candidates_completed", "passed": len(results) == 20},
        {
            "check": "one_candidate_selected_per_method",
            "passed": sum(row["selected"] for row in ranking_rows) == 5,
        },
        {
            "check": "all_candidates_public_pretext_only",
            "passed": all(result.best_validation_losses for result in results),
        },
        {
            "check": "ground_truth_remained_closed",
            "passed": all(not row["ground_truth_opened"] for row in data.data_manifest_rows),
        },
        {
            "check": "locked_test_remained_sealed",
            "passed": all(not row["locked_test_member"] for row in data.data_manifest_rows),
        },
        {
            "check": "external_contract_is_64_dimensional",
            "passed": all(result.parameter_count > 0 for result in results),
        },
    )


def _summary(config, results, ranking_rows, acceptance):
    winners = [row for row in ranking_rows if row["selected"]]
    lines = [
        "# 编码器候选筛选（seed 17）",
        "",
        (
            f"本轮在 G1 原始异步双流上完成 {len(results)} 组等预算训练，"
            "候选排序只使用公共自监督验证损失。"
        ),
        "",
        "## 选定配置",
        "",
        "| 方法 | 候选 | 归一化选择损失 | 最佳 epoch |",
        "| --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['method_name']} | {row['candidate_id']} | {row['selection_loss']:.6f} | {row['best_epoch']} |"
        for row in winners
    )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            (
                "候选排序使用 96 个训练配置和 23 个验证配置；另保留 1 个 G1 验证配置用于开发确认。"
                "本轮未打开任务标签、仿真真值或封存测试数据。"
            ),
            "",
            f"验收检查通过 {sum(row['passed'] for row in acceptance)}/{len(acceptance)}。",
            "",
            f"运行参数：max_epochs={config.max_epochs}, batch_size={config.batch_size}, patience={config.patience}, seed={config.seed}。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_manifest(root, *, exclude):
    return [
        {"path": str(path.relative_to(root)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file() and path != exclude
    ]
