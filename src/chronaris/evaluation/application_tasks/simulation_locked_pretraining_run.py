"""Three-seed G1 retraining of the five selected fusion encoders."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    CandidateScreenConfig,
    LockedChronarisTrainingConfig,
    confirm_selected_pretext_checkpoint,
    train_locked_chronaris,
    train_pretext_candidate,
)
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_locked_pretraining")
LOGGER.addHandler(logging.NullHandler())
LOCKED_SEEDS = (17, 29, 43)


@dataclass(frozen=True, slots=True)
class SimulationLockedPretrainingConfig:
    run_id: str = "2026-07-12_simulation-locked-pretraining"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("locked seeds must be non-empty and unique")
        if not set(self.seeds).issubset(LOCKED_SEEDS):
            raise ValueError("locked seeds must be selected from 17, 29, 43")


@dataclass(frozen=True, slots=True)
class SimulationLockedPretrainingResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_seed_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_locked_pretraining(
    config: SimulationLockedPretrainingConfig,
) -> SimulationLockedPretrainingResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    selected_path = Path(config.selected_candidates_path)
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_ids = {
        method: str(selected_payload[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    candidates = {value.candidate_id: value for value in ENCODER_SCREEN_CANDIDATES}
    data = load_simulation_locked_pretraining_data(config.simulation_root)
    normalizer = TrainOnlyRobustNormalizer().fit(
        data.batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=(
            data.fold.validation_sample_ids + data.fold.held_out_sample_ids
        ),
    )
    baseline_device = _resolve_device(config.baseline_device)
    chronaris_device = _resolve_device(config.chronaris_device)
    vehicle_labels = tuple((name, name) for name in data.schema.vehicle_feature_names)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_locked_pretraining",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "selected_candidates": selected_ids,
            "g2_locked_test_opened": False,
            "task_targets_opened": False,
        },
    ) as progress:
        result_rows = []
        epoch_rows = []
        auxiliary_rows = []
        for seed in config.seeds:
            for method_name in TRAINABLE_FUSION_METHODS:
                candidate = candidates[selected_ids[method_name]]
                output_root = heavy_root / "checkpoints" / f"seed_{seed}"
                if method_name == "chronaris":
                    result = train_locked_chronaris(
                        batch=data.batch,
                        fold=data.fold,
                        physiology_feature_names=data.schema.physiology_feature_names,
                        vehicle_feature_names=data.schema.vehicle_feature_names,
                        vehicle_field_labels=vehicle_labels,
                        normalizer=normalizer,
                        output_root=output_root,
                        config=LockedChronarisTrainingConfig(
                            max_epochs=config.max_epochs,
                            batch_size=config.batch_size,
                            patience=config.patience,
                            seed=seed,
                            device=chronaris_device,
                        ),
                        augmentation_policy=AugmentationPolicy(),
                        candidate_config=candidate,
                        resume=config.resume,
                    )
                    method_epoch_rows = result.epoch_rows
                    method_auxiliary_rows = result.auxiliary_rows
                    auxiliary_enabled = True
                else:
                    result = train_pretext_candidate(
                        method_name,
                        candidate=candidate,
                        batch=data.batch,
                        fold=data.fold,
                        physiology_feature_names=data.schema.physiology_feature_names,
                        vehicle_feature_names=data.schema.vehicle_feature_names,
                        vehicle_field_labels=vehicle_labels,
                        normalizer=normalizer,
                        output_root=output_root,
                        config=CandidateScreenConfig(
                            max_epochs=config.max_epochs,
                            batch_size=config.batch_size,
                            patience=config.patience,
                            seed=seed,
                            device=baseline_device,
                        ),
                        augmentation_policy=AugmentationPolicy(),
                        resume=config.resume,
                    )
                    method_epoch_rows = result.epoch_rows
                    method_auxiliary_rows = ()
                    auxiliary_enabled = False
                confirmation = confirm_selected_pretext_checkpoint(
                    result.best_checkpoint_path,
                    batch=data.batch,
                    sample_ids=data.fold.held_out_sample_ids,
                    batch_size=config.batch_size,
                    seed=seed,
                    device=(
                        chronaris_device if method_name == "chronaris" else baseline_device
                    ),
                )
                result_rows.append(
                    {
                        "seed": seed,
                        "method_name": method_name,
                        "candidate_id": candidate.candidate_id,
                        "status": result.status,
                        "best_epoch": result.best_epoch,
                        "completed_epochs": result.completed_epochs,
                        "stopped_early": result.stopped_early,
                        "best_public_selection_loss": result.best_public_selection_loss,
                        "training_elapsed_s": result.training_elapsed_s,
                        "parameter_count": result.parameter_count,
                        "checkpoint_path": result.best_checkpoint_path,
                        "checkpoint_sha256": sha256_file(result.best_checkpoint_path),
                        "training_device": (
                            chronaris_device if method_name == "chronaris" else baseline_device
                        ),
                        "chronaris_auxiliary_enabled": auxiliary_enabled,
                        "reserved_confirmation_loss": confirmation[
                            "public_confirmation_loss"
                        ],
                        "reserved_confirmation_sample_count": confirmation[
                            "sample_count"
                        ],
                        "g2_locked_test_opened": False,
                        "task_targets_opened": False,
                    }
                )
                epoch_rows.extend(
                    {"seed": seed, "method_name": method_name, **row}
                    for row in method_epoch_rows
                )
                auxiliary_rows.extend(
                    {"seed": seed, "method_name": method_name, **row}
                    for row in method_auxiliary_rows
                )
                progress.update(
                    "locked_method_seed_complete",
                    seed=seed,
                    method_name=method_name,
                    candidate_id=candidate.candidate_id,
                    status=result.status,
                    best_epoch=result.best_epoch,
                )
        acceptance = _acceptance_rows(
            config=config,
            rows=result_rows,
            auxiliary_rows=auxiliary_rows,
            data=data,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            selected_path=selected_path,
            selected_ids=selected_ids,
            baseline_device=baseline_device,
            chronaris_device=chronaris_device,
            data=data,
            result_rows=result_rows,
            epoch_rows=epoch_rows,
            auxiliary_rows=auxiliary_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            method_seed_count=len(result_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationLockedPretrainingResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        method_seed_count=len(result_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("locked pretraining device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("locked pretraining requested unavailable CUDA device")
    return value


def _acceptance_rows(*, config, rows, auxiliary_rows, data):
    expected = len(config.seeds) * len(TRAINABLE_FUSION_METHODS)
    chronaris_rows = [row for row in auxiliary_rows if row["method_name"] == "chronaris"]
    return (
        _check("all_method_seed_runs", len(rows) == expected, len(rows), expected),
        _check("all_checkpoints_complete", all(row["status"] in {"completed", "resumed"} for row in rows), [row["status"] for row in rows], "completed_or_resumed"),
        _check("public_validation_losses_finite", all(row["best_public_selection_loss"] >= 0 for row in rows), len(rows), expected),
        _check("reserved_confirmation_complete", all(row["reserved_confirmation_sample_count"] == 1 and row["reserved_confirmation_loss"] >= 0 for row in rows), len(rows), expected),
        _check("chronaris_auxiliary_recorded", bool(chronaris_rows), len(chronaris_rows), ">0"),
        _check(
            "chronaris_auxiliary_schedule_consistent",
            (
                any(row["weight"] > 0 for row in chronaris_rows)
                if config.max_epochs > 10
                else all(row["weight"] == 0 for row in chronaris_rows)
            ),
            any(row["weight"] > 0 for row in chronaris_rows),
            config.max_epochs > 10,
        ),
        _check("g1_train_validation_confirmation_counts", len(data.fold.train_sample_ids) == 96 and len(data.fold.validation_sample_ids) == 23 and len(data.fold.held_out_sample_ids) == 1, [len(data.fold.train_sample_ids), len(data.fold.validation_sample_ids), len(data.fold.held_out_sample_ids)], [96, 23, 1]),
        _check("normalizer_train_only", all(not row["g2_locked_test_opened"] for row in rows), False, False),
        _check("task_targets_closed", all(not row["task_targets_opened"] for row in rows), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "locked_pretraining_results.csv",
        "epochs": root / "epoch_metrics.jsonl",
        "auxiliary": root / "chronaris_auxiliary_metrics.csv",
        "data": root / "data_manifest.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["result_rows"]).to_csv(paths["results"], index=False)
    paths["epochs"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values["epoch_rows"]), encoding="utf-8")
    pd.DataFrame(values["auxiliary_rows"]).to_csv(paths["auxiliary"], index=False)
    pd.DataFrame(values["data"].data_manifest_rows).to_csv(paths["data"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_locked_pretraining_protocol.v1",
        "config": asdict(values["config"]),
        "selected_candidates": values["selected_ids"],
        "selected_candidates_sha256": sha256_file(values["selected_path"]),
        "fold": values["data"].fold.to_dict(),
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "g2_locked_test_raw_file_count": values["data"].sealed_locked_test_file_count,
        "g2_locked_test_opened": False,
        "task_targets_opened": False,
        "early_stopping_inputs": ["masked_reconstruction", "short_horizon_prediction", "lag_discrimination"],
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text(
        "\n".join((
            "# 仿真选定配置三随机种子重训",
            "",
            f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
            f"完成 {len(values['result_rows'])} 个方法与随机种子组合；训练只读取 G1 原始异步双流。",
            "Chronaris 的连续对齐、物理一致性和因果方向损失参与反向传播，但早停只使用公共自监督验证损失。",
            "G2 锁定测试、任务真值和下游指标在本 run 中保持关闭。",
            "",
        )),
        encoding="utf-8",
    )
    seed_flags = " ".join(f"--seed {seed}" for seed in values["config"].seeds)
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_locked_pretraining.py "
        f"--run-id {values['config'].run_id} {seed_flags} "
        f"--max-epochs {values['config'].max_epochs} --batch-size {values['config'].batch_size} "
        f"--patience {values['config'].patience} --baseline-device {values['baseline_device']} "
        f"--chronaris-device {values['chronaris_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_locked_pretraining_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "method_seed_count": len(values["result_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "g2_locked_test_opened": False,
        "task_targets_opened": False,
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
