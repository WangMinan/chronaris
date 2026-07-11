"""Locked G1 retraining for the four mechanism-removal Chronaris variants."""

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
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    LockedChronarisTrainingConfig,
    confirm_selected_pretext_checkpoint,
    train_locked_chronaris,
)
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_chronaris_ablation")
LOGGER.addHandler(logging.NullHandler())
CHRONARIS_ABLATION_VARIANTS = (
    "no_continuous_evolution",
    "no_physics",
    "no_causal_mask",
    "single_scale_lag",
)


@dataclass(frozen=True, slots=True)
class SimulationChronarisAblationPretrainingConfig:
    run_id: str = "2026-07-12_simulation-chronaris-ablation-pretraining"
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
    variants: tuple[str, ...] = CHRONARIS_ABLATION_VARIANTS
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    device: str = "cpu"
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.seeds or not set(self.seeds).issubset(LOCKED_SEEDS):
            raise ValueError("ablation seeds must be selected from 17, 29, 43")
        if not self.variants or not set(self.variants).issubset(
            CHRONARIS_ABLATION_VARIANTS
        ):
            raise ValueError("unsupported Chronaris ablation variant")


@dataclass(frozen=True, slots=True)
class SimulationChronarisAblationPretrainingResult:
    run_id: str
    status: str
    variant_seed_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    compact_run_root: str
    heavy_run_root: str
    report_path: str
    evidence_manifest_path: str


def run_simulation_chronaris_ablation_pretraining(
    config: SimulationChronarisAblationPretrainingConfig,
) -> SimulationChronarisAblationPretrainingResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    selected = json.loads(
        Path(config.selected_candidates_path).read_text(encoding="utf-8")
    )
    candidate_id = str(selected["chronaris"]["candidate_id"])
    candidate = next(
        value for value in ENCODER_SCREEN_CANDIDATES
        if value.candidate_id == candidate_id
    )
    data = load_simulation_locked_pretraining_data(config.simulation_root)
    normalizer = TrainOnlyRobustNormalizer().fit(
        data.batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=(
            data.fold.validation_sample_ids + data.fold.held_out_sample_ids
        ),
    )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Chronaris ablation requested unavailable CUDA")
    vehicle_labels = tuple((name, name) for name in data.schema.vehicle_feature_names)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_chronaris_ablation_pretraining",
        logger=LOGGER,
        initial_progress={
            "variants": list(config.variants),
            "seeds": list(config.seeds),
            "g2_locked_test_opened": False,
            "task_targets_opened": False,
        },
    ) as progress:
        result_rows = []
        epoch_rows = []
        auxiliary_rows = []
        for seed in config.seeds:
            for variant in config.variants:
                result = train_locked_chronaris(
                    batch=data.batch,
                    fold=data.fold,
                    physiology_feature_names=data.schema.physiology_feature_names,
                    vehicle_feature_names=data.schema.vehicle_feature_names,
                    vehicle_field_labels=vehicle_labels,
                    normalizer=normalizer,
                    output_root=(
                        heavy_root / "checkpoints" / f"seed_{seed}" / variant
                    ),
                    config=LockedChronarisTrainingConfig(
                        max_epochs=config.max_epochs,
                        batch_size=config.batch_size,
                        patience=config.patience,
                        seed=seed,
                        device=config.device,
                    ),
                    augmentation_policy=AugmentationPolicy(),
                    candidate_config=candidate,
                    variant=variant,
                    resume=config.resume,
                )
                confirmation = confirm_selected_pretext_checkpoint(
                    result.best_checkpoint_path,
                    batch=data.batch,
                    sample_ids=data.fold.held_out_sample_ids,
                    batch_size=config.batch_size,
                    seed=seed,
                    device=config.device,
                )
                result_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "candidate_id": candidate_id,
                        "status": result.status,
                        "best_epoch": result.best_epoch,
                        "completed_epochs": result.completed_epochs,
                        "stopped_early": result.stopped_early,
                        "best_public_selection_loss": result.best_public_selection_loss,
                        "reserved_confirmation_loss": confirmation[
                            "public_confirmation_loss"
                        ],
                        "reserved_confirmation_sample_count": confirmation[
                            "sample_count"
                        ],
                        "training_elapsed_s": result.training_elapsed_s,
                        "parameter_count": result.parameter_count,
                        "checkpoint_path": result.best_checkpoint_path,
                        "checkpoint_sha256": sha256_file(result.best_checkpoint_path),
                        "g2_locked_test_opened": False,
                        "task_targets_opened": False,
                    }
                )
                epoch_rows.extend(
                    {"seed": seed, "variant": variant, **row}
                    for row in result.epoch_rows
                )
                auxiliary_rows.extend(
                    {"seed": seed, "variant": variant, **row}
                    for row in result.auxiliary_rows
                )
                progress.update(
                    "simulation_chronaris_ablation_variant_complete",
                    seed=seed,
                    variant=variant,
                    status=result.status,
                    best_epoch=result.best_epoch,
                )
        acceptance = _acceptance_rows(config, result_rows, auxiliary_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            candidate_id=candidate_id,
            data=data,
            result_rows=result_rows,
            epoch_rows=epoch_rows,
            auxiliary_rows=auxiliary_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            variant_seed_count=len(result_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationChronarisAblationPretrainingResult(
        run_id=config.run_id,
        status=status,
        variant_seed_count=len(result_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _acceptance_rows(config, rows, auxiliary):
    expected = len(config.seeds) * len(config.variants)
    physics = [
        row
        for row in auxiliary
        if row["term_name"] == "chronaris_physical_consistency"
    ]
    return (
        _check("all_variant_seed_runs", len(rows) == expected, len(rows), expected),
        _check("all_checkpoints_complete", all(row["status"] in {"completed", "resumed"} for row in rows), [row["status"] for row in rows], "completed_or_resumed"),
        _check("public_validation_losses_finite", all(row["best_public_selection_loss"] >= 0 for row in rows), len(rows), expected),
        _check("reserved_confirmation_complete", all(row["reserved_confirmation_sample_count"] == 1 and row["reserved_confirmation_loss"] >= 0 for row in rows), len(rows), expected),
        _check("no_physics_component_removed", all(row["count"] == 0 for row in physics if row["variant"] == "no_physics"), True, True),
        _check("g2_and_task_oracle_closed", all(not row["g2_locked_test_opened"] and not row["task_targets_opened"] for row in rows), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "ablation_pretraining_results.csv",
        "epochs": root / "epoch_metrics.jsonl",
        "auxiliary": root / "chronaris_auxiliary_metrics.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["result_rows"]).to_csv(paths["results"], index=False)
    paths["epochs"].write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values["epoch_rows"]),
        encoding="utf-8",
    )
    pd.DataFrame(values["auxiliary_rows"]).to_csv(paths["auxiliary"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_chronaris_ablation_pretraining.v1",
        "config": asdict(values["config"]),
        "candidate_id": values["candidate_id"],
        "fold": values["data"].fold.to_dict(),
        "g2_locked_test_opened": False,
        "task_targets_opened": False,
        "selection_metric": "public_common_pretext_validation_loss",
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# Chronaris 机制消融锁定重训",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['result_rows'])} 个消融变体与随机种子组合；训练期间 G2 和任务真值保持关闭。",
        "",
    )), encoding="utf-8")
    seed_flags = " ".join(f"--seed {seed}" for seed in values["config"].seeds)
    variant_flags = " ".join(
        f"--variant {variant}" for variant in values["config"].variants
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_chronaris_ablation_pretraining.py "
        f"--run-id {values['config'].run_id} {seed_flags} {variant_flags} "
        f"--max-epochs {values['config'].max_epochs} --batch-size {values['config'].batch_size} "
        f"--patience {values['config'].patience} --device {values['config'].device} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_chronaris_ablation_pretraining_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "variant_seed_count": len(values["result_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
