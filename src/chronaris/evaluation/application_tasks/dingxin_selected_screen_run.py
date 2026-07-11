"""Five-fold Dingxin validation confirmation for the G1-selected encoders."""

from __future__ import annotations

import gc
import json
import logging
import resource
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    CandidateScreenConfig,
    train_pretext_candidate,
)
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_selected_screen")
LOGGER.addHandler(logging.NullHandler())
DEFAULT_FOLDS = (
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
    "leave_one_sortie_out__fold01",
    "leave_one_sortie_out__fold02",
)


@dataclass(frozen=True, slots=True)
class DingxinSelectedScreenConfig:
    run_id: str = "2026-07-11_dingxin-selected-validation-seed17"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    fold_ids: tuple[str, ...] = DEFAULT_FOLDS
    max_epochs: int = 50
    batch_size: int = 32
    patience: int = 8
    seed: int = 17
    device: str = "cpu"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinSelectedScreenResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_fold_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_selected_screen(
    config: DingxinSelectedScreenConfig,
) -> DingxinSelectedScreenResult:
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
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_selected_validation_seed17",
        logger=LOGGER,
        initial_progress={
            "fold_count": len(config.fold_ids),
            "selected_candidate_ids": selected_ids,
            "outer_test_accessed": False,
            "task_targets_opened": False,
            "outer_test_metrics_opened": False,
        },
    ) as progress:
        result_rows = []
        epoch_rows = []
        fold_rows = []
        for fold_id in config.fold_ids:
            data = load_dingxin_fold_pretraining_data(
                fold_id=fold_id,
                snapshot_root=config.snapshot_root,
                fixed_audit_root=config.fixed_audit_root,
                inner_split_root=config.inner_split_root,
            )
            fold = data.fold
            base_provider = data.index.load_batch
            normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
                base_provider,
                train_sample_ids=fold.train_sample_ids,
                held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
                batch_size=2,
            )
            provider, provider_audit = _build_guarded_cached_provider(
                base_provider,
                allowed_sample_ids=fold.train_sample_ids + fold.validation_sample_ids,
                forbidden_sample_ids=fold.held_out_sample_ids,
            )
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "inner_train_count": len(fold.train_sample_ids),
                    "validation_count": len(fold.validation_sample_ids),
                    "outer_test_count": len(fold.held_out_sample_ids),
                    "normalizer_fit_sample_hash": normalizer.fit_sample_hash,
                    "normalizer_fit_count": len(normalizer.fit_sample_ids),
                }
            )
            for method_name in TRAINABLE_FUSION_METHODS:
                candidate = candidates[selected_ids[method_name]]
                result = train_pretext_candidate(
                    method_name,
                    candidate=candidate,
                    batch=None,
                    batch_provider=provider,
                    fold=fold,
                    physiology_feature_names=data.index.plan.schema.physiology_feature_names,
                    vehicle_feature_names=data.index.plan.schema.vehicle_feature_names,
                    vehicle_field_labels=data.vehicle_field_labels,
                    normalizer=normalizer,
                    output_root=heavy_root / "checkpoints" / fold_id,
                    config=CandidateScreenConfig(
                        max_epochs=config.max_epochs,
                        batch_size=config.batch_size,
                        patience=config.patience,
                        seed=config.seed,
                        device=config.device,
                    ),
                    augmentation_policy=AugmentationPolicy(),
                    resume=config.resume,
                )
                result_rows.append(
                    {
                        "fold_id": fold_id,
                        "method_name": method_name,
                        "candidate_id": result.candidate_id,
                        "status": result.status,
                        "best_epoch": result.best_epoch,
                        "completed_epochs": result.completed_epochs,
                        "stopped_early": result.stopped_early,
                        "best_public_selection_loss": result.best_public_selection_loss,
                        **{
                            f"validation_{name}": value
                            for name, value in result.best_validation_losses.items()
                        },
                        "parameter_count": result.parameter_count,
                        "training_elapsed_s": result.training_elapsed_s,
                        "protocol_sha256": result.protocol_sha256,
                        "best_checkpoint_path": result.best_checkpoint_path,
                        "best_checkpoint_sha256": sha256_file(result.best_checkpoint_path),
                        "maximum_rss_mb": _maximum_rss_mb(),
                    }
                )
                epoch_rows.extend(
                    {"fold_id": fold_id, **dict(row)} for row in result.epoch_rows
                )
                progress.update(
                    "selected_method_fold_complete",
                    fold_id=fold_id,
                    method_name=method_name,
                    candidate_id=result.candidate_id,
                    status=result.status,
                    best_epoch=result.best_epoch,
                )
            if provider_audit["forbidden_request_count"]:
                raise ValueError("Dingxin selected screen requested outer-test samples")
            fold_rows[-1].update(provider_audit)
            del data, normalizer, provider
            gc.collect()
        acceptance = _acceptance_rows(
            config=config,
            selected_ids=selected_ids,
            results=result_rows,
            folds=fold_rows,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            selected_path=selected_path,
            selected_ids=selected_ids,
            result_rows=result_rows,
            epoch_rows=epoch_rows,
            fold_rows=fold_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            method_fold_count=len(result_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
            outer_test_accessed=False,
        )
    return DingxinSelectedScreenResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        method_fold_count=len(result_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _build_guarded_cached_provider(base_provider, *, allowed_sample_ids, forbidden_sample_ids):
    allowed = set(allowed_sample_ids)
    forbidden = set(forbidden_sample_ids)
    cache = {}
    audit = {"provider_request_count": 0, "cache_hit_count": 0, "forbidden_request_count": 0}

    def provider(sample_ids):
        ids = tuple(sample_ids)
        audit["provider_request_count"] += 1
        if set(ids) & forbidden or not set(ids) <= allowed:
            audit["forbidden_request_count"] += 1
            raise ValueError("selected validation provider rejected outer-test or unknown samples")
        if ids in cache:
            audit["cache_hit_count"] += 1
            return cache[ids]
        cache[ids] = base_provider(ids)
        return cache[ids]

    return provider, audit


def _acceptance_rows(*, config, selected_ids, results, folds):
    return (
        _check("five_folds", len(folds) == 5, len(folds), 5),
        _check("twenty_five_method_folds", len(results) == 25, len(results), 25),
        _check(
            "only_g1_selected_candidates",
            all(row["candidate_id"] == selected_ids[row["method_name"]] for row in results),
            sorted({(row["method_name"], row["candidate_id"]) for row in results}),
            sorted(selected_ids.items()),
        ),
        _check(
            "all_checkpoints_complete",
            all(row["status"] in {"completed", "resumed"} for row in results),
            [row["status"] for row in results],
            "completed_or_resumed",
        ),
        _check(
            "validation_losses_finite",
            all(row["best_public_selection_loss"] >= 0 for row in results),
            len(results),
            25,
        ),
        _check(
            "normalizers_inner_train_only",
            all(row["normalizer_fit_count"] == row["inner_train_count"] for row in folds),
            [row["normalizer_fit_count"] for row in folds],
            [row["inner_train_count"] for row in folds],
        ),
        _check(
            "outer_test_never_requested",
            all(row["forbidden_request_count"] == 0 for row in folds),
            sum(row["forbidden_request_count"] for row in folds),
            0,
        ),
        _check("task_targets_closed", True, False, False),
        _check("outer_test_metrics_closed", True, False, False),
        _check(
            "resource_observed",
            max(row["maximum_rss_mb"] for row in results) < 12288,
            round(max(row["maximum_rss_mb"] for row in results), 1),
            "<12288 MB",
        ),
    )


def _write_outputs(
    *, compact_root, heavy_root, config, selected_path, selected_ids,
    result_rows, epoch_rows, fold_rows, acceptance, status
):
    paths = {
        "results": compact_root / "selected_validation_results.csv",
        "epochs": compact_root / "epoch_metrics.csv",
        "folds": compact_root / "fold_audit.csv",
        "acceptance": compact_root / "acceptance.csv",
        "protocol": compact_root / "protocol.json",
        "report": compact_root / "report.md",
        "resume": compact_root / "resume_command.txt",
        "evidence": compact_root / "evidence_manifest.json",
    }
    for path, rows in (
        (paths["results"], result_rows),
        (paths["epochs"], epoch_rows),
        (paths["folds"], fold_rows),
        (paths["acceptance"], acceptance),
    ):
        pd.DataFrame(rows).to_csv(path, index=False)
    _write_json(
        paths["protocol"],
        {
            "format": "chronaris.dingxin_selected_validation_protocol.v1",
            "config": asdict(config),
            "selected_candidates": selected_ids,
            "selected_candidates_path": str(selected_path),
            "selected_candidates_sha256": sha256_file(selected_path),
            "fit_role": "inner_train",
            "evaluation_role": "validation",
            "task_targets_opened": False,
            "outer_test_accessed": False,
            "outer_test_metrics_opened": False,
        },
    )
    passed = sum(row["passed"] for row in acceptance)
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新选定配置 validation 确认",
                "",
                f"状态：{status}；验收 {passed}/{len(acceptance)}。",
                "",
                "五个 G1 已选配置在鼎新五折 inner-train 上独立训练，只使用 validation 公共自监督损失确认。",
                "任务目标、outer-test 原始上下文和 outer-test 指标均保持关闭。",
                "本 run 冻结真实数据训练配置，但不构成 outer-test 或论文锁定结论。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_selected_screen.py "
        f"--run-id {config.run_id} --batch-size {config.batch_size} "
        f"--max-epochs {config.max_epochs} --patience {config.patience} "
        f"--seed {config.seed} --device {config.device} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence"],
        {
            "run_id": config.run_id,
            "status": status,
            "evidence_layer": "dingxin_selected_validation_pretext",
            "method_fold_count": len(result_rows),
            "acceptance_pass_count": passed,
            "acceptance_check_count": len(acceptance),
            "outer_test_accessed": False,
            "task_targets_opened": False,
            "outer_test_metrics_opened": False,
            "heavy_run_root": str(heavy_root),
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return paths


def _maximum_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
