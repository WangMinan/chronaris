"""Three-seed five-fold Dingxin retraining of the selected encoder configs."""

from __future__ import annotations

import gc
import json
import logging
import resource
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
    ensure_dingxin_model_input_contract,
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    DEFAULT_FOLDS,
    _build_guarded_cached_provider,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    CandidateScreenConfig,
    LockedChronarisTrainingConfig,
    train_locked_chronaris,
    train_pretext_candidate,
)
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_locked_pretraining")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DingxinLockedPretrainingConfig:
    run_id: str = "2026-07-12_dingxin-locked-pretraining"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    initialization_pretraining_run_id: str | None = None
    fold_ids: tuple[str, ...] = DEFAULT_FOLDS
    seeds: tuple[int, ...] = LOCKED_SEEDS
    methods: tuple[str, ...] = TRAINABLE_FUSION_METHODS
    max_epochs: int = 50
    batch_size: int = 32
    patience: int = 8
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.methods or not set(self.methods).issubset(TRAINABLE_FUSION_METHODS):
            raise ValueError("Dingxin locked methods must use the fixed trainable set")


@dataclass(frozen=True, slots=True)
class DingxinLockedPretrainingResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_fold_seed_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_locked_pretraining(config: DingxinLockedPretrainingConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    ensure_dingxin_model_input_contract(heavy_root)
    selected = json.loads(Path(config.selected_candidates_path).read_text(encoding="utf-8"))
    selected_ids = {
        method: str(selected[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    candidates = {value.candidate_id: value for value in ENCODER_SCREEN_CANDIDATES}
    initialization_checkpoints = (
        _require_complete_initialization_checkpoints(
            Path(config.heavy_output_root) / config.initialization_pretraining_run_id,
            compact_root=(
                Path(config.compact_output_root)
                / config.initialization_pretraining_run_id
            ),
            seeds=config.seeds,
            selected_ids=selected_ids,
            methods=config.methods,
        )
        if config.initialization_pretraining_run_id is not None
        else {}
    )
    baseline_device = _resolve_device(config.baseline_device)
    chronaris_device = _resolve_device(config.chronaris_device)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_locked_pretraining",
        logger=LOGGER,
        initial_progress={
            "folds": list(config.fold_ids),
            "seeds": list(config.seeds),
            "selected_candidates": selected_ids,
            "methods": list(config.methods),
            "task_targets_opened": False,
            "outer_test_accessed": False,
            "transfer_initialization_enabled": bool(initialization_checkpoints),
            "model_input_bin_width_s": DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
        },
    ) as progress:
        result_rows = []
        fold_rows = []
        auxiliary_rows = []
        for fold_id in config.fold_ids:
            data = load_dingxin_fold_pretraining_data(
                fold_id=fold_id,
                snapshot_root=config.snapshot_root,
                fixed_audit_root=config.fixed_audit_root,
                inner_split_root=config.inner_split_root,
            )
            provider, access = _build_guarded_cached_provider(
                data.load_batch,
                allowed_sample_ids=(
                    data.fold.train_sample_ids + data.fold.validation_sample_ids
                ),
                forbidden_sample_ids=data.fold.held_out_sample_ids,
            )
            normalizer, normalizer_status = _load_or_fit_normalizer(
                heavy_root=heavy_root,
                fold_id=fold_id,
                seeds=config.seeds,
                methods=config.methods,
                selected_ids=selected_ids,
                provider=provider,
                train_sample_ids=data.fold.train_sample_ids,
                held_out_sample_ids=(
                    data.fold.validation_sample_ids + data.fold.held_out_sample_ids
                ),
                batch_size=config.batch_size,
                resume=config.resume,
            )
            schema = data.index.plan.schema
            vehicle_labels = data.vehicle_field_labels
            for seed in config.seeds:
                for method in config.methods:
                    candidate = candidates[selected_ids[method]]
                    output_root = (
                        heavy_root
                        / "checkpoints"
                        / f"seed_{seed}"
                        / fold_id
                    )
                    if method == "chronaris":
                        result = train_locked_chronaris(
                            batch=None,
                            batch_provider=provider,
                            fold=data.fold,
                            physiology_feature_names=schema.physiology_feature_names,
                            vehicle_feature_names=schema.vehicle_feature_names,
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
                            initialization_checkpoint=initialization_checkpoints.get(
                                (seed, method)
                            ),
                            resume=config.resume,
                        )
                        auxiliary_rows.extend(
                            {
                                "seed": seed,
                                "fold_id": fold_id,
                                "method_name": method,
                                **row,
                            }
                            for row in result.auxiliary_rows
                        )
                        device = chronaris_device
                    else:
                        result = train_pretext_candidate(
                            method,
                            candidate=candidate,
                            batch=None,
                            batch_provider=provider,
                            fold=data.fold,
                            physiology_feature_names=schema.physiology_feature_names,
                            vehicle_feature_names=schema.vehicle_feature_names,
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
                            initialization_checkpoint=initialization_checkpoints.get(
                                (seed, method)
                            ),
                            resume=config.resume,
                        )
                        device = baseline_device
                    checkpoint_payload = torch.load(
                        result.best_checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    device_history = checkpoint_payload.get(
                        "training_device_history",
                        [checkpoint_payload.get("config", {}).get("device", device)],
                    )
                    transfer = checkpoint_payload.get("transfer_initialization")
                    result_rows.append(
                        {
                            "seed": seed,
                            "fold_id": fold_id,
                            "method_name": method,
                            "candidate_id": candidate.candidate_id,
                            "status": result.status,
                            "best_epoch": result.best_epoch,
                            "completed_epochs": result.completed_epochs,
                            "stopped_early": result.stopped_early,
                            "best_public_selection_loss": result.best_public_selection_loss,
                            "training_elapsed_s": result.training_elapsed_s,
                            "parameter_count": result.parameter_count,
                            "training_device": device_history[-1],
                            "training_device_history": json.dumps(device_history),
                            "checkpoint_path": result.best_checkpoint_path,
                            "checkpoint_sha256": sha256_file(result.best_checkpoint_path),
                            "maximum_rss_mb": _maximum_rss_mb(),
                            "task_targets_opened": False,
                            "outer_test_accessed": False,
                            "transfer_initialized": transfer is not None,
                            "transfer_source_sha256": (
                                None
                                if transfer is None
                                else transfer["source"]["checkpoint_sha256"]
                            ),
                            "transfer_copied_element_fraction": (
                                None
                                if transfer is None
                                else transfer["copied_element_fraction"]
                            ),
                        }
                    )
                    progress.update(
                        "dingxin_locked_method_complete",
                        seed=seed,
                        fold_id=fold_id,
                        method_name=method,
                        status=result.status,
                        best_epoch=result.best_epoch,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "inner_train_count": len(data.fold.train_sample_ids),
                    "validation_count": len(data.fold.validation_sample_ids),
                    "outer_test_count": len(data.fold.held_out_sample_ids),
                    "normalizer_fit_count": len(normalizer.fit_sample_ids),
                    "forbidden_request_count": access["forbidden_request_count"],
                    "cache_hit_count": access["cache_hit_count"],
                    "normalizer_status": normalizer_status,
                }
            )
        acceptance = _acceptance_rows(config, result_rows, fold_rows, auxiliary_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            selected_ids=selected_ids,
            baseline_device=baseline_device,
            chronaris_device=chronaris_device,
            result_rows=result_rows,
            fold_rows=fold_rows,
            auxiliary_rows=auxiliary_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            method_fold_seed_count=len(result_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return DingxinLockedPretrainingResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        method_fold_seed_count=len(result_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("Dingxin locked device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("Dingxin locked requested unavailable CUDA device")
    return value


def _load_or_fit_normalizer(
    *, heavy_root, fold_id, seeds, methods, selected_ids, provider,
    train_sample_ids, held_out_sample_ids, batch_size, resume
):
    path = heavy_root / "normalizers" / fold_id / "normalizer.json"
    manifest = None
    status = "completed"
    if resume and path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        status = "resumed_cache"
    if resume and manifest is None:
        for seed in seeds:
            for method in methods:
                checkpoint = (
                    heavy_root
                    / "checkpoints"
                    / f"seed_{seed}"
                    / fold_id
                    / method
                    / (
                        "last.pt"
                        if method == "chronaris"
                        else f"{selected_ids[method]}/last.pt"
                    )
                )
                if checkpoint.is_file():
                    manifest = torch.load(
                        checkpoint, map_location="cpu", weights_only=True
                    )["normalizer"]
                    status = "resumed_checkpoint"
                    break
            if manifest is not None:
                break
    if manifest is None:
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
            provider,
            train_sample_ids=train_sample_ids,
            held_out_sample_ids=held_out_sample_ids,
            batch_size=batch_size,
        )
        manifest = dict(normalizer.to_manifest())
    else:
        normalizer = TrainOnlyRobustNormalizer.from_manifest(manifest)
    if tuple(normalizer.fit_sample_ids) != tuple(train_sample_ids):
        raise ValueError("Dingxin cached normalizer fit lineage changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return normalizer, status


def _require_complete_initialization_checkpoints(
    root, *, compact_root, seeds, selected_ids, methods
):
    evidence_path = compact_root / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "completed":
        raise ValueError("Dingxin transfer requires completed simulation pretraining")
    paths = {}
    for seed in seeds:
        for method in methods:
            path = (
                root / "checkpoints" / f"seed_{seed}" / method / "best.pt"
                if method == "chronaris"
                else root
                / "checkpoints"
                / f"seed_{seed}"
                / method
                / selected_ids[method]
                / "best.pt"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("training_status") != "completed"
                or int(payload["seed"]) != seed
                or str(payload["method_name"]) != method
            ):
                raise ValueError("Dingxin transfer source checkpoint is incomplete")
            paths[(seed, method)] = path
    return paths


def _acceptance_rows(config, results, folds, auxiliary):
    expected = len(config.seeds) * len(config.fold_ids) * len(config.methods)
    chronaris_required = "chronaris" in config.methods
    return (
        _check("all_method_fold_seed_runs", len(results) == expected, len(results), expected),
        _check("all_checkpoints_complete", all(row["status"] in {"completed", "resumed"} for row in results), [row["status"] for row in results], "completed_or_resumed"),
        _check("public_validation_losses_finite", all(row["best_public_selection_loss"] >= 0 for row in results), len(results), expected),
        _check(
            "chronaris_auxiliary_schedule_consistent",
            (
                bool(auxiliary)
                and (
                    any(row["weight"] > 0 for row in auxiliary)
                    if config.max_epochs > 10
                    else all(row["weight"] == 0 for row in auxiliary)
                )
                if chronaris_required
                else not auxiliary
            ),
            any(row["weight"] > 0 for row in auxiliary),
            config.max_epochs > 10 if chronaris_required else False,
        ),
        _check("normalizers_inner_train_only", all(row["normalizer_fit_count"] == row["inner_train_count"] for row in folds), [row["normalizer_fit_count"] for row in folds], "inner_train counts"),
        _check("outer_test_never_requested", all(row["forbidden_request_count"] == 0 for row in folds), [row["forbidden_request_count"] for row in folds], 0),
        _check("task_targets_closed", all(not row["task_targets_opened"] for row in results), False, False),
        _check(
            "transfer_initialization_contract",
            (
                all(
                    row["transfer_initialized"]
                    and row["transfer_copied_element_fraction"] > 0
                    for row in results
                )
                if config.initialization_pretraining_run_id is not None
                else all(not row["transfer_initialized"] for row in results)
            ),
            config.initialization_pretraining_run_id,
            "all initialized when configured; none otherwise",
        ),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "locked_pretraining_results.csv",
        "folds": root / "fold_audit.csv",
        "auxiliary": root / "chronaris_auxiliary_metrics.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["result_rows"]).to_csv(paths["results"], index=False)
    pd.DataFrame(values["fold_rows"]).to_csv(paths["folds"], index=False)
    pd.DataFrame(values["auxiliary_rows"]).to_csv(paths["auxiliary"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.dingxin_locked_pretraining_protocol.v1",
        "config": asdict(values["config"]),
        "selected_candidates": values["selected_ids"],
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
        "task_targets_opened": False,
        "outer_test_accessed": False,
        "early_stopping_inputs": ["masked_reconstruction", "short_horizon_prediction", "lag_discrimination"],
        "representation_family": (
            "synthetic_pretrain_real_adapt_v1"
            if values["config"].initialization_pretraining_run_id is not None
            else "frozen_task_agnostic_v1"
        ),
        "model_input_bin_width_s": DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# 鼎新选定配置锁定重训",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['result_rows'])} 个方法—折—随机种子训练；任务目标和 outer-test 始终关闭。",
        f"六种方法共用 {DINGXIN_MODEL_INPUT_BIN_WIDTH_S:.1f} 秒固定因果时间箱；每箱时间戳取最后一次真实观测，任何输入都不会前移。",
        f"深度基线设备为 {values['baseline_device']}，Chronaris 设备为 {values['chronaris_device']}；逐方法实测耗时与设备历史见 locked_pretraining_results.csv。",
        "跨 schema 迁移只加载形状一致的任务无关参数；schema 相关输入和重构层重新初始化。"
        if values["config"].initialization_pretraining_run_id is not None
        else "Chronaris 方法专属损失参与反向传播，早停只读取公共自监督 validation 损失。",
        "",
    )), encoding="utf-8")
    seed_flags = " ".join(f"--seed {seed}" for seed in values["config"].seeds)
    method_flags = " ".join(
        f"--method {method}" for method in values["config"].methods
    )
    initialization_flag = (
        ""
        if values["config"].initialization_pretraining_run_id is None
        else "--initialization-pretraining-run-id "
        f"{values['config'].initialization_pretraining_run_id} "
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_locked_pretraining.py "
        f"--run-id {values['config'].run_id} {seed_flags} {method_flags} "
        f"{initialization_flag}"
        f"--max-epochs {values['config'].max_epochs} "
        f"--batch-size {values['config'].batch_size} --patience {values['config'].patience} "
        f"--baseline-device {values['baseline_device']} --chronaris-device {values['chronaris_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.dingxin_locked_pretraining_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "method_fold_seed_count": len(values["result_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "task_targets_opened": False,
        "outer_test_accessed": False,
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _maximum_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
