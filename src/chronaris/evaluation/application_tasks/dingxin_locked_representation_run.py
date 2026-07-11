"""Export six-method Dingxin representations for every locked fold and seed."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    DEFAULT_FOLDS,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.modeling.training import (
    TRAINABLE_FUSION_METHODS,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import (
    CheckpointRegistry,
    ResumableOOFExporter,
    build_checkpoint_record,
    load_fusion_stream_batch,
    validate_fusion_method_alignment,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_locked_representations")
LOGGER.addHandler(logging.NullHandler())
SIX_METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)


@dataclass(frozen=True, slots=True)
class DingxinLockedRepresentationConfig:
    run_id: str = "2026-07-12_dingxin-locked-representations"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_dingxin-locked-pretraining"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    fold_ids: tuple[str, ...] = DEFAULT_FOLDS
    seeds: tuple[int, ...] = LOCKED_SEEDS
    fit_batch_size: int = 8
    export_batch_size: int = 8
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinLockedRepresentationResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    seed_fold_count: int
    export_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_locked_representations(config: DingxinLockedRepresentationConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    pretraining_protocol = json.loads(
        (
            Path(config.compact_output_root)
            / config.pretraining_run_id
            / "protocol.json"
        ).read_text(encoding="utf-8")
    )
    representation_family = str(pretraining_protocol.get("representation_family"))
    if representation_family not in {
        "frozen_task_agnostic_v1",
        "synthetic_pretrain_real_adapt_v1",
    }:
        raise ValueError("Dingxin pretraining representation family is unsupported")
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(config.selected_candidates_path).read_text(encoding="utf-8"))
    selected_ids = {
        method: str(selected[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    checkpoints = require_complete_dingxin_locked_checkpoints(
        pretraining_root,
        seeds=config.seeds,
        fold_ids=config.fold_ids,
        selected_ids=selected_ids,
    )
    baseline_device = _resolve_device(config.baseline_device)
    chronaris_device = _resolve_device(config.chronaris_device)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_locked_representation_export",
        logger=LOGGER,
        initial_progress={
            "checkpoint_count_verified_before_outer_test": len(checkpoints),
            "task_targets_opened": False,
            "outer_test_metrics_opened": False,
            "representation_family": representation_family,
        },
    ) as progress:
        seed_fold_rows = []
        export_rows = []
        for seed in config.seeds:
            for fold_id in config.fold_ids:
                data = load_dingxin_fold_pretraining_data(
                    fold_id=fold_id,
                    snapshot_root=config.snapshot_root,
                    fixed_audit_root=config.fixed_audit_root,
                    inner_split_root=config.inner_split_root,
                )
                provider = data.index.load_batch
                fold_root = compact_root / "folds" / f"seed_{seed}" / fold_id
                fold_root.mkdir(parents=True, exist_ok=True)
                registry = CheckpointRegistry(fold_root / "checkpoint_registry.json")
                adapters = {}
                normalizer = None
                normalizer_hashes = set()
                for method in TRAINABLE_FUSION_METHODS:
                    path = checkpoints[(seed, fold_id, method)]
                    device = chronaris_device if method == "chronaris" else baseline_device
                    encoder, _heads, loaded_normalizer, payload = (
                        load_common_pretraining_checkpoint(path, device=device)
                    )
                    if payload["fold"]["fold_id"] != fold_id:
                        raise ValueError("Dingxin locked checkpoint fold mismatch")
                    normalizer_hashes.add(
                        loaded_normalizer.to_manifest()["transform_sha256"]
                    )
                    normalizer = loaded_normalizer if normalizer is None else normalizer
                    record = build_checkpoint_record(
                        method_name=method,
                        fold=data.fold,
                        checkpoint_path=path,
                        seed=seed,
                    )
                    registry.register(record, replace_existing=True)
                    adapters[method] = TrainedFusionAdapter(
                        encoder=encoder,
                        normalizer=loaded_normalizer,
                        fold_id=fold_id,
                        checkpoint_sha256=record.checkpoint_sha256,
                    )
                if len(normalizer_hashes) != 1 or normalizer is None:
                    raise ValueError("Dingxin locked fold normalizers differ by method")
                naive_path = (
                    heavy_root
                    / "checkpoints"
                    / f"seed_{seed}"
                    / fold_id
                    / "naive_time_sync"
                    / "best.pt"
                )
                if not naive_path.is_file() or not config.resume:
                    naive = NaiveTimeSyncEncoder().fit_from_batch_provider(
                        provider,
                        train_sample_ids=data.fold.train_sample_ids,
                        held_out_sample_ids=(
                            data.fold.validation_sample_ids
                            + data.fold.held_out_sample_ids
                        ),
                        normalizer=normalizer,
                        batch_size=config.fit_batch_size,
                    )
                    save_naive_time_sync_checkpoint(naive_path, encoder=naive)
                naive_record = build_checkpoint_record(
                    method_name="naive_time_sync",
                    fold=data.fold,
                    checkpoint_path=naive_path,
                    seed=seed,
                )
                registry.register(naive_record, replace_existing=True)
                adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
                    encoder=load_naive_time_sync_checkpoint(naive_path),
                    fold_id=fold_id,
                    checkpoint_sha256=naive_record.checkpoint_sha256,
                )
                exporter = ResumableOOFExporter(
                    heavy_root / "representations" / f"seed_{seed}" / fold_id,
                    resume=config.resume,
                )
                outputs = {method: {} for method in SIX_METHODS}
                local_rows = []
                for method in SIX_METHODS:
                    for role in ("train", "validation", "held_out"):
                        result = exporter.export_from_batch_provider(
                            encoder=adapters[method],
                            batch_provider=provider,
                            checkpoint=registry.require(method, fold_id),
                            export_role=role,
                            batch_size=config.export_batch_size,
                        )
                        local_rows.append(result)
                        outputs[method][role] = load_fusion_stream_batch(
                            result.output_root
                        )
                        export_rows.append({"seed": seed, **result.to_dict()})
                alignments = {
                    role: validate_fusion_method_alignment(
                        [outputs[method][role] for method in SIX_METHODS]
                    )
                    for role in ("train", "validation", "held_out")
                }
                _write_json(fold_root / "split_manifest.json", data.fold.to_dict())
                _write_json(
                    fold_root / "representation_export_manifest.json",
                    {
                        "format": "chronaris.dingxin_locked_representation_fold.v1",
                        "seed": seed,
                        "fold_id": fold_id,
                        "exports": [row.to_dict() for row in local_rows],
                        "alignment_sha256": alignments,
                        "task_targets_opened": False,
                        "outer_test_metrics_opened": False,
                        "representation_family": representation_family,
                    },
                )
                seed_fold_rows.append(
                    {
                        "seed": seed,
                        "fold_id": fold_id,
                        "checkpoint_count": len(registry.records),
                        "export_count": len(local_rows),
                        "alignment_role_count": len(alignments),
                        "outer_test_representation_exported": True,
                        "outer_test_metrics_opened": False,
                    }
                )
                progress.update(
                    "dingxin_locked_seed_fold_representation_complete",
                    seed=seed,
                    fold_id=fold_id,
                    export_count=len(local_rows),
                )
                del adapters, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        acceptance = _acceptance_rows(config, seed_fold_rows, export_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            baseline_device=baseline_device,
            chronaris_device=chronaris_device,
            seed_fold_rows=seed_fold_rows,
            export_rows=export_rows,
            acceptance=acceptance,
            status=status,
            representation_family=representation_family,
        )
        progress.finish(
            status=status,
            seed_fold_count=len(seed_fold_rows),
            export_count=len(export_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return DingxinLockedRepresentationResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        seed_fold_count=len(seed_fold_rows),
        export_count=len(export_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def require_complete_dingxin_locked_checkpoints(root, *, seeds, fold_ids, selected_ids):
    paths = {}
    for seed in seeds:
        for fold_id in fold_ids:
            for method in TRAINABLE_FUSION_METHODS:
                path = (
                    root / "checkpoints" / f"seed_{seed}" / fold_id / method / "best.pt"
                    if method == "chronaris"
                    else root / "checkpoints" / f"seed_{seed}" / fold_id / method / selected_ids[method] / "best.pt"
                )
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("training_status") != "completed" or int(payload["seed"]) != seed:
                    raise ValueError("Dingxin locked checkpoint incomplete or seed-mismatched")
                paths[(seed, fold_id, method)] = path
    return paths


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("Dingxin locked representation device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("Dingxin locked representation requested unavailable CUDA")
    return value


def _acceptance_rows(config, folds, exports):
    expected_folds = len(config.seeds) * len(config.fold_ids)
    expected_exports = expected_folds * 18
    return (
        _check("all_seed_folds", len(folds) == expected_folds, len(folds), expected_folds),
        _check("six_checkpoints_per_seed_fold", all(row["checkpoint_count"] == 6 for row in folds), [row["checkpoint_count"] for row in folds], 6),
        _check("eighteen_exports_per_seed_fold", all(row["export_count"] == 18 for row in folds), [row["export_count"] for row in folds], 18),
        _check("all_representation_exports", len(exports) == expected_exports, len(exports), expected_exports),
        _check("three_aligned_roles", all(row["alignment_role_count"] == 3 for row in folds), [row["alignment_role_count"] for row in folds], 3),
        _check("outer_test_metrics_closed", all(not row["outer_test_metrics_opened"] for row in folds), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "folds": root / "seed_fold_inventory.csv",
        "exports": root / "representation_inventory.jsonl",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["seed_fold_rows"]).to_csv(paths["folds"], index=False)
    paths["exports"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values["export_rows"]), encoding="utf-8")
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.dingxin_locked_representation_protocol.v1",
        "config": asdict(values["config"]),
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
        "checkpoint_set_verified_before_outer_test": True,
        "task_targets_opened": False,
        "outer_test_metrics_opened": False,
        "representation_family": values["representation_family"],
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# 鼎新三随机种子五折六方法表示导出",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['seed_fold_rows'])} 个随机种子—折和 {len(values['export_rows'])} 份统一表示。",
        "所有训练 checkpoint 完成后才导出 outer-test 表示；任务目标、consumer 和 outer-test 指标保持关闭。",
        "",
    )), encoding="utf-8")
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_locked_representations.py "
        f"--run-id {values['config'].run_id} "
        f"--pretraining-run-id {values['config'].pretraining_run_id} "
        f"--export-batch-size {values['config'].export_batch_size} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.dingxin_locked_representation_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "seed_fold_count": len(values["seed_fold_rows"]),
        "export_count": len(values["export_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "outer_test_metrics_opened": False,
        "representation_family": values["representation_family"],
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
