"""Export complete simplified-task representation catalogs from locked encoders."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_locked_representation_run import (
    require_complete_dingxin_locked_checkpoints,
)
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
    DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
    CheckpointRegistry,
    ResumableOOFExporter,
    build_checkpoint_record,
    load_fusion_stream_batch,
    validate_fusion_method_alignment,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

from .simple_downstream_protocol import SIMPLE_DOWNSTREAM_METHODS


@dataclass(frozen=True, slots=True)
class SimpleDownstreamRepresentationConfig:
    run_id: str = "2026-07-16_simple-downstream-representations-smoke"
    pretraining_run_id: str = "2026-07-16_simple-downstream-pretraining-smoke"
    task_protocol_run_id: str = "2026-07-16_simple-downstream-protocol"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    fold_ids: tuple[str, ...] = ("leave_one_sortie_out__fold01",)
    seeds: tuple[int, ...] = (17,)
    fit_batch_size: int = 8
    export_batch_size: int = 8
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    maneuver_history_policy: str = DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimpleDownstreamRepresentationResult:
    run_id: str
    status: str
    compact_root: str
    heavy_root: str
    seed_fold_count: int
    export_count: int
    report_path: str
    evidence_manifest_path: str


def run_simple_downstream_representations(
    config: SimpleDownstreamRepresentationConfig,
) -> SimpleDownstreamRepresentationResult:
    if config.maneuver_history_policy != DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY:
        raise ValueError("simplified primary representations require past kinematics")
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    task_root = Path(config.heavy_output_root) / config.task_protocol_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    task_contexts = pd.read_csv(task_root / "context_manifest.csv")
    task_folds = pd.read_csv(task_root / "fold_manifest.csv")
    _validate_task_catalog(task_contexts, task_folds)

    pretraining_protocol_path = (
        Path(config.compact_output_root) / config.pretraining_run_id / "protocol.json"
    )
    pretraining_protocol = json.loads(
        pretraining_protocol_path.read_text(encoding="utf-8")
    )
    if pretraining_protocol.get("status", "completed") != "completed":
        raise ValueError("simplified representation source pretraining is incomplete")
    if pretraining_protocol.get("task_targets_opened") is not False:
        raise ValueError("simplified representation encoder opened task targets")
    if (
        pretraining_protocol.get("maneuver_history_policy")
        != config.maneuver_history_policy
    ):
        raise ValueError("simplified representation field policy mismatch")
    representation_family = str(pretraining_protocol["representation_family"])

    selected = json.loads(
        Path(config.selected_candidates_path).read_text(encoding="utf-8")
    )
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
    seed_fold_rows = []
    export_rows = []
    for seed in config.seeds:
        for fold_id in config.fold_ids:
            data = load_dingxin_fold_pretraining_data(
                fold_id=fold_id,
                snapshot_root=config.snapshot_root,
                fixed_audit_root=config.fixed_audit_root,
                inner_split_root=config.inner_split_root,
                maneuver_history_policy=config.maneuver_history_policy,
            )
            role_ids = _consumer_role_ids(task_contexts, task_folds, fold_id)
            available = set(data.index.contexts["context_id"].astype(str))
            missing = sorted(
                (set(role_ids["consumer_train"]) | set(role_ids["held_out"]))
                - available
            )
            if missing:
                raise ValueError(f"task contexts missing from representation index: {missing[:5]}")

            fold_root = compact_root / "folds" / f"seed_{seed}" / fold_id
            fold_root.mkdir(parents=True, exist_ok=True)
            registry = CheckpointRegistry(fold_root / "checkpoint_registry.json")
            adapters = {}
            normalizer = None
            normalizer_hashes = set()
            for method in TRAINABLE_FUSION_METHODS:
                checkpoint_path = checkpoints[(seed, fold_id, method)]
                device = chronaris_device if method == "chronaris" else baseline_device
                encoder, _heads, loaded_normalizer, payload = (
                    load_common_pretraining_checkpoint(checkpoint_path, device=device)
                )
                if payload["fold"]["fold_id"] != fold_id:
                    raise ValueError("simplified checkpoint fold mismatch")
                normalizer_hashes.add(
                    loaded_normalizer.to_manifest()["transform_sha256"]
                )
                normalizer = loaded_normalizer if normalizer is None else normalizer
                record = build_checkpoint_record(
                    method_name=method,
                    fold=data.fold,
                    checkpoint_path=checkpoint_path,
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
                raise ValueError("simplified fold normalizers differ by method")

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
                    data.load_batch,
                    train_sample_ids=data.fold.train_sample_ids,
                    held_out_sample_ids=(
                        data.fold.validation_sample_ids + data.fold.held_out_sample_ids
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
            outputs = {method: {} for method in SIMPLE_DOWNSTREAM_METHODS}
            local_rows = []
            for method in SIMPLE_DOWNSTREAM_METHODS:
                for role in ("consumer_train", "held_out"):
                    result = exporter.export_from_batch_provider(
                        encoder=adapters[method],
                        batch_provider=data.load_batch,
                        checkpoint=registry.require(method, fold_id),
                        export_role=role,
                        batch_size=config.export_batch_size,
                        catalog_sample_ids=role_ids[role],
                    )
                    local_rows.append(result)
                    outputs[method][role] = load_fusion_stream_batch(result.output_root)
                    export_rows.append({"seed": seed, **result.to_dict()})
            alignments = {
                role: validate_fusion_method_alignment(
                    [outputs[method][role] for method in SIMPLE_DOWNSTREAM_METHODS]
                )
                for role in ("consumer_train", "held_out")
            }
            _write_json(
                fold_root / "catalog_manifest.json",
                {
                    "format": "chronaris.simple_representation_catalog.v1",
                    "seed": seed,
                    "fold_id": fold_id,
                    "roles": {key: list(value) for key, value in role_ids.items()},
                    "alignment_sha256": alignments,
                    "schema_id": data.index.plan.schema.schema_id,
                    "schema_sha256": data.index.plan.schema.schema_sha256,
                    "maneuver_history_policy": config.maneuver_history_policy,
                    "task_targets_opened": False,
                    "outer_metrics_opened": False,
                },
            )
            seed_fold_rows.append(
                {
                    "seed": seed,
                    "fold_id": fold_id,
                    "consumer_train_count": len(role_ids["consumer_train"]),
                    "held_out_count": len(role_ids["held_out"]),
                    "checkpoint_count": len(registry.records),
                    "export_count": len(local_rows),
                    "alignment_role_count": len(alignments),
                    "schema_id": data.index.plan.schema.schema_id,
                    "schema_sha256": data.index.plan.schema.schema_sha256,
                }
            )
            del adapters, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    acceptance = _acceptance_rows(config, seed_fold_rows, export_rows)
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"
    paths = _write_outputs(
        compact_root=compact_root,
        heavy_root=heavy_root,
        config=config,
        selected_ids=selected_ids,
        seed_fold_rows=seed_fold_rows,
        export_rows=export_rows,
        acceptance=acceptance,
        status=status,
        representation_family=representation_family,
        baseline_device=baseline_device,
        chronaris_device=chronaris_device,
        task_context_hash=sha256_file(task_root / "context_manifest.csv"),
        pretraining_protocol_hash=sha256_file(pretraining_protocol_path),
    )
    return SimpleDownstreamRepresentationResult(
        run_id=config.run_id,
        status=status,
        compact_root=str(compact_root),
        heavy_root=str(heavy_root),
        seed_fold_count=len(seed_fold_rows),
        export_count=len(export_rows),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _consumer_role_ids(contexts, folds, fold_id):
    row = folds[folds["fold_id"].astype(str) == str(fold_id)]
    if len(row) != 1:
        raise ValueError(f"task catalog has no unique fold: {fold_id}")
    held_out_sortie = str(row.iloc[0]["held_out_sortie"])
    ordered = contexts.sort_values(
        ["sortie_id", "view_id", "target_start_offset_ms"], kind="mergesort"
    )
    return {
        "consumer_train": tuple(
            ordered[ordered["sortie_id"].astype(str) != held_out_sortie][
                "context_id"
            ].astype(str)
        ),
        "held_out": tuple(
            ordered[ordered["sortie_id"].astype(str) == held_out_sortie][
                "context_id"
            ].astype(str)
        ),
    }


def _validate_task_catalog(contexts, folds):
    if len(contexts) != 90 or contexts["context_id"].duplicated().any():
        raise ValueError("simplified representation task catalog must contain 90 IDs")
    if contexts["vehicle_context_id"].nunique() != 60:
        raise ValueError("simplified representation catalog must contain 60 vehicles")
    if len(folds) != 2 or folds["fold_id"].duplicated().any():
        raise ValueError("simplified representation catalog must contain two folds")


def _acceptance_rows(config, folds, exports):
    expected_units = len(config.seeds) * len(config.fold_ids)
    expected_exports = expected_units * len(SIMPLE_DOWNSTREAM_METHODS) * 2
    return (
        _check("all_seed_folds", len(folds) == expected_units, len(folds), expected_units),
        _check(
            "six_checkpoints_per_seed_fold",
            all(row["checkpoint_count"] == 6 for row in folds),
            [row["checkpoint_count"] for row in folds],
            6,
        ),
        _check(
            "two_complete_catalogs_per_method",
            len(exports) == expected_exports,
            len(exports),
            expected_exports,
        ),
        _check(
            "ninety_contexts_per_seed_fold",
            all(row["consumer_train_count"] + row["held_out_count"] == 90 for row in folds),
            [row["consumer_train_count"] + row["held_out_count"] for row in folds],
            90,
        ),
        _check(
            "future_prediction_schema",
            all(
                row["schema_id"] == "dingxin_future_prediction_common_observed.v1"
                for row in folds
            ),
            [row["schema_id"] for row in folds],
            "dingxin_future_prediction_common_observed.v1",
        ),
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
    paths["exports"].write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in values["export_rows"]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(
        paths["protocol"],
        {
            "format": "chronaris.simple_downstream_representation_protocol.v1",
            "status": values["status"],
            "config": asdict(values["config"]),
            "selected_candidates": values["selected_ids"],
            "baseline_device": values["baseline_device"],
            "chronaris_device": values["chronaris_device"],
            "representation_family": values["representation_family"],
            "model_input_bin_width_s": DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
            "task_context_sha256": values["task_context_hash"],
            "pretraining_protocol_sha256": values["pretraining_protocol_hash"],
            "task_targets_opened": False,
            "outer_metrics_opened": False,
        },
    )
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新简化下游统一表示导出",
                "",
                f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
                f"完成 {len(values['seed_fold_rows'])} 个随机种子—折和 {len(values['export_rows'])} 份表示目录。",
                "所有方法均覆盖完整的消费者训练清单与留出架次清单；编码器未读取下游目标，指标保持关闭。",
                "",
            )
        ),
        encoding="utf-8",
    )
    seed_flags = " ".join(
        f"--seed {seed}" for seed in values["config"].seeds
    )
    fold_flags = " ".join(
        f"--fold-id {fold_id}" for fold_id in values["config"].fold_ids
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/dingxin/run_simple_downstream_representations.py "
        f"--run-id {values['config'].run_id} "
        f"--pretraining-run-id {values['config'].pretraining_run_id} "
        f"--task-protocol-run-id {values['config'].task_protocol_run_id} "
        f"{seed_flags} {fold_flags} "
        f"--fit-batch-size {values['config'].fit_batch_size} "
        f"--export-batch-size {values['config'].export_batch_size} "
        f"--baseline-device {values['baseline_device']} "
        f"--chronaris-device {values['chronaris_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence"],
        {
            "format": "chronaris.simple_downstream_representation_evidence.v1",
            "run_id": values["config"].run_id,
            "status": values["status"],
            "seed_fold_count": len(values["seed_fold_rows"]),
            "export_count": len(values["export_rows"]),
            "acceptance_pass_count": passed,
            "acceptance_check_count": len(values["acceptance"]),
            "task_targets_opened": False,
            "outer_metrics_opened": False,
            "heavy_run_root": str(values["heavy_root"]),
            "output_paths": {key: str(path) for key, path in paths.items()},
        },
    )
    return paths


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("simplified representation device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("simplified representation requested unavailable CUDA")
    return value


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
