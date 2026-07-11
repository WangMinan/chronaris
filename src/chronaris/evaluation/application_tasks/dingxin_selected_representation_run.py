"""Export six-method Dingxin representations from the selected validation checkpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    ensure_dingxin_model_input_contract,
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    DEFAULT_FOLDS,
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


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_selected_representations")
LOGGER.addHandler(logging.NullHandler())
SIX_METHODS = (
    *TRAINABLE_FUSION_METHODS[:2],
    "naive_time_sync",
    *TRAINABLE_FUSION_METHODS[2:],
)


@dataclass(frozen=True, slots=True)
class DingxinSelectedRepresentationConfig:
    run_id: str = "2026-07-12_dingxin-selected-representations-seed17"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    selected_screen_run_id: str = "2026-07-11_dingxin-selected-validation-seed17"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    fold_ids: tuple[str, ...] = DEFAULT_FOLDS
    seed: int = 17
    fit_batch_size: int = 2
    export_batch_size: int = 2
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinSelectedRepresentationResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    fold_count: int
    export_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_selected_representations(config: DingxinSelectedRepresentationConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    screen_heavy_root = Path(config.heavy_output_root) / config.selected_screen_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    ensure_dingxin_model_input_contract(screen_heavy_root)
    selected = json.loads(Path(config.selected_candidates_path).read_text(encoding="utf-8"))
    selected_ids = {
        method: str(selected[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_selected_representation_export",
        logger=LOGGER,
        initial_progress={
            "fold_count": len(config.fold_ids),
            "outer_test_representation_export": True,
            "task_targets_opened": False,
            "outer_test_metrics_opened": False,
        },
    ) as progress:
        fold_rows = []
        export_rows = []
        for fold_id in config.fold_ids:
            data = load_dingxin_fold_pretraining_data(
                fold_id=fold_id,
                snapshot_root=config.snapshot_root,
                fixed_audit_root=config.fixed_audit_root,
                inner_split_root=config.inner_split_root,
            )
            provider = data.load_batch
            fold = data.fold
            fold_compact_root = compact_root / "folds" / fold_id
            fold_compact_root.mkdir(parents=True, exist_ok=True)
            registry = CheckpointRegistry(fold_compact_root / "checkpoint_registry.json")
            adapters = {}
            normalizer = None
            normalizer_hashes = set()
            for method_name in TRAINABLE_FUSION_METHODS:
                checkpoint_path = (
                    screen_heavy_root
                    / "checkpoints"
                    / fold_id
                    / method_name
                    / selected_ids[method_name]
                    / "best.pt"
                )
                encoder, _heads, loaded_normalizer, payload = (
                    load_common_pretraining_checkpoint(checkpoint_path)
                )
                if payload["fold"]["fold_id"] != fold_id:
                    raise ValueError("selected checkpoint fold mismatch")
                if payload["candidate_config"]["candidate_id"] != selected_ids[method_name]:
                    raise ValueError("selected checkpoint candidate mismatch")
                normalizer_hashes.add(
                    loaded_normalizer.to_manifest()["transform_sha256"]
                )
                normalizer = loaded_normalizer if normalizer is None else normalizer
                record = build_checkpoint_record(
                    method_name=method_name,
                    fold=fold,
                    checkpoint_path=checkpoint_path,
                    seed=config.seed,
                )
                registry.register(record, replace_existing=True)
                adapters[method_name] = TrainedFusionAdapter(
                    encoder=encoder,
                    normalizer=loaded_normalizer,
                    fold_id=fold_id,
                    checkpoint_sha256=record.checkpoint_sha256,
                )
            if len(normalizer_hashes) != 1 or normalizer is None:
                raise ValueError("selected checkpoints do not share one fold normalizer")
            naive_path = heavy_root / "checkpoints" / fold_id / "naive_time_sync" / "best.pt"
            if not naive_path.exists() or not config.resume:
                naive_encoder = NaiveTimeSyncEncoder().fit_from_batch_provider(
                    provider,
                    train_sample_ids=fold.train_sample_ids,
                    held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
                    normalizer=normalizer,
                    batch_size=config.fit_batch_size,
                )
                save_naive_time_sync_checkpoint(naive_path, encoder=naive_encoder)
                naive_status = "completed"
            else:
                naive_status = "resumed"
            naive_encoder = load_naive_time_sync_checkpoint(naive_path)
            naive_record = build_checkpoint_record(
                method_name="naive_time_sync",
                fold=fold,
                checkpoint_path=naive_path,
                seed=config.seed,
            )
            registry.register(naive_record, replace_existing=True)
            adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
                encoder=naive_encoder,
                fold_id=fold_id,
                checkpoint_sha256=naive_record.checkpoint_sha256,
            )
            exporter = ResumableOOFExporter(
                heavy_root / "representations" / fold_id,
                resume=config.resume,
            )
            outputs = {method: {} for method in SIX_METHODS}
            fold_exports = []
            for method_name in SIX_METHODS:
                for role in ("train", "validation", "held_out"):
                    result = exporter.export_from_batch_provider(
                        encoder=adapters[method_name],
                        batch_provider=provider,
                        checkpoint=registry.require(method_name, fold_id),
                        export_role=role,
                        batch_size=config.export_batch_size,
                    )
                    fold_exports.append(result)
                    outputs[method_name][role] = load_fusion_stream_batch(result.output_root)
                    export_rows.append(
                        {"fold_id": fold_id, **result.to_dict()}
                    )
            alignment = {
                role: validate_fusion_method_alignment(
                    [outputs[method][role] for method in SIX_METHODS]
                )
                for role in ("train", "validation", "held_out")
            }
            manifest = {
                "format": "chronaris.dingxin_selected_representation_exports.v1",
                "fold_id": fold_id,
                "export_count": len(fold_exports),
                "alignment_sha256": alignment,
                "exports": [result.to_dict() for result in fold_exports],
                "outer_test_representation_exported": True,
                "task_targets_opened": False,
                "outer_test_metrics_opened": False,
            }
            _write_json(fold_compact_root / "split_manifest.json", fold.to_dict())
            _write_json(
                fold_compact_root / "representation_export_manifest.json",
                manifest,
            )
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "checkpoint_count": len(registry.records),
                    "export_count": len(fold_exports),
                    "initial_completed_count": sum(
                        result.status == "completed" for result in fold_exports
                    ),
                    "initial_resumed_count": sum(
                        result.status == "resumed" for result in fold_exports
                    ),
                    "normalizer_transform_sha256": next(iter(normalizer_hashes)),
                    "naive_status": naive_status,
                    "alignment_role_count": len(alignment),
                    "outer_test_metrics_opened": False,
                }
            )
            progress.update("selected_fold_export_complete", fold_id=fold_id)
        acceptance = _acceptance_rows(fold_rows, export_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            fold_rows=fold_rows,
            export_rows=export_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            fold_count=len(fold_rows),
            export_count=len(export_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return DingxinSelectedRepresentationResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        fold_count=len(fold_rows),
        export_count=len(export_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _acceptance_rows(folds, exports):
    return (
        _check("five_folds", len(folds) == 5, len(folds), 5),
        _check("six_checkpoints_per_fold", all(row["checkpoint_count"] == 6 for row in folds), [row["checkpoint_count"] for row in folds], 6),
        _check("eighteen_exports_per_fold", all(row["export_count"] == 18 for row in folds), [row["export_count"] for row in folds], 18),
        _check("ninety_exports", len(exports) == 90, len(exports), 90),
        _check("three_aligned_roles", all(row["alignment_role_count"] == 3 for row in folds), [row["alignment_role_count"] for row in folds], 3),
        _check("outer_test_metrics_closed", all(not row["outer_test_metrics_opened"] for row in folds), False, False),
        _check("task_targets_closed", True, False, False),
    )


def _write_outputs(*, compact_root, heavy_root, config, fold_rows, export_rows, acceptance, status):
    paths = {
        "folds": compact_root / "fold_inventory.csv",
        "exports": compact_root / "representation_inventory.jsonl",
        "acceptance": compact_root / "acceptance.csv",
        "report": compact_root / "report.md",
        "resume": compact_root / "resume_command.txt",
        "evidence": compact_root / "evidence_manifest.json",
    }
    import pandas as pd

    pd.DataFrame(fold_rows).to_csv(paths["folds"], index=False)
    pd.DataFrame(acceptance).to_csv(paths["acceptance"], index=False)
    paths["exports"].write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in export_rows),
        encoding="utf-8",
    )
    passed = sum(row["passed"] for row in acceptance)
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新选定配置六方法表示导出",
                "",
                f"状态：{status}；验收 {passed}/{len(acceptance)}。",
                "五折各导出六方法 train/validation/outer-test 三角色表示，共 90 份。",
                "outer-test 仅生成冻结表示；任务目标、consumer 和指标保持关闭。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_selected_representations.py "
        f"--run-id {config.run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence"],
        {
            "run_id": config.run_id,
            "status": status,
            "evidence_layer": "dingxin_selected_representations",
            "fold_count": len(fold_rows),
            "export_count": len(export_rows),
            "acceptance_pass_count": passed,
            "acceptance_check_count": len(acceptance),
            "outer_test_representation_exported": True,
            "outer_test_metrics_opened": False,
            "task_targets_opened": False,
            "heavy_run_root": str(heavy_root),
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
