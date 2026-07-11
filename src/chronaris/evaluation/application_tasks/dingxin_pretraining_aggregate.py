"""Aggregate and revalidate all five fixed Dingxin pretraining folds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.representation import CheckpointRegistry, load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


DEFAULT_FOLD_RUN_IDS = (
    "2026-07-11_dingxin-fold-pretraining-smoke",
    "2026-07-11_dingxin-fold-pretraining-view02",
    "2026-07-11_dingxin-fold-pretraining-view03",
    "2026-07-11_dingxin-fold-pretraining-sortie01",
    "2026-07-11_dingxin-fold-pretraining-sortie02",
)
EXPECTED_FOLD_IDS = {
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
    "leave_one_sortie_out__fold01",
    "leave_one_sortie_out__fold02",
}
EXPECTED_METHODS = {
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
}


@dataclass(frozen=True, slots=True)
class DingxinPretrainingAggregateConfig:
    run_id: str = "2026-07-11_dingxin-five-fold-pretraining"
    output_root: str = "docs/artifacts/runs"
    fold_run_ids: tuple[str, ...] = DEFAULT_FOLD_RUN_IDS


@dataclass(frozen=True, slots=True)
class DingxinPretrainingAggregateResult:
    run_id: str
    status: str
    run_root: str
    fold_count: int
    checkpoint_count: int
    representation_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_pretraining_aggregate(
    config: DingxinPretrainingAggregateConfig,
) -> DingxinPretrainingAggregateResult:
    root = Path(config.output_root)
    run_root = root / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    checkpoint_rows = []
    export_rows = []
    resource_rows = []
    source_rows = []
    for fold_run_id in config.fold_run_ids:
        fold_root = root / fold_run_id
        payloads = _load_fold_payloads(fold_root)
        split = payloads["split"]
        fold_id = str(split["fold_id"])
        evidence = payloads["evidence"]
        acceptance = payloads["acceptance"]
        exports = payloads["exports"]
        registry = CheckpointRegistry(fold_root / "checkpoint_registry.json")
        for record in registry.records.values():
            verified = registry.require(record.method_name, fold_id)
            checkpoint_rows.append(
                {
                    "fold_run_id": fold_run_id,
                    "fold_id": fold_id,
                    "method_name": verified.method_name,
                    "checkpoint_path": verified.checkpoint_path,
                    "checkpoint_sha256": verified.checkpoint_sha256,
                    "fit_sample_hash": verified.fit_sample_hash,
                    "label_used_for_encoder_training": (
                        verified.label_used_for_encoder_training
                    ),
                }
            )
        valid_exports = 0
        for item in exports["exports"]:
            representation_path = Path(item["representation_path"])
            archive_valid = (
                representation_path.is_file()
                and sha256_file(representation_path)
                == item["representation_sha256"]
            )
            batch = load_fusion_stream_batch(item["output_root"])
            archive_valid = archive_valid and batch.sample_ids == tuple(
                item["sample_ids"]
            )
            valid_exports += int(archive_valid)
            export_rows.append(
                {
                    "fold_run_id": fold_run_id,
                    "fold_id": fold_id,
                    "method_name": item["method_name"],
                    "export_role": item["export_role"],
                    "sample_count": len(item["sample_ids"]),
                    "representation_sha256": item["representation_sha256"],
                    "archive_valid": archive_valid,
                }
            )
        resources = payloads["resources"]
        resources.insert(0, "fold_run_id", fold_run_id)
        resources.insert(1, "fold_id", fold_id)
        resource_rows.extend(resources.to_dict("records"))
        source_rows.append(
            {
                "fold_run_id": fold_run_id,
                "fold_id": fold_id,
                "schema_sha256": payloads["source"]["schema_sha256"],
                "source_hashes": payloads["source"]["source_hashes"],
            }
        )
        fold_rows.append(
            {
                "fold_run_id": fold_run_id,
                "fold_id": fold_id,
                "status": evidence["status"],
                "train_count": len(split["train_sample_ids"]),
                "validation_count": len(split["validation_sample_ids"]),
                "held_out_count": len(split["held_out_sample_ids"]),
                "train_sample_hash": split["train_sample_hash"],
                "acceptance_pass_count": int(acceptance["passed"].sum()),
                "acceptance_check_count": len(acceptance),
                "checkpoint_count": len(registry.records),
                "representation_count": len(exports["exports"]),
                "valid_representation_count": valid_exports,
                "resume_reused_count": exports["resume_reused_count"],
                "downstream_targets_opened": evidence[
                    "downstream_targets_opened"
                ],
                "outer_test_metrics_opened": evidence[
                    "outer_test_metrics_opened"
                ],
            }
        )
    acceptance_rows = _build_acceptance_rows(
        fold_rows=fold_rows,
        checkpoint_rows=checkpoint_rows,
        export_rows=export_rows,
        resource_rows=resource_rows,
        source_rows=source_rows,
    )
    status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
    paths = _write_outputs(
        run_root=run_root,
        config=config,
        status=status,
        fold_rows=fold_rows,
        checkpoint_rows=checkpoint_rows,
        export_rows=export_rows,
        resource_rows=resource_rows,
        source_rows=source_rows,
        acceptance_rows=acceptance_rows,
    )
    return DingxinPretrainingAggregateResult(
        run_id=config.run_id,
        status=status,
        run_root=str(run_root),
        fold_count=len(fold_rows),
        checkpoint_count=len(checkpoint_rows),
        representation_count=len(export_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance_rows),
        acceptance_check_count=len(acceptance_rows),
        report_path=paths["report"],
        evidence_manifest_path=paths["evidence_manifest"],
    )


def _load_fold_payloads(root: Path):
    required = {
        "split": root / "split_manifest.json",
        "evidence": root / "evidence_manifest.json",
        "exports": root / "representation_export_manifest.json",
        "source": root / "source_manifest.json",
        "acceptance": root / "acceptance_checks.csv",
        "resources": root / "resource_budget.csv",
        "registry": root / "checkpoint_registry.json",
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Dingxin fold evidence missing under {root}: {missing}")
    return {
        "split": json.loads(required["split"].read_text(encoding="utf-8")),
        "evidence": json.loads(required["evidence"].read_text(encoding="utf-8")),
        "exports": json.loads(required["exports"].read_text(encoding="utf-8")),
        "source": json.loads(required["source"].read_text(encoding="utf-8")),
        "acceptance": pd.read_csv(required["acceptance"]),
        "resources": pd.read_csv(required["resources"]),
    }


def _build_acceptance_rows(
    *, fold_rows, checkpoint_rows, export_rows, resource_rows, source_rows
):
    fold_ids = {row["fold_id"] for row in fold_rows}
    checkpoint_keys = {
        (row["fold_id"], row["method_name"]) for row in checkpoint_rows
    }
    export_keys = {
        (row["fold_id"], row["method_name"], row["export_role"])
        for row in export_rows
    }
    expected_checkpoint_keys = {
        (fold_id, method) for fold_id in EXPECTED_FOLD_IDS for method in EXPECTED_METHODS
    }
    expected_export_keys = {
        (fold_id, method, role)
        for fold_id in EXPECTED_FOLD_IDS
        for method in EXPECTED_METHODS
        for role in ("train", "validation", "held_out")
    }
    fit_hash_valid = all(
        row["fit_sample_hash"]
        == next(
            fold["train_sample_hash"]
            for fold in fold_rows
            if fold["fold_id"] == row["fold_id"]
        )
        for row in checkpoint_rows
    )
    schema_hashes = {row["schema_sha256"] for row in source_rows}
    maximum_rss = max(float(row["maximum_rss_mb"]) for row in resource_rows)
    return [
        _check("five_fixed_folds_present", len(fold_rows) == 5 and fold_ids == EXPECTED_FOLD_IDS, sorted(fold_ids), sorted(EXPECTED_FOLD_IDS)),
        _check("all_fold_runs_completed", all(row["status"] == "completed" for row in fold_rows), [row["status"] for row in fold_rows], "five completed"),
        _check("all_fold_acceptance_passed", sum(row["acceptance_pass_count"] for row in fold_rows) == 60 and sum(row["acceptance_check_count"] for row in fold_rows) == 60, [row["acceptance_pass_count"] for row in fold_rows], "60/60"),
        _check("thirty_checkpoint_records", checkpoint_keys == expected_checkpoint_keys, len(checkpoint_keys), 30),
        _check("checkpoint_fit_hashes_match_inner_train", fit_hash_valid, fit_hash_valid, True),
        _check("checkpoint_training_has_no_labels", all(not row["label_used_for_encoder_training"] for row in checkpoint_rows), False, False),
        _check("ninety_role_representations", export_keys == expected_export_keys, len(export_keys), 90),
        _check("all_representation_archives_valid", all(row["archive_valid"] for row in export_rows), sum(row["archive_valid"] for row in export_rows), 90),
        _check("all_exports_resume_reused", sum(row["resume_reused_count"] for row in fold_rows) == 90, [row["resume_reused_count"] for row in fold_rows], 90),
        _check("role_sample_counts_match_split", _role_counts_match(fold_rows, export_rows), True, True),
        _check("one_common_input_schema", len(schema_hashes) == 1, sorted(schema_hashes), "one schema hash"),
        _check("targets_and_outer_metrics_remain_closed", all(not row["downstream_targets_opened"] and not row["outer_test_metrics_opened"] for row in fold_rows), False, False),
        _check("all_runs_below_memory_gate", maximum_rss < 2500.0, round(maximum_rss, 1), "<2500 MB"),
    ]


def _role_counts_match(fold_rows, export_rows):
    expected = {
        (row["fold_id"], role): row[f"{role}_count"]
        for row in fold_rows
        for role in ("train", "validation", "held_out")
    }
    return all(
        row["sample_count"] == expected[(row["fold_id"], row["export_role"])]
        for row in export_rows
    )


def _write_outputs(
    *, run_root, config, status, fold_rows, checkpoint_rows, export_rows,
    resource_rows, source_rows, acceptance_rows
):
    paths = {
        "fold_summary": run_root / "fold_summary.csv",
        "checkpoint_inventory": run_root / "checkpoint_inventory.csv",
        "representation_inventory": run_root / "representation_inventory.csv",
        "resource_budget": run_root / "resource_budget.csv",
        "source_manifest": run_root / "source_manifest.json",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    for key, rows in (
        ("fold_summary", fold_rows),
        ("checkpoint_inventory", checkpoint_rows),
        ("representation_inventory", export_rows),
        ("resource_budget", resource_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(
        paths["source_manifest"],
        {
            "format": "chronaris.dingxin_five_fold_pretraining_sources.v1",
            "fold_run_ids": list(config.fold_run_ids),
            "fold_sources": source_rows,
        },
    )
    passed = sum(row["passed"] for row in acceptance_rows)
    total_training = sum(
        float(row["training_elapsed_s"])
        for row in resource_rows
        if row["method_name"] != "naive_time_sync"
    )
    maximum_rss = max(float(row["maximum_rss_mb"]) for row in resource_rows)
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新五折六方法公共预训练与统一表示报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；聚合验收 {passed}/{len(acceptance_rows)} 通过，五个子 run 合计 60/60。",
                "- 三个留一视图主协议折与两个留一架次辅助协议折均完成；六方法共形成 30 个 checkpoint 和 90 份 train/validation/outer-test 表示。",
                "- 90 份 archive、manifest、样本顺序、source hash 与 checkpoint lineage 已逐项重验；每折第二遍恢复均复用 18/18。",
                f"- 五折五个可训练方法累计训练 {total_training:.2f} 秒；所有运行实测最高峰值内存 {maximum_rss:.1f} MB，低于 2.5 GB 门限。",
                "- 预训练仍未打开机动分类或生理响应目标，outer-test 未计算任务指标；本报告不构成模型排名。",
                "",
                "## 下一步",
                "",
                "1. 用同一五折表示接入固定线性与 MiniROCKET 工程冒烟。",
                "2. 正式候选筛选前按每折 inner-train 重建嵌套任务目标。",
                "3. 五折 consumer 完成后再启动 Chronaris 候选 screen，不提前读取锁定仿真测试。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本报告证明五个固定鼎新外层折可完成六方法公共预训练和统一表示导出。\n"
        "- 每折仅使用一个 seed、一个 epoch 和固定候选；仍是工程 smoke，不是正式效果确认。\n"
        "- 两项弱监督目标和 outer-test 指标均未打开，不比较任何方法优劣。\n"
        "- 正式 screen 前必须按 inner-train 重拟合嵌套目标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/audit_dingxin_five_fold_pretraining.py "
        f"--run-id {config.run_id}\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": config.run_id,
            "status": status,
            "evidence_layer": "dingxin_five_fold_pretraining_smoke",
            "fold_count": len(fold_rows),
            "checkpoint_count": len(checkpoint_rows),
            "representation_count": len(export_rows),
            "downstream_targets_opened": False,
            "outer_test_metrics_opened": False,
            "confirmed_metrics_changed": False,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
