"""Validation-only consumers using inner-train-fitted Dingxin targets."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_models import (
    DingxinConsumerConfig,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_run import (
    _load_fold_outputs,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_runtime import (
    run_dingxin_method_consumers,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    load_dingxin_nested_fold_consumer_targets,
)
from chronaris.evaluation.application_tasks.dingxin_pretraining_aggregate import (
    DEFAULT_FOLD_RUN_IDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_nested_validation")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DingxinNestedConsumerConfig:
    run_id: str = "2026-07-11_dingxin-nested-validation"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    nested_target_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-nested-targets/nested_targets.csv"
    )
    fold_run_ids: tuple[str, ...] = DEFAULT_FOLD_RUN_IDS
    seed: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinNestedConsumerResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_fold_count: int
    metric_count: int
    fusion_gain_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_nested_validation_consumers(
    config: DingxinNestedConsumerConfig,
) -> DingxinNestedConsumerResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    consumer_config = DingxinConsumerConfig(random_state=config.seed)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_nested_validation_consumers",
        logger=LOGGER,
        initial_progress={
            "threshold_scope": "inner_train_nested",
            "evaluation_roles": ["validation"],
            "outer_test_metrics_opened": False,
            "confirmed_metrics_changed": False,
        },
    ) as progress:
        result_rows = []
        metric_rows = []
        resource_rows = []
        target_manifests = []
        prediction_match_count = 0
        for fold_run_id in config.fold_run_ids:
            fold_root = Path(config.compact_output_root) / fold_run_id
            split = json.loads(
                (fold_root / "split_manifest.json").read_text(encoding="utf-8")
            )
            fold_id = str(split["fold_id"])
            targets = load_dingxin_nested_fold_consumer_targets(
                fold_id=fold_id,
                nested_target_path=config.nested_target_path,
            )
            target_manifests.append(targets.to_manifest())
            all_outputs, _rows, _alignment = _load_fold_outputs(fold_root)
            for method_name in APPLICATION_METHODS:
                outputs = {
                    role: all_outputs[method_name][role]
                    for role in ("train", "validation")
                }
                initial = run_dingxin_method_consumers(
                    method_name=method_name,
                    fold_id=fold_id,
                    outputs=outputs,
                    targets=targets,
                    output_root=heavy_root / "consumers",
                    config=consumer_config,
                    evaluation_roles=("validation",),
                    resume=config.resume,
                )
                resumed = run_dingxin_method_consumers(
                    method_name=method_name,
                    fold_id=fold_id,
                    outputs=outputs,
                    targets=targets,
                    output_root=heavy_root / "consumers",
                    config=consumer_config,
                    evaluation_roles=("validation",),
                    resume=True,
                )
                if initial.metric_rows != resumed.metric_rows:
                    raise ValueError("nested validation metrics changed on resume")
                prediction_match = (
                    initial.manifest["prediction_sha256"]
                    == resumed.manifest["prediction_sha256"]
                )
                prediction_match_count += int(prediction_match)
                metric_rows.extend(dict(row) for row in initial.metric_rows)
                resource_rows.extend(dict(row) for row in initial.resource_rows)
                result_rows.append(
                    {
                        "fold_run_id": fold_run_id,
                        "fold_id": fold_id,
                        "method_name": method_name,
                        "initial_status": initial.status,
                        "resume_component_count": sum(
                            value == "resumed"
                            for value in resumed.component_status.values()
                        ),
                        "prediction_hash_match": prediction_match,
                        "protocol_sha256": initial.protocol_sha256,
                        "consumer_config_sha256": _mapping_hash(
                            asdict(consumer_config)
                        ),
                        "threshold_scope": targets.threshold_scope,
                        "evaluation_roles": "validation",
                        "outer_test_metrics_opened": False,
                    }
                )
                progress.update(
                    "nested_method_fold_complete",
                    fold_id=fold_id,
                    method_name=method_name,
                )
        fusion_gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=(
                "naive_time_sync",
                "mult",
                "contiformer",
                "chronaris",
            ),
        )
        acceptance_rows = _acceptance_rows(
            result_rows,
            metric_rows,
            fusion_gain_rows,
            target_manifests,
            prediction_match_count,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            config=config,
            status=status,
            consumer_config=consumer_config,
            result_rows=result_rows,
            resource_rows=resource_rows,
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            target_manifests=target_manifests,
            acceptance_rows=acceptance_rows,
            heavy_root=str(heavy_root),
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            metric_count=len(metric_rows),
            fusion_gain_count=len(fusion_gain_rows),
        )
        return DingxinNestedConsumerResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            method_fold_count=len(result_rows),
            metric_count=len(metric_rows),
            fusion_gain_count=len(fusion_gain_rows),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _acceptance_rows(results, metrics, gains, targets, prediction_match_count):
    unavailable = [row for row in metrics if row["status"] == "unavailable"]
    return [
        _check("thirty_method_fold_bundles", len(results) == 30, len(results), 30),
        _check("five_nested_target_folds", len(targets) == 5, len(targets), 5),
        _check("all_targets_inner_train_nested", all(item["threshold_scope"] == "inner_train_nested" for item in targets), False, False),
        _check("resume_reuses_sixty_components", sum(row["resume_component_count"] for row in results) == 60, sum(row["resume_component_count"] for row in results), 60),
        _check("prediction_hashes_stable", prediction_match_count == 30, prediction_match_count, 30),
        _check("fixed_consumer_config", len({row["consumer_config_sha256"] for row in results}) == 1, 1, 1),
        _check("validation_only_metric_matrix", len(metrics) == 840 and {row["role"] for row in metrics} == {"validation"}, len(metrics), 840),
        _check("outer_test_metrics_closed", all(not row["outer_test_metrics_opened"] for row in results) and "held_out" not in {row["role"] for row in metrics}, False, False),
        _check("metric_status_structured", all(row["status"] == "available" or row["reason"] == "metric_not_defined" for row in metrics), len(unavailable), "structured"),
        _check("fixed_class_macro_f1_present", sum(row["metric"] == "macro_f1" and row["task"] == "maneuver_intensity_classification" for row in metrics) == 60, True, True),
        _check("expected_fusion_gain_matrix", len(gains) == 560, len(gains), 560),
        _check("smoke_only_not_confirmed", all(row["smoke_only"] and row["threshold_scope"] == "inner_train_nested" for row in metrics), False, False),
    ]


def _write_outputs(
    *, compact_root, config, status, consumer_config, result_rows,
    resource_rows, metric_rows, fusion_gain_rows, target_manifests,
    acceptance_rows, heavy_root
):
    paths = {
        "consumer_inventory": compact_root / "consumer_inventory.csv",
        "resource_budget": compact_root / "resource_budget.csv",
        "metric_long": compact_root / "metric_long.csv",
        "fusion_gain": compact_root / "fusion_gain.csv",
        "target_manifest": compact_root / "target_manifest.json",
        "consumer_protocol": compact_root / "consumer_protocol.json",
        "acceptance_checks": compact_root / "acceptance_checks.csv",
        "report": compact_root / "report.md",
        "claim_boundary": compact_root / "claim_boundary.md",
        "resume_command": compact_root / "resume_command.txt",
        "evidence_manifest": compact_root / "evidence_manifest.json",
    }
    for key, rows in (
        ("consumer_inventory", result_rows), ("resource_budget", resource_rows),
        ("metric_long", metric_rows), ("fusion_gain", fusion_gain_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(paths["target_manifest"], {"format": "chronaris.dingxin_nested_consumer_targets.v1", "folds": target_manifests})
    _write_json(paths["consumer_protocol"], {"format": "chronaris.dingxin_nested_validation_protocol.v1", "config": asdict(consumer_config), "fit_role": "train", "evaluation_roles": ["validation"], "outer_test_metrics_opened": False, "method_specific_hyperparameters": False})
    passed = sum(row["passed"] for row in acceptance_rows)
    available = sum(row["status"] == "available" for row in metric_rows)
    paths["report"].write_text("\n".join((
        "# 鼎新嵌套目标 validation consumer 报告", "", "## 结论", "",
        f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
        "- 五折六方法使用 inner-train 嵌套目标拟合固定线性与 MiniROCKET consumer，只评价 validation。",
        f"- 生成 {len(metric_rows)} 条 validation-only 指标，其中 {available} 条可计算；双流增益接口 {len(fusion_gain_rows)} 条。",
        "- outer-test 没有进入评价角色，不生成预测或指标；本 run 仍为正式 screen 前的协议确认。",
        "- 本报告不形成方法排名或论文确认结论。", "", "## 下一步", "",
        "1. 冻结本协议，进入 seed 17 候选 screen。", "2. screen 期间继续禁止读取 outer-test 与仿真锁定测试。", ""
    )), encoding="utf-8")
    paths["claim_boundary"].write_text("# 论断边界\n\n- 只评价 validation，outer-test 保持关闭。\n- 嵌套目标仍是弱监督任务。\n- 当前指标不进入论文确认表。\n", encoding="utf-8")
    paths["resume_command"].write_text(f"/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/evaluation/application_tasks/run_dingxin_nested_validation.py --run-id {config.run_id} --resume\n", encoding="utf-8")
    _write_json(paths["evidence_manifest"], {"run_id": config.run_id, "status": status, "evidence_layer": "dingxin_nested_validation_smoke", "metric_count": len(metric_rows), "fusion_gain_count": len(fusion_gain_rows), "outer_test_metrics_opened": False, "confirmed_metrics_changed": False, "heavy_run_root": heavy_root, "output_paths": {key: str(value) for key, value in paths.items()}})
    return {key: str(value) for key, value in paths.items()}


def _mapping_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
