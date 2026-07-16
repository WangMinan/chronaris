"""Run fixed lightweight consumers on simplified Dingxin representations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

from .simple_downstream_consumers import (
    SimpleConsumerConfig,
    fit_simple_downstream_consumer,
    maneuver_metric_summary,
    physiology_metric_summary,
)
from .simple_downstream_protocol import SIMPLE_DOWNSTREAM_METHODS


@dataclass(frozen=True, slots=True)
class SimpleDownstreamRunConfig:
    run_id: str = "2026-07-16_simple-downstream-consumer-smoke"
    representation_run_id: str = (
        "2026-07-16_simple-downstream-representations-smoke"
    )
    task_protocol_run_id: str = "2026-07-16_simple-downstream-protocol"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    fold_ids: tuple[str, ...] = ("leave_one_sortie_out__fold01",)
    seeds: tuple[int, ...] = (17,)
    methods: tuple[str, ...] = SIMPLE_DOWNSTREAM_METHODS
    evaluation_scope: str = "engineering_smoke"
    classification_c: float = 1.0
    regression_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not self.methods or not set(self.methods).issubset(SIMPLE_DOWNSTREAM_METHODS):
            raise ValueError("simplified consumer methods are unsupported")
        if self.evaluation_scope not in {"engineering_smoke", "formal_confirmation"}:
            raise ValueError("simplified consumer evaluation scope is unsupported")
        if self.classification_c != 1.0 or self.regression_alpha != 1.0:
            raise ValueError("simplified consumer parameters are frozen at 1.0")


@dataclass(frozen=True, slots=True)
class SimpleDownstreamRunResult:
    run_id: str
    status: str
    compact_root: str
    heavy_root: str
    evaluated_unit_count: int
    metrics_long_path: str
    metrics_summary_path: str
    report_path: str
    evidence_manifest_path: str


def run_simple_downstream_consumers(
    config: SimpleDownstreamRunConfig,
) -> SimpleDownstreamRunResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    representation_root = (
        Path(config.compact_output_root) / config.representation_run_id
    )
    representation_heavy_root = (
        Path(config.heavy_output_root) / config.representation_run_id
    )
    task_root = Path(config.heavy_output_root) / config.task_protocol_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    representation_protocol_path = representation_root / "protocol.json"
    representation_protocol = json.loads(
        representation_protocol_path.read_text(encoding="utf-8")
    )
    if representation_protocol.get("status") != "completed":
        raise ValueError("simplified consumer representations are incomplete")
    if representation_protocol.get("task_targets_opened") is not False:
        raise ValueError("simplified consumer representation encoder used task labels")
    inventory = pd.read_json(
        representation_root / "representation_inventory.jsonl", lines=True
    )
    maneuver_targets = pd.read_csv(task_root / "maneuver_targets.csv")
    physiology_targets = pd.read_csv(task_root / "physiology_targets.csv")
    _validate_requested_units(config, inventory)

    metrics_rows = []
    unit_rows = []
    prediction_inventory = []
    for seed in config.seeds:
        for fold_id in config.fold_ids:
            fold_maneuver = maneuver_targets[
                maneuver_targets["fold_id"].astype(str) == fold_id
            ].copy()
            fold_physiology = physiology_targets[
                physiology_targets["fold_id"].astype(str) == fold_id
            ].copy()
            for method in config.methods:
                rows = inventory[
                    (inventory["seed"].astype(int) == seed)
                    & (inventory["fold_id"].astype(str) == fold_id)
                    & (inventory["method_name"].astype(str) == method)
                ]
                train = _load_inventory_role(rows, "consumer_train")
                held_out = _load_inventory_role(rows, "held_out")
                consumer = fit_simple_downstream_consumer(
                    pooled_embedding=train.pooled_embedding,
                    sample_ids=train.sample_ids,
                    maneuver_targets=fold_maneuver,
                    physiology_targets=fold_physiology,
                    config=SimpleConsumerConfig(
                        classification_c=config.classification_c,
                        regression_alpha=config.regression_alpha,
                        random_state=seed,
                    ),
                )
                predictions = consumer.predict(held_out.pooled_embedding)
                held_out_maneuver = fold_maneuver[
                    fold_maneuver["split_role"].astype(str) == "held_out"
                ]
                held_out_physiology = fold_physiology[
                    fold_physiology["split_role"].astype(str) == "held_out"
                ]
                maneuver_summary, maneuver_predictions = maneuver_metric_summary(
                    held_out_maneuver,
                    sample_ids=held_out.sample_ids,
                    score_prediction=predictions["maneuver_score"],
                    class_probability=predictions["maneuver_probability"],
                )
                physiology_summary, physiology_field_metrics = (
                    physiology_metric_summary(
                        held_out_physiology,
                        sample_ids=held_out.sample_ids,
                        standardized_prediction=predictions[
                            "physiology_standardized"
                        ],
                        fields=consumer.physiology_fields,
                    )
                )
                unit_root = heavy_root / f"seed_{seed}" / fold_id / method
                unit_root.mkdir(parents=True, exist_ok=True)
                maneuver_path = unit_root / "maneuver_predictions.csv"
                physiology_path = unit_root / "physiology_predictions.csv"
                physiology_fields_path = unit_root / "physiology_field_metrics.csv"
                _serializable_maneuver_predictions(maneuver_predictions).to_csv(
                    maneuver_path, index=False
                )
                _physiology_prediction_frame(
                    held_out_physiology,
                    sample_ids=held_out.sample_ids,
                    prediction=predictions["physiology_standardized"],
                    fields=consumer.physiology_fields,
                ).to_csv(physiology_path, index=False)
                physiology_field_metrics.to_csv(
                    physiology_fields_path, index=False
                )
                prediction_inventory.append(
                    {
                        "seed": seed,
                        "fold_id": fold_id,
                        "method_name": method,
                        "maneuver_predictions_path": str(maneuver_path),
                        "maneuver_predictions_sha256": sha256_file(maneuver_path),
                        "physiology_predictions_path": str(physiology_path),
                        "physiology_predictions_sha256": sha256_file(physiology_path),
                        "physiology_field_metrics_path": str(physiology_fields_path),
                        "physiology_field_metrics_sha256": sha256_file(
                            physiology_fields_path
                        ),
                    }
                )
                metrics_rows.extend(
                    _metric_rows(
                        seed,
                        fold_id,
                        method,
                        "future_maneuver",
                        maneuver_summary,
                    )
                )
                metrics_rows.extend(
                    _metric_rows(
                        seed,
                        fold_id,
                        method,
                        "future_physiology",
                        physiology_summary,
                    )
                )
                unit_rows.append(
                    {
                        "seed": seed,
                        "fold_id": fold_id,
                        "method_name": method,
                        "consumer_train_count": len(train.sample_ids),
                        "held_out_count": len(held_out.sample_ids),
                        "maneuver_vehicle_context_count": maneuver_summary[
                            "independent_vehicle_context_count"
                        ],
                        "physiology_field_count": physiology_summary["field_count"],
                        "status": "completed",
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = _summarize_metrics(metrics)
    paths = {
        "metrics_long": compact_root / "metrics_long.csv",
        "metrics_summary": compact_root / "metrics_summary.csv",
        "units": compact_root / "unit_inventory.csv",
        "predictions": compact_root / "prediction_inventory.csv",
        "acceptance": compact_root / "acceptance.csv",
        "protocol": compact_root / "protocol.json",
        "report": compact_root / "report.md",
        "resume": compact_root / "resume_command.txt",
        "evidence": compact_root / "evidence_manifest.json",
    }
    metrics.to_csv(paths["metrics_long"], index=False)
    metrics_summary.to_csv(paths["metrics_summary"], index=False)
    pd.DataFrame(unit_rows).to_csv(paths["units"], index=False)
    pd.DataFrame(prediction_inventory).to_csv(paths["predictions"], index=False)
    acceptance = _acceptance_rows(config, unit_rows, metrics)
    pd.DataFrame(acceptance).to_csv(paths["acceptance"], index=False)
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"
    _write_json(
        paths["protocol"],
        {
            "format": "chronaris.simple_downstream_consumer_protocol.v1",
            "status": status,
            "config": asdict(config),
            "representation_protocol_sha256": sha256_file(
                representation_protocol_path
            ),
            "task_protocol_run_id": config.task_protocol_run_id,
            "outer_metrics_opened": True,
            "result_driven_tuning_allowed": False,
        },
    )
    _write_report(paths["report"], config, status, acceptance, unit_rows)
    paths["resume"].write_text(_resume_command(config) + "\n", encoding="utf-8")
    _write_json(
        paths["evidence"],
        {
            "format": "chronaris.simple_downstream_consumer_evidence.v1",
            "run_id": config.run_id,
            "status": status,
            "evaluation_scope": config.evaluation_scope,
            "evaluated_unit_count": len(unit_rows),
            "acceptance_pass_count": sum(row["passed"] for row in acceptance),
            "acceptance_check_count": len(acceptance),
            "outer_metrics_opened": True,
            "result_driven_tuning_allowed": False,
            "confirmed_metrics_changed": False,
            "representation_heavy_root": str(representation_heavy_root),
            "heavy_run_root": str(heavy_root),
            "output_paths": {key: str(path) for key, path in paths.items()},
        },
    )
    return SimpleDownstreamRunResult(
        run_id=config.run_id,
        status=status,
        compact_root=str(compact_root),
        heavy_root=str(heavy_root),
        evaluated_unit_count=len(unit_rows),
        metrics_long_path=str(paths["metrics_long"]),
        metrics_summary_path=str(paths["metrics_summary"]),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _load_inventory_role(rows, role):
    selected = rows[rows["export_role"].astype(str) == role]
    if len(selected) != 1:
        raise ValueError(f"representation inventory has no unique {role} row")
    return load_fusion_stream_batch(str(selected.iloc[0]["output_root"]))


def _validate_requested_units(config, inventory):
    for seed in config.seeds:
        for fold_id in config.fold_ids:
            for method in config.methods:
                unit = inventory[
                    (inventory["seed"].astype(int) == seed)
                    & (inventory["fold_id"].astype(str) == fold_id)
                    & (inventory["method_name"].astype(str) == method)
                ]
                if set(unit["export_role"].astype(str)) != {
                    "consumer_train",
                    "held_out",
                }:
                    raise ValueError(
                        f"incomplete simplified representation unit: {seed}/{fold_id}/{method}"
                    )


def _metric_rows(seed, fold_id, method, task_name, summary):
    return [
        {
            "seed": seed,
            "fold_id": fold_id,
            "method_name": method,
            "task_name": task_name,
            "metric_name": name,
            "metric_value": value,
        }
        for name, value in summary.items()
        if value is not None
    ]


def _summarize_metrics(metrics):
    numeric = metrics[pd.to_numeric(metrics["metric_value"], errors="coerce").notna()].copy()
    numeric["metric_value"] = numeric["metric_value"].astype(float)
    return (
        numeric.groupby(["method_name", "task_name", "metric_name"], as_index=False)
        .agg(
            mean=("metric_value", "mean"),
            std=("metric_value", "std"),
            minimum=("metric_value", "min"),
            maximum=("metric_value", "max"),
            unit_count=("metric_value", "count"),
        )
        .sort_values(["task_name", "metric_name", "method_name"], kind="mergesort")
    )


def _serializable_maneuver_predictions(frame):
    output = frame.drop(columns=["class_probability"]).copy()
    probability = np.stack(frame["class_probability"].to_list())
    output["probability_low"] = probability[:, 0]
    output["probability_medium"] = probability[:, 1]
    output["probability_high"] = probability[:, 2]
    return output


def _physiology_prediction_frame(targets, *, sample_ids, prediction, fields):
    ids = tuple(str(value) for value in sample_ids)
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = targets[targets["selected"].astype(bool)].pivot(
        index="context_id", columns="field_name", values="future_standardized"
    ).reindex(index=ids, columns=fields)
    current = targets[targets["selected"].astype(bool)].pivot(
        index="context_id", columns="field_name", values="current_standardized"
    ).reindex(index=ids, columns=fields)
    rows = []
    for sample_index, context_id in enumerate(ids):
        for field_index, field_name in enumerate(fields):
            rows.append(
                {
                    "context_id": context_id,
                    "field_name": field_name,
                    "future_standardized": truth.iloc[sample_index, field_index],
                    "current_standardized": current.iloc[sample_index, field_index],
                    "prediction_standardized": prediction[sample_index, field_index],
                }
            )
    return pd.DataFrame(rows)


def _acceptance_rows(config, units, metrics):
    expected = len(config.seeds) * len(config.fold_ids) * len(config.methods)
    return (
        _check("all_requested_units", len(units) == expected, len(units), expected),
        _check(
            "all_units_completed",
            all(row["status"] == "completed" for row in units),
            [row["status"] for row in units],
            "completed",
        ),
        _check(
            "fixed_consumer_parameters",
            config.classification_c == 1.0 and config.regression_alpha == 1.0,
            [config.classification_c, config.regression_alpha],
            [1.0, 1.0],
        ),
        _check(
            "metrics_finite",
            np.isfinite(pd.to_numeric(metrics["metric_value"], errors="coerce")).all(),
            len(metrics),
            "all finite",
        ),
    )


def _write_report(path, config, status, acceptance, units):
    passed = sum(row["passed"] for row in acceptance)
    scope_line = (
        "本轮仅为工程冒烟，分数不得用于修改协议、模型或消费者，也不进入论文正式主表。"
        if config.evaluation_scope == "engineering_smoke"
        else "本轮为协议冻结后的正式确认，结果打开后不再进行结果驱动调参。"
    )
    Path(path).write_text(
        "\n".join(
            (
                "# 鼎新简化下游消费者运行",
                "",
                f"状态：{status}；验收 {passed}/{len(acceptance)}。",
                f"完成 {len(units)} 个方法—折—随机种子评价单元。",
                scope_line,
                "未来机动按独立飞机上下文聚合，未来生理状态按字段报告并计算相对持久性技能。",
                "",
            )
        ),
        encoding="utf-8",
    )


def _resume_command(config):
    seed_flags = " ".join(f"--seed {seed}" for seed in config.seeds)
    fold_flags = " ".join(f"--fold-id {fold}" for fold in config.fold_ids)
    method_flags = " ".join(f"--method {method}" for method in config.methods)
    return (
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/dingxin/run_simple_downstream.py "
        f"--run-id {config.run_id} "
        f"--representation-run-id {config.representation_run_id} "
        f"--task-protocol-run-id {config.task_protocol_run_id} "
        f"--evaluation-scope {config.evaluation_scope} "
        f"{seed_flags} {fold_flags} {method_flags}"
    )


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
