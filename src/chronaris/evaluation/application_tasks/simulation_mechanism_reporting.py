"""Compact reporting for the locked simulation mechanism recovery task."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def write_simulation_mechanism_outputs(**values):
    root = values["compact_root"]
    paths = {
        "models": root / "model_inventory.csv",
        "evaluations": root / "evaluation_inventory.csv",
        "metrics": root / "metric_long.csv",
        "paired": root / "paired_statistics.csv",
        "g1_targets": root / "g1_target_manifest.json",
        "stress_targets": root / "stress_target_manifest.json",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "downstream_protocol.json",
        "report": root / "report.md",
        "claim": root / "claim_boundary.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    for key, rows in (
        ("models", values["model_rows"]),
        ("evaluations", values["evaluation_rows"]),
        ("metrics", values["metric_rows"]),
        ("paired", values["paired_rows"]),
        ("acceptance", values["acceptance"]),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(paths["g1_targets"], values["g1_target_manifest"])
    _write_json(paths["stress_targets"], values["stress_target_manifest"])
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_mechanism_downstream_protocol.v1",
        "config": asdict(values["config"]),
        "fit_role": "g1_train",
        "hyperparameter_selection_role": "g1_validation",
        "locked_evaluation_role": "g2_stress",
        "independent_statistical_unit": "trajectory",
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# 时间偏移与响应时延恢复锁定评估",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['model_rows'])} 个 G1 恢复探针和 {len(values['evaluation_rows'])} 个 G2 场景评价。",
        "",
    )), encoding="utf-8")
    paths["claim"].write_text(
        "# 结论边界\n\n恢复指标验证半物理仿真中表示对已知观测偏移和生理响应时延的可辨识性，不等同于鼎新现场人工标注。\n",
        encoding="utf-8",
    )
    prediction_path = values["heavy_root"] / "prediction_rows.csv"
    pd.DataFrame(values["prediction_rows"]).to_csv(prediction_path, index=False)
    paths["resume"].write_text(
        _resume_command(values["config"]),
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_mechanism_consumer_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "model_count": len(values["model_rows"]),
        "evaluation_count": len(values["evaluation_rows"]),
        "metric_count": len(values["metric_rows"]),
        "paired_statistic_count": len(values["paired_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": str(values["heavy_root"]),
        "prediction_path": str(prediction_path),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _resume_command(config):
    args = [
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python",
        "scripts/evaluation/application_tasks/run_simulation_mechanism_consumers.py",
        "--run-id", config.run_id,
        "--mechanism-representation-run-id",
        config.mechanism_representation_run_id,
        "--stress-representation-run-id", config.stress_representation_run_id,
        "--resume",
    ]
    for seed in config.seeds:
        args.extend(("--seed", str(seed)))
    return " ".join(shlex.quote(str(value)) for value in args) + "\n"


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
