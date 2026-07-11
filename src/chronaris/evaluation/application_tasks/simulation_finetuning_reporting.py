"""Compact evidence writers for the simulation fine-tuning auxiliary table."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def write_simulation_finetuning_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "finetuning_inventory.csv",
        "training": root / "training_history.csv",
        "exports": root / "representation_inventory.csv",
        "metrics": root / "metric_long.csv",
        "gains": root / "fusion_gain.csv",
        "paired": root / "paired_statistics.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "claim": root / "claim_boundary.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    for key, rows in (
        ("results", values["result_rows"]),
        ("training", values["training_rows"]),
        ("exports", values["export_rows"]),
        ("metrics", values["metric_rows"]),
        ("gains", values["gain_rows"]),
        ("paired", values["paired_rows"]),
        ("acceptance", values["acceptance"]),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(
        paths["protocol"],
        {
            "format": "chronaris.simulation_end_to_end_finetuning_protocol.v1",
            "config": asdict(values["config"]),
            "representation_family": "end_to_end_finetuned_v1",
            "label_used_for_encoder_training": True,
            "fit_role": "train",
            "early_stopping_role": "validation",
            "locked_evaluation_role": "held_out",
            "losses": [
                "workload_class_cross_entropy",
                "standardized_workload_mse",
                "maneuver_state_cross_entropy",
            ],
            "naive_time_sync_update_mode": "head_only_nonparametric_encoder",
            "baseline_device": values["baseline_device"],
            "chronaris_device": values["chronaris_device"],
            "target_manifest": values["target_manifest"],
        },
    )
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text(
        "\n".join(
            (
                "# 仿真端到端微调辅助评估",
                "",
                f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
                f"完成 {len(values['result_rows'])} 个方法—随机种子组合，输出 {len(values['metric_rows'])} 条指标。",
                "该表使用任务标签更新编码器和任务头，仅作为适配能力辅助证据，与冻结任务无关表示主表严格分开。",
                "朴素时间同步没有可学习编码器，因此只训练同容量任务头并作为非参数控制。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim"].write_text(
        "# 结论边界\n\n本 run 使用仿真任务真值进行端到端适配，不替代冻结表示质量结论，也不等同于鼎新人工工作负荷评价。\n",
        encoding="utf-8",
    )
    pd.DataFrame(values["unit_rows"]).to_csv(
        values["heavy_root"] / "unit_score_rows.csv", index=False
    )
    seed_flags = " ".join(f"--seed {seed}" for seed in values["config"].seeds)
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_end_to_end_finetuning.py "
        f"--run-id {values['config'].run_id} {seed_flags} "
        f"--learning-rate {values['config'].learning_rate} "
        f"--max-epochs {values['config'].max_epochs} "
        f"--patience {values['config'].patience} "
        f"--batch-size {values['config'].batch_size} "
        f"--baseline-device {values['baseline_device']} "
        f"--chronaris-device {values['chronaris_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence"],
        {
            "format": "chronaris.simulation_end_to_end_finetuning_evidence.v1",
            "run_id": values["config"].run_id,
            "status": values["status"],
            "representation_family": "end_to_end_finetuned_v1",
            "method_seed_count": len(values["result_rows"]),
            "metric_count": len(values["metric_rows"]),
            "acceptance_pass_count": passed,
            "acceptance_check_count": len(values["acceptance"]),
            "heavy_run_root": str(values["heavy_root"]),
            "output_paths": {key: str(path) for key, path in paths.items()},
        },
    )
    return paths


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
