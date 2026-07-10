"""Compact evidence writers for the Chronaris continuous adapter smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def write_chronaris_continuous_adapter_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    adapter_protocol: Mapping[str, object],
    path_rows: Sequence[Mapping[str, object]],
    lag_rows: Sequence[Mapping[str, object]],
    physics_rows: Sequence[Mapping[str, object]],
    ablation_rows: Sequence[Mapping[str, object]],
    causality_rows: Sequence[Mapping[str, object]],
    sensitivity_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    transform_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    acceptance_rows: Sequence[Mapping[str, object]],
    heavy_run_root: str,
) -> Mapping[str, str]:
    paths = {
        "adapter_protocol": run_root / "adapter_protocol.json",
        "continuous_path_audit": run_root / "continuous_path_audit.csv",
        "lag_scale_boundary_audit": run_root / "lag_scale_boundary_audit.csv",
        "physics_availability": run_root / "physics_availability.csv",
        "ablation_config_diff": run_root / "ablation_config_diff.csv",
        "attention_causality_audit": run_root / "attention_causality_audit.csv",
        "dual_stream_sensitivity": run_root / "dual_stream_sensitivity.csv",
        "parameter_budget": run_root / "parameter_budget.csv",
        "fold_transform_manifest": run_root / "fold_transform_manifest.json",
        "checkpoint_registry": run_root / "checkpoint_registry.json",
        "representation_export_manifest": run_root / "representation_export_manifest.json",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["adapter_protocol"], adapter_protocol)
    for key, rows in (
        ("continuous_path_audit", path_rows),
        ("lag_scale_boundary_audit", lag_rows),
        ("physics_availability", physics_rows),
        ("ablation_config_diff", ablation_rows),
        ("attention_causality_audit", causality_rows),
        ("dual_stream_sensitivity", sensitivity_rows),
        ("parameter_budget", parameter_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(paths["fold_transform_manifest"], transform_manifest)
    _write_json(paths["representation_export_manifest"], export_manifest)
    paths["report"].write_text(
        _render_report(
            status=status,
            path_rows=path_rows,
            physics_rows=physics_rows,
            ablation_rows=ablation_rows,
            causality_rows=causality_rows,
            sensitivity_rows=sensitivity_rows,
            parameter_rows=parameter_rows,
            acceptance_rows=acceptance_rows,
            export_manifest=export_manifest,
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 验证 Chronaris 连续双流、物理可用性和秒级因果融合进入同一任务无关表示路径。\n"
        "- 主干仅使用随机初始化和训练折归一化，尚未执行公共自监督训练，不产生下游任务指标。\n"
        "- 物理项 active 表示语义与有效残差对可计算，不表示随机初始化模型已获得物理一致性。\n"
        "- 四项消融只完成配置与前向路径审计，正式效果差异须等待锁定训练与统一下游评估。\n"
        "- 仿真读取仅限原始观测归档；本 run 未打开真值文件。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_chronaris_continuous_adapter_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "chronaris_continuous_adapter_smoke_only",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "downstream_metrics_produced": False,
            "simulation_oracle_opened": False,
            "heavy_run_root": heavy_run_root,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _render_report(
    *,
    status,
    path_rows,
    physics_rows,
    ablation_rows,
    causality_rows,
    sensitivity_rows,
    parameter_rows,
    acceptance_rows,
    export_manifest,
) -> str:
    passed = sum(bool(row["passed"]) for row in acceptance_rows)
    worst_future = max(row["future_perturbation_max_abs_delta"] for row in causality_rows)
    min_physiology = min(row["physiology_history_max_abs_delta"] for row in sensitivity_rows)
    min_vehicle = min(row["vehicle_history_max_abs_delta"] for row in sensitivity_rows)
    active_by_dataset = {
        dataset: sum(row["status"] == "active" for row in physics_rows if row["dataset_id"] == dataset)
        for dataset in ("simulation", "dingxin")
    }
    lines = [
        "# Chronaris 连续融合生产主干冒烟报告",
        "",
        "## 结论",
        "",
        f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
        "- 原始异步双流已通过独立 ODE-RNN、96 点连续查询、物理可用性和三尺度因果融合形成 64 维任务无关表示。",
        f"- 仿真与鼎新各完成一个留出折输出；当前新建 {export_manifest['current_run_built_count']} 个、复用 {export_manifest['current_run_reused_count']} 个，恢复复核复用 {export_manifest['resume_verification_reused_count']} 个。",
        f"- 未来扰动对历史输出的最大变化为 {worst_future:.3e}；生理/航电历史扰动的最小表示变化为 {min_physiology:.3e}/{min_vehicle:.3e}。",
        f"- 仿真与鼎新分别有 {active_by_dataset['simulation']}/{active_by_dataset['dingxin']} 个物理项可计算；不可用项保留空值与原因。",
        f"- 两条数据路径共记录 {sum(row['observation_update_count'] for row in path_rows)} 次观测更新，四项固定消融完成 {len(ablation_rows)} 次有限值前向。",
        "- 本 run 没有公共自监督训练或下游任务指标，不能据此比较 Chronaris 与基线效果。",
        "",
        "## 参数与运行审计",
        "",
        "| 数据 | 输入字段 | 参数量 | 前向耗时（秒） | ODE 方法 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in parameter_rows:
        lines.append(
            f"| {row['dataset_label']} | {row['input_feature_count']} | "
            f"{row['parameter_count']} | {row['forward_elapsed_s']:.3f} | "
            f"{row['ode_method']} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 为五个可训练编码器实现公共 masked reconstruction、短期预测和时延判别训练器。",
            "2. 运行单 fold、单 epoch 的六方法训练—导出—线性探针闭环烟雾测试。",
            "3. 闭环通过后按同一增强和预算启动 seed 17 开发筛选，不读取仿真锁定测试真值。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
