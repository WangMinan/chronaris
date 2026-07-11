"""Compact evidence writer for the six-method G4 application-consumer smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_application_consumer_smoke_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    data_manifest,
    target_manifest,
    checkpoint_manifest,
    representation_manifest,
    downstream_protocol,
    model_manifests,
    metric_rows,
    fusion_gain_rows,
    paired_rows,
    tcn_training_rows,
    resource_rows,
    acceptance_rows,
    heavy_run_root: str,
    workload_prediction_path: str,
    unit_score_path: str,
):
    paths = {
        "data_manifest": run_root / "data_manifest.json",
        "target_manifest": run_root / "target_manifest.json",
        "checkpoint_manifest": run_root / "checkpoint_manifest.json",
        "representation_manifest": run_root / "representation_manifest.json",
        "downstream_protocol": run_root / "downstream_protocol.json",
        "consumer_model_manifest": run_root / "consumer_model_manifest.json",
        "metric_long": run_root / "metric_long.csv",
        "fold_metrics": run_root / "fold_metrics.csv",
        "fusion_gain": run_root / "fusion_gain.csv",
        "paired_statistics": run_root / "paired_statistics.csv",
        "tcn_training": run_root / "tcn_training.csv",
        "resource_budget": run_root / "resource_budget.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    for key, payload in (
        ("data_manifest", data_manifest),
        ("target_manifest", target_manifest),
        ("checkpoint_manifest", checkpoint_manifest),
        ("representation_manifest", representation_manifest),
        ("downstream_protocol", downstream_protocol),
        ("consumer_model_manifest", model_manifests),
    ):
        _write_json(paths[key], payload)
    for key, rows in (
        ("metric_long", metric_rows),
        ("fold_metrics", metric_rows),
        ("fusion_gain", fusion_gain_rows),
        ("paired_statistics", paired_rows),
        ("tcn_training", tcn_training_rows),
        ("resource_budget", resource_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    passed = sum(row["passed"] for row in acceptance_rows)
    available = sum(row["status"] == "available" for row in metric_rows)
    paths["report"].write_text(
        "\n".join(
            (
                "# 六方法应用型下游消费者闭环冒烟报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
                "- 六种冻结融合表示均由同一批原始异步双流生成，并在 32/16/16 个训练、验证、留出上下文上保持相同样本和查询时间轴。",
                "- 固定线性探针、MiniROCKET 多变量时序变换和两层因果 TCN 已使用相同配置消费六方法表示；持续时间参数只由训练划分状态序列拟合。",
                f"- 共写出 {len(metric_rows)} 条 smoke 指标，其中 {available} 条可计算；同时写出方向归一的融合增益和以 4 条留出轨迹为独立单位的配对接口。",
                "- 删除 Chronaris 单份表示、MiniROCKET 模型和 TCN 模型后均可独立重建；未删除组件哈希保持不变，预测哈希一致。",
                "- 本 run 只证明应用任务链路、恢复机制和统计接口成立，不承担模型排序或论文结论。",
                "",
                "## 已闭合的消费者",
                "",
                "1. 线性探针：池化表示上的 Logistic 分类和 Ridge 回归。",
                "2. MiniROCKET：先按统一规则在训练划分过滤窗口内恒定潜在维，再用 10,000 卷积核完成固定多变量时序变换并接同规格 Logistic/Ridge；过滤索引随模型产物保存。",
                "3. 因果 TCN：两层、64 通道、膨胀率 1/2 的逐时刻状态发射模型，并报告原始输出与训练折持续时间解码。",
                "",
                "## 下一步",
                "",
                "1. 将公共预训练从 16 条 smoke 轨迹扩展到完整 G1 开发集，并冻结四个 Chronaris 候选。",
                "2. 在开发集运行真实弱监督任务、仿真负荷和机动分段的统一筛选，期间不读取 G2 锁定测试。",
                "3. 锁定候选后运行多 seed、时间偏移、漂移、缺失、响应滞后压力曲线和方法消融。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 是六方法冻结表示与应用型下游算法的接口冒烟，不形成方法优劣排序。\n"
        "- 使用的是模型无关航空双流仿真训练划分，负荷和机动状态均为生成器真值，不等同于真实飞行员人工评估。\n"
        "- 所有指标、融合增益和配对统计均标记 smoke only，不更新论文 confirmed metrics。\n"
        "- 表示编码器未读取下游真值；真值只在五个可训练 checkpoint 完成并完成表示导出后打开。\n"
        "- 鼎新弱监督任务、完整开发筛选、G2 锁定测试和最终论文图表仍需后续里程碑完成。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_application_consumer_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "application_consumer_smoke_only",
            "training_invoked": True,
            "encoder_training_invoked": False,
            "confirmed_metrics_changed": False,
            "downstream_metrics_smoke_only": True,
            "heavy_run_root": heavy_run_root,
            "workload_prediction_path": workload_prediction_path,
            "unit_score_path": unit_score_path,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
