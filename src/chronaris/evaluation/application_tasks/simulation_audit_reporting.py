"""Compact artifact and report writers for the simulation benchmark audit."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.evaluation.application_tasks.simulation_audit_figures import (
    render_simulation_audit_figures,
)


def write_simulation_audit_outputs(
    *,
    config,
    benchmark,
    validation: pd.DataFrame,
    paired: pd.DataFrame,
    acceptance_rows: Sequence[Mapping[str, object]],
    compact_root: Path,
    status: str,
) -> dict[str, str]:
    """Write reviewable compact evidence while keeping raw arrays out of Git."""

    validation_path = compact_root / "oracle_validation.csv"
    validation.to_csv(validation_path, index=False)
    paired_path = compact_root / "paired_observation_audit.csv"
    paired.to_csv(paired_path, index=False)
    acceptance_path = compact_root / "acceptance_checks.csv"
    pd.DataFrame(acceptance_rows).to_csv(acceptance_path, index=False)
    scenario_coverage = _scenario_coverage(validation)
    scenario_path = compact_root / "scenario_coverage.csv"
    scenario_coverage.to_csv(scenario_path, index=False)
    split_path = compact_root / "generator_family_split.json"
    _write_json(split_path, benchmark.split_identity)
    config_path = compact_root / "generator_config.json"
    heavy_manifest = json.loads(
        Path(benchmark.simulation_manifest_path).read_text(encoding="utf-8")
    )
    _write_json(
        config_path,
        {
            "audit_config": asdict(config),
            "duration_s": heavy_manifest["duration_s"],
            "truth_rate_hz": heavy_manifest["truth_rate_hz"],
            "split_specs": heavy_manifest["split_specs"],
            "observation_scenarios": heavy_manifest["observation_scenarios"],
        },
    )
    figures = render_simulation_audit_figures(
        validation=validation,
        output_root=compact_root / "figures",
    )
    figure_path = compact_root / "figure_manifest.json"
    _write_json(figure_path, {"figures": list(figures)})
    report_path = compact_root / "report.md"
    report_path.write_text(
        _render_report(
            mode=config.mode,
            status=status,
            benchmark=benchmark,
            acceptance_rows=acceptance_rows,
            scenario_coverage=scenario_coverage,
        ),
        encoding="utf-8",
    )
    claim_path = compact_root / "claim_boundary.md"
    claim_path.write_text(
        "# 论断边界\n\n"
        "- 本 run 是方法无关仿真真值与压力场景验证，不是新增鼎新真实数据。\n"
        "- 仿真负荷是生成器内部状态，不是人工工作负荷评价。\n"
        "- 仿真指标不得与鼎新真实指标求平均。\n"
        "- 本 run 未训练待比较模型，也未修改既有确认指标。\n",
        encoding="utf-8",
    )
    resume_path = compact_root / "resume_command.txt"
    resume_path.write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/simulation/generate_aviation_dual_stream.py "
        f"--mode {config.mode} --heavy-run-id {config.heavy_run_id} "
        f"--compact-run-id {config.compact_run_id} --resume\n",
        encoding="utf-8",
    )
    evidence_path = compact_root / "evidence_manifest.json"
    _write_json(
        evidence_path,
        {
            "run_id": config.compact_run_id,
            "status": status,
            "evidence_layer": "method_independent_simulation_truth",
            "mode": config.mode,
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "heavy_run_root": benchmark.run_root,
            "heavy_simulation_manifest": benchmark.simulation_manifest_path,
            "output_paths": {
                "oracle_validation": str(validation_path),
                "paired_observation_audit": str(paired_path),
                "acceptance_checks": str(acceptance_path),
                "scenario_coverage": str(scenario_path),
                "generator_family_split": str(split_path),
                "generator_config": str(config_path),
                "figure_manifest": str(figure_path),
                "report": str(report_path),
                "claim_boundary": str(claim_path),
                "resume_command": str(resume_path),
                "progress": str(compact_root / "progress.json"),
                "run_log": str(compact_root / "run.log"),
            },
        },
    )
    return {
        "report_path": str(report_path),
        "evidence_manifest_path": str(evidence_path),
    }


def _scenario_coverage(validation: pd.DataFrame) -> pd.DataFrame:
    result = (
        validation.groupby(
            ["split_id", "generator_family", "scenario_id"],
            sort=True,
        )
        .agg(
            trajectory_count=("trajectory_id", "nunique"),
            mean_vehicle_missing_ratio=("vehicle_missing_ratio", "mean"),
            mean_physiology_missing_ratio=("physiology_missing_ratio", "mean"),
            workload_low_ratio=("workload_low_ratio", "mean"),
            workload_medium_ratio=("workload_medium_ratio", "mean"),
            workload_high_ratio=("workload_high_ratio", "mean"),
        )
        .reset_index()
    )
    split_order = {
        value: index
        for index, value in enumerate(
            ("train", "validation", "locked_test", "smoke_g1", "smoke_g2")
        )
    }
    scenario_order = {
        value: index
        for index, value in enumerate(
            (
                "clean_asynchronous",
                "sampling_jitter",
                "clock_offset_and_drift",
                "random_missing",
                "block_missing_and_long_lag",
                "mixed_severe",
            )
        )
    }
    result["_split_order"] = result["split_id"].map(split_order)
    result["_scenario_order"] = result["scenario_id"].map(scenario_order)
    return (
        result.sort_values(["_split_order", "_scenario_order"])
        .drop(columns=["_split_order", "_scenario_order"])
        .reset_index(drop=True)
    )


def _render_report(
    *,
    mode: str,
    status: str,
    benchmark,
    acceptance_rows,
    scenario_coverage,
) -> str:
    failed = [str(row["check_id"]) for row in acceptance_rows if not row["passed"]]
    mode_label = {"smoke": "冒烟验证", "formal": "正式基准"}.get(mode, mode)
    status_label = {"completed": "完成", "partial": "部分完成"}.get(status, status)
    lines = [
        "# 航空人机异步双流仿真审计报告",
        "",
        "## 结论",
        "",
        f"- 运行模式：{mode_label}；状态：{status_label}。",
        (
            f"- 潜在架次：{benchmark.latent_sortie_count}；"
            f"成对观测场景：{benchmark.observed_scenario_count}。"
        ),
        (
            f"- 验收检查：{len(acceptance_rows) - len(failed)}/"
            f"{len(acceptance_rows)} 通过。"
        ),
        f"- 复用已校验场景：{benchmark.resumed_scenario_count}。",
        "- 生成器只接收场景、飞行员参数档案和随机种子；不读取待比较方法或评价结果。",
    ]
    if failed:
        lines.append("- 未通过项：" + "、".join(failed) + "。")
    lines.extend(
        [
            "",
            "## 场景覆盖",
            "",
            "| 数据划分 | 生成族 | 场景 | 轨迹数 | 航电缺失 | 生理缺失 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in scenario_coverage.itertuples(index=False):
        lines.append(
            f"| {_display_split(row.split_id)} | "
            f"{_display_family(row.generator_family)} | "
            f"{_display_scenario(row.scenario_id)} | "
            f"{row.trajectory_count} | {row.mean_vehicle_missing_ratio:.1%} | "
            f"{row.mean_physiology_missing_ratio:.1%} |"
        )
    next_steps = (
        [
            "1. 冒烟验证通过后生成 96/24/48 个正式潜在架次及六个规范场景。",
            "2. 正式数据通过后再增加锁定测试单因素压力扫描。",
            "3. 六方法只能读取 `raw_dual_stream.npz`，任务和机制指标单独读取真值文件。",
        ]
        if mode == "smoke"
        else [
            "1. 基于锁定测试轨迹增加时间抖动、偏移、漂移、缺失、时延和信噪比单因素压力扫描。",
            "2. 建立统一双流输入与融合表示合同，六方法只读取观测文件。",
            "3. 任务构造和机制指标单独读取真值文件，不向编码器暴露真值字段。",
        ]
    )
    lines.extend(["", "## 下一步", "", *next_steps, ""])
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _display_split(value: str) -> str:
    return {
        "smoke_g1": "G1 状态空间（冒烟）",
        "smoke_g2": "G2 事件样条（冒烟）",
        "train": "训练集",
        "validation": "验证集",
        "locked_test": "锁定测试集",
    }.get(value, value)


def _display_family(value: str) -> str:
    return {
        "g1_state_space": "G1 状态空间生成族",
        "g2_event_spline": "G2 事件样条生成族",
    }.get(value, value)


def _display_scenario(value: str) -> str:
    return {
        "clean_asynchronous": "干净异步",
        "sampling_jitter": "采样抖动",
        "clock_offset_and_drift": "时钟偏移与漂移",
        "random_missing": "随机缺失",
        "block_missing_and_long_lag": "连续缺失与长时延",
        "mixed_severe": "混合重度压力",
    }.get(value, value)
