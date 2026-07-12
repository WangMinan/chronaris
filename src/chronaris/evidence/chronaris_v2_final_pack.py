"""Build the final locked Chronaris v2 gap-evidence package."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_protocol import (
    PRIMARY_METRIC_THRESHOLDS,
)
from chronaris.evidence.downstream_application_data import (
    METHOD_LABELS,
    STRESS_LABELS,
    build_mechanism_mae_table,
    build_stress_heatmap_table,
)
from chronaris.evidence.downstream_application_figures import (
    configure_chinese_matplotlib,
)


METRIC_LABELS = {
    "dingxin_maneuver_macro_f1": "鼎新机动强度分类 Macro-F1",
    "dingxin_high_response_auprc": "鼎新高生理响应识别 AUPRC",
    "dingxin_response_rmse": "鼎新连续生理响应 RMSE",
    "simulation_load_macro_f1": "仿真负荷分类 Macro-F1",
    "simulation_load_rmse": "仿真负荷回归 RMSE",
    "simulation_segmentation_macro_f1": "仿真机动分段 Macro-F1",
}


@dataclass(frozen=True, slots=True)
class ChronarisV2FinalPackConfig:
    run_id: str = "2026-07-13_chronaris-v2-final-evidence-pack"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    promotion_run_id: str = "2026-07-13_chronaris-v2-promotion-audit"
    dingxin_consumer_run_id: str = "2026-07-13_chronaris-v2-dingxin-locked-consumers"
    sealed_consumer_run_id: str = "2026-07-13_chronaris-v2-sealed-confirmation-consumers"
    v2_mechanism_run_id: str = "2026-07-13_chronaris-v2-mechanism-consumers"
    v1_mechanism_run_id: str = "2026-07-13_chronaris-v1-sealed-reference-mechanism"
    ablation_consumer_run_id: str = "2026-07-13_chronaris-v2-simulation-ablation-consumers"
    public_confirmation_run_id: str = "2026-07-13_chronaris-v2-public-adapter-confirmation"


def run_chronaris_v2_final_pack(
    config: ChronarisV2FinalPackConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2FinalPackConfig()
    root = Path(resolved.compact_output_root) / resolved.run_id
    root.mkdir(parents=True, exist_ok=True)
    promotion_root = Path(resolved.compact_output_root) / resolved.promotion_run_id
    sealed_root = Path(resolved.compact_output_root) / resolved.sealed_consumer_run_id
    v2_mechanism_root = Path(resolved.compact_output_root) / resolved.v2_mechanism_run_id
    v1_mechanism_root = Path(resolved.compact_output_root) / resolved.v1_mechanism_run_id
    ablation_root = Path(resolved.compact_output_root) / resolved.ablation_consumer_run_id
    public_root = Path(resolved.heavy_output_root) / resolved.public_confirmation_run_id
    for run_root in (
        promotion_root, sealed_root, v2_mechanism_root,
        v1_mechanism_root, ablation_root,
    ):
        _require_status(run_root / "evidence_manifest.json", {"completed"})
    _require_status(public_root / "evidence_manifest.json", {"completed", "partial"})
    audit = json.loads((promotion_root / "promotion_audit.json").read_text(encoding="utf-8"))
    primary = pd.read_csv(promotion_root / "primary_method_panel.csv")
    gate_details = pd.read_csv(promotion_root / "additional_gate_details.csv")
    stress = pd.read_csv(sealed_root / "stress_slopes.csv")
    v2_mechanism = pd.read_csv(v2_mechanism_root / "metric_long.csv")
    v1_mechanism = pd.read_csv(v1_mechanism_root / "metric_long.csv")
    ablation = pd.read_csv(ablation_root / "full_ablation_metric_delta.csv")
    public = json.loads((public_root / "fusion_refresh_summary.json").read_text(encoding="utf-8"))
    public_confirm = pd.read_csv(public["confirm_leaderboard_csv"])
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    primary_summary = _primary_summary(primary)
    stress_table = build_stress_heatmap_table(stress)
    mechanism_table = _mechanism_comparison(v1_mechanism, v2_mechanism)
    ablation_table = _ablation_primary(ablation)
    public_table = _public_summary(public_confirm)
    primary_summary.to_csv(tables / "six_primary_metrics.csv", index=False)
    gate_details.to_csv(tables / "additional_gate_details.csv", index=False)
    stress_table.to_csv(tables / "missingness_and_stress_slopes.csv")
    mechanism_table.to_csv(tables / "sealed_timing_mechanism.csv", index=False)
    ablation_table.to_csv(tables / "formal_ablation_deltas.csv", index=False)
    public_table.to_csv(tables / "public_adapter_confirmation.csv", index=False)
    configure_chinese_matplotlib()
    figure_paths = {
        "primary": _plot_primary(primary_summary, figures / "six_primary_metrics.png"),
        "stress": _plot_stress(stress_table, figures / "stress_slopes.png"),
        "mechanism": _plot_mechanism(mechanism_table, figures / "timing_mechanism.png"),
        "ablation": _plot_ablation(ablation_table, figures / "formal_ablation.png"),
    }
    report_path = root / "report.md"
    report_path.write_text(
        _report(audit, primary_summary, public_table, figure_paths),
        encoding="utf-8",
    )
    (root / "protocol.json").write_text(
        json.dumps({
            "format": "chronaris.v2_final_evidence_pack_protocol.v1",
            "config": asdict(resolved),
            "confirmed_v1_evidence_changed": False,
            "results_returned_to_same_development_round": False,
            "dingxin_evidence_role": "fixed_data_same_protocol_recheck",
            "public_second_stream_role": "context_construction",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evidence/run_chronaris_v2_final_pack.py "
        f"--run-id {resolved.run_id}\n",
        encoding="utf-8",
    )
    (root / "evidence_manifest.json").write_text(
        json.dumps({
            "format": "chronaris.v2_final_evidence_pack.v1",
            "run_id": resolved.run_id,
            "status": "completed",
            "promoted": bool(audit["promoted"]),
            "paper_main_model": audit["paper_main_model"],
            "figure_paths": {key: str(value) for key, value in figure_paths.items()},
            "report_path": str(report_path),
            "confirmed_v1_evidence_changed": False,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def _primary_summary(primary):
    grouped = primary.groupby(["metric_name", "method", "direction"], as_index=False)["value"].mean()
    grouped["threshold"] = grouped["metric_name"].map(
        {name: value[1] for name, value in PRIMARY_METRIC_THRESHOLDS.items()}
    )
    grouped["metric_label"] = grouped["metric_name"].map(METRIC_LABELS)
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS)
    grouped["threshold_passed"] = np.where(
        grouped["direction"].eq("higher"),
        grouped["value"] > grouped["threshold"],
        grouped["value"] < grouped["threshold"],
    )
    return grouped


def _mechanism_comparison(v1, v2):
    first = build_mechanism_mae_table(v1).query("method == 'chronaris'").rename(columns={"value": "v1_mae_s"})
    second = build_mechanism_mae_table(v2).query("method == 'chronaris'").rename(columns={"value": "v2_mae_s"})
    merged = first[["target", "v1_mae_s"]].merge(second[["target", "v2_mae_s"]], on="target", validate="one_to_one")
    merged["v2_over_v1"] = merged["v2_mae_s"] / merged["v1_mae_s"]
    merged["within_10_percent"] = merged["v2_over_v1"] <= 1.10
    return merged


def _ablation_primary(frame):
    primary = {
        ("simulated_future_workload_classification", "minirocket", "macro_f1"),
        ("simulated_future_workload_regression", "minirocket", "rmse"),
        ("simulated_maneuver_state_segmentation", "causal_tcn_duration", "frame_macro_f1"),
    }
    selected = frame[frame.apply(lambda row: (row["task"], row["consumer"], row["metric"]) in primary, axis=1)]
    return selected.groupby(["ablation_method", "task", "consumer", "metric"], as_index=False)["full_advantage_normalized"].mean()


def _public_summary(frame):
    columns = [
        "dataset_id", "candidate_id", "seed", "fold_count",
        "combined_macro_f1", "combined_balanced_accuracy",
        "benchmark_only_macro_f1", "loft_only_macro_f1",
        "mean_rmse", "mean_mae", "n_back_rmse", "heat_the_chair_rmse",
    ]
    available = [column for column in columns if column in frame.columns]
    result = frame[available].copy()
    if result.empty or set(result["dataset_id"]) != {
        "nasa_csm", "uab_workload_dataset"
    }:
        raise ValueError("public confirmation does not cover NASA and UAB")
    return result.sort_values(["dataset_id", "seed"], kind="stable")


def _plot_primary(frame, path):
    methods = list(dict.fromkeys(frame["method"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, (metric, group) in zip(axes.flat, frame.groupby("metric_name", sort=False), strict=True):
        group = group.set_index("method").reindex(methods)
        axis.bar(range(len(group)), group["value"], color="#4C78A8")
        axis.axhline(float(group["threshold"].iloc[0]), color="#D62728", linestyle="--", label="晋级硬门槛")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(range(len(group)), [METHOD_LABELS[value] for value in group.index], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def _plot_stress(table, path):
    fig, axis = plt.subplots(figsize=(11, 4.8))
    image = axis.imshow(table.to_numpy(float), cmap="RdYlGn", aspect="auto")
    axis.set_yticks(range(len(table)), [METHOD_LABELS[value] for value in table.index])
    axis.set_xticks(
        range(len(table.columns)),
        [STRESS_LABELS.get(value, str(value)) for value in table.columns],
        rotation=30,
        ha="right",
    )
    axis.set_title("七类观测压力下的主指标平均退化斜率（越高越好）")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    return _save(fig, path)


def _plot_mechanism(table, path):
    fig, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(table)); width = 0.36
    axis.bar(x - width / 2, table["v1_mae_s"], width, label="Chronaris v1")
    axis.bar(x + width / 2, table["v2_mae_s"], width, label="Chronaris v2")
    labels = table["target"].map({
        "relative_clock_offset_magnitude_s": "时钟偏移幅值",
        "primary_physiology_response_lag_s": "生理响应时延",
    })
    axis.set_xticks(x, labels)
    axis.set_ylabel("平均绝对误差（秒）")
    axis.set_title("同一独立封存族上的时间机制恢复")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def _plot_ablation(table, path):
    pivot = table.pivot(index="ablation_method", columns="task", values="full_advantage_normalized")
    pivot = pivot.rename(
        index={
            "chronaris_no_corrected_physics": "去修正物理约束",
            "chronaris_no_missingness_curriculum": "去缺失课程",
        },
        columns={
            "simulated_future_workload_classification": "仿真负荷分类",
            "simulated_future_workload_regression": "仿真负荷回归",
            "simulated_maneuver_state_segmentation": "仿真机动分段",
        },
    )
    fig, axis = plt.subplots(figsize=(10, 4.8))
    pivot.plot.bar(ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("完整模型方向归一优势")
    axis.set_title("Chronaris v2 修正物理约束与缺失课程的正式消融")
    axis.set_xlabel("")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _report(audit, primary, public, figures):
    failed = [row["metric_name"] for row in audit["metric_audits"] if not row["passed"]]
    return "\n".join((
        "# Chronaris v2 锁定复验与差距证据包",
        "",
        f"结论：v2 不晋级；论文主模型保持为 `{audit['paper_main_model']}`。",
        f"未通过的预声明主指标：{', '.join(METRIC_LABELS[value] for value in failed)}。",
        "鼎新结果属于固定数据同协议复验；独立仿真族、公开数据适配和正式消融均在配置锁定后运行，结果不回流当前开发轮次。",
        f"六主指标图：`{figures['primary']}`；压力斜率图：`{figures['stress']}`。",
        f"时间机制图：`{figures['mechanism']}`；正式消融图：`{figures['ablation']}`。",
        f"公开数据适配汇总包含 {len(public)} 个数据集—任务条目，第二输入流仅作上下文构造。",
        "",
    ))


def _require_status(path, allowed):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") not in allowed:
        raise ValueError(f"upstream evidence is not complete: {path}")
