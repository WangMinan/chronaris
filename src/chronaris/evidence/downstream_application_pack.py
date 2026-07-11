"""Assemble locked downstream runs into one thesis-facing evidence package."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evidence.downstream_application_data import (
    DINGXIN_PRIMARY,
    METHOD_LABELS,
    SIMULATION_PRIMARY,
    build_ablation_primary_table,
    build_mechanism_mae_table,
    build_primary_metric_table,
    build_stress_heatmap_table,
)
from chronaris.evidence.downstream_application_figures import (
    configure_chinese_matplotlib,
    plot_ablation_advantage,
    plot_dingxin_representative_case,
    plot_mechanism_recovery,
    plot_method_primary_metrics,
    plot_simulation_oracle_case,
    plot_stress_slope_heatmap,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.evidence.downstream_application_pack")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DownstreamEvidencePackConfig:
    run_id: str = "2026-07-12_downstream-evidence-pack"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    dingxin_consumer_run_id: str = (
        "2026-07-12_dingxin-locked-consumers-coalesced"
    )
    simulation_consumer_run_id: str = "2026-07-12_simulation-locked-consumers"
    simulation_representation_run_id: str = (
        "2026-07-12_simulation-locked-representations"
    )
    stress_consumer_run_id: str = "2026-07-12_simulation-locked-stress-consumers"
    mechanism_consumer_run_id: str = "2026-07-12_simulation-mechanism-consumers"
    ablation_consumer_run_id: str = (
        "2026-07-12_simulation-chronaris-ablation-consumers"
    )
    finetuning_run_id: str = "2026-07-12_simulation-end-to-end-finetuning"
    public_adapter_run_id: str | None = "2026-06-07_public-adapter-calibration"
    transfer_consumer_run_id: str | None = (
        "2026-07-12_dingxin-synthetic-pretrain-adapt-consumers-coalesced"
    )


@dataclass(frozen=True, slots=True)
class DownstreamEvidencePackResult:
    run_id: str
    status: str
    run_root: str
    figure_count: int
    evidence_row_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_downstream_evidence_pack(config: DownstreamEvidencePackConfig):
    root = Path(config.compact_output_root) / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    upstream = _load_upstream_evidence(config)
    with open_task_eval_run_observer(
        run_root=root,
        run_id=config.run_id,
        stage_name="downstream_thesis_evidence_pack",
        logger=LOGGER,
        initial_progress={
            "upstream_run_ids": list(upstream),
            "reader_visible_language": "zh-CN",
            "real_and_simulated_evidence_mixed": False,
        },
    ) as progress:
        dingxin = pd.read_csv(
            Path(config.compact_output_root)
            / config.dingxin_consumer_run_id
            / "main_view_fold_summary.csv"
        )
        simulation = pd.read_csv(
            Path(config.compact_output_root)
            / config.simulation_consumer_run_id
            / "metric_long.csv"
        )
        stress = pd.read_csv(
            Path(config.compact_output_root)
            / config.stress_consumer_run_id
            / "stress_slopes.csv"
        )
        mechanism = pd.read_csv(
            Path(config.compact_output_root)
            / config.mechanism_consumer_run_id
            / "metric_long.csv"
        )
        ablation = pd.read_csv(
            Path(config.compact_output_root)
            / config.ablation_consumer_run_id
            / "full_ablation_metric_delta.csv"
        )
        dingxin_table = build_primary_metric_table(
            dingxin,
            DINGXIN_PRIMARY,
            value_column="mean",
        )
        simulation_table = build_primary_metric_table(
            simulation,
            SIMULATION_PRIMARY,
            value_column="value",
            role="held_out",
        )
        stress_table = build_stress_heatmap_table(stress)
        mechanism_table = build_mechanism_mae_table(mechanism)
        ablation_table = build_ablation_primary_table(ablation)
        dingxin_predictions = pd.read_csv(
            Path(config.heavy_output_root)
            / config.dingxin_consumer_run_id
            / "prediction_rows.csv"
        )
        simulation_predictions = pd.read_csv(
            Path(config.heavy_output_root)
            / config.simulation_consumer_run_id
            / "workload_predictions.csv"
        )
        simulation_sample_manifest = pd.read_csv(
            Path(config.compact_output_root)
            / config.simulation_representation_run_id
            / "data_manifest.csv"
        )
        tables = root / "tables"
        tables.mkdir(exist_ok=True)
        dingxin_table.to_csv(tables / "dingxin_primary_metrics.csv", index=False)
        simulation_table.to_csv(tables / "simulation_primary_metrics.csv", index=False)
        stress_table.to_csv(tables / "stress_degradation_slopes.csv")
        mechanism_table.to_csv(tables / "mechanism_recovery_mae.csv", index=False)
        ablation_table.to_csv(tables / "chronaris_ablation_advantage.csv", index=False)
        figures_root = root / "figures"
        figures = (
            plot_method_primary_metrics(
                dingxin_table,
                DINGXIN_PRIMARY,
                title="鼎新现有双流数据的下游任务表现",
                path=figures_root / "dingxin_method_comparison.png",
            ),
            plot_method_primary_metrics(
                simulation_table,
                SIMULATION_PRIMARY,
                title="模型无关半物理仿真的锁定下游表现",
                path=figures_root / "simulation_clean_method_comparison.png",
            ),
            plot_stress_slope_heatmap(
                stress_table,
                figures_root / "simulation_stress_degradation.png",
            ),
            plot_mechanism_recovery(
                mechanism_table,
                figures_root / "simulation_mechanism_recovery.png",
            ),
            plot_ablation_advantage(
                ablation_table,
                SIMULATION_PRIMARY,
                figures_root / "chronaris_mechanism_ablation.png",
            ),
            plot_dingxin_representative_case(
                dingxin_predictions,
                figures_root / "dingxin_representative_case.png",
            ),
            plot_simulation_oracle_case(
                simulation_predictions,
                simulation_sample_manifest,
                figures_root / "simulation_oracle_case.png",
            ),
        )
        figure_rows = [
            {
                "figure_id": path.stem,
                "path": str(path),
                "sha256": sha256_file(path),
                "language": "zh-CN",
                "source_runs": _figure_source_runs(path.stem, config),
            }
            for path in figures
        ]
        pd.DataFrame(figure_rows).to_csv(root / "figure_manifest.csv", index=False)
        evidence_rows = _evidence_matrix_rows(config, upstream)
        pd.DataFrame(evidence_rows).to_csv(root / "evidence_matrix.csv", index=False)
        winner_rows = _winner_rows(dingxin_table, "鼎新") + _winner_rows(
            simulation_table, "半物理仿真"
        )
        pd.DataFrame(winner_rows).to_csv(root / "primary_metric_leaders.csv", index=False)
        acceptance = _acceptance_rows(
            upstream=upstream,
            figures=figure_rows,
            evidence_rows=evidence_rows,
            dingxin_table=dingxin_table,
            simulation_table=simulation_table,
            stress_table=stress_table,
            mechanism_table=mechanism_table,
            ablation_table=ablation_table,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        pd.DataFrame(acceptance).to_csv(root / "acceptance.csv", index=False)
        font_family = configure_chinese_matplotlib()
        paths = _write_documents(
            root=root,
            config=config,
            upstream=upstream,
            figures=figure_rows,
            evidence_rows=evidence_rows,
            winners=winner_rows,
            acceptance=acceptance,
            font_family=font_family,
            status=status,
        )
        progress.finish(
            status=status,
            figure_count=len(figures),
            evidence_row_count=len(evidence_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return DownstreamEvidencePackResult(
        run_id=config.run_id,
        status=status,
        run_root=str(root),
        figure_count=len(figures),
        evidence_row_count=len(evidence_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _load_upstream_evidence(config):
    required = {
        "dingxin_real_weak_supervision": config.dingxin_consumer_run_id,
        "simulation_clean_locked": config.simulation_consumer_run_id,
        "simulation_stress_locked": config.stress_consumer_run_id,
        "simulation_mechanism_recovery": config.mechanism_consumer_run_id,
        "simulation_chronaris_ablation": config.ablation_consumer_run_id,
        "simulation_end_to_end_auxiliary": config.finetuning_run_id,
    }
    if config.transfer_consumer_run_id is not None:
        transfer = Path(config.compact_output_root) / config.transfer_consumer_run_id
        if transfer.is_dir():
            required["dingxin_synthetic_pretrain_adaptation"] = (
                config.transfer_consumer_run_id
            )
    result = {}
    for role, run_id in required.items():
        evidence_path = (
            Path(config.compact_output_root) / run_id / "evidence_manifest.json"
        )
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"downstream evidence pack requires completed run: {run_id}")
        result[role] = {
            "run_id": run_id,
            "evidence_path": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "format": payload.get("format"),
            "status": payload.get("status"),
        }
    if config.public_adapter_run_id is not None:
        public_root = (
            Path(config.compact_output_root) / config.public_adapter_run_id
        )
        summary_path = public_root / "public_adapter_calibration_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("evidence_layer") != "public_adapter_calibration":
            raise ValueError("public adapter evidence layer is invalid")
        result["public_dataset_adaptation"] = {
            "run_id": config.public_adapter_run_id,
            "evidence_path": str(summary_path),
            "evidence_sha256": sha256_file(summary_path),
            "format": "chronaris.public_adapter_calibration.legacy_v1",
            "status": "completed_legacy_evidence",
        }
    return result


def _evidence_matrix_rows(config, upstream):
    definitions = {
        "dingxin_real_weak_supervision": (
            "鼎新现有真实双流",
            "机动强度分类与机动诱发生理响应",
            "验证真实数据链路与应用形态",
            "弱监督任务，不等同于人工工作负荷或专家科目真值",
        ),
        "simulation_clean_locked": (
            "模型无关半物理仿真",
            "仿真负荷评估与机动状态分段",
            "利用已知真值比较六方法下游质量",
            "仿真真值不替代鼎新现场效果",
        ),
        "simulation_stress_locked": (
            "模型无关半物理仿真",
            "七因素压力与混合重压",
            "验证不规则采样、漂移、缺失与时延鲁棒性",
            "压力曲线只说明生成场景内的退化规律",
        ),
        "simulation_mechanism_recovery": (
            "模型无关半物理仿真",
            "时钟偏移与生理响应时延恢复",
            "验证融合表示保留的时间机制信息",
            "恢复真值来自生成器 oracle",
        ),
        "simulation_chronaris_ablation": (
            "模型无关半物理仿真",
            "连续演化、物理约束、因果掩码与多尺度时延消融",
            "解释 Chronaris 增益来源",
            "机制贡献不等同于真实因果效应",
        ),
        "simulation_end_to_end_auxiliary": (
            "模型无关半物理仿真",
            "任务标签端到端微调",
            "补充模型适配能力",
            "独立辅助表，不替代冻结表示质量主结论",
        ),
        "dingxin_synthetic_pretrain_adaptation": (
            "仿真预训练到鼎新适配",
            "无标签真实训练折适配",
            "检查仿真预训练的迁移价值",
            "六方法必须使用相同额外数据预算",
        ),
        "public_dataset_adaptation": (
            "UAB/NASA 公开数据",
            "公开工作负荷与认知状态适配",
            "提供人体状态任务的外部适配参照",
            "上下文构造第二输入流不等价于鼎新真实航电流",
        ),
    }
    return [
        {
            "evidence_role": role,
            "data_layer": definitions[role][0],
            "task": definitions[role][1],
            "supported_claim": definitions[role][2],
            "claim_boundary": definitions[role][3],
            **payload,
        }
        for role, payload in upstream.items()
    ]


def _winner_rows(table, data_layer):
    rows = []
    for (task, metric, direction), group in table.groupby(
        ["task_label", "metric_label", "direction"], sort=False
    ):
        winner = (
            group.loc[group["mean"].idxmax()]
            if direction == "higher"
            else group.loc[group["mean"].idxmin()]
        )
        rows.append(
            {
                "data_layer": data_layer,
                "task": task,
                "metric": metric,
                "direction": direction,
                "leading_method": winner["method"],
                "leading_method_label": METHOD_LABELS[winner["method"]],
                "mean": float(winner["mean"]),
                "std": float(winner["std"]),
                "descriptive_only": True,
            }
        )
    return rows


def _figure_source_runs(figure_id, config):
    if figure_id.startswith("dingxin"):
        return config.dingxin_consumer_run_id
    if figure_id.startswith("simulation_clean"):
        return config.simulation_consumer_run_id
    if figure_id.startswith("simulation_oracle"):
        return config.simulation_consumer_run_id
    if "stress" in figure_id:
        return config.stress_consumer_run_id
    if "mechanism_recovery" in figure_id:
        return config.mechanism_consumer_run_id
    return config.ablation_consumer_run_id


def _acceptance_rows(**values):
    return (
        _check("all_required_upstreams_completed", len(values["upstream"]) >= 6, len(values["upstream"]), ">=6"),
        _check("seven_chinese_figures", len(values["figures"]) == 7 and all(row["language"] == "zh-CN" for row in values["figures"]), len(values["figures"]), 7),
        _check("evidence_layers_not_mixed", len(values["evidence_rows"]) >= 6 and all(row["claim_boundary"] for row in values["evidence_rows"]), len(values["evidence_rows"]), ">=6"),
        _check("dingxin_primary_table", len(values["dingxin_table"]) == 18, len(values["dingxin_table"]), 18),
        _check("simulation_primary_table", len(values["simulation_table"]) == 18, len(values["simulation_table"]), 18),
        _check("seven_stress_factors", values["stress_table"].shape[1] == 7, values["stress_table"].shape[1], 7),
        _check("two_mechanism_targets", values["mechanism_table"]["target"].nunique() == 2, values["mechanism_table"]["target"].nunique(), 2),
        _check("four_ablation_variants", values["ablation_table"]["ablation_method"].nunique() == 4, values["ablation_table"]["ablation_method"].nunique(), 4),
    )


def _write_documents(**values):
    root = values["root"]
    paths = {
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "claim": root / "claim_boundary.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    _write_json(paths["protocol"], {
        "format": "chronaris.downstream_evidence_pack_protocol.v1",
        "config": asdict(values["config"]),
        "upstream": values["upstream"],
        "font_family": values["font_family"],
        "primary_metrics_are_predeclared": True,
        "real_and_simulated_evidence_mixed": False,
    })
    winner_lines = [
        f"- {row['data_layer']}—{row['task']}：{row['leading_method_label']}，{row['metric']}={row['mean']:.4f}。"
        for row in values["winners"]
    ]
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text(
        "\n".join(
            (
                "# 固定数据下游评估论文证据包",
                "",
                f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
                f"汇总 {len(values['evidence_rows'])} 层证据和 {len(values['figures'])} 幅中文论文图。",
                "",
                "## 预声明主指标的描述性领先方法",
                "",
                *winner_lines,
                "",
                "这些领先关系是结果汇总，不替代折级或轨迹级配对统计；真实弱监督、仿真真值与端到端微调保持独立表述。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim"].write_text(
        "# 结论边界\n\n鼎新结果只支撑现有真实双流上的弱监督应用任务；仿真结果支撑已知生成真值下的机制、压力和消融判断；端到端微调只说明任务适配能力。三类证据不得互相替代。\n",
        encoding="utf-8",
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evidence/build_downstream_application_pack.py "
        f"--run-id {values['config'].run_id}\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.downstream_evidence_pack.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "figure_count": len(values["figures"]),
        "evidence_row_count": len(values["evidence_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "upstream": values["upstream"],
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
