"""One-shot, fail-closed promotion audit for the locked Chronaris v2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_protocol import (
    audit_v2_promotion,
)
from chronaris.evidence.downstream_application_data import (
    DINGXIN_PRIMARY,
    SIMULATION_PRIMARY,
    build_mechanism_mae_table,
    build_stress_heatmap_table,
)


CANONICAL_CONFIRMATION_SCENARIO = "timestamp_jitter_000ms"
SINGLE_STREAM_TOLERANCE = 0.02
PRIMARY_NAMES = (
    "dingxin_maneuver_macro_f1",
    "dingxin_high_response_auprc",
    "dingxin_response_rmse",
    "simulation_load_macro_f1",
    "simulation_load_rmse",
    "simulation_segmentation_macro_f1",
)


@dataclass(frozen=True, slots=True)
class ChronarisV2PromotionConfig:
    run_id: str = "2026-07-13_chronaris-v2-promotion-audit"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    dingxin_consumer_run_id: str = (
        "2026-07-13_chronaris-v2-dingxin-locked-consumers"
    )
    simulation_confirmation_consumer_run_id: str = (
        "2026-07-13_chronaris-v2-sealed-confirmation-consumers"
    )
    mechanism_consumer_run_id: str = (
        "2026-07-13_chronaris-v2-mechanism-consumers"
    )
    v1_mechanism_consumer_run_id: str = (
        "2026-07-12_simulation-mechanism-consumers"
    )
    locked_configuration_path: str = (
        "docs/artifacts/runs/2026-07-12_chronaris-v2-dingxin-inner-"
        "confirmation-r3/locked_configuration.json"
    )
    task_independent_diagnostics_path: str = (
        "docs/artifacts/runs/2026-07-12_chronaris-v2-hyperparameter-"
        "diagnostics-seed17-r3/candidate_diagnostics.csv"
    )
    canonical_confirmation_scenario: str = CANONICAL_CONFIRMATION_SCENARIO


@dataclass(frozen=True, slots=True)
class ChronarisV2PromotionResult:
    run_id: str
    status: str
    promoted: bool
    paper_main_model: str
    compact_run_root: str
    report_path: str
    audit_path: str


def run_chronaris_v2_promotion_audit(
    config: ChronarisV2PromotionConfig,
) -> ChronarisV2PromotionResult:
    root = Path(config.compact_output_root) / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    for run_id in (
        config.dingxin_consumer_run_id,
        config.simulation_confirmation_consumer_run_id,
        config.mechanism_consumer_run_id,
    ):
        _require_completed_evidence(Path(config.compact_output_root) / run_id)
    dingxin = pd.read_csv(
        Path(config.compact_output_root)
        / config.dingxin_consumer_run_id
        / "main_view_fold_summary.csv"
    )
    simulation_root = (
        Path(config.compact_output_root)
        / config.simulation_confirmation_consumer_run_id
    )
    simulation = pd.read_csv(simulation_root / "metric_long.csv")
    seed_metrics, primary_panels = build_locked_seed_metric_rows(
        dingxin,
        simulation,
        canonical_scenario=config.canonical_confirmation_scenario,
    )
    recall_rows = build_class_recall_rows(
        Path(config.heavy_output_root)
        / config.dingxin_consumer_run_id
        / "prediction_rows.csv"
    )
    mechanism = pd.read_csv(
        Path(config.compact_output_root)
        / config.mechanism_consumer_run_id
        / "metric_long.csv"
    )
    v1_mechanism = pd.read_csv(
        Path(config.compact_output_root)
        / config.v1_mechanism_consumer_run_id
        / "metric_long.csv"
    )
    slopes = pd.read_csv(simulation_root / "stress_slopes.csv")
    diagnostics = pd.read_csv(config.task_independent_diagnostics_path)
    locked = json.loads(
        Path(config.locked_configuration_path).read_text(encoding="utf-8")
    )
    gate_details, gates = build_additional_gate_rows(
        primary_panels=primary_panels,
        recall_rows=recall_rows,
        mechanism=mechanism,
        v1_mechanism=v1_mechanism,
        slopes=slopes,
        diagnostics=diagnostics,
        locked_candidate_id=str(locked["candidate"]["candidate_id"]),
    )
    audit = audit_v2_promotion(seed_metrics, additional_gates=gates)
    audit.update(
        {
            "run_id": config.run_id,
            "config": asdict(config),
            "canonical_confirmation_scenario": config.canonical_confirmation_scenario,
            "single_stream_no_harm_absolute_tolerance": SINGLE_STREAM_TOLERANCE,
            "locked_results_returned_to_development": False,
        }
    )
    pd.DataFrame(seed_metrics).to_csv(root / "primary_seed_metrics.csv", index=False)
    pd.DataFrame(primary_panels).to_csv(root / "primary_method_panel.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(root / "maneuver_class_recall.csv", index=False)
    pd.DataFrame(gate_details).to_csv(root / "additional_gate_details.csv", index=False)
    audit_path = root / "promotion_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = "promoted" if audit["promoted"] else "not_promoted"
    report_path = root / "report.md"
    report_path.write_text(
        "\n".join(
            (
                "# Chronaris v2 一次性锁定晋级审计",
                "",
                f"结论：{status}；论文主模型保持为 `{audit['paper_main_model']}`。",
                "六项主指标和附加门禁均由锁定产物自动抽取；确认结果不回流当前开发轮次。",
                "",
            )
        ),
        encoding="utf-8",
    )
    evidence = {
        "format": "chronaris.v2_promotion_evidence.v1",
        "run_id": config.run_id,
        "status": "completed",
        "promoted": bool(audit["promoted"]),
        "paper_main_model": audit["paper_main_model"],
        "confirmed_v1_evidence_changed": False,
        "locked_results_returned_to_development": False,
        "output_paths": {
            "audit": str(audit_path),
            "report": str(report_path),
        },
    }
    (root / "evidence_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ChronarisV2PromotionResult(
        run_id=config.run_id,
        status=status,
        promoted=bool(audit["promoted"]),
        paper_main_model=str(audit["paper_main_model"]),
        compact_run_root=str(root),
        report_path=str(report_path),
        audit_path=str(audit_path),
    )


def build_locked_seed_metric_rows(dingxin, simulation, *, canonical_scenario):
    panels: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    dingxin_names = PRIMARY_NAMES[:3]
    simulation_names = PRIMARY_NAMES[3:]
    for name, specification in zip(dingxin_names, DINGXIN_PRIMARY, strict=True):
        panel = _select_panel(dingxin, specification, value_column="mean")
        _append_primary_rows(seed_rows, panels, panel, name=name)
    simulation = simulation[simulation["scenario_id"].eq(canonical_scenario)]
    for name, specification in zip(
        simulation_names, SIMULATION_PRIMARY, strict=True
    ):
        panel = _select_panel(simulation, specification, value_column="value")
        _append_primary_rows(seed_rows, panels, panel, name=name)
    return seed_rows, panels


def _select_panel(frame, specification, *, value_column):
    panel = frame.copy()
    for column in ("task", "consumer", "metric"):
        panel = panel[panel[column].eq(specification[column])]
    required = {"seed", "method", "direction", value_column}
    if panel.empty or not required.issubset(panel.columns):
        raise ValueError(f"locked primary panel unavailable: {specification}")
    grouped = (
        panel.groupby(["seed", "method", "direction"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "value"})
    )
    return grouped


def _append_primary_rows(seed_rows, panels, panel, *, name):
    methods = set(panel["method"])
    if methods != {
        "physiology_only", "vehicle_only", "naive_time_sync",
        "mult", "contiformer", "chronaris",
    }:
        raise ValueError(f"primary metric {name} does not cover all six methods")
    for seed, group in panel.groupby("seed"):
        if len(group) != 6:
            raise ValueError(f"primary metric {name} seed {seed} is incomplete")
        direction = str(group["direction"].iloc[0])
        chronaris = float(group.loc[group["method"].eq("chronaris"), "value"].iloc[0])
        best = float(group["value"].max() if direction == "higher" else group["value"].min())
        seed_rows.append(
            {
                "metric_name": name,
                "seed": int(seed),
                "value": chronaris,
                "rank_first": bool(np.isclose(chronaris, best) or (
                    chronaris > best if direction == "higher" else chronaris < best
                )),
            }
        )
        for row in group.to_dict("records"):
            panels.append({"metric_name": name, **row})


def build_class_recall_rows(prediction_path):
    frame = pd.read_csv(prediction_path)
    selected = frame[
        frame["task"].eq("maneuver_intensity_classification")
        & frame["consumer"].eq("minirocket")
        & frame["role"].eq("held_out")
    ].copy()
    if selected.empty:
        raise ValueError("Dingxin maneuver prediction rows are unavailable")
    rows = []
    for (seed, method, class_id), group in selected.groupby(
        ["seed", "method", "truth"], sort=True
    ):
        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "class_id": int(class_id),
                "recall": float((group["prediction"] == group["truth"]).mean()),
                "sample_count": len(group),
            }
        )
    return rows


def build_additional_gate_rows(
    *, primary_panels, recall_rows, mechanism, v1_mechanism, slopes,
    diagnostics, locked_candidate_id,
):
    details: list[dict[str, object]] = []
    panel = pd.DataFrame(primary_panels)
    no_harm = True
    for (metric_name, seed), group in panel.groupby(["metric_name", "seed"]):
        direction = str(group["direction"].iloc[0])
        chronaris = float(group.loc[group["method"].eq("chronaris"), "value"].iloc[0])
        singles = group[group["method"].isin(("physiology_only", "vehicle_only"))]
        best_single = float(
            singles["value"].max() if direction == "higher" else singles["value"].min()
        )
        passed = (
            chronaris >= best_single - SINGLE_STREAM_TOLERANCE
            if direction == "higher"
            else chronaris <= best_single + SINGLE_STREAM_TOLERANCE
        )
        no_harm &= passed
        details.append({
            "gate": "single_stream_no_harm", "item": f"{metric_name}:seed_{seed}",
            "actual": chronaris, "reference": best_single, "passed": passed,
        })
    recalls = pd.DataFrame(recall_rows)
    recall_gate = True
    for class_id, group in recalls.groupby("class_id"):
        mean = group.groupby("method")["recall"].mean()
        chronaris = float(mean["chronaris"])
        baseline = float(mean.drop("chronaris").max())
        passed = chronaris >= baseline - 0.05
        recall_gate &= passed
        details.append({
            "gate": "class_recall_gap_within_0_05", "item": f"class_{class_id}",
            "actual": chronaris, "reference": baseline, "passed": passed,
        })
    v2_mechanism = build_mechanism_mae_table(mechanism)
    v1_mechanism = build_mechanism_mae_table(v1_mechanism)
    time_gate = True
    for target in sorted(set(v2_mechanism["target"])):
        actual = float(v2_mechanism.query("target == @target and method == 'chronaris'")["value"].iloc[0])
        reference = float(v1_mechanism.query("target == @target and method == 'chronaris'")["value"].iloc[0])
        passed = actual <= reference * 1.10
        time_gate &= passed
        details.append({
            "gate": "time_mechanism_within_10_percent", "item": target,
            "actual": actual, "reference": reference, "passed": passed,
        })
    heatmap = build_stress_heatmap_table(slopes)
    missing_gate = True
    for factor in ("random_missing_rate", "contiguous_gap_s"):
        ranks = heatmap[factor].rank(method="min", ascending=False)
        rank = int(ranks["chronaris"])
        passed = rank <= 3
        missing_gate &= passed
        details.append({
            "gate": "missingness_slopes_top_three", "item": factor,
            "actual": rank, "reference": 3, "passed": passed,
        })
    selected = diagnostics[diagnostics["candidate_id"].eq(locked_candidate_id)]
    if len(selected) != 1:
        raise ValueError("locked candidate has no unique task-independent diagnostic")
    diagnostic = selected.iloc[0]
    causal = bool(diagnostic["causal_future_invariance_passed"])
    pooling = bool(diagnostic["invalid_query_pooling_passed"])
    details.extend((
        {"gate": "causal_future_invariance", "item": locked_candidate_id,
         "actual": causal, "reference": True, "passed": causal},
        {"gate": "invalid_queries_excluded_from_pooling", "item": locked_candidate_id,
         "actual": pooling, "reference": True, "passed": pooling},
    ))
    return details, {
        "single_stream_no_harm": bool(no_harm),
        "class_recall_gap_within_0_05": bool(recall_gate),
        "time_mechanism_within_10_percent": bool(time_gate),
        "missingness_slopes_top_three": bool(missing_gate),
        "causal_future_invariance": causal,
        "invalid_queries_excluded_from_pooling": pooling,
    }


def _require_completed_evidence(root):
    evidence = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "completed":
        raise ValueError(f"promotion audit requires completed locked evidence: {root}")
