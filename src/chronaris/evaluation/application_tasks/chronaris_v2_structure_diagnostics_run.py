"""Diagnose and rank completed structure candidates without downstream labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_candidate_diagnostics import (
    diagnose_chronaris_candidate,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.evaluation.application_tasks.chronaris_v2_structure_figures import (
    render_structure_gate_figure,
)
from chronaris.modeling.training import rank_task_independent_candidates


@dataclass(frozen=True, slots=True)
class ChronarisV2StructureDiagnosticsConfig:
    run_id: str = "2026-07-12_chronaris-v2-structure-diagnostics-seed17"
    structure_training_run_id: str = (
        "2026-07-12_chronaris-v2-structure-screen-seed17"
    )
    compact_output_root: str = "docs/artifacts/runs"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    physiology_reference_checkpoint: str = (
        "artifacts/application_evaluation/"
        "2026-07-11_encoder-candidate-screen-seed17/"
        "checkpoints/physiology_only/A/best.pt"
    )
    vehicle_reference_checkpoint: str = (
        "artifacts/application_evaluation/"
        "2026-07-11_encoder-candidate-screen-seed17/"
        "checkpoints/vehicle_only/C/best.pt"
    )
    device: str = "cuda"
    require_all_candidates: bool = True


def run_chronaris_v2_structure_diagnostics(
    config: ChronarisV2StructureDiagnosticsConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2StructureDiagnosticsConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    training_root = (
        Path(resolved.compact_output_root) / resolved.structure_training_run_id
    )
    training_rows = pd.read_csv(training_root / "candidate_training.csv").to_dict(
        orient="records"
    )
    expected_count = 8 if resolved.require_all_candidates else len(training_rows)
    if len(training_rows) != expected_count:
        raise ValueError(
            f"structure diagnostics require {expected_count} trained/reference candidates"
        )
    if any(
        row["status"] not in {"immutable_reference", "completed", "resumed"}
        for row in training_rows
    ):
        raise ValueError("structure diagnostics found an incomplete candidate")
    data = load_simulation_locked_pretraining_data(resolved.simulation_root)
    common = {
        "physiology_reference_checkpoint": resolved.physiology_reference_checkpoint,
        "vehicle_reference_checkpoint": resolved.vehicle_reference_checkpoint,
        "batch": data.batch,
        "train_sample_ids": data.fold.train_sample_ids,
        "validation_sample_ids": data.fold.validation_sample_ids,
        "fold_id": data.fold.fold_id,
        "device": resolved.device,
    }
    reference_row = next(
        row for row in training_rows if row["candidate_id"] == "structure_01_v1"
    )
    v1_result = diagnose_chronaris_candidate(
        candidate_checkpoint=reference_row["checkpoint_path"],
        candidate_id="structure_01_v1",
        **common,
    )
    clock_baseline = v1_result.evidence.clock_offset_mae_s
    lag_baseline = v1_result.evidence.response_lag_mae_s
    results = [
        replace(
            v1_result,
            evidence=replace(
                v1_result.evidence,
                v1_clock_offset_mae_s=clock_baseline,
                v1_response_lag_mae_s=lag_baseline,
            ),
        )
    ]
    for row in training_rows:
        if row["candidate_id"] == "structure_01_v1":
            continue
        results.append(
            diagnose_chronaris_candidate(
                candidate_checkpoint=_final_training_checkpoint(
                    row["checkpoint_path"]
                ),
                candidate_id=str(row["candidate_id"]),
                v1_clock_offset_mae_s=clock_baseline,
                v1_response_lag_mae_s=lag_baseline,
                **common,
            )
        )
    evidence = tuple(result.evidence for result in results)
    ranking = rank_task_independent_candidates(evidence, top_k=3)
    pd.DataFrame(row.to_dict() for row in evidence).to_csv(
        compact_root / "candidate_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(ranking).to_csv(
        compact_root / "task_independent_ranking.csv",
        index=False,
    )
    figure_path = render_structure_gate_figure(
        compact_root / "task_independent_ranking.csv",
        compact_root / "structure_gate_overview.png",
    )
    _write_json(
        compact_root / "diagnostic_details.json",
        {
            result.evidence.candidate_id: result.details for result in results
        },
    )
    selected = [row for row in ranking if row["selected_for_inner_validation"]]
    acceptance = (
        _check("all_requested_candidates_diagnosed", len(results) == expected_count),
        _check(
            "forbidden_sources_closed",
            all(
                not result.details["task_labels_opened"]
                and not result.details["simulation_oracle_opened"]
                and not result.details["locked_test_opened"]
                for result in results
            ),
        ),
        _check("v1_time_baseline_available", clock_baseline >= 0 and lag_baseline >= 0),
        _check(
            "ranking_selects_gate_passed_only",
            all(row["gate_passed"] for row in selected),
        ),
    )
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    status = "ranked" if selected else "gates_failed"
    _write_json(
        compact_root / "protocol.json",
        {
            "format": "chronaris.v2_structure_diagnostics_protocol.v1",
            "config": asdict(resolved),
            "selection_inputs": [
                "public_self_supervised_validation_loss",
                "single_stream_fidelity",
                "representation_health",
                "synthetic_time_mechanism_recovery",
                "causal_and_pooling_invariants",
                "parameter_count",
            ],
            "downstream_labels_opened": False,
            "outer_test_opened": False,
            "sealed_confirmation_opened": False,
        },
    )
    (compact_root / "summary.md").write_text(
        "\n".join(
            (
                "# Chronaris v2 结构候选任务无关门禁",
                "",
                f"状态：{status}；诊断 {len(results)} 个候选，门内候选 {sum(row['gate_passed'] for row in ranking)} 个，前三复核候选 {len(selected)} 个。",
                "航电/生理保真比定义为：最佳单流在同一原始语义目标上的归一化恢复误差，除以候选融合表示的对应恢复误差；达到 0.98 表示融合后恢复能力不比单流低超过 2%。",
                "排序未读取机动分类、生理响应、outer-test 或封存仿真确认结果。",
                "若门内候选为空，不启动超参数筛选；需形成新的预注册开发轮次修复表示保真或时间机制。",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_structure_diagnostics_evidence.v1",
            "run_id": resolved.run_id,
            "status": status,
            "selected_candidate_ids": [row["candidate_id"] for row in selected],
            "figure_path": str(figure_path),
            "confirmed_metrics_changed": False,
        },
    )
    return compact_root


def _final_training_checkpoint(best_checkpoint_path):
    best = Path(best_checkpoint_path)
    last = best.with_name("last.pt")
    if not last.is_file():
        raise FileNotFoundError(f"final task-independent checkpoint is missing: {last}")
    return last


def _check(name: str, passed: bool):
    return {"check": name, "passed": bool(passed)}


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
