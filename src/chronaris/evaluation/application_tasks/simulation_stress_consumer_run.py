"""Apply clean-fitted consumers to every frozen G2 stress representation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
)
from chronaris.evaluation.application_tasks.application_frozen_evaluation import (
    evaluate_frozen_application_consumers,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_audit import (
    build_paired_unit_statistic_rows,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    build_guarded_application_consumer_targets,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.evaluation.application_tasks.simulation_stress_context_data import (
    load_simulation_stress_context_data,
)
from chronaris.evaluation.application_tasks.simulation_stress_metrics import (
    build_stress_slope_rows,
    stress_scenario_metadata,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import TRAINABLE_FUSION_METHODS
from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream import (
    locked_stress_observation_scenarios,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_stress_consumers")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationStressConsumerConfig:
    run_id: str = "2026-07-12_simulation-locked-stress-consumers"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-locked-pretraining"
    clean_consumer_run_id: str = "2026-07-12_simulation-locked-consumers"
    stress_representation_run_id: str = (
        "2026-07-12_simulation-locked-stress-representations"
    )
    stress_generation_run_id: str = "2026-07-12_aviation-simulation-locked-stress"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationStressConsumerResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    evaluation_count: int
    metric_count: int
    slope_count: int
    paired_statistic_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_stress_consumers(config: SimulationStressConsumerConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    clean_consumer_root = Path(config.heavy_output_root) / config.clean_consumer_run_id
    stress_representation_root = (
        Path(config.heavy_output_root) / config.stress_representation_run_id
    )
    stress_root = Path(config.heavy_output_root) / config.stress_generation_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    _require_completed_evidence(
        Path(config.compact_output_root) / config.clean_consumer_run_id
    )
    _require_completed_evidence(
        Path(config.compact_output_root) / config.stress_representation_run_id
    )
    checkpoint_paths = _checkpoint_paths(
        pretraining_root,
        config.selected_candidates_path,
        config.seeds,
    )
    clean_targets = json.loads(
        (
            Path(config.compact_output_root)
            / config.clean_consumer_run_id
            / "target_manifest.json"
        ).read_text(encoding="utf-8")
    )
    frozen_thresholds = tuple(clean_targets["workload_thresholds_train_only"])
    scenario_ids = tuple(
        scenario.scenario_id for scenario in locked_stress_observation_scenarios()
    )
    metadata = stress_scenario_metadata()
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_locked_stress_consumers",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "scenario_count": len(scenario_ids),
            "consumer_refit_allowed": False,
            "frozen_thresholds": list(frozen_thresholds),
        },
    ) as progress:
        evaluation_rows = []
        metric_rows = []
        unit_rows = []
        workload_rows = []
        paired_rows = []
        for scenario_id in scenario_ids:
            data = load_simulation_stress_context_data(
                stress_root,
                scenario_id=scenario_id,
            )
            targets = build_guarded_application_consumer_targets(
                data,
                completed_pretraining_checkpoints=tuple(
                    checkpoint_paths[(config.seeds[0], method)]
                    for method in TRAINABLE_FUSION_METHODS
                ),
                smoke_only=False,
                workload_thresholds=frozen_thresholds,
            )
            for seed in config.seeds:
                scenario_unit_rows = []
                for method in APPLICATION_METHODS:
                    representation_root = (
                        stress_representation_root
                        / "representations"
                        / f"seed_{seed}"
                        / method
                        / scenario_id
                    )
                    output = load_fusion_stream_batch(representation_root)
                    observation_available = dict(zip(
                        output.sample_ids, output.valid_mask.any(dim=1).tolist(), strict=True
                    ))
                    result = evaluate_frozen_application_consumers(
                        method_name=method,
                        output=output,
                        targets=targets,
                        model_root=(
                            clean_consumer_root / "consumers" / f"seed_{seed}"
                        ),
                        output_root=heavy_root / "predictions" / f"seed_{seed}",
                        fold_id=(
                            f"simulation_g1_to_g2_stress__seed_{seed}__{scenario_id}"
                        ),
                        evaluation_id=scenario_id,
                        seed=seed,
                    )
                    tagged_metrics = [
                        {
                            "scenario_id": scenario_id,
                            **metadata[scenario_id],
                            **dict(row),
                        }
                        for row in result.metric_rows
                    ]
                    tagged_units = [
                        {"seed": seed, "scenario_id": scenario_id, **dict(row)}
                        for row in result.unit_score_rows
                    ]
                    metric_rows.extend(tagged_metrics)
                    unit_rows.extend(tagged_units)
                    scenario_unit_rows.extend(tagged_units)
                    workload_rows.extend(
                        {"seed": seed, "scenario_id": scenario_id, **dict(row),
                         "observation_available": observation_available[row["sample_id"]]}
                        for row in result.workload_prediction_rows
                    )
                    evaluation_rows.append(
                        {
                            "seed": seed,
                            "scenario_id": scenario_id,
                            "method_name": method,
                            "metric_count": len(result.metric_rows),
                            "consumer_refit": False,
                            "source_model_protocol_sha256": result.source_model_protocol_sha256,
                            "prediction_sha256": result.prediction_sha256,
                        }
                    )
                paired_subset = [
                    row
                    for row in scenario_unit_rows
                    if row["method"] in {"chronaris", "mult", "contiformer"}
                    and (
                        (row["consumer"] == "minirocket" and row["metric"] in {"classification_correct", "regression_absolute_error"})
                        or (row["consumer"] == "causal_tcn_duration" and row["metric"] == "frame_accuracy")
                    )
                ]
                paired_rows.extend(
                    {"seed": seed, "scenario_id": scenario_id, **row}
                    for row in build_paired_unit_statistic_rows(
                        paired_subset,
                        sample_manifest_rows=data.sample_manifest_rows,
                        seed=seed,
                        smoke_only=False,
                    )
                )
                progress.update(
                    "stress_scenario_consumer_complete",
                    seed=seed,
                    scenario_id=scenario_id,
                )
        fusion_gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=("naive_time_sync", "mult", "contiformer", "chronaris"),
        )
        slope_rows = build_stress_slope_rows(metric_rows)
        acceptance = _acceptance_rows(
            config=config,
            evaluations=evaluation_rows,
            metrics=metric_rows,
            slopes=slope_rows,
            paired=paired_rows,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            frozen_thresholds=frozen_thresholds,
            evaluation_rows=evaluation_rows,
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            slope_rows=slope_rows,
            paired_rows=paired_rows,
            workload_rows=workload_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            evaluation_count=len(evaluation_rows),
            metric_count=len(metric_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationStressConsumerResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        evaluation_count=len(evaluation_rows),
        metric_count=len(metric_rows),
        slope_count=len(slope_rows),
        paired_statistic_count=len(paired_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _checkpoint_paths(root, selected_path, seeds):
    selected = json.loads(Path(selected_path).read_text(encoding="utf-8"))
    result = {}
    for seed in seeds:
        for method in TRAINABLE_FUSION_METHODS:
            candidate = str(selected[method]["candidate_id"])
            path = (
                root / "checkpoints" / f"seed_{seed}" / method / "best.pt"
                if method == "chronaris"
                else root / "checkpoints" / f"seed_{seed}" / method / candidate / "best.pt"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("training_status") != "completed":
                raise ValueError("stress consumer checkpoint is incomplete")
            result[(seed, method)] = path
    return result


def _require_completed_evidence(root):
    payload = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"stress consumers require completed evidence: {root}")


def _acceptance_rows(*, config, evaluations, metrics, slopes, paired):
    expected = len(config.seeds) * 35 * 6
    return (
        _check("all_frozen_evaluations", len(evaluations) == expected, len(evaluations), expected),
        _check("no_consumer_refit", all(not row["consumer_refit"] for row in evaluations), False, False),
        _check("formal_metric_scope", bool(metrics) and all(not row["smoke_only"] for row in metrics), len(metrics), ">0 formal rows"),
        _check("all_stress_factors_have_slopes", {row["stress_factor"] for row in slopes} == {"timestamp_jitter_ms", "absolute_clock_offset_s", "absolute_clock_drift_ppm", "random_missing_rate", "contiguous_gap_s", "additional_physiology_lag_s", "snr_degradation_db"}, sorted({row["stress_factor"] for row in slopes}), "seven factors"),
        _check("mixed_severe_evaluated", any(row["scenario_id"] == "mixed_severe" for row in metrics), True, True),
        _check("paired_trajectory_unit", bool(paired) and all(row["independent_unit_count"] == 48 for row in paired), len(paired), ">0 with 48 units"),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "evaluations": root / "evaluation_inventory.csv",
        "metrics": root / "metric_long.csv",
        "gains": root / "fusion_gain.csv",
        "slopes": root / "stress_slopes.csv",
        "paired": root / "paired_statistics.csv",
        "unobserved": values["heavy_root"] / "unobserved_predictions.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "claim": root / "claim_boundary.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    for key, rows in (
        ("evaluations", values["evaluation_rows"]),
        ("metrics", values["metric_rows"]),
        ("gains", values["fusion_gain_rows"]),
        ("slopes", values["slope_rows"]),
        ("paired", values["paired_rows"]),
        ("acceptance", values["acceptance"]),
        ("unobserved", [row for row in values["workload_rows"] if not row["observation_available"]]),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_stress_consumer_protocol.v1",
        "config": asdict(values["config"]),
        "frozen_workload_thresholds": list(values["frozen_thresholds"]),
        "consumer_refit_allowed": False,
        "paired_bootstrap_repetitions": 2000,
        "paired_permutation_repetitions": 10000,
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# 受控仿真压力场景下游评估",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['evaluation_rows'])} 个冻结 consumer 评估、{len(values['metric_rows'])} 条指标和 {len(values['slope_rows'])} 条退化斜率。",
        "所有 consumer 与负荷阈值均冻结自 G1 clean train/validation，压力场景不重训、不调参。",
        "无观测样本仍进入同一冻结消费者和总体指标；其预测单列保存，供解释无信息输入下的输出，不作为有观测恢复能力的证据。",
        "",
    )), encoding="utf-8")
    paths["claim"].write_text(
        "# 结论边界\n\n压力曲线验证半物理生成机制下的观测鲁棒性，不等同于新增鼎新现场样本。\n",
        encoding="utf-8",
    )
    workload_path = values["heavy_root"] / "workload_predictions.csv"
    pd.DataFrame(values["workload_rows"]).to_csv(workload_path, index=False)
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_stress_consumers.py "
        f"--run-id {values['config'].run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_stress_consumer_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "evaluation_count": len(values["evaluation_rows"]),
        "metric_count": len(values["metric_rows"]),
        "stress_slope_count": len(values["slope_rows"]),
        "paired_statistic_count": len(values["paired_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": str(values["heavy_root"]),
        "workload_prediction_path": str(workload_path),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
