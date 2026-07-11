"""Locked representation probes for clock-offset and response-lag recovery."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from chronaris.evaluation.application_tasks.application_consumer_smoke_audit import (
    build_paired_unit_statistic_rows,
)
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_representation_run import (
    MECHANISM_METHODS,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_targets import (
    CLOCK_OFFSET_TARGET,
    RESPONSE_LAG_TARGET,
    build_simulation_mechanism_targets,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream import (
    canonical_observation_scenarios,
    locked_stress_observation_scenarios,
)


LOGGER = logging.getLogger(
    "chronaris.pipelines.task_eval.simulation_mechanism_consumers"
)
LOGGER.addHandler(logging.NullHandler())
TARGETS = (CLOCK_OFFSET_TARGET, RESPONSE_LAG_TARGET)
TARGET_TOLERANCE_S = {CLOCK_OFFSET_TARGET: 0.5, RESPONSE_LAG_TARGET: 2.0}


@dataclass(frozen=True, slots=True)
class SimulationMechanismConsumerConfig:
    run_id: str = "2026-07-12_simulation-mechanism-consumers"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    mechanism_representation_run_id: str = (
        "2026-07-12_simulation-mechanism-representations"
    )
    stress_representation_run_id: str = (
        "2026-07-12_simulation-locked-stress-representations"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationMechanismConsumerResult:
    run_id: str
    status: str
    model_count: int
    evaluation_count: int
    metric_count: int
    paired_statistic_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    compact_run_root: str
    heavy_run_root: str
    report_path: str
    evidence_manifest_path: str


def run_simulation_mechanism_consumers(
    config: SimulationMechanismConsumerConfig,
) -> SimulationMechanismConsumerResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    mechanism_root = (
        Path(config.heavy_output_root) / config.mechanism_representation_run_id
    )
    stress_root = Path(config.heavy_output_root) / config.stress_representation_run_id
    mechanism_compact = (
        Path(config.compact_output_root) / config.mechanism_representation_run_id
    )
    stress_compact = (
        Path(config.compact_output_root) / config.stress_representation_run_id
    )
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    _require_completed_evidence(mechanism_compact)
    _require_completed_evidence(stress_compact)
    g1_manifest = pd.read_csv(mechanism_compact / "data_manifest.csv").to_dict(
        "records"
    )
    stress_manifest = pd.read_csv(stress_compact / "data_manifest.csv").to_dict(
        "records"
    )
    g1_targets = build_simulation_mechanism_targets(
        g1_manifest,
        scenarios=canonical_observation_scenarios(),
        representation_evidence_completed=True,
    )
    stress_targets = build_simulation_mechanism_targets(
        stress_manifest,
        scenarios=locked_stress_observation_scenarios(),
        representation_evidence_completed=True,
    )
    g1_scenarios = tuple(
        value.scenario_id for value in canonical_observation_scenarios()
    )
    stress_scenarios = tuple(
        value.scenario_id for value in locked_stress_observation_scenarios()
    )
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_mechanism_consumers",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "methods": list(MECHANISM_METHODS),
            "oracle_opened_after_representations": True,
            "g2_used_for_model_selection": False,
        },
    ) as progress:
        model_rows = []
        evaluation_rows = []
        metric_rows = []
        prediction_rows = []
        unit_rows = []
        for seed in config.seeds:
            for method in MECHANISM_METHODS:
                train = _load_g1_role(
                    mechanism_root,
                    seed=seed,
                    method=method,
                    role="train",
                    scenarios=g1_scenarios,
                )
                validation = _load_g1_role(
                    mechanism_root,
                    seed=seed,
                    method=method,
                    role="validation",
                    scenarios=g1_scenarios,
                )
                models = {}
                for target_name in TARGETS:
                    model, selected_alpha, model_status, protocol_sha256 = _fit_or_load(
                        output_root=heavy_root / "models" / f"seed_{seed}" / method,
                        method=method,
                        seed=seed,
                        target_name=target_name,
                        train=train,
                        validation=validation,
                        targets=g1_targets,
                        alpha_grid=config.alpha_grid,
                        resume=config.resume,
                    )
                    models[target_name] = model
                    model_rows.append(
                        {
                            "seed": seed,
                            "method_name": method,
                            "target_name": target_name,
                            "status": model_status,
                            "selected_alpha": selected_alpha,
                            "protocol_sha256": protocol_sha256,
                            "fit_role": "g1_train",
                            "selection_role": "g1_validation",
                            "g2_used_for_model_selection": False,
                        }
                    )
                for scenario_id in stress_scenarios:
                    output = load_fusion_stream_batch(
                        stress_root
                        / "representations"
                        / f"seed_{seed}"
                        / method
                        / scenario_id
                    )
                    values = output.pooled_embedding.detach().cpu().numpy()
                    for target_name, model in models.items():
                        truth = stress_targets.values(output.sample_ids, target_name)
                        prediction = np.asarray(model.predict(values), dtype=np.float64)
                        rows, units = _evaluate_predictions(
                            seed=seed,
                            method=method,
                            scenario_id=scenario_id,
                            target_name=target_name,
                            sample_ids=output.sample_ids,
                            truth=truth,
                            prediction=prediction,
                            targets=stress_targets,
                        )
                        metric_rows.extend(rows)
                        unit_rows.extend(units)
                        prediction_rows.extend(
                            {
                                "seed": seed,
                                "method": method,
                                "scenario_id": scenario_id,
                                "target": target_name,
                                "sample_id": sample_id,
                                "trajectory_id": trajectory_id,
                                "truth": float(truth[index]),
                                "prediction": float(prediction[index]),
                            }
                            for index, (sample_id, trajectory_id) in enumerate(
                                zip(
                                    output.sample_ids,
                                    stress_targets.trajectory_ids(output.sample_ids),
                                    strict=True,
                                )
                            )
                        )
                        evaluation_rows.append(
                            {
                                "seed": seed,
                                "method_name": method,
                                "scenario_id": scenario_id,
                                "target_name": target_name,
                                "context_count": len(output.sample_ids),
                                "trajectory_count": len(
                                    set(
                                        stress_targets.trajectory_ids(
                                            output.sample_ids
                                        )
                                    )
                                ),
                            }
                        )
                progress.update(
                    "simulation_mechanism_method_complete",
                    seed=seed,
                    method_name=method,
                )
        paired_rows = []
        for seed in config.seeds:
            for scenario_id in stress_scenarios:
                paired_rows.extend(
                    {
                        "seed": seed,
                        "scenario_id": scenario_id,
                        **row,
                    }
                    for row in build_paired_unit_statistic_rows(
                        [
                            row
                            for row in unit_rows
                            if row["seed"] == seed
                            and row["scenario_id"] == scenario_id
                        ],
                        sample_manifest_rows=stress_manifest,
                        reference_method="chronaris",
                        seed=seed,
                        smoke_only=False,
                    )
                )
        acceptance = _acceptance_rows(
            config=config,
            models=model_rows,
            evaluations=evaluation_rows,
            metrics=metric_rows,
            paired=paired_rows,
            stress_scenarios=stress_scenarios,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            model_rows=model_rows,
            evaluation_rows=evaluation_rows,
            metric_rows=metric_rows,
            paired_rows=paired_rows,
            prediction_rows=prediction_rows,
            g1_target_manifest=g1_targets.manifest,
            stress_target_manifest=stress_targets.manifest,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            model_count=len(model_rows),
            evaluation_count=len(evaluation_rows),
            metric_count=len(metric_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationMechanismConsumerResult(
        run_id=config.run_id,
        status=status,
        model_count=len(model_rows),
        evaluation_count=len(evaluation_rows),
        metric_count=len(metric_rows),
        paired_statistic_count=len(paired_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _load_g1_role(root, *, seed, method, role, scenarios):
    sample_ids = []
    pooled = []
    hashes = []
    for scenario_id in scenarios:
        output = load_fusion_stream_batch(
            root
            / "representations"
            / f"seed_{seed}"
            / method
            / role
            / scenario_id
        )
        sample_ids.extend(output.sample_ids)
        pooled.append(output.pooled_embedding.detach().cpu().numpy())
        hashes.append(output.checkpoint_sha256)
    if len(set(hashes)) != 1:
        raise ValueError("mechanism role representation checkpoint changed")
    return {
        "sample_ids": tuple(sample_ids),
        "pooled": np.concatenate(pooled, axis=0),
        "checkpoint_sha256": hashes[0],
    }


def _fit_or_load(
    *, output_root, method, seed, target_name, train, validation, targets,
    alpha_grid, resume
):
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / f"{target_name}.joblib"
    manifest_path = output_root / f"{target_name}.json"
    protocol = {
        "format": "chronaris.simulation_mechanism_probe.v1",
        "method": method,
        "seed": seed,
        "target": target_name,
        "alpha_grid": list(alpha_grid),
        "train_sample_ids": list(train["sample_ids"]),
        "validation_sample_ids": list(validation["sample_ids"]),
        "checkpoint_sha256": train["checkpoint_sha256"],
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    if resume and model_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_sha256") == protocol_sha256:
            return (
                joblib.load(model_path),
                float(manifest["selected_alpha"]),
                "resumed",
                protocol_sha256,
            )
    model, selected_alpha = fit_regressor(
        train["pooled"],
        targets.values(train["sample_ids"], target_name),
        validation["pooled"],
        targets.values(validation["sample_ids"], target_name),
        alpha_values=alpha_grid,
        scaler_with_mean=True,
    )
    temporary = model_path.with_suffix(".tmp")
    joblib.dump(model, temporary)
    temporary.replace(model_path)
    manifest_path.write_text(
        json.dumps(
            {
                **protocol,
                "protocol_sha256": protocol_sha256,
                "selected_alpha": selected_alpha,
                "fit_role": "g1_train",
                "selection_role": "g1_validation",
                "g2_used_for_model_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return model, selected_alpha, "completed", protocol_sha256


def _evaluate_predictions(
    *, seed, method, scenario_id, target_name, sample_ids, truth, prediction,
    targets
):
    trajectory_ids = targets.trajectory_ids(sample_ids)
    frame = pd.DataFrame(
        {"trajectory_id": trajectory_ids, "truth": truth, "prediction": prediction}
    ).groupby("trajectory_id", sort=True).mean()
    error = frame["prediction"].to_numpy() - frame["truth"].to_numpy()
    tolerance = TARGET_TOLERANCE_S[target_name]
    correlation = spearmanr(frame["truth"], frame["prediction"]).statistic
    metrics = (
        ("mae_s", float(np.mean(np.abs(error))), "lower"),
        ("rmse_s", float(np.sqrt(np.mean(error ** 2))), "lower"),
        ("tolerance_hit_rate", float(np.mean(np.abs(error) <= tolerance)), "higher"),
        (
            "spearman",
            None if not np.isfinite(correlation) else float(correlation),
            "higher",
        ),
    )
    metric_rows = [
        {
            "seed": seed,
            "method": method,
            "scenario_id": scenario_id,
            "task": "sim_clock_lag_recovery_v1",
            "target": target_name,
            "consumer": "ridge_recovery_probe",
            "metric": name,
            "value": value,
            "direction": direction,
            "available": value is not None,
            "trajectory_count": len(frame),
            "tolerance_s": tolerance,
            "smoke_only": False,
        }
        for name, value, direction in metrics
    ]
    unit_rows = [
        {
            "seed": seed,
            "scenario_id": scenario_id,
            "method": method,
            "consumer": f"ridge_recovery_probe__{target_name}",
            "metric": "absolute_error_s",
            "sample_id": sample_id,
            "role": "held_out",
            "value": float(abs(prediction[index] - truth[index])),
            "direction": "lower",
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    return metric_rows, unit_rows


def _require_completed_evidence(root):
    payload = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"mechanism consumer requires completed evidence: {root}")


def _acceptance_rows(*, config, models, evaluations, metrics, paired, stress_scenarios):
    expected_models = len(config.seeds) * len(MECHANISM_METHODS) * len(TARGETS)
    expected_evaluations = expected_models * len(stress_scenarios)
    return (
        _check("all_seed_method_target_models", len(models) == expected_models, len(models), expected_models),
        _check("g1_validation_selection_only", all(not row["g2_used_for_model_selection"] and row["selection_role"] == "g1_validation" for row in models), True, True),
        _check("all_stress_evaluations", len(evaluations) == expected_evaluations, len(evaluations), expected_evaluations),
        _check("forty_eight_trajectory_units", all(row["trajectory_count"] == 48 for row in evaluations), sorted({row["trajectory_count"] for row in evaluations}), 48),
        _check("formal_metrics_available", bool(metrics) and any(row["available"] for row in metrics), len(metrics), ">0"),
        _check("paired_trajectory_statistics", bool(paired) and all(row["independent_unit_count"] == 48 for row in paired), len(paired), ">0 with 48 units"),
    )


def _write_outputs(**values):
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
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_mechanism_consumers.py "
        f"--run-id {values['config'].run_id} --resume\n",
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


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
