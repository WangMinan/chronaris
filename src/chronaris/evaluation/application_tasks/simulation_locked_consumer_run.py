"""Locked G1-to-G2 downstream consumers for three seeds and six methods."""

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
from chronaris.evaluation.application_tasks.application_consumer_runtime import (
    ApplicationConsumerProtocol,
    run_application_method_consumers,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_audit import (
    build_paired_unit_statistic_rows,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    build_guarded_application_consumer_targets,
)
from chronaris.evaluation.application_tasks.application_consumers import (
    LinearConsumerConfig,
    MiniRocketConsumerConfig,
    TCNConsumerConfig,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
)
from chronaris.evaluation.application_tasks.simulation_locked_context_data import (
    load_simulation_locked_context_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import TRAINABLE_FUSION_METHODS
from chronaris.representation import load_fusion_stream_batch


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_locked_consumers")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationLockedConsumerConfig:
    run_id: str = "2026-07-12_simulation-locked-consumers"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-locked-pretraining"
    representation_run_id: str = "2026-07-12_simulation-locked-representations"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    minirocket_kernels: int = 10_000
    tcn_device: str = "auto"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationLockedConsumerResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_seed_count: int
    metric_count: int
    fusion_gain_count: int
    paired_statistic_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_locked_consumers(config: SimulationLockedConsumerConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    representation_root = Path(config.heavy_output_root) / config.representation_run_id
    representation_compact = Path(config.compact_output_root) / config.representation_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    representation_evidence = json.loads(
        (representation_compact / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    if representation_evidence.get("status") != "completed":
        raise ValueError("locked consumers require completed representation evidence")
    checkpoint_paths = _checkpoint_paths(
        pretraining_root,
        config.selected_candidates_path,
        config.seeds,
    )
    data = load_simulation_locked_context_data(config.simulation_root)
    targets = build_guarded_application_consumer_targets(
        data,
        completed_pretraining_checkpoints=tuple(
            checkpoint_paths[(config.seeds[0], method)]
            for method in TRAINABLE_FUSION_METHODS
        ),
        smoke_only=False,
    )
    tcn_device = _resolve_device(config.tcn_device)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_locked_consumers",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "checkpoint_count": len(checkpoint_paths),
            "task_oracle_opened_after_representation": True,
            "locked_metrics_opened": True,
        },
    ) as progress:
        result_rows = []
        metric_rows = []
        workload_rows = []
        unit_rows = []
        training_rows = []
        resource_rows = []
        for seed in config.seeds:
            protocol = _formal_protocol(
                seed=seed,
                minirocket_kernels=config.minirocket_kernels,
                tcn_device=tcn_device,
            )
            for method in APPLICATION_METHODS:
                outputs = _load_outputs(
                    representation_root,
                    seed=seed,
                    method=method,
                )
                result = run_application_method_consumers(
                    method_name=method,
                    outputs=outputs,
                    targets=targets,
                    output_root=heavy_root / "consumers" / f"seed_{seed}",
                    fold_id=f"simulation_g1_to_g2_clean_locked__seed_{seed}",
                    protocol=protocol,
                    resume=config.resume,
                )
                result_rows.append(
                    {
                        "seed": seed,
                        "method_name": method,
                        "status": result.status,
                        "protocol_sha256": result.protocol_sha256,
                        "linear_classification_c": result.model_manifest[
                            "linear_selected_classification_c"
                        ],
                        "linear_regression_alpha": result.model_manifest[
                            "linear_selected_regression_alpha"
                        ],
                        "minirocket_classification_c": result.model_manifest[
                            "minirocket_selected_classification_c"
                        ],
                        "minirocket_regression_alpha": result.model_manifest[
                            "minirocket_selected_regression_alpha"
                        ],
                        "hyperparameter_selection_role": result.model_manifest[
                            "hyperparameter_selection_role"
                        ],
                    }
                )
                metric_rows.extend(dict(row) for row in result.metric_rows)
                workload_rows.extend(
                    {"seed": seed, **dict(row)}
                    for row in result.workload_prediction_rows
                )
                unit_rows.extend(
                    {"seed": seed, **dict(row)} for row in result.unit_score_rows
                )
                training_rows.extend(
                    {"seed": seed, "method": method, **dict(row)}
                    for row in result.tcn_training_rows
                )
                resource_rows.extend(
                    {"seed": seed, **dict(row)} for row in result.resource_rows
                )
                progress.update(
                    "locked_method_consumer_complete",
                    seed=seed,
                    method_name=method,
                    status=result.status,
                )
        fusion_gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=("naive_time_sync", "mult", "contiformer", "chronaris"),
        )
        paired_rows = []
        for seed in config.seeds:
            paired_rows.extend(
                {"seed": seed, **row}
                for row in build_paired_unit_statistic_rows(
                    [row for row in unit_rows if row["seed"] == seed],
                    sample_manifest_rows=data.sample_manifest_rows,
                    seed=seed,
                    smoke_only=False,
                )
            )
        acceptance = _acceptance_rows(
            config=config,
            results=result_rows,
            metrics=metric_rows,
            paired=paired_rows,
            targets=targets,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            tcn_device=tcn_device,
            result_rows=result_rows,
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            paired_rows=paired_rows,
            training_rows=training_rows,
            resource_rows=resource_rows,
            workload_rows=workload_rows,
            targets=targets,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            method_seed_count=len(result_rows),
            metric_count=len(metric_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationLockedConsumerResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        method_seed_count=len(result_rows),
        metric_count=len(metric_rows),
        fusion_gain_count=len(fusion_gain_rows),
        paired_statistic_count=len(paired_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _formal_protocol(*, seed, minirocket_kernels, tcn_device):
    return ApplicationConsumerProtocol(
        linear=LinearConsumerConfig(random_state=seed, tune_on_validation=True),
        minirocket=MiniRocketConsumerConfig(
            n_kernels=minirocket_kernels,
            random_state=seed,
            tune_on_validation=True,
        ),
        tcn=TCNConsumerConfig(
            kernel_size=5,
            epochs=40,
            patience=6,
            seed=seed,
            device=tcn_device,
        ),
    )


def _checkpoint_paths(root, selected_path, seeds):
    selected = json.loads(Path(selected_path).read_text(encoding="utf-8"))
    paths = {}
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
                raise ValueError("locked consumer checkpoint is incomplete")
            paths[(seed, method)] = path
    return paths


def _load_outputs(root, *, seed, method):
    base = root / "representations" / f"seed_{seed}" / method
    return {
        role: load_fusion_stream_batch(base / role)
        for role in ("train", "validation", "held_out")
    }


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("locked consumer TCN device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("locked consumer requested unavailable CUDA device")
    return value


def _acceptance_rows(*, config, results, metrics, paired, targets):
    expected = len(config.seeds) * len(APPLICATION_METHODS)
    held_out = [row for row in metrics if row["role"] == "held_out"]
    return (
        _check("all_method_seed_consumers", len(results) == expected, len(results), expected),
        _check("all_consumers_complete", all(row["status"] in {"completed", "resumed"} for row in results), [row["status"] for row in results], "completed_or_resumed"),
        _check("validation_hyperparameter_selection", all(row["hyperparameter_selection_role"] == "validation" for row in results), {row["hyperparameter_selection_role"] for row in results}, {"validation"}),
        _check("formal_metric_scope", bool(metrics) and all(not row["smoke_only"] for row in metrics), len(metrics), ">0 formal rows"),
        _check("g2_metrics_available", bool(held_out) and all(row["role"] == "held_out" for row in held_out), len(held_out), ">0"),
        _check("paired_trajectory_statistics", bool(paired) and all(row["independent_unit_count"] == 48 for row in paired), len(paired), ">0 with 48 units"),
        _check("oracle_contract_limited", set(targets.manifest["allowed_oracle_fields"]) == {"true_time_s", "workload", "maneuver_state"}, targets.manifest["allowed_oracle_fields"], ["true_time_s", "workload", "maneuver_state"]),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "consumer_inventory.csv",
        "metrics": root / "metric_long.csv",
        "gains": root / "fusion_gain.csv",
        "paired": root / "paired_statistics.csv",
        "training": root / "tcn_training.csv",
        "resources": root / "resource_metrics.csv",
        "targets": root / "target_manifest.json",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "downstream_protocol.json",
        "report": root / "report.md",
        "claim": root / "claim_boundary.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    for key, rows in (
        ("results", values["result_rows"]),
        ("metrics", values["metric_rows"]),
        ("gains", values["fusion_gain_rows"]),
        ("paired", values["paired_rows"]),
        ("training", values["training_rows"]),
        ("resources", values["resource_rows"]),
        ("acceptance", values["acceptance"]),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(paths["targets"], values["targets"].manifest)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_locked_downstream_protocol.v1",
        "config": asdict(values["config"]),
        "consumer": asdict(_formal_protocol(seed=17, minirocket_kernels=values["config"].minirocket_kernels, tcn_device=values["tcn_device"])),
        "fit_role": "train",
        "hyperparameter_selection_role": "validation",
        "locked_evaluation_role": "held_out",
        "independent_statistical_unit": "trajectory",
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# G1 到 G2 锁定下游评估",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['result_rows'])} 个方法与随机种子组合、{len(values['metric_rows'])} 条任务指标。",
        "下游超参数只由 G1 validation 选择；G2 指标以 48 条潜在轨迹为配对统计单位。",
        "",
    )), encoding="utf-8")
    paths["claim"].write_text(
        "# 结论边界\n\n本 run 验证模型无关半物理仿真中的已知机动与负荷真值，不替代鼎新现场人工评估。\n",
        encoding="utf-8",
    )
    workload_path = values["heavy_root"] / "workload_predictions.csv"
    pd.DataFrame(values["workload_rows"]).to_csv(workload_path, index=False)
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_locked_consumers.py "
        f"--run-id {values['config'].run_id} --minirocket-kernels {values['config'].minirocket_kernels} --tcn-device {values['tcn_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_locked_consumer_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "method_seed_count": len(values["result_rows"]),
        "metric_count": len(values["metric_rows"]),
        "fusion_gain_count": len(values["fusion_gain_rows"]),
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
