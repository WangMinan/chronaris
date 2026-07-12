"""Formal G2 downstream comparison of full Chronaris and four mechanism ablations."""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

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
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_pretraining_run import (
    CHRONARIS_ABLATION_VARIANTS,
)
from chronaris.evaluation.application_tasks.simulation_locked_context_data import (
    load_simulation_locked_context_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_consumer_run import (
    _checkpoint_paths as _locked_checkpoint_paths,
)
from chronaris.evaluation.application_tasks.simulation_locked_representation_run import (
    _locked_v2_candidate_id,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import TRAINABLE_FUSION_METHODS
from chronaris.representation import load_fusion_stream_batch


LOGGER = logging.getLogger(
    "chronaris.pipelines.task_eval.simulation_chronaris_ablation_consumers"
)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationChronarisAblationConsumerConfig:
    run_id: str = "2026-07-12_simulation-chronaris-ablation-consumers"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-chronaris-ablation-pretraining"
    representation_run_id: str = "2026-07-12_simulation-chronaris-ablation-representations"
    full_pretraining_run_id: str = "2026-07-12_simulation-locked-pretraining"
    baseline_pretraining_run_id: str | None = None
    locked_configuration_path: str | None = None
    full_consumer_run_id: str = "2026-07-12_simulation-locked-consumers"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    variants: tuple[str, ...] = CHRONARIS_ABLATION_VARIANTS
    minirocket_kernels: int = 10_000
    tcn_device: str = "auto"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationChronarisAblationConsumerResult:
    run_id: str
    status: str
    variant_seed_count: int
    metric_count: int
    paired_statistic_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    compact_run_root: str
    heavy_run_root: str
    report_path: str
    evidence_manifest_path: str


def run_simulation_chronaris_ablation_consumers(
    config: SimulationChronarisAblationConsumerConfig,
) -> SimulationChronarisAblationConsumerResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    representation_root = Path(config.heavy_output_root) / config.representation_run_id
    representation_compact = Path(config.compact_output_root) / config.representation_run_id
    full_consumer_compact = Path(config.compact_output_root) / config.full_consumer_run_id
    full_consumer_heavy = Path(config.heavy_output_root) / config.full_consumer_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    _require_completed_evidence(representation_compact, "ablation representations")
    _require_completed_evidence(full_consumer_compact, "full locked consumers")
    checkpoint_paths = _full_checkpoint_paths(config)
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
        stage_name="simulation_chronaris_ablation_consumers",
        logger=LOGGER,
        initial_progress={
            "variants": list(config.variants),
            "seeds": list(config.seeds),
            "full_locked_consumer_verified": True,
            "task_oracle_opened_after_representation": True,
        },
    ) as progress:
        result_rows = []
        metric_rows = []
        unit_rows = []
        training_rows = []
        resource_rows = []
        workload_rows = []
        for seed in config.seeds:
            protocol = _formal_protocol(
                seed=seed,
                minirocket_kernels=config.minirocket_kernels,
                tcn_device=tcn_device,
            )
            for variant in config.variants:
                method_name = f"chronaris_{variant}"
                result = run_application_method_consumers(
                    method_name=method_name,
                    outputs=_load_outputs(
                        representation_root,
                        seed=seed,
                        method_name=method_name,
                    ),
                    targets=targets,
                    output_root=heavy_root / "consumers" / f"seed_{seed}",
                    fold_id=f"simulation_g1_to_g2_ablation__seed_{seed}",
                    protocol=protocol,
                    resume=config.resume,
                )
                result_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "method_name": method_name,
                        "status": result.status,
                        "protocol_sha256": result.protocol_sha256,
                        "hyperparameter_selection_role": result.model_manifest[
                            "hyperparameter_selection_role"
                        ],
                    }
                )
                metric_rows.extend(dict(row) for row in result.metric_rows)
                unit_rows.extend(
                    {"seed": seed, **dict(row)} for row in result.unit_score_rows
                )
                training_rows.extend(
                    {"seed": seed, "method": method_name, **dict(row)}
                    for row in result.tcn_training_rows
                )
                resource_rows.extend(
                    {"seed": seed, **dict(row)} for row in result.resource_rows
                )
                workload_rows.extend(
                    {"seed": seed, **dict(row)}
                    for row in result.workload_prediction_rows
                )
                progress.update(
                    "simulation_chronaris_ablation_consumer_complete",
                    seed=seed,
                    variant=variant,
                    status=result.status,
                )
        full_metrics = pd.read_csv(full_consumer_compact / "metric_long.csv")
        full_metrics = full_metrics[full_metrics["method"] == "chronaris"]
        full_units = pd.read_csv(full_consumer_heavy / "unit_score_rows.csv")
        full_units = full_units[full_units["method"] == "chronaris"]
        comparison_metrics = tuple(full_metrics.to_dict("records")) + tuple(metric_rows)
        comparison_units = tuple(full_units.to_dict("records")) + tuple(unit_rows)
        paired_rows = []
        for seed in config.seeds:
            paired_rows.extend(
                {"seed": seed, **row}
                for row in build_paired_unit_statistic_rows(
                    [row for row in comparison_units if int(row["seed"]) == seed],
                    sample_manifest_rows=data.sample_manifest_rows,
                    reference_method="chronaris",
                    seed=seed,
                    smoke_only=False,
                )
            )
        delta_rows = _build_full_ablation_metric_deltas(comparison_metrics)
        acceptance = _acceptance_rows(
            config=config,
            results=result_rows,
            metrics=metric_rows,
            paired=paired_rows,
            deltas=delta_rows,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            tcn_device=tcn_device,
            result_rows=result_rows,
            metric_rows=metric_rows,
            delta_rows=delta_rows,
            paired_rows=paired_rows,
            training_rows=training_rows,
            resource_rows=resource_rows,
            workload_rows=workload_rows,
            unit_rows=unit_rows,
            targets=targets,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            variant_seed_count=len(result_rows),
            metric_count=len(metric_rows),
            paired_statistic_count=len(paired_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationChronarisAblationConsumerResult(
        run_id=config.run_id,
        status=status,
        variant_seed_count=len(result_rows),
        metric_count=len(metric_rows),
        paired_statistic_count=len(paired_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _build_full_ablation_metric_deltas(rows):
    frame = pd.DataFrame(rows)
    selected = frame[frame["role"] == "held_out"].copy()
    keys = ["seed", "task", "consumer", "metric", "direction"]
    output = []
    for key, group in selected.groupby(keys, sort=True, dropna=False):
        full = group[group["method"] == "chronaris"]
        if len(full) != 1 or not _metric_row_available(full.iloc[0]):
            continue
        full_value = float(full.iloc[0]["value"])
        for _, row in group.iterrows():
            if row["method"] == "chronaris" or not _metric_row_available(row):
                continue
            ablation_value = float(row["value"])
            raw = full_value - ablation_value
            normalized = raw if key[-1] == "higher" else -raw
            output.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "full_method": "chronaris",
                    "ablation_method": row["method"],
                    "full_value": full_value,
                    "ablation_value": ablation_value,
                    "full_advantage_normalized": normalized,
                    "positive_favors_full": True,
                }
            )
    return tuple(output)


def _metric_row_available(row):
    if "status" in row.index and pd.notna(row["status"]):
        return str(row["status"]) == "available"
    if "available" in row.index and pd.notna(row["available"]):
        return bool(row["available"])
    raise ValueError("metric row must declare status or available")


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


def _full_checkpoint_paths(config):
    root = Path(config.heavy_output_root) / config.full_pretraining_run_id
    baseline_root = (
        Path(config.heavy_output_root)
        / (config.baseline_pretraining_run_id or config.full_pretraining_run_id)
    )
    return _locked_checkpoint_paths(
        root,
        config.selected_candidates_path,
        config.seeds,
        baseline_root=baseline_root,
        locked_candidate_id=_locked_v2_candidate_id(
            config.locked_configuration_path
        ),
    )


def _load_outputs(root, *, seed, method_name):
    base = root / "representations" / f"seed_{seed}" / method_name
    return {
        role: load_fusion_stream_batch(base / role)
        for role in ("train", "validation", "held_out")
    }


def _require_completed_evidence(root, label):
    evidence = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "completed":
        raise ValueError(f"{label} evidence is incomplete")


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("ablation TCN device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("ablation consumers requested unavailable CUDA")
    return value


def _acceptance_rows(*, config, results, metrics, paired, deltas):
    expected = len(config.seeds) * len(config.variants)
    return (
        _check("all_variant_seed_consumers", len(results) == expected, len(results), expected),
        _check("all_consumers_complete", all(row["status"] in {"completed", "resumed"} for row in results), [row["status"] for row in results], "completed_or_resumed"),
        _check("validation_hyperparameter_selection", all(row["hyperparameter_selection_role"] == "validation" for row in results), True, True),
        _check("formal_metric_scope", bool(metrics) and all(not row["smoke_only"] for row in metrics), len(metrics), ">0 formal rows"),
        _check("full_ablation_metric_deltas", bool(deltas), len(deltas), ">0"),
        _check("paired_trajectory_statistics", bool(paired) and all(row["independent_unit_count"] == 48 for row in paired), len(paired), ">0 with 48 units"),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "consumer_inventory.csv",
        "metrics": root / "metric_long.csv",
        "deltas": root / "full_ablation_metric_delta.csv",
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
        ("deltas", values["delta_rows"]),
        ("paired", values["paired_rows"]),
        ("training", values["training_rows"]),
        ("resources", values["resource_rows"]),
        ("acceptance", values["acceptance"]),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    _write_json(paths["targets"], values["targets"].manifest)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_chronaris_ablation_downstream.v1",
        "config": asdict(values["config"]),
        "fit_role": "train",
        "hyperparameter_selection_role": "validation",
        "locked_evaluation_role": "held_out",
        "reference_method": "chronaris",
        "independent_statistical_unit": "trajectory",
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# Chronaris 机制消融锁定下游评估",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['result_rows'])} 个消融 consumer；与完整模型按 48 条 G2 轨迹配对比较。",
        "",
    )), encoding="utf-8")
    paths["claim"].write_text(
        "# 结论边界\n\n消融差异用于解释仿真条件下的机制贡献，不替代鼎新现场人工评估。\n",
        encoding="utf-8",
    )
    workload_path = values["heavy_root"] / "workload_predictions.csv"
    unit_path = values["heavy_root"] / "unit_score_rows.csv"
    pd.DataFrame(values["workload_rows"]).to_csv(workload_path, index=False)
    pd.DataFrame(values["unit_rows"]).to_csv(unit_path, index=False)
    paths["resume"].write_text(
        _resume_command(values["config"], tcn_device=values["tcn_device"]),
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_chronaris_ablation_consumer_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "variant_seed_count": len(values["result_rows"]),
        "metric_count": len(values["metric_rows"]),
        "paired_statistic_count": len(values["paired_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": str(values["heavy_root"]),
        "workload_prediction_path": str(workload_path),
        "unit_score_path": str(unit_path),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _resume_command(config, *, tcn_device):
    args = [
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python",
        "scripts/evaluation/application_tasks/run_simulation_chronaris_ablation_consumers.py",
        "--run-id", config.run_id,
        "--pretraining-run-id", config.pretraining_run_id,
        "--representation-run-id", config.representation_run_id,
        "--full-pretraining-run-id", config.full_pretraining_run_id,
        "--full-consumer-run-id", config.full_consumer_run_id,
        "--minirocket-kernels", str(config.minirocket_kernels),
        "--tcn-device", tcn_device,
        "--resume",
    ]
    for seed in config.seeds:
        args.extend(("--seed", str(seed)))
    for variant in config.variants:
        args.extend(("--variant", variant))
    for flag, value in (
        ("--baseline-pretraining-run-id", config.baseline_pretraining_run_id),
        ("--locked-configuration-path", config.locked_configuration_path),
    ):
        if value is not None:
            args.extend((flag, value))
    return " ".join(shlex.quote(str(value)) for value in args) + "\n"


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
