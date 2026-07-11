"""Generate and audit the frozen G2 single-factor stress extension."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream import (
    SimulationBenchmarkConfig,
    formal_split_specs,
    generate_benchmark,
    locked_stress_observation_scenarios,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_stress_generation")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationStressGenerationConfig:
    heavy_run_id: str = "2026-07-12_aviation-simulation-locked-stress"
    compact_run_id: str = "2026-07-12_aviation-simulation-locked-stress-audit"
    heavy_output_root: str = "artifacts/application_evaluation"
    compact_output_root: str = "docs/artifacts/runs"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationStressGenerationResult:
    status: str
    heavy_run_root: str
    compact_run_root: str
    latent_sortie_count: int
    observed_scenario_count: int
    resumed_scenario_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_stress_generation(config: SimulationStressGenerationConfig):
    compact_root = Path(config.compact_output_root) / config.compact_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    scenarios = locked_stress_observation_scenarios()
    locked_split = formal_split_specs()[-1]
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.compact_run_id,
        stage_name="simulation_locked_stress_generation",
        logger=LOGGER,
        initial_progress={
            "latent_sortie_count": locked_split.latent_sortie_count,
            "stress_scenario_count": len(scenarios),
            "method_names_accepted_by_generator": False,
        },
    ) as progress:
        result = generate_benchmark(
            SimulationBenchmarkConfig(
                run_id=config.heavy_run_id,
                output_root=config.heavy_output_root,
                split_specs=(locked_split,),
                observation_scenarios=scenarios,
                paired_observation_seed=True,
                resume=config.resume,
            ),
            progress_callback=lambda event, fields: progress.update(event, **fields),
        )
        acceptance = _acceptance_rows(result, scenarios)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            config=config,
            result=result,
            scenarios=scenarios,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            latent_sortie_count=result.latent_sortie_count,
            observed_scenario_count=result.observed_scenario_count,
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationStressGenerationResult(
        status=status,
        heavy_run_root=result.run_root,
        compact_run_root=str(compact_root),
        latent_sortie_count=result.latent_sortie_count,
        observed_scenario_count=result.observed_scenario_count,
        resumed_scenario_count=result.resumed_scenario_count,
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _acceptance_rows(result, scenarios):
    paired = result.paired_rows
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    source_root = Path("src/chronaris/simulation/aviation_dual_stream")
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in source_root.glob("*.py")
    )
    seeds_by_trajectory = {}
    for row in json.loads(
        Path(result.simulation_manifest_path).read_text(encoding="utf-8")
    )["scenario_rows"]:
        seeds_by_trajectory.setdefault(row["trajectory_id"], set()).add(
            int(row["observation_seed"])
        )
    return (
        _check("forty_eight_g2_sorties", result.latent_sortie_count == 48, result.latent_sortie_count, 48),
        _check("thirty_five_stress_scenarios", len(scenarios) == 35 and len(set(scenario_ids)) == 35, len(scenarios), 35),
        _check("all_paired_observations", result.observed_scenario_count == 48 * 35, result.observed_scenario_count, 1680),
        _check("g2_generator_family_only", all(row["split_id"] == "locked_test" for row in result.validation_rows), {row["split_id"] for row in result.validation_rows}, {"locked_test"}),
        _check("latent_truth_shared_per_trajectory", len(paired) == 48 and all(row["latent_hash_shared"] and row["trajectory_id_shared"] and row["scenario_ids_unique"] for row in paired), len(paired), 48),
        _check("observation_randomness_paired_per_trajectory", len(seeds_by_trajectory) == 48 and all(len(values) == 1 for values in seeds_by_trajectory.values()), max(map(len, seeds_by_trajectory.values())), 1),
        _check("method_independent_source", all(value not in source for value in ("contiformer", "candidate_id")), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "scenarios": root / "stress_scenarios.csv",
        "validation": root / "validation_summary.csv",
        "paired": root / "paired_summary.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame([scenario.to_dict() for scenario in values["scenarios"]]).to_csv(paths["scenarios"], index=False)
    pd.DataFrame(values["result"].validation_rows).to_csv(paths["validation"], index=False)
    pd.DataFrame(values["result"].paired_rows).to_csv(paths["paired"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_locked_stress_protocol.v1",
        "config": asdict(values["config"]),
        "scenario_count": len(values["scenarios"]),
        "scenarios": [scenario.to_dict() for scenario in values["scenarios"]],
        "independent_unit": "latent_trajectory",
        "method_names_accepted_by_generator": False,
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# G2 锁定压力场景生成与审计",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"48 条 G2 潜在轨迹各生成 {len(values['scenarios'])} 个成对观测版本，共 {values['result'].observed_scenario_count} 个场景。",
        "每条轨迹的状态、事件和负荷真值保持不变，只改变单因素观测条件或 mixed-severe 组合条件。",
        "",
    )), encoding="utf-8")
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/simulation/generate_aviation_locked_stress.py --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_locked_stress_evidence.v1",
        "run_id": values["config"].compact_run_id,
        "status": values["status"],
        "latent_sortie_count": values["result"].latent_sortie_count,
        "observed_scenario_count": values["result"].observed_scenario_count,
        "resumed_scenario_count": values["result"].resumed_scenario_count,
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": values["result"].run_root,
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
