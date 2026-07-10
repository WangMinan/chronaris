"""Generate, validate and package the synthetic aviation benchmark."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pandas as pd

from chronaris.evaluation.application_tasks.simulation_audit_checks import (
    build_simulation_acceptance_rows,
)
from chronaris.evaluation.application_tasks.simulation_audit_reporting import (
    write_simulation_audit_outputs,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream import (
    SimulationBenchmarkConfig,
    formal_split_specs,
    generate_benchmark,
    smoke_observation_scenarios,
    smoke_split_specs,
)
from chronaris.simulation.aviation_dual_stream.config import canonical_observation_scenarios


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_audit")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationAuditConfig:
    mode: str = "smoke"
    heavy_run_id: str = "2026-07-10_aviation-simulation-smoke"
    compact_run_id: str = "2026-07-10_aviation-simulation-smoke-audit"
    heavy_output_root: str = "artifacts/application_evaluation"
    compact_output_root: str = "docs/artifacts/runs"
    resume: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "formal"}:
            raise ValueError("mode must be smoke or formal")


@dataclass(frozen=True, slots=True)
class SimulationAuditResult:
    mode: str
    status: str
    heavy_run_root: str
    compact_run_root: str
    latent_sortie_count: int
    observed_scenario_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_audit(config: SimulationAuditConfig) -> SimulationAuditResult:
    compact_root = Path(config.compact_output_root) / config.compact_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.compact_run_id,
        stage_name="aviation_simulation_audit",
        logger=LOGGER,
        initial_progress={
            "mode": config.mode,
            "training_invoked": False,
            "confirmed_metrics_changed": False,
        },
    ) as progress:
        split_specs = smoke_split_specs() if config.mode == "smoke" else formal_split_specs()
        scenarios = (
            smoke_observation_scenarios()
            if config.mode == "smoke"
            else canonical_observation_scenarios()
        )
        benchmark = generate_benchmark(
            SimulationBenchmarkConfig(
                run_id=config.heavy_run_id,
                output_root=config.heavy_output_root,
                duration_s=60.0 if config.mode == "smoke" else 180.0,
                split_specs=split_specs,
                observation_scenarios=scenarios,
                resume=config.resume,
            ),
            progress_callback=lambda event, fields: progress.update(event, **dict(fields)),
        )
        validation = pd.DataFrame(benchmark.validation_rows)
        paired = pd.DataFrame(benchmark.paired_rows)
        acceptance_rows = build_simulation_acceptance_rows(
            mode=config.mode,
            validation=validation,
            paired=paired,
            split_identity=benchmark.split_identity,
            expected_latent_count=sum(split.latent_sortie_count for split in split_specs),
            expected_scenario_count=(
                sum(split.latent_sortie_count for split in split_specs) * len(scenarios)
            ),
        )
        status = "completed" if all(bool(row["passed"]) for row in acceptance_rows) else "partial"
        output_paths = write_simulation_audit_outputs(
            config=config,
            benchmark=benchmark,
            validation=validation,
            paired=paired,
            acceptance_rows=acceptance_rows,
            compact_root=compact_root,
            status=status,
        )
        pass_count = sum(bool(row["passed"]) for row in acceptance_rows)
        progress.finish(
            status=status,
            latent_sortie_count=benchmark.latent_sortie_count,
            observed_scenario_count=benchmark.observed_scenario_count,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            resumed_scenario_count=benchmark.resumed_scenario_count,
        )
        return SimulationAuditResult(
            mode=config.mode,
            status=status,
            heavy_run_root=benchmark.run_root,
            compact_run_root=str(compact_root),
            latent_sortie_count=benchmark.latent_sortie_count,
            observed_scenario_count=benchmark.observed_scenario_count,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=output_paths["report_path"],
            evidence_manifest_path=output_paths["evidence_manifest_path"],
        )


def audit_existing_simulation(config: SimulationAuditConfig) -> SimulationAuditResult:
    """Rebuild compact audit outputs from an existing ignored heavy run."""

    heavy_root = Path(config.heavy_output_root) / config.heavy_run_id
    manifest_path = heavy_root / "simulation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_rows = _read_jsonl(heavy_root / "validation_rows.jsonl")
    paired_rows = _read_jsonl(heavy_root / "paired_rows.jsonl")
    benchmark = SimpleNamespace(
        run_root=str(heavy_root),
        simulation_manifest_path=str(manifest_path),
        latent_sortie_count=int(manifest["latent_sortie_count"]),
        observed_scenario_count=int(manifest["observed_scenario_count"]),
        resumed_scenario_count=int(manifest["observed_scenario_count"]),
        split_identity=dict(manifest["split_identity"]),
    )
    compact_root = Path(config.compact_output_root) / config.compact_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.compact_run_id,
        stage_name="aviation_simulation_existing_audit",
        logger=LOGGER,
        initial_progress={
            "mode": config.mode,
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
        },
    ) as progress:
        validation = pd.DataFrame(validation_rows)
        paired = pd.DataFrame(paired_rows)
        acceptance_rows = build_simulation_acceptance_rows(
            mode=config.mode,
            validation=validation,
            paired=paired,
            split_identity=benchmark.split_identity,
            expected_latent_count=benchmark.latent_sortie_count,
            expected_scenario_count=benchmark.observed_scenario_count,
        )
        status = "completed" if all(bool(row["passed"]) for row in acceptance_rows) else "partial"
        output_paths = write_simulation_audit_outputs(
            config=config,
            benchmark=benchmark,
            validation=validation,
            paired=paired,
            acceptance_rows=acceptance_rows,
            compact_root=compact_root,
            status=status,
        )
        pass_count = sum(bool(row["passed"]) for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
        )
        return SimulationAuditResult(
            mode=config.mode,
            status=status,
            heavy_run_root=str(heavy_root),
            compact_run_root=str(compact_root),
            latent_sortie_count=benchmark.latent_sortie_count,
            observed_scenario_count=benchmark.observed_scenario_count,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=output_paths["report_path"],
            evidence_manifest_path=output_paths["evidence_manifest_path"],
        )


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
