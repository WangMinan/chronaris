"""Smoke audit packaging tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.simulation_audit import (
    SimulationAuditConfig,
    audit_existing_simulation,
    run_simulation_audit,
)


def test_simulation_smoke_audit_writes_compact_evidence(tmp_path) -> None:
    result = run_simulation_audit(
        SimulationAuditConfig(
            mode="smoke",
            heavy_run_id="smoke-heavy",
            compact_run_id="smoke-audit",
            heavy_output_root=str(tmp_path / "heavy"),
            compact_output_root=str(tmp_path / "compact"),
            resume=True,
        )
    )

    assert result.status == "completed"
    assert result.latent_sortie_count == 4
    assert result.observed_scenario_count == 8
    compact_root = Path(result.compact_run_root)
    for name in (
        "oracle_validation.csv",
        "paired_observation_audit.csv",
        "acceptance_checks.csv",
        "scenario_coverage.csv",
        "generator_family_split.json",
        "generator_config.json",
        "figure_manifest.json",
        "report.md",
        "claim_boundary.md",
        "resume_command.txt",
        "evidence_manifest.json",
        "progress.json",
        "run.log",
    ):
        assert (compact_root / name).exists(), name
    checks = pd.read_csv(compact_root / "acceptance_checks.csv")
    assert checks["passed"].all()
    figures = json.loads((compact_root / "figure_manifest.json").read_text(encoding="utf-8"))
    assert len(figures["figures"]) == 4
    assert all(Path(row["path"]).exists() for row in figures["figures"])

    rebuilt = audit_existing_simulation(
        SimulationAuditConfig(
            mode="smoke",
            heavy_run_id="smoke-heavy",
            compact_run_id="smoke-audit-rebuilt",
            heavy_output_root=str(tmp_path / "heavy"),
            compact_output_root=str(tmp_path / "compact"),
        )
    )
    assert rebuilt.status == "completed"
    assert rebuilt.acceptance_pass_count == rebuilt.acceptance_check_count
