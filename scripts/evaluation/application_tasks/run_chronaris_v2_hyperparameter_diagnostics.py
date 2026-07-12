#!/usr/bin/env python3
"""Diagnose and select the v2 hyperparameter top three."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_hyperparameter_diagnostics_run import (  # noqa: E402
    ChronarisV2HyperparameterDiagnosticsConfig,
    run_chronaris_v2_hyperparameter_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-hyperparameter-diagnostics-seed17",
    )
    parser.add_argument(
        "--hyperparameter-training-run-id",
        default="2026-07-12_chronaris-v2-hyperparameter-screen-seed17",
    )
    parser.add_argument(
        "--structure-diagnostics-run-id",
        default="2026-07-12_chronaris-v2-structure-diagnostics-seed17",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    root = run_chronaris_v2_hyperparameter_diagnostics(
        ChronarisV2HyperparameterDiagnosticsConfig(
            run_id=args.run_id,
            hyperparameter_training_run_id=args.hyperparameter_training_run_id,
            structure_diagnostics_run_id=args.structure_diagnostics_run_id,
            device=args.device,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
