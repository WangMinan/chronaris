#!/usr/bin/env python3
"""Run task-independent diagnostics for the v2 structure screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_structure_diagnostics_run import (  # noqa: E402
    ChronarisV2StructureDiagnosticsConfig,
    run_chronaris_v2_structure_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-structure-diagnostics-seed17",
    )
    parser.add_argument(
        "--structure-training-run-id",
        default="2026-07-12_chronaris-v2-structure-screen-seed17",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = run_chronaris_v2_structure_diagnostics(
        ChronarisV2StructureDiagnosticsConfig(
            run_id=args.run_id,
            structure_training_run_id=args.structure_training_run_id,
            device=args.device,
            require_all_candidates=not args.allow_partial,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
