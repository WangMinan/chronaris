#!/usr/bin/env python3
"""Audit all frozen simulation hard gates before public outer evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_simulation_gates import (
    run_simulation_gate_audit,
)


REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--protocol-version", choices=("v3.2.3",), default="v3.2.3"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    _require_clean_source()
    audit = run_simulation_gate_audit(
        compact_root=(
            REPO
            / "docs/artifacts/runs/2026-09-04_thesis-simulation-gates-v3p2p3"
        ),
        heavy_root=(
            REPO
            / "artifacts/application_evaluation/"
            "2026-09-04_thesis-simulation-gates-v3p2p3"
        ),
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(audit["gates"], ensure_ascii=False, indent=2))
    return 0 if audit["all_hard_gates_passed"] else 2


def _require_clean_source():
    status = subprocess.check_output(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
        ),
        cwd=REPO,
        text=True,
    )
    if status:
        raise RuntimeError("simulation gate audit requires committed source")


if __name__ == "__main__":
    raise SystemExit(main())
