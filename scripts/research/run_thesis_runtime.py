#!/usr/bin/env python3
"""Run the frozen Chronaris CUDA offline-replay benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_runtime import (
    run_thesis_runtime_benchmark,
)


REPO = Path(__file__).resolve().parents[2]
GATES = (
    REPO
    / "docs/artifacts/runs/2026-09-03_thesis-simulation-gates-v3p2p2/"
    "gate_audit.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--protocol-version", choices=("v3.2.2",), default="v3.2.2"
    )
    args = parser.parse_args()
    _require_ready_source()
    audit = run_thesis_runtime_benchmark(
        output_root=REPO
        / "docs/artifacts/runs/2026-09-03_thesis-runtime-v3p2p2",
        device=args.device,
    )
    print(json.dumps(audit["rows"], ensure_ascii=False, indent=2))


def _require_ready_source():
    if not GATES.is_file() or not json.loads(GATES.read_text(encoding="utf-8")).get(
        "all_hard_gates_passed"
    ):
        raise RuntimeError("runtime benchmark requires passed simulation gates")
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
        raise RuntimeError("runtime benchmark requires committed source")


if __name__ == "__main__":
    main()
