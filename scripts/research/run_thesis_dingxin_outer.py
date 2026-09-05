#!/usr/bin/env python3
"""Run the frozen Dingxin grouped confirmation once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_dingxin_outer_run import (
    ThesisDingxinOuterConfig,
    run_thesis_dingxin_outer,
)


REPO = Path(__file__).resolve().parents[2]
SIMULATION_GATES = (
    REPO
    / "docs/artifacts/runs/2026-09-03_thesis-simulation-gates-v3p2p2/"
    "gate_audit.json"
)


def main():
    args = _parse_args()
    _require_clean_source()
    if not SIMULATION_GATES.is_file():
        raise RuntimeError("Dingxin confirmation requires simulation gate audit")
    gates = json.loads(SIMULATION_GATES.read_text(encoding="utf-8"))
    if gates.get("all_hard_gates_passed") is not True:
        raise RuntimeError("Dingxin confirmation blocked by simulation hard gate")
    state = run_thesis_dingxin_outer(
        ThesisDingxinOuterConfig(
            protocol_version=args.protocol_version,
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            seeds=tuple(args.seeds),
            device=args.device,
            resume=args.resume,
        )
    )
    print(
        json.dumps(
            {
                "completed_units": len(state["completed_units"]),
                "completed": state["completed"],
                "outer_results_opened": state["outer_results_opened"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if state["completed"] else 2


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
        raise RuntimeError("Dingxin confirmation requires committed source")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 43))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--protocol-version", choices=("v3.2.2",), default="v3.2.2"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
