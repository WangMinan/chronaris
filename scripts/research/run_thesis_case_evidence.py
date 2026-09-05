#!/usr/bin/env python3
"""Build input-rule-selected Dingxin mechanism cases."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_case_evidence import (
    run_thesis_case_evidence,
)


REPO = Path(__file__).resolve().parents[2]
OUTER = (
    REPO / "docs/artifacts/runs/2026-09-03_thesis-dingxin-confirmation-v3p2p2/"
    "outer_results.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--protocol-version", choices=("v3.2.2",), default="v3.2.2"
    )
    args = parser.parse_args()
    _require_ready_source()
    audit = run_thesis_case_evidence(
        output_root=REPO
        / "docs/artifacts/runs/2026-09-03_thesis-case-evidence-v3p2p2",
        device=args.device,
    )
    print(json.dumps(audit["pairing_diagnostic"], ensure_ascii=False, indent=2))


def _require_ready_source():
    if not OUTER.is_file() or not json.loads(OUTER.read_text(encoding="utf-8")).get(
        "completed"
    ):
        raise RuntimeError("case evidence requires completed Dingxin confirmation")
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
        raise RuntimeError("case evidence requires committed source")


if __name__ == "__main__":
    main()
