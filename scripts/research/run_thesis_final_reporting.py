#!/usr/bin/env python3
"""Aggregate the completed public and Dingxin frozen outer evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_final_reporting import (
    build_thesis_outer_summary,
)


REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-version", choices=("v3.2.2",), default="v3.2.2"
    )
    parser.parse_args()
    _require_clean_source()
    audit = build_thesis_outer_summary(
        native_compact_root=(
            REPO
            / "docs/artifacts/runs/2026-09-03_thesis-native-outer-v3p2p2"
        ),
        native_heavy_root=(
            REPO
            / "artifacts/application_evaluation/"
            "2026-09-03_thesis-native-outer-v3p2p2"
        ),
        dingxin_compact_root=(
            REPO
            / "docs/artifacts/runs/"
            "2026-09-03_thesis-dingxin-confirmation-v3p2p2"
        ),
        output_root=(
            REPO
            / "docs/artifacts/runs/2026-09-03_thesis-final-summary-v3p2p2"
        ),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


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
        raise RuntimeError("final thesis reporting requires committed source")


if __name__ == "__main__":
    main()
