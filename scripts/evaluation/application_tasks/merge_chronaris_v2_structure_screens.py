#!/usr/bin/env python3
"""Merge device-partitioned Chronaris v2 structure-screen evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_structure_merge import (  # noqa: E402
    ChronarisV2StructureMergeConfig,
    merge_chronaris_v2_structure_screens,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-structure-screen-seed17-combined",
    )
    parser.add_argument("--source-run-id", action="append", dest="source_run_ids")
    args = parser.parse_args()
    config = ChronarisV2StructureMergeConfig(
        run_id=args.run_id,
        source_run_ids=(
            tuple(args.source_run_ids)
            if args.source_run_ids
            else ChronarisV2StructureMergeConfig().source_run_ids
        ),
    )
    print(merge_chronaris_v2_structure_screens(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
