#!/usr/bin/env python3
"""Run the Dingxin core-task feasibility and safe-fusion audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from chronaris.evaluation.application_tasks.core_feasibility_run import (
    DingxinCoreFeasibilityConfig,
    run_dingxin_core_feasibility,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default="2026-07-14_dingxin-core-feasibility"
    )
    parser.add_argument(
        "--compact-output-root", default="docs/artifacts/runs"
    )
    parser.add_argument(
        "--heavy-output-root", default="artifacts/application_evaluation"
    )
    parser.add_argument(
        "--source-workspace-root",
        default="/home/wangminan/projects/chronaris",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = run_dingxin_core_feasibility(
        DingxinCoreFeasibilityConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            source_workspace_root=args.source_workspace_root,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
