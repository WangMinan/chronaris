#!/usr/bin/env python3
"""Run Dingxin task-protocol repair and cross-view stability audit."""

from __future__ import annotations

import argparse
import json

from chronaris.evaluation.application_tasks.task_stability_run import (
    DingxinTaskStabilityConfig,
    run_dingxin_task_stability,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-workspace-root",
        default="/home/wangminan/projects/chronaris",
    )
    args = parser.parse_args()
    result = run_dingxin_task_stability(
        DingxinTaskStabilityConfig(
            source_workspace_root=args.source_workspace_root,
            resume=args.resume,
        )
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "decision": result.decision,
                "allow_safe_fusion": result.allow_safe_fusion,
                "allow_task_aware_research": result.allow_task_aware_research,
                "compact_run_root": result.compact_run_root,
                "heavy_run_root": result.heavy_run_root,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
