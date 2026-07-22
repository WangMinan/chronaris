#!/usr/bin/env python3
"""Run the frozen task-aware safe-residual screening for Dingxin."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from chronaris.evaluation.application_tasks.task_aware_safe_residual_run import (
    DingxinSafeResidualConfig,
    run_dingxin_task_aware_safe_residual,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-id",
        default="2026-07-15_dingxin-task-aware-safe-residual",
    )
    parser.add_argument(
        "--source-workspace-root",
        default="/home/wangminan/projects/chronaris",
    )
    parser.add_argument(
        "--stability-workspace-root",
        default="/home/wangminan/projects/chronaris-dingxin-task-stability-20260714",
    )
    parser.add_argument(
        "--matched-workspace-root",
        default="/home/wangminan/projects/chronaris-dingxin-target-reconstruction-20260715",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--partial-max-epochs", type=int, default=24)
    parser.add_argument("--partial-patience", type=int, default=6)
    args = parser.parse_args()
    result = run_dingxin_task_aware_safe_residual(
        DingxinSafeResidualConfig(
            run_id=args.run_id,
            source_workspace_root=args.source_workspace_root,
            stability_workspace_root=args.stability_workspace_root,
            matched_workspace_root=args.matched_workspace_root,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            partial_max_epochs=args.partial_max_epochs,
            partial_patience=args.partial_patience,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
