#!/usr/bin/env python3
"""Run Dingxin stage 3A residual activation and conditional stage-3B gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from chronaris.evaluation.application_tasks.residual_activation_run import (
    DingxinResidualActivationConfig,
    run_dingxin_residual_activation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-id",
        default="2026-07-15_dingxin-residual-activation",
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
    parser.add_argument(
        "--stage2-workspace-root",
        default="/home/wangminan/projects/chronaris-dingxin-task-aware-safe-residual-20260715",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run_dingxin_residual_activation(
        DingxinResidualActivationConfig(
            run_id=args.run_id,
            source_workspace_root=args.source_workspace_root,
            stability_workspace_root=args.stability_workspace_root,
            matched_workspace_root=args.matched_workspace_root,
            stage2_workspace_root=args.stage2_workspace_root,
            seed=args.seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
