#!/usr/bin/env python3
"""Run Dingxin target reconstruction and matched-clean baseline screening."""

from __future__ import annotations

import argparse
import json

from chronaris.evaluation.application_tasks.target_reconstruction_run import (
    DingxinTargetReconstructionConfig,
    run_dingxin_target_reconstruction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-id",
        default="2026-07-15_dingxin-target-reconstruction-confirmation",
    )
    parser.add_argument(
        "--source-workspace-root",
        default="/home/wangminan/projects/chronaris",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = run_dingxin_target_reconstruction(
        DingxinTargetReconstructionConfig(
            run_id=args.run_id,
            source_workspace_root=args.source_workspace_root,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict_result(result), ensure_ascii=False, indent=2))


def asdict_result(result):
    return {
        "status": result.status,
        "decision": result.decision,
        "allow_safe_fusion": result.allow_safe_fusion,
        "allow_task_aware_research": result.allow_task_aware_research,
        "compact_run_root": result.compact_run_root,
        "heavy_run_root": result.heavy_run_root,
        "report_path": result.report_path,
    }


if __name__ == "__main__":
    main()
