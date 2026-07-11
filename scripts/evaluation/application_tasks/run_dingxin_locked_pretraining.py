#!/usr/bin/env python3
"""Run three-seed five-fold Dingxin retraining of the selected configs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.application_tasks import (  # noqa: E402
    DingxinLockedPretrainingConfig,
    run_dingxin_locked_pretraining,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="2026-07-12_dingxin-locked-pretraining-coalesced"
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument(
        "--method",
        action="append",
        choices=("physiology_only", "vehicle_only", "mult", "contiformer", "chronaris"),
        default=[],
    )
    parser.add_argument("--fold-id", action="append", default=[])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--initialization-pretraining-run-id")
    parser.add_argument("--baseline-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chronaris-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_dingxin_locked_pretraining(
        DingxinLockedPretrainingConfig(
            run_id=args.run_id,
            seeds=tuple(args.seed) or (17, 29, 43),
            methods=tuple(args.method) or (
                "physiology_only",
                "vehicle_only",
                "mult",
                "contiformer",
                "chronaris",
            ),
            fold_ids=(
                tuple(args.fold_id)
                if args.fold_id
                else DingxinLockedPretrainingConfig().fold_ids
            ),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            initialization_pretraining_run_id=args.initialization_pretraining_run_id,
            baseline_device=args.baseline_device,
            chronaris_device=args.chronaris_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
