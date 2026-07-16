#!/usr/bin/env python3
"""Export three-seed five-fold six-method Dingxin representations."""

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
    DingxinLockedRepresentationConfig,
    run_dingxin_locked_representations,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402
from chronaris.representation import (  # noqa: E402
    DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY,
    DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="2026-07-12_dingxin-locked-representations-coalesced"
    )
    parser.add_argument(
        "--pretraining-run-id",
        default="2026-07-12_dingxin-locked-pretraining-coalesced",
    )
    parser.add_argument("--fit-batch-size", type=int, default=8)
    parser.add_argument("--export-batch-size", type=int, default=8)
    parser.add_argument("--baseline-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chronaris-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--fold-id", action="append", default=[])
    parser.add_argument(
        "--maneuver-history-policy",
        choices=(
            DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY,
            DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
        ),
        default=DINGXIN_EXCLUDE_MANEUVER_HISTORY_POLICY,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_dingxin_locked_representations(
        DingxinLockedRepresentationConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            seeds=tuple(args.seed) or DingxinLockedRepresentationConfig().seeds,
            fold_ids=(
                tuple(args.fold_id)
                if args.fold_id
                else DingxinLockedRepresentationConfig().fold_ids
            ),
            fit_batch_size=args.fit_batch_size,
            export_batch_size=args.export_batch_size,
            baseline_device=args.baseline_device,
            chronaris_device=args.chronaris_device,
            maneuver_history_policy=args.maneuver_history_policy,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
