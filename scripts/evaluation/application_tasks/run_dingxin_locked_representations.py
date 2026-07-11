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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_dingxin_locked_representations(
        DingxinLockedRepresentationConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            fit_batch_size=args.fit_batch_size,
            export_batch_size=args.export_batch_size,
            baseline_device=args.baseline_device,
            chronaris_device=args.chronaris_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
