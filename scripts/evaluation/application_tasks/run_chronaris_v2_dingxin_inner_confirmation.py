#!/usr/bin/env python3
"""Run the three-view Dingxin task-independent v2 confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_dingxin_inner_confirmation import (  # noqa: E402
    ChronarisV2DingxinInnerConfirmationConfig,
    run_chronaris_v2_dingxin_inner_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-dingxin-inner-confirmation",
    )
    parser.add_argument(
        "--hyperparameter-diagnostics-run-id",
        default="2026-07-12_chronaris-v2-hyperparameter-diagnostics-seed17",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()
    root = run_chronaris_v2_dingxin_inner_confirmation(
        ChronarisV2DingxinInnerConfirmationConfig(
            run_id=args.run_id,
            hyperparameter_diagnostics_run_id=args.hyperparameter_diagnostics_run_id,
            device=args.device,
            max_epochs=args.max_epochs,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
