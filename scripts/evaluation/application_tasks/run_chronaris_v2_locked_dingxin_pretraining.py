#!/usr/bin/env python3
"""Run locked Chronaris v2 Dingxin retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_locked_dingxin_pretraining import (  # noqa: E402
    ChronarisV2LockedDingxinConfig,
    run_chronaris_v2_locked_dingxin_pretraining,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-13_chronaris-v2-dingxin-locked-pretraining",
    )
    parser.add_argument(
        "--locked-configuration-path",
        default=(
            "docs/artifacts/runs/"
            "2026-07-12_chronaris-v2-dingxin-inner-confirmation-r3/"
            "locked_configuration.json"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()
    root = run_chronaris_v2_locked_dingxin_pretraining(
        ChronarisV2LockedDingxinConfig(
            run_id=args.run_id,
            locked_configuration_path=args.locked_configuration_path,
            device=args.device,
            max_epochs=args.max_epochs,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
