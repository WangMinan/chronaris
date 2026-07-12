#!/usr/bin/env python3
"""Run the Chronaris v2 task-independent training smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_training_smoke import (  # noqa: E402
    ChronarisV2TrainingSmokeConfig,
    run_chronaris_v2_training_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-training-smoke",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    root = run_chronaris_v2_training_smoke(
        ChronarisV2TrainingSmokeConfig(
            run_id=args.run_id,
            device=args.device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
