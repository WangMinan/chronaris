#!/usr/bin/env python3
"""Run the pre-registered Chronaris v2 physiology-fidelity repair round."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_fidelity_repair import (  # noqa: E402
    ChronarisV2FidelityRepairConfig,
    run_chronaris_v2_fidelity_repair,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-fidelity-repair-seed17",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=3)
    args = parser.parse_args()
    root = run_chronaris_v2_fidelity_repair(
        ChronarisV2FidelityRepairConfig(
            run_id=args.run_id,
            device=args.device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            max_candidates=args.max_candidates,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
