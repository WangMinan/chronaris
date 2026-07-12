#!/usr/bin/env python3
"""Train the two isolatable Chronaris v2 formal mechanism ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_simulation_ablation_pretraining import (  # noqa: E402
    ChronarisV2SimulationAblationConfig,
    run_chronaris_v2_simulation_ablation_pretraining,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-13_chronaris-v2-simulation-ablation-pretraining")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    root = run_chronaris_v2_simulation_ablation_pretraining(
        ChronarisV2SimulationAblationConfig(
            run_id=args.run_id,
            device=args.device,
            max_epochs=args.max_epochs,
            resume=args.resume,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
