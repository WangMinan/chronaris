#!/usr/bin/env python3
"""Export locked Chronaris v2 formal ablation representations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_simulation_ablation_representations import (  # noqa: E402
    ChronarisV2SimulationAblationRepresentationConfig,
    run_chronaris_v2_simulation_ablation_representations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-13_chronaris-v2-simulation-ablation-representations")
    parser.add_argument("--pretraining-run-id", default="2026-07-13_chronaris-v2-simulation-ablation-pretraining")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    root = run_chronaris_v2_simulation_ablation_representations(
        ChronarisV2SimulationAblationRepresentationConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            device=args.device,
            resume=args.resume,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
