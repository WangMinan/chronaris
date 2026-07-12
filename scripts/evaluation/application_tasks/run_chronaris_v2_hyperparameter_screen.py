#!/usr/bin/env python3
"""Run the gated Chronaris v2 24-candidate hyperparameter screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_hyperparameter_screen import (  # noqa: E402
    ChronarisV2HyperparameterScreenConfig,
    run_chronaris_v2_hyperparameter_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v2-hyperparameter-screen-seed17",
    )
    parser.add_argument(
        "--architecture-gate-run-id",
        default="2026-07-12_chronaris-v2-direct-residual-repair-seed17-r1",
    )
    parser.add_argument(
        "--architecture-gate-candidate-id",
        default="direct_residual_01_causal_query",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=24)
    args = parser.parse_args()
    root = run_chronaris_v2_hyperparameter_screen(
        ChronarisV2HyperparameterScreenConfig(
            run_id=args.run_id,
            architecture_gate_run_id=args.architecture_gate_run_id,
            architecture_gate_candidate_id=args.architecture_gate_candidate_id,
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
