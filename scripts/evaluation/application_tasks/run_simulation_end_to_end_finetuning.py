#!/usr/bin/env python3
"""Run the label-aware auxiliary fine-tuning table on locked simulation data."""

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
    SimulationFineTuningConfig,
    run_simulation_end_to_end_finetuning,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_simulation-end-to-end-finetuning")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chronaris-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_simulation_end_to_end_finetuning(
        SimulationFineTuningConfig(
            run_id=args.run_id,
            seeds=tuple(args.seed) or (17, 29, 43),
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            baseline_device=args.baseline_device,
            chronaris_device=args.chronaris_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
