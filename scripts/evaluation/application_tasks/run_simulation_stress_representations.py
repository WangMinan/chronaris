#!/usr/bin/env python3
"""Export locked six-method representations across all G2 stress levels."""

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
    SimulationStressRepresentationConfig,
    run_simulation_stress_representations,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_simulation-locked-stress-representations")
    parser.add_argument("--export-batch-size", type=int, default=32)
    parser.add_argument("--baseline-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chronaris-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_simulation_stress_representations(
        SimulationStressRepresentationConfig(
            run_id=args.run_id,
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
