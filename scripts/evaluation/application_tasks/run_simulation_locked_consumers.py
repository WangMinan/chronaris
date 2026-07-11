#!/usr/bin/env python3
"""Run formal downstream consumers on G1-to-G2 locked representations."""

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
    SimulationLockedConsumerConfig,
    run_simulation_locked_consumers,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_simulation-locked-consumers")
    parser.add_argument("--minirocket-kernels", type=int, default=10_000)
    parser.add_argument("--tcn-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_simulation_locked_consumers(
        SimulationLockedConsumerConfig(
            run_id=args.run_id,
            minirocket_kernels=args.minirocket_kernels,
            tcn_device=args.tcn_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
