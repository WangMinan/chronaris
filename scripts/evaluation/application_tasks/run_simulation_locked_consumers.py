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
    parser.add_argument("--pretraining-run-id", default="2026-07-12_simulation-locked-pretraining")
    parser.add_argument("--baseline-pretraining-run-id")
    parser.add_argument("--locked-configuration-path")
    parser.add_argument("--representation-run-id", default="2026-07-12_simulation-locked-representations")
    parser.add_argument("--minirocket-kernels", type=int, default=10_000)
    parser.add_argument("--tcn-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_simulation_locked_consumers(
        SimulationLockedConsumerConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            baseline_pretraining_run_id=args.baseline_pretraining_run_id,
            locked_configuration_path=args.locked_configuration_path,
            representation_run_id=args.representation_run_id,
            minirocket_kernels=args.minirocket_kernels,
            tcn_device=args.tcn_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
