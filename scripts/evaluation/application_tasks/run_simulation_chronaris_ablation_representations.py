#!/usr/bin/env python3
"""Export locked G1-to-G2 representations for Chronaris ablations."""

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
    CHRONARIS_ABLATION_VARIANTS,
    SimulationChronarisAblationRepresentationConfig,
    run_simulation_chronaris_ablation_representations,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_simulation-chronaris-ablation-representations")
    parser.add_argument("--pretraining-run-id", default="2026-07-12_simulation-chronaris-ablation-pretraining")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--export-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_simulation_chronaris_ablation_representations(
        SimulationChronarisAblationRepresentationConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            seeds=tuple(args.seed) or (17, 29, 43),
            variants=tuple(args.variant) or CHRONARIS_ABLATION_VARIANTS,
            export_batch_size=args.export_batch_size,
            device=args.device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
